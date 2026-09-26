"""Profile a fresh GUI preprocess selection, without touching dataset caches.

Timers include work on background threads. Nested/overlapping timers must not be
summed as elapsed time. Qt uses an offscreen window; compositor/display latency
is excluded. Timed spans record their thread IDs to expose overlapping work.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from root_tracker.config import Config
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.gui.main_window import MainWindow, ProcessingState
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.io import series_cache
from root_tracker.preprocessing import background as background_module, gpu_background
from root_tracker.registration.template_matcher import fft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/in_vitro.yaml')
    parser.add_argument('--groups', nargs='+', default=['RT_26_2-1', 'RT_26_2-40', 'RT_26_2-4'])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-root', default='.')
    parser.add_argument('--barcodes', action='store_true', help='Also run uncached barcode detection before preprocessing')
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(['preprocess-profile'])
    records = []
    for repeat in range(args.repeats):
        for group in args.groups:
            config = Config.from_yaml(args.config)
            pipeline = RootTrackingPipeline(config)
            groups = pipeline.loader.create_series()  # Metadata/masks only; no pipeline cache.
            series = groups[group]
            config.data.detect_barcodes = args.barcodes
            events, errors = [], []
            lock = threading.Lock()
            start = 0

            def instrument(stack, obj, name, label):
                original = getattr(obj, name)
                @wraps(original)
                def measured(*positional, **keywords):
                    begin = time.perf_counter()
                    try:
                        return original(*positional, **keywords)
                    finally:
                        end = time.perf_counter()
                        with lock:
                            events.append({'name': label, 'start': begin-start,
                                           'seconds': end-begin, 'thread': threading.get_ident()})
                stack.enter_context(patch.object(obj, name, measured))

            with tempfile.TemporaryDirectory(prefix='root-tracker-profile-', dir=args.cache_root) as folder:
                config.data.input = str(Path(folder).resolve())
                config.data.output = str(Path(folder).resolve() / 'exports')
                with patch.object(MainWindow, '_load_last_folder'):
                    window = MainWindow(config)
                window._pipeline = pipeline
                window._series_dict = groups
                window._image_tree.set_series(groups)
                window._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
                window._workflow_bar.blockSignals(True)
                window._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
                window._workflow_bar.blockSignals(False)
                window._settings_panel.set_step(WorkflowStep.PREPROCESS)
                window._auto_preview_action.setChecked(True)
                window._image_tree.select_image(series.images[0])
                window.show()
                app.processEvents()
                np.random.seed(0)
                with ExitStack() as stack:
                    for obj, names in [
                        (pipeline, ['preprocess_series', 'preprocess_image', 'register_series']),
                        (pipeline.cropper, ['rotate', 'auto_crop_to_blue_background']),
                        (pipeline.green_detector, ['find_green_contours', 'cluster_plant_positions', 'mask_green_in_image']),
                        (pipeline.background_remover, ['remove_gradient', '_median_background', 'compute_canny_edges']),
                        (background_module, ['try_subtract_median']),
                        (gpu_background._CudaMedian, ['filter']),
                        (pipeline.registrator, ['align_to_template', '_match_location']),
                        (cv2, ['imread', 'medianBlur', 'matchTemplate', 'copyMakeBorder', 'integral']),
                        (fft, ['rfft2', 'irfft2']),
                        (series_cache, ['save_series']),
                        (window, ['_display_image']),
                        (window._image_viewer, ['set_image']),
                        (window._image_tree, ['refresh_status']),
                        (pipeline.barcode_reader, ['read_file']),
                    ]:
                        for name in names:
                            instrument(stack, obj, name, name)
                    # Avoid unattended modal dialogs; report failures instead.
                    stack.enter_context(patch('root_tracker.gui.main_window.QMessageBox.critical',
                                              side_effect=lambda *a: errors.append(str(a[-1]))))
                    start = time.perf_counter()
                    worker = None
                    try:
                        window._on_image_selected(series.images[0])
                        while window._state != ProcessingState.IDLE:
                            worker = window._worker or worker
                            app.processEvents()
                            if time.perf_counter() - start > 120:
                                window._on_cancel_prediction()
                                raise TimeoutError(f'Preprocessing timed out: {group}')
                            time.sleep(.002)
                        app.processEvents()
                        elapsed = time.perf_counter() - start
                    finally:
                        if worker is not None:
                            worker.cancel()
                            worker.wait()
                try:
                    if errors:
                        raise RuntimeError('; '.join(errors))
                    if not series.pipeline_state.preprocessed or window._image_viewer._pixmap_item is None:
                        raise RuntimeError('Processing or preview did not complete')
                    totals = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
                    for event in events:
                        totals[event['name']]['calls'] += 1
                        totals[event['name']]['seconds'] += event['seconds']
                    record = {'group': group, 'repeat': repeat, 'images': len(series.images),
                              'elapsed_seconds': elapsed, 'barcodes': args.barcodes,
                              'gpu_backend_active': gpu_background._backend is not None,
                              'cpu_count': os.cpu_count(), 'opencv_threads': cv2.getNumThreads(),
                              'load_average': os.getloadavg(),
                              'image_shapes': [list(img.image.shape) if img.image is not None else None for img in series.images],
                              'timers': dict(totals), 'events': sorted(events, key=lambda e: e['start'])}
                    records.append(record)
                    Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
                    print(group, repeat, json.dumps({'elapsed': elapsed, 'timers': dict(totals)}), flush=True)
                finally:
                    window.close()
                    window.deleteLater()
                    app.processEvents()
    print(f'Saved {len(records)} GUI profiles to {args.output}', flush=True)


if __name__ == '__main__':
    main()
