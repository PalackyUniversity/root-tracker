"""Research wall timers for two local groups; no processing cache reads/writes."""
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import sys
import time
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.preprocessing.background import BackgroundRemover
from root_tracker.registration.template_matcher import ImageRegistrator
from root_tracker.tracking import RootSkeletonizer, RootThresholder, RootLinker
from root_tracker.tracking import main_root, temporal_debris
from root_tracker.config import Config

totals = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
def timed(original, name):
    @wraps(original)
    def wrapper(*args, **kwargs):
        begin = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            totals[name]['calls'] += 1
            totals[name]['seconds'] += time.perf_counter() - begin
    return wrapper

records = []
with ExitStack() as stack:
    for obj, names in [
        (RootTrackingPipeline, ['preprocess_series', 'register_series', 'track_and_analyze_series']),
        (BackgroundRemover, ['remove_gradient', '_median_background', 'compute_canny_edges']),
        (ImageRegistrator, ['_coarse_match_location', '_match_location']),
        (RootSkeletonizer, ['skeletonize_mask']),
        (RootThresholder, ['threshold', 'filter_small_contours']),
        (RootLinker, ['link_corners']),
        (main_root, ['select_main_geometry']),
        (temporal_debris, ['filter_static_islands']),
    ]:
        for name in names:
            stack.enter_context(patch.object(obj, name, timed(getattr(obj, name), name)))
    for repeat in range(2):
        for group in ('RT_25_10-1', 'RT_25_10-10'):
            totals.clear()
            config = Config.from_yaml(ROOT / 'configs/in_vitro.yaml')
            pipeline = RootTrackingPipeline(config)
            series = pipeline.loader.create_series()[group]
            with tempfile.TemporaryDirectory(prefix='root-tracker-gpu-profile-') as folder:
                config.data.output = folder
                pipeline.preprocess_series(series)
                pipeline.register_series(series)
                pipeline.track_and_analyze_series(series)
            row = {'group': group, 'repeat': repeat, 'timers': dict(totals)}
            records.append(row)
            print(json.dumps(row), flush=True)
            Path('docs/performance-data/gpu-feasibility-2026-09-29/wall-timers.json').write_text(json.dumps(records, indent=2))
