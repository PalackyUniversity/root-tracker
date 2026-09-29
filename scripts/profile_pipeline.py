"""Profile fresh groups and fingerprint outputs, without touching user caches.

Windows memory counters need no extra packages. Inclusive function timers
overlap and must not be summed. Use separate processes with --module-root for
before/after comparisons. --trace-lines adds overhead; use only for attribution.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import ctypes
from dataclasses import fields
from functools import wraps
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch


def memory_snapshot():
    if sys.platform != 'win32':
        return {}
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'commit_limit',
                                             'commit_available', 'virtual', 'virtual_available', 'extended')]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                         'peak_nonpaged', 'nonpaged', 'commit', 'peak_commit', 'private')]
    status, process = MemoryStatus(), ProcessMemory()
    status.length, process.cb = ctypes.sizeof(status), ctypes.sizeof(process)
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)) or not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(process), process.cb):
        return {}
    return {'rss': process.rss, 'private': process.private, 'peak_rss': process.peak_rss,
            'peak_commit': process.peak_commit, 'page_faults': process.faults,
            'system_total': status.total, 'system_available': status.available}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--config', default='configs/in_vitro.yaml')
    parser.add_argument('--groups', nargs='+', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare')
    parser.add_argument('--trace-lines', action='store_true')
    parser.add_argument('--batch-worker', action='store_true', help='Run in a spawned worker process')
    parser.add_argument('--opencv-threads', type=int)
    args = parser.parse_args()
    if args.batch_worker:
        import multiprocessing as mp
        if mp.current_process().name == 'MainProcess':
            worker = mp.get_context('spawn').Process(target=main)
            worker.start()
            worker.join()
            if worker.exitcode:
                raise SystemExit(worker.exitcode)
            return
    sys.path.insert(0, args.module_root)
    if args.opencv_threads is not None:
        import cv2
        cv2.setNumThreads(args.opencv_threads)
    import numpy as np
    from root_tracker.config import Config
    from root_tracker.pipeline import RootTrackingPipeline
    from root_tracker.preprocessing.background import BackgroundRemover
    from root_tracker.registration.template_matcher import ImageRegistrator
    from root_tracker.tracking import RootSkeletonizer, RootLinker
    from root_tracker.tracking import main_root, temporal_debris

    def digest(value):
        if isinstance(value, np.ndarray):
            return {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, dict):
            return {str(k): digest(v) for k, v in value.items()}
        if isinstance(value, set):
            return hashlib.sha256(repr(sorted(value)).encode()).hexdigest()
        if isinstance(value, (tuple, list)):
            return [digest(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, 'isoformat'):
            return value.isoformat()
        return value

    def fingerprint(series, statistics):
        images = []
        for img in series.images:
            images.append({f.name: digest(getattr(img, f.name)) for f in fields(img)
                           if f.name not in ('rsml_document', 'rsml_unmasked_document', 'rsml_original_document')})
        return {'images': images, 'statistics': [digest(s.to_dict()) for s in statistics]}

    def retained_arrays(series):
        arrays = {}
        def visit(value):
            if isinstance(value, np.ndarray):
                while isinstance(value.base, np.ndarray):
                    value = value.base
                arrays[id(value)] = value.nbytes
            elif isinstance(value, dict):
                for v in value.values():
                    visit(v)
        for img in series.images:
            for f in fields(img):
                visit(getattr(img, f.name))
        return sum(arrays.values())

    references = {} if not args.compare else {
        row['group']: row['outputs'] for row in json.loads(Path(args.compare).read_text())}
    records = []
    for repeat in range(args.repeats):
        for group in args.groups:
            gc.collect()
            config = Config.from_yaml(args.config)
            pipeline = RootTrackingPipeline(config)
            series = pipeline.loader.create_series()[group]
            np.random.seed(0)
            timers = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
            samples, line_times, active = [], defaultdict(lambda: [0, 0.0]), {}
            def timed(original, name):
                @wraps(original)
                def wrapper(*pos, **kw):
                    begin = time.perf_counter()
                    try:
                        return original(*pos, **kw)
                    finally:
                        timers[name]['calls'] += 1
                        timers[name]['seconds'] += time.perf_counter() - begin
                return wrapper
            def trace(frame, event, arg):
                if frame.f_code.co_name not in ('select_main_geometry', '_main_portions', 'temporal_support', 'publish'):
                    return None
                now = time.perf_counter()
                old = active.get(id(frame))
                if old:
                    line_times[old[0]][0] += 1
                    line_times[old[0]][1] += now - old[1]
                key = f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}'
                if event == 'return':
                    active.pop(id(frame), None)
                else:
                    active[id(frame)] = (key, now)
                return trace
            stop = threading.Event()
            def sample():
                while not stop.is_set():
                    samples.append(memory_snapshot())
                    stop.wait(.05)
            before = memory_snapshot()
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            stages = {}
            try:
                with tempfile.TemporaryDirectory(prefix='root-tracker-perf-') as folder, ExitStack() as stack:
                    config.data.output = folder
                    for obj, names in [
                        (BackgroundRemover, ['remove_gradient', '_median_background']),
                        (ImageRegistrator, ['_coarse_match_location', '_match_location']),
                        (RootSkeletonizer, ['skeletonize_mask']),
                        (RootLinker, ['link_corners']),
                        (main_root, ['select_main_geometry']),
                        (temporal_debris, ['filter_static_islands'])]:
                        for name in names:
                            stack.enter_context(patch.object(obj, name, timed(getattr(obj, name), name)))
                    for stage, fn in [('preprocess', lambda: pipeline.preprocess_series(series)),
                                      ('registration', lambda: pipeline.register_series(series)),
                                      ('tracking', lambda: pipeline.track_and_analyze_series(series))]:
                        start, cpu = time.perf_counter(), time.process_time()
                        if args.trace_lines and stage == 'tracking':
                            sys.settrace(trace)
                        try:
                            result = fn()
                        finally:
                            sys.settrace(None)
                        stages[stage] = {'seconds': time.perf_counter() - start,
                                         'cpu_seconds': time.process_time() - cpu,
                                         'memory': memory_snapshot(),
                                         'retained_array_bytes': retained_arrays(series)}
                    statistics = result
            finally:
                stop.set()
                sampler.join()
            outputs = fingerprint(series, statistics)
            peaks = {k: max((s[k] for s in samples if k in s), default=None)
                     for k in ('rss', 'private')}
            if samples and 'system_available' in samples[0]:
                peaks['min_system_available'] = min(s['system_available'] for s in samples)
            row = {'group': group, 'repeat': repeat, 'before': before, 'stages': stages,
                   'peak_memory': peaks, 'timers': dict(timers), 'outputs': outputs,
                   'line_times': dict(sorted(line_times.items(), key=lambda p: -p[1][1]))}
            records.append(row)
            Path(args.output).write_text(json.dumps(records, indent=2))
            if args.compare and outputs != references[group]:
                raise AssertionError(f'Output mismatch: {group}')
            print(json.dumps({k: v for k, v in row.items() if k not in ('outputs', 'line_times')}), flush=True)
            del pipeline, series, statistics, result


if __name__ == '__main__':
    main()
