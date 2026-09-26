"""Benchmark fresh groups without reading/writing the dataset's pipeline cache.

Use --module-root to compare an original checkout with the working tree.
Fingerprints include every output image and plant assignment, not just totals.
OS file caches and imported-library startup are deliberately outside the timing.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module-root', default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--config', default='configs/in_vitro.yaml')
parser.add_argument('--groups', nargs='+', default=['RT_26_2-1', 'RT_26_2-20', 'RT_26_2-40', 'RT_26_2-60', 'RT_26_2-75', 'RT_26_2-4'])
parser.add_argument('--output', required=True)
parser.add_argument('--repeats', type=int, default=1)
parser.add_argument('--cache-root', default='.', help='Use the dataset disk for temporary cache writes')
parser.add_argument('--compare', help='Reference JSON; fail if any output differs')
args = parser.parse_args()
references = {} if not args.compare else {
    row['group']: row['outputs'] for row in json.loads(Path(args.compare).read_text())}
sys.path.insert(0, args.module_root)

import numpy as np
from root_tracker.config import Config
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.io import series_cache


def fingerprint(series, statistics):
    result = {'images': [], 'statistics': [s.to_dict() for s in statistics]}
    for img in series.images:
        row = {}
        for field in ('image', 'process', 'canny', 'diff', 'image_annotated'):
            value = getattr(img, field)
            row[field] = None if value is None else {
                'shape': value.shape,
                'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
            }
        for field in ('positions_x', 'positions_y', 'green_areas', 'total_length',
                      'total_area', 'new_area', 'new_parts', 'plant_length', 'longest'):
            row[field] = getattr(img, field)
        row['colored_samples'] = hashlib.sha256(repr([
            (int(k), sorted((int(x), int(y)) for x, y in v))
            for k, v in sorted(img.colored_samples.items())
        ]).encode()).hexdigest()
        result['images'].append(row)
    return result


records = []
for repeat in range(args.repeats):
    for group in args.groups:
        config = Config.from_yaml(args.config)
        pipeline = RootTrackingPipeline(config)
        # create_series loads optional user masks, but never pipeline caches.
        series = pipeline.loader.create_series()[group]
        np.random.seed(0)  # Original KMeans uses NumPy's global random state.
        timings = {}
        cache_bytes = {}
        with tempfile.TemporaryDirectory(prefix='root-tracker-benchmark-', dir=args.cache_root) as cache_dir:
            config.data.output = str(Path(cache_dir) / 'exports')
            Path(config.data.output).mkdir()
            for stage, fn in (
                ('preprocess', lambda: pipeline.preprocess_series(series)),
                ('registration', lambda: pipeline.register_series(series)),
                ('preprocess_cache', lambda: series_cache.save_series(series, config)),
                ('track', lambda: pipeline.track_and_analyze_series(series)),
                ('track_cache', lambda: series_cache.save_series(series, config)),
            ):
                if stage.endswith('cache'):
                    config.data.input = cache_dir
                start = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    value = fn()
                timings[stage] = time.perf_counter() - start
                if stage == 'track':
                    statistics = value
                if stage.endswith('cache') and value is None:
                    raise RuntimeError('Cache write failed')
                if stage.endswith('cache'):
                    cache_bytes[stage] = value.stat().st_size
            gpu_module = sys.modules.get('root_tracker.preprocessing.gpu_background')
            record = {'gpu_backend_active': bool(gpu_module and gpu_module._backend is not None),
                      'group': group, 'repeat': repeat, 'timings': timings,
                      'cache_bytes': cache_bytes,
                      'outputs': fingerprint(series, statistics)}
            records.append(record)
            print(group, json.dumps(timings), flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2,
                default=lambda x: x.item() if isinstance(x, np.generic) else x.isoformat()))
            if args.compare:
                # Round-trip normalizes tuples, dates and NumPy scalars.
                outputs = json.loads(Path(args.output).read_text())[-1]['outputs']
                if outputs != references[group]:
                    raise AssertionError(f'Output mismatch for {group}')
                print(f'{group}: exact output match', flush=True)
