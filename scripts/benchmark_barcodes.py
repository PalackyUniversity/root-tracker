"""Compare uncached barcode reads over every image in the configured dataset."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from root_tracker.config import Config
from root_tracker.io.loader import ImageLoader
from root_tracker.io.barcode import BarcodeReader
from root_tracker.pipeline import RootTrackingPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/in_vitro.yaml')
    parser.add_argument('--reader-file', help='Original barcode.py for baseline comparison')
    parser.add_argument('--output', required=True)
    parser.add_argument('--compare', help='Fail on lost or changed detections from this baseline JSON')
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 5),
                        help='Concurrent barcode reads (GUI uses up to 4)')
    args = parser.parse_args()
    reader_type = BarcodeReader
    if args.reader_file:
        spec = importlib.util.spec_from_file_location('reference_barcode', args.reader_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reader_type = module.BarcodeReader
    reader = reader_type()
    config = Config.from_yaml(args.config)
    groups = ImageLoader(config).create_series()
    pipeline = RootTrackingPipeline(config)
    rows = []
    group_times = []
    items = [item for group in sorted(groups) for item in groups[group].images]
    expected = {item.path: item.barcode for item in items}
    order = {item.path: i for i, item in enumerate(items)}

    def timed_read(path):
        start = time.perf_counter()
        if hasattr(reader, 'read_file'):
            text, rect = reader.read_file(path)
            loaded = None
            shape = None
        else:
            image = cv2.imread(path)
            loaded = time.perf_counter()
            if image is None:
                raise RuntimeError(f'Cannot read {path}')
            shape = image.shape
            text, rect = reader.read_fast(image)
        finished = time.perf_counter()
        rows.append(dict(path=path, expected=expected[path], text=text, rect=rect,
                         shape=shape, load_seconds=None if loaded is None else loaded-start,
                         decode_seconds=None if loaded is None else finished-loaded,
                         total_seconds=finished-start))
        return text, rect

    # Exercise the same bounded iterator used by both GUI barcode actions.
    pipeline.barcode_reader = SimpleNamespace(read_file=timed_read)
    for group in sorted(groups):
        offset = len(rows)
        start = time.perf_counter()
        list(pipeline.iter_detect_barcodes(groups[group].images, max_workers=args.workers))
        elapsed = time.perf_counter() - start
        if len(rows) - offset != len(groups[group].images):
            raise RuntimeError(f'Barcode reader raised an exception in {group}')
        group_times.append(elapsed)
        for row in rows[offset:]:
            row.update(group=group, group_wall_seconds=elapsed, workers=args.workers)
        rows.sort(key=lambda row: order[row['path']])
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f'{len(rows)}/{len(items)} images; found {sum(bool(r["text"]) for r in rows)}', flush=True)
    print(json.dumps(dict(images=len(rows), found=sum(bool(r['text']) for r in rows),
                          mismatches=sum(bool(r['text']) and r['text'].lower()!=r['expected'].lower() for r in rows),
                          seconds=sum(group_times), workers=args.workers,
                          slowest_group_seconds=max(group_times))))
    if args.compare:
        reference = {r['path']: r for r in json.loads(Path(args.compare).read_text())}
        if set(reference) != {r['path'] for r in rows}:
            raise AssertionError('Dataset image list changed')
        regressions = [r['path'] for r in rows
                       if reference[r['path']]['text'] and
                       reference[r['path']]['text'] != r['text']]
        if regressions:
            raise AssertionError(f'Lost or changed barcode detections: {regressions}')
        print('All baseline detections preserved', flush=True)


if __name__ == '__main__':
    main()
