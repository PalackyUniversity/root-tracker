#!/usr/bin/env python3
"""Inspect an RSML document and optionally save an exact copy."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from root_tracker.io.rsml import read_rsml, save_rsml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='RSML document to inspect')
    parser.add_argument('--output', type=Path, help='Save an exact, unmodified copy')
    args = parser.parse_args()
    try:
        document = read_rsml(args.input)
        print(f'{len(document.plant_ids)} plants, {len(document.roots)} roots')
        print(f'Scale: {document.resolution:g} pixels per {document.unit}')
        print(f'Image reference: {document.image_name or "not supplied"}')
        for root in document.roots:
            print(f'Plant {root.plant_id}, root {root.id}: {len(root.points)} points, '
                  f'polyline length {root.length:g} px')
        if args.output:
            save_rsml(document, args.output)
            print(f'Exact RSML copy saved to: {args.output}')
    except (OSError, ValueError) as exc:
        print(f'RSML error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
