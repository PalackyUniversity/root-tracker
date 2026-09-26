"""Export assigned skeleton paths without changing tracking or its measurements."""
from copy import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from ..config import Config
from ..models import ImageSeries
from . import series_cache


def trace_paths(points: np.ndarray) -> list[list[tuple[int, int]]]:
    """Walk every skeleton edge once; retain isolated pixels as single points.

    Diagonal shortcuts across an existing orthogonal corner are excluded. Paths
    split at junctions, and cycles close explicitly. These are geometric paths,
    not an inference about primary/lateral biological roots.
    """
    pixels = {tuple(map(int, point)) for point in points}
    adjacency = {}
    for x, y in pixels:
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                target = (x + dx, y + dy)
                if not (dx or dy) or target not in pixels:
                    continue
                if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                neighbors.append(target)
        adjacency[x, y] = sorted(neighbors, key=lambda p: (p[1], p[0]))
    used = set()
    paths = []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, neighbor):
        path = [start, neighbor]
        used.add(edge(start, neighbor))
        while len(adjacency[path[-1]]) == 2:
            candidates = [p for p in adjacency[path[-1]] if edge(path[-1], p) not in used]
            if not candidates:
                break
            point = candidates[0]
            used.add(edge(path[-1], point))
            path.append(point)
        return path

    ordered = sorted(pixels, key=lambda p: (p[1], p[0]))
    for point in ordered:
        if not adjacency[point]:
            paths.append([point])
        elif len(adjacency[point]) != 2:
            for neighbor in adjacency[point]:
                if edge(point, neighbor) not in used:
                    paths.append(walk(point, neighbor))
    for point in ordered:
        for neighbor in adjacency[point]:
            if edge(point, neighbor) not in used:
                paths.append(walk(point, neighbor))
    return paths


def _current(series: ImageSeries, config: Config) -> bool:
    state = series.pipeline_state
    return (state.preprocessed and state.tracked
            and state.preprocess_config_hash == config.preprocess_config_hash()
            and state.tracking_config_hash == config.tracking_config_hash())


def _export_source(series: ImageSeries, config: Config, image_index: int | None = None) -> ImageSeries:
    selected = series.images if image_index is None else [series.images[image_index]]
    native = [image for image in selected if image.rsml_document is None]
    if native and not _current(series, config):
        raise ValueError(f'{series.group}: tracking is stale; track this series before RSML export')
    if any(image.image is None or image.rsml_samples is None for image in selected if image.rsml_document is None):
        # Cache restoration must never populate or overwrite live tracker state.
        restored = copy(series)
        restored.images = [copy(image) for image in series]
        restored.pipeline_state = copy(series.pipeline_state)
        if not series_cache.load_series(restored, config, require_identity=True) or not _current(restored, config):
            raise ValueError(f'{series.group}: export data is unavailable; track this series again')
        series = restored
    selected = series.images if image_index is None else [series.images[image_index]]
    if not selected or any(image.image is None or image.rsml_samples is None for image in selected if image.rsml_document is None):
        raise ValueError(f'{series.group}: no RSML geometry in this cache; track this series again')
    return series


def _safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:60] or 'series'


def _document(image, image_name: str, group: str, index: int) -> ET.Element:
    root = ET.Element('rsml')
    metadata = ET.SubElement(root, 'metadata')
    for key, value in (
        ('version', '1'), ('unit', 'pixel'), ('resolution', '1'),
        ('last-modified', datetime.now(timezone.utc).isoformat()),
        ('software', 'Root Tracker'), ('user', ''),
        ('file-key', f'{group}/{index}/{image.filename}'),
    ):
        ET.SubElement(metadata, key).text = value
    source = ET.SubElement(metadata, 'image')
    ET.SubElement(source, 'name').text = image_name
    ET.SubElement(source, 'captured').text = image.date.isoformat()
    sequence = ET.SubElement(metadata, 'time-sequence')
    ET.SubElement(sequence, 'label').text = group
    ET.SubElement(sequence, 'index').text = str(index)
    # Registration does not establish temporal root IDs.
    ET.SubElement(sequence, 'unified').text = 'false'
    scene = ET.SubElement(root, 'scene')
    for plant_id, points in sorted(image.rsml_samples.items()):
        plant = ET.SubElement(scene, 'plant', id=str(plant_id + 1), label=f'Plant {plant_id + 1}')
        for root_index, path in enumerate(trace_paths(points), 1):
            element = ET.SubElement(plant, 'root', id=f'p{plant_id + 1}-r{root_index}', label='Skeleton path')
            polyline = ET.SubElement(ET.SubElement(element, 'geometry'), 'polyline')
            # A degenerate two-point polyline preserves isolated detections at
            # zero length and remains readable by tools requiring line segments.
            if len(path) == 1:
                path = path * 2
            for x, y in path:
                ET.SubElement(polyline, 'point', x=str(x), y=str(y))
    ET.indent(root)
    return root


def export_series_rsml(series: ImageSeries, config: Config, directory: Path,
                       *, image_index: int | None = None) -> list[Path]:
    """Save one RSML and matching processed PNG per frame in a new series folder.

    When image_index is given, export only that frame while retaining its
    original time-sequence index and full-series cache identity.
    Selected frames are staged before publication. Existing destinations are
    refused; callers should select a fresh output directory for another export.
    """
    if image_index is not None and not 0 <= image_index < len(series.images):
        raise ValueError('The selected image is not in this series')
    series = _export_source(series, config, image_index)
    frames = list(enumerate(series)) if image_index is None else [(image_index, series.images[image_index])]
    for _, image in frames:
        if image.rsml_document is not None:
            continue
        height, width = image.image.shape[:2]
        for points in image.rsml_samples.values():
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('Invalid RSML export pixel coordinates; track again')
            if len(points) and (np.any(points < 0) or np.any(points[:, 0] >= width)
                                or np.any(points[:, 1] >= height)):
                raise ValueError('RSML export coordinates fall outside the processed image')
    directory = Path(directory)
    digest = hashlib.sha256(series.group.encode()).hexdigest()[:12]
    suffix = '' if image_index is None else f'-image-{image_index:04d}'
    target = directory / f'{_safe_name(series.group)}-{digest}{suffix}'
    if target.exists():
        raise FileExistsError(f'RSML destination already exists: {target}')
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.rsml-', dir=directory))
    names = []
    try:
        for index, image in frames:
            stem = f'{index:04d}-{_safe_name(Path(image.filename).stem)}'
            if image.rsml_document is not None:
                from .rsml_replacement import _load_background
                document = image.rsml_document
                frame_dir = staging / stem
                frame_dir.mkdir()
                # Keep all XML bytes. Supply its referenced image only when the
                # reference is a safe filename; never follow external paths.
                reference = document.image_name
                safe_image = (reference and Path(reference).name == reference
                              and reference not in ('.', '..') and '\\' not in reference
                              and Path(reference).suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
                png_name = reference if safe_image else 'preview.png'
                background = image.rsml_background
                if background is None:
                    background = _load_background(image)
                if not cv2.imwrite(str(frame_dir / png_name), background):
                    raise OSError('Could not export replacement background')
                (frame_dir / 'roots.rsml').write_bytes(document.source_bytes)
                names.append(str(Path(stem) / 'roots.rsml'))
                continue
            png_name = stem + '.png'
            if not cv2.imwrite(str(staging / png_name), image.image):
                raise OSError(f'Could not write RSML image: {png_name}')
            filename = stem + '.rsml'
            ET.ElementTree(_document(image, png_name, series.group, index)).write(
                staging / filename, encoding='utf-8', xml_declaration=True)
            names.append(filename)
        # mkdir claims the target exclusively, including in competing exports.
        target.mkdir()
        try:
            for item in staging.iterdir():
                os.replace(item, target / item.name)
        except BaseException:
            shutil.rmtree(target)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [target / name for name in names]
