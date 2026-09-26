"""Durable, per-image RSML replacements independent of disposable caches."""
import os
import shutil
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .rsml import RSMLDocument, parse_rsml
from ..analysis.rsml_statistics import apply_measurements, refresh_statistics


def replacement_path(image_path):
    path = Path(image_path)
    return path.parent / '.root_tracker_rsml' / (path.name + '.npz')


def _validate(document, background):
    if not document.plant_ids:
        raise ValueError('Replacement RSML must contain at least one plant (which may have no roots).')
    height, width = background.shape[:2]
    for root in document.roots:
        if not root.points:
            raise ValueError('Every replacement root needs polyline geometry for measurements.')
        if any(len(point) != 2 for point in root.points):
            raise ValueError('Root Tracker replacement requires 2D RSML geometry.')
        if any(not (0 <= point[0] < width and 0 <= point[1] < height) for point in root.points):
            raise ValueError('RSML coordinates do not fit this image. Use RSML in this image\'s pixel coordinates.')


def replace_roots(image, document: RSMLDocument):
    """Validate and persist before changing memory; retain the original XML."""
    # Freeze the coordinate frame at replacement time. Subsequent crops/masks
    # must not move or erase the imported roots.
    background = image.rsml_background if image.rsml_document is not None else image.image
    if background is None and image.rsml_document is not None:
        background = _load_background(image)
    if background is None:
        background = cv2.imread(image.path)
    if background is None:
        raise ValueError('Cannot load the image for this RSML replacement.')
    _validate(document, background)
    ok, png = cv2.imencode('.png', background)
    if not ok:
        raise OSError('Could not save the replacement image background.')
    destination = replacement_path(image.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.replacement-', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, rsml=np.frombuffer(document.source_bytes, dtype=np.uint8), background=png)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    image.rsml_document = document
    image.rsml_background = background.copy()
    image.image_annotated = None
    apply_measurements(image)


def load_replacement(image):
    path = replacement_path(image.path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        document = parse_rsml(data['rsml'].tobytes())
    image.rsml_document = document
    image.rsml_background = None  # Decode lazily when displaying the image.
    image.image_annotated = None
    apply_measurements(image)
    return True


def _load_background(image):
    with np.load(replacement_path(image.path), allow_pickle=False) as data:
        background = cv2.imdecode(data['background'], cv2.IMREAD_COLOR)
    if background is None:
        raise ValueError('The saved RSML replacement background is unreadable.')
    return background


def render_replacement(image):
    if image.rsml_background is None:
        image.rsml_background = _load_background(image)
    annotated = image.rsml_background.copy()
    colors = [(255, 80, 40), (40, 180, 40), (60, 60, 255), (200, 70, 200)]
    for root in image.rsml_document.roots:
        points = np.rint(np.asarray(root.points)).astype(np.int32)
        color = colors[root.plant_index % len(colors)]
        if len(points) == 1:
            cv2.circle(annotated, tuple(points[0]), 2, color, -1)
        elif len(points):
            cv2.polylines(annotated, [points], False, color, 2, cv2.LINE_AA)
    cv2.putText(annotated, 'RSML', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 80, 200), 1, cv2.LINE_AA)
    return annotated


def restore_measurements(series):
    if any(image.rsml_document is not None for image in series):
        series.pipeline_state.last_statistics = refresh_statistics(series, series.pipeline_state.last_statistics)


def move_replacement(source, destination):
    """Move the persistent replacement alongside an image moved to/from aside."""
    source = replacement_path(source)
    if source.exists():
        target = replacement_path(destination)
        if target.exists():
            raise FileExistsError(f'RSML replacement already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)


def move_image_with_replacement(source, destination):
    """Move both durable files; roll the replacement back if the image move fails."""
    had_replacement = replacement_path(source).exists()
    move_replacement(source, destination)
    try:
        shutil.move(source, destination)
    except Exception:
        if had_replacement:
            move_replacement(destination, source)
        raise
