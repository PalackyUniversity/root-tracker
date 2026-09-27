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


def replace_roots(image, document: RSMLDocument, *, manual=False, persist=True):
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
    if manual and image.rsml_document is None and image.tracking_overlay is not None:
        background = background.copy()
        pixels = image.tracking_overlay
        background[pixels[:, 0], pixels[:, 1]] = pixels[:, 2:]
    _validate(document, background)
    original = image.rsml_original_document
    if manual and image.rsml_document is not None and image.rsml_unmasked_document is None:
        original = image.rsml_document
    if persist:
        _save_replacement(image, document, background, manual, original)
    image.rsml_document = document
    image.rsml_unmasked_document = document if manual else None
    image.rsml_root_sources = tuple(range(len(document.roots))) if manual else ()
    image.rsml_original_document = original if manual else None
    # Once frozen, the background can be shared by successive save snapshots.
    image.rsml_background = background if image.rsml_background is background else background.copy()
    image.image_annotated = None
    apply_measurements(image)


def _save_replacement(image, document, background, manual, original):
    destination = replacement_path(image.path)
    # Geometry edits do not change the frozen image. Reuse its PNG instead of
    # re-encoding millions of pixels on every deletion or reassignment.
    if image.rsml_document is not None and destination.exists():
        with np.load(destination, allow_pickle=False) as data:
            png = data['background']
    else:
        ok, png = cv2.imencode('.png', background)
        if not ok:
            raise OSError('Could not save the replacement image background.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.replacement-', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, rsml=np.frombuffer(document.source_bytes, dtype=np.uint8), background=png,
                     manual=manual, original=np.frombuffer(original.source_bytes if original and manual else b'', dtype=np.uint8))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reset_manual_roots(image):
    """Restore an import or return a manually edited image to native tracking."""
    if image.rsml_unmasked_document is None:
        return False
    if image.rsml_original_document is not None:
        replace_roots(image, image.rsml_original_document)
    else:
        replacement_path(image.path).unlink(missing_ok=True)
        image.rsml_document = image.rsml_unmasked_document = None
        image.rsml_original_document = image.rsml_background = None
        image.rsml_root_sources = ()
        image.clear_tracking_results()
    return True


def load_replacement(image):
    path = replacement_path(image.path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        document = parse_rsml(data['rsml'].tobytes())
        original = parse_rsml(data['original'].tobytes()) if 'original' in data and data['original'].size else None
        # Earlier manual corrections did not store an explicit origin flag.
        import xml.etree.ElementTree as ET
        tree = ET.fromstring(document.source_bytes)
        ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
        manual = (bool(data['manual']) if 'manual' in data else
                  tree.findtext(f'{ns}metadata/{ns}software') == 'Root Tracker')
    image.rsml_document = document
    image.rsml_unmasked_document = document if manual else None
    image.rsml_root_sources = tuple(range(len(document.roots))) if manual else ()
    image.rsml_original_document = original
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


def root_colors(document):
    """Keep native plant colors when native geometry is manually corrected."""
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(document.source_bytes)
    ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
    if tree.findtext(f'{ns}metadata/{ns}software') == 'Root Tracker':
        from ..tracking.root_linker import PLANT_COLORS
        return PLANT_COLORS
    return [(255, 80, 40), (40, 180, 40), (60, 60, 255), (200, 70, 200)]


def render_replacement(image, *, show_max_root_depth=True):
    import json
    import xml.etree.ElementTree as ET
    if image.rsml_background is None:
        image.rsml_background = _load_background(image)
    annotated = image.rsml_background.copy()
    colors = root_colors(image.rsml_document)
    tree = ET.fromstring(image.rsml_document.source_bytes)
    ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
    elements = list(tree.find(ns + 'scene').iter(ns + 'root'))
    highlights = []
    native = any('root-tracker-main-points' in element.attrib for element in elements)
    for root, element in zip(image.rsml_document.roots, elements):
        points = np.rint(np.asarray(root.points)).astype(np.int32)
        color = colors[root.plant_index % len(colors)]
        if len(points) == 1:
            cv2.circle(annotated, tuple(points[0]), 2, color, -1)
        elif len(points):
            cv2.polylines(annotated, [points], False, color, 3 if native else 2, cv2.LINE_AA)
        main = {tuple(point) for point in json.loads(element.get('root-tracker-main-points', '[]'))}
        marked = np.array([tuple(point) in main for point in points], dtype=bool)
        transitions = np.diff(np.r_[False, marked, False].astype(np.int8))
        bright = tuple(min(c + 170, 255) for c in color)
        for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            highlights.append((points[start:end], bright))
    for points, color in highlights:
        if len(points) == 1:
            cv2.circle(annotated, tuple(points[0]), 6, color, -1)
        else:
            cv2.polylines(annotated, [points], False, color, 12, cv2.LINE_AA)
    if native and show_max_root_depth:
        for index in range(len(image.rsml_document.plant_ids)):
            points = [p for r in image.rsml_document.roots if r.plant_index == index for p in r.points]
            if not points:
                continue
            top = tuple(map(int, min(points, key=lambda p: p[1])))
            bottom = tuple(map(int, max(points, key=lambda p: p[1])))
            cv2.line(annotated, top, (top[0] + 100, top[1]), (255, 255, 255), 1)
            cv2.line(annotated, (top[0] + 100, top[1]), (top[0] + 100, bottom[1]), (255, 255, 255), 3)
            cv2.line(annotated, (top[0] + 100, bottom[1]), bottom, (255, 255, 255), 1)
    cv2.putText(annotated, 'Edited' if image.rsml_unmasked_document is not None else 'RSML', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 80, 200), 1, cv2.LINE_AA)
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
