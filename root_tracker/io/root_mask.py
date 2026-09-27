"""Apply reversible group exclusions to manually corrected root geometry."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

from .rsml import parse_rsml


def visible_runs(points, mask):
    """Split a polyline at masked pixels, including between sparse vertices."""
    points = np.asarray(points, dtype=float)
    if not len(points):
        return []
    samples = [points[:1]]
    for start, end in zip(points, points[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)))))
        samples.append(np.linspace(start, end, count + 1)[1:])
    samples = np.concatenate(samples)
    pixels = np.rint(samples).astype(np.int64)
    height, width = mask.shape[:2]
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    visible = np.ones(len(samples), dtype=bool)
    visible[inside] = mask[pixels[inside, 1], pixels[inside, 0]] == 0
    if visible.all():
        return [points]
    transitions = np.diff(np.r_[False, visible, False].astype(np.int8))
    return [samples[start:end] for start, end in zip(np.flatnonzero(transitions == 1),
                                                   np.flatnonzero(transitions == -1))]


def masked_document(document, mask):
    """Return visible geometry plus its root indices in the unmasked document."""
    if mask is None or not np.any(mask):
        return document, tuple(range(len(document.roots)))
    tree = ET.fromstring(document.source_bytes)
    ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
    plants = tree.find(ns + 'scene').findall(ns + 'plant')
    elements = [element for plant in plants for element in plant.iter(ns + 'root')]
    owners = {child: parent for parent in tree.iter() for child in parent}
    for element in elements:
        owners[element].remove(element)
    fragments = []
    source_indices = {}
    for index, (root, element) in enumerate(zip(document.roots, elements)):
        runs = visible_runs(root.points, mask)
        fragments.append([])
        for part, points in enumerate(runs):
            fragment = deepcopy(element)
            if len(runs) > 1:
                fragment.set('id', f'{root.id}-visible-{part + 1}')
            polyline = fragment.find(f'{ns}geometry/{ns}polyline')
            for point in list(polyline):
                if point.tag == ns + 'point':
                    polyline.remove(point)
            for x, y in points:
                ET.SubElement(polyline, ns + 'point', x=str(x), y=str(y))
            parent_parts = fragments[root.parent_index] if root.parent_index is not None else []
            parent = parent_parts[0] if parent_parts else plants[root.plant_index]
            parent.append(fragment)
            fragments[-1].append(fragment)
            source_indices[fragment] = index
    mapping = tuple(source_indices[element] for plant in plants for element in plant.iter(ns + 'root'))
    return parse_rsml(ET.tostring(tree, encoding='utf-8', xml_declaration=True)), mapping


def apply_root_mask(image, mask):
    """Keep stored corrections intact, replacing only their effective geometry."""
    baseline = image.rsml_unmasked_document
    if baseline is None:
        return
    document, mapping = masked_document(baseline, mask)
    if image.rsml_document.source_bytes != document.source_bytes:
        image.rsml_document = document
        image.image_annotated = None
    image.rsml_root_sources = mapping
