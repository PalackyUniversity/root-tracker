"""Edit individual detections through the durable per-image geometry store."""
import xml.etree.ElementTree as ET

from .rsml import parse_rsml
from .rsml_export import _document
from .rsml_replacement import replace_roots


def editable_document(image, *, group=None, image_index=0):
    """Return a snapshot without modifying the image on selection alone."""
    if image.rsml_document is not None:
        return image.rsml_document
    if image.rsml_samples is None:
        return None
    return parse_rsml(ET.tostring(_document(image, image.filename, group if group is not None else image.barcode, image_index)))


def edit_roots(image, document, selected, *, plant_index=None, delete=False, persist=True, mask=None):
    """Persist selected edits atomically; unselected child roots stay in place.

    XML elements retain their geometry and extensions. Children whose parent
    is removed or moves to another plant become primary roots of their plant.
    Returns the new selection indices after the document's traversal changes.
    """
    selected = set(selected)
    if not selected or not selected <= set(range(len(document.roots))):
        raise ValueError('Select an existing root detection first.')
    if not delete and (plant_index is None or not 0 <= plant_index < len(document.plant_ids)):
        raise ValueError('Choose an existing plant assignment.')
    if image.rsml_document is None and mask is not None and image.rsml_unmasked_samples is not None:
        return _edit_masked_native(image, document, selected, plant_index, delete, persist)
    if image.rsml_unmasked_document is not None and document is image.rsml_document:
        selected = {image.rsml_root_sources[index] for index in selected}
        document = image.rsml_unmasked_document
    tree = ET.fromstring(document.source_bytes)
    ns = tree.tag.split('}')[0] + '}' if tree.tag.startswith('{') else ''
    plants = tree.find(ns + 'scene').findall(ns + 'plant')
    elements = [element for plant in plants for element in plant.iter(ns + 'root')]
    owners = {child: parent for parent in tree.iter() for child in parent}
    # Detach roots first; rebuilding avoids accidentally deleting descendants.
    for element in elements:
        owners[element].remove(element)
    destinations = [plant_index if i in selected and not delete else root.plant_index
                    for i, root in enumerate(document.roots)]
    for i, (element, root) in enumerate(zip(elements, document.roots)):
        if delete and i in selected:
            continue
        parent = root.parent_index
        if (parent is not None and not (delete and parent in selected)
                and destinations[parent] == destinations[i]):
            elements[parent].append(element)
        else:
            plants[destinations[i]].append(element)
    order = [element for plant in plants for element in plant.iter(ns + 'root')]
    new_selection = {order.index(elements[i]) for i in selected} if not delete else set()
    edited = parse_rsml(ET.tostring(tree, encoding='utf-8', xml_declaration=True))
    replace_roots(image, edited, manual=True, persist=persist)
    return new_selection


def _edit_masked_native(image, document, selected, plant_index, delete, persist):
    """Edit visible detections within the full native geometry, preserving gaps.

    A native mask can remove unmasked fragments through contour filtering too.
    Starting from full pixel ownership keeps those fragments and boundary edges
    available to Restore, without restoring deliberately deleted detections.
    """
    from copy import copy
    import numpy as np
    samples = {key: {tuple(map(int, point)) for point in points}
               for key, points in image.rsml_unmasked_samples.items()}
    visible = {tuple(map(int, point)) for root in document.roots for point in root.points}
    # Masked tracking can assign a visible pixel differently. Its current
    # ownership is authoritative; otherwise deletion could reveal an old owner.
    for points in samples.values():
        points.difference_update(visible)
    retained = {key: set() for key in samples}
    changed = set()
    selected_points = {}
    for index, root in enumerate(document.roots):
        owner = int(root.plant_id) - 1
        pixels = {tuple(map(int, point)) for point in root.points}
        samples.setdefault(owner, set()).update(pixels)
        if index in selected:
            selected_points.setdefault(owner, set()).update(pixels)
            changed.update(pixels)
        else:
            retained.setdefault(owner, set()).update(pixels)
    for owner, pixels in selected_points.items():
        samples[owner].difference_update(pixels - retained.get(owner, set()))
    if not delete:
        target = int(document.plant_ids[plant_index]) - 1
        samples.setdefault(target, set()).update(changed)
    snapshot = copy(image)
    snapshot.rsml_samples = {owner: np.asarray(sorted(points), dtype=np.int32).reshape(-1, 2)
                             for owner, points in samples.items()}
    if not delete and image.main_root_samples is not None:
        transferred = [(target, x, y) for owner, x, y in image.main_root_samples
                       if (x, y) in selected_points.get(owner, set())]
        snapshot.main_root_samples = np.concatenate(
            [image.main_root_samples, np.asarray(transferred, dtype=np.int32).reshape(-1, 3)])
    # Preserve frame metadata from the visible document while replacing geometry.
    tree = ET.fromstring(document.source_bytes)
    generated = _document(snapshot, image.filename, image.barcode, 0)
    scene = tree.find('scene')
    tree.remove(scene)
    tree.append(generated.find('scene'))
    edited = parse_rsml(ET.tostring(tree, encoding='utf-8', xml_declaration=True))
    new_selection = (set() if delete else
                     {index for index, root in enumerate(edited.roots)
                      if root.plant_index == plant_index and any(tuple(point) in changed for point in root.points)})
    replace_roots(image, edited, manual=True, persist=persist)
    return new_selection
