"""Measurements from authoritative RSML geometry, never from old pixel results."""
import math


def plant_slots(document):
    """Keep standard 1-based plant IDs; otherwise use document order for display.

    Arbitrary IDs remain in rsml_plant_id and are matched by that identity when
    comparing two imports. They are not guessed to match native tracker slots.
    """
    try:
        ids = [int(value) for value in document.plant_ids]
        if sorted(ids) == list(range(1, len(ids) + 1)):
            return ids, True
    except ValueError:
        pass
    return list(range(1, len(document.plant_ids) + 1)), False


def image_statistics(image):
    doc = image.rsml_document
    slots, _ = plant_slots(doc)
    grouped = [[] for _ in doc.plant_ids]
    for root in doc.roots:
        grouped[root.plant_index].append(root)
    total = sum(root.length for root in doc.roots)
    records = []
    for index, roots in enumerate(grouped):
        primary = [root for root in roots if root.parent_index is None]
        main = max(primary, key=lambda root: root.length, default=None)
        base = main.points[0] if main is not None and main.points else None
        ys = [point[1] for point in main.points] if main is not None else []
        records.append(dict(
            image_date=image.date, image_barcode=image.barcode,
            image_barcode_read=image.barcode_read, image_path=image.path,
            image_total_area=None, image_total_length=total,
            image_new_area=None, image_new_parts=None, image_area_change=None,
            plant_id=slots[index], plant_center_x=base[0] if base else None,
            plant_center_y=base[1] if base else None, plant_green_area=None,
            plant_root_count=len(roots), plant_total_length=sum(root.length for root in roots),
            plant_total_length_RGR=None,
            plant_main_root_depth=max(ys) - min(ys) if ys else 0.,
            plant_main_root_length=main.length if main is not None else 0.,
            plant_main_root_length_RGR=None,
            root_source='rsml', rsml_plant_id=doc.plant_ids[index],
        ))
    return records


def apply_measurements(image):
    records = image_statistics(image)
    image.total_length = sum(record['plant_total_length'] for record in records)
    image.total_area = image.new_area = image.new_parts = None
    image.plant_length = [0.] * len(records)
    image.longest = [0.] * len(records)
    for record in records:
        index = record['plant_id'] - 1
        image.plant_length[index] = record['plant_total_length']
        image.longest[index] = record['plant_main_root_length']
    # Imported roots must not become seeds for later native tracking.
    image.colored_samples = {}
    image.rsml_samples = None


def _rgr(current, previous, days):
    if current is None or previous is None or current <= 0 or previous <= 0 or days <= 0:
        return None
    return (math.log(current) - math.log(previous)) / days


def refresh_statistics(series, statistics):
    """Replace stale rows and update rates on both sides of replaced frames."""
    cached = {}
    for item in statistics:
        record = item.to_dict() if hasattr(item, 'to_dict') else dict(item)
        cached.setdefault(record['image_path'], []).append(record)
    result = []
    previous_image = None
    previous_records = []
    for image in series.images:
        if image.rsml_document is not None:
            from ..io.root_mask import apply_root_mask
            apply_root_mask(image, series.user_mask)
            apply_measurements(image)
            records = image_statistics(image)
        else:
            records = cached.get(image.path, [])
        previous_imported = previous_image is not None and previous_image.rsml_document is not None
        if image.rsml_document is not None or previous_imported:
            days = (image.date - previous_image.date).days if previous_image else 0
            current_numeric = image.rsml_document is None or plant_slots(image.rsml_document)[1]
            previous_numeric = not previous_imported or plant_slots(previous_image.rsml_document)[1]
            for record in records:
                prev = None
                if current_numeric and previous_numeric:
                    prev = next((r for r in previous_records if r['plant_id'] == record['plant_id']), None)
                elif image.rsml_document is not None and previous_imported:
                    source_id = record.get('rsml_plant_id')
                    matches = [r for r in previous_records if source_id and r.get('rsml_plant_id') == source_id]
                    if len(matches) == 1 and sum(r.get('rsml_plant_id') == source_id for r in records) == 1:
                        prev = matches[0]
                for key in ('plant_total_length', 'plant_main_root_length'):
                    record[key + '_RGR'] = _rgr(record.get(key), prev.get(key) if prev else None, days)
                # Pixel difference measurements cannot measure growth from an
                # imported polyline whose segmented root area is unknown.
                record['image_new_area'] = None
                record['image_new_parts'] = None
                record['image_area_change'] = None
            if previous_imported:
                image.new_area = image.new_parts = None
        for record in records:
            record.setdefault('root_source', 'tracked')
        result.extend(records)
        previous_image, previous_records = image, records
    return result
