"""Orient unsupported upward tips from their observed root junction."""
import numpy as np
from .temporal_fragments import contour_path


def orient_terminals(uppers, lowers, pairs, colored, origins, previous):
    """Prefer a rooted physical contact to a gap longer than the visible twig.

    A free upper endpoint is not necessarily a new root: laterals can grow
    upward. Keep stem arrivals, established ownership and short supported gaps
    unchanged. Reverse only terminals whose other end has an independently
    rooted incoming segment at its physical junction.
    """
    by_point = {lower['point']: lower for lower in lowers}
    by_end = {upper.get('lower_point'): upper for upper in uppers}
    parents = dict(pairs)
    for upper in uppers:
        top, bottom = upper['point'], upper.get('lower_point')
        lower = by_point.get(bottom)
        if upper.get('junction_id') or lower is None or not lower.get('junction_id'):
            continue
        pixels = set(map(tuple, upper['contour'][:, 0]))
        if any(pixels & points for points in (previous or {}).values()):
            continue
        parent = parents.get(top)
        if parent in origins:
            continue
        length = np.linalg.norm(np.diff(contour_path(upper).astype(float), axis=0), axis=1).sum()
        if parent is not None and np.linalg.norm(np.subtract(top, parent)) <= length:
            continue
        junction = lower['junction_id']
        def independent(point):
            # A base-linked descendant may already be colored and return to
            # this junction. It cannot root its own ancestor after reversal.
            seen = {bottom}
            while point in by_end:
                if point in seen:
                    return False
                seen.add(point)
                point = parents.get(by_end[point]['point'])
            return point not in seen
        arrivals = [corner['point'] for corner in lowers
                    if corner['point'] != bottom and corner.get('junction_id') == junction
                    and corner['point'] in colored and independent(corner['point'])]
        if not arrivals:
            continue
        anchor = min(arrivals, key=lambda point: np.linalg.norm(np.subtract(point, bottom)))
        upper['point'], upper['lower_point'] = bottom, top
        upper['angle'], lower['angle'] = lower['angle'], upper['angle']
        upper['junction_id'] = lower.pop('junction_id')
        if 'width_profile' in upper:
            upper['width_profile'] = upper['width_profile'][::-1].copy()
        lower['point'] = top
        by_point.pop(bottom)
        by_point[top] = lower
        by_end.pop(bottom)
        by_end[top] = upper
        colored.pop(bottom, None)
        # Other detached fragments may already use the old bottom as a gap
        # anchor. It now faces INTO this junction, so retain those connections
        # through the independently rooted arrival instead of orphaning them.
        pairs[:] = [(child, anchor if previous_anchor == bottom else previous_anchor)
                    for child, previous_anchor in pairs if child != top]
        parents = dict(pairs)
