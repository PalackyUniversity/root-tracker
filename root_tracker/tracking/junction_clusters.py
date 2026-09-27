"""Treat a short connector inside a crossing as a bidirectional junction."""
import numpy as np
from .temporal_fragments import contour_path


def junction_clusters(uppers, lowers):
    """Return junction representatives, internal segments and original sides.

    A connector shorter than its two endpoint diameters has no independent
    interior outside the junction bulges. Routing it as a directed root makes
    a nearly horizontal crossing depend on which endpoint sorts first.
    """
    representatives = {}
    def representative(junction):
        while junction in representatives:
            junction = representatives[junction]
        return junction

    bridges = {}
    sides = {}
    incoming, outgoing = {}, {}
    for i, upper in enumerate(uppers):
        incoming.setdefault(lowers.get(upper.get('lower_point'), {}).get('junction_id'), set()).add(i)
        outgoing.setdefault(upper.get('junction_id'), set()).add(i)
    for i, upper in enumerate(uppers):
        first = upper.get('junction_id')
        last = lowers.get(upper.get('lower_point'), {}).get('junction_id')
        sides[i] = first
        widths = upper.get('width_profile')
        if not first or not last or first == last or widths is None or not len(widths):
            continue
        # Only a level connector is ambiguous under the router's downward
        # ordering. Sloping corridors retain their directed topology.
        if upper['point'][1] != upper['lower_point'][1]:
            continue
        # A stem seed or U-shaped tip may itself be the only arrival. Collapse
        # only a corridor with independent entries and exits at BOTH ends.
        if any(not (incoming.get(j, set()) - {i})
               or not (outgoing.get(j, set()) - {i}) for j in (first, last)):
            continue
        path = contour_path(upper)
        length = np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1).sum()
        if length >= float(widths[0] + widths[-1]):
            continue
        bridges[i] = (first, last)
        a, b = representative(first), representative(last)
        if a != b:
            representatives[max(a, b)] = min(a, b)
    return representative, bridges, sides


def crossed_bridges(bridges, start, end):
    """Find the local physical route, independent of contour orientation."""
    pending = [(start, [])]
    seen = set()
    for point, path in pending:
        if point == end:
            return path
        if point in seen:
            continue
        seen.add(point)
        for index, (a, b) in bridges.items():
            if point in (a, b):
                pending.append((b if point == a else a, path + [index]))
    return []
