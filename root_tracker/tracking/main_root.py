"""Select a main path by established geometry, then by new tip depth."""
import numpy as np
from scipy.spatial import cKDTree

from .junction_router import plant_ids
from .temporal_fragments import contour_path


def _main_portions(upper, main_tree, lateral_tree):
    """Return current contour pieces with established lateral runs excluded."""
    if main_tree is None or lateral_tree is None:
        return [upper['contour']], upper['point'], upper['lower_point']
    path = contour_path(upper)
    width = np.asarray(upper.get('width_profile', np.full(len(path), 2.))).reshape(-1)
    if len(width) != len(path):
        width = np.full(len(path), np.median(width) if len(width) else 2.)
    radius = np.maximum(np.sqrt(2), width / 2)
    main_distance = main_tree.query(path)[0]
    lateral_distance = lateral_tree.query(path)[0]
    main = main_distance <= radius
    lateral = (lateral_distance <= radius) & ~main
    def sustained(mask):
        confirmed = np.zeros(len(path), bool)
        indices = np.flatnonzero(mask)
        # Reuse ownership's sustained-run criterion; a single junction contact
        # cannot relabel the adjoining root's role.
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1):
            if len(run) >= 5:
                confirmed[run[0]:run[-1]+1] = True
        return confirmed

    confirmed = sustained(lateral) & ~main
    main = sustained(main)
    if not confirmed.any():
        return [upper['contour']], upper['point'], upper['lower_point']
    if not main.any():
        return [], None, None
    # A merged incoming lateral must not become the main's ancestry. Beyond
    # the first confirmed main run, keep distal continuation: an old gap in
    # the main can have left intermediate observed pixels without a main role.
    # Absence of that role is not evidence that the root branches away.
    confirmed[np.flatnonzero(main)[0]:] = False
    if not confirmed.any():
        return [upper['contour']], upper['point'], upper['lower_point']
    seeds = np.flatnonzero(main | confirmed)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    right = np.minimum(np.searchsorted(seeds, np.arange(len(path))), len(seeds)-1)
    left = np.maximum(right-1, 0)
    nearest = np.where(abs(arc - arc[seeds[left]]) <= abs(arc[seeds[right]] - arc),
                       seeds[left], seeds[right])
    keep = ~confirmed[nearest]
    indices = np.flatnonzero(keep)
    pieces = []
    for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if len(run):
            points = path[run]
            # Open curves must retrace themselves: drawContours(FILLED) would
            # otherwise close the endpoints across unobserved background.
            closed = np.concatenate((points, points[-2:0:-1]))
            pieces.append(closed.reshape(-1, 1, 2))
    return pieces, tuple(path[indices[0]]), tuple(path[indices[-1]])


def select_main_geometry(upper, colored, pairs, plant, previous, lowers=(), previous_roots=None):
    """Return the tip and contour pieces preserving the preceding main path.

    Historical pixels vote for their nearest observed segment within its measured
    foreground radius. Subdivision and skeleton thinning therefore do not reset
    main-root identity. A deeper lateral can win only when history cannot tell
    the paths apart (e.g. genuinely new growth beyond the old tip).
    """
    owned = [u for u in upper if plant in plant_ids(colored.get(u.get('lower_point')))]
    if not owned:
        return None, {}
    parents = dict(pairs)
    support = {}
    applied_shift = np.zeros(2)
    if previous:
        points, indices, radii = [], [], []
        for u in owned:
            xy = contour_path(u)
            width = np.asarray(u.get('width_profile', np.full(len(xy), 2.))).reshape(-1)
            if len(width) != len(xy):
                width = np.full(len(xy), np.median(width) if len(width) else 2.)
            points.extend(xy)
            indices.extend([u['contour_index']] * len(xy))
            # One diagonal raster step also accommodates skeleton tie-breaking.
            radii.extend(np.maximum(np.sqrt(2), width / 2))
        if points:
            tree = cKDTree(points)
            history = np.asarray(list(previous), dtype=float)
            shift = np.zeros(2)
            # Only other established roots may estimate residual motion. Fitting
            # the main itself can snap a vanished main onto a nearby lateral.
            anchors = set(previous_roots or ()) - set(previous)
            if anchors:
                anchors = np.asarray(list(anchors), dtype=float)
                seen = set()
                while tuple(shift) not in seen:
                    seen.add(tuple(shift))
                    _, nearest = tree.query(anchors + shift)
                    update = np.rint(np.median(np.asarray(points)[nearest] - anchors, axis=0))
                    if np.array_equal(update, shift):
                        break
                    shift = update
            distance, nearest = tree.query(history)
            if np.any(shift):
                shifted_distance, shifted_nearest = tree.query(history + shift)
                radius = np.asarray(radii)
                direct_support = np.count_nonzero(distance <= radius[nearest])
                shifted_support = np.count_nonzero(shifted_distance <= radius[shifted_nearest])
                # Missing lateral anchors must not drag an observed main away.
                # Prefer no motion unless independent alignment improves support.
                if shifted_support > direct_support:
                    distance, nearest = shifted_distance, shifted_nearest
                    applied_shift = shift
            matched = np.asarray(indices)[nearest[distance <= np.asarray(radii)[nearest]]]
            ids, counts = np.unique(matched, return_counts=True)
            support = dict(zip(ids, counts))

    # A current segment can contain both an old main and an old lateral.
    # Preserve those roles before tracing; otherwise a merged contour promotes
    # its entire lateral prefix, and even its upstream ancestors, to main.
    main_tree = (cKDTree(np.vstack((np.asarray(list(previous)),
                                    np.asarray(list(previous)) + applied_shift)))
                 if previous else None)
    lateral = set(previous_roots or ()) - set(previous)
    lateral_tree = cKDTree(np.asarray(list(lateral)) + applied_shift) if lateral else None
    original_by_tip = {u['lower_point']: u for u in owned}

    def established_ancestor(node):
        visited = set()
        parent = node.get('parent_points', {}).get(plant, parents.get(node['point']))
        while parent in original_by_tip and parent not in visited:
            visited.add(parent)
            ancestor = original_by_tip[parent]
            if support.get(ancestor['contour_index'], 0):
                return True
            parent = ancestor.get('parent_points', {}).get(plant, parents.get(ancestor['point']))
        return False

    ancestors_of_main = set()
    for node in owned:
        if not support.get(node['contour_index'], 0):
            continue
        visited = set()
        parent = node.get('parent_points', {}).get(plant, parents.get(node['point']))
        while parent in original_by_tip and parent not in visited:
            visited.add(parent)
            ancestor = original_by_tip[parent]
            ancestors_of_main.add(ancestor['contour_index'])
            parent = ancestor.get('parent_points', {}).get(plant, parents.get(ancestor['point']))

    geometries, by_tip = {}, {}
    for u in owned:
        pieces, first, last = _main_portions(u, main_tree, lateral_tree)
        has_ancestor = established_ancestor(u)
        if not pieces:
            if not (has_ancestor and u['contour_index'] in ancestors_of_main):
                continue
            # Keep graph connectivity between established main observations
            # without promoting an ambiguous intervening contour to main.
            first, last = u['point'], u['lower_point']
        geometries[u['contour_index']] = pieces
        node = dict(u, lower_point=last)
        if first != u['point'] and not has_ancestor:
            node['parent_points'] = {plant: None}
            node.pop('junction_id', None)
        by_tip[last] = node
    if not by_tip:
        return None, {}

    incoming = {}
    if support:
        for lower in lowers:
            if lower.get('junction_id') and lower['point'] in by_tip:
                incoming.setdefault(lower['junction_id'], []).append(lower['point'])

    lower_by_point = {lower['point']: lower for lower in lowers}

    def physical_parent(node, point):
        junction = node.get('junction_id')
        if junction and lower_by_point.get(point, {}).get('junction_id') == junction:
            return True
        component = node.get('component_id')
        return component is not None and by_tip.get(point, {}).get('component_id') == component

    direct_support = set()
    if previous:
        direct_tree = cKDTree(np.asarray(list(previous)))
        for node in by_tip.values():
            path = contour_path(node)
            width = np.asarray(node.get('width_profile', [2.]))
            radius = max(np.sqrt(2), float(np.median(width)) / 2)
            if np.any(direct_tree.query(path)[0] <= radius):
                direct_support.add(node['contour_index'])

    memo = {}

    def trace(point, visiting):
        if point not in by_tip or point in visiting:
            return 0, []
        if point in memo and not set(memo[point][1]).intersection(
                by_tip[p]["contour_index"] for p in visiting):
            return memo[point]
        u = by_tip[point]
        parent = u.get('parent_points', {}).get(plant, parents.get(u['point']))
        candidates = [parent]
        # Reconnection can give one plant several physical upstream routes.
        # Preserve its established main route instead of inheriting whichever
        # parent the ownership solver happened to use for that same plant.
        candidates.extend(p for p in incoming.get(u.get('junction_id'), [])
                          if p != point and p[1] <= point[1] and p != parent)
        chosen, (score, path) = max(
            ((p, trace(p, visiting | {point})) for p in candidates),
            key=lambda item: (item[1][0], physical_parent(u, item[0])))
        own_support = support.get(u['contour_index'], 0)
        # New distal growth can continue an established main. An unsupported
        # basal gap parent cannot extend it backwards into a newly detected
        # leaf edge or another root; physical connections remain admissible.
        if (own_support and not score and not physical_parent(u, chosen)
                and not direct_support.intersection(path)):
            path = []
        result = own_support + score, [u['contour_index']] + path
        memo[point] = result
        return result

    best = None
    for tip in by_tip:
        if not geometries[by_tip[tip]['contour_index']]:
            continue
        support_score, path = trace(tip, set())
        score = (support_score, tip[1])
        if best is None or score > best[0]:
            best = score, tip, path
    if best is None:
        return None, {}
    return best[1], {index: geometries[index] for index in best[2] if geometries[index]}


def select_main_path(upper, colored, pairs, plant, previous, lowers=(), previous_roots=None):
    """Compatibility view of main geometry as original contour indices."""
    tip, geometry = select_main_geometry(upper, colored, pairs, plant, previous,
                                         lowers, previous_roots)
    return tip, list(geometry)
