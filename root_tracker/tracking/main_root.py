"""Select a main path by established geometry, then by new tip depth."""
import numpy as np
from scipy.spatial import cKDTree

from .junction_router import plant_ids
from .temporal_fragments import contour_path


def has_main_axis(geometry, upper):
    """A compact first observation has no resolved longitudinal identity yet."""
    pieces = [c[:, 0] for contours in geometry.values() for c in contours]
    if not pieces:
        return False
    widths = [np.asarray(u.get('width_profile', [2.])).reshape(-1)
              for u in upper if u['contour_index'] in geometry]
    extent = np.linalg.norm(np.ptp(np.concatenate(pieces), axis=0))
    return extent > np.median(np.concatenate(widths))


def _connect_main_fragments(by_tip, original_by_tip, parents, plant):
    """Recover mutually nearest forward gaps within an already assigned plant.

    Ownership may link both fragments directly to the stem. That is not a
    physical fork. Infer main ancestry only for free arrivals and free tips,
    never across a known lateral, and leave the ownership graph untouched.
    """
    def parent(node):
        return node.get('parent_points', {}).get(plant, parents.get(node['point']))

    def ancestry(node):
        point = parent(node)
        seen = set()
        while point in original_by_tip and point not in seen:
            seen.add(point)
            point = parent(original_by_tip[point])
        seen.add(point)
        return seen

    used = {parent(node) for node in by_tip.values()}
    arrivals = [node for node in by_tip.values()
                if parent(node) is not None
                and (parent(node) not in original_by_tip or parent(node) in by_tip)
                and not node.get('junction_id')]
    tips = [node for point, node in by_tip.items() if point not in used]
    paths = {node['contour_index']: contour_path(node).astype(float)
             for node in by_tip.values()}
    choices = []
    for child in arrivals:
        for upstream in tips:
            if child is upstream or parent(child) not in ancestry(upstream):
                continue
            gap = np.subtract(child['point'], upstream['lower_point']).astype(float)
            distance = np.linalg.norm(gap)
            if gap[1] <= 0 or distance == 0:
                continue
            directions, lengths = [], []
            for node, at_start in ((upstream, False), (child, True)):
                path = paths[node['contour_index']]
                if not at_start:
                    path = path[::-1]
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                lengths.append(arc[-1])
                direction = path[min(np.searchsorted(arc, distance), len(path)-1)] - path[0]
                directions.append(direction if at_start else -direction)
            # Use the adjoining observed arms, not the extent of an unrelated
            # connected component. A newly visible tip can be shorter than its
            # gap while the established upstream arm supplies the scale.
            if distance > max(lengths):
                continue
            # Each arm must point more along the gap than across it.
            if all(np.dot(gap, d) > abs(gap[0]*d[1] - gap[1]*d[0]) for d in directions):
                choices.append((distance, child['lower_point'], upstream['lower_point']))
    nearest_child, nearest_parent = {}, {}
    def record(mapping, point, distance, candidate):
        if point not in mapping:
            mapping[point] = distance, candidate
        elif mapping[point][0] == distance:
            mapping[point] = distance, None

    for distance, child, upstream in sorted(choices):
        record(nearest_parent, child, distance, upstream)
        record(nearest_child, upstream, distance, child)
    for child, (_, upstream) in nearest_parent.items():
        if upstream is not None and nearest_child[upstream][1] == child:
            node = by_tip[child]
            node['parent_points'] = dict(node.get('parent_points', {}))
            node['parent_points'][plant] = upstream


def _main_portions(upper, main_tree, lateral_tree, *, main_bridge=False):
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
        # Sparse lateral contact may not erase a bridge between established
        # main observations. Without those observations on both sides, it is
        # not evidence that an unrelated branch has become the main.
        if main_bridge and np.count_nonzero(confirmed) * 2 < len(path):
            return [upper['contour']], upper['point'], upper['lower_point']
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
    recovery_only = set()
    for u in owned:
        has_ancestor = established_ancestor(u)
        pieces, first, last = _main_portions(
            u, main_tree, lateral_tree,
            main_bridge=has_ancestor and u['contour_index'] in ancestors_of_main)
        if not pieces and has_ancestor:
            # Keep a sparsely contested moved terminal available for recovery,
            # without allowing it to vote itself into an ordinary main branch.
            recovered, start, end = _main_portions(u, main_tree, lateral_tree, main_bridge=True)
            if recovered:
                pieces, first, last = recovered, start, end
                recovery_only.add(u['contour_index'])
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

    _connect_main_fragments(by_tip, original_by_tip, parents, plant)

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

    # Counted near-pixel votes can prefer a short lateral at the old centerline
    # over a whole main that moved sideways. Compare each supported candidate
    # against the entire historical route, not just its close coincidences.
    route_distances = {}
    if previous:
        reference = np.asarray(list(previous), dtype=float)
        for index, pieces in geometries.items():
            if not pieces:
                continue
            tree = cKDTree(np.concatenate([piece[:, 0] for piece in pieces]))
            route_distances[index] = np.minimum(tree.query(reference)[0],
                                                tree.query(reference + applied_shift)[0])

    def route_quality(path, supported):
        if route_distances and supported:
            distances = [route_distances[index] for index in path if index in route_distances]
            if distances:
                return 1, -float(np.minimum.reduce(distances).mean())
        return 0, 0.

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
        own_support = support.get(u['contour_index'], 0)
        chosen, (score, path) = max(
            ((p, trace(p, visiting | {point})) for p in candidates),
            key=lambda item: (*route_quality([u['contour_index']] + item[1][1],
                                             own_support + item[1][0]),
                              physical_parent(u, item[0])))
        # New distal growth can continue an established main. An unsupported
        # basal gap parent cannot extend it backwards into a newly detected
        # leaf edge or another root; physical connections remain admissible.
        if (own_support and not score and not physical_parent(u, chosen)
                and not direct_support.intersection(path)):
            path = []
        result = own_support + score, [u['contour_index']] + path
        memo[point] = result
        return result

    # An internal arrival is not a tip when an admissible physical continuation
    # exists. At a reconnection, counting its old samples separately would let
    # the losing incoming arm beat the complete selected route.
    continued = set()
    for node in by_tip.values():
        if not geometries[node['contour_index']] or node['contour_index'] in recovery_only:
            continue
        parent = node.get('parent_points', {}).get(plant, parents.get(node['point']))
        if parent != node['lower_point']:
            continued.add(parent)
        continued.update(p for p in incoming.get(node.get('junction_id'), [])
                         if p != node['lower_point'] and p[1] <= node['point'][1])
    tips = [tip for tip in by_tip if tip not in continued
            and geometries[by_tip[tip]['contour_index']]]
    if not tips:
        tips = [tip for tip in by_tip if geometries[by_tip[tip]['contour_index']]]
    candidates = [(trace(tip, set()), tip) for tip in tips]
    established = [item for item in candidates if not recovery_only.intersection(item[0][1])]
    if not established:
        return None, {}
    (support_score, path), tip = max(established, key=lambda item: (item[0][0], item[1][1]))
    recent = set(previous or ()) & set(previous_roots or ())
    if recent and tip[1] < max(y for _, y in recent):
        previous_depth = max(y for _, y in recent)
        continuations = [item for item in candidates if item[1][1] >= previous_depth
                         and (not recovery_only.intersection(item[0][1])
                              or item[1][1] >= max(y for _, y in previous))]
        if continuations:
            # Pixel votes alone can prefer a short lateral near the old
            # centerline after the whole main moves sideways. Recover only
            # when that vote would truncate an observed established route.
            (score, alternative), endpoint = max(continuations,
                key=lambda item: (*route_quality(item[0][1], item[0][0]), item[1][1]))
            if route_quality(alternative, score) > route_quality(path, support_score):
                path, tip = alternative, endpoint
    return tip, {index: geometries[index] for index in path if geometries[index]}


def select_main_path(upper, colored, pairs, plant, previous, lowers=(), previous_roots=None):
    """Compatibility view of main geometry as original contour indices."""
    tip, geometry = select_main_geometry(upper, colored, pairs, plant, previous,
                                         lowers, previous_roots)
    return tip, list(geometry)
