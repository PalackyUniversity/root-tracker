"""Select a main path by established geometry, then by new tip depth."""
import numpy as np
from scipy.spatial import cKDTree

from .junction_router import plant_ids
from .temporal_fragments import contour_path


def select_main_path(upper, colored, pairs, plant, previous, lowers=(), previous_roots=None):
    """Return tip and contour indices, preserving the preceding main path.

    Historical pixels vote for their nearest observed segment within its measured
    foreground radius. Subdivision and skeleton thinning therefore do not reset
    main-root identity. A deeper lateral can win only when history cannot tell
    the paths apart (e.g. genuinely new growth beyond the old tip).
    """
    owned = [u for u in upper if plant in plant_ids(colored.get(u.get('lower_point')))]
    if not owned:
        return None, []
    by_tip = {u['lower_point']: u for u in owned}
    parents = dict(pairs)
    support = {}
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
            matched = np.asarray(indices)[nearest[distance <= np.asarray(radii)[nearest]]]
            ids, counts = np.unique(matched, return_counts=True)
            support = dict(zip(ids, counts))

    incoming = {}
    if support:
        for lower in lowers:
            if lower.get('junction_id') and lower['point'] in by_tip:
                incoming.setdefault(lower['junction_id'], []).append(lower['point'])

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
        score, path = max((trace(p, visiting | {point}) for p in candidates),
                          key=lambda result: result[0])
        result = support.get(u['contour_index'], 0) + score, [u['contour_index']] + path
        memo[point] = result
        return result

    best = None
    for tip in by_tip:
        support_score, path = trace(tip, set())
        score = (support_score, tip[1])
        if best is None or score > best[0]:
            best = score, tip, path
    return best[1], best[2]
