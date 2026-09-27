"""Route plant identities through physical skeleton junctions."""

import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def plant_ids(assignment):
    """Singleton assignments remain integers; shared assignments preserve order."""
    return assignment if isinstance(assignment, tuple) else (assignment,)


def route_junctions(uppers, lowers, origins, base_pairs, n_clusters, previous_samples=None):
    """Resolve physical merges/splits without inventing connections by proximity.

    Each shared flow retains its pre-merge tangent until it splits. A joint
    minimum-turn assignment reserves a distinct exit for each incoming plant
    when enough exits exist. Additional exits are ordinary lateral branches.
    """
    # Short terminal twigs removed by the length filter reserve their real
    # outgoing direction, so an arriving root need not steal another exit.
    real_count = len(uppers)
    sinks = [dict(exit_corner)
             for upper in uppers for exit_corner in upper.get('terminal_exits', [])]
    uppers = list(uppers) + sinks
    lower_by_point = {c['point']: c for c in lowers}
    by_bottom = {u['lower_point']: i for i, u in enumerate(uppers) if 'lower_point' in u}
    base_parent = dict(base_pairs)
    incoming = {}
    tasks = {}
    for corner in lowers:
        if corner.get('junction_id'):
            incoming.setdefault(corner['junction_id'], []).append(corner['point'])
    for i, upper in enumerate(uppers):
        upper.pop('parent_points', None)
        junction = upper.get('junction_id')
        key = ('junction', junction) if junction else ('segment', i)
        tasks.setdefault(key, []).append(i)

    colored = dict(origins)
    flows = {point: {pid: 90.} for point, pid in origins.items()}
    samples = {pid: set() for pid in range(n_clusters)}
    pairs = []
    processed = set()
    previous_trees = None
    shared_previous_trees = {}

    def temporal_support(upper, ids):
        nonlocal previous_trees
        if not previous_samples or 'contour' not in upper:
            return None, ()
        if previous_trees is None:
            previous_trees = {pid: cKDTree(np.asarray(list(pixels)))
                              for pid, pixels in previous_samples.items() if pixels}
        if not previous_trees:
            return None, ()
        pixels = np.unique(upper['contour'][:, 0], axis=0)
        # Small registration/thinning shifts must not erase otherwise strong
        # temporal evidence. Pixels near multiple plants, including previously
        # shared sections, provide no exclusive evidence for either identity.
        near = {pid: np.isfinite(tree.query(pixels, distance_upper_bound=2.000001)[0])
                for pid, tree in previous_trees.items()}
        continuing_ids = set()
        for pair in combinations(sorted(ids), 2):
            if pair not in shared_previous_trees:
                common = previous_samples.get(pair[0], set()) & previous_samples.get(pair[1], set())
                shared_previous_trees[pair] = cKDTree(np.asarray(list(common))) if common else None
            tree = shared_previous_trees[pair]
            if tree is not None:
                matches = np.isfinite(tree.query(pixels, distance_upper_bound=2.000001)[0])
                if np.count_nonzero(matches) >= max(5, .6 * len(pixels)):
                    continuing_ids.update(pair)
        continuing = tuple(pid for pid in ids if pid in continuing_ids)
        exclusive = np.sum(list(near.values()), axis=0) == 1
        overlaps = {pid: int(np.count_nonzero(matches & exclusive)) for pid, matches in near.items()}
        best = max(overlaps, key=overlaps.get)
        count = overlaps[best]
        if best in ids and count >= max(5, .2 * len(pixels)) and count >= .8 * sum(overlaps.values()):
            return (best, count / len(pixels)), continuing
        return None, continuing

    def publish(index, routes):
        upper = uppers[index]
        processed.add(index)
        if index >= real_count or not routes or 'lower_point' not in upper:
            return
        ids = tuple(routes)
        bottom = upper['lower_point']
        colored[bottom] = ids[0] if len(ids) == 1 else ids
        upper['parent_points'] = {pid: value[0] for pid, value in routes.items()}
        for parent in dict.fromkeys(upper['parent_points'].values()):
            pairs.append((upper['point'], parent))
        tangent = lower_by_point.get(bottom, {}).get('angle', (upper['angle'] + 180) % 360)
        flows[bottom] = {pid: value[1] if len(ids) > 1 else tangent
                         for pid, value in routes.items()}
        if 'contour' in upper:
            pixels = set(map(tuple, upper['contour'][:, 0].tolist()))
            for pid in ids:
                samples[pid].update(pixels)

    def ready(point):
        return point not in by_bottom or by_bottom[point] in processed

    def process_task(key, indices, physical=True):
        indices = [i for i in indices if i not in processed]
        if not indices:
            return
        parents = incoming.get(key[1], []) if key[0] == 'junction' and physical else []
        if parents and not all(ready(point) for point in parents):
            return
        options = {}
        # Stable image-left to image-right order, including a shared flow's
        # preserved order when both plants have the same endpoint.
        for point in sorted(parents, key=lambda p: p[0]):
            for pid, angle in flows.get(point, {}).items():
                options.setdefault(pid, []).append((point, angle))
        if options:
            ids = list(options)
            indices.sort(key=lambda i: (uppers[i]['point'][0], uppers[i]['lower_point'][0]))
            scores = np.empty((len(ids), len(indices)))
            choices = {}
            history = {index: temporal_support(uppers[index], ids) if len(ids) > 1 else (None, ())
                       for index in indices}
            for row, pid in enumerate(ids):
                for col, index in enumerate(indices):
                    angle = (uppers[index]['angle'] + 180) % 360
                    def turn(route):
                        return abs((angle - route[1] + 180) % 360 - 180)
                    route = min(options[pid], key=turn)
                    # Only break equal-direction ties by lateral order.
                    scores[row, col] = turn(route) + abs(
                        row / max(1, len(ids) - 1) - col / max(1, len(indices) - 1)) * 1e-6
                    supported, _ = history[index]
                    if supported is not None and supported[0] != pid:
                        scores[row, col] += 180. * supported[1]
                    choices[row, col] = route
            routes = {col: {} for col in range(len(indices))}
            rows, cols = linear_sum_assignment(scores)
            for row, col in zip(rows, cols):
                routes[col][ids[row]] = choices[row, col]
            # Fewer exits than identities: keep the remaining identities in
            # the best shared exit, instead of discarding them at a merge.
            for row in set(range(len(ids))) - set(rows):
                col = int(np.argmin(scores[row]))
                routes[col][ids[row]] = choices[row, col]
            # More exits than identities: keep normal same-plant branching.
            for col in set(range(len(indices))) - set(cols):
                row = int(np.argmin(scores[:, col]))
                routes[col][ids[row]] = choices[row, col]
            for col, index in enumerate(indices):
                # A newly grown lateral is not proof that previously shared
                # roots have separated. Preserve both identities on a strongly
                # supported continuation of the existing shared geometry.
                _, continuing = history[index]
                if len(continuing) > 1:
                    for pid in continuing:
                        routes[col][pid] = choices[ids.index(pid), col]
                publish(index, {pid: routes[col][pid] for pid in ids if pid in routes[col]})
        else:
            # Away from physical junctions retain the existing gap/temporal
            # link, but inherit its current identities rather than stale IDs.
            for index in sorted(indices, key=lambda i: uppers[i]['point'][1]):
                parent = base_parent.get(uppers[index]['point'])
                if parent is not None and not ready(parent):
                    continue
                publish(index, {pid: (parent, angle)
                                for pid, angle in flows.get(parent, {}).items()})

    pending = list(tasks)
    while pending:
        before = len(processed)
        for key in pending:
            process_task(key, tasks[key])
        pending = [key for key in pending if any(i not in processed for i in tasks[key])]
        if len(processed) == before:
            # A folded/ambiguous junction may not form a directed acyclic
            # graph. Fall back to its rooted base links rather than fabricate
            # a cycle or a connection to an unassigned fragment.
            for key in pending:
                process_task(key, tasks[key], physical=False)
            pending = [key for key in pending if any(i not in processed for i in tasks[key])]
            if len(processed) == before:
                break
    return pairs, colored, samples
