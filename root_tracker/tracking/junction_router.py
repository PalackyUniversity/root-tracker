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
    original_uppers = uppers
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

    from .contact_evidence import ContactEvidence
    contacts = ContactEvidence(uppers, lower_by_point, incoming, tasks, by_bottom,
                               historical_owner=lambda index: evidence(index)[0])
    contact_proof = {}
    contact_extents = {}

    colored = dict(origins)
    flows = {point: {pid: 90.} for point, pid in origins.items()}
    samples = {pid: set() for pid in range(n_clusters)}
    pairs = []
    processed = set()
    previous_trees = None
    shared_previous_trees = {}
    evidence_cache = {}
    historical_fragments = {}
    proven_upgrades = {}

    def temporal_support(upper):
        nonlocal previous_trees
        if not previous_samples or 'contour' not in upper:
            return None, ()
        if previous_trees is None:
            previous_trees = {pid: cKDTree(np.asarray(list(pixels)))
                              for pid, pixels in previous_samples.items() if pixels}
        if not previous_trees:
            return None, ()
        from .temporal_fragments import contour_path
        pixels = contour_path(upper)
        # Small registration/thinning shifts must not erase otherwise strong
        # temporal evidence. Pixels near multiple plants, including previously
        # shared sections, provide no exclusive evidence for either identity.
        distances = {pid: tree.query(pixels, distance_upper_bound=8.000001)[0]
                     for pid, tree in previous_trees.items()}
        near = {pid: distance <= 2.000001 for pid, distance in distances.items()}
        nearest_history = np.min(list(distances.values()), axis=0)
        # A merged current segment may contain different historical owners.
        # Keep sustained exclusive runs individually instead of allowing the
        # majority owner to replace the minority's established pixels.
        closest = {pid: (distance <= 2.000001)
                   & (distance <= nearest_history + .000001)
                   for pid, distance in distances.items()}
        unique = np.sum(list(closest.values()), axis=0) == 1
        for pid, matches in closest.items():
            indices = np.flatnonzero(matches & unique)
            for run in np.split(indices, np.flatnonzero(np.diff(indices) > 3) + 1):
                if len(run) >= 5:
                    historical_fragments.setdefault(id(upper), []).append(
                        ((pid,), pixels[run[0]:run[-1] + 1], True))
        continuing_ids = set()
        for pair in combinations(sorted(previous_trees), 2):
            if pair not in shared_previous_trees:
                common = previous_samples.get(pair[0], set()) & previous_samples.get(pair[1], set())
                shared_previous_trees[pair] = cKDTree(np.asarray(list(common))) if common else None
            tree = shared_previous_trees[pair]
            if tree is not None:
                shared_distances = tree.query(pixels, distance_upper_bound=8.000001)[0]
                matches = (np.isfinite(shared_distances)
                           & (shared_distances <= nearest_history + .000001))
                # Measure support against observed historical pixels, not new
                # growth or a differently split/currently longer segment.
                observed = np.count_nonzero(np.any(list(near.values()), axis=0))
                if np.count_nonzero(matches) >= 5:
                    indices = np.flatnonzero(matches)
                    if (indices[-1] - indices[0] + 1 >= .6 * len(pixels)
                            and np.count_nonzero(matches) >= .6 * observed
                            and np.count_nonzero(unique) < 5):
                        continuing_ids.update(pair)
                    else:
                        boundaries = [i + 1 for i, (left, right) in enumerate(
                            zip(indices[:-1], indices[1:]))
                            if right - left > 3 or np.any(unique[left + 1:right])]
                        for run in np.split(indices, boundaries):
                            if len(run) >= 5:
                                historical_fragments.setdefault(id(upper), []).append(
                                    (pair, pixels[run[0]:run[-1] + 1], False))
        continuing = tuple(sorted(continuing_ids))
        # Broaden only when the tighter match is inconclusive. Ambiguous
        # pixels near two separate roots never vote for either identity.
        for tolerance in (2., 4., 8.):
            near = {pid: distance <= tolerance + .000001 for pid, distance in distances.items()}
            exclusive = np.sum(list(near.values()), axis=0) == 1
            overlaps = {pid: int(np.count_nonzero(matches & exclusive)) for pid, matches in near.items()}
            best = max(overlaps, key=overlaps.get)
            count = overlaps[best]
            minimum = max(5, .2 * len(pixels)) if tolerance == 2. else max(10, .5 * len(pixels))
            if count >= minimum and count >= .8 * sum(overlaps.values()):
                return (best, count / len(pixels)), continuing
        return None, continuing

    def evidence(index):
        if index not in evidence_cache:
            evidence_cache[index] = temporal_support(uppers[index])
        return evidence_cache[index]

    def downstream_owner(index, pid):
        """Require an established exit before extending a contact into a root."""
        lower = lower_by_point.get(uppers[index].get('lower_point'), {})
        children = tasks.get(('junction', lower.get('junction_id')), [])
        return any((evidence(child)[0] is not None and evidence(child)[0][0] == pid)
                   or pid in evidence(child)[1] for child in children if child != index)

    def publish(index, routes):
        upper = uppers[index]
        processed.add(index)
        if index >= real_count or 'lower_point' not in upper:
            return
        supported, continuing = evidence(index)
        # A disconnected fragment needs majority historical support. The looser
        # contact vote disambiguates rooted paths but cannot establish a root.
        if not routes and not continuing and (supported is None or supported[1] < .5):
            return
        if len(routes) > 1 and index not in contact_extents and not contact_proof.get(index):
            options_here = {pid: [route] for pid, route in routes.items()}
            winner = supported[0] if supported is not None and supported[0] in routes else next(iter(routes))
            extent = contacts.width_support(index, options_here, winner)
            if extent is not None:
                contact_extents[index] = extent
                contact_proof[index] = set(routes)
        # An exit proves continuation only for a root physically arriving at
        # this segment's entrance. A gap/temporal base link can come from one
        # old pixel at the far end; its downstream neighbor must not thereby
        # acquire the entire established upstream root.
        arrivals = incoming.get(upper.get('junction_id'), [])
        proven_upgrades[id(upper)] = {
            pid for pid, route in routes.items()
            if route[0] in arrivals and downstream_owner(index, pid)
        } | contact_proof.get(index, set())
        required = set(continuing)
        if supported is not None:
            required.add(supported[0])
            # Touching an established root is not evidence that the arriving
            # root follows it. Require its own downstream continuation.
            routes = {pid: route for pid, route in routes.items()
                      if pid in required or pid in proven_upgrades[id(upper)]}
        for pid in sorted(required):
            if pid in routes:
                continue
            candidates = [(point, flow[pid]) for point, flow in flows.items()
                          if pid in flow and point[1] <= upper['point'][1]]
            if candidates:
                routes[pid] = min(candidates, key=lambda route: sum(
                    (a - b) ** 2 for a, b in zip(route[0], upper['point'])))
        if not routes:
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
        # Carry individual calibers through a bundle, rather than treating
        # its combined width as the diameter of each participating root.
        contacts.remember(index, routes, index in contact_extents)
        extent = contact_extents.get(index)
        if extent is not None:
            from .temporal_fragments import contour_path
            path = contour_path(upper)
            cut, survivor = extent
            if cut < len(path) and survivor in routes:
                historical_fragments.setdefault(id(upper), []).extend([
                    (tuple(routes), path[:cut], False),
                    ((survivor,), path[cut:], True)])
                # The width observation proves sharing only before this tip.
                proven_upgrades[id(upper)] = set()
                flows[bottom] = {survivor: tangent}
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
            history = {index: evidence(index)
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
                    # A measured parallel bundle preserves transverse order
                    # at separation. Original entry headings can point across
                    # each other after a long curved shared section.
                    if (len(parents) == 1 and parents[0] in contacts.parallel
                            and len(indices) == len(ids)):
                        scores[row, col] = abs(row - col)
                    supported, _ = history[index]
                    if supported is not None and supported[0] != pid:
                        scores[row, col] += 180. * supported[1]
                    choices[row, col] = route
            routes = {col: {} for col in range(len(indices))}
            rows, cols = linear_sum_assignment(scores)
            for row, col in zip(rows, cols):
                routes[col][ids[row]] = choices[row, col]
            # A root may terminate at a contact. A shortage of exits alone
            # cannot establish sharing. Legacy callers without a foreground
            # profile retain the geometry-only behavior.
            for row in set(range(len(ids))) - set(rows):
                col = int(np.argmin(scores[row]))
                index = indices[col]
                if 'width_profile' in uppers[index]:
                    winner = next(iter(routes[col]))
                    candidates = set(routes[col]) | {ids[row]}
                    corridor_options = {pid: options[pid] for pid in ids if pid in candidates}
                    proof, extent = contacts.supported(index, corridor_options, winner)
                    if ids[row] not in proof:
                        continue
                    contact_proof.setdefault(index, set()).update(proof)
                    if extent is not None:
                        contact_extents[index] = extent
                routes[col][ids[row]] = choices[row, col]
            # More exits than identities: keep normal same-plant branching.
            for col in set(range(len(indices))) - set(cols):
                row = int(np.argmin(scores[:, col]))
                routes[col][ids[row]] = choices[row, col]
            # Sharing evidence uses the local contact geometry. When it
            # instead supports one new continuation, compare undistorted
            # approaches to choose its owner before creating temporal history.
            if len(indices) == 1 and len(ids) > 1 and len(routes[0]) == 1:
                index = indices[0]
                supported, continuing = history[index]
                if supported is None and not continuing and 'width_profile' in uppers[index]:
                    winner = next(iter(routes[0]))
                    owner = contacts.exclusive_owner(index, options, winner)
                    routes[0] = {owner: choices[ids.index(owner), 0]}
            for col, index in enumerate(indices):
                # A newly grown lateral is not proof that previously shared
                # roots have separated. Preserve both identities on a strongly
                # supported continuation of the existing shared geometry.
                _, continuing = history[index]
                if len(parents) == 1 and parents[0] in contacts.bundles:
                    winner = next(iter(routes[col]))
                    extent = contacts.width_support(index, options, winner)
                    if extent is not None:
                        contact_extents[index] = extent
                        contact_proof[index] = set(ids)
                        for row, pid in enumerate(ids):
                            routes[col][pid] = choices[row, col]
                if len(continuing) > 1:
                    for pid in continuing:
                        if pid in ids:
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
            # Resolve only a root dependency cycle, then retry physical
            # routing. Bulk fallback would process downstream contacts before
            # their delayed arrivals, permanently discarding sharing evidence.
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            task_by_index = {index: key for key, indices in tasks.items()
                             for index in indices}
            positions = {key: i for i, key in enumerate(pending)}
            dependencies = np.zeros((len(pending), len(pending)), dtype=bool)
            for key in pending:
                parents = list(incoming.get(key[1], [])) if key[0] == 'junction' else []
                parents.extend(base_parent.get(uppers[i]['point']) for i in tasks[key])
                for parent in parents:
                    index = by_bottom.get(parent)
                    dependency = task_by_index.get(index)
                    if index is not None and dependency is None:
                        # An internal connector is published by its junction
                        # task rather than being a standalone task.
                        dependency = ('junction', lower_by_point.get(parent, {}).get('junction_id'))
                    if index not in processed and dependency in positions:
                        dependencies[positions[key], positions[dependency]] = True
            _, components = connected_components(
                csr_matrix(dependencies), directed=True, connection='strong')
            component_dependencies = {
                component: set(components[np.any(dependencies[components == component], axis=0)]) - {component}
                for component in dict.fromkeys(components)}
            order, remaining = [], set(component_dependencies)
            while remaining:
                roots = [component for component in component_dependencies
                         if component in remaining
                         and not (component_dependencies[component] & remaining)]
                order.extend(roots)
                remaining.difference_update(roots)
            # If an upstream cycle has no rooted base link at all, it cannot
            # recover. Try the next component's own grounded links rather than
            # abandon an otherwise recoverable downstream fragment.
            for component in order:
                members = np.flatnonzero(components == component)
                for position in members:
                    key = pending[position]
                    process_task(key, tasks[key], physical=False)
                    if len(processed) > before:
                        break
                if len(processed) > before:
                    break
            pending = [key for key in pending if any(i not in processed for i in tasks[key])]
            if len(processed) == before:
                break
    if historical_fragments:
        from .temporal_fragments import preserve_historical_fragments
        preserve_historical_fragments(
            original_uppers, historical_fragments, pairs, colored, samples, proven_upgrades)
    return pairs, colored, samples
