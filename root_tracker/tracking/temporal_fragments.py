"""Retain historical ownership boundaries within current root segments."""

import numpy as np


def contour_path(upper):
    points = upper['contour'][:, 0]
    start = np.flatnonzero(np.all(points == upper['point'], axis=1))[0]
    ordered = np.roll(points, -int(start), axis=0)
    end = np.flatnonzero(np.all(ordered == upper['lower_point'], axis=1))[0]
    return ordered[:int(end) + 1]


def preserve_historical_fragments(uppers, history, pairs, colored, samples, upgrades):
    """Split the caller's segments at historical ownership boundaries.

    Every fragment gets its own ownership and parent endpoints, keeping drawing,
    root measurements and exported samples consistent with the routed graph.
    """
    redirects = {}
    for upper in list(uppers):
        shared = history.get(id(upper))
        assignment = colored.get(upper.get('lower_point'))
        if not shared or assignment is None:
            continue
        owners = assignment if isinstance(assignment, tuple) else (assignment,)
        path = contour_path(upper)
        labels = [set(owners) for _ in path]
        shared_labels = [set() for _ in path]
        proven = upgrades.get(id(upper), set())
        for plant_pair, pixels, exclusive in shared:
            historical = set(map(tuple, pixels.tolist()))
            for index, point in enumerate(path):
                if tuple(point) in historical:
                    if exclusive:
                        labels[index] = set(plant_pair) | (labels[index] & proven)
                    else:
                        shared_labels[index].update(plant_pair)
        for index, shared_owners in enumerate(shared_labels):
            if shared_owners:
                labels[index] = shared_owners | (labels[index] & proven)
        if all(label == set(owners) for label in labels):
            continue
        original_parents = upper.get('parent_points', {})
        parents = dict(original_parents)
        path_points = set(map(tuple, path.tolist()))
        for pid in set.union(*labels) - set(parents):
            entry = next(point for point, label in zip(path, labels) if pid in label)
            candidates = [point for point, value in colored.items()
                          if pid in (value if isinstance(value, tuple) else (value,))
                          and point[1] <= entry[1] and point not in path_points]
            if candidates:
                parents[pid] = min(candidates, key=lambda point: sum(
                    (a - b) ** 2 for a, b in zip(point, entry)))
        allowed = set(parents)
        labels = [label & allowed for label in labels]
        starts = [0] + [i for i in range(1, len(path)) if labels[i] != labels[i - 1]]
        ends = starts[1:] + [len(path)]
        for plant_samples in samples.values():
            plant_samples.difference_update(path_points)
        pairs[:] = [(top, parent) for top, parent in pairs if top != upper['point']]
        fragments = []
        for start, end in zip(starts, ends):
            part = path[start:end]
            ids = sorted(labels[start], key=lambda pid: (parents[pid][0], pid))
            if not ids:
                continue
            fragment = dict(upper)
            fragment.pop('terminal_exits', None)
            fragment.pop('junction_id', None)
            fragment['point'] = tuple(part[0])
            fragment['lower_point'] = tuple(part[-1])
            fragment['contour'] = np.concatenate((part, part[-2:0:-1]))[:, None].copy()
            fragment['parent_points'] = {pid: parents[pid] for pid in ids}
            bottom = fragment['lower_point']
            colored[bottom] = ids[0] if len(ids) == 1 else tuple(ids)
            for pid in ids:
                samples[pid].update(map(tuple, part.tolist()))
                parents[pid] = bottom
            for parent in set(fragment['parent_points'].values()):
                if parent != fragment['point']:
                    pairs.append((fragment['point'], parent))
            fragments.append(fragment)
        if fragments:
            for pid in owners:
                if pid not in labels[-1]:
                    redirects[upper['lower_point'], pid] = parents[pid]
            upper.clear()
            upper.update(fragments[0])
            uppers.extend(fragments[1:])
    # A child may have been routed before its parent's terminal portion was
    # split. Follow the last fragment of the same plant, never another owner.
    for upper in uppers:
        for pid, parent in upper.get('parent_points', {}).items():
            seen = set()
            while (parent, pid) in redirects and parent not in seen:
                seen.add(parent)
                parent = redirects[parent, pid]
            upper['parent_points'][pid] = parent
    pending = [upper for upper in uppers if upper.get('parent_points')]
    reached = set(colored) - {upper['lower_point'] for upper in pending}
    ordered_pairs = []
    while pending:
        ready = [upper for upper in pending
                 if set(upper['parent_points'].values()) <= reached]
        if not ready:
            raise ValueError('Temporal root fragments produced a cyclic parent graph')
        ready_ids = {id(upper) for upper in ready}
        for upper in ready:
            ordered_pairs.extend((upper['point'], parent)
                                 for parent in upper['parent_points'].values()
                                 if upper['point'] != parent)
            reached.add(upper['lower_point'])
        pending = [upper for upper in pending if id(upper) not in ready_ids]
    pairs[:] = list(dict.fromkeys(ordered_pairs))
