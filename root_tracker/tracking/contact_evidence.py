"""Evidence for two identities occupying one physical skeleton corridor.

A missing exit is a possible root tip, not evidence of a shared root. Sharing
needs either independent continuations beyond the corridor or a foreground
width better explained by the envelope of the arriving roots than one root.
All width comparisons use local measurements, in the same image units.
"""

import numpy as np
from scipy.ndimage import median_filter

from .temporal_fragments import contour_path


def turn(a, b):
    return abs((a - b + 180.) % 360. - 180.)


class ContactEvidence:
    def __init__(self, uppers, lower_by_point, incoming, tasks, by_bottom):
        self.uppers = uppers
        self.lowers = lower_by_point
        self.incoming = incoming
        self.tasks = tasks
        self.by_bottom = by_bottom
        self.sections = {}
        self.bundles = set()
        self.parallel = set()
        self.travel = {}

    def independent_exits(self, index, options):
        """Match all arrivals at the far junction, including local arrivals.

        A lateral entering the far end must reserve its own continuation; it
        cannot also prove that the same plant travelled through the corridor.
        Walk degree-two vertices so splitting a contour does not change proof.
        """
        visited = set()
        while index not in visited:
            visited.add(index)
            bottom = self.uppers[index].get('lower_point')
            junction = self.lowers.get(bottom, {}).get('junction_id')
            children = self.tasks.get(('junction', junction), []) if junction else []
            local = [p for p in self.incoming.get(junction, []) if p != bottom]
            if len(children) == 1 and not local:
                index = children[0]
                continue
            if len(children) < 2:
                return set()
            entries = [(pid, angle) for pid, routes in options.items()
                       for _, angle in routes[:1]]
            entries += [(None, self.lowers[p]['angle']) for p in local]
            scores = np.asarray([[turn(angle, (self.uppers[c]['angle'] + 180) % 360)
                                  for c in children] for _, angle in entries])
            # Output count alone is not proof: an ordinary lateral creates
            # another output too. Each continuation must uniquely prefer a
            # distinct arrival and proceed forward along that arrival.
            owners = set()
            for column in scores.T:
                row = int(np.argmin(column))
                if (column[row] < 90 and np.count_nonzero(
                        np.isclose(column, column[row])) == 1
                        and entries[row][0] is not None):
                    owners.add(entries[row][0])
            return owners
        return set()

    def width_support(self, index, options, winner):
        """Compare measured width with single-root and bundle envelopes.

        Exclude each junction's own bulge by measuring an incoming segment's
        interior. Median residuals keep isolated segmentation specks from
        establishing sharing. An equal fit is deliberately inconclusive.
        """
        upper = self.uppers[index]
        profile = np.asarray(upper.get('width_profile', []), dtype=float)
        if not len(profile):
            return None
        widths, centers = self.arrivals(index, options)
        if not widths or len(widths) != len(options):
            return None
        intervals = [(centers[pid] - width / 2, centers[pid] + width / 2)
                     for pid, width in widths.items()]
        if winner not in widths:
            return None
        bundle = max(b for _, b in intervals) - min(a for a, _ in intervals)
        path = contour_path(upper)
        spacing = np.median(np.linalg.norm(np.diff(path, axis=0), axis=1)) if len(path) > 1 else 1.
        radius = max(1, int(round(max(widths.values()) / max(spacing, np.finfo(float).eps))))
        measured = median_filter(profile, size=2 * radius + 1, mode='nearest')
        # Exclude the junction bulge. Evidence must last at least one local
        # diameter beyond it; short corridors require topological evidence.
        initial = measured[radius:2 * radius]
        if len(initial) < radius:
            return None
        # Skeleton and distance-transform locations are quantized to pixels.
        # Prefer the simpler single-root explanation within that uncertainty.
        single_residual = np.maximum(0., abs(initial - widths[winner]) - spacing)
        if np.median(abs(initial - bundle)) >= np.median(single_residual):
            return None
        # Locate a sustained return to a single-root caliber. The window is
        # measured in root diameters, so rescaling the image preserves it.
        shared_error = abs(measured - bundle)
        tail_width = np.median(measured[-max(radius, len(measured) // 4):])
        residuals = {pid: abs(value - tail_width) for pid, value in widths.items()}
        best = min(residuals.values())
        # A local neck followed by a wider bundle is not a root ending.
        # Require the terminal portion itself to favor the single-root model.
        if best >= abs(tail_width - bundle):
            return len(path), winner
        candidates = [pid for pid, error in residuals.items() if np.isclose(error, best)]
        single_error = abs(measured - widths[candidates[0]])
        contraction = single_error < shared_error
        cut = None
        for i in range(2 * radius, len(measured) - radius):
            if np.all(contraction[i:i + radius]):
                cut = i
                break
        if cut is None:
            return len(path), winner
        if len(candidates) == 1:
            survivor = candidates[0]
        else:
            # Equal-caliber roots: use which outer boundary continues across
            # the narrowing. Project the centerline shift onto the local
            # normal, and preserve the arriving transverse order.
            before = path[max(0, cut - 2 * radius):cut].astype(float)
            after = path[cut:min(len(path), cut + 2 * radius)].astype(float)
            direction = before[-1] - before[0]
            norm = np.linalg.norm(direction)
            if norm == 0 or len(after) < 2:
                return len(path), winner
            local_normal = np.array([-direction[1], direction[0]]) / norm
            shift = float(np.dot(after.mean(axis=0) - before.mean(axis=0), local_normal))
            if abs(shift) < spacing:
                return len(path), winner
            survivor = max(candidates, key=lambda pid: shift * centers[pid])
        return cut, survivor

    def arrivals(self, index, options):
        angle = np.deg2rad((self.uppers[index]['angle'] + 180) % 360)
        normal = np.array([-np.sin(angle), np.cos(angle)])
        widths, centers = {}, {}
        for pid, routes in options.items():
            point, _ = routes[0]
            inherited = self.sections.get(point, {}).get(pid)
            if inherited is not None:
                width, offset = inherited
            else:
                parent = self.by_bottom.get(point)
                if parent is None:
                    continue
                values = np.asarray(self.uppers[parent].get('width_profile', []), dtype=float)
                if not len(values):
                    continue
                interior = values[len(values) // 4:max(len(values) // 4 + 1, 3 * len(values) // 4)]
                width, offset = float(np.median(interior)), 0.
            widths[pid] = width
            centers[pid] = float(np.dot(point, normal)) + offset
        return widths, centers

    def remember(self, index, routes, bundle=False):
        upper = self.uppers[index]
        if 'width_profile' not in upper:
            return
        bottom = upper['lower_point']
        if len(routes) == 1:
            values = np.asarray(upper['width_profile'])
            interior = values[len(values) // 4:max(len(values) // 4 + 1, 3 * len(values) // 4)]
            self.sections[bottom] = {next(iter(routes)): (float(np.median(interior)), 0.)}
            return
        widths, centers = self.arrivals(index, {pid: [route] for pid, route in routes.items()})
        if len(widths) != len(routes):
            return
        left = min(centers[p] - widths[p] / 2 for p in widths)
        right = max(centers[p] + widths[p] / 2 for p in widths)
        middle = (left + right) / 2
        # The arrivals bound the predicted envelope. Once observed, calibrate
        # lane spacing to the actual bundle silhouette: roots can overlap in
        # projection rather than remaining side by side at their entry gap.
        values = np.asarray(upper['width_profile'])
        diameter = max(widths.values())
        path = contour_path(upper)
        spacing = np.median(np.linalg.norm(np.diff(path, axis=0), axis=1)) if len(path) > 1 else 1.
        window = max(1, int(round(diameter / max(spacing, np.finfo(float).eps))))
        observed = float(np.median(values[:2 * window]))
        excess = right - left - diameter
        contraction = np.clip((observed - diameter) / excess, 0., 1.) if excess > 0 else 1.
        self.sections[bottom] = {pid: (widths[pid], (centers[pid] - middle) * contraction)
                                 for pid in widths}
        if bundle or any(route[0] in self.bundles for route in routes.values()):
            self.bundles.add(bottom)
            distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            distance += max((self.travel.get(route[0], 0.) for route in routes.values()), default=0.)
            self.travel[bottom] = distance
            angles = [route[1] for route in routes.values()]
            separation = max(turn(a, b) for a in angles for b in angles)
            sine = np.sin(np.deg2rad(separation / 2))
            # Two crossing strips overlap over a distance set by their own
            # calibers and intersection angle. Beyond that footprint, the
            # evidence describes a parallel bundle, whose lanes keep order.
            footprint = sum(widths.values()) / sine if sine > 0 else np.inf
            if distance > footprint or any(route[0] in self.parallel for route in routes.values()):
                self.parallel.add(bottom)

    def supported(self, index, options, winner):
        if len(options) < 2:
            return set(), None
        independent = self.independent_exits(index, options)
        if len(independent) > 1:
            return independent, None
        extent = self.width_support(index, options, winner)
        if extent is not None:
            return set(options), extent
        return set(), None
