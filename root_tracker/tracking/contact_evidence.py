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
    def __init__(self, uppers, lower_by_point, incoming, tasks, by_bottom,
                 historical_owner=None):
        self.uppers = uppers
        self.lowers = lower_by_point
        self.incoming = incoming
        self.tasks = tasks
        self.by_bottom = by_bottom
        self.sections = {}
        self.bundles = set()
        self.parallel = set()
        self.travel = {}
        self.tangents = {}
        self.historical_owner = historical_owner or (lambda index: None)

    def incoming_direction(self, point, fallback):
        """Estimate approach outside the junction's thinning distortion.

        The last skeleton pixels are pulled towards the other root at a
        contact. Fit the preceding centerline over one measured diameter,
        leaving one diameter between the fit and the junction endpoint.
        Use arc length so this is invariant to sampling density and scale.
        """
        index = self.by_bottom.get(point)
        if index is None or not self.lowers.get(point, {}).get('junction_id'):
            return fallback
        return self._endpoint_direction(index, True, fallback)

    def outgoing_direction(self, index):
        upper = self.uppers[index]
        fallback = (upper['angle'] + 180) % 360
        if not upper.get('junction_id'):
            return fallback
        return self._endpoint_direction(index, False, fallback)

    def _endpoint_direction(self, index, at_end, fallback):
        key = (index, at_end)
        if key in self.tangents:
            return self.tangents[key]
        upper = self.uppers[index]
        widths = np.asarray(upper.get('width_profile', []), dtype=float)
        if not len(widths):
            return fallback
        diameter = float(np.median(widths))
        path = contour_path(upper).astype(float)
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        # A short bridge has no uncontaminated interior between its two
        # junctions. Keep its original direction instead of fitting the other
        # junction's distorted pixels or a truncated observation window.
        other_junction = (upper.get('junction_id') if at_end else
                          self.lowers.get(upper['lower_point'], {}).get('junction_id'))
        needed_length = (3 if other_junction else 2) * diameter
        if distances[-1] < needed_length:
            return fallback
        if diameter <= 0:
            return fallback
        low, high = ((distances[-1] - 2 * diameter, distances[-1] - diameter)
                     if at_end else (diameter, 2 * diameter))
        boundaries = np.column_stack([np.interp([low, high], distances, path[:, axis])
                                      for axis in range(2)])
        sample = np.vstack((boundaries[:1], path[(distances > low) & (distances < high)],
                            boundaries[1:]))
        # Integrate the line-fit moments along each polyline edge. Giving
        # every vertex equal weight would bias the direction towards densely
        # sampled pieces (and make equivalent contours route differently).
        weights = np.linalg.norm(np.diff(sample, axis=0), axis=1)
        weights /= weights.sum()
        mean = np.sum(weights[:, None] * (sample[:-1] + sample[1:]) / 2, axis=0)
        first, last = sample[:-1] - mean, sample[1:] - mean
        covariance = ((first.T * weights) @ first + (last.T * weights) @ last) / 3
        cross = (first.T * weights) @ last
        covariance += (cross + cross.T) / 6
        _, vectors = np.linalg.eigh(covariance)
        vector = vectors[:, -1]
        if np.dot(vector, sample[-1] - sample[0]) < 0:
            vector = -vector
        angle = float(np.rad2deg(np.arctan2(vector[1], vector[0])) % 360)
        self.tangents[key] = angle
        return angle

    def exclusive_owner(self, index, options, winner):
        """Resolve a new single continuation after sharing was ruled out.

        Do not reinterpret the identities inside an already shared arrival:
        its common centerline has no separate approach direction per plant.
        """
        if any(len(routes) != 1 for routes in options.values()):
            return winner
        points = [routes[0][0] for routes in options.values()]
        if len(set(points)) != len(points):
            return winner
        outgoing = self.outgoing_direction(index)
        scores = {pid: turn(outgoing, self.incoming_direction(routes[0][0], routes[0][1]))
                  for pid, routes in options.items()}
        best = min(scores, key=scores.get)
        return winner if np.isclose(scores[best], scores[winner]) else best

    def independent_exits(self, index, options):
        """Match all arrivals at the far junction, including local arrivals.

        A lateral entering the far end must reserve its own continuation; it
        cannot also prove that the same plant travelled through the corridor.
        Walk degree-two vertices so splitting a contour does not change proof.
        """
        established = self.historical_owner(index)
        widths, _ = self.arrivals(index, options)
        angles = [routes[0][1] for routes in options.values() if len(routes) == 1]
        crossing_limit = 0.
        if len(widths) == len(options) == len(angles) == 2:
            separation = turn(*angles)
            if separation > 0:
                crossing_limit = sum(widths.values()) / np.sin(np.deg2rad(separation / 2))
        distance = 0.
        previous_end = None
        visited = set()
        while index not in visited:
            visited.add(index)
            path = contour_path(self.uppers[index]).astype(float)
            distance += float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if previous_end is not None:
                distance += float(np.linalg.norm(path[0] - previous_end))
            previous_end = path[-1]
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
            for child, column in zip(children, scores.T):
                row = int(np.argmin(column))
                owner = entries[row][0]
                historical = self.historical_owner(child)
                # An old lateral belongs to its established root, even if its
                # heading happens to match a new arrival. A new exit can prove
                # a crossing only inside the overlap footprint of the arriving
                # strips; a distant side twig cannot upgrade an old trunk.
                if historical is not None and historical[0] != owner:
                    continue
                if (established is not None and owner != established[0]
                        and historical is None and distance > crossing_limit):
                    continue
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
        # A short piece between junctions cannot contain the required
        # interior window. For an already established bundle, continue the
        # observation along its straightest physical exit; a lateral does not
        # reset the measurement window merely by splitting the skeleton.
        measured_profile = profile
        cursor, visited = index, {index}
        inherited = any(route[0] in self.bundles for routes in options.values()
                        for route in routes)
        while inherited and len(measured_profile) < 2 * radius:
            bottom = self.uppers[cursor].get('lower_point')
            junction = self.lowers.get(bottom, {}).get('junction_id')
            children = [c for c in self.tasks.get(('junction', junction), [])
                        if c not in visited and 'width_profile' in self.uppers[c]]
            if not children:
                break
            direction = self.lowers.get(bottom, {}).get(
                'angle', self.outgoing_direction(cursor))
            child = min(children, key=lambda c: turn(direction, self.outgoing_direction(c)))
            if turn(direction, self.outgoing_direction(child)) >= 90:
                break
            measured_profile = np.r_[measured_profile, self.uppers[child]['width_profile']]
            visited.add(child)
            cursor = child
        measured = median_filter(measured_profile, size=2 * radius + 1, mode='nearest')
        # Exclude the junction bulge. Evidence must last at least one local
        # diameter beyond it; short corridors require topological evidence.
        # An existing bundle already supplies the entrance evidence. A short
        # wide continuation can end at the next junction; requiring another
        # whole diameter past that junction would discard its shared prefix.
        initial = (measured[:radius] if inherited and len(path) < 2 * radius
                   else measured[radius:2 * radius])
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
        if cut is None or cut >= len(path):
            # A narrowing observed in the lookahead belongs to a later
            # segment. This whole piece remains shared; its own centerline
            # cannot determine the survivor at that downstream boundary.
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
        return min(cut, len(path)), survivor

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
        contraction = (np.clip((observed - diameter) / excess, 0., 1.)
                       if excess > 0 and len(path) >= 2 * window else 1.)
        self.sections[bottom] = {pid: (widths[pid], (centers[pid] - middle) * contraction)
                                 for pid in widths}
        if bundle or len(routes) > 1:
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
            if ((bundle and distance > footprint)
                    or any(route[0] in self.parallel for route in routes.values())):
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
