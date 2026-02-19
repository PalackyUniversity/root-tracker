"""
Root linking and plant assignment.

Links root segments based on angles and positions, and assigns them to plants.
"""

import math
import numpy as np
import cv2

from ..config import Config


# Colors for each plant (BGR format)
PLANT_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]


class RootLinker:
    """
    Links root segments and assigns them to plants.

    Uses spatial proximity and directional compatibility to link
    root segments into continuous roots and assign them to their
    parent plants.

    Args:
        config: Configuration object.
    """

    # Maximum distance (px) for matching a segment's upper endpoint to the
    # same endpoint recorded in the previous frame.
    _ENDPOINT_MATCH_THRESHOLD = 15

    # Squared BGR distance below which a pixel is considered to match a plant
    # color (normal or bright variant).  Excludes near-white pixels (indicator
    # lines etc.) whose distance to the nearest bright plant color is ~14 000.
    _PIXEL_MAX_DIST_SQ = 5000

    # Minimum fraction of a segment's pixels that must vote for the same plant
    # before we consider the color vote authoritative.
    _MIN_COLOR_VOTE_FRACTION = 0.3

    def __init__(self, config: Config) -> None:
        self.config = config

    def get_color(self, plant_id: int) -> tuple[int, int, int]:
        """Get the color for a plant."""
        return PLANT_COLORS[plant_id % len(PLANT_COLORS)]

    def compute_distance(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float]
    ) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        )

    def _extrapolate_point(
        self,
        point: tuple[float, float],
        direction: tuple[float, float],
        curvature: tuple[float, float],
        distance: float
    ) -> tuple[float, float]:
        """
        Extrapolate a point along its curve by a given distance.

        Uses: pos(t) = point + direction * t + 0.5 * curvature * t * t_curv
        where t_curv is capped to avoid wild extrapolations from noisy
        curvature estimates over short traces.
        """
        t_curv = min(distance, 25.0)
        return (
            point[0] + direction[0] * distance + 0.5 * curvature[0] * distance * t_curv,
            point[1] + direction[1] * distance + 0.5 * curvature[1] * distance * t_curv,
        )

    def _classify_pixel(self, pixel) -> int:
        """
        Return the plant_id (0-based) whose color best matches a BGR pixel,
        or -1 if the pixel is too dark or too ambiguous.

        Checks both the normal plant color and its brightened variant
        (used for the main-root overlay).
        """
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        if b + g + r < 30:          # near-black background
            return -1

        best_dist = self._PIXEL_MAX_DIST_SQ
        best_pid = -1

        for pid, (cb, cg, cr) in enumerate(PLANT_COLORS):
            for add in (0, 170):    # normal and bright (+170, capped) variants
                db = b - min(cb + add, 255)
                dg = g - min(cg + add, 255)
                dr = r - min(cr + add, 255)
                dist = db * db + dg * dg + dr * dr
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid

        return best_pid

    def _vote_plant_from_annotated(
        self,
        contour: np.ndarray,
        annotated: np.ndarray
    ) -> int | None:
        """
        Sample the previous annotated image at each skeleton pixel of a
        segment and return the plant_id that appears most often, provided it
        accounts for at least _MIN_COLOR_VOTE_FRACTION of the votes.

        The annotated image has every root painted 3-12 px thick in its plant
        color, so pixel-level lookup is far more robust than 1-px skeleton
        intersection, even with a 1-2 px registration shift.

        Returns None when no clear winner is found (genuinely new growth).
        """
        h, w = annotated.shape[:2]
        votes: dict[int, int] = {}

        for pt in contour[:, 0]:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < h and 0 <= x < w:
                pid = self._classify_pixel(annotated[y, x])
                if pid >= 0:
                    votes[pid] = votes.get(pid, 0) + 1

        if not votes:
            return None

        total = sum(votes.values())
        best_pid = max(votes, key=lambda p: votes[p])
        if votes[best_pid] / total >= self._MIN_COLOR_VOTE_FRACTION:
            return best_pid
        return None

    def compute_link_cost(
        self,
        upper_corner: dict,
        lower_corner: dict,
        min_diff_x: float,
        plant_positions_x: list[int],
        plant_positions_y: list[int]
    ) -> float:
        """
        Compute the cost of linking two corners.

        Lower cost means better link. Uses extrapolation from each
        corner's direction and curvature to predict where the root
        would continue, then measures how well the prediction matches
        the actual candidate position.
        """
        up_point = upper_corner['point']
        low_point = lower_corner['point']

        if low_point[1] > up_point[1]:
            return math.inf

        distance = self.compute_distance(up_point, low_point)

        is_plant_center = (
            low_point[0] in plant_positions_x and
            low_point[1] in plant_positions_y
        )

        if is_plant_center:
            if abs(up_point[0] - low_point[0]) > min_diff_x // 2:
                return math.inf
            if abs(up_point[1] - low_point[1]) > min_diff_x * 2 // 3:
                return math.inf
        elif distance > min_diff_x // 2:
            return math.inf

        up_dir = upper_corner.get('direction', (0.0, -1.0))
        up_curv = upper_corner.get('curvature', (0.0, 0.0))
        low_dir = lower_corner.get('direction', (0.0, 1.0))
        low_curv = lower_corner.get('curvature', (0.0, 0.0))

        up_ext = self._extrapolate_point(up_point, up_dir, up_curv, distance)
        low_ext = self._extrapolate_point(low_point, low_dir, low_curv, distance)

        ext_error_up = self.compute_distance(up_ext, low_point)
        ext_error_low = self.compute_distance(low_ext, up_point)
        ext_cost = (ext_error_up + ext_error_low) / 2.0

        alignment = up_dir[0] * low_dir[0] + up_dir[1] * low_dir[1]
        straightness_penalty = (1.0 + alignment) / 2.0

        cost = (0.25 * distance
                + 0.6 * ext_cost
                + 0.15 * min_diff_x * straightness_penalty)

        return cost

    def link_corners(
        self,
        upper_corners: list[dict],
        lower_corners: list[dict],
        plant_positions_x: list[int],
        plant_positions_y: list[int],
        previous_upper_assignments: dict[tuple, int] = None,
        previous_annotated: np.ndarray = None,
    ) -> tuple[list[tuple], dict[tuple, int], dict[int, set], dict[tuple, int]]:
        """
        Link upper corners to lower corners and assign plant IDs.

        Temporal consistency is enforced through two complementary mechanisms,
        applied in order before the geometric chain:

        1. **Upper-endpoint matching** — a segment's upper endpoint (the
           branching point from the root above) is stable across frames; if it
           is within _ENDPOINT_MATCH_THRESHOLD px of a known endpoint from the
           previous frame, that frame's plant ID is locked in.

        2. **Annotated-image color vote** — for segments that have no endpoint
           match (e.g. newly-split segments whose endpoints moved), each pixel
           of the segment is sampled in the previous frame's annotated image.
           Because roots are painted 3-12 px thick, this lookup is robust to
           1-2 px registration shifts.  The majority plant color wins.

        Only segments that match neither criterion (genuine new growth) fall
        through to the geometric chain.

        Args:
            upper_corners: List of upper corner info dicts.
            lower_corners: List of lower corner info dicts.
            plant_positions_x: X coordinates of plant centers.
            plant_positions_y: Y coordinates of plant centers.
            previous_upper_assignments: upper_endpoint → plant_id from the
                previous frame (returned as 4th value of this function).
            previous_annotated: Annotated BGR image from the previous frame,
                used for pixel-level plant color lookup.

        Returns:
            Tuple of:
            - List of (upper_point, lower_point) pairs
            - Dict mapping points to plant IDs
            - Dict mapping plant IDs to sets of skeleton pixels
            - Dict mapping this frame's upper endpoints to plant IDs
              (pass as previous_upper_assignments on the next frame)
        """
        if not upper_corners or not plant_positions_x:
            return [], {}, {}, {}

        min_diff_x = min(np.diff(sorted(plant_positions_x))) if len(plant_positions_x) > 1 else 100

        colored = {
            (x, y): n
            for n, (x, y) in enumerate(zip(plant_positions_x, plant_positions_y))
        }

        upper_corners_sorted = sorted(upper_corners, key=lambda c: c['point'][1])
        lower_corners_sorted = sorted(lower_corners, key=lambda c: c['point'][1])

        pairs = []
        colored_samples = {n: set() for n in range(self.config.n_clusters)}
        upper_assignments: dict[tuple, int] = {}
        used = set(lower_corner['point'] for lower_corner in lower_corners_sorted[:len(plant_positions_x)])

        for upper in upper_corners_sorted:
            up_point = upper['point']
            forced_plant_id = None

            # --- Step 1: upper-endpoint matching ---
            # A segment's upper endpoint is where it branches from the root
            # above — the most stable feature across frames.  Match it to the
            # nearest endpoint recorded in the previous frame.
            if previous_upper_assignments:
                best_dist = float('inf')
                for prev_pt, prev_pid in previous_upper_assignments.items():
                    d = self.compute_distance(up_point, prev_pt)
                    if d < best_dist:
                        best_dist = d
                        forced_plant_id = prev_pid
                if best_dist > self._ENDPOINT_MATCH_THRESHOLD:
                    forced_plant_id = None

            # --- Step 2: annotated-image color vote (fallback) ---
            # For segments whose endpoint has no close match (e.g. newly formed
            # due to skeleton splitting), sample the previous annotated image.
            # Roots are painted 3-12 px thick there, so this is reliable even
            # with small registration shifts.
            if forced_plant_id is None and previous_annotated is not None and 'contour' in upper:
                forced_plant_id = self._vote_plant_from_annotated(
                    upper['contour'], previous_annotated
                )

            # --- Step 3: geometric chain + fallback ---
            costs = []
            for lower in lower_corners_sorted:
                cost = self.compute_link_cost(
                    upper, lower, min_diff_x,
                    plant_positions_x, plant_positions_y
                )
                costs.append((cost, lower))

            if costs:
                min_cost, best_lower = min(costs, key=lambda x: x[0])

                if min_cost != math.inf:
                    low_point = best_lower['point']
                    used.add(low_point)

                    # Plant ID priority:
                    # 1. Temporal lock (endpoint match or color vote) — never
                    #    changes an established classification.
                    # 2. Geometric chain propagation — for genuinely new growth.
                    # 3. Nearest already-colored point — last resort.
                    if forced_plant_id is not None:
                        plant_id = forced_plant_id
                    elif low_point in colored:
                        plant_id = colored[low_point]
                    else:
                        distances = [
                            (self.compute_distance(up_point, p), pid)
                            for p, pid in colored.items()
                        ]
                        if distances:
                            _, plant_id = min(distances, key=lambda x: x[0])
                        else:
                            plant_id = 0

                    if 'lower_point' in upper:
                        colored[upper['lower_point']] = plant_id

                    pairs.append((up_point, low_point))
                    upper_assignments[up_point] = plant_id

                    if 'contour' in upper:
                        pixels = set(map(tuple, upper['contour'][:, 0]))
                        colored_samples[plant_id] = colored_samples[plant_id].union(pixels)

        return pairs, colored, colored_samples, upper_assignments

    def draw_annotations(
        self,
        image: np.ndarray,
        plant_positions_x: list[int],
        plant_positions_y: list[int],
        pairs: list[tuple],
        colored: dict[tuple, int]
    ) -> np.ndarray:
        """Draw plant annotations and root links on an image."""
        result = image.copy()

        for i, (x, y) in enumerate(zip(plant_positions_x, plant_positions_y)):
            color = self.get_color(i)
            cv2.circle(result, (x, y), 10, color, 4)
            cv2.putText(result, str(i + 1), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        for upper, lower in pairs:
            cv2.line(result, upper, lower, (255, 255, 255), 1)

        return result
