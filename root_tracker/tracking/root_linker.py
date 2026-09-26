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
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def get_color(self, plant_id: int) -> tuple[int, int, int]:
        """Get the color for a plant."""
        return PLANT_COLORS[plant_id % len(PLANT_COLORS)]
    
    def compute_distance(
        self, 
        point1: tuple[int, int], 
        point2: tuple[int, int]
    ) -> float:
        """Compute Euclidean distance between two points."""
        return math.hypot(float(point1[0]) - float(point2[0]),
                          float(point1[1]) - float(point2[1]))
    
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
        
        Lower cost means better link. Considers:
        - Spatial distance
        - Angle compatibility
        - Whether the link crosses plant positions
        
        Args:
            upper_corner: Upper corner info dict.
            lower_corner: Lower corner info dict.
            min_diff_x: Minimum x-difference between plants.
            plant_positions_x: X coords of plant centers.
            plant_positions_y: Y coords of plant centers.
            
        Returns:
            Link cost (infinity if link should be rejected).
        """
        up_point = upper_corner['point']
        low_point = lower_corner['point']
        up_angle = upper_corner['angle']
        low_angle = lower_corner['angle']
        
        # Only link if upper is above lower
        if low_point[1] > up_point[1]:
            return math.inf
        
        # Basic distance
        distance = self.compute_distance(up_point, low_point)
        
        # Check if linking to a plant center (special case)
        is_plant_center = (
            low_point[0] in plant_positions_x and 
            low_point[1] in plant_positions_y
        )
        
        if is_plant_center:
            # Don't link to distant plant centers
            if abs(up_point[0] - low_point[0]) > min_diff_x // 2:
                return math.inf
            if abs(up_point[1] - low_point[1]) > min_diff_x * 2 // 3:
                return math.inf
        elif distance > min_diff_x // 2:
            return math.inf
        
        # Angle compatibility cost
        cost = distance
        cost += abs(up_angle - low_angle) * min_diff_x / 180 / 4
        
        # Prediction matching cost
        predicted_angle = math.degrees(np.arctan2(
            low_point[1] - up_point[1], 
            low_point[0] - up_point[0]
        )) + 180
        
        cost += abs(predicted_angle - up_angle) * min_diff_x / 180 / 2
        cost += abs(predicted_angle - low_angle) * min_diff_x / 180 / 2
        
        return cost
    
    def link_corners(
        self,
        upper_corners: list[dict],
        lower_corners: list[dict],
        plant_positions_x: list[int],
        plant_positions_y: list[int],
        previous_colored_samples: dict[int, set] = None
    ) -> tuple[list[tuple], dict[tuple, int], dict[int, set]]:
        """
        Link upper corners to lower corners and assign plant IDs.
        
        Args:
            upper_corners: List of upper corner info dicts.
            lower_corners: List of lower corner info dicts.
            plant_positions_x: X coordinates of plant centers.
            plant_positions_y: Y coordinates of plant centers.
            previous_colored_samples: Sample pixels from previous frame for consistency.
            
        Returns:
            Tuple of:
            - List of (upper_point, lower_point) pairs
            - Dict mapping points to plant IDs
            - Dict mapping plant IDs to sets of skeleton pixels
        """
        if not upper_corners or not plant_positions_x:
            return [], {}, {}
        
        min_diff_x = min(np.diff(sorted(plant_positions_x))) if len(plant_positions_x) > 1 else 100
        
        # Initialize plant assignments
        colored = {
            (x, y): n 
            for n, (x, y) in enumerate(zip(plant_positions_x, plant_positions_y))
        }
        
        # Sort corners by y-coordinate
        upper_corners_sorted = sorted(upper_corners, key=lambda c: c['point'][1])
        lower_corners_sorted = sorted(lower_corners, key=lambda c: c['point'][1])
        
        pairs = []
        colored_samples = {n: set() for n in range(self.config.n_clusters)}
        used = set(lower_corner['point'] for lower_corner in lower_corners_sorted[:len(plant_positions_x)])
        
        for upper in upper_corners_sorted:
            up_point = upper['point']
            forced_plant_id = None

            # Consistency check: look for overlap with previous frame
            if previous_colored_samples and 'contour' in upper:
                current_pixels = set(map(tuple, upper['contour'][:, 0].tolist()))
                
                # Check overlap with each plant from previous frame
                for plant_id, prev_pixels in previous_colored_samples.items():
                    if not prev_pixels:
                        continue
                    
                    # Compute intersection
                    # We can assume small overlap is sufficient
                    intersection_size = len(current_pixels.intersection(prev_pixels))
                    if intersection_size > 0:
                        forced_plant_id = plant_id
                        break
            
            # If we found a forced match, we try to link to that plant
            if forced_plant_id is not None:
                # Find nearest colored point belonging to this plant
                plant_points = [p for p, pid in colored.items() if pid == forced_plant_id]
                if plant_points:
                    # Find nearest point in that plant's set
                    best_point = min(plant_points, key=lambda p: abs(p[0] - up_point[0]) + abs(p[1] - up_point[1]))

                    if 'lower_point' in upper:
                        colored[upper['lower_point']] = forced_plant_id
                    
                    pairs.append((up_point, best_point))
                    
                    if 'contour' in upper:
                        pixels = set(map(tuple, upper['contour'][:, 0].tolist()))
                        colored_samples[forced_plant_id].update(pixels)
                        
                    continue

            # Standard linking logic (if not forced)
            
            # Find best matching lower corner
            costs = []
            for lower in lower_corners_sorted:
                if lower['point'][1] > up_point[1]:
                    break
                
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
                    
                    # Assign plant ID based on nearest colored point
                    if low_point in colored:
                        plant_id = colored[low_point]
                    else:
                        # Find nearest colored point
                        distances = [
                            (self.compute_distance(up_point, p), pid)
                            for p, pid in colored.items()
                        ]
                        if distances:
                            _, plant_id = min(distances, key=lambda x: x[0])
                        else:
                            plant_id = 0
                    
                    # Propagate color to upper corner
                    if 'lower_point' in upper:
                        colored[upper['lower_point']] = plant_id
                    
                    pairs.append((up_point, low_point))
                    
                    # Track pixels for this plant
                    if 'contour' in upper:
                        pixels = set(map(tuple, upper['contour'][:, 0].tolist()))
                        colored_samples[plant_id].update(pixels)
        
        return pairs, colored, colored_samples
    
    def draw_annotations(
        self,
        image: np.ndarray,
        plant_positions_x: list[int],
        plant_positions_y: list[int],
        pairs: list[tuple],
        colored: dict[tuple, int]
    ) -> np.ndarray:
        """
        Draw plant annotations and root links on an image.
        
        Args:
            image: Image to annotate (will be modified in place).
            plant_positions_x: X coordinates of plants.
            plant_positions_y: Y coordinates of plants.
            pairs: List of (upper, lower) point pairs.
            colored: Point to plant ID mapping.
            
        Returns:
            Annotated image.
        """
        result = image.copy()
        
        # Draw plant centers
        for i, (x, y) in enumerate(zip(plant_positions_x, plant_positions_y)):
            color = self.get_color(i)
            cv2.circle(result, (x, y), 10, color, 4)
            cv2.putText(result, str(i + 1), (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        
        # Draw links
        for upper, lower in pairs:
            cv2.line(result, upper, lower, (255, 255, 255), 1)
        
        return result
