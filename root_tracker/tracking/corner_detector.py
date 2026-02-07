"""
Corner detection for root segment endpoints.

Identifies endpoints of root segments and computes their angles.
"""

import math
import numpy as np

from ..config import Config


# Neighbor offsets for 8-connectivity
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


class CornerDetector:
    """
    Detects endpoints (corners) of root segments and computes their angles.
    
    Analyzes skeleton contours to find their endpoints and calculates
    the direction angle at each endpoint for use in root linking.
    
    Args:
        config: Configuration object.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._angle_sample_length = 10  # Pixels to sample for angle calculation
    
    def find_contour_endpoints(
        self, 
        contour: np.ndarray, 
        skeleton: np.ndarray
    ) -> list[tuple[int, int]]:
        """
        Find the two endpoints of a root segment contour.
        
        Endpoints are pixels with only one neighbor in the skeleton.
        
        Args:
            contour: Contour array from cv2.findContours.
            skeleton: Binary skeleton image.
            
        Returns:
            List of (x, y) coordinates of endpoints.
        """
        corners = []
        
        for point in contour:
            pixel_x, pixel_y = point[0]
            neighbor_count = 0
            
            for dx, dy in NEIGHBOR_OFFSETS:
                nx, ny = pixel_x + dx, pixel_y + dy
                
                # Check bounds
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbor_count += 1
            
            if neighbor_count == 1:
                corners.append((pixel_x, pixel_y))
        
        return corners
    
    def trace_from_endpoint(
        self, 
        contour: np.ndarray, 
        start_point: tuple[int, int], 
        max_length: int = 10
    ) -> list[tuple[int, int]]:
        """
        Trace the contour starting from an endpoint.
        
        Follows neighboring pixels to create a short path from the endpoint.
        
        Args:
            contour: Contour array.
            start_point: Starting (x, y) coordinate.
            max_length: Maximum number of pixels to trace.
            
        Returns:
            List of (x, y) coordinates along the trace.
        """
        trace = [start_point]
        contour_set = {tuple(p[0]) for p in contour}
        
        while len(trace) < max_length:
            current = trace[-1]
            found_next = False
            
            for dx, dy in NEIGHBOR_OFFSETS:
                candidate = (current[0] + dx, current[1] + dy)
                
                if candidate in contour_set and candidate not in trace:
                    if abs(candidate[0] - current[0]) <= 1 and abs(candidate[1] - current[1]) <= 1:
                        trace.append(candidate)
                        found_next = True
                        break
            
            if not found_next:
                break
        
        return trace
    
    def compute_angle(self, trace: list[tuple[int, int]], reverse: bool = False) -> float:
        """
        Compute the direction angle from a trace of points.
        
        Args:
            trace: List of (x, y) coordinates.
            reverse: If True, reverse the trace before computing.
            
        Returns:
            Angle in degrees (0-360).
        """
        if len(trace) < 2:
            return 90.0  # Default pointing downward
        
        if reverse:
            trace = trace[::-1]
        
        # Compute average differential
        diff = np.mean(np.diff(trace, axis=0), axis=0)
        
        angle = math.degrees(math.atan2(diff[1], diff[0]))
        if reverse:
            angle = (angle + 180) % 360
        else:
            angle = angle % 360
        
        return angle
    
    def analyze_contour_corners(
        self, 
        contour: np.ndarray, 
        skeleton: np.ndarray
    ) -> tuple[dict | None, dict | None]:
        """
        Analyze a contour to find its upper and lower corners with angles.
        
        Args:
            contour: Contour array.
            skeleton: Skeleton image.
            
        Returns:
            Tuple of (upper_corner_info, lower_corner_info) dicts, or None if no corners.
            Each dict contains: 'point', 'angle', 'contour_index'.
        """
        corners = self.find_contour_endpoints(contour, skeleton)
        
        if len(corners) != 2:
            return None, None
        
        # Determine which corner is upper (smaller y) and lower (larger y)
        if corners[0][1] < corners[1][1]:
            upper_point, lower_point = corners[0], corners[1]
        else:
            upper_point, lower_point = corners[1], corners[0]
        
        # Trace from each corner
        upper_trace = self.trace_from_endpoint(contour, upper_point, self._angle_sample_length)
        lower_trace = self.trace_from_endpoint(contour, lower_point, self._angle_sample_length)
        
        # Compute angles
        upper_angle = self.compute_angle(upper_trace[::-1], reverse=False)
        lower_angle = self.compute_angle(lower_trace[::-1], reverse=True)
        
        upper_info = {
            'point': upper_point,
            'angle': upper_angle,
        }
        
        lower_info = {
            'point': lower_point,
            'angle': lower_angle,
        }
        
        return upper_info, lower_info
