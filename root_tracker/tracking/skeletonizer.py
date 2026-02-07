"""
Root skeletonization for extracting root structure.

Converts binary root masks to single-pixel-wide skeletons and
identifies branch points.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..config import Config


# Kernel for counting neighbors (used to find branch points)
KERNEL_CENTER = 10
KERNEL_COUNT = np.array([
    [1, 1, 1],
    [1, KERNEL_CENTER, 1],
    [1, 1, 1]
], dtype=np.float32)


class RootSkeletonizer:
    """
    Creates and processes root skeletons.
    
    Converts binary root masks to single-pixel-wide skeletons,
    identifies branch/intersection points, and splits the skeleton
    into individual segments.
    
    Args:
        config: Configuration object.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._min_segment_length = 15  # Minimum pixels for a valid segment
    
    def skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a skeleton from a binary mask.
        
        Args:
            mask: Binary mask (values 0 or 1).
            
        Returns:
            Skeleton image (float32, values 0 or 1).
        """
        return skeletonize(mask).astype(np.float32)
    
    def find_endpoints(self, skeleton: np.ndarray) -> list[tuple[int, int]]:
        """
        Find endpoint pixels in the skeleton (pixels with only 1 neighbor).
        
        Args:
            skeleton: Skeleton image.
            
        Returns:
            List of (x, y) coordinates of endpoints.
        """
        # Convolve with neighbor counting kernel
        conv = cv2.filter2D(src=skeleton, ddepth=-1, kernel=KERNEL_COUNT)
        
        # Remove center pixel contribution
        conv = conv - KERNEL_CENTER
        
        # Endpoints have exactly 1 neighbor
        endpoints = [(x, y) for y, x in zip(*np.where(conv == 1))]
        
        return endpoints
    
    def find_intersections(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Find intersection/branch points in the skeleton.
        
        Args:
            skeleton: Skeleton image.
            
        Returns:
            Binary mask of intersection points (dilated for removal).
        """
        # Convolve with neighbor counting kernel
        conv = cv2.filter2D(src=skeleton, ddepth=-1, kernel=KERNEL_COUNT)
        
        # Remove center pixel contribution
        conv = conv - KERNEL_CENTER
        
        # Intersections have 3+ neighbors
        intersections = np.zeros_like(conv, dtype=np.uint8)
        intersections[conv >= 3] = 255
        
        # Dilate to create removal mask
        intersections = cv2.dilate(intersections, None, iterations=1)
        
        return intersections
    
    def split_at_intersections(
        self, 
        skeleton: np.ndarray, 
        intersections: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """
        Split skeleton at intersection points into separate segments.
        
        Args:
            skeleton: Skeleton image.
            intersections: Intersection mask.
            
        Returns:
            Tuple of (split skeleton, list of contours for each segment).
        """
        # Remove intersections from skeleton
        split_skeleton = skeleton.astype(int) - intersections.astype(int)
        split_skeleton[split_skeleton < 0] = 0
        split_skeleton = split_skeleton.astype(np.uint8) * 255
        
        # Find contours (each is a separate segment)
        contours, _ = cv2.findContours(
            split_skeleton, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # Filter short segments
        valid_contours = [cnt for cnt in contours if len(cnt) > self._min_segment_length]
        
        # Redraw with only valid contours
        result = np.zeros_like(split_skeleton)
        cv2.drawContours(result, valid_contours, -1, 255, 1)
        
        return result, valid_contours
    
    def mask_above_plants(
        self, 
        skeleton: np.ndarray, 
        positions_x: list[int], 
        positions_y: list[int]
    ) -> np.ndarray:
        """
        Mask out skeleton regions above plant positions.
        
        Removes skeleton pixels above the plant stems to focus on roots.
        
        Args:
            skeleton: Skeleton image.
            positions_x: X coordinates of plant positions.
            positions_y: Y coordinates of plant positions.
            
        Returns:
            Skeleton with above-plant regions masked.
        """
        result = skeleton.copy()
        width_fraction = skeleton.shape[1] // 12
        
        for x, y in zip(positions_x, positions_y):
            left = max(x - width_fraction, 0)
            right = min(x + width_fraction, skeleton.shape[1] - 1)
            result[:y, left:right] = 0
        
        return result
    
    def get_total_length(self, skeleton: np.ndarray) -> int:
        """
        Get the total length of the skeleton in pixels.
        
        Args:
            skeleton: Skeleton image.
            
        Returns:
            Total number of skeleton pixels.
        """
        return cv2.countNonZero(skeleton)
