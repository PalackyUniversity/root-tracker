"""
Green area detection for plant stem identification.

Detects green plant parts and clusters them to identify individual plants.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ..config import Config
from .colors import color_mask


class GreenAreaDetector:
    """
    Detects green plant areas and identifies plant centroids.
    
    Uses HSV color thresholding to find green areas, then clusters
    them to identify individual plant positions.
    
    Args:
        config: Configuration object with green detection settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def detect_green_mask(self, image: np.ndarray, *, hsv=None) -> np.ndarray:
        """
        Create a binary mask of green areas in the image.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Binary mask where green areas are white.
        """
        green_threshold = color_mask(image, self.config.green.hsv_lower,
                                     self.config.green.hsv_upper, hsv=hsv)

        # Clean up the mask
        green_threshold = cv2.dilate(green_threshold, None, iterations=3)
        green_threshold = cv2.erode(green_threshold, None, iterations=1)
        
        return green_threshold
    
    def find_green_contours(
        self, 
        image: np.ndarray
    ) -> tuple[list, np.ndarray]:
        """
        Find contours of green areas in the image.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Tuple of (contours list, green mask).
        """
        green_mask = self.detect_green_mask(image)
        contours, _ = cv2.findContours(
            green_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # Filter by size
        valid_contours = [
            cnt for cnt in contours
            if len(cnt) > self.config.green.min_count 
            and cv2.contourArea(cnt) > self.config.green.min_area
        ]
        
        return valid_contours, green_mask
    
    def cluster_plant_positions(
        self, 
        contours: list
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Cluster green contours into plant positions using KMeans.
        
        Args:
            contours: List of contours from green detection.
            
        Returns:
            Tuple of (x_positions, y_positions, areas) for each cluster,
            sorted left-to-right by x position.
        """
        # Early frames can show fewer stems than the configured plant count.
        # Keep preprocessing their pixels; tracking uses the shared origins
        # from other dates. Partial clusters would shift plant identities.
        if len(contours) < self.config.n_clusters:
            return [], [], []
        
        # Extract centroids and areas from contours
        positions_x = []
        positions_y = []
        areas = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate area within contour
            cnt_shifted = cnt.copy()
            cnt_shifted[:, 0, 0] -= x
            cnt_shifted[:, 0, 1] -= y
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt_shifted], -1, 255, cv2.FILLED)
            area = cv2.countNonZero(mask)
            
            positions_x.append(x + w / 2)
            positions_y.append(y + h / 2)
            areas.append(area)
        
        # Cluster by x position
        positions_x_arr = np.array(positions_x).reshape(-1, 1)
        positions_y_arr = np.array(positions_y).reshape(-1, 1)
        areas_arr = np.array(areas).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=self.config.n_clusters).fit_predict(positions_x_arr)
        
        # Calculate median positions and sum areas for each cluster
        x_medians = []
        y_medians = []
        area_sums = []
        
        for i in range(self.config.n_clusters):
            mask = kmeans == i
            x_medians.append(int(np.around(np.median(positions_x_arr[mask]))))
            y_medians.append(int(np.around(np.median(positions_y_arr[mask]))))
            area_sums.append(int(np.sum(areas_arr[mask])))
        
        # Sort by x position (left to right)
        sorted_data = sorted(zip(x_medians, y_medians, area_sums), key=lambda z: z[0])
        
        return (
            [d[0] for d in sorted_data],
            [d[1] for d in sorted_data],
            [d[2] for d in sorted_data]
        )
    
    def mask_green_in_image(
        self, 
        image: np.ndarray, 
        contours: list
    ) -> np.ndarray:
        """
        Black out green areas in the image (to remove plant stems from root detection).
        
        Args:
            image: Input BGR image.
            contours: Green contours to mask out.
            
        Returns:
            Image with green areas blacked out.
        """
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 0, 0), cv2.FILLED)
        return result
    
    def find_crop_start(self, contours: list, image_height: int) -> int:
        """
        Find the y-coordinate where roots start (below green areas).
        
        Args:
            contours: Green contours.
            image_height: Height of the image.
            
        Returns:
            Y coordinate for the top of the root region.
        """
        if not contours:
            return 0
        
        min_y = image_height
        for cnt in contours:
            y = cv2.boundingRect(cnt)[1]
            if y < min_y:
                min_y = y
        
        return min_y
