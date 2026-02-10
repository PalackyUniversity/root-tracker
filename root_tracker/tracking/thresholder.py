"""
Root thresholding for segmentation.

Applies dual-threshold approach to segment roots from background.
"""

import cv2
import numpy as np

from ..config import Config


class RootThresholder:
    """
    Segments roots using dual-threshold approach.
    
    Uses a low and high threshold to identify roots:
    - Regions passing the low threshold that also contain pixels passing
      the high threshold are considered roots.
    
    Args:
        config: Configuration object with threshold settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply dual-threshold segmentation to identify roots.
        
        Args:
            image: Grayscale processed image.
            
        Returns:
            Binary mask of root regions.
        """
        # Apply low and high thresholds
        thresh_low = cv2.threshold(
            image, 
            self.config.threshold.low, 
            255, 
            cv2.THRESH_BINARY
        )[1]
        thresh_high = cv2.threshold(
            image, 
            self.config.threshold.high, 
            255, 
            cv2.THRESH_BINARY
        )[1]
        
        # Find contours in low threshold
        contours, hierarchy = cv2.findContours(
            thresh_low, 
            cv2.RETR_CCOMP, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if hierarchy is None:
            return np.zeros_like(image)
        
        # Keep contours that have significant high-threshold overlap
        thresh = np.zeros_like(thresh_low)
        to_draw = []
        ignore = []
        
        for cnt_n, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
            temp_mask = np.zeros_like(thresh_low)
            cv2.drawContours(temp_mask, [cnt], 0, 255, cv2.FILLED)
            
            # Check if this is an outer contour (not a hole)
            if h[3] == -1:
                if len(cnt) > self.config.threshold.min_contour_length:
                    # Check if enough of the contour passes high threshold
                    high_overlap = cv2.countNonZero(cv2.bitwise_and(thresh_high, temp_mask))
                    total_area = cv2.countNonZero(temp_mask)
                    if total_area > 0 and high_overlap / total_area > 0.2:
                        to_draw.append(cnt)
                    else:
                        ignore.append(cnt_n)
                else:
                    ignore.append(cnt_n)
        
        # Add holes of valid contours
        for cnt_n, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
            if h[3] != -1 and h[3] not in ignore:
                to_draw.append(cnt)
        
        cv2.drawContours(thresh, to_draw, -1, 255, cv2.FILLED)
        
        return thresh
    
    def apply_margins(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove root detections near side margins (box edges).
        
        Args:
            mask: Binary root mask.
            
        Returns:
            Mask with margins cleared.
        """
        if len(mask.shape) == 3:
            h, w, _ = mask.shape
        else:
            h, w = mask.shape
            
        top = round(self.config.margin_top * h)
        bottom = round(self.config.margin_bottom * h)
        left = round(self.config.margin_left * w)
        right = round(self.config.margin_right * w)
        
        result = mask.copy()
        if top > 0:
            result[:top, :] = 0
        if bottom > 0:
            result[-bottom:, :] = 0
        if left > 0:
            result[:, :left] = 0
        if right > 0:
            result[:, -right:] = 0
            
        return result

    def apply_user_mask(self, mask: np.ndarray, user_mask: np.ndarray | None) -> np.ndarray:
        """
        Apply user-defined mask to remove root detections in masked areas.

        Args:
            mask: Binary root detection mask.
            user_mask: User-defined mask (255 = remove roots, 0 = keep).

        Returns:
            Filtered mask with user-masked areas removed.
        """
        if user_mask is None:
            return mask

        # Remove detections where user_mask is 255
        return cv2.bitwise_and(mask, cv2.bitwise_not(user_mask))

    def filter_small_contours(self, mask: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Remove small contours from the mask.
        
        Args:
            mask: Binary root mask.
            
        Returns:
            Tuple of (filtered mask, list of valid contours).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.config.threshold.min_contour_area
        ]
        
        result = np.zeros_like(mask)
        cv2.drawContours(result, valid_contours, -1, 1, cv2.FILLED)
        
        return result, valid_contours
    
    def compute_new_growth(
        self, 
        diff_image: np.ndarray, 
        margins: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        """
        Compute metrics for newly grown roots from difference image.
        
        Args:
            diff_image: Difference between consecutive images.
            margins: Tuple of (top, bottom, left, right) margins in pixels.
            
        Returns:
            Tuple of (new_area, new_parts_count).
        """
        thresh_new = cv2.threshold(
            diff_image, 
            self.config.threshold.low, 
            255, 
            cv2.THRESH_BINARY
        )[1]
        
        # Apply margins
        top, bottom, left, right = margins
        if top > 0:
            thresh_new[:top, :] = 0
        if bottom > 0:
            thresh_new[-bottom:, :] = 0
        if left > 0:
            thresh_new[:, :left] = 0
        if right > 0:
            thresh_new[:, -right:] = 0
        
        new_area = cv2.countNonZero(thresh_new)
        
        # Count new root parts
        contours, _ = cv2.findContours(thresh_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_parts = len([
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.config.threshold.min_contour_area 
            and len(cnt) > self.config.threshold.min_contour_length
        ])
        
        return new_area, new_parts
