"""
Image cropping and rotation utilities.

Handles automatic cropping based on blue background detection and image rotation.
"""

import cv2
import numpy as np

from ..config import Config


# Rotation mapping for OpenCV
ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class ImageCropper:
    """
    Handles image rotation and automatic cropping.
    
    Uses blue background detection to find the region of interest and
    crops the image accordingly.
    
    Args:
        config: Configuration object with crop settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image according to configuration.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Rotated image, or original if rotation is 0.
        """
        if self.config.rotation == 0:
            return image
        
        rotation_code = ROTATION_MAP.get(self.config.rotation)
        if rotation_code is None:
            raise ValueError(f"Invalid rotation: {self.config.rotation}")
        
        return cv2.rotate(image, rotation_code)
    
    def auto_crop_to_blue_background(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically crop image based on blue background detection.
        
        Finds the bounding box of all blue regions and crops to that area.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Cropped image containing the blue background region.
        """
        # Convert to HSV and threshold for blue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(
            hsv, 
            self.config.crop.blue_hsv_lower, 
            self.config.crop.blue_hsv_upper
        )
        
        # Speed optimization - erode to remove noise
        thresh = cv2.erode(thresh, None, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Find bounding box of all significant contours
        min_x, min_y = image.shape[1], image.shape[0]
        max_x, max_y = 0, 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
        
        # Crop to bounding box
        cropped = image[min_y:max_y, min_x:max_x]
        
        # Apply vertical crop ratios
        height = cropped.shape[0]
        top = round(height * self.config.crop.top_ratio)
        bottom = round(height * self.config.crop.bottom_ratio)
        
        return cropped[top:bottom]
    
    def margin_offsets(self, height: int, width: int) -> tuple[int, int, int, int]:
        """
        Convert the configured margin fractions to pixel sizes.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Tuple of (top, bottom, left, right) margins in pixels.
        """
        return (
            round(self.config.margin_top * height),
            round(self.config.margin_bottom * height),
            round(self.config.margin_left * width),
            round(self.config.margin_right * width),
        )

    def crop_sides(self, image: np.ndarray) -> np.ndarray:
        """
        Apply side margin cropping (removes box edges).
        
        Args:
            image: Input image.
            
        Returns:
            Image with side margins removed.
        """
        margin = round(self.config.margin * image.shape[1])
        result = image.copy()
        result[:, :margin] = 0
        result[:, -margin:] = 0
        return result
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full cropping pipeline: rotate and auto-crop.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Rotated and cropped image.
        """
        rotated = self.rotate(image)
        cropped = self.auto_crop_to_blue_background(rotated)
        return cropped
