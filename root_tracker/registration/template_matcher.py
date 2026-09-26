"""
Template matching based image registration.

Aligns time series images using template matching on edge features.
"""

import cv2
import numpy as np

from ..config import Config
from ..models import ImageData


class ImageRegistrator:
    """
    Registers (aligns) images in a time series.
    
    Uses template matching on Canny edge images to align subsequent
    images to maintain consistent geometry across the time series.
    
    Args:
        config: Configuration object with registration settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._margin_divisor = 4  # Margin is 1/4 of image dimensions
    
    def align_to_template(
        self, 
        template_canny: np.ndarray,
        target_image: np.ndarray,
        target_canny: np.ndarray,
        target_process: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Align a target image to a template using template matching.
        
        Args:
            template_canny: Canny edges of the template (reference) image.
            target_image: BGR image to align.
            target_canny: Canny edges of the target image.
            target_process: Processed grayscale of the target image.
            
        Returns:
            Tuple of (aligned_image, aligned_canny, aligned_process, offset_x, offset_y).
        """
        h, w = template_canny.shape[:2]
        
        # Calculate margins for border expansion
        mx = target_canny.shape[1] // self._margin_divisor
        my = target_canny.shape[0] // self._margin_divisor
        
        # Add border to target images
        target_canny_padded = cv2.copyMakeBorder(
            target_canny, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0
        )
        target_image_padded = cv2.copyMakeBorder(
            target_image, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0
        )
        target_process_padded = cv2.copyMakeBorder(
            target_process, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0
        )
        
        # Perform template matching
        result = cv2.matchTemplate(target_canny_padded, template_canny, cv2.TM_CCOEFF)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        left, top = max_loc
        
        # Crop aligned images to template size
        aligned_image = target_image_padded[top:top + h, left:left + w]
        aligned_canny = target_canny_padded[top:top + h, left:left + w]
        aligned_process = target_process_padded[top:top + h, left:left + w]
        
        # Calculate position offsets for updating coordinates
        offset_x = mx - left
        offset_y = my - top
        
        return aligned_image, aligned_canny, aligned_process, offset_x, offset_y
    
    def register_series(self, images: list[ImageData]) -> None:
        """
        Register all images in a series to the first image.
        
        Modifies ImageData objects in place with aligned images and
        updated position coordinates.
        
        Args:
            images: List of ImageData objects with canny and process attributes set.
        """
        if len(images) < 2:
            return
            
        # If registration is disabled, just compute difference images without alignment
        if not self.config.registration.enabled:
            for n in range(len(images) - 1):
                if images[n].process is None or images[n + 1].process is None:
                    continue
                    
                # Compute difference image (simple subtraction)
                images[n + 1].diff = cv2.subtract(images[n + 1].process, images[n].process)
            return
        
        for n in range(len(images) - 1):
            template = images[n].canny
            
            if template is None or images[n + 1].canny is None:
                continue
            
            # Align next image to current
            aligned_image, aligned_canny, aligned_process, offset_x, offset_y = (
                self.align_to_template(
                    template,
                    images[n + 1].image,
                    images[n + 1].canny,
                    images[n + 1].process
                )
            )
            
            # Update image data
            images[n + 1].image = aligned_image
            images[n + 1].canny = aligned_canny
            images[n + 1].process = aligned_process
            
            # Update position coordinates
            images[n + 1].positions_x = [x + offset_x for x in images[n + 1].positions_x]
            images[n + 1].positions_y = [y + offset_y for y in images[n + 1].positions_y]
            
            # Compute difference image for new growth detection
            images[n + 1].diff = cv2.subtract(images[n + 1].process, images[n].process)
