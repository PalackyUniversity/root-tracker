"""
Background removal utilities.

Removes background gradients and prepares images for root detection.
"""

import cv2
import numpy as np

from ..config import Config
from ..profiling import profile_operation


class BackgroundRemover:
    """
    Removes background gradient from images.
    
    Uses median blur subtraction to remove uneven lighting and
    background gradients, making roots more visible.
    
    Args:
        config: Configuration object.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._blur_kernel_size = 101  # Odd number for median blur
        self._smooth_kernel_size = 5
    
    @profile_operation("remove_gradient")
    def remove_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background gradient from the image.
        
        Subtracts a heavily blurred version of the image from itself
        to remove gradual lighting changes.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Grayscale image with gradient removed.
        """
        # Compute gradient using median blur
        blurred = cv2.medianBlur(image, self._blur_kernel_size)
        
        # Subtract gradient (convert to int to handle negatives)
        diff = image.astype(int) - blurred.astype(int)
        diff[diff < 0] = 0
        diff = diff.astype(np.uint8)
        
        # Smooth the result
        diff = cv2.medianBlur(diff, self._smooth_kernel_size)
        
        # Convert to grayscale
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        return diff
    
    @profile_operation("canny_edges")
    def compute_canny_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> np.ndarray:
        """
        Compute Canny edge detection on the image.
        
        Args:
            image: Input grayscale or BGR image.
            low_threshold: Lower threshold for edge detection.
            high_threshold: Upper threshold for edge detection.
            
        Returns:
            Binary edge image.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return cv2.Canny(image, low_threshold, high_threshold)
