"""
Background removal utilities.

Removes background gradients and prepares images for root detection.
"""

import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import current_process

from ..config import Config
from .gpu_background import try_subtract_median


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
        # Optional exact GPU histogram median with fused saturated subtraction.
        # Missing CUDA, small images and batch children retain the CPU path.
        diff = try_subtract_median(image, self._blur_kernel_size)
        if diff is None:
            blurred = self._median_background(image)
            diff = cv2.subtract(image, blurred)

        # Smooth the result
        diff = cv2.medianBlur(diff, self._smooth_kernel_size)
        
        # Convert to grayscale
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        return diff

    def _median_background(self, image: np.ndarray) -> np.ndarray:
        """Exact median in overlapping strips, bounded to four CPU workers.

        Keep the full kernel radius around each strip so internal boundaries
        never enter the returned pixels. Batch child processes already run in
        parallel and use a single filter call to avoid nested worker pools.
        """
        workers = min(4, os.cpu_count() or 1)
        if (workers == 1 or image.shape[0] < 1024 or
                current_process().name != 'MainProcess'):
            return cv2.medianBlur(image, self._blur_kernel_size)
        radius = self._blur_kernel_size // 2
        height = image.shape[0]
        result = np.empty_like(image)

        def filter_strip(index):
            start, stop = height * index // workers, height * (index + 1) // workers
            top, bottom = max(0, start - radius), min(height, stop + radius)
            filtered = cv2.medianBlur(image[top:bottom], self._blur_kernel_size)
            result[start:stop] = filtered[start - top:stop - top]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(filter_strip, range(workers)))
        return result
    
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
