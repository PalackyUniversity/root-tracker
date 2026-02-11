"""
Image loading and series creation.

Handles loading images from disk and organizing them into series.
"""

import os
from glob import glob
from datetime import datetime
import parse
import cv2
import numpy as np

from ..config import Config
from ..models import ImageData, ImageSeries
from . import mask_io


class ImageLoader:
    """
    Loads images from disk and creates image series.
    
    Parses filenames to extract metadata and groups images by their
    barcode/group identifier.
    
    Args:
        config: Configuration object with data paths.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def parse_filename(self, filepath: str) -> tuple[str, datetime] | None:
        """
        Parse a filename to extract group and date.
        
        Args:
            filepath: Full path to the image file.
            
        Returns:
            Tuple of (group, date) or None if parsing fails.
        """
        # Extract just the filename without path and extension
        filename = os.path.basename(filepath)
        basename = os.path.splitext(filename)[0]
        
        try:
            parsed = parse.parse(
                self.config.data.filename_template, 
                basename
            ).named
            
            # Handle date format quirks (e.g., removing "_1" suffix)
            date_str = parsed["date"].replace("_1", "")
            date = datetime.strptime(date_str, self.config.data.date_format)
            
            return parsed["group"], date
        except (AttributeError, ValueError, KeyError):
            return None
    
    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            filepath: Path to the image file.
            
        Returns:
            BGR image array.
        """
        return cv2.imread(filepath)
    
    def discover_images(self) -> list[str]:
        """
        Find all image files in the input directory.
        
        Returns:
            List of image file paths.
        """
        # Main directory
        main_pattern = os.path.join(self.config.data.input, "*")
        files = glob(main_pattern)
        
        # Aside directory
        aside_pattern = os.path.join(self.config.data.input, "aside", "*")
        files.extend(glob(aside_pattern))
        
        return files
    
    def create_series(self) -> dict[str, ImageSeries]:
        """
        Load all images and organize into series.
        
        Scans the input directory, parses filenames, and groups
        images by their barcode/group identifier.
        
        Returns:
            Dictionary mapping group names to ImageSeries objects.
        """
        series_dict: dict[str, ImageSeries] = {}
        
        for filepath in self.discover_images():
            parsed = self.parse_filename(filepath)
            if parsed is None:
                continue
            
            group, date = parsed
            barcode = group.split("/")[-1]
            
            image_data = ImageData(
                date=date,
                path=filepath,
                barcode=barcode
            )
            
            if group not in series_dict:
                series_dict[group] = ImageSeries(group=group)

            series_dict[group].add_image(image_data)

        # Auto-load masks for each series
        for series in series_dict.values():
            mask = mask_io.load_mask(series, self.config)
            if mask is not None:
                series.user_mask = mask
                # Initialize working_mask as a copy of user_mask
                series.working_mask = mask.copy()

        return series_dict
    
    def load_series_images(self, series: ImageSeries) -> None:
        """
        Load the actual image data for all images in a series.
        
        This is separate from create_series() to allow lazy loading.
        
        Args:
            series: ImageSeries to load images for.
        """
        for image_data in series.images:
            image_data.image = self.load_image(image_data.path)
