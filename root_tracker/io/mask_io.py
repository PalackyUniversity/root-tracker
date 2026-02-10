"""
Mask persistence module for Root Tracker.

Provides functions to save and load user-defined masks for root removal.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from ..config import Config
from ..models import ImageSeries


def get_mask_path(series: ImageSeries, config: Config) -> Path:
    """
    Get the file path for a series mask.

    Args:
        series: The image series.
        config: Configuration object.

    Returns:
        Path to the mask file ({group_name}_mask.png).
    """
    # Use the first image's directory as the base location
    if not series.images:
        raise ValueError("Cannot get mask path for empty series")

    image_dir = Path(series.images[0].path).parent
    mask_filename = f"{series.barcode}_mask.png"
    return image_dir / mask_filename


def save_mask(series: ImageSeries, config: Config) -> Optional[Path]:
    """
    Save the user mask for a series to disk.

    Args:
        series: The image series with user_mask to save.
        config: Configuration object.

    Returns:
        Path to the saved mask file, or None if no mask to save.
    """
    if series.user_mask is None:
        return None

    try:
        mask_path = get_mask_path(series, config)

        # Save as PNG (lossless, single channel)
        cv2.imwrite(str(mask_path), series.user_mask)

        return mask_path
    except Exception as e:
        raise IOError(f"Failed to save mask for {series.group}: {e}")


def load_mask(series: ImageSeries, config: Config) -> Optional[np.ndarray]:
    """
    Load the user mask for a series from disk.

    Args:
        series: The image series to load mask for.
        config: Configuration object.

    Returns:
        Loaded mask as binary numpy array, or None if no mask file exists.
    """
    if not series.images:
        return None

    try:
        mask_path = get_mask_path(series, config)

        # Check if mask file exists
        if not mask_path.exists():
            return None

        # Load as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return None

        return mask
    except Exception:
        # If loading fails, just return None (mask is optional)
        return None


def delete_mask(series: ImageSeries, config: Config) -> bool:
    """
    Delete the mask file for a series.

    Args:
        series: The image series.
        config: Configuration object.

    Returns:
        True if mask file was deleted, False if it didn't exist.
    """
    if not series.images:
        return False

    try:
        mask_path = get_mask_path(series, config)

        if mask_path.exists():
            mask_path.unlink()
            return True
        return False
    except Exception:
        return False
