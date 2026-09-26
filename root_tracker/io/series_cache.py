"""
Disk-backed series cache for memory-bounded batch processing.

Stores processed image arrays and metadata as .npz files
in a .root_tracker_cache/ directory next to the input images.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..models import ImageSeries

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".root_tracker_cache"

# Image array fields to persist (per ImageData)
_ARRAY_FIELDS = ("image", "process", "canny", "diff", "image_annotated")

# Scalar fields to persist in metadata (per ImageData)
_SCALAR_FIELDS = (
    "total_length", "total_area", "new_area", "new_parts",
)

# List fields to persist in metadata (per ImageData)
_LIST_FIELDS = (
    "plant_length", "longest", "positions_x", "positions_y", "green_areas",
)

# Barcode fields to persist in metadata (per ImageData)
_BARCODE_FIELDS = (
    "barcode_detected", "barcode_read", "barcode_mismatch",
    "barcode_not_found", "barcode_rect",
)


def _sanitize_filename(name: str) -> str:
    """Sanitize a group name for use as a filename."""
    return re.sub(r'[^\w\-.]', '_', name)


def get_cache_path(series: ImageSeries, config: Config) -> Path:
    """
    Get the cache file path for a series.

    Returns:
        Path to the .npz cache file in {input_dir}/.root_tracker_cache/
    """
    input_dir = Path(config.data.input)
    cache_dir = input_dir / CACHE_DIR_NAME
    filename = f"{_sanitize_filename(series.group)}.npz"
    return cache_dir / filename


def save_series(series: ImageSeries, config: Config) -> Optional[Path]:
    """
    Save arrays and metadata to .npz (optionally compressed for smaller files).

    Args:
        series: The image series to cache.
        config: Configuration object (used for path and hash info).

    Returns:
        Path to the saved cache file, or None on failure.
    """
    cache_path = get_cache_path(series, config)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}

        # Per-image arrays
        for idx, img in enumerate(series.images):
            for field_name in _ARRAY_FIELDS:
                arr = getattr(img, field_name, None)
                if arr is not None:
                    arrays[f"{field_name}_{idx}"] = arr

        # User mask
        if series.user_mask is not None:
            arrays["user_mask"] = series.user_mask

        # Build metadata dict
        state = series.pipeline_state
        metadata = {
            "preprocess_config_hash": state.preprocess_config_hash,
            "tracking_config_hash": state.tracking_config_hash,
            "preprocessed": state.preprocessed,
            "tracked": state.tracked,
            "images": [],
            "statistics": [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in state.last_statistics
            ],
        }

        for img in series.images:
            img_meta = {}
            for field_name in _SCALAR_FIELDS:
                img_meta[field_name] = getattr(img, field_name, None)
            for field_name in _LIST_FIELDS:
                val = getattr(img, field_name, [])
                img_meta[field_name] = [int(v) if isinstance(v, (int, np.integer)) else v for v in val]
            for field_name in _BARCODE_FIELDS:
                val = getattr(img, field_name, None)
                if field_name == "barcode_rect" and val is not None:
                    img_meta[field_name] = list(val)
                else:
                    img_meta[field_name] = val
            metadata["images"].append(img_meta)

        # Encode metadata as JSON -> uint8 numpy array
        meta_json = json.dumps(metadata, default=_json_default)
        arrays["_metadata"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

        # Compression dominates interactive latency on full-resolution groups.
        # Both variants are standard NPZ and remain readable by older versions.
        save = np.savez_compressed if config.data.cache_compressed else np.savez
        save(str(cache_path), **arrays)
        logger.debug("Saved cache for series %s to %s", series.group, cache_path)
        return cache_path

    except Exception as e:
        logger.warning("Failed to save cache for series %s: %s", series.group, e)
        return None



def load_series_state(series: ImageSeries, config: Config) -> bool:
    """
    Restore a series' metadata and state from a .npz cache file, WITHOUT loading heavy image arrays.

    Used during startup/loading to quickly populate the UI with processed state.

    Args:
        series: The image series to restore into.
        config: Configuration object.

    Returns:
        True if successfully loaded state, False if cache missing/corrupt.
    """
    cache_path = get_cache_path(series, config)

    if not cache_path.exists():
        return False

    try:
        # Load only metadata
        # We must load the zip file but we extract only _metadata and user_mask
        data = np.load(str(cache_path), allow_pickle=False)

        if "_metadata" not in data:
            data.close()
            return False

        meta_bytes = data["_metadata"].tobytes()
        metadata = json.loads(meta_bytes.decode("utf-8"))

        # Restore pipeline state
        state = series.pipeline_state
        state.preprocess_config_hash = metadata.get("preprocess_config_hash", "")
        state.tracking_config_hash = metadata.get("tracking_config_hash", "")
        state.preprocessed = metadata.get("preprocessed", False)
        state.tracked = metadata.get("tracked", False)

        # Restore statistics
        state.last_statistics = metadata.get("statistics", [])

        # Restore per-image metadata
        images_meta = metadata.get("images", [])
        for idx, img in enumerate(series.images):
            if idx < len(images_meta):
                img_meta = images_meta[idx]
                for field_name in _SCALAR_FIELDS:
                    if field_name in img_meta:
                        setattr(img, field_name, img_meta[field_name])
                for field_name in _LIST_FIELDS:
                    if field_name in img_meta:
                        setattr(img, field_name, list(img_meta[field_name]))
                for field_name in _BARCODE_FIELDS:
                    if field_name in img_meta:
                        val = img_meta[field_name]
                        if field_name == "barcode_rect" and val is not None:
                            setattr(img, field_name, tuple(val))
                        else:
                            setattr(img, field_name, val)

        # Restore user mask if present (usually small enough to keep in memory)
        if "user_mask" in data:
            series.user_mask = data["user_mask"]
            # Initialize working mask copy
            series.working_mask = series.user_mask.copy()

        data.close()
        # logger.debug("Loaded cache state for series %s", series.group)
        return True

    except Exception as e:
        logger.warning("Failed to load cache state for series %s: %s", series.group, e)
        return False


def load_series(series: ImageSeries, config: Config) -> bool:
    """
    Restore a series' arrays and metadata from a .npz cache file.

    Args:
        series: The image series to restore into.
        config: Configuration object.

    Returns:
        True if successfully loaded, False if cache missing/corrupt.
    """
    cache_path = get_cache_path(series, config)

    if not cache_path.exists():
        return False

    try:
        data = np.load(str(cache_path), allow_pickle=False)

        # Load metadata
        if "_metadata" not in data:
            return False

        meta_bytes = data["_metadata"].tobytes()
        metadata = json.loads(meta_bytes.decode("utf-8"))

        # Restore pipeline state
        state = series.pipeline_state
        state.preprocess_config_hash = metadata.get("preprocess_config_hash", "")
        state.tracking_config_hash = metadata.get("tracking_config_hash", "")
        state.preprocessed = metadata.get("preprocessed", False)
        state.tracked = metadata.get("tracked", False)

        # Restore statistics (as raw dicts — export just needs .to_dict() or dicts)
        state.last_statistics = metadata.get("statistics", [])

        # Restore per-image arrays and metadata
        images_meta = metadata.get("images", [])
        for idx, img in enumerate(series.images):
            # Arrays
            for field_name in _ARRAY_FIELDS:
                key = f"{field_name}_{idx}"
                if key in data:
                    setattr(img, field_name, data[key])

            # Scalar + list + barcode metadata
            if idx < len(images_meta):
                img_meta = images_meta[idx]
                for field_name in _SCALAR_FIELDS:
                    if field_name in img_meta:
                        setattr(img, field_name, img_meta[field_name])
                for field_name in _LIST_FIELDS:
                    if field_name in img_meta:
                        setattr(img, field_name, list(img_meta[field_name]))
                for field_name in _BARCODE_FIELDS:
                    if field_name in img_meta:
                        val = img_meta[field_name]
                        if field_name == "barcode_rect" and val is not None:
                            setattr(img, field_name, tuple(val))
                        else:
                            setattr(img, field_name, val)

        # Restore user mask
        if "user_mask" in data:
            series.user_mask = data["user_mask"]

        data.close()
        logger.debug("Loaded cache for series %s from %s", series.group, cache_path)
        return True

    except Exception as e:
        logger.warning("Failed to load cache for series %s: %s", series.group, e)
        return False


def has_valid_cache(series: ImageSeries, config: Config) -> bool:
    """
    Check if a valid cache exists for the series with matching config hashes.

    Only reads metadata, not the full arrays.

    Args:
        series: The image series.
        config: Configuration object.

    Returns:
        True if cache exists and config hashes match.
    """
    cache_path = get_cache_path(series, config)

    if not cache_path.exists():
        return False

    try:
        data = np.load(str(cache_path), allow_pickle=False)

        if "_metadata" not in data:
            data.close()
            return False

        meta_bytes = data["_metadata"].tobytes()
        metadata = json.loads(meta_bytes.decode("utf-8"))
        data.close()

        # Check config hashes match current config
        cached_preprocess_hash = metadata.get("preprocess_config_hash", "")
        cached_tracking_hash = metadata.get("tracking_config_hash", "")

        current_preprocess_hash = config.preprocess_config_hash()
        current_tracking_hash = config.tracking_config_hash()

        return (
            cached_preprocess_hash == current_preprocess_hash
            and cached_tracking_hash == current_tracking_hash
        )

    except Exception as e:
        logger.warning("Failed to check cache for series %s: %s", series.group, e)
        return False


def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
