"""
Root Tracker - A tool for tracking and analyzing plant root growth.

This package provides modules for:
- Image preprocessing (cropping, rotation, background removal)
- Image registration (aligning time series)
- Root tracking and segmentation
- Statistical analysis of root growth
- Export of results

Designed for step-by-step pipeline execution, suitable for GUI integration.
"""

__version__ = "2.0.0"

from .config import Config
from .pipeline import RootTrackingPipeline

__all__ = ["Config", "RootTrackingPipeline", "__version__"]
