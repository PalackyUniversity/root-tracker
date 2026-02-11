"""
Image data models for Root Tracker.

Provides dataclasses for representing images and image series with their
associated metadata and processed results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class ImageData:
    """
    Represents a single image with its metadata and processed results.
    
    This class holds both the original image path/metadata and various
    processed versions of the image used throughout the pipeline.
    
    Attributes:
        date: Timestamp when the image was captured.
        path: Path to the original image file.
        barcode: Group/barcode identifier from the filename.
        barcode_read: Barcode read from the image content (if available).
        
    Processing Results (populated during pipeline execution):
        image: The cropped and preprocessed image (BGR).
        process: Grayscale processed image with gradient removed.
        canny: Edge-detected version using Canny algorithm.
        diff: Difference from previous image in series.
        
    Plant Detection Results:
        green_areas: List of green area sizes per detected plant.
        positions_x: X coordinates of plant centroids.
        positions_y: Y coordinates of plant centroids.
        
    Analysis Results:
        total_length: Total skeleton length of all roots.
        total_area: Total area of all roots.
        new_area: Area of newly grown roots (from diff).
        new_parts: Count of new root segments.
        plant_length: Length of roots per plant.
        longest: Length of main root per plant.
        colored_samples: Pixel sets for each plant's roots (for tracking).
    """
    date: datetime
    path: str
    barcode: str
    
    # Barcode read from image
    barcode_read: str = ""
    barcode_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h) bounding box
    barcode_mismatch: bool = False  # True if detected barcode doesn't match expected
    barcode_detected: bool = False  # True if barcode detection was attempted
    
    # Processed image arrays
    image: Optional[np.ndarray] = field(default=None, repr=False)
    process: Optional[np.ndarray] = field(default=None, repr=False)
    canny: Optional[np.ndarray] = field(default=None, repr=False)
    diff: Optional[np.ndarray] = field(default=None, repr=False)
    image_annotated: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Plant detection results
    green_areas: list[int] = field(default_factory=list)
    positions_x: list[int] = field(default_factory=list)
    positions_y: list[int] = field(default_factory=list)
    
    # Analysis results
    total_length: Optional[int] = None
    total_area: Optional[int] = None
    new_area: Optional[int] = None
    new_parts: Optional[int] = None
    plant_length: list[int] = field(default_factory=list)
    longest: list[int] = field(default_factory=list)
    colored_samples: dict[int, set] = field(default_factory=dict)
    
    def clear_tracking_results(self) -> None:
        """Clear all tracking/analysis results, preserving preprocessing data."""
        self.total_length = None
        self.total_area = None
        self.new_area = None
        self.new_parts = None
        self.plant_length = []
        self.longest = []
        self.colored_samples = {}
        self.image_annotated = None

    def clear_preprocessing_results(self) -> None:
        """Clear all preprocessing results (and tracking, since it depends on them)."""
        self.clear_tracking_results()
        self.image = None
        self.process = None
        self.canny = None
        self.diff = None
        self.green_areas = []
        self.positions_x = []
        self.positions_y = []

    @property
    def is_set_aside(self) -> bool:
        """Check if image is in 'aside' folder."""
        # Simple check if 'aside' is one of the path components
        return 'aside' in Path(self.path).parts

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return Path(self.path).name

    def copy_for_editing(self) -> "ImageData":
        """
        Create a shallow copy for GUI editing purposes.
        
        This allows the GUI to modify results without affecting the original.
        """
        return ImageData(
            date=self.date,
            path=self.path,
            barcode=self.barcode,
            barcode_read=self.barcode_read,
            image=self.image.copy() if self.image is not None else None,
            process=self.process.copy() if self.process is not None else None,
            canny=self.canny.copy() if self.canny is not None else None,
            diff=self.diff.copy() if self.diff is not None else None,
            image_annotated=self.image_annotated.copy() if self.image_annotated is not None else None,
            green_areas=self.green_areas.copy(),
            positions_x=self.positions_x.copy(),
            positions_y=self.positions_y.copy(),
            total_length=self.total_length,
            total_area=self.total_area,
            new_area=self.new_area,
            new_parts=self.new_parts,
            plant_length=self.plant_length.copy(),
            longest=self.longest.copy(),
            colored_samples={k: v.copy() for k, v in self.colored_samples.items()},
        )


@dataclass
class PipelineState:
    """Tracks which pipeline steps have been completed for a series."""
    preprocessed: bool = False
    preprocess_config_hash: str = ""
    tracked: bool = False
    tracking_config_hash: str = ""
    last_statistics: list = field(default_factory=list)

    def invalidate_from(self, step: str) -> None:
        """Invalidate this step and all subsequent steps."""
        steps = ['preprocess', 'track']
        idx = steps.index(step)
        for s in steps[idx:]:
            if s == 'preprocess':
                self.preprocessed = False
                self.preprocess_config_hash = ""
            elif s == 'track':
                self.tracked = False
                self.tracking_config_hash = ""
                self.last_statistics = []


@dataclass
class ImageSeries:
    """
    A collection of images belonging to the same group/barcode.

    Images are stored sorted by date for time-series analysis.

    Attributes:
        group: The group identifier (barcode).
        images: List of ImageData objects, sorted by date.
        user_mask: Applied/committed mask for removing root detections (shared across all images).
        working_mask: Currently edited mask (uncommitted changes).
    """
    group: str
    images: list[ImageData] = field(default_factory=list)
    pipeline_state: PipelineState = field(default_factory=PipelineState)

    # User-defined mask for root removal (applies to all images in series)
    user_mask: Optional[np.ndarray] = field(default=None, repr=False)
    working_mask: Optional[np.ndarray] = field(default=None, repr=False)

    def clear_tracking_results(self) -> None:
        """Clear tracking results for all images and reset pipeline state."""
        for img in self.images:
            img.clear_tracking_results()
        self.pipeline_state.invalidate_from('track')

    def clear_preprocessing_results(self) -> None:
        """Clear all preprocessing and tracking results (including masks)."""
        for img in self.images:
            img.clear_preprocessing_results()
        self.pipeline_state.invalidate_from('preprocess')
        # Mask was drawn on preprocessed image coords — no longer valid
        self.user_mask = None
        self.working_mask = None

    def __post_init__(self) -> None:
        """Sort images by date after initialization."""
        self.sort_by_date()
    
    def sort_by_date(self) -> None:
        """Sort images by their capture date."""
        self.images.sort(key=lambda img: img.date)
    
    def add_image(self, image: ImageData) -> None:
        """Add an image and maintain sorted order."""
        self.images.append(image)
        self.sort_by_date()
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __iter__(self):
        return iter(self.images)
    
    def __getitem__(self, index: int) -> ImageData:
        return self.images[index]
    
    @property
    def date_range(self) -> tuple[datetime, datetime]:
        """Get the date range of images in this series."""
        if not self.images:
            raise ValueError("No images in series")
        return self.images[0].date, self.images[-1].date
    
    @property
    def barcode(self) -> str:
        """Get the barcode (last part of group path)."""
        return self.group.split("/")[-1]
    
    @property
    def has_barcode_warning(self) -> bool:
        """Check if any image has a barcode mismatch."""
        return any(img.barcode_mismatch for img in self.images)

    @property
    def is_set_aside(self) -> bool:
        """Check if all images in series are set aside."""
        # If any image is set aside, the whole group is effectively set aside
        # (since we move the whole group folder usually)
        return all(img.is_set_aside for img in self.images) if self.images else False

    def has_pending_mask_changes(self) -> bool:
        """Check if there are uncommitted mask edits."""
        # No pending changes if working_mask doesn't exist
        if self.working_mask is None:
            return False

        # No pending changes if user_mask doesn't exist and working_mask is all zeros
        if self.user_mask is None:
            return np.any(self.working_mask > 0)

        # Compare working_mask with user_mask
        return not np.array_equal(self.working_mask, self.user_mask)
