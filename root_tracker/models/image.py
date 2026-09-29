"""
Image data models for Root Tracker.

Provides dataclasses for representing images and image series with their
associated metadata and processed results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..io.rsml import RSMLDocument


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
    barcode_not_found: bool = False  # True if detection ran but no barcode found
    barcode_detected: bool = False  # True if barcode detection was attempted
    
    # Processed image arrays
    image: Optional[np.ndarray] = field(default=None, repr=False)
    process: Optional[np.ndarray] = field(default=None, repr=False)
    canny: Optional[np.ndarray] = field(default=None, repr=False)
    diff: Optional[np.ndarray] = field(default=None, repr=False)
    image_annotated: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Sparse pixels beneath depth markers: columns y, x, B, G, R.
    root_depth_background: Optional[np.ndarray] = field(default=None, repr=False)
    # Sparse pixels beneath white root connectors: y, x, B, G, R.
    root_link_background: Optional[np.ndarray] = field(default=None, repr=False)

    # Unmasked main-root pixels: plant index, x, y.
    main_root_samples: Optional[np.ndarray] = field(default=None, repr=False)
    tracking_overlay: Optional[np.ndarray] = field(default=None, repr=False)

    # Affine mapping from detected-plate pixels to the displayed processed image.
    plate_transform: list[float] = field(default_factory=list)

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
    
    # Export-only snapshot; never consumed by tracking. None identifies old caches.
    rsml_samples: dict[int, np.ndarray] | None = field(default=None, repr=False)

    # Native geometry retained before exclusions, so Restore survives manual edits.
    rsml_unmasked_samples: dict[int, np.ndarray] | None = field(default=None, repr=False)

    # User replacements are authoritative and survive cache/result invalidation.
    rsml_document: Optional["RSMLDocument"] = field(default=None, repr=False)
    rsml_background: Optional[np.ndarray] = field(default=None, repr=False)
    # Manual corrections before reversible group exclusions are applied.
    rsml_unmasked_document: Optional["RSMLDocument"] = field(default=None, repr=False)
    rsml_original_document: Optional["RSMLDocument"] = field(default=None, repr=False)
    rsml_root_sources: tuple[int, ...] = field(default_factory=tuple, repr=False)

    # Source metadata is independent of processing caches and settings.
    camera_metadata: dict[str, str] | None = field(default=None, repr=False)
    camera_metadata_error: str = ""

    def clear_tracking_results(self) -> None:
        """Clear all tracking/analysis results, preserving preprocessing data."""
        self.root_depth_background = None
        self.rsml_unmasked_samples = None
        self.main_root_samples = None
        self.tracking_overlay = None
        if self.rsml_document is not None:
            self.colored_samples = {}
            self.rsml_samples = None
            self.image_annotated = None
            return
        self.root_link_background = None
        self.total_length = None
        self.total_area = None
        self.new_area = None
        self.new_parts = None
        self.plant_length = []
        self.longest = []
        self.colored_samples = {}
        self.rsml_samples = None
        self.image_annotated = None

    def tracking_preview(self, show_max_root_depth: bool = True):
        """Hide depth markers without recomputing roots or changing exports."""
        if show_max_root_depth or self.image_annotated is None or self.root_depth_background is None:
            return self.image_annotated
        preview = self.image_annotated.copy()
        pixels = self.root_depth_background
        preview[pixels[:, 0], pixels[:, 1]] = pixels[:, 2:]
        return preview

    def root_link_pixels(self, plant_origins=None):
        """Get connector underlays, recovering them from older tracking caches."""
        if self.root_link_background is None and self.tracking_overlay is not None and self.image is not None:
            from ..config import Config
            from ..tracking.root_linker import RootLinker
            pixels = self.tracking_overlay
            pixels = pixels[np.all(pixels[:, 2:] == 255, axis=1)]
            origins = (plant_origins if plant_origins is not None
                       else list(zip(self.positions_x, self.positions_y)))
            background = RootLinker(Config()).draw_annotations(
                self.image, [x for x, _ in origins],
                [y for _, y in origins], [], {})
            ys, xs = pixels[:, 0], pixels[:, 1]
            self.root_link_background = np.column_stack((ys, xs, background[ys, xs])).astype(np.int32)
        return self.root_link_background

    def root_link_preview(self, preview, plant_origins=None):
        """Separate visible connectors from the bitmap without erasing roots."""
        pixels = self.root_link_pixels(plant_origins)
        if pixels is None or not len(pixels):
            return preview, None
        # Root strokes are painted after connectors and must retain priority.
        pixels = pixels[np.all(preview[pixels[:, 0], pixels[:, 1]] == 255, axis=1)]
        if not len(pixels):
            return preview, None
        preview = preview.copy()
        preview[pixels[:, 0], pixels[:, 1]] = pixels[:, 2:]
        return preview, pixels

    def clear_preprocessing_results(self) -> None:
        """Clear all preprocessing results (and tracking, since it depends on them)."""
        self.clear_tracking_results()
        self.plate_transform = []
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
            camera_metadata=self.camera_metadata.copy() if self.camera_metadata is not None else None,
            camera_metadata_error=self.camera_metadata_error,
            barcode_read=self.barcode_read,
            barcode_mismatch=self.barcode_mismatch,
            barcode_not_found=self.barcode_not_found,
            barcode_detected=self.barcode_detected,
            image=self.image.copy() if self.image is not None else None,
            process=self.process.copy() if self.process is not None else None,
            canny=self.canny.copy() if self.canny is not None else None,
            diff=self.diff.copy() if self.diff is not None else None,
            image_annotated=self.image_annotated.copy() if self.image_annotated is not None else None,
            root_depth_background=self.root_depth_background.copy() if self.root_depth_background is not None else None,
            root_link_background=self.root_link_background.copy() if self.root_link_background is not None else None,
            main_root_samples=self.main_root_samples.copy() if self.main_root_samples is not None else None,
            tracking_overlay=self.tracking_overlay.copy() if self.tracking_overlay is not None else None,
            green_areas=self.green_areas.copy(),
            plate_transform=self.plate_transform.copy(),
            positions_x=self.positions_x.copy(),
            positions_y=self.positions_y.copy(),
            total_length=self.total_length,
            total_area=self.total_area,
            new_area=self.new_area,
            new_parts=self.new_parts,
            plant_length=self.plant_length.copy(),
            longest=self.longest.copy(),
            colored_samples={k: v.copy() for k, v in self.colored_samples.items()},
            rsml_document=self.rsml_document,
            rsml_unmasked_document=self.rsml_unmasked_document,
            rsml_original_document=self.rsml_original_document,
            rsml_root_sources=self.rsml_root_sources,
            rsml_background=self.rsml_background.copy() if self.rsml_background is not None else None,
            rsml_unmasked_samples=({k: v.copy() for k, v in self.rsml_unmasked_samples.items()}
                                   if self.rsml_unmasked_samples is not None else None),
            rsml_samples=({k: v.copy() for k, v in self.rsml_samples.items()}
                          if self.rsml_samples is not None else None),
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
    # Committed processing parameters for this group; paths and GUI state stay global.
    processing_settings: Optional[dict] = field(default=None, repr=False)

    def plant_origins(self, count: int) -> list[tuple[int, int]]:
        """Shared tracking origins, in registered image coordinates.

        Keep per-image detections as inputs, but use the same rounded median
        for preview and tracking. RSML replacements own their geometry.
        """
        origins = []
        for index in range(count):
            points = [(image.positions_x[index], image.positions_y[index])
                      for image in self.images
                      if image.rsml_document is None
                      and index < min(len(image.positions_x), len(image.positions_y))]
            if points:
                x, y = np.median(points, axis=0)
                origins.append((round(x), round(y)))
        return origins

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
    def has_barcode_error(self) -> bool:
        """Check if any image has a barcode mismatch."""
        return any(img.barcode_mismatch for img in self.images)

    @property
    def barcode_error_count(self) -> int:
        """Count number of images with barcode mismatches."""
        return sum(1 for img in self.images if img.barcode_mismatch)

    @property
    def has_barcode_warning(self) -> bool:
        """Check if any image has 'no barcode' warning."""
        return any(img.barcode_not_found for img in self.images)

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
            return bool(np.any(self.working_mask > 0))

        # Compare working_mask with user_mask
        return not np.array_equal(self.working_mask, self.user_mask)
