"""
Configuration management for Root Tracker.

Uses Python dataclasses for type-safe configuration with validation.
"""

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Optional
import yaml
import math


@dataclass
class DataConfig:
    """Configuration for data paths and file patterns."""
    filename_template: str = "{group}.{date}"
    date_format: str = "%d-%m-%y"
    input: str = "data_in_vitro/"
    output: str = "result/"
    statistics: str = "statistics_in_vitro.csv"
    detect_barcodes: bool = True  # Enable/disable barcode detection
    cache_compressed: bool = False  # Faster interactive saves; True saves disk space

    def resolve_paths(self, base_path: Path) -> None:
        """Resolve relative paths to absolute paths based on base_path."""
        self.input = str(base_path / self.input)
        self.output = str(base_path / self.output)
        self.statistics = str(base_path / self.statistics)


@dataclass
class GreenConfig:
    """Configuration for green area detection."""
    min_count: int = 5
    min_area: int = 100
    # HSV range for green detection
    hsv_lower: tuple[int, int, int] = (28, 166, 50)
    hsv_upper: tuple[int, int, int] = (42, 255, 200)


@dataclass
class ThresholdConfig:
    """Configuration for root thresholding."""
    low: int = 20
    high: int = 30
    min_contour_area: int = 100
    min_contour_length: int = 15


@dataclass
class RegistrationConfig:
    """Configuration for image registration."""
    enabled: bool = True
    margin_ratio: float = 0.25  # Search border on each side, relative to image dimensions


@dataclass
class CropConfig:
    """Configuration for image cropping."""
    background_enabled: bool = True
    background_region: str = "largest"  # "largest" or "all" matching patches
    # HSV range for blue background detection
    blue_hsv_lower: tuple[int, int, int] = (70, 0, 0)
    blue_hsv_upper: tuple[int, int, int] = (140, 255, 255)
    # Vertical crop ratios (relative to cropped height)
    top_ratio: float = 0.0
    bottom_ratio: float = 1.0


@dataclass
class Config:
    """
    Main configuration class for Root Tracker.
    
    Provides type-safe access to all configuration values with sensible defaults.
    Can be loaded from a YAML file or constructed programmatically.
    
    Example:
        # Load from YAML
        config = Config.from_yaml("configs/in_vitro.yaml")
        
        # Or create programmatically
        config = Config(n_clusters=6, rotation=180)
    """
    # General settings
    # Margins for cropping sides (0-1)
    margin_top: float = 0.0
    margin_bottom: float = 0.03
    margin_left: float = 0.03
    margin_right: float = 0.03
    
    rotation: float = 180  # Legacy automatic orientation, clockwise degrees
    # Load ROI bounds plate search; preprocess ROI is fixed in plate coordinates.
    # None searches the whole image / uses the configured fixed plate margins.
    load_roi: tuple[float, float, float, float, float] | None = None
    preprocess_roi: tuple[float, float, float, float, float] | None = None
    n_clusters: int = 6  # Number of plants per image
    
    # Nested configs
    data: DataConfig = field(default_factory=DataConfig)
    green: GreenConfig = field(default_factory=GreenConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    
    # Internal state
    _base_path: Optional[Path] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if self.crop.background_region not in ("largest", "all"):
            raise ValueError("background_region must be largest or all")
        if not math.isfinite(self.rotation):
            raise ValueError("rotation must be finite")
        from .preprocessing.roi import validate
        self.load_roi = validate(self.load_roi)
        self.preprocess_roi = validate(self.preprocess_roi)
        
        for name, value in [
            ("margin_top", self.margin_top),
            ("margin_bottom", self.margin_bottom),
            ("margin_left", self.margin_left),
            ("margin_right", self.margin_right)
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
                
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Config instance with values from the file.
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        
        # Parse nested configs
        data_config = DataConfig(**raw.pop("data", {}))
        green_config = GreenConfig(**raw.pop("green", {}))
        threshold_config = ThresholdConfig(**raw.pop("threshold", {}))
        registration_config = RegistrationConfig(**raw.pop("registration", {}))
        crop_config = CropConfig(**raw.pop("crop", {}))
        
        config = cls(
            data=data_config,
            green=green_config,
            threshold=threshold_config,
            registration=registration_config,
            crop=crop_config,
            **raw
        )
        config._base_path = path.parent.parent  # Go up from configs/ to project root
        config.data.resolve_paths(config._base_path)
        
        return config
    
    @property
    def base_path(self) -> Path:
        """Get the base path for the project."""
        if self._base_path is None:
            return Path.cwd()
        return self._base_path

    def preprocess_config_hash(self) -> str:
        """Hash of config values that affect preprocessing."""
        values = (
            "plate-search-fixed-analysis-v2",
            self.rotation, self.n_clusters,
            self.margin_top, self.margin_bottom, self.margin_left, self.margin_right,
            self.green.min_count, self.green.min_area,
            self.green.hsv_lower, self.green.hsv_upper,
            self.crop.top_ratio, self.crop.bottom_ratio,
            self.crop.blue_hsv_lower, self.crop.blue_hsv_upper,
            self.crop.background_enabled, self.crop.background_region,
            self.registration.enabled, self.registration.margin_ratio,
        )
        # Older versions ignored non-default margins. Do not reuse those
        # cached alignments now that this setting controls the search border.
        if self.registration.enabled and self.registration.margin_ratio != 0.25:
            values += ("configurable-registration-border-v1",)
        if self.load_roi is not None or self.preprocess_roi is not None:
            values += ("rotated-crops-v1", self.load_roi, self.preprocess_roi)
        return md5(str(values).encode()).hexdigest()

    def tracking_config_hash(self) -> str:
        """Hash of config values that affect tracking."""
        values = (
            self.threshold.low, self.threshold.high,
            self.threshold.min_contour_area, self.threshold.min_contour_length,
            self.n_clusters,
            self.margin_top, self.margin_bottom, self.margin_left, self.margin_right,
        )
        return md5(str(values).encode()).hexdigest()
