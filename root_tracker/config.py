"""
Configuration management for Root Tracker.

Uses Python dataclasses for type-safe configuration with validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    """Configuration for data paths and file patterns."""
    filename_template: str = "{group}.{date}"
    date_format: str = "%d-%m-%y"
    input: str = "data_in_vitro/"
    output: str = "result/"
    statistics: str = "statistics_in_vitro.csv"

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
    margin_ratio: float = 0.25  # 1/4 of image dimensions


@dataclass
class CropConfig:
    """Configuration for image cropping."""
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
    
    rotation: int = 180  # Rotation in degrees (0, 90, 180, 270)
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
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError(f"rotation must be 0, 90, 180, or 270, got {self.rotation}")
        
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
