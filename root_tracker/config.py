"""
Configuration management for Root Tracker.

Uses Python dataclasses for type-safe configuration with validation.
"""

from dataclasses import dataclass, field, asdict, fields
from hashlib import md5
from pathlib import Path
from typing import Optional
import yaml
import math


@dataclass
class DataConfig:
    """Configuration for data paths and file patterns."""
    filename_template: str = "{group}.{date}"
    date_format: str = "%y-%m-%d"
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
class GuiConfig:
    """Workflow defaults saved alongside processing presets."""
    show_max_root_depth: bool = True
    auto_apply: bool = True
    load_crop_editing: bool = True
    preprocess_crop_editing: bool = False


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
    gui: GuiConfig = field(default_factory=GuiConfig)
    
    # Internal state
    _base_path: Optional[Path] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        for section in (self.data, self.registration, self.crop, self.gui):
            for setting in fields(section):
                if isinstance(setting.default, bool) and type(getattr(section, setting.name)) is not bool:
                    raise ValueError(f'{setting.name} must be true or false')
        if type(self.n_clusters) is not int:
            raise ValueError('Origin counts must be an integer')
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
                
        if not 0 <= self.crop.top_ratio < self.crop.bottom_ratio <= 1:
            raise ValueError('Crop ratios must satisfy 0 ≤ top < bottom ≤ 1')
        if self.margin_top + self.margin_bottom >= 1 or self.margin_left + self.margin_right >= 1:
            raise ValueError('Opposite crop margins must leave some image visible')
        if not 0 <= self.registration.margin_ratio <= 1:
            raise ValueError('Registration margin must be between 0 and 1')
        for section, lower_name, upper_name in ((self.green, 'hsv_lower', 'hsv_upper'),
                                               (self.crop, 'blue_hsv_lower', 'blue_hsv_upper')):
            for name in (lower_name, upper_name):
                value = getattr(section, name)
                if len(value) != 3 or any(type(v) is not int or not 0 <= v <= limit for v, limit in zip(value, (179, 255, 255))):
                    raise ValueError('HSV limits require hue 0–179 and saturation/value 0–255')
            if any(a > b for a, b in zip(getattr(section, lower_name)[1:], getattr(section, upper_name)[1:])):
                raise ValueError('Lower saturation/value must not exceed the upper limit')
        for value in (self.green.min_count, self.green.min_area, self.threshold.min_contour_area, self.threshold.min_contour_length):
            if type(value) is not int or value < 0:
                raise ValueError('Contour limits must be non-negative integers')
        if not 0 <= self.threshold.low <= self.threshold.high <= 255:
            raise ValueError('Root thresholds must satisfy 0 ≤ low ≤ high ≤ 255')
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
        
        config = cls.from_dict(raw)
        config._base_path = path.parent.parent  # Go up from configs/ to project root
        config.data.resolve_paths(config._base_path)
        
        return config
    
    @classmethod
    def from_dict(cls, raw):
        """Read the same schema used by YAML without mutating the input."""
        if not isinstance(raw, dict):
            raise ValueError('Configuration must contain named settings')
        values = dict(raw)
        # Migrate the original processing option to a preview preference.
        threshold = dict(values.get('threshold', {}))
        if 'show_max_root_depth' in threshold:
            gui = dict(values.get('gui', {}))
            gui.setdefault('show_max_root_depth', threshold.pop('show_max_root_depth'))
            values.update(threshold=threshold, gui=gui)
        nested = {}
        for name, kind in (('data', DataConfig), ('green', GreenConfig),
                           ('threshold', ThresholdConfig), ('registration', RegistrationConfig),
                           ('crop', CropConfig), ('gui', GuiConfig)):
            nested[name] = kind(**values.pop(name, {}))
        return cls(**nested, **values)

    def to_dict(self):
        """Return portable, safe-YAML-compatible configuration values."""
        values = asdict(self)
        values.pop('_base_path', None)
        def plain(value):
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value
        return plain(values)

    def to_yaml(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding='utf-8')

    def processing_settings(self) -> dict:
        """Copy the processing parameters without dataset paths or GUI preferences."""
        return {key: value for key, value in self.to_dict().items()
                if key not in ('data', 'gui')}

    def for_series(self, series) -> "Config":
        """Combine global data settings with a group's committed parameters."""
        if series.processing_settings is None:
            return self
        values = self.to_dict()
        values.update(series.processing_settings)
        config = Config.from_dict(values)
        config._base_path = self.base_path
        return config

    @property
    def base_path(self) -> Path:
        """Get the base path for the project."""
        if self._base_path is None:
            return Path.cwd()
        return self._base_path

    def preprocess_config_hash(self) -> str:
        """Hash of config values that affect preprocessing."""
        # YAML/JSON turn tuples into lists; a GUI spinbox may turn 180 into
        # 180.0. Equivalent settings must retain the same cache identity.
        values = (
            "plate-search-fixed-analysis-v2",
            self.data.filename_template, self.data.date_format,
            float(self.rotation), self.n_clusters,
            self.margin_top, self.margin_bottom, self.margin_left, self.margin_right,
            self.green.min_count, self.green.min_area,
            list(self.green.hsv_lower), list(self.green.hsv_upper),
            self.crop.top_ratio, self.crop.bottom_ratio,
            list(self.crop.blue_hsv_lower), list(self.crop.blue_hsv_upper),
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
            "foreground-brightness-v2",
            "plant-connected-links-v1",
            "directional-shared-junctions-v1",
            "root-depth-overlay-v1",
            self.data.filename_template, self.data.date_format,
            self.threshold.low, self.threshold.high,
            self.threshold.min_contour_area, self.threshold.min_contour_length,
            self.n_clusters,
            self.margin_top, self.margin_bottom, self.margin_left, self.margin_right,
        )
        return md5(str(values).encode()).hexdigest()
