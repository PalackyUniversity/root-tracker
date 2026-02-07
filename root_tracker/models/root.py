"""
Root data models for Root Tracker.

Provides dataclasses for representing roots, plant root systems, and statistics.
Designed with future RSML (Root System Markup Language) export in mind.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class Root:
    """
    Represents a single root segment.
    
    This class stores the geometry and properties of an individual root segment,
    which can later be assembled into full root systems.
    
    Note: The structure is designed to facilitate future RSML export,
    where roots are represented hierarchically with parent-child relationships.
    
    Attributes:
        id: Unique identifier for this root segment.
        plant_id: Which plant this root belongs to (0-indexed).
        contour: The contour points defining this root's shape.
        skeleton_points: Points along the skeleton of this root.
        upper_point: The upper endpoint (closer to plant).
        lower_point: The lower endpoint (away from plant).
        angle_upper: Direction angle at upper endpoint (degrees).
        angle_lower: Direction angle at lower endpoint (degrees).
        length: Length of this root segment in pixels.
        parent_id: ID of the parent root segment (for branching), None if primary.
    """
    id: int
    plant_id: int
    contour: np.ndarray = field(repr=False)
    skeleton_points: Optional[np.ndarray] = field(default=None, repr=False)
    upper_point: tuple[int, int] = (0, 0)
    lower_point: tuple[int, int] = (0, 0)
    angle_upper: float = 90.0
    angle_lower: float = 90.0
    length: int = 0
    parent_id: Optional[int] = None
    
    @property
    def is_primary(self) -> bool:
        """Check if this is a primary (main) root."""
        return self.parent_id is None
    
    @property
    def depth(self) -> int:
        """Calculate the vertical depth of this root segment."""
        return abs(self.lower_point[1] - self.upper_point[1])


@dataclass
class PlantRoots:
    """
    Collection of all roots belonging to a single plant.
    
    This class aggregates root segments for a plant and provides methods
    for analysis and future RSML export.
    
    Attributes:
        plant_id: Identifier for this plant (0-indexed within image).
        roots: List of Root objects belonging to this plant.
        centroid: (x, y) position of the plant stem/base.
        green_area: Area of the green (stem) region in pixels.
    """
    plant_id: int
    roots: list[Root] = field(default_factory=list)
    centroid: tuple[int, int] = (0, 0)
    green_area: int = 0
    
    def add_root(self, root: Root) -> None:
        """Add a root to this plant's collection."""
        self.roots.append(root)
    
    @property
    def total_length(self) -> int:
        """Calculate total length of all roots."""
        return sum(root.length for root in self.roots)
    
    @property
    def root_count(self) -> int:
        """Count the number of root tips (endpoints)."""
        return len(self.roots)
    
    @property
    def main_root_depth(self) -> int:
        """Calculate the depth of the deepest root point."""
        if not self.roots:
            return 0
        max_y = max(root.lower_point[1] for root in self.roots)
        min_y = min(root.upper_point[1] for root in self.roots)
        return max_y - min_y
    
    def get_primary_roots(self) -> list[Root]:
        """Get list of primary (main) roots."""
        return [root for root in self.roots if root.is_primary]
    
    def get_lateral_roots(self) -> list[Root]:
        """Get list of lateral (branching) roots."""
        return [root for root in self.roots if not root.is_primary]
    
    # Future RSML support methods (placeholders)
    def to_rsml_polylines(self) -> list[list[tuple[float, float]]]:
        """
        Convert roots to RSML-compatible polylines.
        
        TODO: Implement when adding RSML export support.
        """
        raise NotImplementedError("RSML export not yet implemented")


@dataclass
class PlantStatistics:
    """
    Statistics for a single plant at a single time point.
    
    This dataclass holds all computed metrics for a plant, used for
    CSV export and analysis.
    
    Attributes match the output columns expected in the statistics CSV.
    """
    # Image information
    image_date: datetime
    image_barcode: str
    image_barcode_read: str
    image_path: str
    image_total_area: int
    image_total_length: int
    image_new_area: Optional[int]
    image_new_parts: Optional[int]
    image_area_change: Optional[float]
    
    # Plant information
    plant_id: int  # 1-indexed for output
    plant_center_x: int
    plant_center_y: int
    plant_green_area: int
    plant_root_count: int
    plant_total_length: int
    plant_total_length_rgr: Optional[float]  # Relative Growth Rate
    plant_main_root_depth: int
    plant_main_root_length: int
    plant_main_root_length_rgr: Optional[float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "image_date": self.image_date,
            "image_barcode": self.image_barcode,
            "image_barcode_read": self.image_barcode_read,
            "image_path": self.image_path,
            "image_total_area": self.image_total_area,
            "image_total_length": self.image_total_length,
            "image_new_area": self.image_new_area,
            "image_new_parts": self.image_new_parts,
            "image_area_change": self.image_area_change,
            "plant_id": self.plant_id,
            "plant_center_x": self.plant_center_x,
            "plant_center_y": self.plant_center_y,
            "plant_green_area": self.plant_green_area,
            "plant_root_count": self.plant_root_count,
            "plant_total_length": self.plant_total_length,
            "plant_total_length_RGR": self.plant_total_length_rgr,
            "plant_main_root_depth": self.plant_main_root_depth,
            "plant_main_root_length": self.plant_main_root_length,
            "plant_main_root_length_RGR": self.plant_main_root_length_rgr,
        }
