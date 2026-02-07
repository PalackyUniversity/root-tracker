"""
Statistics calculation for root analysis.

Computes per-plant and per-image statistics including growth rates.
"""

import math
import numpy as np
import cv2
from datetime import datetime

from ..config import Config
from ..models import ImageData, PlantStatistics


class StatisticsCalculator:
    """
    Calculates statistics for root analysis.
    
    Computes various metrics for each plant and image, including
    relative growth rates (RGR) between time points.
    
    Args:
        config: Configuration object.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def compute_relative_growth_rate(
        self, 
        current_value: int, 
        previous_value: int, 
        days_elapsed: int
    ) -> float | None:
        """
        Compute the relative growth rate (RGR).
        
        RGR = (ln(current) - ln(previous)) / time_delta
        
        Args:
            current_value: Current measurement.
            previous_value: Previous measurement.
            days_elapsed: Number of days between measurements.
            
        Returns:
            RGR value, or None if calculation not possible.
        """
        if previous_value <= 0 or current_value <= 0 or days_elapsed <= 0:
            return None
        
        return (np.log(current_value) - np.log(previous_value)) / days_elapsed
    
    def compute_plant_statistics(
        self,
        image_data: ImageData,
        plant_id: int,
        plant_positions_x: list[int],
        plant_positions_y: list[int],
        plant_length: int,
        main_root_depth: int,
        main_root_length: int,
        root_count: int,
        previous_image: ImageData | None = None,
        previous_plant_length: int | None = None,
        previous_main_root_length: int | None = None
    ) -> PlantStatistics:
        """
        Compute statistics for a single plant.
        
        Args:
            image_data: The image data.
            plant_id: Plant index (0-based).
            plant_positions_x: X positions of all plants.
            plant_positions_y: Y positions of all plants.
            plant_length: Total root length for this plant.
            main_root_depth: Depth of main root.
            main_root_length: Length of main root.
            root_count: Number of root endpoints.
            previous_image: Previous time point image (for RGR).
            previous_plant_length: Previous plant length (for RGR).
            previous_main_root_length: Previous main root length (for RGR).
            
        Returns:
            PlantStatistics object.
        """
        # Calculate RGR if previous data available
        plant_length_rgr = None
        main_root_length_rgr = None
        area_change = None
        
        if previous_image is not None:
            days_elapsed = (image_data.date - previous_image.date).days
            
            if previous_plant_length is not None:
                plant_length_rgr = self.compute_relative_growth_rate(
                    plant_length, previous_plant_length, days_elapsed
                )
            
            if previous_main_root_length is not None:
                main_root_length_rgr = self.compute_relative_growth_rate(
                    main_root_length, previous_main_root_length, days_elapsed
                )
            
            if previous_image.total_area and image_data.total_area and image_data.new_area is not None:
                previous_area = previous_image.total_area
                current_area_adjusted = image_data.total_area - image_data.new_area
                if previous_area > 0:
                    area_change = abs(current_area_adjusted - previous_area) / previous_area * 100
        
        return PlantStatistics(
            image_date=image_data.date,
            image_barcode=image_data.barcode,
            image_barcode_read=image_data.barcode_read,
            image_path=image_data.path,
            image_total_area=image_data.total_area or 0,
            image_total_length=image_data.total_length or 0,
            image_new_area=image_data.new_area,
            image_new_parts=image_data.new_parts,
            image_area_change=area_change,
            plant_id=plant_id + 1,  # 1-indexed for output
            plant_center_x=plant_positions_x[plant_id],
            plant_center_y=plant_positions_y[plant_id],
            plant_green_area=image_data.green_areas[plant_id] if plant_id < len(image_data.green_areas) else 0,
            plant_root_count=root_count,
            plant_total_length=plant_length,
            plant_total_length_rgr=plant_length_rgr,
            plant_main_root_depth=main_root_depth,
            plant_main_root_length=main_root_length,
            plant_main_root_length_rgr=main_root_length_rgr,
        )
    
    def validate_monotonic_growth(
        self, 
        statistics: list[dict], 
        key: str
    ) -> list[str]:
        """
        Validate that a metric increases monotonically over time.
        
        Logs warnings for any decreases (which may indicate tracking errors).
        
        Args:
            statistics: List of statistics dictionaries.
            key: The key to validate.
            
        Returns:
            List of warning messages.
        """
        warnings = []
        
        for plant_id in range(1, self.config.n_clusters + 1):
            prev_value = None
            
            for stat in statistics:
                if stat.get("plant_id") != plant_id:
                    continue
                
                current_value = stat.get(key)
                if current_value is None:
                    continue
                
                if prev_value is not None and current_value < prev_value:
                    warnings.append(
                        f"For image '{stat['image_path']}' plant {plant_id}: "
                        f"'{key}' decreased from {prev_value} to {current_value}"
                    )
                
                prev_value = current_value
        
        return warnings
