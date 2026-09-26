"""
Result exporting to CSV and images.

Handles saving analysis results and annotated images.
"""

import os
import cv2
import numpy as np
import pandas as pd

from ..config import Config
from ..models import PlantStatistics


class ResultExporter:
    """
    Exports analysis results to files.
    
    Handles:
    - Saving statistics to CSV
    - Saving annotated images
    - Creating summary reports
    
    Args:
        config: Configuration object with output paths.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.config.data.output, exist_ok=True)
    
    def save_image(self, image: np.ndarray, filename: str) -> str:
        """
        Save an image to the output directory.
        
        Args:
            image: Image array to save.
            filename: Filename (not full path).
            
        Returns:
            Full path to saved image.
        """
        output_path = os.path.join(self.config.data.output, filename)
        cv2.imwrite(output_path, image)
        return output_path
    
    def export_statistics(
        self, 
        statistics: list[PlantStatistics | dict]
    ) -> str:
        """
        Export statistics to CSV file.
        
        Args:
            statistics: List of PlantStatistics objects or dicts.
            
        Returns:
            Path to the saved CSV file.
        """
        # Convert to dicts if needed
        records = [
            stat.to_dict() if isinstance(stat, PlantStatistics) else stat
            for stat in statistics
        ]
        
        df = pd.DataFrame.from_records(records)
        df.to_csv(self.config.data.statistics, index=False)
        
        return self.config.data.statistics
    
    def generate_summary_report(
        self, 
        statistics: list[PlantStatistics | dict]
    ) -> dict:
        """
        Generate a summary report from statistics.
        
        Args:
            statistics: List of statistics.
            
        Returns:
            Summary dictionary with aggregate metrics.
        """
        if not statistics:
            return {}
        
        # Convert to dicts if needed
        records = [
            stat.to_dict() if isinstance(stat, PlantStatistics) else stat
            for stat in statistics
        ]
        
        df = pd.DataFrame.from_records(records)
        
        has_rsml = "root_source" in df and (df["root_source"] == "rsml").any()
        plant_ids = sorted(df["plant_id"].unique()) if has_rsml else range(1, self.config.n_clusters + 1)
        summary = {
            "total_images": df["image_path"].nunique(),
            "total_plants": len(plant_ids),
            "date_range": (
                df["image_date"].min().isoformat() if not df.empty else None,
                df["image_date"].max().isoformat() if not df.empty else None,
            ),
        }
        
        # Per-plant summary
        for plant_id in plant_ids:
            plant_df = df[df["plant_id"] == plant_id]
            if not plant_df.empty:
                summary[f"plant_{plant_id}"] = {
                    "final_total_length": plant_df["plant_total_length"].iloc[-1],
                    "final_main_root_depth": plant_df["plant_main_root_depth"].iloc[-1],
                    "avg_rgr": plant_df["plant_total_length_RGR"].mean(),
                }
        
        return summary
