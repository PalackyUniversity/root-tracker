"""
Main pipeline orchestration for Root Tracker.

Provides step-by-step pipeline execution suitable for GUI integration.
"""

import os
import math
import numpy as np
import cv2
from multiprocessing import freeze_support
from tqdm.contrib.concurrent import process_map

from .config import Config
from .models import ImageData, ImageSeries, PlantStatistics
from .preprocessing import ImageCropper, GreenAreaDetector, BackgroundRemover
from .registration import ImageRegistrator
from .tracking import RootThresholder, RootSkeletonizer, CornerDetector, RootLinker
from .analysis import StatisticsCalculator
from .io import ImageLoader, ResultExporter, BarcodeReader


# Suppress numpy divide warnings (from original code)
np.seterr(divide='ignore')


class RootTrackingPipeline:
    """
    Main pipeline for root tracking and analysis.
    
    Orchestrates all processing steps and provides a step-by-step
    interface suitable for GUI integration.
    
    The pipeline supports:
    1. Loading images and creating series
    2. Preprocessing (crop, rotate, detect plants)
    3. Registration (align time series)
    4. Tracking (segment and link roots)
    5. Analysis (compute statistics)
    6. Export (save results)
    
    Each step can be run independently, allowing a GUI to:
    - Show intermediate results
    - Allow user corrections
    - Apply batch operations (e.g., ignore regions)
    
    Args:
        config: Configuration object.
    
    Example:
        config = Config.from_yaml("configs/in_vitro.yaml")
        pipeline = RootTrackingPipeline(config)
        
        # Run full pipeline
        pipeline.run()
        
        # Or step-by-step
        series = pipeline.load_images()
        for s in series.values():
            pipeline.preprocess_series(s)
            pipeline.register_series(s)
            stats = pipeline.track_and_analyze_series(s)
        pipeline.export_results(all_stats)
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        
        # Initialize components
        self.loader = ImageLoader(config)
        self.cropper = ImageCropper(config)
        self.green_detector = GreenAreaDetector(config)
        self.background_remover = BackgroundRemover(config)
        self.registrator = ImageRegistrator(config)
        self.thresholder = RootThresholder(config)
        self.skeletonizer = RootSkeletonizer(config)
        self.corner_detector = CornerDetector(config)
        self.linker = RootLinker(config)
        self.statistics_calc = StatisticsCalculator(config)
        self.exporter = ResultExporter(config)
        self.barcode_reader = BarcodeReader()
        
        # State
        self._series: dict[str, ImageSeries] = {}
        self._all_statistics: list[PlantStatistics] = []
    
    def load_images(self) -> dict[str, ImageSeries]:
        """
        Step 1: Load all images and create series.
        
        Returns:
            Dictionary mapping group names to ImageSeries.
        """
        self._series = self.loader.create_series()
        return self._series
    
    def detect_barcode_in_image(self, image_data: ImageData) -> bool:
        """
        Detect barcode in an image during load step.
        
        Reads barcode from the full image and compares with expected barcode.
        Sets barcode_read, barcode_rect, and barcode_mismatch fields.
        
        Args:
            image_data: The image to detect barcode in.
            
        Returns:
            True if there's a mismatch warning, False otherwise.
        """
        try:
            # Read barcode from original image (decoupled from preprocessing rotation)
            image = cv2.imread(image_data.path)
            if image is None:
                return False
            
            # Read barcode with bounding box (use fast method)
            barcode_text, rect = self.barcode_reader.read_fast(image)
            
            image_data.barcode_read = barcode_text
            image_data.barcode_rect = rect
            image_data.barcode_detected = True  # Mark as detected (even if no barcode found)
            
            # Check for mismatch (if barcode was detected)
            if barcode_text:
                # Compare with expected barcode (case-insensitive)
                image_data.barcode_mismatch = barcode_text.lower() != image_data.barcode.lower()
            else:
                image_data.barcode_mismatch = False
            
            return image_data.barcode_mismatch
            
        except Exception:
            image_data.barcode_read = ""
            image_data.barcode_rect = None
            image_data.barcode_mismatch = False
            image_data.barcode_detected = True  # Mark as detected even on failure
            return False
    
    def preprocess_image(self, image_data: ImageData) -> None:
        """
        Preprocess a single image.
        
        Applies cropping, rotation, plant detection, and background removal.
        Modifies the ImageData object in place.
        
        Args:
            image_data: The image to preprocess.
        """
        # Load image
        original = cv2.imread(image_data.path)
        
        # Rotate if needed
        if self.config.rotation:
            original = self.cropper.rotate(original)
        
        # Auto-crop to blue background
        cropped = self.cropper.auto_crop_to_blue_background(original)
        
        # Detect green areas (plant stems)
        green_contours, _ = self.green_detector.find_green_contours(cropped)
        
        # Find where to crop (below green areas)
        min_y = self.green_detector.find_crop_start(green_contours, cropped.shape[0])
        
        # Mask out green areas and adjust positions
        origo = cropped.copy()
        cropped = self.green_detector.mask_green_in_image(cropped, green_contours)
        
        # Get plant positions
        try:
            pos_x, pos_y, areas = self.green_detector.cluster_plant_positions(green_contours)
        except Exception:
            # If clustering fails, return early
            return
        
        # Store results (adjusted for crop)
        image_data.positions_x = pos_x
        image_data.positions_y = [y - min_y for y in pos_y]
        image_data.green_areas = areas
        
        # Crop to root region
        cropped = cropped[min_y:].copy()  # .copy() ensures contiguous array
        image_data.image = origo[min_y:].copy()
        
        # Remove background gradient
        image_data.process = self.background_remover.remove_gradient(cropped)
        image_data.canny = self.background_remover.compute_canny_edges(image_data.process)
        
        # Apply margins to the processed image so they are visible in UI
        # This will black out the edges based on config
        image_data.process = self.thresholder.apply_margins(image_data.process)
        image_data.image = self.thresholder.apply_margins(image_data.image)
    
    def preprocess_series(self, series: ImageSeries) -> None:
        """
        Preprocess all images in a series.
        
        Args:
            series: ImageSeries to preprocess.
        """
        for image_data in series.images:
            self.preprocess_image(image_data)
    
    def register_series(self, series: ImageSeries) -> None:
        """
        Step 2: Register (align) images in a series.
        
        Args:
            series: ImageSeries to register.
        """
        self.registrator.register_series(series.images)
    
    def track_and_analyze_series(
        self, 
        series: ImageSeries,
        progress_callback: callable = None
    ) -> list[PlantStatistics]:
        """
        Step 3: Run tracking algorithm and compute statistics.
        
        This is the main processing step that:
        1. Thresholds images to segment roots
        2. Skeletonizes roots
        3. Links root segments
        4. Assigns roots to plants
        5. Computes statistics
        
        Args:
            series: ImageSeries to analyze.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            List of PlantStatistics objects.
        """
        statistics = []
        
        # Compute median positions across series
        pos_x_median = []
        pos_y_median = []
        for i in range(self.config.n_clusters):
            x_values = [img.positions_x[i] for img in series.images if img.positions_x]
            y_values = [img.positions_y[i] for img in series.images if img.positions_y]
            if x_values and y_values:
                pos_x_median.append(round(np.median(x_values)))
                pos_y_median.append(round(np.median(y_values)))
        
        if not pos_x_median:
            return statistics
        
        total_images = len(series.images)
        for idx, image_data in enumerate(series.images):
            if image_data.process is None:
                continue
            
            # Draw plant markers
            for i, (x, y) in enumerate(zip(pos_x_median, pos_y_median)):
                color = self.linker.get_color(i)
                cv2.circle(image_data.image, (x, y), 10, color, 4)
                cv2.putText(image_data.image, str(i + 1), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            
            # Threshold to get root mask
            thresh = self.thresholder.threshold(image_data.process)
            thresh = self.thresholder.apply_margins(thresh)
            
            # Handle new growth from difference
            if image_data.diff is not None:
                h, w = thresh.shape
                margins = (
                    round(self.config.margin_top * h),
                    round(self.config.margin_bottom * h),
                    round(self.config.margin_left * w),
                    round(self.config.margin_right * w)
                )
                new_area, new_parts = self.thresholder.compute_new_growth(
                    image_data.diff, margins
                )
                image_data.new_area = new_area
                image_data.new_parts = new_parts
            else:
                image_data.new_area = None
                image_data.new_parts = None
            
            # Filter small contours
            thresh_filtered, contours = self.thresholder.filter_small_contours(thresh)
            image_data.total_area = cv2.countNonZero(thresh_filtered)
            
            # Skeletonize
            skeleton = self.skeletonizer.skeletonize_mask(thresh_filtered)
            image_data.total_length = self.skeletonizer.get_total_length(skeleton)
            
            # Mask above plants
            skeleton = self.skeletonizer.mask_above_plants(
                skeleton, pos_x_median, pos_y_median
            )
            
            # Find and remove intersections
            intersections = self.skeletonizer.find_intersections(skeleton)
            endpoints = self.skeletonizer.find_endpoints(skeleton)
            skeleton_split, segment_contours = self.skeletonizer.split_at_intersections(
                skeleton, intersections
            )
            
            # Analyze corners of each segment
            upper_corners = []
            lower_corners = []
            
            # Add plant positions as initial lower corners
            for i, (x, y) in enumerate(zip(pos_x_median, pos_y_median)):
                lower_corners.append({
                    'point': (x, y),
                    'angle': 90.0,
                    'plant_id': i
                })
            
            for cnt_n, cnt in enumerate(segment_contours):
                upper_info, lower_info = self.corner_detector.analyze_contour_corners(
                    cnt, skeleton_split
                )
                
                if upper_info and lower_info:
                    upper_info['contour_index'] = cnt_n
                    upper_info['contour'] = cnt
                    upper_info['lower_point'] = lower_info['point']
                    upper_corners.append(upper_info)
                    
                    if lower_info['point'] not in [e for e in endpoints]:
                        lower_corners.append(lower_info)
            
            # Link segments and assign to plants
            pairs, colored, colored_samples = self.linker.link_corners(
                upper_corners, lower_corners, pos_x_median, pos_y_median
            )
            
            image_data.colored_samples = colored_samples
            
            # Draw links on image
            for upper, lower in pairs:
                cv2.line(image_data.image, upper, lower, (255, 255, 255), 1)
            
            # Compute per-plant statistics
            image_data.plant_length = []
            image_data.longest = []
            
            for k in range(self.config.n_clusters):
                # Find roots for this plant
                colored_k = {ck: cv for ck, cv in colored.items() if cv == k}
                plant_contours = [
                    segment_contours[uc['contour_index']]
                    for uc in upper_corners
                    if 'lower_point' in uc and uc['lower_point'] in colored_k
                ]
                
                # Calculate plant total length
                mask_to_count = np.zeros_like(skeleton_split)
                cv2.drawContours(mask_to_count, plant_contours, -1, 255, cv2.FILLED)
                plant_length = cv2.countNonZero(mask_to_count)
                image_data.plant_length.append(plant_length)
                
                # Find main root depth
                if colored_k:
                    top = min(colored_k.keys(), key=lambda z: z[1])
                    bottom = max(colored_k.keys(), key=lambda z: z[1])
                    main_root_depth = bottom[1] - top[1]
                else:
                    main_root_depth = 0
                    top = (pos_x_median[k], pos_y_median[k])
                    bottom = (pos_x_median[k], pos_y_median[k])
                
                # Draw main root indicator
                cv2.line(image_data.image, top, (top[0] + 100, top[1]), (255, 255, 255), 1)
                cv2.line(image_data.image, (top[0] + 100, top[1]), (top[0] + 100, bottom[1]), (255, 255, 255), 3)
                cv2.line(image_data.image, (top[0] + 100, bottom[1]), bottom, (255, 255, 255), 1)
                
                # Trace longest path
                mask_longest = np.zeros_like(skeleton_split)
                cv2.drawContours(mask_longest, plant_contours[:1] if plant_contours else [], -1, 255, cv2.FILLED)
                longest_length = cv2.countNonZero(mask_longest)
                image_data.longest.append(longest_length)
                
                # Color the roots
                color = self.linker.get_color(k)
                bright_color = tuple(min(c + 170, 255) for c in color)
                cv2.drawContours(image_data.image, plant_contours, -1, color, 3)
                
                # Count root endpoints
                used_points = {p for p, _ in pairs}
                root_count = len([
                    uc for uc in upper_corners
                    if 'lower_point' in uc and uc['lower_point'] in colored_k
                    and uc['lower_point'] not in used_points
                ])
                
                # Create statistics record
                previous_image = series.images[idx - 1] if idx > 0 else None
                time_delta = (image_data.date - previous_image.date).days if previous_image else 1
                
                # Calculate RGR values
                plant_length_rgr = None
                main_root_length_rgr = None
                area_change = None
                
                if idx > 0 and previous_image:
                    prev_length = previous_image.plant_length[k] if k < len(previous_image.plant_length) else None
                    prev_longest = previous_image.longest[k] if k < len(previous_image.longest) else None
                    
                    if prev_length and prev_length > 0:
                        plant_length_rgr = (np.log(plant_length) - np.log(prev_length)) / time_delta
                    
                    if prev_longest and prev_longest > 0:
                        main_root_length_rgr = (np.log(longest_length) - np.log(prev_longest)) / time_delta
                    
                    if previous_image.total_area and image_data.total_area and image_data.new_area is not None:
                        current_adjusted = abs((image_data.total_area - image_data.new_area) - previous_image.total_area)
                        area_change = current_adjusted / previous_image.total_area * 100
                
                stats = PlantStatistics(
                    image_date=image_data.date,
                    image_barcode=image_data.barcode,
                    image_barcode_read=image_data.barcode_read,
                    image_path=image_data.path,
                    image_total_area=image_data.total_area or 0,
                    image_total_length=image_data.total_length or 0,
                    image_new_area=image_data.new_area,
                    image_new_parts=image_data.new_parts,
                    image_area_change=area_change,
                    plant_id=k + 1,
                    plant_center_x=pos_x_median[k],
                    plant_center_y=pos_y_median[k],
                    plant_green_area=image_data.green_areas[k] if k < len(image_data.green_areas) else 0,
                    plant_root_count=root_count,
                    plant_total_length=plant_length,
                    plant_total_length_rgr=plant_length_rgr,
                    plant_main_root_depth=main_root_depth,
                    plant_main_root_length=longest_length,
                    plant_main_root_length_rgr=main_root_length_rgr,
                )
                
                statistics.append(stats)
            
            # Save annotated image
            self.exporter.save_image(image_data.image, os.path.basename(image_data.path))
            
            # Update progress (after processing)
            if progress_callback:
                progress_callback(idx + 1, total_images)
        
        # Validate monotonic growth
        for key in ["image_total_area", "image_total_length", "plant_total_length", 
                    "plant_main_root_depth", "plant_main_root_length"]:
            warnings = self.statistics_calc.validate_monotonic_growth(statistics, key)
            for warning in warnings:
                print(warning)
        
        return statistics
    
    def export_results(self, statistics: list[PlantStatistics]) -> str:
        """
        Step 4: Export results to CSV.
        
        Args:
            statistics: List of statistics dictionaries.
            
        Returns:
            Path to the saved CSV file.
        """
        self._all_statistics = statistics
        return self.exporter.export_statistics(statistics)
    
    def process_series_wrapper(self, series: ImageSeries) -> list[PlantStatistics]:
        """
        Process a single series (for multiprocessing).
        
        Runs preprocessing, registration, and tracking.
        
        Args:
            series: ImageSeries to process.
            
        Returns:
            List of statistics dictionaries.
        """
        try:
            self.preprocess_series(series)
            self.register_series(series)
            return self.track_and_analyze_series(series)
        except Exception as e:
            print(f"Error processing {series.group}: {e}")
            return []
    
    def run(self, parallel: bool = True, max_workers: int | None = None) -> str:
        """
        Run the full pipeline.
        
        Args:
            parallel: If True, process series in parallel.
            max_workers: Maximum number of parallel workers.
            
        Returns:
            Path to the saved CSV file.
        """
        # Load images
        series_dict = self.load_images()
        series_list = list(series_dict.values())
        
        if parallel:
            freeze_support()
            if max_workers is None:
                max_workers = os.cpu_count()
            
            # Process in parallel
            all_stats = process_map(
                self._process_series_for_parallel,
                series_list,
                max_workers=max_workers
            )
        else:
            all_stats = []
            for series in series_list:
                stats = self.process_series_wrapper(series)
                all_stats.append(stats)
        
        # Flatten results
        flat_stats = [stat for sublist in all_stats for stat in sublist]
        
        # Export
        return self.export_results(flat_stats)
    
    def _process_series_for_parallel(self, series: ImageSeries) -> list[PlantStatistics]:
        """Process series for parallel execution (creates new pipeline instance)."""
        # Create a new pipeline instance for this process
        pipeline = RootTrackingPipeline(self.config)
        return pipeline.process_series_wrapper(series)


def _process_series_standalone(args: tuple) -> list[PlantStatistics]:
    """Standalone function for multiprocessing (pickle-friendly)."""
    config, series = args
    pipeline = RootTrackingPipeline(config)
    return pipeline.process_series_wrapper(series)
