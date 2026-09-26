"""
Main pipeline orchestration for Root Tracker.

Provides step-by-step pipeline execution suitable for GUI integration.
"""

import os
import math
import numpy as np
import cv2
import pandas as pd
from multiprocessing import freeze_support, current_process
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections.abc import Callable, Iterator
from tqdm.contrib.concurrent import process_map

from .config import Config
from .models import ImageData, ImageSeries, PlantStatistics
from .preprocessing import ImageCropper, GreenAreaDetector, BackgroundRemover
from .registration import ImageRegistrator
from .tracking import RootThresholder, RootSkeletonizer, CornerDetector, RootLinker
from .analysis import StatisticsCalculator
from .io import ImageLoader, ResultExporter, BarcodeReader, series_cache


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
        
        # Try to load cached state for each series
        for series in self._series.values():
            series_cache.load_series_state(series, self.config)
            
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
            # Read barcode with bounding box (use fast method)
            barcode_text, rect = self.barcode_reader.read_file(image_data.path)
            
            image_data.barcode_read = barcode_text
            image_data.barcode_rect = rect
            image_data.barcode_detected = True  # Mark as detected (even if no barcode found)
            
            # Check for mismatch (if barcode was detected)
            if barcode_text:
                # Compare with expected barcode (case-insensitive)
                image_data.barcode_mismatch = barcode_text.lower() != image_data.barcode.lower()
                image_data.barcode_not_found = False
            else:
                image_data.barcode_mismatch = False
                image_data.barcode_not_found = True
            
            return image_data.barcode_mismatch
            
        except Exception:
            image_data.barcode_read = ""
            image_data.barcode_rect = None
            image_data.barcode_mismatch = False
            image_data.barcode_not_found = True
            image_data.barcode_detected = True  # Mark as detected even on failure
            return False
    
    def iter_detect_barcodes(
        self, images: list[ImageData], *,
        cancelled: Callable[[], bool] | None = None,
        max_workers: int = 4,
    ) -> Iterator[ImageData]:
        """Yield checked images as they finish, using at most four native readers.

        JPEG decoding and ZBar release the GIL. Each task owns its image and
        decoder; no Qt or cache writes run in these threads. Keep only one task
        per worker in flight so cancellation does not leave a dataset queued.
        Already checked images are yielded without reading them again. Closing
        the iterator waits for active reads, preventing mutations after cleanup.
        Process-pool pipeline workers retain their serial barcode loop to avoid
        nested parallelism.
        """
        if max_workers < 1:
            raise ValueError('max_workers must be positive')
        is_cancelled = cancelled or (lambda: False)
        pending = []
        for image in images:
            if is_cancelled():
                return
            if image.barcode_detected:
                yield image
            else:
                pending.append(image)
        if not pending or is_cancelled():
            return
        workers = min(max_workers, 4, os.cpu_count() or 1, len(pending))
        remaining = iter(pending)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='barcode') as executor:
            futures = {}
            try:
                for _ in range(workers):
                    if is_cancelled():
                        return
                    image = next(remaining)
                    futures[executor.submit(self.detect_barcode_in_image, image)] = image
                while futures and not is_cancelled():
                    done, _ = wait(futures, timeout=0.05, return_when=FIRST_COMPLETED)
                    for future in done:
                        if is_cancelled():
                            return
                        image = futures.pop(future)
                        future.result()
                        yield image
                        if is_cancelled():
                            return
                        following = next(remaining, None)
                        if following is not None:
                            futures[executor.submit(self.detect_barcode_in_image, following)] = following
            finally:
                for future in futures:
                    future.cancel()

    def preprocess_image(self, image_data: ImageData, *, original: np.ndarray | None = None) -> None:
        """
        Preprocess a single image.
        
        Applies cropping, rotation, plant detection, and background removal.
        Modifies the ImageData object in place.
        
        Args:
            image_data: The image to preprocess.
            original: Optional predecoded BGR photograph (never modified).
        """
        # Load image unless the series reader has prefetched it.
        if original is None:
            original = cv2.imread(image_data.path)
        
        cropped = self.cropper.process(original)

        # Detect green areas (plant stems)
        green_contours, _ = self.green_detector.find_green_contours(cropped)
        
        # Mask out green areas and adjust positions
        # Masking returns its own buffer, so retain the unmasked crop as a view
        # until the final display crop rather than copying the full photograph.
        origo = cropped
        cropped = self.green_detector.mask_green_in_image(cropped, green_contours)
        
        # Get plant positions
        try:
            pos_x, pos_y, areas = self.green_detector.cluster_plant_positions(green_contours)
        except Exception:
            # If clustering fails, return early
            return
        
        from .preprocessing.roi import extract, transform_points
        analysis_box = self.cropper.analysis_roi(cropped.shape)
        cropped, matrix = extract(cropped, analysis_box)
        image_data.image, _ = extract(origo, analysis_box)
        image_data.plate_transform = matrix.ravel().tolist()
        positions = transform_points(list(zip(pos_x, pos_y)), matrix)
        image_data.positions_x = [int(round(p[0])) for p in positions]
        image_data.positions_y = [int(round(p[1])) for p in positions]
        image_data.green_areas = areas
        image_data.process = self.background_remover.remove_gradient(cropped)
        image_data.canny = self.background_remover.compute_canny_edges(image_data.process)

    def preprocess_series(self, series: ImageSeries, progress_callback: callable = None) -> None:
        """
        Preprocess all images in a series.
        
        Args:
            series: ImageSeries to preprocess.
            progress_callback: Optional callback(current, total) for progress.
        """
        total = len(series.images)
        if total < 2 or current_process().name != 'MainProcess':
            for i, image_data in enumerate(series.images):
                self.preprocess_image(image_data)
                if progress_callback:
                    progress_callback(i + 1, total)
            return

        # Decode one photograph ahead while filtering the current image. Keep
        # CV and KMeans in chronological order (including their random state).
        # Batch child processes already overlap reads and do not add a pool.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='image-decode') as reader:
            pending = reader.submit(cv2.imread, series.images[0].path)
            for i, image_data in enumerate(series.images):
                original = pending.result()
                if i + 1 < total:
                    pending = reader.submit(cv2.imread, series.images[i + 1].path)
                self.preprocess_image(image_data, original=original)
                del original
                if progress_callback:
                    progress_callback(i + 1, total)

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

            # Create annotated copy — never mutate image_data.image
            annotated = image_data.image.copy()

            # Draw plant markers
            for i, (x, y) in enumerate(zip(pos_x_median, pos_y_median)):
                color = self.linker.get_color(i)
                cv2.circle(annotated, (x, y), 10, color, 4)
                cv2.putText(annotated, str(i + 1), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            
            # Threshold to get root mask. Margins are already cropped out during
            # preprocessing, so there is no edge border left to clear here.
            thresh = self.thresholder.threshold(image_data.process)

            # Apply user mask if present (series-level mask)
            thresh = self.thresholder.apply_user_mask(thresh, series.user_mask)

            # Handle new growth from difference (margins already cropped out)
            if image_data.diff is not None:
                new_area, new_parts = self.thresholder.compute_new_growth(
                    image_data.diff, (0, 0, 0, 0)
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
            endpoints = set(self.skeletonizer.find_endpoints(skeleton))
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
                    
                    if lower_info['point'] not in endpoints:
                        lower_corners.append(lower_info)
            
            # Retrieve previous colored samples for consistency
            previous_colored_samples = None
            if idx > 0:
                prev_image = series.images[idx - 1]
                if hasattr(prev_image, 'colored_samples'):
                    previous_colored_samples = prev_image.colored_samples
            
            # Link segments and assign to plants
            pairs, colored, colored_samples = self.linker.link_corners(
                upper_corners, lower_corners, pos_x_median, pos_y_median,
                previous_colored_samples=previous_colored_samples
            )
            
            image_data.colored_samples = colored_samples
            
            # Draw links on image
            for upper, lower in pairs:
                cv2.line(annotated, upper, lower, (255, 255, 255), 1)
            
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
                cv2.line(annotated, top, (top[0] + 100, top[1]), (255, 255, 255), 1)
                cv2.line(annotated, (top[0] + 100, top[1]), (top[0] + 100, bottom[1]), (255, 255, 255), 3)
                cv2.line(annotated, (top[0] + 100, bottom[1]), bottom, (255, 255, 255), 1)
                
                # Trace longest path (backtracking from bottom)
                mask_longest = np.zeros_like(skeleton_split)
                conts = []
                last_len = None
                visited = set()  # Prevent cycles in segment graph

                # We need to trace back from the bottom point to the top
                current_bottom = bottom

                while last_len != len(conts):
                    last_len = len(conts)
                    for uc_idx, uc in enumerate(upper_corners):
                        # Find the segment that ends at current_bottom
                        if 'lower_point' in uc and uc['lower_point'] == current_bottom and uc_idx not in visited:
                            visited.add(uc_idx)
                            # Found the segment, add its contour
                            if uc['contour_index'] < len(segment_contours):
                                conts.append(segment_contours[uc['contour_index']])

                            # Now find the pair that connects to the top of this segment
                            for p1, p2 in pairs:
                                if p1 == uc['point']:
                                    current_bottom = p2
                                    break
                            break

                cv2.drawContours(mask_longest, conts, -1, 255, cv2.FILLED)
                longest_length = cv2.countNonZero(mask_longest)
                image_data.longest.append(longest_length)
                
                # Color the roots
                color = self.linker.get_color(k)
                bright_color = tuple(min(c + 170, 255) for c in color)
                
                # Draw all roots normal thickness
                cv2.drawContours(annotated, plant_contours, -1, color, 3)
                
                # Draw main root thicker and brighter
                cv2.drawContours(annotated, conts, -1, bright_color, 12)
                
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
            
            # Store annotated image
            image_data.image_annotated = annotated

            # Save annotated image
            self.exporter.save_image(image_data.image_annotated, os.path.basename(image_data.path))
            
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
    
    def is_tracking_current(self, series: ImageSeries) -> bool:
        """Check if tracking results are still valid for the current config."""
        state = series.pipeline_state
        return (
            state.tracked
            and state.tracking_config_hash == self.config.tracking_config_hash()
            and state.preprocess_config_hash == self.config.preprocess_config_hash()
        )

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

    def export_statistics(self, series_dict: dict[str, ImageSeries]) -> pd.DataFrame:
        """
        Collect processing results from all series and return as DataFrame.
        
        Args:
            series_dict: Dictionary of ImageSeries to export.
            
        Returns:
            pandas DataFrame containing all statistics.
        """
        all_stats = []
        for series in series_dict.values():
            state = series.pipeline_state
            if state.tracked and state.last_statistics:
                all_stats.extend(state.last_statistics)
            else:
                # If not tracked yet, track it now (synchronously)
                stats = self.track_and_analyze_series(series)
                all_stats.extend(stats)
        
        # Convert to records
        records = [
            stat.to_dict() if hasattr(stat, 'to_dict') else stat
            for stat in all_stats
        ]
        
        if not records:
            return pd.DataFrame()
            
        return pd.DataFrame.from_records(records)
    
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


def preprocess_and_cache_worker(args: tuple) -> str:
    """Preprocess a series and save to disk cache. Pickle-friendly for multiprocessing.

    Args:
        args: Tuple of (config, series, optional queue).

    Returns:
        The series group name.
    """
    if len(args) == 3:
        config, series, queue = args
    else:
        config, series = args
        queue = None

    pipeline = RootTrackingPipeline(config)
    
    # Callback to put progress into queue
    def on_progress(current, total):
        if queue:
            # We report 1 unit of progress per image completed to the global counter
            queue.put(1)

    # Detect barcodes if not already done
    for img in series.images:
        if not img.barcode_detected:
            pipeline.detect_barcode_in_image(img)
            
    pipeline.preprocess_series(series, progress_callback=on_progress)
    pipeline.register_series(series)
    series.pipeline_state.preprocessed = True
    series.pipeline_state.preprocess_config_hash = config.preprocess_config_hash()
    series.pipeline_state.invalidate_from('track')
    series_cache.save_series(series, config)
    return series.group


def track_and_cache_worker(args: tuple) -> tuple:
    """Track a series and save to disk cache. Pickle-friendly for multiprocessing.

    Handles preprocessing if needed (not yet done or arrays freed).

    Args:
        args: Tuple of (config, series, optional queue).

    Returns:
        Tuple of (group_name, stats_dicts, state_dict) with lightweight data only.
    """
    if len(args) == 3:
        config, series, queue = args
    else:
        config, series = args
        queue = None

    pipeline = RootTrackingPipeline(config)

    # Callback to put progress into queue
    def on_progress(current, total):
        if queue:
            # We report 1 unit of progress per image completed to the global counter
            queue.put(1)

    # Detect barcodes if not already done
    for img in series.images:
        if not img.barcode_detected:
            pipeline.detect_barcode_in_image(img)

    preprocess_hash = config.preprocess_config_hash()
    state = series.pipeline_state

    # Ensure preprocessed
    if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
        pipeline.preprocess_series(series)
        pipeline.register_series(series)
        state.preprocessed = True
        state.preprocess_config_hash = preprocess_hash
        state.invalidate_from('track')
    elif series.images and series.images[0].image is None:
        # Preprocessed but arrays freed — reload from cache
        series_cache.load_series(series, config)

    stats = pipeline.track_and_analyze_series(series, progress_callback=on_progress)

    state.tracked = True
    state.tracking_config_hash = config.tracking_config_hash()
    state.last_statistics = stats

    series_cache.save_series(series, config)

    # Return lightweight results (no numpy arrays)
    stats_dicts = [s.to_dict() if hasattr(s, 'to_dict') else s for s in stats]
    state_dict = {
        'preprocessed': state.preprocessed,
        'preprocess_config_hash': state.preprocess_config_hash,
        'tracked': state.tracked,
        'tracking_config_hash': state.tracking_config_hash,
    }
    return series.group, stats_dicts, state_dict
