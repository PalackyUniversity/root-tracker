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
        self.rsml_output: str | None = None
    
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
            from .io.rsml_replacement import load_replacement, restore_measurements
            for image in series:
                load_replacement(image)
            restore_measurements(series)
            
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

    def redetect_centroids(self, series: ImageSeries) -> None:
        """Refresh plant centers without rebuilding processed pixels or alignment."""
        from .preprocessing.roi import extraction_geometry, transform_points
        state = series.pipeline_state
        if (any(len(image.plate_transform) != 6 for image in series.images) or
                (state.preprocess_config_hash and
                 state.preprocess_config_hash != self.config.preprocess_config_hash())):
            # Older caches lack the coordinate mapping needed to reuse alignment.
            series.clear_tracking_results()
            self.preprocess_series(series)
            self.register_series(series)
            return

        detected = []
        if not series.images:
            return
        # Overlap JPEG decoding with detection without changing clustering order.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='centroid-decode') as reader:
            pending = reader.submit(cv2.imread, series.images[0].path)
            for index, image in enumerate(series.images):
                original = pending.result()
                if index + 1 < len(series.images):
                    pending = reader.submit(cv2.imread, series.images[index + 1].path)
                if original is None:
                    raise ValueError(f"Cannot read image: {image.path}")
                plate = self.cropper.process(original)
                contours, _ = self.green_detector.find_green_contours(plate)
                x, y, areas = self.green_detector.cluster_plant_positions(contours)
                _, crop_matrix, _ = extraction_geometry(plate.shape, self.cropper.analysis_roi(plate.shape))
                registered = np.asarray(image.plate_transform).reshape(2, 3)
                offset = np.rint(registered[:, 2] - crop_matrix[:, 2]).astype(int)
                # Match preprocessing's rounding BEFORE the integer registration shift.
                positions = np.rint(transform_points(list(zip(x, y)), crop_matrix)).astype(int) + offset
                detected.append((positions[:, 0].tolist(), positions[:, 1].tolist(), areas))

        # Commit only after every image succeeded; preserve existing data on errors.
        series.clear_tracking_results()
        for image, (x, y, areas) in zip(series.images, detected):
            image.positions_x, image.positions_y, image.green_areas = x, y, areas

    def preprocess_series(self, series: ImageSeries, progress_callback: callable = None) -> None:
        """
        Preprocess all images in a series.
        
        Args:
            series: ImageSeries to preprocess.
            progress_callback: Optional callback(current, total) for progress.
        """
        images = [image for image in series if image.rsml_document is None]
        total = len(images)
        if total < 2 or current_process().name != 'MainProcess':
            for i, image_data in enumerate(images):
                self.preprocess_image(image_data)
                if progress_callback:
                    progress_callback(i + 1, total)
            return

        # Decode one photograph ahead while filtering the current image. Keep
        # CV and KMeans in chronological order (including their random state).
        # Batch child processes already overlap reads and do not add a pool.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='image-decode') as reader:
            pending = reader.submit(cv2.imread, images[0].path)
            for i, image_data in enumerate(images):
                original = pending.result()
                if i + 1 < total:
                    pending = reader.submit(cv2.imread, images[i + 1].path)
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
        self.registrator.register_series([image for image in series if image.rsml_document is None])
    
    def track_and_analyze_series(
        self, 
        series: ImageSeries,
        progress_callback: callable = None,
        *, image_callback: callable = None, save_images: bool = True,
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
            image_callback: Called as each image's tracking result is ready.
            save_images: Write annotations during tracking; interactive callers
                can defer these writes until all calculated images are published.

        Returns:
            List of PlantStatistics objects.
        """
        statistics = []
        has_mask = series.user_mask is not None and bool(np.any(series.user_mask))
        if has_mask and any(image.rsml_document is None
                            and (image.rsml_unmasked_samples is None or image.main_root_samples is None)
                            for image in series):
            # Bootstrap once for old caches or datasets first opened with a mask.
            # Subsequent mask strokes reuse this geometry; deletion stays cheap.
            from copy import copy
            unmasked = copy(series)
            unmasked.images = [copy(image) for image in series]
            unmasked.user_mask = unmasked.working_mask = None
            self.track_and_analyze_series(unmasked, save_images=False)
            for image, source in zip(series, unmasked):
                if image.rsml_document is None:
                    image.rsml_unmasked_samples = source.rsml_samples
                    image.main_root_samples = source.main_root_samples

        # Use the same group origins shown in Preprocess.
        origins = series.plant_origins(self.config.n_clusters)
        pos_x_median = [x for x, _ in origins]
        pos_y_median = [y for _, y in origins]
        
        # Invalidate only export geometry at the start of each tracking run.
        for image_data in series.images:
            image_data.rsml_samples = None
            image_data.root_depth_background = None
            if not has_mask and image_data.rsml_document is None:
                image_data.main_root_samples = np.empty((0, 3), dtype=np.int32)
                image_data.tracking_overlay = np.empty((0, 5), dtype=np.int32)
        has_replacements = any(image.rsml_document is not None for image in series)
        if not pos_x_median and not has_replacements:
            for image_data in series.images:
                if image_data.image is not None and image_data.process is not None:
                    image_data.rsml_samples = {}
            return statistics
        
        previous_main = {}
        total_images = len(series.images)
        for idx, image_data in enumerate(series.images):
            if image_data.rsml_document is not None:
                previous_main = {}  # A manual replacement starts a new authoritative geometry.
                from .io.rsml_replacement import render_replacement
                from .io.root_mask import apply_root_mask
                apply_root_mask(image_data, series.user_mask)
                from .analysis.rsml_statistics import apply_measurements, image_statistics
                apply_measurements(image_data)
                statistics.extend(image_statistics(image_data))
                image_data.image_annotated = render_replacement(image_data)
                if image_callback:
                    image_callback(image_data)
                if save_images:
                    self.exporter.save_image(image_data.image_annotated, os.path.basename(image_data.path))
                if progress_callback:
                    progress_callback(idx + 1, total_images)
                continue
            if not pos_x_median or image_data.process is None:
                continue

            # Create annotated copy — never mutate image_data.image
            annotated = image_data.image.copy()
            depth_mask = np.zeros(annotated.shape[:2], dtype=np.uint8)

            # Draw plant markers
            for i, (x, y) in enumerate(zip(pos_x_median, pos_y_median)):
                color = self.linker.get_color(i)
                cv2.circle(annotated, (x, y), 10, color, 4)
                cv2.putText(annotated, str(i + 1), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            
            # Threshold to get root mask. Margins are already cropped out during
            # preprocessing, so there is no edge border left to clear here.
            thresh = self.thresholder.threshold(image_data.process)
            unmasked_thresh = thresh

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
            
            # Preserve observed thin fragments supported by the preceding frame.
            previous_colored_samples = (series.images[idx - 1].colored_samples
                                        if idx > 0 else None)
            thresh_filtered, contours = self.thresholder.filter_small_contours(
                thresh, previous_colored_samples=previous_colored_samples)
            image_data.total_area = cv2.countNonZero(thresh_filtered)
            
            _, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                (unmasked_thresh > 0).astype(np.uint8))

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
            # Include tiny segments discarded by splitting: they still connect
            # nearby parts of a crossing. Label only actual skeleton pixels,
            # so dilation's empty padding cannot join disconnected roots.
            removed_skeleton = ((skeleton > 0) & (skeleton_split == 0)).astype(np.uint8)
            junction_labels = cv2.connectedComponents(removed_skeleton)[1]
            # Filtered terminal twigs still provide real exits at crossings.
            # Keep their directions for routing, without adding their pixels
            # back to measurements or rendered root segments.
            short_mask = ((removed_skeleton > 0) & (intersections == 0)).astype(np.uint8)
            terminal_exits = {}
            for short_contour in cv2.findContours(
                    short_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
                short_upper, short_lower = self.corner_detector.analyze_contour_corners(
                    short_contour, short_mask)
                if short_upper and short_lower:
                    upper_is_tip = short_upper['point'] in endpoints
                    lower_is_tip = short_lower['point'] in endpoints
                    if upper_is_tip == lower_is_tip:
                        continue
                    if upper_is_tip:
                        short_upper, short_lower = short_lower, short_upper
                    x, y = short_upper['point']
                    junction = int(junction_labels[y, x])
                    short_upper['lower_point'] = short_lower['point']
                    short_upper['junction_id'] = junction
                    terminal_exits.setdefault(junction, []).append(short_upper)
            
            # Measure foreground caliber before the skeleton loses width.
            root_width = 2 * cv2.distanceTransform(
                (thresh_filtered > 0).astype(np.uint8), cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE)
            from .tracking.temporal_fragments import contour_path

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
                    for corner in (upper_info, lower_info):
                        x, y = corner['point']
                        adjacent = np.unique(junction_labels[
                            max(0, y - 1):y + 2, max(0, x - 1):x + 2])
                        adjacent = adjacent[adjacent > 0]
                        if len(adjacent) == 1:
                            corner['junction_id'] = int(adjacent[0])
                    if upper_info.get('junction_id') in terminal_exits:
                        upper_info['terminal_exits'] = terminal_exits.pop(upper_info['junction_id'])
                    ux, uy = upper_info['point']
                    component = int(component_labels[uy, ux])
                    upper_info['component_id'] = component
                    width, height = component_stats[component, 2:4]
                    upper_info['component_extent'] = float(np.hypot(width - 1, height - 1))
                    upper_info['component_area'] = int(component_stats[component, cv2.CC_STAT_AREA])
                    upper_info['contour_index'] = cnt_n
                    upper_info['contour'] = cnt
                    upper_info['lower_point'] = lower_info['point']
                    path = contour_path(upper_info)
                    upper_info['width_profile'] = root_width[path[:, 1], path[:, 0]]
                    upper_corners.append(upper_info)
                    
                    if lower_info['point'] not in endpoints:
                        lower_corners.append(lower_info)
            
            # Link segments and assign to plants
            pairs, colored, colored_samples = self.linker.link_corners(
                upper_corners, lower_corners, pos_x_median, pos_y_median,
                previous_colored_samples=previous_colored_samples
            )
            # Temporal ownership boundaries may subdivide a current skeleton
            # segment. Use those same fragments for measurement and drawing.
            segment_contours = [upper['contour'] for upper in upper_corners]
            for contour_index, upper in enumerate(upper_corners):
                upper['contour_index'] = contour_index
            
            image_data.colored_samples = colored_samples
            image_data.rsml_samples = {
                int(k): np.asarray(sorted(v), dtype=np.int32).reshape(-1, 2)
                for k, v in colored_samples.items()
            }
            
            if not has_mask:
                image_data.rsml_unmasked_samples = image_data.rsml_samples

            # Draw links on image
            for upper, lower in pairs:
                cv2.line(annotated, upper, lower, (255, 255, 255), 1)
            
            # Preserve diagnostic markers separately from editable root strokes.
            oy, ox = np.nonzero(np.any(annotated != image_data.image, axis=2))
            image_data.tracking_overlay = np.column_stack((oy, ox, annotated[oy, ox])).astype(np.int32)

            # Compute per-plant statistics
            image_data.plant_length = []
            image_data.longest = []
            main_segments = {}
            current_main_samples = {}
            main_pixels = []
            from .tracking.junction_router import plant_ids
            
            for k in range(self.config.n_clusters):
                # Find roots for this plant
                colored_k = {ck: cv for ck, cv in colored.items() if k in plant_ids(cv)}
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
                
                from .tracking.main_root import has_main_axis, select_main_geometry
                bottom, main_geometry = select_main_geometry(
                    upper_corners, colored, pairs, k, previous_main.get(k, set()),
                    lower_corners,
                    (previous_colored_samples or {}).get(k))
                top = (min(colored_k, key=lambda z: z[1]) if colored_k
                       else (pos_x_median[k], pos_y_median[k]))
                bottom = bottom if bottom is not None else top
                main_root_depth = bottom[1] - top[1]

                # Keep depth markers separate until root annotations are complete.
                cv2.line(depth_mask, top, (top[0] + 100, top[1]), 255, 1)
                cv2.line(depth_mask, (top[0] + 100, top[1]), (top[0] + 100, bottom[1]), 255, 3)
                cv2.line(depth_mask, (top[0] + 100, bottom[1]), bottom, 255, 1)

                mask_longest = np.zeros_like(skeleton_split)
                conts = [contour for pieces in main_geometry.values() for contour in pieces]
                for i in main_geometry:
                    main_segments.setdefault(i, set()).add(k)

                cv2.drawContours(mask_longest, conts, -1, 255, cv2.FILLED)
                longest_length = cv2.countNonZero(mask_longest)
                image_data.longest.append(longest_length)
                ys, xs = np.nonzero(mask_longest)
                current_main_samples[k] = set(zip(xs.tolist(), ys.tolist()))
                # A seed-edge blob can be the only first detection. Show it,
                # but establish identity only once a longitudinal axis exists.
                if len(xs) and (previous_main.get(k) or has_main_axis(main_geometry, upper_corners)):
                    previous_main.setdefault(k, set()).update(zip(xs.tolist(), ys.tolist()))
                if not has_mask:
                    main_pixels.extend(zip(np.full(len(xs), k), xs, ys))
                
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
            
            # Shared segments are painted last so neither plant overwrites
            # the other's half, including on highlighted main-root paths.
            from .tracking.shared_rendering import draw_shared_segments
            draw_shared_segments(annotated, upper_corners, colored, main_segments, self.linker,
                                 main_samples=current_main_samples)

            # Save only the pixels needed to hide the markers instantly.
            ys, xs = np.nonzero(depth_mask)
            if not has_mask:
                image_data.main_root_samples = np.asarray(main_pixels, dtype=np.int32).reshape(-1, 3)
            image_data.root_depth_background = np.column_stack(
                (ys, xs, annotated[ys, xs])
            ).astype(np.int32)
            annotated[ys, xs] = 255
            # Store annotated image (exports retain the depth markers).
            image_data.image_annotated = annotated

            # Publish calculated pixels before any image encoding or cache I/O.
            if image_callback:
                image_callback(image_data)
            if save_images:
                self.exporter.save_image(image_data.image_annotated, os.path.basename(image_data.path))
            
            # Update progress (after processing)
            if progress_callback:
                progress_callback(idx + 1, total_images)
        
        if has_replacements:
            from .analysis.rsml_statistics import refresh_statistics
            return refresh_statistics(series, statistics)

        # Validate monotonic growth
        for key in ["image_total_area", "image_total_length", "plant_total_length", 
                    "plant_main_root_depth", "plant_main_root_length"]:
            warnings = self.statistics_calc.validate_monotonic_growth(statistics, key)
            for warning in warnings:
                print(warning)
        
        return statistics
    
    def is_tracking_current(self, series: ImageSeries) -> bool:
        """Check if tracking results are still valid for the current config."""
        if series.images and all(image.rsml_document is not None for image in series):
            return True
        state = series.pipeline_state
        if any(image.rsml_document is None and image.rsml_samples is not None
               and image.main_root_samples is None for image in series):
            return False  # Upgrade caches that predate editable main-root markings.
        config = self.config.for_series(series)
        if (series.user_mask is not None and np.any(series.user_mask)
                and any(image.rsml_document is None and image.rsml_unmasked_samples is None for image in series)):
            return False  # Upgrade old masked caches with restorable geometry.
        return (
            state.tracked
            and state.tracking_config_hash == config.tracking_config_hash()
            and state.preprocess_config_hash == config.preprocess_config_hash()
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
                stats = state.last_statistics
            else:
                # If not tracked yet, track it now (synchronously)
                pipeline = (self if series.processing_settings is None
                            else RootTrackingPipeline(self.config.for_series(series)))
                stats = pipeline.track_and_analyze_series(series)
            if any(image.rsml_document is not None for image in series):
                from .analysis.rsml_statistics import refresh_statistics
                stats = refresh_statistics(series, stats)
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
        # The CLI's explicit pipeline config governs a fresh full run.
        series.processing_settings = self.config.processing_settings()
        try:
            self.preprocess_series(series)
            self.register_series(series)
            statistics = self.track_and_analyze_series(series)
            if self.rsml_output is not None:
                from pathlib import Path
                from .io.rsml_export import export_series_rsml
                state = series.pipeline_state
                state.preprocessed = state.tracked = True
                state.preprocess_config_hash = self.config.preprocess_config_hash()
                state.tracking_config_hash = self.config.tracking_config_hash()
                export_series_rsml(series, self.config, Path(self.rsml_output))
            return statistics
        except Exception as e:
            print(f"Error processing {series.group}: {e}")
            if self.rsml_output is not None:
                raise
            return []
    
    def run(self, parallel: bool = True, max_workers: int | None = None,
            rsml_output: str | None = None) -> str:
        """
        Run the full pipeline.
        
        Args:
            parallel: If True, process series in parallel.
            max_workers: Maximum number of parallel workers.
            
        Returns:
            Path to the saved CSV file.
        """
        self.rsml_output = rsml_output
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
        pipeline.rsml_output = self.rsml_output
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

    config = config.for_series(series)
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

    config = config.for_series(series)
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
