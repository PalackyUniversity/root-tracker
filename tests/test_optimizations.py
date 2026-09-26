"""Numerical regression cases for performance-sensitive CV operations."""
import unittest
import tempfile
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from root_tracker.config import Config
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.tracking.thresholder import RootThresholder
from root_tracker.tracking.corner_detector import CornerDetector, NEIGHBOR_OFFSETS
from root_tracker.preprocessing.background import BackgroundRemover
from root_tracker.registration.template_matcher import ImageRegistrator
from root_tracker.models import ImageData, ImageSeries
from root_tracker.io import series_cache
from root_tracker.tracking.skeletonizer import RootSkeletonizer
from skimage.morphology import skeletonize


def reference_threshold(image, config):
    low = cv2.threshold(image, config.threshold.low, 255, cv2.THRESH_BINARY)[1]
    high = cv2.threshold(image, config.threshold.high, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(low, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    result = np.zeros_like(image)
    if hierarchy is None:
        return result
    keep, ignore = [], set()
    for n, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt], 0, 255, cv2.FILLED)
        if h[3] == -1:
            area = cv2.countNonZero(mask)
            overlap = cv2.countNonZero(cv2.bitwise_and(high, mask))
            if len(cnt) > config.threshold.min_contour_length and area and overlap / area > .2:
                keep.append(cnt)
            else:
                ignore.add(n)
    keep.extend(c for c, h in zip(contours, hierarchy[0]) if h[3] != -1 and h[3] not in ignore)
    cv2.drawContours(result, keep, -1, 255, cv2.FILLED)
    return result


class OptimizationTests(unittest.TestCase):
    def test_predecoded_image_matches_regular_preprocessing_without_mutation(self):
        config = Config(n_clusters=1, rotation=0)
        image = np.full((128, 192, 3), (255, 0, 0), np.uint8)
        green = cv2.cvtColor(np.uint8([[[35, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
        image[20:50, 70:90] = green
        original = image.copy()
        with tempfile.TemporaryDirectory() as folder:
            config.data.output = folder
            path = str(Path(folder) / 'input.png')
            cv2.imwrite(path, image)
            pipeline = RootTrackingPipeline(config)
            regular = ImageData(datetime(2026, 1, 1), path, 'group')
            predecoded = ImageData(datetime(2026, 1, 1), 'not-on-disk.png', 'group')
            np.random.seed(0)
            pipeline.preprocess_image(regular)
            np.random.seed(0)
            pipeline.preprocess_image(predecoded, original=image)
            self.assertIsNotNone(regular.process)
            for field in ('image', 'process', 'canny'):
                np.testing.assert_array_equal(getattr(regular, field), getattr(predecoded, field))
            self.assertEqual(regular.positions_x, predecoded.positions_x)
            self.assertEqual(regular.positions_y, predecoded.positions_y)
            np.testing.assert_array_equal(image, original)
            series = ImageSeries('group', [ImageData(datetime(2026, 1, i+1), path, 'group')
                                          for i in range(3)])
            progress = []
            pipeline.preprocess_series(series, progress_callback=lambda i, n: progress.append((i, n)))
            self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])
            for item in series.images:
                np.testing.assert_array_equal(item.process, regular.process)
            interrupted = ImageSeries('group', [ImageData(datetime(2026, 1, i+1), path, 'group')
                                               for i in range(3)])
            def cancel(current, total):
                raise InterruptedError('Cancelled')
            with self.assertRaises(InterruptedError):
                pipeline.preprocess_series(interrupted, progress_callback=cancel)
            self.assertIsNotNone(interrupted.images[0].process)
            self.assertTrue(all(item.process is None for item in interrupted.images[1:]))

    def test_registration_score_preserves_full_search_offsets(self):
        rng = np.random.default_rng(76)
        registrator = ImageRegistrator(Config())
        template = (rng.random((731, 809)) < .012).astype(np.uint8) * 255
        target = (rng.random((1103, 1307)) < .008).astype(np.uint8) * 255
        target[311:1042, 417:1226] = template
        # Full search must retain large translations, not just nearby matches.
        expected = cv2.minMaxLoc(cv2.matchTemplate(target, template, cv2.TM_CCOEFF))[3]
        self.assertEqual(registrator._match_location(target, template), expected)
        # Blank/tied and non-binary inputs must preserve OpenCV's tie behavior.
        for sample in (np.zeros_like(template), template // 3):
            expected = cv2.minMaxLoc(cv2.matchTemplate(target, sample, cv2.TM_CCOEFF))[3]
            self.assertEqual(registrator._match_location(target, sample), expected)

    def test_skeleton_components_match_full_frame(self):
        mask = np.zeros((600, 700), np.uint8)
        cv2.rectangle(mask, (0, 0), (25, 599), 1, -1)
        cv2.ellipse(mask, (250, 220), (110, 190), 20, 0, 360, 1, 9)
        cv2.line(mask, (450, 0), (699, 500), 1, 13)
        cv2.circle(mask, (250, 220), 14, 1, -1)
        rng = np.random.default_rng(91)
        for sample in [mask, np.zeros_like(mask), np.ones_like(mask),
                       (rng.random(mask.shape) > .8).astype(np.uint8)]:
            np.testing.assert_array_equal(
                RootSkeletonizer(Config()).skeletonize_mask(sample),
                skeletonize(sample).astype(np.float32))

    def test_cache_roundtrip_compressed_and_uncompressed(self):
        image = np.random.default_rng(9).integers(0, 256, (33, 45, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            config = Config()
            config.data.input = directory
            for compressed in (True, False):
                config.data.cache_compressed = compressed
                original = ImageSeries(group='test')
                original.add_image(ImageData(datetime(2026, 1, 1), 'input.jpg', 'test'))
                original.images[0].image = image
                original.images[0].positions_x = [4, 15]
                original.images[0].plate_transform = [0.96, -0.28, -41.5, 0.28, 0.96, -17.25]
                original.pipeline_state.preprocessed = True
                self.assertIsInstance(series_cache.save_series(original, config), Path)
                restored = ImageSeries(group='test')
                restored.add_image(ImageData(datetime(2026, 1, 1), 'input.jpg', 'test'))
                self.assertTrue(series_cache.load_series(restored, config))
                np.testing.assert_array_equal(restored.images[0].image, image)
                self.assertEqual(restored.images[0].positions_x, [4, 15])
                self.assertEqual(restored.images[0].plate_transform, original.images[0].plate_transform)
                self.assertTrue(restored.pipeline_state.preprocessed)

    def test_threshold_holes_borders_noise_and_empty(self):
        config = Config()
        config.threshold.low, config.threshold.high = 20, 30
        rng = np.random.default_rng(41)
        structured = np.zeros((121, 177), np.uint8)
        cv2.rectangle(structured, (0, 0), (70, 90), 25, -1)
        cv2.rectangle(structured, (3, 8), (30, 80), 80, -1)
        cv2.circle(structured, (40, 40), 13, 0, -1)
        cv2.circle(structured, (40, 40), 4, 90, -1)
        for image in [structured, np.zeros_like(structured),
                      rng.integers(0, 60, structured.shape, dtype=np.uint8),
                      structured[::2, ::2]]:
            np.testing.assert_array_equal(RootThresholder(config).threshold(image),
                                          reference_threshold(image, config))

    def test_endpoints_preserve_order_duplicates_and_bounds(self):
        rng = np.random.default_rng(17)
        for shape in [(1, 1), (1, 19), (23, 1), (31, 27)]:
            skeleton = (rng.random(shape) > .65).astype(np.uint8) * 255
            points = np.column_stack(np.where(skeleton > 0)[::-1]).astype(np.int32)
            contour = np.concatenate([points, points[::-1]]).reshape(-1, 1, 2)
            expected = []
            for x, y in contour[:, 0]:
                count = sum(0 <= y+dy < shape[0] and 0 <= x+dx < shape[1]
                            and skeleton[y+dy, x+dx] > 0 for dx, dy in NEIGHBOR_OFFSETS)
                if count == 1:
                    expected.append((x, y))
            self.assertEqual(CornerDetector(Config()).find_contour_endpoints(contour, skeleton), expected)

    def test_background_saturating_subtraction_is_exact(self):
        image = np.random.default_rng(19).integers(0, 256, (117, 139, 3), dtype=np.uint8)
        expected = np.maximum(image.astype(int) - cv2.medianBlur(image, 101).astype(int), 0).astype(np.uint8)
        expected = cv2.cvtColor(cv2.medianBlur(expected, 5), cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(BackgroundRemover(Config()).remove_gradient(image), expected)

    def test_large_background_preserves_tile_boundaries(self):
        image = np.random.default_rng(21).integers(0, 256, (1081, 301, 3), dtype=np.uint8)
        expected = cv2.subtract(image, cv2.medianBlur(image, 101))
        expected = cv2.cvtColor(cv2.medianBlur(expected, 5), cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(BackgroundRemover(Config()).remove_gradient(image), expected)


if __name__ == '__main__':
    unittest.main()
