"""Crop geometry preserves pixels, coordinates and legacy automatic behavior."""
import unittest
import cv2
import numpy as np
from root_tracker.config import Config
from root_tracker.preprocessing.cropper import ImageCropper
from root_tracker.preprocessing import roi
from root_tracker.preprocessing.colors import color_mask


class RoiGeometryTests(unittest.TestCase):
    def test_axis_aligned_crop_and_coordinates(self):
        image = np.arange(80*100, dtype=np.float32).reshape(80, 100)
        box = roi.from_bounds(10, 12, 60, 40, image.shape)
        actual, matrix = roi.extract(image, box)
        np.testing.assert_array_equal(actual, image[12:52, 10:70])
        np.testing.assert_allclose(roi.transform_points([(10, 12), (30, 20)], matrix), [(0, 0), (20, 8)])

    def test_cardinal_boxes_equal_opencv_rotations(self):
        image = np.random.default_rng(0).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        for rotation in (0, 90, 180, 270):
            config = Config(rotation=rotation)
            cropper = ImageCropper(config)
            oriented = cropper.rotate(image)
            h, w = oriented.shape[:2]
            box = roi.unrotate_box(roi.from_bounds(3, 4, w-8, h-9, oriented.shape), oriented.shape, image.shape, rotation)
            actual, _ = roi.extract(image, box)
            np.testing.assert_array_equal(actual, oriented[4:h-5, 3:w-5])

    def test_noncardinal_extract_keeps_landmark_coordinates(self):
        image = np.zeros((240, 300), np.uint8)
        cv2.circle(image, (150, 120), 4, 255, -1)
        actual, matrix = roi.extract(image, (.5, .5, .5, .5, 23.7))
        point = roi.transform_points([(150, 120)], matrix)[0]
        self.assertEqual(actual.shape, (120, 150))
        self.assertGreater(actual[round(point[1]), round(point[0])], 200)

    def test_color_range_supports_hue_wrap_and_cleanup(self):
        hsv = np.zeros((20, 30, 3), np.uint8)
        hsv[:, :10] = (178, 240, 230)
        hsv[:, 10:20] = (2, 240, 230)
        hsv[:, 20:] = (90, 240, 230)
        mask = color_mask(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (170, 200, 200), (10, 255, 255))
        self.assertTrue(np.all(mask[:, :20] == 255))
        self.assertFalse(np.any(mask[:, 20:]))

    def test_two_samples_define_range_in_either_order_and_across_red(self):
        from root_tracker.preprocessing.colors import range_from_samples
        for first, second, expected in (
            ((100, 200, 80), (120, 150, 220), ((100, 150, 80), (120, 200, 220))),
            ((178, 190, 110), (2, 250, 210), ((178, 190, 110), (2, 250, 210))),
        ):
            self.assertEqual(range_from_samples(first, second), expected)
            self.assertEqual(range_from_samples(second, first), expected)

    def test_no_matches_in_manual_search_keeps_search_area_and_its_orientation(self):
        config = Config()
        config.load_roi = (.5, .5, .5, .5, 0)
        image = np.zeros((80, 100, 3), np.uint8)
        image[20:60, 25:75] = (20, 40, 100)
        np.testing.assert_array_equal(ImageCropper(config).process(image), image[20:60, 25:75])

    def test_no_blue_matches_keeps_image_available_for_manual_crop(self):
        image = np.zeros((40, 50, 3), np.uint8)
        self.assertEqual(ImageCropper(Config(rotation=0)).process(image).shape, image.shape)

    def test_config_accepts_arbitrary_rotation_and_roundtrips_rois(self):
        import tempfile
        from pathlib import Path
        config = Config(rotation=17.5, load_roi=(.5, .5, .7, .8, -11), preprocess_roi=(.5, .5, .8, .9, 8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.yaml'
            import yaml
            from dataclasses import asdict
            values = asdict(config)
            values.pop('_base_path')
            path.write_text(yaml.safe_dump(values))
            restored = Config.from_yaml(path)
        self.assertEqual(restored.load_roi, config.load_roi)
        self.assertEqual(restored.preprocess_roi, config.preprocess_roi)
        self.assertEqual(restored.rotation, 17.5)


if __name__ == '__main__':
    unittest.main()

class PlateCropSemanticsTests(unittest.TestCase):
    def test_load_box_limits_search_then_detector_finds_plate_inside_it(self):
        image = np.zeros((240, 320, 3), np.uint8)
        image[10:230, :45] = (200, 60, 20)  # unrelated blue object outside search
        image[60:200, 100:260] = (200, 60, 20)
        config = Config(rotation=180, load_roi=roi.from_bounds(70, 30, 220, 190, image.shape))
        config.crop.top_ratio, config.crop.bottom_ratio = .1, .85
        actual = ImageCropper(config).process(image)
        np.testing.assert_array_equal(actual, image[63:197, 103:257])

    def test_plate_detection_ignores_smaller_matching_regions(self):
        image = np.zeros((200, 300, 3), np.uint8)
        image[30:170, 80:240] = (200, 60, 20)
        image[5:25, 5:25] = (200, 60, 20)
        result = ImageCropper(Config(rotation=0)).process(image)
        np.testing.assert_array_equal(result, image[33:167, 83:237])

    def test_default_analysis_box_is_fixed_relative_to_plate(self):
        from root_tracker.gui.roi_editor import RoiEditor
        config = Config(rotation=0)
        first = np.zeros((200, 300, 3), np.uint8)
        second = first.copy()
        second[60:80] = cv2.cvtColor(np.uint8([[[35, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
        self.assertEqual(RoiEditor._analysis_box(first, config), RoiEditor._analysis_box(second, config))

    def test_fixed_preprocess_crop_follows_plate_when_plate_moves(self):
        from datetime import datetime
        from unittest.mock import patch
        from root_tracker.pipeline import RootTrackingPipeline
        from root_tracker.models import ImageData
        config = Config(rotation=0, n_clusters=1, preprocess_roi=(.5, .6, .7, .6, 0))
        pipeline = RootTrackingPipeline(config)
        results = []
        for x, y in ((40, 30), (90, 50)):
            image = np.zeros((260, 340, 3), np.uint8)
            image[y:y+180, x:x+180] = (200, 60, 20)
            image[y+70:y+150, x+75:x+78] = (230, 230, 230)
            data = ImageData(datetime(2026, 1, 1), 'unused.png', 'a')
            with patch.object(pipeline.green_detector, 'cluster_plant_positions', return_value=([70], [50], [100])):
                pipeline.preprocess_image(data, original=image)
            results.append(data)
        for field in ('image', 'process', 'canny', 'positions_x', 'positions_y'):
            np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))

    def test_region_modes_and_disabled_detection(self):
        image = np.zeros((200, 300, 3), np.uint8)
        image[30:170, 80:240] = (200, 60, 20)
        image[5:25, 5:25] = (200, 60, 20)
        config = Config(rotation=0)
        cropper = ImageCropper(config)
        self.assertEqual(cropper.blue_bounds(image), (83, 33, 154, 134))
        before = config.preprocess_config_hash()
        config.crop.background_region = 'all'
        self.assertEqual(cropper.blue_bounds(image), (8, 8, 229, 159))
        self.assertNotEqual(before, config.preprocess_config_hash())
        config.crop.background_enabled = False
        from unittest.mock import patch
        with patch.object(cropper, 'blue_mask', side_effect=AssertionError('Detection must be disabled')):
            np.testing.assert_array_equal(cropper.process(image), image)
