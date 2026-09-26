"""Preprocessing pixels and ownership must survive copy elimination."""
from datetime import datetime
from itertools import combinations
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from root_tracker.config import Config
from root_tracker.models import ImageData
from root_tracker.pipeline import RootTrackingPipeline


def reference_preprocess(pipeline, original):
    """The previous copy-heavy path, using the same CV operations."""
    if pipeline.config.rotation:
        original = pipeline.cropper.rotate(original)
    cropped = pipeline.cropper.auto_crop_to_blue_background(original)
    contours, _ = pipeline.green_detector.find_green_contours(cropped)
    min_y = pipeline.green_detector.find_crop_start(contours, cropped.shape[0])
    unmasked = cropped.copy()
    cropped = pipeline.green_detector.mask_green_in_image(cropped, contours)
    pos_x, pos_y, areas = pipeline.green_detector.cluster_plant_positions(contours)
    cropped = cropped[min_y:].copy()
    image = unmasked[min_y:].copy()
    process = pipeline.background_remover.remove_gradient(cropped)
    canny = pipeline.background_remover.compute_canny_edges(process)
    height, width = image.shape[:2]
    top, bottom, left, right = pipeline.cropper.margin_offsets(height, width)
    roi = (slice(top, height - bottom), slice(left, width - right))
    return {
        'image': image[roi].copy(),
        'process': process[roi].copy(),
        'canny': canny[roi].copy(),
        'positions_x': [x - left for x in pos_x],
        'positions_y': [y - min_y - top for y in pos_y],
        'green_areas': areas,
    }


class PreprocessCopyTests(unittest.TestCase):
    def photograph(self, green=True):
        image = np.zeros((113, 157, 3), np.uint8)
        rng = np.random.default_rng(17)
        image[9:-9, 12:-12, 0] = rng.integers(170, 255, (95, 133), np.uint8)
        cv2.line(image, (59, 36), (72, 100), (230, 230, 230), 3)
        if green:
            color = cv2.cvtColor(np.uint8([[[35, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
            image[20:43, 48:69] = color
        return image

    def test_exact_pixels_and_independent_owned_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for rotation in (0, 90, 180, 270):
                for margins in ((0, 0, 0, 0), (.1, .05, .07, .09)):
                    for green in (False, True):
                        with self.subTest(rotation=rotation, margins=margins, green=green):
                            config = Config(rotation=rotation, n_clusters=1)
                            (config.margin_top, config.margin_bottom,
                             config.margin_left, config.margin_right) = margins
                            config.data.output = directory
                            pipeline = RootTrackingPipeline(config)
                            # One fixed centroid isolates copy behavior from KMeans RNG.
                            positions = ([50], [30], [300]) if green else ([], [], [])
                            image = self.photograph(green)
                            before = image.copy()
                            image.flags.writeable = False
                            with patch.object(pipeline.green_detector, 'cluster_plant_positions',
                                              return_value=positions):
                                expected = reference_preprocess(pipeline, image)
                                actual = ImageData(datetime(2026, 1, 1), 'unused.png', 'test')
                                pipeline.preprocess_image(actual, original=image)
                            for field, value in expected.items():
                                np.testing.assert_array_equal(getattr(actual, field), value)
                            np.testing.assert_array_equal(image, before)
                            arrays = [actual.image, actual.process, actual.canny]
                            for array in arrays:
                                self.assertTrue(array.flags.owndata)
                                self.assertTrue(array.flags.c_contiguous)
                                self.assertFalse(np.shares_memory(array, image))
                            for first, second in combinations(arrays, 2):
                                self.assertFalse(np.shares_memory(first, second))

    def test_cluster_failure_does_not_change_input_or_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(rotation=0)
            config.data.output = directory
            pipeline = RootTrackingPipeline(config)
            original = self.photograph()
            before = original.copy()
            data = ImageData(datetime(2026, 1, 1), 'unused.png', 'test')
            previous = np.ones((3, 5, 3), np.uint8)
            data.image = previous
            with patch.object(pipeline.green_detector, 'cluster_plant_positions',
                              side_effect=ValueError('not enough stems')):
                pipeline.preprocess_image(data, original=original)
            self.assertIs(data.image, previous)
            self.assertIsNone(data.process)
            np.testing.assert_array_equal(original, before)


if __name__ == '__main__':
    unittest.main()
