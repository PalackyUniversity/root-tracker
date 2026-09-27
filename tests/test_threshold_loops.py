"""Root loops must be judged by foreground brightness, not enclosed area."""
import unittest

import cv2
import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.thresholder import RootThresholder


class ThresholdLoopTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.threshold.low = 20
        self.config.threshold.high = 30
        self.thresholder = RootThresholder(self.config)

    def test_bright_root_survives_closing_a_large_loop(self):
        closed = np.zeros((160, 160), np.uint8)
        cv2.rectangle(closed, (20, 20), (140, 140), 80, 2)
        opened = closed.copy()
        opened[19:22, 70:80] = 0
        for image in (opened, closed):
            with self.subTest(closed=image is closed):
                expected = (image > 20).astype(np.uint8) * 255
                np.testing.assert_array_equal(self.thresholder.threshold(image), expected)

    def test_bright_island_cannot_rescue_a_dim_enclosing_loop(self):
        image = np.zeros((160, 160), np.uint8)
        cv2.rectangle(image, (20, 20), (140, 140), 25, 2)
        image[40:120, 40:120] = 80
        expected = np.zeros_like(image)
        expected[40:120, 40:120] = 255
        np.testing.assert_array_equal(self.thresholder.threshold(image), expected)

    def test_dim_loop_is_still_rejected(self):
        image = np.zeros((160, 160), np.uint8)
        cv2.rectangle(image, (20, 20), (140, 140), 25, 2)
        self.assertEqual(cv2.countNonZero(self.thresholder.threshold(image)), 0)
