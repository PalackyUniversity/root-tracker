"""Optional GPU histogram median must preserve OpenCV pixels and CPU fallback."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np

from root_tracker.preprocessing import gpu_background


class GPUBackgroundTests(unittest.TestCase):
    def test_cpu_opt_out_and_small_images_do_not_initialize_cuda(self):
        with patch.object(gpu_background, '_get_backend', side_effect=AssertionError('Must not initialize CUDA')):
            with patch.dict(os.environ, {'ROOT_TRACKER_GPU': '0'}):
                self.assertIsNone(gpu_background.try_subtract_median(np.zeros((1200, 1200, 3), np.uint8), 101))
            self.assertIsNone(gpu_background.try_subtract_median(np.zeros((10, 10, 3), np.uint8), 101))
            with patch.object(gpu_background, 'current_process', return_value=SimpleNamespace(name='ForkProcess-1')):
                self.assertIsNone(gpu_background.try_subtract_median(np.zeros((1200, 1200, 3), np.uint8), 101))

    def test_missing_gpu_falls_back_without_touching_input(self):
        image = np.full((1024, 1024, 3), 23, np.uint8)
        with patch.dict(os.environ, {'ROOT_TRACKER_GPU': 'auto'}), \
                patch.object(gpu_background, '_get_backend', return_value=None):
            self.assertIsNone(gpu_background.try_subtract_median(image, 101))
        self.assertTrue(np.all(image == 23))

    def test_runtime_gpu_failure_disables_backend_and_preserves_input(self):
        image = np.full((1024, 1024, 3), 23, np.uint8)
        def failed(*args, **kwargs):
            raise RuntimeError('simulated allocation failure')
        with patch.dict(os.environ, {'ROOT_TRACKER_GPU': 'auto'}), \
                patch.object(gpu_background, '_backend', SimpleNamespace(filter=failed)), \
                patch.object(gpu_background, '_unavailable', False):
            with self.assertLogs(gpu_background.log, level='WARNING'):
                self.assertIsNone(gpu_background.try_subtract_median(image, 101))
            self.assertIsNone(gpu_background._get_backend())
            self.assertTrue(np.all(image == 23))

    def test_gpu_matches_replicated_borders_channels_and_subtraction(self):
        backend = gpu_background._get_backend()
        if backend is None:
            self.skipTest('Optional CUDA backend unavailable')
        rng = np.random.default_rng(10)
        for shape in ((1, 17), (23, 1, 3), (39, 57, 3), (519, 19, 3)):
            for kind in ('random', 'zero', 'white'):
                image = (rng.integers(0, 256, shape, dtype=np.uint8) if kind == 'random'
                         else np.full(shape, 255 if kind == 'white' else 0, np.uint8))
                original = image.copy()
                image.setflags(write=False)
                with self.subTest(shape=shape, kind=kind):
                    expected = cv2.medianBlur(image, 101)
                    np.testing.assert_array_equal(backend.filter(image, 101, subtract=False), expected)
                    np.testing.assert_array_equal(backend.filter(image, 101, subtract=True), cv2.subtract(image, expected))
                    np.testing.assert_array_equal(image, original)


if __name__ == '__main__':
    unittest.main()
