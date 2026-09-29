"""Batch capacity follows memory pressure rather than logical CPU count."""
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from root_tracker.resources import (
    batch_worker_count, batch_thread_count, estimate_series_memory, initialize_batch_worker,
)

GIB = 1024 ** 3


def worker_thread_count():
    import cv2
    return cv2.getNumThreads()


class ResourceTests(unittest.TestCase):
    def test_spawned_initializer_limits_only_child_opencv_threads(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        before = worker_thread_count()
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                                 initializer=initialize_batch_worker, initargs=(2,)) as executor:
            self.assertEqual(executor.submit(worker_thread_count).result(timeout=30), 2)
        self.assertEqual(worker_thread_count(), before)

    def test_native_thread_budget_scales_down_as_pool_grows(self):
        with patch('root_tracker.resources.os.cpu_count', return_value=16):
            self.assertEqual([batch_thread_count(n) for n in (1, 4, 8, 16)], [4, 4, 2, 1])

    def test_pressure_limits_sixteen_core_machine_to_one_worker(self):
        with patch('root_tracker.resources.system_memory', return_value=(32 * GIB, 3 * GIB)), \
                patch('root_tracker.resources.os.cpu_count', return_value=16), \
                patch('root_tracker.resources.estimate_series_memory', return_value=2 * GIB):
            self.assertEqual(batch_worker_count([object()] * 50), 1)

    def test_available_memory_group_count_and_explicit_cap(self):
        with patch('root_tracker.resources.system_memory', return_value=(32 * GIB, 14 * GIB)), \
                patch('root_tracker.resources.os.cpu_count', return_value=16), \
                patch('root_tracker.resources.estimate_series_memory', return_value=2 * GIB):
            self.assertEqual(batch_worker_count([object()] * 50), 6)
            self.assertEqual(batch_worker_count([object()] * 2), 2)
            self.assertEqual(batch_worker_count([object()] * 50, 3), 3)

    def test_largest_group_bounds_concurrent_work(self):
        with patch('root_tracker.resources.system_memory', return_value=(32 * GIB, 14 * GIB)), \
                patch('root_tracker.resources.os.cpu_count', return_value=16), \
                patch('root_tracker.resources.estimate_series_memory', side_effect=[GIB, 8 * GIB]):
            self.assertEqual(batch_worker_count([object(), object()]), 1)

    def test_unknown_or_exhausted_memory_still_allows_serial_progress(self):
        for memory in (None, (32 * GIB, 0)):
            with patch('root_tracker.resources.system_memory', return_value=memory):
                self.assertEqual(batch_worker_count([]), 1)
                self.assertEqual(batch_worker_count([SimpleNamespace(images=[])]), 1)
        with self.assertRaises(ValueError):
            batch_worker_count([], 0)

    def test_photo_headers_are_used_without_decoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photo.png'
            Image.new('RGB', (1000, 2000)).save(path)
            group = SimpleNamespace(images=[SimpleNamespace(path=path, image=None)])
            with patch.object(Image.Image, 'load', side_effect=AssertionError('decoded')):
                self.assertEqual(estimate_series_memory(group), GIB)
            # Full-resolution estimates grow with image size and group length.
            group.images = [SimpleNamespace(image=np.empty((6000, 4000, 0), np.uint8))] * 4
            self.assertGreater(estimate_series_memory(group), 3 * GIB)


if __name__ == '__main__':
    unittest.main()
