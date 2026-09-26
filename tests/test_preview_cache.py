import unittest
from threading import Event
import numpy as np
from root_tracker.gui.preview_cache import PreviewCache


class PreviewCacheTests(unittest.TestCase):
    def test_revisit_reuses_exact_pixels_and_oldest_entry_is_evicted(self):
        cache = PreviewCache(max_bytes=24, max_entries=2)
        self.addCleanup(cache.close)
        a = (np.full((2, 2, 3), 1, np.uint8), None)
        b = (np.full((2, 2, 3), 2, np.uint8), None)
        c = (np.full((2, 2, 3), 3, np.uint8), None)
        cache.get('a', lambda: a)
        cache.get('b', lambda: b)
        self.assertIs(cache.get('a', lambda: self.fail('Decoded twice')), a)
        cache.get('c', lambda: c)
        self.assertIs(cache.get('b', lambda: b), b)
        self.assertNotIn('a', cache._entries)

    def test_clear_during_prefetch_does_not_restore_stale_entries(self):
        cache = PreviewCache()
        self.addCleanup(cache.close)
        started, finish = Event(), Event()
        self.addCleanup(finish.set)
        def prepare():
            started.set()
            finish.wait(3)
            return np.zeros((3, 3, 3), np.uint8), None
        cache.prefetch([('old', prepare)])
        self.assertTrue(started.wait(2))
        future = cache._pending['old']
        cache.clear()
        finish.set()
        future.result(timeout=2)
        self.assertNotIn('old', cache._entries)
