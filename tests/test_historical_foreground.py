"""Small observed fragments of established roots survive area filtering."""
import unittest

import numpy as np

from root_tracker.config import Config
from root_tracker.tracking.thresholder import RootThresholder
from root_tracker.tracking.root_linker import RootLinker


class HistoricalForegroundTests(unittest.TestCase):
    def test_fragmented_thin_root_survives_without_restoring_missing_pixels(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:33, 10:35] = 255
        mask[30:33, 45:70] = 255
        mask[65:68, 10:35] = 255  # Equally small unsupported foreground.
        history = {0: {(x, 29) for x in range(10, 70)}}
        result, _ = RootThresholder(Config()).filter_small_contours(
            mask, previous_colored_samples=history)
        self.assertTrue(np.all(result[30:33, 10:35]))
        self.assertTrue(np.all(result[30:33, 45:70]))
        self.assertFalse(np.any(result[65:68, 10:35]))
        self.assertFalse(np.any(result[30:33, 35:45]))
        self.assertFalse(np.any(result[mask == 0]))

    def test_one_historical_contact_does_not_rescue_small_debris(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:33, 10:35] = 255
        result, _ = RootThresholder(Config()).filter_small_contours(
            mask, previous_colored_samples={0: {(10, 31)}})
        self.assertFalse(np.any(result))

    def test_disconnected_shifted_fragment_can_recover_its_historical_owner(self):
        upper = {'point': (102, 200), 'lower_point': (102, 225), 'angle': 270.,
                 'contour': np.array([[[102, y]] for y in range(200, 226)], dtype=np.int32)}
        _, colored, samples = RootLinker(Config(n_clusters=1)).link_corners(
            [upper], [{'point': (100, 10), 'angle': 90., 'plant_id': 0}],
            [100], [10], {0: {(100, y) for y in range(200, 226)}})
        self.assertEqual(colored.get((102, 225)), 0)
        self.assertEqual(samples[0], {(102, y) for y in range(200, 226)})

    def test_minority_historical_contact_cannot_root_disconnected_fragment(self):
        upper = {'point': (102, 200), 'lower_point': (102, 225), 'angle': 270.,
                 'contour': np.array([[[102, y]] for y in range(200, 226)], dtype=np.int32)}
        _, colored, samples = RootLinker(Config(n_clusters=1)).link_corners(
            [upper], [{'point': (100, 10), 'angle': 90., 'plant_id': 0}],
            [100], [10], {0: {(100, y) for y in range(200, 206)}})
        self.assertNotIn((102, 225), colored)
        self.assertFalse(samples[0])

class RecordedHistoricalForegroundTests(unittest.TestCase):
    def test_rt12_and_rt7_observed_fragments_survive(self):
        from pathlib import Path
        with np.load(Path(__file__).parent / 'fixtures/historical_foreground.npz') as data:
            for group, boxes in [(12, [(15, 25, 36, 39), (62, 16, 20, 12)]),
                                 (7, [(17, 23, 39, 9)])]:
                mask = data[f'rt{group}_mask']
                history = {0: set(map(tuple, data[f'rt{group}_history']))}
                before, _ = RootThresholder(Config()).filter_small_contours(mask)
                after, _ = RootThresholder(Config()).filter_small_contours(
                    mask, previous_colored_samples=history)
                for x, y, w, h in boxes:
                    with self.subTest(group=group, box=(x, y, w, h)):
                        self.assertEqual(np.count_nonzero(before[y:y+h, x:x+w]), 0)
                        self.assertGreater(np.count_nonzero(after[y:y+h, x:x+w]), 0)
                self.assertFalse(np.any(after[mask == 0]))


if __name__ == '__main__':
    unittest.main()
