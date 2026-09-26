"""Export must preserve tracker state and survive its cache boundaries."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
import cv2

from root_tracker.config import Config
from root_tracker.models import ImageData, ImageSeries
from root_tracker.io import series_cache
from root_tracker.io.rsml import read_rsml
from root_tracker.io.rsml_export import trace_paths, export_series_rsml


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config = Config(n_clusters=1)
        self.config.data.input = str(self.base)
        self.config.data.output = str(self.base / 'images')
        self.image = ImageData(datetime(2026, 1, 2), 'same.png', 'p')
        self.image.image = np.zeros((12, 10, 3), np.uint8)
        self.image.image[2, 3] = (20, 40, 60)
        self.image.rsml_samples = {0: np.array([[3, 2], [3, 3], [3, 4]])}
        self.image.plant_length = [3]
        self.series = ImageSeries('../unsafe/group', [self.image])
        state = self.series.pipeline_state
        state.preprocessed = state.tracked = True
        state.preprocess_config_hash = self.config.preprocess_config_hash()
        state.tracking_config_hash = self.config.tracking_config_hash()

    def test_single_image_export_retains_original_index_after_cache_reload(self):
        other = self.image.copy_for_editing()
        other.path = 'second.png'
        other.rsml_samples[0][:, 0] = 7
        self.series.images.append(other)
        series_cache.save_series(self.series, self.config)
        for image in self.series:
            image.image = None
            image.rsml_samples = None
        files = export_series_rsml(self.series, self.config, self.base / 'export', image_index=1)
        self.assertEqual(len(list((self.base / 'export').rglob('*.rsml'))), 1)
        self.assertEqual(read_rsml(files[0]).roots[0].points[0], (7., 2.))
        self.assertEqual(ET.parse(files[0]).findtext('./metadata/time-sequence/index'), '1')
        self.assertIsNone(other.image)
        export_series_rsml(self.series, self.config, self.base / 'export', image_index=0)
        self.assertEqual(len(list((self.base / 'export').rglob('*.rsml'))), 2)

    def test_straight_and_disconnected_paths(self):
        points = np.array([[2, 1], [2, 2], [2, 3], [8, 8]])
        self.assertEqual(trace_paths(points), [[(2, 1), (2, 2), (2, 3)], [(8, 8)]])
        self.assertEqual(trace_paths(points[::-1]), trace_paths(points))
        self.assertEqual(trace_paths(np.empty((0, 2), int)), [])

    def test_branch_and_cycle_have_each_edge_once(self):
        for points, expected_edges in [
            ([(1, 0), (1, 1), (0, 1), (2, 1)], 3),
            ([(1, 0), (2, 1), (1, 2), (0, 1)], 4),
            ([(0, 0), (1, 1), (2, 2)], 2),
        ]:
            with self.subTest(points=points):
                paths = trace_paths(np.array(points))
                edges = [frozenset((a, b)) for path in paths for a, b in zip(path, path[1:])]
                self.assertEqual(len(edges), expected_edges)
                self.assertEqual(len(set(edges)), expected_edges)

    def test_export_coordinates_image_and_no_mutation(self):
        files = export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertEqual(len(files), 1)
        doc = read_rsml(files[0])
        self.assertEqual(doc.roots[0].points, ((3., 2.), (3., 3.), (3., 4.)))
        self.assertEqual(doc.unit, 'pixel')
        np.testing.assert_array_equal(cv2.imread(str(files[0].parent / doc.image_name)), self.image.image)
        self.assertEqual(self.image.plant_length, [3])
        self.assertTrue(self.series.pipeline_state.tracked)
        self.assertEqual(self.image.colored_samples, {})
        self.assertTrue(files[0].is_relative_to(self.base / 'export'))
        tree = ET.parse(files[0])
        self.assertEqual(tree.findtext('./metadata/resolution'), '1')
        with self.assertRaises(FileExistsError):
            export_series_rsml(self.series, self.config, self.base / 'export')

    def test_duplicate_names_and_empty_geometry(self):
        other = self.image.copy_for_editing()
        other.rsml_samples = {}
        self.series.images.append(other)
        files = export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertEqual(len(set(files)), 2)
        self.assertEqual(read_rsml(files[1]).roots, ())

    def test_cache_copy_and_invalidation_do_not_restore_tracking_samples(self):
        copy = self.image.copy_for_editing()
        copy.rsml_samples[0][0, 0] = 9
        self.assertEqual(self.image.rsml_samples[0][0, 0], 3)
        self.assertIsNotNone(series_cache.save_series(self.series, self.config))
        fresh = ImageSeries(self.series.group, [ImageData(self.image.date, self.image.path, 'p')])
        self.assertTrue(series_cache.load_series(fresh, self.config))
        np.testing.assert_array_equal(fresh[0].rsml_samples[0], self.image.rsml_samples[0])
        self.assertEqual(fresh[0].colored_samples, {})
        fresh.clear_tracking_results()
        self.assertIsNone(fresh[0].rsml_samples)

    def test_evicted_export_loads_cache_without_mutating_live_series(self):
        series_cache.save_series(self.series, self.config)
        self.image.image = None
        self.image.rsml_samples = None
        files = export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertEqual(len(files), 1)
        self.assertIsNone(self.image.image)
        self.assertIsNone(self.image.rsml_samples)

    def test_colliding_cache_names_cannot_mislabel_geometry(self):
        from copy import deepcopy
        self.series.group = 'a/b'
        series_cache.save_series(self.series, self.config)
        other = deepcopy(self.series)
        other.group = 'a_b'
        other[0].rsml_samples[0][:, 0] = 8
        series_cache.save_series(other, self.config)
        self.image.image = None
        self.image.rsml_samples = None
        with self.assertRaises(ValueError):
            export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertFalse((self.base / 'export').exists())

    def test_changed_image_identity_cannot_reuse_export_cache(self):
        series_cache.save_series(self.series, self.config)
        self.image.path = 'different.png'
        self.image.image = None
        self.image.rsml_samples = None
        with self.assertRaises(ValueError):
            export_series_rsml(self.series, self.config, self.base / 'export')

    def test_old_cache_and_stale_tracking_fail_before_writing(self):
        self.image.rsml_samples = None
        series_cache.save_series(self.series, self.config)
        with self.assertRaisesRegex(ValueError, 'track|Track'):
            export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertFalse((self.base / 'export').exists())
        self.series.pipeline_state.tracking_config_hash = 'old'
        with self.assertRaises(ValueError):
            export_series_rsml(self.series, self.config, self.base / 'export')

    def test_image_failure_leaves_no_partial_series(self):
        with patch('root_tracker.io.rsml_export.cv2.imwrite', return_value=False):
            with self.assertRaises(OSError):
                export_series_rsml(self.series, self.config, self.base / 'export')
        self.assertEqual(list((self.base / 'export').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
