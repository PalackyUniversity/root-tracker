"""Batch workers and cache reloads retain each group's processing parameters."""
from datetime import datetime
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from root_tracker.config import Config
from root_tracker.io import series_cache
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline, preprocess_and_cache_worker, track_and_cache_worker


class GroupSettingsTests(unittest.TestCase):
    def test_batch_export_and_cache_reload_use_each_groups_parameters(self):
        with TemporaryDirectory() as directory:
            base = Config()
            base.data.input = base.data.output = directory
            groups = {}
            for name, count, area in [('a', 3, 111), ('b', 6, 100)]:
                config = Config.from_dict(base.to_dict())
                config.n_clusters = count
                config.threshold.min_contour_area = area
                image = ImageData(datetime(2026, 4, 27), f'{name}.png', name)
                image.barcode_detected = True
                series = ImageSeries(name, [image])
                series.processing_settings = config.processing_settings()
                groups[name] = series

            # Isolate costly image analysis; run the actual worker, cache and export paths.
            def analyze(pipeline, series, **kwargs):
                return [{'group': series.group, 'origins': pipeline.config.n_clusters,
                         'area_limit': pipeline.config.threshold.min_contour_area}]

            with patch.object(RootTrackingPipeline, 'preprocess_series'), \
                    patch.object(RootTrackingPipeline, 'register_series'), \
                    patch.object(RootTrackingPipeline, 'track_and_analyze_series', autospec=True, side_effect=analyze):
                for series in groups.values():
                    preprocess_and_cache_worker((base, series))
                    self.assertEqual(series.pipeline_state.preprocess_config_hash,
                                     base.for_series(series).preprocess_config_hash())
                    track_and_cache_worker((base, series))

            restored = {}
            for name, original in groups.items():
                fresh = ImageSeries(name, original.images)
                self.assertTrue(series_cache.load_series_state(fresh, base))
                self.assertTrue(RootTrackingPipeline(base).is_tracking_current(fresh))
                restored[name] = fresh
            rows = RootTrackingPipeline(base).export_statistics(restored).to_dict('records')
            self.assertEqual(rows, [{'group': 'a', 'origins': 3, 'area_limit': 111},
                                    {'group': 'b', 'origins': 6, 'area_limit': 100}])
            self.assertEqual(base.n_clusters, 6)
            self.assertEqual(base.threshold.min_contour_area, 100)
