"""Authoritative RSML survives tracking, cache resets, and reloads."""
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch
import math
import numpy as np
import cv2

from root_tracker.config import Config
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.io.rsml import read_rsml
from root_tracker.io.rsml_replacement import replace_roots, load_replacement, refresh_statistics, render_replacement
from root_tracker.io import series_cache

RAW = b'''<rsml xmlns:v="urn:vendor"><metadata><unit>pixel</unit><resolution>1</resolution></metadata>
<scene><plant id="1"><root id="main"><geometry><polyline>
<point x="10" y="10"/><point x="10" y="20"/></polyline></geometry>
<v:unknown>preserve</v:unknown><root id="lateral"><geometry><polyline>
<point x="10" y="15"/><point x="13" y="19"/></polyline></geometry></root>
</root></plant></scene></rsml>'''


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config = Config(n_clusters=1)
        self.config.data.input = str(self.base)
        self.config.data.output = str(self.base / 'output')
        self.source = self.base / 'external.rsml'
        self.source.write_bytes(RAW)
        self.image = ImageData(datetime(2026, 1, 1), str(self.base / 'p.01-01-26.png'), 'p')
        self.image.image = np.zeros((40, 40, 3), np.uint8)
        self.image.process = np.zeros((40, 40), np.uint8)
        cv2.imwrite(self.image.path, self.image.image)
        self.image.total_area = 999
        self.image.plant_length = [999]
        self.series = ImageSeries('p', [self.image])

    def replace(self, image=None):
        replace_roots(image or self.image, read_rsml(self.source))

    def test_replacement_drives_metrics_and_survives_clear(self):
        self.replace()
        self.assertEqual(self.image.total_length, 15.)
        self.assertEqual(self.image.plant_length, [15.])
        self.assertEqual(self.image.longest, [10.])
        self.assertIsNone(self.image.total_area)
        records = refresh_statistics(self.series, [])
        self.assertEqual(records[0]['plant_main_root_depth'], 10.)
        self.assertEqual(records[0]['plant_root_count'], 2)
        self.assertEqual(records[0]['root_source'], 'rsml')
        self.assertEqual(records[0]['rsml_plant_id'], '1')
        self.image.clear_preprocessing_results()
        self.assertEqual(self.image.rsml_document.source_bytes, RAW)
        self.assertEqual(self.image.total_length, 15.)
        self.assertGreater(np.count_nonzero(render_replacement(self.image)), 0)
        self.assertEqual(self.image.copy_for_editing().rsml_document.source_bytes, RAW)

    def test_persists_outside_cache_and_original_document(self):
        self.replace()
        self.source.unlink()
        series_cache.save_series(self.series, self.config)
        import shutil
        shutil.rmtree(self.base / '.root_tracker_cache')
        fresh = ImageData(self.image.date, self.image.path, 'p')
        self.assertTrue(load_replacement(fresh))
        self.assertEqual(fresh.rsml_document.source_bytes, RAW)
        self.assertEqual(fresh.total_length, 15.)
        self.assertGreater(np.count_nonzero(render_replacement(fresh)), 0)
        pipeline = RootTrackingPipeline(self.config)
        loaded = pipeline.load_images()['p'][0]
        self.assertEqual(loaded.total_length, 15.)
        self.assertEqual(loaded.rsml_document.source_bytes, RAW)

    def test_tracking_skips_replacements_and_csv_ignores_stale_cache_stats(self):
        self.replace()
        self.series.pipeline_state.tracked = True
        self.series.pipeline_state.last_statistics = [{'image_path': self.image.path, 'plant_id': 1, 'plant_total_length': 999}]
        pipeline = RootTrackingPipeline(self.config)
        with patch.object(pipeline.thresholder, 'threshold', side_effect=AssertionError('must not track RSML')):
            records = pipeline.track_and_analyze_series(self.series, save_images=False)
        self.assertEqual(records[0]['plant_total_length'], 15.)
        frame = pipeline.export_statistics({'p': self.series})
        self.assertEqual(frame.iloc[0]['plant_total_length'], 15.)
        self.assertTrue(math.isnan(float(frame.iloc[0]['image_total_area'])) if frame.iloc[0]['image_total_area'] is not None else True)

    def test_adjacent_growth_rates_use_replacement_values(self):
        before = ImageData(self.image.date - timedelta(days=1), 'before.png', 'p')
        after = ImageData(self.image.date + timedelta(days=1), 'after.png', 'p')
        self.series.images = [before, self.image, after]
        self.replace()
        cached = [dict(image_path=before.path, plant_id=1, plant_total_length=5., plant_main_root_length=5.),
                  dict(image_path=self.image.path, plant_id=1, plant_total_length=999.),
                  dict(image_path=after.path, plant_id=1, plant_total_length=30., plant_main_root_length=20.,
                       plant_total_length_RGR=999., image_new_area=999, image_area_change=999.)]
        records = refresh_statistics(self.series, cached)
        self.assertAlmostEqual(records[1]['plant_total_length_RGR'], math.log(3))
        self.assertAlmostEqual(records[2]['plant_total_length_RGR'], math.log(2))
        self.assertIsNone(records[2]['image_new_area'])
        self.assertIsNone(records[2]['image_area_change'])

    def test_invalid_replacement_does_not_destroy_existing_one(self):
        self.replace()
        self.source.write_bytes(RAW.replace(b'x="13"', b'x="1300"'))
        with self.assertRaises(ValueError):
            self.replace()
        fresh = ImageData(self.image.date, self.image.path, 'p')
        load_replacement(fresh)
        self.assertEqual(fresh.rsml_document.source_bytes, RAW)

    def test_cache_reload_cannot_restore_old_native_measurements(self):
        series_cache.save_series(self.series, self.config)
        self.replace()
        series_cache.load_series(self.series, self.config)
        self.assertEqual(self.image.total_length, 15.)
        self.assertIsNone(self.image.total_area)

    def test_reexport_uses_original_rsml_even_without_native_tracking(self):
        self.replace()
        from root_tracker.io.rsml_export import export_series_rsml
        self.series.clear_preprocessing_results()
        files = export_series_rsml(self.series, self.config, self.base / 'export', image_index=0)
        self.assertEqual(files[0].read_bytes(), RAW)

    def test_mixed_tracking_uses_new_rates_but_not_imported_tracking_seeds(self):
        native = ImageData(self.image.date + timedelta(days=1), str(self.base / 'next.png'), 'p')
        native.image = np.zeros((180, 180, 3), np.uint8)
        native.process = np.zeros((180, 180), np.uint8)
        cv2.line(native.process, (80, 35), (80, 160), 255, 5)
        native.positions_x, native.positions_y, native.green_areas = [80], [20], [100]
        self.series.images.append(native)
        self.replace()
        self.image.colored_samples = {0: {(10, 10)}}  # stale pre-replacement tracker data
        pipeline = RootTrackingPipeline(self.config)
        observed = []
        original_link = pipeline.linker.link_corners
        def link(*args, **kwargs):
            observed.append(kwargs.get('previous_colored_samples'))
            return original_link(*args, **kwargs)
        with patch.object(pipeline.linker, 'link_corners', side_effect=link):
            records = pipeline.track_and_analyze_series(self.series, save_images=False)
        self.assertEqual(observed, [{}])
        self.assertEqual(records[0]['plant_total_length'], 15.)
        self.assertGreater(records[1]['plant_total_length'], 0)
        self.assertAlmostEqual(records[1]['plant_total_length_RGR'], math.log(records[1]['plant_total_length'] / 15))
        self.assertIsNone(native.new_area)

    def test_move_keeps_replacement_attached_to_image(self):
        self.replace()
        from root_tracker.io.rsml_replacement import move_image_with_replacement
        destination = self.base / 'aside' / Path(self.image.path).name
        destination.parent.mkdir()
        move_image_with_replacement(self.image.path, str(destination))
        fresh = ImageData(self.image.date, str(destination), 'p')
        self.assertTrue(load_replacement(fresh))
        self.assertEqual(fresh.total_length, 15.)
        self.assertFalse(Path(self.image.path).exists())

    def test_all_replaced_series_is_current_without_native_preprocessing(self):
        self.replace()
        self.assertTrue(RootTrackingPipeline(self.config).is_tracking_current(self.series))

    def test_preprocessing_and_registration_do_not_touch_replaced_frames(self):
        self.replace()
        pipeline = RootTrackingPipeline(self.config)
        before = self.image.image.copy()
        # A replaced frame needs no automatic image-processing inputs.
        with patch.object(pipeline, 'preprocess_image', side_effect=AssertionError('must skip replacement')):
            pipeline.preprocess_series(self.series)
        pipeline.register_series(self.series)
        np.testing.assert_array_equal(self.image.image, before)
        self.assertEqual(self.image.total_length, 15.)

    def test_summary_includes_plants_from_rsml_beyond_configured_count(self):
        self.source.write_bytes(RAW.replace(b'</scene>', b'<plant id="2"/></scene>'))
        self.replace()
        pipeline = RootTrackingPipeline(self.config)
        records = refresh_statistics(self.series, [])
        summary = pipeline.exporter.generate_summary_report(records)
        self.assertEqual(summary['total_plants'], 2)
        self.assertEqual(summary['plant_1']['final_total_length'], 15.)
        self.assertEqual(summary['plant_2']['final_total_length'], 0.)

    def test_main_root_depth_uses_same_primary_axis_as_main_length(self):
        self.source.write_bytes(RAW.replace(b'x="13" y="19"', b'x="13" y="35"'))
        self.replace()
        record = refresh_statistics(self.series, [])[0]
        self.assertEqual(record['plant_main_root_length'], 10.)
        self.assertEqual(record['plant_main_root_depth'], 10.)
        self.assertGreater(record['plant_total_length'], 30.)
