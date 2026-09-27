"""Working settings and cached results survive restarting the GUI."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.presets import PresetStore
from root_tracker.io import series_cache
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline


class SessionRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_restart_restores_working_settings_and_recognizes_cached_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat)
            store = PresetStore(root / 'presets')
            config = Config()
            config.data.input = directory
            store.save('Defaults', config)
            with patch('root_tracker.gui.main_window.QSettings', return_value=settings), \
                    patch.object(MainWindow, '_get_preset_store', return_value=store), \
                    patch.object(MainWindow, '_reload_images'):
                window = MainWindow()
                working = window._config
                working.load_roi = (.5, .5, .8, .9, 5.)
                working.green.hsv_lower = (30, 170, 55)
                working.registration.margin_ratio = .3
                working.threshold.min_contour_length = 23
                series = ImageSeries('plant', [ImageData(datetime(2026, 4, 27), 'plant.26-04-27.JPG', 'plant')])
                series.processing_settings = working.processing_settings()
                state = series.pipeline_state
                state.preprocessed = state.tracked = True
                state.preprocess_config_hash = working.preprocess_config_hash()
                state.tracking_config_hash = working.tracking_config_hash()
                state.last_statistics = [{'group': 'plant', 'length': 42}]
                series_cache.save_series(series, working)
                window._series_dict = {'plant': series}
                window.close()

                reopened = MainWindow()
                try:
                    restored = reopened._config
                    self.assertEqual(reopened._session_group_settings['plant'], series.processing_settings)
                    self.assertEqual(restored.load_roi, working.load_roi)
                    self.assertEqual(tuple(restored.green.hsv_lower), working.green.hsv_lower)
                    self.assertEqual(restored.threshold.min_contour_length, 23)
                    fresh = ImageSeries('plant', series.images)
                    self.assertTrue(series_cache.load_series_state(fresh, restored))
                    self.assertTrue(RootTrackingPipeline(restored).is_tracking_current(fresh))
                    self.assertEqual(fresh.pipeline_state.last_statistics, [{'group': 'plant', 'length': 42}])
                    self.assertIsNone(store.load('Defaults').load_roi)
                finally:
                    reopened.close()

                with patch.object(MainWindow, '_load_last_folder'):
                    explicit = MainWindow(Config(n_clusters=2))
                    self.assertEqual(explicit._config.n_clusters, 2)
                    explicit.close()

    def test_configuration_serialization_preserves_cache_identity(self):
        config = Config(rotation=180)
        restored = Config.from_dict(config.to_dict())
        restored.rotation = 180.0
        self.assertEqual(config.preprocess_config_hash(), restored.preprocess_config_hash())
        self.assertEqual(config.tracking_config_hash(), restored.tracking_config_hash())

    def test_invalid_or_missing_dataset_session_falls_back_to_preset(self):
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat)
            store = PresetStore(root / 'presets')
            config = Config(n_clusters=9)
            config.data.input = directory
            store.save('Defaults', config)
            missing = Config()
            missing.data.input = str(root / 'missing')
            for session in ('invalid json', json.dumps({'config': missing.to_dict()})):
                with self.subTest(session=session):
                    settings.setValue('last_session', session)
                    with patch('root_tracker.gui.main_window.QSettings', return_value=settings), \
                            patch.object(MainWindow, '_get_preset_store', return_value=store), \
                            patch.object(MainWindow, '_reload_images'):
                        window = MainWindow()
                        try:
                            self.assertEqual(window._config.n_clusters, 9)
                            self.assertEqual(window._active_preset_name, 'Defaults')
                        finally:
                            window.close()
