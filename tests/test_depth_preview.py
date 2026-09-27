"""Depth visibility redraws cached results without scheduling processing."""
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.io import series_cache
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline


class DepthPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.config = Config(n_clusters=1)
        self.config.data.input = self.config.data.output = folder.name
        self.config.threshold.min_contour_area = self.config.threshold.min_contour_length = 1
        self.image = ImageData(datetime(2026, 1, 1), folder.name + '/a.png', 'a')
        self.image.image = np.zeros((180, 220, 3), np.uint8)
        self.image.process = np.zeros((180, 220), np.uint8)
        self.image.process[20:155, 48:53] = 255
        self.image.positions_x, self.image.positions_y = [50], [10]
        self.series = ImageSeries('a', [self.image])
        RootTrackingPipeline(self.config).track_and_analyze_series(self.series, save_images=False)

    def test_checkbox_redraws_immediately_and_preserves_pending_edits_and_zoom(self):
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(self.config)
        def dispose_window():
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(dispose_window)
        window._workflow_bar.blockSignals(True)
        window._workflow_bar.set_current_step(WorkflowStep.TRACK)
        window._workflow_bar.blockSignals(False)
        panel = window._settings_panel
        panel.set_step(WorkflowStep.TRACK)
        window._current_image = self.image
        window._display_image(self.image)
        viewer = window._image_viewer
        viewer.zoom_in()
        transform = viewer._view.transform()
        before = viewer._pixmap_item.pixmap().toImage()
        config_hash = self.config.tracking_config_hash()
        annotated = self.image.image_annotated.copy()
        panel._show_max_root_depth_cb.setChecked(False)
        self.assertNotEqual(before, viewer._pixmap_item.pixmap().toImage())
        self.assertEqual(transform, viewer._view.transform())
        self.assertFalse(panel.is_dirty())
        self.assertFalse(window._auto_apply_timer.isActive())
        self.assertEqual(config_hash, self.config.tracking_config_hash())
        np.testing.assert_array_equal(annotated, self.image.image_annotated)
        panel._min_contour_area_spin.setValue(2)
        draft = panel.get_current_values()
        panel._show_max_root_depth_cb.setChecked(True)
        self.assertEqual(before, viewer._pixmap_item.pixmap().toImage())
        self.assertEqual(draft, panel.get_current_values())
        self.assertTrue(panel.is_dirty())

    def test_depth_background_survives_cache_round_trip(self):
        hidden = self.image.tracking_preview(False).copy()
        self.assertFalse(np.array_equal(hidden, self.image.image_annotated))
        self.assertIsNotNone(series_cache.save_series(self.series, self.config))
        self.image.clear_tracking_results()
        self.assertIsNone(self.image.root_depth_background)
        self.assertTrue(series_cache.load_series(self.series, self.config))
        np.testing.assert_array_equal(hidden, self.image.tracking_preview(False))
        copy = self.image.copy_for_editing()
        np.testing.assert_array_equal(hidden, copy.tracking_preview(False))
