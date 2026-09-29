"""Depth visibility redraws cached results without scheduling processing."""
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import cv2
from PySide6.QtCore import QCoreApplication, QEvent, QPointF
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.image_viewer import ImageViewer
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
        self.assertIsNotNone(viewer._depth_overlay)
        self.assertIsNotNone(viewer._link_overlay)
        config_hash = self.config.tracking_config_hash()
        annotated = self.image.image_annotated.copy()
        panel._show_max_root_depth_cb.setChecked(False)
        self.assertIsNone(viewer._depth_overlay)
        self.assertIsNotNone(viewer._link_overlay)
        self.assertEqual(before, viewer._pixmap_item.pixmap().toImage())
        self.assertEqual(transform, viewer._view.transform())
        self.assertFalse(panel.is_dirty())
        self.assertFalse(window._auto_apply_timer.isActive())
        self.assertEqual(config_hash, self.config.tracking_config_hash())
        np.testing.assert_array_equal(annotated, self.image.image_annotated)
        panel._min_contour_area_spin.setValue(2)
        draft = panel.get_current_values()
        panel._show_max_root_depth_cb.setChecked(True)
        self.assertIsNotNone(viewer._depth_overlay)
        self.assertEqual(before, viewer._pixmap_item.pixmap().toImage())
        self.assertEqual(draft, panel.get_current_values())
        self.assertTrue(panel.is_dirty())

    def test_marker_stays_bright_at_fractional_positions_during_pan_and_zoom(self):
        viewer = ImageViewer()
        viewer.resize(500, 500)
        viewer.show()
        self.addCleanup(viewer.close)
        viewer.set_image(np.zeros((1000, 1000, 3), np.uint8))
        mask = np.zeros((1000, 1000), np.uint8)
        cv2.line(mask, (100, 200), (900, 200), 255, 1)
        cv2.line(mask, (900, 200), (900, 800), 255, 3)
        cv2.line(mask, (900, 800), (100, 800), 255, 1)
        viewer.set_root_depth_pixels(np.column_stack(np.nonzero(mask)))
        self.app.processEvents()
        view = viewer._view
        for scale in (.08, .2, .37, 1., 2., 5.):
            for point in (QPointF(500.5, 200.5), QPointF(500.5, 800.5), QPointF(900.5, 500.5)):
                for phase in (0., .25, .5, .75):
                    center = view.viewport().rect().center()
                    viewer._set_viewport_transform(QTransform(
                        scale, 0, 0, scale, center.x() - scale * point.x() + phase,
                        center.y() - scale * point.y() + phase))
                    for pan in (-2, 0, 2):
                        with self.subTest(scale=scale, point=point, phase=phase, pan=pan):
                            view.verticalScrollBar().setValue(pan)
                            self.app.processEvents()
                            snapshot = view.viewport().grab().toImage()
                            screen = view.mapFromScene(point) * snapshot.devicePixelRatio()
                            brightness = max(snapshot.pixelColor(x, y).red()
                                             for x in range(screen.x()-2, screen.x()+3)
                                             for y in range(screen.y()-2, screen.y()+3))
                            self.assertGreaterEqual(brightness, 240)
                            # Horizontal and vertical sections must both stay
                            # two screen pixels wide, even at maximum zoom.
                            if point.x() == 900.5:
                                colors = [snapshot.pixelColor(x, screen.y()).red()
                                          for x in range(screen.x()-4, screen.x()+5)]
                            else:
                                colors = [snapshot.pixelColor(screen.x(), y).red()
                                          for y in range(screen.y()-4, screen.y()+5)]
                            width = sum(colors) / 255
                            self.assertAlmostEqual(width, 2 * snapshot.devicePixelRatio(), delta=.15)
        viewer.set_image(np.zeros((1000, 1000, 3), np.uint8), preserve_view=True)
        self.assertIsNone(viewer._depth_overlay)

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

    def test_connectors_survive_cache_reload_and_old_cache_recovery(self):
        hidden, pixels = self.image.root_link_preview(self.image.tracking_preview(False))
        self.assertGreater(len(pixels), 0)
        np.testing.assert_array_equal(hidden[pixels[:, 0], pixels[:, 1]], pixels[:, 2:])
        underlay = self.image.root_link_background.copy()
        cache = series_cache.save_series(self.series, self.config)
        self.image.clear_tracking_results()
        self.assertIsNone(self.image.root_link_background)
        self.assertTrue(series_cache.load_series(self.series, self.config))
        np.testing.assert_array_equal(underlay, self.image.root_link_background)
        np.testing.assert_array_equal(underlay, self.image.copy_for_editing().root_link_background)
        # Earlier caches only store the diagnostic marker pixels.
        with np.load(cache, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if not key.startswith('root_link_background_')}
        np.savez(cache, **arrays)
        self.image.clear_tracking_results()
        self.assertTrue(series_cache.load_series(self.series, self.config))
        actual, actual_pixels = self.image.root_link_preview(self.image.tracking_preview(False))
        np.testing.assert_array_equal(hidden, actual)
        np.testing.assert_array_equal(pixels, actual_pixels)

    def test_connectors_survive_manual_edit_reload_and_result_invalidation(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import load_replacement, render_replacement
        underlay = self.image.root_link_background.copy()
        edit_roots(self.image, editable_document(self.image), {0}, plant_index=0)
        fresh = ImageData(self.image.date, self.image.path, self.image.barcode)
        self.assertTrue(load_replacement(fresh))
        np.testing.assert_array_equal(underlay, fresh.root_link_background)
        fresh.clear_preprocessing_results()
        np.testing.assert_array_equal(underlay, fresh.root_link_background)
        rendered = render_replacement(fresh, show_max_root_depth=False)
        hidden, pixels = fresh.root_link_preview(rendered)
        self.assertGreater(len(pixels), 0)
        np.testing.assert_array_equal(hidden[pixels[:, 0], pixels[:, 1]], pixels[:, 2:])
        # Connector removal must preserve root highlights and export pixels.
        np.testing.assert_array_equal(hidden[100, 50], rendered[100, 50])
        np.testing.assert_array_equal(rendered[pixels[:, 0], pixels[:, 1]], 255)

    def test_root_connector_stays_two_pixels_wide_when_panning_and_zooming(self):
        viewer = ImageViewer()
        viewer.resize(500, 500)
        viewer.show()
        self.addCleanup(viewer.close)
        viewer.set_image(np.zeros((1000, 1000, 3), np.uint8))
        mask = np.zeros((1000, 1000), np.uint8)
        cv2.line(mask, (500, 100), (500, 900), 255, 1)
        viewer.set_root_link_pixels(np.column_stack(np.nonzero(mask)))
        self.app.processEvents()
        view = viewer._view
        point = QPointF(500.5, 500.5)
        for scale in (.08, .2, .37, 1., 2., 5.):
            for phase in (0., .25, .5, .75):
                center = view.viewport().rect().center()
                viewer._set_viewport_transform(QTransform(
                    scale, 0, 0, scale, center.x() - scale * point.x() + phase,
                    center.y() - scale * point.y() + phase))
                for pan in (-2, 0, 2):
                    with self.subTest(scale=scale, phase=phase, pan=pan):
                        view.horizontalScrollBar().setValue(pan)
                        self.app.processEvents()
                        snapshot = view.viewport().grab().toImage()
                        screen = view.mapFromScene(point) * snapshot.devicePixelRatio()
                        colors = [snapshot.pixelColor(x, screen.y()).red()
                                  for x in range(screen.x()-4, screen.x()+5)]
                        self.assertGreaterEqual(max(colors), 240)
                        self.assertAlmostEqual(sum(colors) / 255, 2 * snapshot.devicePixelRatio(), delta=.15)
        viewer.set_image(None)
        self.assertIsNone(viewer._link_overlay)

    def test_edited_native_roots_use_the_overlay_and_clear_it_when_leaving_track(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import render_replacement, replacement_depth_pixels
        document = editable_document(self.image)
        edit_roots(self.image, document, {0}, plant_index=0)
        hidden = render_replacement(self.image, show_max_root_depth=False)
        pixels = replacement_depth_pixels(self.image)
        expected = hidden.copy()
        expected[pixels[:, 0], pixels[:, 1]] = 255
        np.testing.assert_array_equal(expected, render_replacement(self.image))
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(self.config)
        def dispose_window():
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.addCleanup(dispose_window)
        window._workflow_bar.blockSignals(True)
        window._workflow_bar.set_current_step(WorkflowStep.TRACK)
        window._current_image = self.image
        window._display_image(self.image)
        viewer = window._image_viewer
        self.assertIsNotNone(viewer._depth_overlay)
        self.assertIsNotNone(viewer._link_overlay)
        bitmap = viewer._pixmap_item.pixmap().toImage()
        self.config.gui.show_max_root_depth = False
        window._display_image(self.image, preserve_view=True)
        self.assertIsNone(viewer._depth_overlay)
        self.assertEqual(bitmap, viewer._pixmap_item.pixmap().toImage())
        self.config.gui.show_max_root_depth = True
        window._display_image(self.image, preserve_view=True)
        self.assertIsNotNone(viewer._depth_overlay)
        window._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
        window._display_image(self.image)
        self.assertIsNone(viewer._depth_overlay)
        self.assertIsNone(viewer._link_overlay)
