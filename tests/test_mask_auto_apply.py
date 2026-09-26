"""Automatic mask edits use the tracking stage and remain reversible."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF, QEvent
from PySide6.QtGui import QWheelEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow, ProcessingState, ProcessingContext, ProcessWorker
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.io import mask_io
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline


class MaskAutoApplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = Config(n_clusters=1)
        config.data.input = config.data.output = folder.name
        config.data.detect_barcodes = False
        config.threshold.min_contour_area = 1
        config.threshold.min_contour_length = 1
        with patch.object(MainWindow, '_load_last_folder'):
            self.w = MainWindow(config)
        self.addCleanup(self.w.close)
        self.w._pipeline = RootTrackingPipeline(config)
        self.image = ImageData(datetime(2026, 1, 1), str(Path(folder.name) / 'a.png'), 'a')
        self.image.image = np.zeros((40, 40, 3), np.uint8)
        self.image.process = np.zeros((40, 40), np.uint8)
        self.image.process[10:35, 18:23] = 255
        self.image.positions_x, self.image.positions_y = [20], [5]
        self.series = ImageSeries('a', [self.image])
        self.series.pipeline_state.preprocessed = True
        self.series.pipeline_state.preprocess_config_hash = config.preprocess_config_hash()
        self.w._series_dict = {'a': self.series}
        self.w._current_series = self.series
        self.w._current_image = self.image
        self.w._restore_settings_draft(WorkflowStep.TRACK)
        self.w._workflow_bar.set_current_step(WorkflowStep.TRACK)
        self.panel = self.w._settings_panel
        self.w._auto_preview_action.setChecked(True)
        self.w._sync_auto_apply_controls()
        # Run the real tracking computation synchronously at the thread boundary.
        self.runs = []
        def track(series):
            worker = ProcessWorker(self.w._pipeline, series, config, 'track')
            worker.persist_mask = self.w._mask_refresh_active
            worker._run_track()
            self.runs.append(self.image.total_area)
        worker = patch.object(self.w, '_start_tracking', side_effect=track)
        worker.start()
        self.addCleanup(worker.stop)
        self.w._pipeline.track_and_analyze_series(self.series)
        self.original_area = self.image.total_area
        self.assertGreater(self.original_area, 0)

    def paint(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[20:, :] = 255
        self.w._image_viewer.set_mask_data(self.series.user_mask, mask)
        self.w._on_mask_modified()
        return mask

    def test_auto_mask_updates_tracking_and_reset_reuses_preprocessing(self):
        processed = self.image.process
        mask = self.paint()
        QTest.qWait(50)
        np.testing.assert_array_equal(self.series.user_mask, mask)
        np.testing.assert_array_equal(mask_io.load_mask(self.series, self.w._config), mask)
        self.assertEqual(len(self.runs), 1)
        self.assertLess(self.image.total_area, self.original_area)
        self.assertIs(self.image.process, processed)
        self.assertTrue(self.series.pipeline_state.preprocessed)
        self.assertTrue(self.panel._apply_btn.isHidden())
        self.assertTrue(self.panel._discard_btn.isHidden())
        self.assertFalse(self.panel._auto_reset_btn.isHidden())
        self.assertTrue(self.panel._auto_reset_btn.isEnabled())
        self.panel._auto_reset_btn.click()
        self.assertEqual(self.image.total_area, self.original_area)
        self.assertFalse(np.any(self.series.user_mask))
        self.assertFalse(self.panel._auto_reset_btn.isEnabled())
        self.assertFalse(mask_io.get_mask_path(self.series, self.w._config).exists())

    def test_manual_mode_stages_edits_until_enabled(self):
        self.w._auto_preview_action.setChecked(False)
        mask = self.paint()
        self.w._apply_auto_settings()
        self.assertIsNone(self.series.user_mask)
        self.assertEqual(self.image.total_area, self.original_area)
        self.assertFalse(self.panel._apply_btn.isHidden())
        self.assertFalse(self.panel._discard_btn.isHidden())
        self.assertTrue(self.panel._auto_reset_btn.isHidden())
        self.w._auto_preview_action.setChecked(True)
        self.w._apply_auto_settings()
        np.testing.assert_array_equal(self.series.user_mask, mask)

    def test_clear_and_reset_restore_existing_mask(self):
        baseline = np.zeros((40, 40), np.uint8)
        baseline[30:, :] = 255
        self.series.user_mask = baseline.copy()
        self.series.working_mask = baseline.copy()
        self.w._image_viewer.set_mask_data(self.series.user_mask, self.series.working_mask)
        self.w._on_mask_erase_all()
        self.w._apply_auto_settings()
        self.assertFalse(np.any(self.series.user_mask))
        self.panel._auto_reset_btn.click()
        np.testing.assert_array_equal(self.series.user_mask, baseline)

    def test_busy_tracking_defers_latest_mask_and_navigation_does_not_start_worker(self):
        mask = self.paint()
        self.w._state = ProcessingState.TRACKING
        self.w._apply_auto_settings()
        self.assertIsNone(self.series.user_mask)
        self.w._state = ProcessingState.IDLE
        self.w._apply_auto_settings(evaluate=False)
        np.testing.assert_array_equal(self.series.user_mask, mask)
        self.assertEqual(self.runs, [])
        self.assertFalse(self.panel._is_dirty)

    def test_mask_drawing_is_blocked_while_tracking_and_resumes_afterwards(self):
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        self.w.show()
        viewer.show()
        self.panel.show()
        self.app.processEvents()
        self.panel._mask_controls._tools.button(1).click()
        position = viewer._view.mapFromScene(QPointF(20, 25))
        self.w._lock_ui()
        QTest.mouseClick(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertIsNone(viewer.get_working_mask())
        self.w._unlock_ui()
        QTest.mouseClick(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertTrue(np.any(viewer.get_working_mask()))

    def test_auto_mask_refresh_allows_next_stroke_and_queues_latest_mask(self):
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        self.w.show()
        viewer.show()
        self.panel.show()
        self.app.processEvents()
        self.panel._mask_controls._tools.button(1).click()
        self.panel._mask_controls._diameter.setValue(3)
        self.paint()
        def start(series):
            ProcessingContext(self.w, ProcessingState.TRACKING, series).__enter__()
        with patch.object(self.w, '_start_tracking', side_effect=start):
            self.w._apply_auto_settings()
        self.assertTrue(self.panel._mask_controls.isEnabled())
        self.assertFalse(self.panel._min_contour_area_spin.isEnabled())
        first_applied = self.series.user_mask.copy()
        self.app.processEvents()
        # Keep the small brush target precise when mapping to viewport pixels.
        viewer._set_zoom(1.0)
        position = viewer._view.mapFromScene(QPointF(10, 10))
        QTest.mouseClick(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertGreater(self.series.working_mask[10, 10], 0)
        np.testing.assert_array_equal(self.series.user_mask, first_applied)
        latest = self.series.working_mask.copy()
        self.w._apply_auto_settings()  # In-flight results must not change inputs.
        np.testing.assert_array_equal(self.series.user_mask, first_applied)
        self.w._pipeline.track_and_analyze_series(self.series)
        self.w._on_tracking_finished(True, '')
        np.testing.assert_array_equal(self.series.working_mask, latest)
        self.w._apply_auto_settings()
        np.testing.assert_array_equal(self.series.user_mask, latest)

    def test_cancelled_refresh_still_persists_committed_mask(self):
        mask = self.paint()
        self.series.user_mask = mask.copy()
        worker = ProcessWorker(self.w._pipeline, self.series, self.w._config, 'track')
        worker.persist_mask = True
        worker.cancel()
        worker._run_track()
        np.testing.assert_array_equal(mask_io.load_mask(self.series, self.w._config), mask)
        self.assertEqual(self.image.total_area, self.original_area)

    def test_right_drag_temporarily_restores_then_returns_to_previous_action(self):
        self.w._auto_preview_action.setChecked(False)
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        self.w.show()
        viewer.show()
        self.panel.show()
        self.app.processEvents()
        controls = self.panel._mask_controls
        controls._diameter.setValue(8)
        for index, restore_tool in ((1, 'brush_eraser'), (2, 'rect_eraser')):
            for original_action in (0, 1):
                with self.subTest(shape=index, action=original_action):
                    controls._tools.button(index).click()
                    controls._operations.button(original_action).click()
                    original_tool = viewer._mask_tool
                    mask = np.full((40, 40), 255, np.uint8)
                    self.series.user_mask = mask.copy()
                    self.series.working_mask = mask.copy()
                    viewer.set_mask_data(self.series.user_mask, self.series.working_mask)
                    start = viewer._view.mapFromScene(QPointF(12, 12))
                    end = viewer._view.mapFromScene(QPointF(25, 25))
                    QTest.mousePress(viewer._view.viewport(), Qt.MouseButton.RightButton, pos=start)
                    self.assertEqual(viewer._mask_tool.value, restore_tool)
                    self.assertTrue(controls._operations.button(1).isChecked())
                    QTest.mouseMove(viewer._view.viewport(), end)
                    QTest.mouseRelease(viewer._view.viewport(), Qt.MouseButton.RightButton, pos=end)
                    self.assertEqual(self.series.working_mask[18, 18], 0)
                    self.assertEqual(self.series.working_mask[1, 1], 255)
                    self.assertEqual(viewer._mask_tool, original_tool)
                    self.assertTrue(controls._operations.button(original_action).isChecked())
                    self.assertEqual(controls._diameter.value(), 8)

    def test_alt_temporarily_pans_without_painting_and_restores_tool(self):
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        viewer.show()
        self.w.show()
        self.app.processEvents()
        controls = self.panel._mask_controls
        view = viewer._view
        viewer.fit_in_view()
        for shape in (1, 2):
            with self.subTest(shape=shape):
                controls._tools.button(shape).click()
                original_tool = viewer._mask_tool
                original_size = viewer._brush_size
                mask = np.zeros((40, 40), np.uint8)
                viewer.set_mask_data(None, mask.copy())
                # Alt must work immediately even while the tool button owns focus.
                QTest.keyPress(controls._tools.button(shape), Qt.Key.Key_Alt)
                self.assertEqual(viewer._mask_tool.value, 'move')
                self.assertTrue(controls._tools.button(0).isChecked())
                start = view.viewport().rect().center()
                center_before = view.mapToScene(start)
                QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.AltModifier, start)
                end = start + QPoint(35, 20)
                move = QMouseEvent(QEvent.Type.MouseMove, QPointF(end),
                                   QPointF(view.viewport().mapToGlobal(end)),
                                   Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.AltModifier)
                self.app.sendEvent(view.viewport(), move)
                # Releasing Alt mid-drag must not turn the remaining drag into paint.
                QTest.keyRelease(view, Qt.Key.Key_Alt)
                self.assertEqual(viewer._mask_tool.value, 'move')
                QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton,
                                   pos=start + QPoint(35, 20))
                self.assertNotEqual(view.mapToScene(start), center_before)
                self.assertEqual(viewer._mask_tool, original_tool)
                self.assertEqual(viewer._brush_size, original_size)
                self.assertTrue(controls._tools.button(shape).isChecked())
                np.testing.assert_array_equal(viewer._working_mask, mask)
                self.assertFalse(viewer._drawing)
                self.assertIsNone(viewer._mask_draw_button)
                QTest.keyPress(view, Qt.Key.Key_Alt)
                QTest.keyRelease(view, Qt.Key.Key_Alt)
                self.assertEqual(viewer._mask_tool, original_tool)

    def test_alt_pan_restores_on_window_deactivation(self):
        controls = self.panel._mask_controls
        viewer = self.w._image_viewer
        controls._tools.button(1).click()
        original = viewer._mask_tool
        QTest.keyPress(viewer._view, Qt.Key.Key_Alt)
        self.assertTrue(controls._tools.button(0).isChecked())
        self.app.sendEvent(self.w, QEvent(QEvent.Type.WindowDeactivate))
        self.assertEqual(viewer._mask_tool, original)
        self.assertTrue(controls._tools.button(1).isChecked())

    def test_shift_scroll_changes_brush_diameter_without_zoom_and_respects_limits(self):
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        controls = self.panel._mask_controls
        controls._tools.button(1).click()
        view = viewer._view
        position = view.viewport().rect().center()
        zoom = view.transform().m11()
        def scroll(delta, modifiers=Qt.KeyboardModifier.ShiftModifier):
            event = QWheelEvent(QPointF(position), QPointF(view.viewport().mapToGlobal(position)),
                                QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                                modifiers, Qt.ScrollPhase.NoScrollPhase, False)
            view.wheelEvent(event)
        scroll(120)
        self.assertEqual(controls._diameter.value(), 105)
        self.assertEqual(viewer._brush_size, 105)
        scroll(-120)
        self.assertEqual(controls._diameter.value(), 100)
        scroll(60)
        self.assertEqual(controls._diameter.value(), 100)
        scroll(60)
        self.assertEqual(controls._diameter.value(), 105)
        controls._operations.button(1).click()
        controls._diameter.setValue(998)
        scroll(120)
        self.assertEqual(controls._diameter.value(), 999)
        controls._diameter.setValue(2)
        scroll(-120)
        self.assertEqual(controls._diameter.value(), 1)
        self.assertEqual(viewer._brush_size, 1)
        self.assertEqual(view.transform().m11(), zoom)
        scroll(120, Qt.KeyboardModifier.NoModifier)
        self.assertGreater(view.transform().m11(), zoom)
        self.assertFalse(self.panel._is_dirty)

    def test_completed_mask_stroke_does_not_wait_for_settings_debounce(self):
        self.paint()
        QTest.qWait(50)
        self.assertEqual(len(self.runs), 1)
        self.assertLess(self.image.total_area, self.original_area)

    def test_worker_publishes_calculated_image_before_export_and_cache(self):
        worker = ProcessWorker(self.w._pipeline, self.series, self.w._config, 'track')
        worker.persist_mask = True
        events = []
        worker.image_ready.connect(lambda image: events.append(('ready', image)))
        real_save = self.w._pipeline.exporter.save_image
        def save(image, path):
            self.assertTrue(events)
            self.assertIs(events[0][1], self.image)
            self.assertIsNotNone(self.image.image_annotated)
            return real_save(image, path)
        with patch.object(self.w._pipeline.exporter, 'save_image', side_effect=save):
            worker._run_track()
        self.assertEqual(events, [('ready', self.image)])

    def test_calculated_image_is_displayed_while_tracking_is_still_running(self):
        self.w._mask_refresh_active = True
        self.w._state = ProcessingState.TRACKING
        self.image.image_annotated = np.full((40, 40, 3), (0, 0, 255), np.uint8)
        self.w._on_tracking_image_ready(self.image)
        shown = self.w._image_viewer._pixmap_item.pixmap().toImage()
        self.assertEqual(shown.pixelColor(20, 20).red(), 255)
        self.assertEqual(self.w._state, ProcessingState.TRACKING)
        self.w._state = ProcessingState.IDLE

    def test_geometry_change_invalidates_mask_reset_baseline(self):
        self.paint()
        self.w._apply_auto_settings()
        self.w._invalidate_preprocessing(self.series)
        self.w._sync_auto_apply_controls()
        self.assertFalse(self.panel._auto_reset_btn.isEnabled())

    def test_removal_buttons_follow_loaded_cleared_and_restored_masks(self):
        self.w._auto_preview_action.setChecked(False)
        viewer = self.w._image_viewer
        viewer.set_image(self.image.image)
        self.w.show()
        self.app.processEvents()
        viewer._set_zoom(3)
        controls = self.panel._mask_controls
        controls._tools.button(1).click()
        restore = controls._operations.button(1)
        clear = controls._clear_button
        self.assertFalse(restore.isEnabled())
        self.assertFalse(clear.isEnabled())
        mask = self.paint()
        self.assertTrue(restore.isEnabled())
        self.assertTrue(clear.isEnabled())
        clear.click()
        self.assertFalse(restore.isEnabled())
        self.assertFalse(clear.isEnabled())
        # Reload an applied mask, then erase all of it with a single stroke.
        viewer.set_mask_data(mask.copy(), None)
        self.assertTrue(restore.isEnabled())
        self.assertTrue(clear.isEnabled())
        restore.click()
        controls._diameter.setValue(100)
        position = viewer._view.mapFromScene(QPointF(20, 20))
        QTest.mouseClick(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertFalse(np.any(viewer.get_working_mask()))
        self.assertFalse(restore.isEnabled())
        self.assertFalse(clear.isEnabled())
        # The committed mask is still present, but availability follows the draft.
        self.assertTrue(np.any(viewer._applied_mask))
