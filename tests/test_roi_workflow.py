"""Exercise crop/color drafts through the actual GUI and processing configuration."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow, ProcessingState
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline
from root_tracker.io import mask_io


class RoiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        config = Config(rotation=0, n_clusters=1)
        config.data.input = self.folder.name
        config.data.output = self.folder.name
        config.data.detect_barcodes = False
        self.source = np.zeros((240, 320, 3), np.uint8)
        self.source[20:225, 15:305] = (200, 80, 20)
        green = cv2.cvtColor(np.uint8([[[35, 200, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
        self.source[45:85, 130:160] = green
        cv2.line(self.source, (147, 80), (166, 213), (245, 245, 245), 3)
        self.groups = []
        for name in ('a', 'b'):
            path = str(Path(self.folder.name)/(name+'.png'))
            cv2.imwrite(path, self.source)
            self.groups.append(ImageSeries(name, [ImageData(datetime(2026, 1, 1), path, name)]))
        with patch.object(MainWindow, '_load_last_folder'):
            self.w = MainWindow(config)
        self.addCleanup(self.w.close)
        self.w._pipeline = RootTrackingPipeline(config)
        self.w._series_dict = {s.group: s for s in self.groups}
        self.w._image_tree.set_series(self.w._series_dict)
        self.w._auto_preview_action.setChecked(False)
        # These tests exercise settings/preview behavior; selection now always
        # schedules processing, so isolate the worker boundary explicitly.
        self._worker_patches = {}
        for method in ('_preprocess_group', '_auto_process_for_tracking'):
            worker_patch = patch.object(self.w, method)
            worker_patch.start()
            self._worker_patches[method] = worker_patch
            self.addCleanup(worker_patch.stop)
        self.w._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
        self.w._current_series = self.groups[0]
        self.w._current_image = self.groups[0].images[0]
        self.w._restore_settings_draft()
        self.w._display_image(self.w._current_image)

    def test_manual_hsv_preview_updates_during_drag_and_survives_navigation(self):
        control = self.w._settings_panel._color_control
        control.swatches[0].click()
        picker = control._picker
        QTest.qWait(80)
        viewer = self.w._image_viewer
        before = viewer._pixmap_item.pixmap().toImage()
        # Continuous edits faster than the refresh interval must not starve it.
        for value in (120, 125, 130, 135, 140):
            picker.editors[0].controls[0].setValue(value)
            QTest.qWait(25)
        self.assertNotEqual(viewer._pixmap_item.pixmap().toImage(), before)
        self.w._activate_group_settings(self.groups[1])
        self.assertIsNone(control._picker)
        self.w._activate_group_settings(self.groups[0])
        self.assertEqual(self.w._settings_panel._color_control.values()[0][0], 140)

    def test_full_resolution_mask_and_coordinates_survive_live_preview(self):
        from root_tracker.preprocessing.cropper import ImageCropper
        source = cv2.resize(self.source, (2400, 1800), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(self.w._current_image.path, source)
        self.w._roi_editor.reset()
        self.w._display_image(self.w._current_image)
        control = self.w._settings_panel._color_control
        control.swatch.click()
        viewer = self.w._image_viewer
        before_transform = viewer._view.transform()
        before_center = viewer._view.mapToScene(viewer._view.viewport().rect().center())
        masks = []
        original = ImageCropper.blue_mask

        def capture(cropper, image, **kwargs):
            result = original(cropper, image, **kwargs)
            masks.append(result.copy())
            return result

        with patch.object(ImageCropper, 'blue_mask', capture):
            self.w._roi_editor.refresh()
        np.testing.assert_array_equal(masks[-1], original(ImageCropper(self.w._config), source))
        self.assertLess(viewer._pixmap_item.pixmap().width(), source.shape[1])
        self.assertEqual(viewer._pixmap_item.sceneBoundingRect().width(), source.shape[1])
        self.assertEqual(viewer._scene_to_image_coords(QPointF(2200, 1700)), (2200, 1700))
        self.assertEqual(viewer._view.transform(), before_transform)
        self.assertEqual(viewer._view.mapToScene(viewer._view.viewport().rect().center()), before_center)
        control._picker.reject()
        self.w._roi_editor.refresh()
        self.assertEqual(viewer._pixmap_item.pixmap().width(), source.shape[1])
        self.assertEqual(viewer._view.transform(), before_transform)
        self.assertEqual(viewer._view.mapToScene(viewer._view.viewport().rect().center()), before_center)

    def test_dialog_sampling_hides_dialog_and_enters_single_limit_mode(self):
        control = self.w._settings_panel._color_control
        control.swatch.click()
        control._picker.editors[0].sample_button.click()
        self.assertFalse(control._picker.isVisible())
        self.assertTrue(control.pick_button.isChecked())
        self.assertTrue(self.w._image_viewer._color_picking)

    def test_unchanged_color_values_preserve_existing_cache_hash(self):
        from root_tracker.gui.roi_editor import apply_editor_values
        config = self.w._config
        config.crop.blue_hsv_lower = list(config.crop.blue_hsv_lower)
        config.crop.blue_hsv_upper = list(config.crop.blue_hsv_upper)
        before = config.preprocess_config_hash()
        apply_editor_values(config, self.w._settings_panel.get_current_values())
        self.assertEqual(config.preprocess_config_hash(), before)

    def test_load_box_and_color_drafts_survive_navigation_and_discard(self):
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        self.assertIsNotNone(viewer._crop_overlay)
        box = (.5, .5, .6, .7, 12.)
        viewer.crop_changed.emit(box)
        panel._color_control.swatch.click()
        panel._color_control._picker.editors[0].controls[0].setValue(90)
        self.w._on_group_selected(self.groups[1])
        self.assertIsNone(panel.get_current_values()['load_roi'])
        self.w._on_group_selected(self.groups[0])
        self.assertEqual(panel.get_current_values()['load_roi'], box)
        self.assertEqual(panel.get_current_values()['blue_lower'][0], 90)
        panel._discard_btn.click()
        self.w._roi_editor.refresh()
        self.assertIsNone(panel.get_current_values()['load_roi'])
        self.assertFalse(panel.is_dirty())
        self.assertIsNone(self.w._config.load_roi)

    def test_load_apply_invalidates_results_and_saved_mask_without_running_preprocess(self):
        series = self.groups[0]
        series.pipeline_state.preprocessed = True
        series.user_mask = np.ones((30, 40), np.uint8)*255
        mask_io.save_mask(series, self.w._config)
        panel = self.w._settings_panel
        panel.set_crop((.5, .5, .6, .6, 15.))
        panel._apply_btn.click()
        self.assertEqual(self.w._config.load_roi, (.5, .5, .6, .6, 15.))
        self.assertEqual(self.w._state, ProcessingState.IDLE)
        self.assertIsNone(self.w._worker)
        self.assertFalse(series.pipeline_state.preprocessed)
        self.assertIsNone(mask_io.load_mask(series, self.w._config))
        self.assertFalse(panel.is_dirty())

    def test_local_crop_change_preserves_other_groups_masks(self):
        other = self.groups[1]
        other.pipeline_state.preprocessed = True
        other.user_mask = np.ones((30, 40), np.uint8)*255
        mask_io.save_mask(other, self.w._config)
        self.w._settings_panel.set_crop((.5, .5, .7, .7, 15.))
        self.w._settings_panel._apply_btn.click()
        self.assertIsNotNone(other.user_mask)
        self.assertIsNotNone(mask_io.load_mask(other, self.w._config))
        self.assertTrue(other.pipeline_state.preprocessed)

    def test_preprocess_editor_uses_uncropped_source_and_picks_green(self):
        self.w._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
        self.w._on_step_changed(WorkflowStep.PREPROCESS)
        panel = self.w._settings_panel
        self.assertFalse(hasattr(panel, '_rotation_combo'))
        self.assertFalse(hasattr(panel, '_margin_top_spin'))
        panel._crop_edit_btn.setChecked(True)
        self.w._roi_editor.refresh()
        canvas = self.w._roi_editor._canvas
        self.assertLess(canvas.shape[1], self.source.shape[1])
        self.assertIsNotNone(self.w._image_viewer._crop_overlay)
        y, x = np.argwhere(cv2.inRange(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV), (28, 166, 50), (42, 255, 200)) > 0)[100]
        panel._color_control.swatch.click()
        panel._color_control._picker.editors[0].sample_button.click()
        self.w._roi_editor._sample_color(int(x), int(y))
        lower, upper = panel._color_control.values()
        self.assertLessEqual(lower[0], 35)
        self.assertGreaterEqual(upper[0], 35)
        self.assertTrue(panel._color_control.preview.isChecked())
        self.assertNotEqual(lower, self.w._config.green.hsv_lower)
        self.w._roi_editor.refresh()
        self.assertTrue(panel.is_dirty())

    def show_window(self):
        self.w.show()
        self.w._image_viewer.show()
        self.w._settings_panel.show()
        self.app.processEvents()
        self.w._image_viewer.fit_in_view()

    def enter_preprocess(self):
        self.w._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
        self.w._on_step_changed(WorkflowStep.PREPROCESS)

    def test_preview_updates_draft_crop_without_applying(self):
        self.enter_preprocess()
        panel = self.w._settings_panel
        panel._crop_edit_btn.setChecked(True)
        self.w._roi_editor.refresh()
        shape = self.w._roi_editor._canvas.shape
        box = (.5, .5, .6, .6, 0.)
        panel.set_crop(box)
        panel._crop_edit_btn.setChecked(False)
        self.w._roi_editor.refresh()
        pixmap = self.w._image_viewer._pixmap_item.pixmap()
        self.assertEqual((pixmap.height(), pixmap.width()), (round(shape[0]*.6), round(shape[1]*.6)))
        self.assertEqual(panel.get_current_values()['preprocess_roi'], box)
        self.assertIsNone(self.w._config.preprocess_roi)
        self.assertTrue(panel.is_dirty())

    def test_applied_crop_previews_in_other_groups_before_processing(self):
        self.w._config.preprocess_roi = (.5, .5, .6, .6, 0.)
        self.enter_preprocess()
        self.w._on_group_selected(self.groups[1])
        source = self.w._pipeline.cropper.process(self.source)
        pixmap = self.w._image_viewer._pixmap_item.pixmap()
        self.assertEqual((pixmap.height(), pixmap.width()),
                         (round(source.shape[0]*.6), round(source.shape[1]*.6)))
        self.assertIsNone(self.groups[1].images[0].process)

    def test_apply_runs_worker_with_previewed_crop_and_keeps_it_after_reload(self):
        from root_tracker.io import series_cache
        self.enter_preprocess()
        panel = self.w._settings_panel
        box = (.5, .5, .6, .6, 17.)
        panel.set_crop(box)
        self.w._evaluation_timer.stop()
        self._worker_patches['_preprocess_group'].stop()
        panel._apply_btn.click()
        worker = self.w._worker
        self.assertIsNotNone(worker)
        try:
            for _ in range(500):
                QTest.qWait(10)
                if self.w._state == ProcessingState.IDLE:
                    break
            self.assertEqual(self.w._state, ProcessingState.IDLE)
        finally:
            worker.cancel()
            worker.wait()
        series = self.groups[0]
        self.assertEqual(self.w._config.preprocess_roi, box)
        self.assertTrue(series.pipeline_state.preprocessed)
        pixels = series.images[0].image.copy()
        source = self.w._pipeline.cropper.process(self.source)
        self.assertEqual(pixels.shape[:2], (round(source.shape[0]*.6), round(source.shape[1]*.6)))
        self.w._free_series_arrays(series)
        self.assertTrue(series_cache.load_series(series, self.w._config))
        np.testing.assert_array_equal(series.images[0].image, pixels)

    def test_segmentation_and_picker_keep_current_processed_crop(self):
        data = self.w._current_image
        self.w._pipeline.preprocess_image(data)
        self.enter_preprocess()
        self.show_window()
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        original_shape = (viewer._pixmap_item.pixmap().height(), viewer._pixmap_item.pixmap().width())
        self.assertEqual(original_shape, data.image.shape[:2])
        before = viewer._view.viewportTransform()
        panel._color_control.preview.setChecked(True)
        self.w._roi_editor.refresh()
        self.assertEqual(self.w._roi_editor._canvas.shape[:2], original_shape)
        self.assertEqual(viewer._view.viewportTransform(), before)
        panel._color_control.pick_button.setChecked(True)
        self.w._roi_editor.refresh()
        self.assertEqual(self.w._roi_editor._canvas.shape[:2], original_shape)
        self.assertEqual(viewer._view.viewportTransform(), before)

    def test_real_picker_click_samples_image_and_keeps_view(self):
        self.show_window()
        viewer = self.w._image_viewer
        control = self.w._settings_panel._color_control
        before = viewer._view.viewportTransform()
        control.swatch.click()
        control._picker.editors[0].sample_button.click()
        QTest.qWait(130)
        self.assertTrue(viewer._color_picking)
        self.assertEqual(viewer._view.viewport().cursor().shape(), Qt.CursorShape.CrossCursor)
        point = viewer._view.mapFromScene(QPointF(80, 100))
        before_range = control.values()
        QTest.mouseClick(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=point)
        QTest.qWait(130)
        self.assertFalse(control.pick_button.isChecked())
        self.assertTrue(control.preview.isChecked())
        self.assertTrue(self.w._settings_panel.is_dirty())
        lower, upper = control.values()
        self.assertEqual(upper, before_range[1])
        hue = int(cv2.cvtColor(self.source, cv2.COLOR_BGR2HSV)[100,80,0])
        self.assertLessEqual(lower[0], hue)
        self.assertGreaterEqual(upper[0], hue)
        self.assertEqual(viewer._view.viewportTransform(), before)

    def test_cancel_sampling_keeps_range_and_no_draft(self):
        self.show_window()
        control = self.w._settings_panel._color_control
        before = control.values()
        control.swatch.click()
        control._picker.editors[0].sample_button.click()
        self.w._roi_editor._cancel_pick()
        self.assertEqual(control.values(), before)
        self.assertFalse(self.w._settings_panel.is_dirty())
        self.assertFalse(self.w._image_viewer._color_picking)

    def test_pending_sample_is_cancelled_when_switching_groups(self):
        control = self.w._settings_panel._color_control
        before = control.values()
        control.swatch.click()
        control._picker.editors[0].sample_button.click()
        self.w._on_group_selected(self.groups[1])
        self.assertFalse(control.pick_button.isChecked())
        self.assertEqual(control.values(), before)
        self.assertFalse(self.w._settings_panel.is_dirty())

    def test_repeated_refreshes_never_translate_view(self):
        self.show_window()
        viewer = self.w._image_viewer
        image = np.zeros((2000, 3000, 3), np.uint8)
        viewer.set_image(image)
        viewer._set_zoom(.12)
        viewer._view.centerOn(1200, 900)
        before = viewer._view.viewportTransform()
        for _ in range(12):
            viewer.set_image(image, preserve_view=True)
        self.assertEqual(viewer._view.viewportTransform(), before)

    def test_real_mouse_moves_crop_without_panning_and_lock_disables_handles(self):
        w = self.w
        w.show()
        w._image_viewer.show()
        w._settings_panel.show()
        self.app.processEvents()
        w._image_viewer.fit_in_view()
        w._settings_panel.set_crop((.5, .5, .4, .4, 0.))
        w._roi_editor.refresh()
        view = w._image_viewer._view
        start = view.mapFromScene(QPointF(160, 120))
        end = view.mapFromScene(QPointF(175, 130))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end, delay=20)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        box = w._settings_panel.get_current_values()['load_roi']
        self.assertAlmostEqual(box[0]*320, 175, delta=2)
        self.assertAlmostEqual(box[1]*240, 130, delta=2)
        w._roi_editor.set_locked(True)
        self.assertFalse(w._image_viewer._crop_overlay.isEnabled())
        w._roi_editor.set_locked(False)
        self.assertTrue(w._image_viewer._crop_overlay.isEnabled())

    def test_load_search_box_and_detected_plate_are_separate(self):
        from root_tracker.preprocessing import roi
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        search = roi.from_bounds(5, 10, 310, 225, self.source.shape)
        panel.set_crop(search)
        self.w._roi_editor.refresh()
        self.assertEqual(viewer._crop_overlay.box, search)
        plate_bounds = viewer._plate_outline.polygon().boundingRect()
        self.assertGreater(plate_bounds.left(), 5)
        self.assertLess(plate_bounds.right(), 315)
        self.assertGreater(plate_bounds.top(), 10)
        self.assertLess(plate_bounds.bottom(), 235)
        panel._crop_edit_btn.setChecked(False)
        self.w._roi_editor.refresh()
        self.assertIsNotNone(viewer._plate_outline)
        self.assertIsNone(viewer._crop_overlay)
        self.assertLess(viewer._pixmap_item.pixmap().width(), self.source.shape[1])

    def test_revisiting_load_image_reuses_pixels_and_plate_detection(self):
        self.w._roi_editor.reset()
        from root_tracker.preprocessing.cropper import ImageCropper
        original = ImageCropper.plate_outline
        calls = []
        def detect(cropper, image):
            calls.append(image.shape)
            return original(cropper, image)
        with patch('root_tracker.gui.roi_editor.cv2.imread', wraps=cv2.imread) as read, \
                patch.object(ImageCropper, 'plate_outline', detect):
            for group in self.groups + self.groups:
                self.w._current_image = group.images[0]
                self.w._display_image(self.w._current_image)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(len(calls), 2)
        self.assertIs(self.w._roi_editor._canvas, self.w._roi_editor._source_canvas)

    def test_load_preview_does_not_reload_processed_arrays(self):
        self.w._current_series.pipeline_state.preprocessed = True
        with patch('root_tracker.gui.main_window.series_cache.load_series') as load:
            self.w._display_image(self.w._current_image)
            load.assert_not_called()

    def test_edit_crop_preserves_landmark_screen_positions_with_rotation_and_alignment(self):
        from root_tracker.preprocessing import roi
        for angle in (0., 17.):
            self.w._config.preprocess_roi = (.5, .55, .7, .7, angle)
            self.enter_preprocess()
            data = self.w._current_image
            self.w._pipeline.preprocess_image(data, original=self.source)
            matrix = np.asarray(data.plate_transform).reshape(2, 3).copy()
            matrix[:, 2] += (7, -4)  # registration translation
            data.plate_transform = matrix.ravel().tolist()
            self.show_window()
            self.w._settings_panel._crop_edit_btn.setChecked(False)
            self.w._display_image(data)
            view = self.w._image_viewer._view
            self.w._image_viewer._set_zoom(.7)
            points = np.array([[100., 100.], [130., 160.], [180., 120.]])
            cropped = roi.transform_points(points, matrix)
            before = [view.viewportTransform().map(QPointF(*(p+.5))) for p in cropped]
            self.w._settings_panel._crop_edit_btn.setChecked(True)
            self.w._roi_editor.refresh()
            for point, expected in zip(points, before):
                actual = view.viewportTransform().map(QPointF(*(point+.5)))
                self.assertAlmostEqual(actual.x(), expected.x(), places=5)
                self.assertAlmostEqual(actual.y(), expected.y(), places=5)
            self.w._settings_panel._crop_edit_btn.setChecked(False)
            self.w._roi_editor.refresh()
            for point, expected in zip(cropped, before):
                actual = view.viewportTransform().map(QPointF(*(point+.5)))
                self.assertAlmostEqual(actual.x(), expected.x(), places=5)
                self.assertAlmostEqual(actual.y(), expected.y(), places=5)

    def test_load_detection_modes_draft_and_box_only_finished_preview(self):
        from root_tracker.preprocessing import roi
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        panel._background_region.setCurrentIndex(1)
        self.assertEqual(panel.get_current_values()['background_region'], 'all')
        panel._background_enabled.setChecked(False)
        panel.set_crop(roi.from_bounds(30, 40, 200, 150, self.source.shape))
        panel._crop_edit_btn.setChecked(False)
        self.w._roi_editor.refresh()
        np.testing.assert_array_equal(self.w._roi_editor._canvas, self.source[40:190, 30:230])
        self.assertFalse(panel._color_control.swatch.isEnabled())
        self.assertTrue(panel._background_enabled.isEnabled())
        self.assertTrue(panel._color_control.isAncestorOf(panel._background_enabled))
        self.assertTrue(panel._color_control.isAncestorOf(panel._background_region))
        self.assertFalse(panel._background_region.isEnabled())
        self.assertIsNone(viewer._plate_outline)
        self.w._on_group_selected(self.groups[1])
        self.assertTrue(panel.get_current_values()['background_enabled'])
        self.w._on_group_selected(self.groups[0])
        self.assertFalse(panel.get_current_values()['background_enabled'])
        self.assertEqual(panel.get_current_values()['background_region'], 'all')
        panel._discard_btn.click()
        self.assertTrue(panel.get_current_values()['background_enabled'])

    def test_finished_load_preview_keeps_drawn_box_and_overlay_coordinates(self):
        from root_tracker.preprocessing import roi
        from root_tracker.preprocessing.cropper import ImageCropper
        panel = self.w._settings_panel
        panel.set_crop(roi.from_bounds(5, 10, 310, 225, self.source.shape))
        panel._crop_edit_btn.setChecked(False)
        self.w._roi_editor.refresh()
        config = Config(rotation=0, load_roi=panel.get_current_values()['load_roi'])
        expected, matrix = ImageCropper(config).search_image(self.source)
        np.testing.assert_array_equal(self.w._roi_editor._canvas, expected)
        np.testing.assert_allclose(self.w._roi_editor._frame_matrix[:2], matrix)
        self.assertIsNone(self.w._roi_editor.load_preview_rect((0, 0, 5, 5)))
        canvas = self.w._roi_editor._canvas
        panel._background_region.setCurrentIndex(1)
        panel._color_control.set_range((100, 50, 20), (130, 255, 255))
        self.w._roi_editor.refresh()
        self.assertIs(self.w._roi_editor._canvas, canvas)
        np.testing.assert_array_equal(self.w._roi_editor._canvas, expected)
        np.testing.assert_allclose(self.w._roi_editor._frame_matrix[:2], matrix)

    def test_reset_state_and_overlays_survive_rotated_load_crop(self):
        self.show_window()
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        self.assertFalse(panel._auto_crop_btn.isEnabled())
        panel.set_crop((.5, .5, .9, .9, 25.))
        self.assertTrue(panel._auto_crop_btn.isEnabled())
        image = self.w._current_image
        image.barcode_rect = (100, 80, 60, 30)
        image.barcode_read = 'sample'
        self.w._config.data.detect_barcodes = True
        self.w._roi_editor.refresh()
        panel._crop_edit_btn.setChecked(False)
        self.w._roi_editor.refresh()
        before = viewer._view.viewportTransform().map(viewer._barcode_items[0].pos())
        plate_before = [viewer._view.viewportTransform().map(p) for p in viewer._plate_outline.polygon()]
        for editing in (True, False):
            panel._crop_edit_btn.setChecked(editing)
            self.w._roi_editor.refresh()
            after = viewer._view.viewportTransform().map(viewer._barcode_items[0].pos())
            self.assertAlmostEqual(before.x(), after.x(), places=5)
            self.assertAlmostEqual(before.y(), after.y(), places=5)
            self.assertIsNotNone(viewer._plate_outline)
            for actual, expected in zip(viewer._plate_outline.polygon(), plate_before):
                point = viewer._view.viewportTransform().map(actual)
                self.assertAlmostEqual(point.x(), expected.x(), places=5)
                self.assertAlmostEqual(point.y(), expected.y(), places=5)
        panel._auto_crop_btn.click()
        self.assertFalse(panel._auto_crop_btn.isEnabled())

    def test_finishing_rotated_crop_straightens_view_and_edit_keeps_orientation(self):
        from root_tracker.preprocessing import roi
        self.show_window()
        panel, viewer = self.w._settings_panel, self.w._image_viewer
        for step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            if step == WorkflowStep.PREPROCESS:
                self.enter_preprocess()
            panel._crop_edit_btn.setChecked(True)
            panel.set_crop((.5, .5, .7, .7, 32.))
            self.w._roi_editor.refresh()
            panel._crop_edit_btn.setChecked(False)
            self.w._roi_editor.refresh()
            transform = viewer._view.viewportTransform()
            self.assertAlmostEqual(transform.m12(), 0., places=8)
            self.assertAlmostEqual(transform.m21(), 0., places=8)
            self.assertGreater(transform.m11(), 0.)
            self.assertGreater(transform.m22(), 0.)
            points = np.array([[100., 100.], [130., 120.]])
            cropped = roi.transform_points(points, self.w._roi_editor._frame_matrix[:2])
            expected = [transform.map(QPointF(*(point+.5))) for point in cropped]
            panel._crop_edit_btn.setChecked(True)
            self.w._roi_editor.refresh()
            for point, previous in zip(points, expected):
                actual = viewer._view.viewportTransform().map(QPointF(*(point+.5)))
                self.assertAlmostEqual(actual.x(), previous.x(), places=5)
                self.assertAlmostEqual(actual.y(), previous.y(), places=5)

    def test_centroid_redetection_reuses_processed_arrays_and_registration(self):
        self.w._config.preprocess_roi = (.5, .5, .8, .8, 17.)
        self.enter_preprocess()
        pipeline, series = self.w._pipeline, self.w._current_series
        image = self.w._current_image
        pipeline.preprocess_image(image, original=self.source)
        image.plate_transform[2] += 7
        image.plate_transform[5] -= 4
        expected_x = [x+7 for x in image.positions_x]
        expected_y = [y-4 for y in image.positions_y]
        arrays = (image.image, image.process, image.canny)
        image.positions_x = [-100]
        image.positions_y = [-100]
        image.total_length = 100
        with patch.object(pipeline, 'preprocess_image') as preprocess, patch.object(pipeline, 'register_series') as register:
            pipeline.redetect_centroids(series)
            preprocess.assert_not_called()
            register.assert_not_called()
        self.assertEqual(image.positions_x, expected_x)
        self.assertEqual(image.positions_y, expected_y)
        self.assertIsNone(image.total_length)
        for before, after in zip(arrays, (image.image, image.process, image.canny)):
            self.assertIs(before, after)

    def test_centroid_redetection_failure_keeps_existing_centers(self):
        self.enter_preprocess()
        pipeline, series = self.w._pipeline, self.w._current_series
        image = self.w._current_image
        pipeline.preprocess_image(image, original=self.source)
        expected = image.positions_x.copy()
        image.total_length = 100
        with patch.object(pipeline.green_detector, 'cluster_plant_positions', side_effect=ValueError('detection failed')):
            with self.assertRaises(ValueError):
                pipeline.redetect_centroids(series)
        self.assertEqual(image.positions_x, expected)
        self.assertEqual(image.total_length, 100)

    def test_centroid_redetection_falls_back_for_older_cache(self):
        pipeline, series = self.w._pipeline, self.w._current_series
        with patch.object(pipeline, 'preprocess_series') as preprocess, patch.object(pipeline, 'register_series') as register:
            pipeline.redetect_centroids(series)
            preprocess.assert_called_once_with(series)
            register.assert_called_once_with(series)

    def test_plate_outline_follows_barcode_warning_and_visibility(self):
        image = self.w._current_image
        self.w._config.data.detect_barcodes = True
        cases = [
            (False, False, False, '', '#9ca3af'),
            (True, False, False, 'a', '#22c55e'),
            (True, True, False, 'wrong', '#ff9800'),
            (True, False, True, '', '#ff9800'),
        ]
        for editing in (True, False):
            self.w._settings_panel._crop_edit_btn.setChecked(editing)
            for detected, mismatch, missing, read, expected in cases:
                image.barcode_detected = detected
                image.barcode_mismatch = mismatch
                image.barcode_not_found = missing
                image.barcode_read = read
                self.w._roi_editor.refresh()
                pen = self.w._image_viewer._plate_outline.pen()
                self.assertEqual(pen.color().name(), expected)
                self.assertEqual(pen.style(), Qt.PenStyle.DashLine)
        with patch.object(self.w._settings, 'setValue'):
            self.w._on_detect_barcodes_toggled(False)
            self.assertEqual(self.w._image_viewer._plate_outline.pen().color().name(), '#22c55e')
            self.w._on_detect_barcodes_toggled(True)
            self.assertEqual(self.w._image_viewer._plate_outline.pen().color().name(), '#ff9800')

    def test_auto_apply_load_edits_and_reset_restore_session_baseline(self):
        panel = self.w._settings_panel
        self.w._auto_preview_action.setChecked(True)
        original = panel.get_current_values().copy()
        self.assertTrue(panel._apply_btn.isHidden())
        self.assertTrue(panel._discard_btn.isHidden())
        self.assertFalse(panel._auto_reset_btn.isHidden())
        panel.set_crop((.5, .5, .8, .7, 12.))
        self.w._apply_auto_settings()
        self.assertEqual(self.w._config.load_roi, (.5, .5, .8, .7, 12.))
        self.assertFalse(panel._is_dirty)
        self.assertTrue(panel.crop_editing())
        panel._background_region.setCurrentIndex(1)
        self.w._apply_auto_settings()
        self.assertEqual(self.w._config.crop.background_region, 'all')
        panel._auto_reset_btn.click()
        self.assertEqual(panel.get_current_values(), original)
        self.assertIsNone(self.w._config.load_roi)
        self.assertEqual(self.w._config.crop.background_region, 'largest')
        self.assertFalse(panel._auto_reset_btn.isEnabled())

    def test_auto_apply_preprocess_and_tracking_settings(self):
        self.enter_preprocess()
        self.w._auto_preview_action.setChecked(True)
        panel = self.w._settings_panel
        original = self.w._config.n_clusters
        panel._n_clusters_spin.setValue(original+1)
        self.w._apply_auto_settings()
        self.assertEqual(self.w._config.n_clusters, original+1)
        self.w._preprocess_group.assert_called_with(self.w._current_series, force=True)
        self.assertEqual(panel._color_control.title(), 'Plants')
        self.assertEqual(panel._redetect_btn.text(), 'Reset origins')
        self.w._restore_settings_draft(WorkflowStep.TRACK)
        self.assertTrue(panel._apply_btn.isHidden())
        self.assertTrue(panel._discard_btn.isHidden())
        self.assertFalse(panel._auto_reset_btn.isHidden())
        before = self.w._config.threshold.min_contour_area
        panel._min_contour_area_spin.setValue(before+10)
        with patch.object(self.w, '_start_tracking'):
            self.w._apply_auto_settings()
        self.assertEqual(self.w._config.threshold.min_contour_area, before+10)
        self.assertTrue(panel._auto_reset_btn.isEnabled())

    def test_auto_apply_reset_survives_group_navigation(self):
        panel = self.w._settings_panel
        self.w._auto_preview_action.setChecked(True)
        panel.set_crop((.5, .5, .8, .8, 0.))
        self.w._apply_auto_settings()
        self.w._on_group_selected(self.groups[1])
        self.w._on_group_selected(self.groups[0])
        self.assertTrue(panel._auto_reset_btn.isEnabled())
        panel._auto_reset_btn.click()
        self.assertIsNone(self.w._config.load_roi)

    def test_selection_evaluates_even_when_auto_apply_is_off(self):
        self.enter_preprocess()
        self.w._auto_preview_action.setChecked(False)
        self.w._preprocess_group.reset_mock()
        self.w._on_group_selected(self.groups[1])
        self.w._preprocess_group.assert_called_once_with(self.groups[1])

    def test_auto_apply_waits_for_color_dialog_confirmation(self):
        self.w._auto_preview_action.setChecked(True)
        panel = self.w._settings_panel
        original = tuple(self.w._config.crop.blue_hsv_lower)
        panel._color_control.swatch.click()
        picker = panel._color_control._picker
        picker.editors[0].controls[0].setValue(90)
        self.w._apply_auto_settings()
        self.assertEqual(tuple(self.w._config.crop.blue_hsv_lower), original)
        picker.reject()
        self.w._apply_auto_settings()
        self.assertEqual(tuple(self.w._config.crop.blue_hsv_lower), original)

    def test_auto_apply_timer_coalesces_edits(self):
        self.enter_preprocess()
        self.w._evaluation_timer.stop()
        self.w._auto_preview_action.setChecked(True)
        self.w._preprocess_group.reset_mock()
        panel = self.w._settings_panel
        panel._n_clusters_spin.setValue(2)
        panel._n_clusters_spin.setValue(3)
        panel._n_clusters_spin.setValue(4)
        QTest.qWait(450)
        self.assertEqual(self.w._config.n_clusters, 4)
        self.w._preprocess_group.assert_called_once_with(self.w._current_series, force=True)

    def test_auto_apply_commits_pending_edit_before_switching_groups(self):
        self.w._auto_preview_action.setChecked(True)
        self.w._settings_panel.set_crop((.5, .5, .7, .8, 0.))
        self.w._on_group_selected(self.groups[1])
        self.assertIsNone(self.w._config.load_roi)
        self.w._on_group_selected(self.groups[0])
        self.assertEqual(self.w._config.load_roi, (.5, .5, .7, .8, 0.))
        self.w._settings_panel._auto_reset_btn.click()
        self.assertIsNone(self.w._config.load_roi)


if __name__ == '__main__':
    unittest.main()
