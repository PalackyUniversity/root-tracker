"""A chained operation has one lifetime, percentage and ETA."""
import unittest
from unittest.mock import patch
from datetime import datetime
import numpy as np
from PySide6.QtWidgets import QApplication
from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow, ProcessingContext, ProcessingState
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.models import ImageData, ImageSeries


class WorkflowProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        with patch.object(MainWindow, '_load_last_folder'):
            self.window = MainWindow(Config())
        self.addCleanup(self.window.close)

    def test_chain_keeps_one_clock_and_never_restarts_percentage(self):
        w = self.window
        w._auto_process_pending_preprocess = True
        w._auto_process_pending_tracking = True
        w._begin_operation_progress('barcode')
        started = w._operation_progress.started
        w._on_worker_progress(95, 100)
        values = [w._progress_bar.value()]
        w._finish_operation_progress(True, continuing=True)
        w._begin_operation_progress('preprocess')
        values.append(w._progress_bar.value())
        w._on_worker_progress(75, 100)
        values.append(w._progress_bar.value())
        self.assertIn('remaining', w._processing_label.text())
        w._finish_operation_progress(True, continuing=True)
        w._begin_operation_progress('track')
        w._on_worker_progress(100, 100)
        values.append(w._progress_bar.value())
        self.assertLess(values[-1], 100, 'Saving/completion must finish before 100%')
        self.assertEqual(w._operation_progress.started, started)
        self.assertEqual(values, sorted(values))
        w._finish_operation_progress(True, continuing=False)
        self.assertEqual(w._progress_bar.value(), 100)
        self.assertIsNone(w._operation_progress)

    def test_cancel_does_not_report_complete_and_next_run_starts_fresh(self):
        w = self.window
        w._begin_operation_progress('preprocess')
        w._on_worker_progress(50, 100)
        w._finish_operation_progress(False, continuing=False)
        self.assertLess(w._progress_bar.value(), 100)
        w._begin_operation_progress('track')
        self.assertEqual(w._progress_bar.value(), 0)
        self.assertEqual(w._operation_progress.operations, ['track'])

    def test_preprocess_displays_centroids_and_other_steps_clear_them(self):
        w = self.window
        image = ImageData(datetime(2026, 1, 1), '', 'group')
        image.image = np.zeros((100, 150, 3), np.uint8)
        image.positions_x = [25, 80]
        image.positions_y = [30, 40]
        w._current_series = ImageSeries('group', [image])
        w._current_image = image
        with patch.object(w, '_ensure_series_loaded'), patch.object(w._workflow_bar, 'get_current_step', return_value=WorkflowStep.PREPROCESS):
            w._display_image(image)
            self.assertEqual(w._image_viewer.get_centroids(), [(25, 30), (80, 40)])
            w._display_image(image)
            self.assertEqual(len(w._image_viewer._centroid_items), 2)
        with patch.object(w, '_ensure_series_loaded'), patch.object(w._workflow_bar, 'get_current_step', return_value=WorkflowStep.TRACK):
            w._display_image(image)
            self.assertEqual(w._image_viewer.get_centroids(), [])

    def test_preprocess_shows_group_origins_in_both_display_paths(self):
        w = self.window
        images = []
        for day, x, y in ((1, 80, 20), (2, 100, 40), (3, 90, 30)):
            image = ImageData(datetime(2026, 1, day), '', 'group')
            image.image = np.zeros((100, 150, 3), np.uint8)
            image.positions_x, image.positions_y = [25, x], [10, y]
            images.append(image)
        w._current_series = ImageSeries('group', images)
        # LOAD exercises the fallback; PREPROCESS exercises the ROI presenter.
        for panel_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            w._settings_panel.set_step(panel_step)
            with patch.object(w, '_ensure_series_loaded'), patch.object(w._workflow_bar, 'get_current_step', return_value=WorkflowStep.PREPROCESS):
                for image in images:
                    w._current_image = image
                    w._display_image(image)
                    self.assertEqual(w._image_viewer.get_centroids(), [(25, 10), (90, 30)])

    def test_dragging_origin_updates_group_and_invalidates_tracking(self):
        w = self.window
        images = [ImageData(datetime(2026, 1, day), '', 'group') for day in (1, 2, 3)]
        for image in images:
            image.positions_x, image.positions_y = [25, 80], [10, 20]
            image.image_annotated = np.zeros((100, 150, 3), np.uint8)
        series = ImageSeries('group', images)
        series.pipeline_state.tracked = True
        w._current_series, w._current_image = series, images[0]
        w._on_centroid_moved(1, 95, 35)
        for image in images:
            self.assertEqual((image.positions_x, image.positions_y), ([25, 95], [10, 35]))
            self.assertIsNone(image.image_annotated)
        self.assertFalse(series.pipeline_state.tracked)

    def test_settings_and_their_labels_explain_the_controls(self):
        from PySide6.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QFormLayout
        panel = self.window._settings_panel
        for step in (WorkflowStep.PREPROCESS, WorkflowStep.TRACK):
            panel.set_step(step)
            self.app.sendPostedEvents()
            for kind in (QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton):
                for widget in panel._settings_container.findChildren(kind):
                    self.assertTrue(widget.statusTip(), type(widget).__name__)
                    self.assertFalse(widget.toolTip(), type(widget).__name__)
            for form in panel._settings_container.findChildren(QFormLayout):
                for row in range(form.rowCount()):
                    label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    if label and label.widget():
                        self.assertTrue(label.widget().statusTip(), label.widget().text())
                        self.assertFalse(label.widget().toolTip())

    def test_completion_summary_is_absent_from_status_bar_and_tree(self):
        w = self.window
        image = ImageData(datetime(2026, 1, 1), '', 'group')
        w._image_tree.set_series({'group': ImageSeries('group', [image])})
        w._image_tree.set_step(WorkflowStep.PREPROCESS, w._config)
        self.assertFalse(hasattr(w, '_step_summary'))
        self.assertNotIn('/', w._image_tree._tree.topLevelItem(0).text(1))
        self.assertFalse(hasattr(w._image_tree, '_status_label'))

    def test_marker_stays_legible_and_movement_keeps_image_coordinates(self):
        from root_tracker.gui.image_viewer import DraggableCentroid
        moved = []
        marker = DraggableCentroid(25, 30, radius=7, callback=lambda *args: moved.append(args))
        marker.setPos(40, 60)
        self.assertEqual(marker.get_center(), (40, 60))
        self.assertEqual(moved[-1], (0, 40, 60))
        self.assertTrue(marker.flags() & marker.GraphicsItemFlag.ItemIgnoresTransformations)

    def test_real_worker_chain_has_single_completion(self):
        import time
        from unittest.mock import Mock
        from PySide6.QtTest import QTest
        from root_tracker.gui.main_window import ProcessWorker
        w = self.window
        image = ImageData(datetime(2026, 1, 1), '', 'group')
        image.image = np.zeros((100, 150, 3), np.uint8)
        series = ImageSeries('group', [image])
        w._current_series = series
        w._current_image = image
        w._series_dict = {'group': series}
        w._image_tree.set_series(w._series_dict)
        pipeline = Mock()
        def barcodes(images, cancelled):
            time.sleep(.01)
            image.barcode_detected = True
            yield image
        def preprocess(series, progress_callback):
            time.sleep(.01)
            progress_callback(1, 1)
        def track(series, progress_callback):
            time.sleep(.01)
            progress_callback(1, 1)
            return []
        pipeline.iter_detect_barcodes.side_effect = barcodes
        pipeline.preprocess_series.side_effect = preprocess
        pipeline.track_and_analyze_series.side_effect = track
        pipeline.is_tracking_current.return_value = False
        w._pipeline = pipeline
        w._config.data.detect_barcodes = True
        workers = []
        values = []
        w._progress_bar.valueChanged.connect(values.append)
        def worker_factory(*args):
            worker = ProcessWorker(*args)
            workers.append(worker)
            return worker
        with patch('root_tracker.gui.main_window.ProcessWorker', side_effect=worker_factory), patch('root_tracker.io.series_cache.save_series'), patch.object(w, '_display_image'):
            w._auto_process_for_tracking(series)
            try:
                for _ in range(300):
                    self.app.processEvents()
                    if w._state == ProcessingState.IDLE:
                        break
                    QTest.qWait(10)
                self.assertEqual(w._state, ProcessingState.IDLE)
            finally:
                for worker in workers:
                    worker.cancel()
                    worker.wait()
        self.assertEqual(len(workers), 3)
        self.assertEqual(values, sorted(values))
        self.assertEqual(values.count(100), 1)
        self.assertTrue(series.pipeline_state.tracked)

    def test_redetection_preserves_view_and_button_help_uses_status_bar(self):
        from unittest.mock import Mock
        from PySide6.QtWidgets import QLabel
        w = self.window
        image = ImageData(datetime(2026, 1, 1), '', 'group')
        image.image = np.zeros((400, 600, 3), np.uint8)
        w._current_image = image
        w._current_series = ImageSeries('group', [image])
        w._pipeline = Mock()
        w._image_viewer.resize(800, 600)
        w._settings_panel.set_step(WorkflowStep.PREPROCESS)
        with patch.object(w, '_ensure_series_loaded'), patch.object(w._workflow_bar, 'get_current_step', return_value=WorkflowStep.PREPROCESS), patch('root_tracker.io.series_cache.save_series'), patch('root_tracker.gui.main_window.QMessageBox.critical') as error:
            w.show()
            w._image_viewer.show()
            w._display_image(image)
            self.app.processEvents()  # Settle the initial window layout before user zoom/pan.
            w._image_viewer._set_zoom(2)
            view = w._image_viewer._view
            view.centerOn(220, 170)
            before = view.mapToScene(view.viewport().rect().center())
            zoom = view.transform().m11()
            w._on_redetect_plants()
            error.assert_not_called()
            self.assertAlmostEqual(view.transform().m11(), zoom)
            self.assertLessEqual((view.mapToScene(view.viewport().rect().center()) - before).manhattanLength(), 1)
        panel = w._settings_panel
        self.assertFalse(panel._apply_btn.styleSheet())
        self.assertFalse(any('Settings are shared' in label.text() for label in panel.findChildren(QLabel)))
        self.assertIn('this group only', panel._apply_btn.statusTip())
        self.assertFalse(panel._apply_btn.toolTip())
        w._begin_operation_progress('preprocess')
        w._on_worker_progress(50, 100)
        self.assertNotIn('About', w._processing_label.text())

    def test_unsaved_indicator_does_not_move_settings_or_apply_buttons(self):
        w = self.window
        panel = w._settings_panel
        panel.set_step(WorkflowStep.TRACK)
        panel.show()
        w.show()
        self.app.processEvents()
        field_before = panel._min_contour_area_spin.mapTo(w, panel._min_contour_area_spin.rect().topLeft())
        apply_before = panel._apply_btn.mapTo(w, panel._apply_btn.rect().topLeft())
        panel._min_contour_area_spin.setValue(panel._min_contour_area_spin.value() + 1)
        self.app.processEvents()
        self.assertFalse(hasattr(panel, '_status_indicator'))
        self.assertTrue(panel._discard_btn.isEnabled())
        self.assertEqual(panel._min_contour_area_spin.mapTo(w, panel._min_contour_area_spin.rect().topLeft()), field_before)
        self.assertEqual(panel._apply_btn.mapTo(w, panel._apply_btn.rect().topLeft()), apply_before)
        panel._mark_clean()
        self.app.processEvents()
        self.assertEqual(panel._apply_btn.mapTo(w, panel._apply_btn.rect().topLeft()), apply_before)

    def test_equal_next_button_edge_margins(self):
        w = self.window
        w.show()
        self.app.processEvents()
        next_corner = w._next_step_btn.mapTo(w, w._next_step_btn.rect().bottomRight())
        right = w.width() - 1 - next_corner.x()
        bottom = w.height() - 1 - next_corner.y()
        self.assertLessEqual(abs(right - bottom), 1)

    def test_pending_edits_are_shared_between_panel_and_current_group_only(self):
        w = self.window
        panel = w._settings_panel
        images = [ImageData(datetime(2026, 1, 1), '', name) for name in ('one', 'two')]
        series = [ImageSeries(name, [image]) for name, image in zip(('one', 'two'), images)]
        w._current_series = series[0]
        w._current_image = images[0]
        w._series_dict = {group.group: group for group in series}
        w._image_tree.set_series(w._series_dict)
        w._image_tree.set_step(WorkflowStep.PREPROCESS, w._config)
        panel.set_step(WorkflowStep.PREPROCESS)
        spin = panel._n_clusters_spin
        original = spin.value()
        spin.setValue(original + 1)
        group = w._image_tree._tree.topLevelItem(0)
        other = w._image_tree._tree.topLevelItem(1)
        self.assertEqual(group.text(1), '●')
        self.assertEqual(group.child(0).text(1), '●')
        self.assertEqual(other.text(1), '○')
        self.assertIn('Unapplied', group.toolTip(1))
        spin.setValue(original)
        self.assertEqual(group.text(1), '○')
        self.assertFalse(hasattr(panel, '_status_indicator'))
        spin.setValue(original + 1)
        panel.apply_requested.disconnect()  # Test Apply's edit lifecycle without running CV.
        panel._on_apply_clicked()
        self.assertEqual(group.text(1), '○')
        spin.setValue(spin.value() + 1)
        with patch.object(w, '_display_image'), patch.object(w, '_ensure_series_loaded'), patch.object(w._image_tree, 'get_selected_series', return_value=series[1]):
            w._on_image_selected(images[1])
        self.assertEqual(group.text(1), '●')
        self.assertEqual(other.text(1), '○')
        self.assertFalse(hasattr(panel, '_status_indicator'))

    def test_apply_tracking_settings_preserves_view_after_completion(self):
        from unittest.mock import Mock
        w = self.window
        image = ImageData(datetime(2026, 1, 1), '', 'group')
        image.image = np.zeros((400, 600, 3), np.uint8)
        w._current_image = image
        w._current_series = ImageSeries('group', [image])
        w._pipeline = Mock()
        w._auto_preview_action.setChecked(True)
        w._settings_panel.set_step(WorkflowStep.TRACK)
        with patch.object(w, '_ensure_series_loaded'), patch.object(w._workflow_bar, 'get_current_step', return_value=WorkflowStep.TRACK), patch.object(w, '_on_track_roots') as start:
            w.show()
            w._image_viewer.show()
            w._display_image(image)
            self.app.processEvents()
            view = w._image_viewer._view
            w._image_viewer._set_zoom(2)
            view.centerOn(220, 170)
            center = view.mapToScene(view.viewport().rect().center())
            zoom = view.transform().m11()
            spin = w._settings_panel._min_contour_area_spin
            spin.setValue(spin.value() + 1)
            w._on_apply_settings()
            start.assert_called_once()
            image.image_annotated = image.image.copy()
            w._on_tracking_finished(True, '')
            self.assertAlmostEqual(view.transform().m11(), zoom)
            self.assertLessEqual((view.mapToScene(view.viewport().rect().center()) - center).manhattanLength(), 1)

    def test_transition_keeps_progress_visible_and_ui_locked(self):
        w = self.window
        w._auto_process_pending_tracking = True
        w._begin_operation_progress('preprocess')
        context = ProcessingContext(w, ProcessingState.PREPROCESSING)
        context.__enter__()
        observed = []
        context.next_operation = lambda: observed.append((w._progress_bar.isHidden(), w._image_tree.isEnabled()))
        context.__exit__(None, None, None)
        self.assertEqual(observed, [(False, False)])
