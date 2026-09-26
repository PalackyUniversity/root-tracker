"""Step status must stay accurate without depending on resident image arrays."""
from datetime import datetime
import tempfile
import unittest
from unittest.mock import patch, Mock

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.models import ImageData, ImageSeries
from root_tracker.gui.image_tree import ImageTree
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.gui.main_window import MainWindow, ProcessingContext, ProcessingState, ProcessWorker
from root_tracker.io import series_cache


class TreeStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.config = Config()
        self.images = [ImageData(datetime(2026, 1, i + 1), f'/images/{i}.jpg', 'group')
                       for i in range(3)]
        self.series = ImageSeries('group', self.images)
        self.tree = ImageTree()
        self.tree.set_series({'group': self.series})
        self.addCleanup(self.tree.close)

    def status(self, index):
        return self.tree._tree.topLevelItem(0).child(index).data(1, Qt.ItemDataRole.UserRole)

    def mark_preprocessed(self):
        self.series.pipeline_state.preprocessed = True
        self.series.pipeline_state.preprocess_config_hash = self.config.preprocess_config_hash()
        for image in self.images[:2]:
            image.positions_x = [15, 30]
            image.positions_y = [1, 2]

    def test_preprocess_counts_only_successful_images_even_when_arrays_are_freed(self):
        self.mark_preprocessed()
        self.tree.set_step(WorkflowStep.PREPROCESS, self.config)
        self.assertEqual([self.status(i) for i in range(3)], ['done', 'done', 'pending'])
        self.assertIn('2 of 3', self.tree._tree.topLevelItem(0).toolTip(1))
        self.assertEqual(self.tree._tree.topLevelItem(0).text(1), '○')

    def test_tracking_is_distinct_and_zero_root_length_counts_as_done(self):
        self.mark_preprocessed()
        self.series.pipeline_state.tracked = True
        self.series.pipeline_state.tracking_config_hash = self.config.tracking_config_hash()
        self.images[0].total_length = 0
        self.tree.set_step(WorkflowStep.TRACK, self.config)
        self.assertEqual([self.status(i) for i in range(3)], ['done', 'pending', 'pending'])

    def test_changed_preprocessing_settings_invalidate_tracking_status(self):
        self.mark_preprocessed()
        self.series.pipeline_state.tracked = True
        self.series.pipeline_state.tracking_config_hash = self.config.tracking_config_hash()
        self.images[0].total_length = 10
        self.config.rotation = 90
        self.tree.set_step(WorkflowStep.TRACK, self.config)
        self.assertEqual(self.status(0), 'outdated')
        self.series.clear_preprocessing_results()
        self.tree.refresh_status()
        self.assertEqual(self.status(0), 'pending')

    def test_status_refresh_preserves_selection_expansion_and_warnings(self):
        self.images[0].barcode_detected = True
        self.images[0].barcode_not_found = True
        self.tree.refresh()
        self.tree.set_step(WorkflowStep.LOAD, self.config)
        self.tree.select_image(self.images[0])
        group = self.tree._tree.topLevelItem(0)
        group.setExpanded(False)
        warning = group.child(0).toolTip(0)
        self.images[1].barcode_detected = True
        self.tree.refresh_status()
        self.assertIs(self.tree.get_selected_data(), self.images[0])
        self.assertFalse(group.isExpanded())
        self.assertEqual(group.child(0).toolTip(0), warning)
        self.assertEqual(self.status(0), 'done')
        self.assertIn('checked', group.child(0).toolTip(1).lower())
        self.assertTrue(group.child(0).data(1, Qt.ItemDataRole.AccessibleTextRole))

    def test_barcode_warnings_replace_right_status_and_toggle_without_rebuilding(self):
        self.config.data.detect_barcodes = True
        self.images[0].barcode_detected = True
        self.images[0].barcode_not_found = True
        self.images[1].barcode_detected = True
        self.images[1].barcode_mismatch = True
        self.images[1].barcode_read = 'wrong'
        self.tree.refresh()
        self.tree.set_step(WorkflowStep.LOAD, self.config)
        group = self.tree._tree.topLevelItem(0)
        self.assertEqual(group.text(0), 'group')
        for i in (0, 1):
            self.assertEqual(group.child(i).text(0), self.images[i].date.strftime('%Y-%m-%d'))
            self.assertEqual(group.child(i).text(1), '⚠')
        self.assertEqual(group.text(1), '⚠')
        self.assertIn('wrong', group.child(1).toolTip(1))
        self.config.data.detect_barcodes = False
        self.tree.refresh_status()
        self.assertIs(self.tree._tree.topLevelItem(0), group)
        self.assertEqual(group.text(1), '✓')
        for i in (0, 1):
            self.assertEqual(group.child(i).text(1), '✓')
            self.assertNotIn('No barcode detected', group.child(i).toolTip(1))
        self.config.data.detect_barcodes = True
        self.tree.refresh_status()
        self.assertEqual(group.text(1), '⚠')
        self.assertEqual(group.child(0).text(1), '⚠')

    def test_pending_dot_overlays_success_without_invalidating_results(self):
        self.mark_preprocessed()
        self.tree.set_step(WorkflowStep.PREPROCESS, self.config)
        group = self.tree._tree.topLevelItem(0)
        self.assertEqual(group.child(0).text(1), '✓')
        self.tree.set_pending_changes(self.series)
        self.assertEqual(group.child(0).text(1), '●')
        self.assertTrue(self.series.pipeline_state.preprocessed)
        self.assertEqual(self.status(0), 'done')
        self.config.data.detect_barcodes = True
        self.images[0].barcode_detected = True
        self.images[0].barcode_not_found = True
        self.tree.refresh_status()
        self.assertEqual(group.child(0).text(1), '⚠')
        self.assertIn('Unapplied', group.child(0).toolTip(1))
        self.tree.set_pending_changes(None)
        self.assertEqual(group.child(1).text(1), '✓')

    def test_disabled_barcode_check_shows_loaded_and_aside_is_excluded(self):
        self.config.data.detect_barcodes = False
        self.images[2].path = '/images/aside/2.jpg'
        self.tree.refresh()
        self.tree.set_step(WorkflowStep.LOAD, self.config)
        self.assertEqual(self.status(0), 'done')
        self.assertEqual(self.status(2), 'excluded')
        self.assertEqual(self.tree._tree.topLevelItem(0).text(1), '✓')

    def test_active_group_is_not_reported_as_complete_until_registration_finishes(self):
        self.mark_preprocessed()
        self.tree.set_step(WorkflowStep.PREPROCESS, self.config)
        self.tree.set_processing(self.series, WorkflowStep.PREPROCESS)
        self.assertEqual(self.status(0), 'running')
        self.tree.set_processing(None, None)
        self.assertEqual(self.status(0), 'done')

    def test_failed_rerun_invalidates_old_completion(self):
        self.mark_preprocessed()
        self.series.pipeline_state.tracked = True
        pipeline = Mock()
        pipeline.preprocess_series.side_effect = RuntimeError('failed')
        worker = ProcessWorker(pipeline, self.series, self.config)
        with self.assertRaises(RuntimeError):
            worker._run_preprocess()
        self.assertFalse(self.series.pipeline_state.preprocessed)
        self.assertFalse(self.series.pipeline_state.tracked)

    def test_interrupted_tracking_preserves_only_preprocessing_completion(self):
        self.mark_preprocessed()
        self.series.pipeline_state.tracked = True
        pipeline = Mock()
        pipeline.track_and_analyze_series.side_effect = InterruptedError('cancelled')
        worker = ProcessWorker(pipeline, self.series, self.config, 'track')
        with self.assertRaises(InterruptedError):
            worker._run_track()
        self.assertTrue(self.series.pipeline_state.preprocessed)
        self.assertFalse(self.series.pipeline_state.tracked)

    def test_batch_context_marks_and_releases_active_groups(self):
        self.mark_preprocessed()
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(self.config)
        self.addCleanup(window.close)
        window._series_dict = {'group': self.series}
        window._image_tree.set_series(window._series_dict)
        window._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
        window._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
        item = window._image_tree._tree.topLevelItem(0).child(0)
        with ProcessingContext(window, ProcessingState.BATCH_PREPROCESS, [self.series]) as context:
            self.assertEqual(item.data(1, Qt.ItemDataRole.UserRole), 'running')
            context.series = []
            window._refresh_tree_status()
            self.assertEqual(item.data(1, Qt.ItemDataRole.UserRole), 'done')

    def test_metadata_only_cache_restore_preserves_completion(self):
        self.mark_preprocessed()
        with tempfile.TemporaryDirectory() as directory:
            self.config.data.input = directory
            self.assertIsNotNone(series_cache.save_series(self.series, self.config))
            restored = ImageSeries('group', [ImageData(img.date, img.path, img.barcode) for img in self.images])
            self.assertTrue(series_cache.load_series_state(restored, self.config))
            self.assertIsNone(restored.images[0].image)
            self.tree.set_series({'group': restored})
            self.tree.set_step(WorkflowStep.PREPROCESS, self.config)
            self.assertEqual([self.status(i) for i in range(3)], ['done', 'done', 'pending'])

    def test_main_window_switch_and_processing_context_update_status(self):
        self.mark_preprocessed()
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(self.config)
        self.addCleanup(window.close)
        window._series_dict = {'group': self.series}
        window._image_tree.set_series(window._series_dict)
        window._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
        window._workflow_bar.set_current_step(WorkflowStep.PREPROCESS)
        with patch.object(window, '_handle_enter_preprocess'):
            window._on_step_changed(WorkflowStep.PREPROCESS)
        item = window._image_tree._tree.topLevelItem(0).child(0)
        self.assertEqual(item.data(1, Qt.ItemDataRole.UserRole), 'done')
        with ProcessingContext(window, ProcessingState.PREPROCESSING, self.series):
            self.assertEqual(item.data(1, Qt.ItemDataRole.UserRole), 'running')
        self.assertEqual(item.data(1, Qt.ItemDataRole.UserRole), 'done')


if __name__ == '__main__':
    unittest.main()
