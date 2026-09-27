"""Pending edits survive navigation and can be discarded without touching results."""
import unittest
import tempfile
from datetime import datetime
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.models import ImageData, ImageSeries


class SettingsDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = Config()
        config.data.input = folder.name
        config.data.output = folder.name
        with patch.object(MainWindow, '_load_last_folder'):
            self.w = MainWindow(config)
        self.addCleanup(self.w.close)
        self.groups = [ImageSeries(name, [ImageData(datetime(2026, 1, 1), '', name)]) for name in ('a', 'b')]
        self.w._series_dict = {s.group: s for s in self.groups}
        self.w._image_tree.set_series(self.w._series_dict)
        self.w._settings_panel.set_step(WorkflowStep.TRACK)
        self.w._image_tree.set_step(WorkflowStep.TRACK, self.w._config)
        self.w._auto_preview_action.setChecked(False)
        # These tests exercise settings/preview behavior; selection now always
        # schedules processing, so isolate the worker boundary explicitly.
        for method in ('_preprocess_group', '_auto_process_for_tracking', '_on_track_roots', '_detect_barcodes_in_group'):
            worker_patch = patch.object(self.w, method)
            worker_patch.start()
            self.addCleanup(worker_patch.stop)

    def select(self, group):
        with patch.object(self.w, '_ensure_series_loaded'), patch.object(self.w, '_display_image'):
            self.w._on_group_selected(group)

    def test_registration_border_follows_checkbox_and_restored_draft(self):
        panel = self.w._settings_panel
        panel.set_step(WorkflowStep.PREPROCESS)
        self.assertTrue(panel._reg_margin_spin.isEnabled())
        panel._reg_enabled_cb.setChecked(False)
        self.assertFalse(panel._reg_margin_spin.isEnabled())
        panel.set_pending_values({'reg_enabled': True, 'reg_margin': .1})
        self.assertTrue(panel._reg_margin_spin.isEnabled())
        self.assertEqual(panel.get_current_values()['reg_margin'], .1)
        panel.set_pending_values({'reg_enabled': False})
        self.assertFalse(panel._reg_margin_spin.isEnabled())

    def test_two_groups_keep_separate_drafts_and_discard_only_current(self):
        panel = self.w._settings_panel
        self.select(self.groups[0])
        baseline = panel._min_contour_area_spin.value()
        panel._min_contour_area_spin.setValue(baseline + 10)
        self.select(self.groups[1])
        self.assertEqual(panel._min_contour_area_spin.value(), baseline)
        panel._min_contour_area_spin.setValue(baseline + 20)
        self.select(self.groups[0])
        self.assertEqual(panel._min_contour_area_spin.value(), baseline + 10)
        self.assertEqual([self.w._image_tree._tree.topLevelItem(i).text(1) for i in (0, 1)], ['●', '●'])
        panel._discard_btn.click()
        self.assertEqual(panel._min_contour_area_spin.value(), baseline)
        self.assertFalse(panel._discard_btn.isEnabled())
        self.select(self.groups[1])
        self.assertEqual(panel._min_contour_area_spin.value(), baseline + 20)
        self.assertFalse(hasattr(panel, '_status_indicator'))

    def test_mask_erase_is_a_draft_and_discard_restores_applied_mask(self):
        series = self.groups[0]
        series.user_mask = np.full((20, 30), 255, np.uint8)
        series.working_mask = series.user_mask.copy()
        self.select(series)
        self.w._image_viewer.set_mask_data(series.user_mask, series.working_mask)
        with patch('root_tracker.gui.main_window.mask_io.delete_mask') as delete:
            self.w._on_mask_erase_all()
            delete.assert_not_called()
        self.assertTrue(series.user_mask.all())
        self.assertFalse(series.working_mask.any())
        self.select(self.groups[1])
        self.select(series)
        self.assertTrue(self.w._settings_panel._discard_btn.isEnabled())
        with patch.object(self.w, '_display_image'):
            self.w._settings_panel._discard_btn.click()
        np.testing.assert_array_equal(series.working_mask, series.user_mask)

    def test_drafts_survive_step_changes_and_discard_is_step_specific(self):
        self.select(self.groups[0])
        w = self.w
        panel = w._settings_panel
        panel._min_contour_area_spin.setValue(111)
        w._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
        with patch.object(w, '_display_image'):
            w._on_step_changed(WorkflowStep.PREPROCESS)
            panel._n_clusters_spin.setValue(9)
            w._on_step_changed(WorkflowStep.TRACK)
            self.assertEqual(panel._min_contour_area_spin.value(), 111)
            panel._discard_btn.click()
            self.assertEqual(panel._min_contour_area_spin.value(), w._config.threshold.min_contour_area)
            w._on_step_changed(WorkflowStep.PREPROCESS)
            self.assertEqual(panel._n_clusters_spin.value(), 9)

    def test_applying_one_group_keeps_other_groups_draft(self):
        from unittest.mock import Mock
        self.select(self.groups[0])
        panel = self.w._settings_panel
        panel._min_contour_area_spin.setValue(111)
        self.select(self.groups[1])
        panel._min_contour_length_spin.setValue(23)
        self.select(self.groups[0])
        self.w._pipeline = Mock()
        with patch.object(self.w._workflow_bar, 'get_current_step', return_value=WorkflowStep.TRACK), patch.object(self.w, '_display_image'):
            panel._apply_btn.click()
        self.assertEqual(self.w._config.threshold.min_contour_area, 111)
        self.select(self.groups[1])
        self.assertEqual(panel._min_contour_area_spin.value(), 100)
        self.assertEqual(panel._min_contour_length_spin.value(), 23)
        self.assertTrue(panel._discard_btn.isEnabled())

    def test_applied_tracking_settings_only_invalidate_selected_group(self):
        from root_tracker.pipeline import RootTrackingPipeline
        self.w._pipeline = RootTrackingPipeline(self.w._config)
        for series in self.groups:
            state = series.pipeline_state
            state.preprocessed = state.tracked = True
            state.preprocess_config_hash = self.w._config.preprocess_config_hash()
            state.tracking_config_hash = self.w._config.tracking_config_hash()
            series.images[0].positions_x = [10]
            series.images[0].total_length = 42
        self.select(self.groups[0])
        self.w._settings_panel._min_contour_area_spin.setValue(111)
        self.w._on_apply_settings(evaluate=False)
        self.assertTrue(self.w._pipeline.is_tracking_current(self.groups[1]))
        tree = self.w._image_tree
        tree.set_step(WorkflowStep.TRACK, self.w._config)
        self.assertEqual(tree._tree.topLevelItem(1).text(1), '✓')
        self.select(self.groups[1])
        self.assertEqual(self.w._config.threshold.min_contour_area, 100)
        self.select(self.groups[0])
        self.assertEqual(self.w._config.threshold.min_contour_area, 111)

    def test_update_all_copies_already_applied_settings_to_other_groups(self):
        from root_tracker.pipeline import RootTrackingPipeline
        self.w._pipeline = RootTrackingPipeline(self.w._config)
        self.groups[1].processing_settings = self.w._config.processing_settings()
        self.groups[1].processing_settings['n_clusters'] = 9
        self.select(self.groups[0])
        self.w._settings_panel._min_contour_area_spin.setValue(111)
        self.w._on_apply_settings(evaluate=False)
        with patch.object(self.w._workflow_bar, 'get_current_step', return_value=WorkflowStep.TRACK):
            self.w._on_apply_all_settings()
        self.select(self.groups[1])
        self.assertEqual(self.w._config.threshold.min_contour_area, 111)
        self.assertEqual(self.w._config.n_clusters, 9)

    def test_tree_status_column_is_compact(self):
        self.assertEqual(self.w._image_tree._tree.columnWidth(1), 30)
        self.assertLess(self.w._image_tree._tree.indentation(), 20)
