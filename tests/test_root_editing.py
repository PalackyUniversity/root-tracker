"""Manual changes must persist, leave other detections intact, and export."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from root_tracker.models import ImageData, ImageSeries
from root_tracker.io.rsml import parse_rsml
from root_tracker.io.rsml_replacement import load_replacement


class RootEditingTests(unittest.TestCase):
    def test_reassign_with_cleared_mask_keeps_main_highlight_in_new_color(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import render_replacement, root_colors
        self.image.main_root_samples = np.array([(0, 20, y) for y in range(20, 61)])
        self.image.rsml_unmasked_samples = self.image.rsml_samples
        doc = editable_document(self.image)
        edit_roots(self.image, doc, {0}, plant_index=1, mask=np.zeros((100, 100), np.uint8))
        fresh = ImageData(self.image.date, self.image.path, 'p')
        self.assertTrue(load_replacement(fresh))
        rendered = render_replacement(fresh, show_max_root_depth=False)
        color = root_colors(fresh.rsml_document)[1]
        np.testing.assert_array_equal(rendered[30, 20], [min(c + 170, 255) for c in color])

    def test_main_root_highlight_survives_delete_mask_and_reload(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import render_replacement, root_colors
        from root_tracker.io.root_mask import apply_root_mask
        self.image.rsml_samples[0] = np.array(
            [(20, y) for y in range(20, 61)] + [(40, y) for y in range(20, 41)])
        self.image.main_root_samples = np.array([(0, 20, y) for y in range(20, 61)])
        doc = editable_document(self.image)
        edit_roots(self.image, doc, {next(i for i, r in enumerate(doc.roots) if r.plant_index == 1)}, delete=True)
        fresh = ImageData(self.image.date, self.image.path, 'p')
        self.assertTrue(load_replacement(fresh))
        mask = np.zeros((100, 100), np.uint8)
        mask[50:] = 255
        apply_root_mask(fresh, mask)
        rendered = render_replacement(fresh)
        color = root_colors(fresh.rsml_document)[0]
        np.testing.assert_array_equal(rendered[30, 20], [min(c + 170, 255) for c in color])
        np.testing.assert_array_equal(rendered[30, 40], color)
        np.testing.assert_array_equal(rendered[60, 20], [0, 0, 0])
        np.testing.assert_array_equal(rendered[20, 90], [255, 255, 255])
        hidden = render_replacement(fresh, show_max_root_depth=False)
        np.testing.assert_array_equal(hidden[20, 90], [0, 0, 0])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = ImageData(datetime(2026, 1, 1), str(Path(self.temp.name) / 'roots.png'), 'p')
        self.image.image = np.zeros((100, 100, 3), np.uint8)
        self.image.rsml_samples = {0: np.array([[20, y] for y in range(20, 61)]),
                                   1: np.array([[60, y] for y in range(20, 61)])}
        self.series = ImageSeries('p', [self.image])

    def test_reassignment_and_deletion_survive_reload_and_export(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_export import export_series_rsml
        from root_tracker.config import Config
        doc = editable_document(self.image)
        self.assertEqual(len(doc.roots), 2)
        edit_roots(self.image, doc, {0}, plant_index=1)
        self.assertEqual([r.plant_index for r in self.image.rsml_document.roots], [1, 1])
        self.assertEqual(self.image.plant_length, [0, 80])
        fresh = ImageData(self.image.date, self.image.path, 'p')
        self.assertTrue(load_replacement(fresh))
        self.assertEqual(fresh.plant_length, [0, 80])
        edit_roots(fresh, fresh.rsml_document, {0}, delete=True)
        self.assertEqual(len(fresh.rsml_document.roots), 1)
        files = export_series_rsml(ImageSeries('p', [fresh]), Config(), Path(self.temp.name) / 'out')
        exported = parse_rsml(files[0].read_bytes())
        self.assertEqual(len(exported.roots), 1)
        self.assertEqual(exported.roots[0].plant_index, 1)

    def test_edit_parent_does_not_delete_or_reassign_unselected_children(self):
        from root_tracker.io.root_editing import edit_roots
        doc = parse_rsml(b'<rsml><scene><plant id="1"><root id="a"><geometry><polyline><point x="1" y="1"/></polyline></geometry><root id="b"><geometry><polyline><point x="2" y="2"/></polyline></geometry></root></root></plant><plant id="2"/></scene></rsml>')
        edit_roots(self.image, doc, {0}, plant_index=1)
        roots = {r.id: r for r in self.image.rsml_document.roots}
        self.assertEqual(roots['b'].plant_index, 0)
        self.assertIsNone(roots['b'].parent_index)
        self.assertEqual(roots['a'].plant_index, 1)
        doc = self.image.rsml_document
        index = next(i for i, r in enumerate(doc.roots) if r.id == 'a')
        edit_roots(self.image, doc, {index}, delete=True)
        self.assertEqual([r.id for r in self.image.rsml_document.roots], ['b'])

    def test_edited_later_frame_preserves_sequence_metadata(self):
        import xml.etree.ElementTree as ET
        from root_tracker.io.root_editing import editable_document, edit_roots
        doc = editable_document(self.image, group='experiment/p', image_index=7)
        edit_roots(self.image, doc, {0}, plant_index=1)
        xml = ET.fromstring(self.image.rsml_document.source_bytes)
        self.assertEqual(xml.findtext('metadata/time-sequence/index'), '7')
        self.assertEqual(xml.findtext('metadata/time-sequence/label'), 'experiment/p')
        self.assertEqual(xml.findtext('metadata/file-key'), 'experiment/p/7/roots.png')

    def test_mask_after_manual_edit_updates_geometry_and_restores_corrections(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import restore_measurements
        from root_tracker.pipeline import RootTrackingPipeline
        from root_tracker.config import Config
        from root_tracker.io.rsml_export import export_series_rsml
        edit_roots(self.image, editable_document(self.image), {0}, plant_index=1)
        self.series.user_mask = np.zeros((100, 100), np.uint8)
        self.series.user_mask[35:46, :40] = 255
        config = Config(n_clusters=2)
        records = RootTrackingPipeline(config).track_and_analyze_series(self.series, save_images=False)
        self.assertLess(self.image.total_length, 80)
        self.assertEqual(len(self.image.rsml_document.roots), 3)
        self.assertTrue(all(r.plant_index == 1 for r in self.image.rsml_document.roots))
        self.assertEqual(records[1]['plant_total_length'], self.image.total_length)
        files = export_series_rsml(self.series, config, Path(self.temp.name) / 'masked')
        exported = parse_rsml(files[0].read_bytes())
        self.assertEqual(len(exported.roots), 3)
        # Editing a remaining fragment must not erase the masked geometry.
        edit_roots(self.image, self.image.rsml_document, {0}, plant_index=0)
        restore_measurements(self.series)
        self.assertLess(self.image.plant_length[0], 40)
        self.series.user_mask = None
        restore_measurements(self.series)
        self.assertEqual(self.image.plant_length, [40, 40])
        fresh = ImageData(self.image.date, self.image.path, 'p')
        self.assertTrue(load_replacement(fresh))
        self.assertEqual(fresh.plant_length, [40, 40])
        # Clearing the mask must never resurrect an explicitly deleted root.
        edit_roots(fresh, fresh.rsml_document, {0}, delete=True)
        restored = ImageSeries('p', [fresh])
        restored.user_mask = np.full((100, 100), 255, np.uint8)
        restore_measurements(restored)
        self.assertEqual(fresh.total_length, 0)
        restored.user_mask = None
        restore_measurements(restored)
        self.assertEqual(fresh.total_length, 40)

    def test_reset_manual_changes_to_import_preserves_original_import_after_reload(self):
        from root_tracker.io.root_editing import edit_roots
        from root_tracker.io.rsml_replacement import replace_roots, reset_manual_roots
        original = parse_rsml(b'<rsml><scene><plant id="1"><root id="a"><geometry><polyline><point x="1" y="1"/><point x="1" y="10"/></polyline></geometry></root></plant></scene></rsml>')
        replace_roots(self.image, original)
        edit_roots(self.image, original, {0}, delete=True)
        fresh = ImageData(self.image.date, self.image.path, 'p')
        load_replacement(fresh)
        self.assertEqual(fresh.total_length, 0)
        self.assertTrue(reset_manual_roots(fresh))
        self.assertEqual(fresh.rsml_document.source_bytes, original.source_bytes)
        self.assertIsNone(fresh.rsml_unmasked_document)
        self.assertEqual(fresh.total_length, 9)

    def test_restoring_earlier_mask_keeps_full_paths_and_filtered_fragments(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import restore_measurements, reset_manual_roots
        full = {key: value.copy() for key, value in self.image.rsml_samples.items()}
        mask = np.zeros((100, 100), np.uint8)
        mask[35:46, :40] = 255
        for filtered_away in (False, True):
            self.image.rsml_unmasked_samples = full
            self.image.rsml_samples = {0: np.empty((0, 2), dtype=np.int32) if filtered_away else
                                      full[0][(full[0][:, 1] < 35) | (full[0][:, 1] > 45)], 1: full[1]}
            doc = editable_document(self.image)
            selected = {i for i, root in enumerate(doc.roots) if root.plant_index == 1}
            edit_roots(self.image, doc, selected, plant_index=1, mask=mask)
            self.series.user_mask = None
            restore_measurements(self.series)
            self.assertEqual(self.image.plant_length, [40, 40])
            self.assertEqual(len(self.image.rsml_document.roots), 2)
            reset_manual_roots(self.image)

    def test_masked_native_edit_respects_current_ownership_instead_of_old_baseline(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        from root_tracker.io.rsml_replacement import reset_manual_roots
        full = {key: value.copy() for key, value in self.image.rsml_samples.items()}
        mask = np.zeros((100, 100), np.uint8)
        mask[80:, :] = 255
        for shared in (False, True):
            self.image.rsml_unmasked_samples = full
            self.image.rsml_samples = {0: full[0] if shared else np.empty((0, 2), np.int32),
                                       1: full[0]}
            doc = editable_document(self.image)
            selected = {i for i, root in enumerate(doc.roots) if root.plant_index == 1}
            edit_roots(self.image, doc, selected, delete=True, mask=mask)
            remaining = [r for r in self.image.rsml_document.roots if r.points[0][0] == 20]
            self.assertEqual(len(remaining), 1 if shared else 0)
            reset_manual_roots(self.image)

    def test_failed_save_leaves_live_results_unchanged(self):
        from root_tracker.io.root_editing import editable_document, edit_roots
        doc = editable_document(self.image)
        with patch('root_tracker.io.rsml_replacement.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                edit_roots(self.image, doc, {0}, delete=True)
        self.assertIsNone(self.image.rsml_document)
        self.assertEqual(len(self.image.rsml_samples[0]), 41)


class RootInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_browsing_defers_geometry_until_selecting_the_current_image(self):
        from copy import copy
        from root_tracker.gui.main_window import MainWindow
        from root_tracker.gui.workflow_bar import WorkflowStep
        from root_tracker.config import Config
        from root_tracker.io.root_editing import editable_document
        fixture = RootEditingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        other = copy(fixture.image)
        other.path = str(Path(fixture.temp.name) / 'other.png')
        other.rsml_samples = {0: np.array([[80, y] for y in range(20, 61)])}
        fixture.series.images.append(other)
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(Config(n_clusters=2))
        self.addCleanup(window.close)
        window._remember_session = False
        window._current_series = fixture.series
        window._workflow_bar.blockSignals(True)
        window._workflow_bar.set_current_step(WorkflowStep.TRACK)
        window._workflow_bar.blockSignals(False)
        window.show()
        with patch('root_tracker.gui.root_editor.editable_document', wraps=editable_document) as build:
            for image in (fixture.image, other, fixture.image, other):
                window._current_image = image
                window._display_image(image)
            self.assertEqual(build.call_count, 0, 'Browsing must not trace roots or round-trip XML')
            self.app.processEvents()
            viewer = window._image_viewer
            view = viewer._view
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.mapFromScene(QPointF(80, 40)))
            self.assertEqual(viewer.selected_roots, {0})
            self.assertEqual(viewer._root_document.roots[0].points[0], (80., 20.))
            self.assertEqual(build.call_count, 1)
            QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.mapFromScene(QPointF(80, 40)))
            self.assertEqual(build.call_count, 1)
            window._root_editor.present(None, WorkflowStep.PREPROCESS)
            self.assertEqual(viewer.selected_roots, set())
            self.assertFalse(viewer._root_editing_active())

    def test_click_multiselect_and_drag_copy_target_assignment(self):
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.io.root_editing import editable_document
        fixture = RootEditingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(600, 600)
        viewer.show()
        viewer.set_image(fixture.image.image)
        viewer.set_root_document(editable_document(fixture.image))
        self.app.processEvents()
        view = viewer._view
        a = view.mapFromScene(QPointF(20, 40))
        b = view.mapFromScene(QPointF(60, 40))
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=a)
        self.assertEqual(viewer.selected_roots, {0})
        QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.ControlModifier, b)
        self.assertEqual(viewer.selected_roots, {0, 1})
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=a)
        assignments = []
        viewer.root_assignment_requested.connect(assignments.append)
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=a)
        QTest.mouseMove(view.viewport(), b)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=b)
        self.assertEqual(assignments, [1])
        viewer.set_root_document(None)
        self.assertEqual(viewer.selected_roots, set())

    def test_panel_edits_refresh_measurements_and_clear_on_navigation(self):
        from root_tracker.gui.main_window import MainWindow, ProcessingState
        from root_tracker.gui.workflow_bar import WorkflowStep
        from root_tracker.config import Config
        fixture = RootEditingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(Config(n_clusters=2))
        self.addCleanup(window.close)
        window._remember_session = False
        window._workflow_bar.blockSignals(True)
        window._workflow_bar.set_current_step(WorkflowStep.TRACK)
        window._workflow_bar.blockSignals(False)
        window._settings_panel.set_step(WorkflowStep.TRACK)
        window._current_image = fixture.image
        window._current_series = fixture.series
        window._display_image(fixture.image)
        viewer = window._image_viewer
        viewer.select_roots({0})
        panel = window._settings_panel
        self.assertFalse(panel._selected_roots_group.isHidden())
        panel.root_assignment_requested.emit(1)
        self.assertEqual(fixture.image.plant_length, [0, 80])
        self.assertEqual(len(fixture.series.pipeline_state.last_statistics), 2)
        self.assertEqual(viewer.selected_roots, {0})
        window._state = ProcessingState.TRACKING
        panel.root_delete_requested.emit()
        self.assertEqual(len(fixture.image.rsml_document.roots), 2)
        window._state = ProcessingState.IDLE
        panel._delete_roots_btn.click()
        self.assertEqual(fixture.image.plant_length, [0, 40])
        self.assertTrue(panel._selected_roots_group.isHidden())
        viewer.select_roots({0})
        window._root_editor.present(None, WorkflowStep.PREPROCESS)
        self.assertEqual(viewer.selected_roots, set())

    def test_mask_mode_does_not_select_and_empty_drop_does_not_assign(self):
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.gui.masking_tools import MaskTool
        from root_tracker.io.root_editing import editable_document
        fixture = RootEditingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(600, 600)
        viewer.show()
        viewer.set_image(fixture.image.image)
        viewer.set_root_document(editable_document(fixture.image))
        self.app.processEvents()
        view = viewer._view
        a = view.mapFromScene(QPointF(20, 40))
        empty = view.mapFromScene(QPointF(80, 80))
        assignments = []
        viewer.root_assignment_requested.connect(assignments.append)
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=a)
        QTest.mouseMove(view.viewport(), empty)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=empty)
        self.assertEqual(assignments, [])
        viewer.select_roots(set())
        viewer.set_mask_tool(MaskTool.BRUSH)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=a)
        self.assertEqual(viewer.selected_roots, set())
        self.assertGreater(np.count_nonzero(viewer._working_mask), 0)

    def test_rectangle_and_brush_select_without_changing_exclusions(self):
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.gui.masking_tools import MaskTool
        from root_tracker.io.root_editing import editable_document
        fixture = RootEditingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(600, 600)
        viewer.show()
        viewer.set_image(fixture.image.image)
        viewer.set_root_document_loader(lambda: editable_document(fixture.image))
        mask = np.zeros((100, 100), np.uint8)
        viewer.set_mask_data(mask, mask.copy())
        changes = []
        viewer.mask_modified.connect(lambda: changes.append(True))
        self.app.processEvents()
        view = viewer._view
        viewer.set_mask_tool(MaskTool.RECT_SELECT)
        start = view.mapFromScene(QPointF(10, 30))
        end = view.mapFromScene(QPointF(70, 50))
        QTest.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
        self.assertEqual(viewer.selected_roots, {0, 1})
        viewer.set_mask_tool(MaskTool.BRUSH_SELECT, 5)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.mapFromScene(QPointF(20, 40)))
        self.assertEqual(viewer.selected_roots, {0})
        self.assertEqual(changes, [])
        np.testing.assert_array_equal(viewer.get_working_mask(), mask)
