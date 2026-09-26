from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QGraphicsPathItem

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow, ProcessingState
from root_tracker.gui.dialogs.rsml_dialog import RSMLDialog, RSMLExportWorker
from root_tracker.io.rsml import read_rsml
from test_rsml import EXTENDED
import test_rsml_export


class RSMLGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['rsml-test', '-platform', 'offscreen'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'test.rsml'
        self.source.write_bytes(EXTENDED)

    def test_dialog_displays_hierarchy_projection_and_saves_exact_copy(self):
        dialog = RSMLDialog(read_rsml(self.source))
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.tree.topLevelItemCount(), 1)
        main = dialog.tree.topLevelItem(0).child(0)
        self.assertEqual(main.child(0).text(0), 'lateral')
        self.assertEqual(sum(isinstance(i, QGraphicsPathItem) for i in dialog.scene.items()), 2)
        self.assertIn('XY projection', dialog.summary.text())
        output = Path(self.tmp.name) / 'copy.rsml'
        with patch('root_tracker.gui.dialogs.rsml_dialog.QFileDialog.getSaveFileName', return_value=(str(output), '')):
            dialog.save_copy()
        self.assertEqual(output.read_bytes(), EXTENDED)

    def test_context_replacement_is_authoritative_and_marked_in_tree(self):
        import test_rsml_replacement
        from root_tracker.pipeline import RootTrackingPipeline
        from root_tracker.gui.workflow_bar import WorkflowStep
        from PySide6.QtCore import Qt
        fixture = test_rsml_replacement.ReplacementTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(fixture.config)
        self.addCleanup(window.close)
        window._pipeline = RootTrackingPipeline(fixture.config)
        window._series_dict = {'p': fixture.series}
        window._image_tree.set_series(window._series_dict)
        window._current_series = fixture.series
        window._current_image = fixture.image
        with patch('root_tracker.gui.main_window.QFileDialog.getOpenFileName', return_value=(str(fixture.source), '')):
            window._on_replace_rsml(fixture.image)
        self.assertEqual(fixture.image.total_length, 15.)
        self.assertFalse(hasattr(window, '_rsml_import_action'))
        row = window._image_tree._tree.topLevelItem(0).child(0)
        for step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS, WorkflowStep.TRACK):
            window._image_tree.set_step(step, fixture.config)
            self.assertEqual(row.text(1), 'RSML')
            self.assertEqual(row.data(1, Qt.ItemDataRole.UserRole), 'rsml')
            self.assertIn('replaced', row.toolTip(1).lower())
        window._image_tree.set_pending_changes(fixture.series)
        fixture.image.barcode_mismatch = True
        window._image_tree.refresh_status()
        self.assertEqual(row.text(1), 'RSML')
        fixture.image.clear_preprocessing_results()
        window._image_tree.refresh_status()
        self.assertEqual(row.text(1), 'RSML')

    def test_context_menu_targets_clicked_image_and_not_groups(self):
        from root_tracker.gui.image_tree import ImageTree
        from root_tracker.gui.menus import RoundedMenu
        fixture = test_rsml_export.ExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        other = fixture.image.copy_for_editing()
        other.path = 'second.png'
        fixture.series.images.append(other)
        tree = ImageTree()
        self.addCleanup(tree.close)
        tree.set_series({fixture.series.group: fixture.series})
        tree.resize(500, 500)
        tree.show()
        tree._tree.expandAll()
        self.app.processEvents()
        group = tree._tree.topLevelItem(0)
        tree._tree.setCurrentItem(group.child(0))
        clicked = []
        replacements = []
        tree.rsml_replace_requested.connect(replacements.append)
        tree.rsml_export_requested.connect(clicked.append)
        menus = []
        def choose(menu, position):
            replace_actions = [a for a in menu.actions() if a.text() == 'Replace with RSML…']
            if replace_actions:
                replace_actions[0].trigger()
            actions = [a for a in menu.actions() if a.text() == 'Export to RSML…']
            menus.append(len(actions))
            if actions:
                actions[0].trigger()
        with patch.object(RoundedMenu, 'exec', choose):
            tree._show_context_menu(tree._tree.visualItemRect(group.child(1)).center())
            tree._show_context_menu(tree._tree.visualItemRect(group).center())
        self.assertEqual(menus, [1, 0])
        self.assertEqual(len(clicked), 1)
        self.assertIs(clicked[0], other)
        self.assertEqual(len(replacements), 1)
        self.assertIs(replacements[0], other)

    def test_export_worker_produces_files_and_reports_failure(self):
        fixture = test_rsml_export.ExportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        other = fixture.image.copy_for_editing()
        other.path = 'second.png'
        fixture.series.images.append(other)
        destination = fixture.base / 'export'
        worker = RSMLExportWorker([fixture.series], fixture.config, destination, image_index=1)
        results = []
        worker.completed.connect(lambda ok, text: results.append((ok, text)))
        worker.run()
        self.assertTrue(results[0][0])
        files = list(destination.rglob('*.rsml'))
        self.assertEqual(len(files), 1)
        self.assertIn('second', files[0].name)
        worker.run()
        self.assertFalse(results[1][0])
