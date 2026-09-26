"""Presets use the CLI schema, persist safely, and cover every setting."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QMessageBox, QLineEdit
from root_tracker.config import Config
from root_tracker.gui.presets import PresetStore
from root_tracker.gui.dialogs.presets_dialog import ConfigEditor, PresetsDialog
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.workflow_bar import WorkflowStep


class PresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = PresetStore(Path(self.temp.name) / 'presets')
        self.config = Config()
        self.config.data.input = self.temp.name
        self.config.data.output = str(Path(self.temp.name)/'output')
        self.config.data.statistics = str(Path(self.temp.name)/'stats.csv')

    def test_yaml_roundtrip_all_configuration_fields(self):
        self.config.load_roi = (.5, .6, .7, .8, 15.)
        self.config.preprocess_roi = (.4, .5, .8, .7, -30.)
        self.config.gui.load_crop_editing = False
        self.config.gui.preprocess_crop_editing = True
        self.config.gui.auto_apply = False
        self.config.crop.background_enabled = False
        self.store.save('Example', self.config)
        loaded = Config.from_yaml(self.store.path('Example'))
        self.assertEqual(loaded.to_dict(), self.config.to_dict())
        self.assertNotIn('!!python', self.store.path('Example').read_text())
        self.store.delete('Example')
        self.assertEqual(self.store.names(), [])

    def test_all_fields_have_editor_controls(self):
        editor = ConfigEditor(self.config)
        self.addCleanup(editor.deleteLater)
        expected = set()
        for key, value in self.config.to_dict().items():
            if isinstance(value, dict):
                expected.update(f'{key}.{name}' for name in value)
            else:
                expected.add(key)
        self.assertEqual(set(editor._readers), expected)
        self.assertEqual(editor.value().to_dict(), self.config.to_dict())
        editor.findChild(QDoubleSpinBox, 'rotation').setValue(-17.5)
        editor.findChild(QSpinBox, 'n_clusters').setValue(8)
        result = editor.value()
        self.assertEqual(result.rotation, -17.5)
        self.assertEqual(result.n_clusters, 8)

    def test_validation_leaves_existing_preset_untouched(self):
        self.store.save('Example', self.config)
        original = self.store.path('Example').read_bytes()
        self.config.crop.top_ratio = .9
        self.config.crop.bottom_ratio = .1
        with self.assertRaises(ValueError):
            self.store.save('Example', self.config)
        self.assertEqual(self.store.path('Example').read_bytes(), original)
        for name in ('../outside', '/absolute', 'a/b', ''):
            with self.assertRaises(ValueError):
                self.store.path(name)

    def test_deleted_bundled_presets_are_not_recreated(self):
        bundled = Path(self.temp.name)/'configs'
        bundled.mkdir()
        self.config.to_yaml(bundled/'Example.yaml')
        self.store.initialize(bundled)
        self.store.delete('Example')
        self.store.initialize(bundled)
        self.assertNotIn('Example', self.store.names())

    def test_manager_create_edit_delete(self):
        dialog = PresetsDialog(self.store, self.config)
        self.addCleanup(dialog.deleteLater)
        with patch('root_tracker.gui.dialogs.presets_dialog.QInputDialog.getText', return_value=('New preset', True)):
            dialog.new()
        self.assertIn('New preset', self.store.names())
        dialog.editor.findChild(QSpinBox, 'n_clusters').setValue(9)
        dialog.save()
        self.assertEqual(self.store.load('New preset').n_clusters, 9)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.delete()
        self.assertEqual(self.store.names(), [])

    def test_apply_switches_paths_workflow_and_retains_config_references(self):
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(Config())
        self.addCleanup(window.close)
        original_reference = window._config
        self.config.gui.load_crop_editing = False
        self.config.gui.auto_apply = False
        self.config.n_clusters = 9
        self.config.crop.background_enabled = False
        with patch.object(window, '_reload_images') as reload_images, patch.object(window, '_get_preset_store', return_value=self.store), patch.object(window._settings, 'setValue'):
            window._apply_preset(self.config, 'Example')
            reload_images.assert_called_once()
        self.assertIs(window._config, original_reference)
        self.assertIs(window._settings_panel._config, original_reference)
        self.assertEqual(window._config.data.input, self.temp.name)
        self.assertEqual(window._config.data.output, self.config.data.output)
        self.assertEqual(window._config.n_clusters, 9)
        self.assertFalse(window._settings_panel.crop_editing())
        self.assertFalse(window._settings_panel._background_enabled.isChecked())
        self.assertFalse(window._auto_preview_action.isChecked())
        self.assertEqual(window._workflow_bar.get_current_step(), WorkflowStep.LOAD)

    def test_manager_import_export_is_cli_compatible(self):
        source = Path(self.temp.name)/'source.yaml'
        target = Path(self.temp.name)/'export.yaml'
        self.config.gui.load_crop_editing = False
        self.config.to_yaml(source)
        dialog = PresetsDialog(self.store, self.config)
        self.addCleanup(dialog.deleteLater)
        with patch('root_tracker.gui.dialogs.presets_dialog.QFileDialog.getOpenFileName', return_value=(str(source), '')), patch('root_tracker.gui.dialogs.presets_dialog.QInputDialog.getText', return_value=('Imported', True)):
            dialog.import_yaml()
        self.assertEqual(self.store.load('Imported').to_dict(), self.config.to_dict())
        with patch('root_tracker.gui.dialogs.presets_dialog.QFileDialog.getSaveFileName', return_value=(str(target), '')):
            dialog.export_yaml()
        self.assertEqual(Config.from_yaml(target).to_dict(), self.config.to_dict())

    def test_crop_editing_defaults_are_independent_of_detection(self):
        from root_tracker.gui.settings_panel import SettingsPanel
        self.config.gui.load_crop_editing = False
        self.config.gui.preprocess_crop_editing = True
        panel = SettingsPanel(self.config)
        self.addCleanup(panel.deleteLater)
        self.assertFalse(panel.crop_editing())
        self.assertTrue(panel._background_enabled.isChecked())
        panel.set_step(WorkflowStep.PREPROCESS)
        self.assertTrue(panel.crop_editing())
        self.assertEqual(panel._crop_edit_btn.text(), 'Finish editing')

    def test_use_preset_reloads_real_dataset_with_its_filename_rules(self):
        import cv2
        import numpy as np
        folder = Path(self.temp.name)/'photos'
        folder.mkdir()
        cv2.imwrite(str(folder/'plant.01-01-26.png'), np.full((60, 80, 3), (200, 80, 20), np.uint8))
        self.config.data.input = str(folder)
        self.config.data.detect_barcodes = False
        self.config.rotation = 0
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(Config())
        self.addCleanup(window.close)
        with patch.object(window, '_get_preset_store', return_value=self.store), patch.object(window._settings, 'setValue'):
            window._apply_preset(self.config, 'Example')
        self.assertEqual(list(window._series_dict), ['plant'])
        self.assertIsNotNone(window._current_image)
        self.assertEqual(Path(window._current_image.path).parent, folder)
        self.assertIs(window._pipeline.config, window._config)
        self.assertIsNotNone(window._image_viewer._pixmap_item)

    def test_saved_preset_startup_and_explicit_config_precedence(self):
        self.config.n_clusters = 9
        self.store.save('Startup', self.config)
        settings = Mock()
        settings.value.side_effect = lambda key, default=None: str(self.store.path('Startup')) if key == 'active_preset' else default
        with patch('root_tracker.gui.main_window.QSettings', return_value=settings), patch.object(MainWindow, '_get_preset_store', return_value=self.store), patch.object(MainWindow, '_reload_images') as reload_images, patch.object(MainWindow, '_load_last_folder') as last_folder:
            window = MainWindow()
            self.addCleanup(window.close)
            self.assertEqual(window._config.n_clusters, 9)
            reload_images.assert_called_once()
            last_folder.assert_not_called()
            reload_images.reset_mock()
            explicit = MainWindow(Config(n_clusters=2))
            self.addCleanup(explicit.close)
            self.assertEqual(explicit._config.n_clusters, 2)
            reload_images.assert_not_called()

    def test_menu_offers_management_and_saved_presets(self):
        self.store.save('Example', self.config)
        with patch.object(MainWindow, '_load_last_folder'):
            window = MainWindow(Config())
        self.addCleanup(window.close)
        self.assertTrue(window._presets_menu.actions())
        with patch.object(window, '_get_preset_store', return_value=self.store):
            window._refresh_presets_menu()
        self.assertEqual([action.text() for action in window._presets_menu.actions() if not action.isSeparator()], ['Manage presets…', 'Example'])

    def test_first_launch_uses_defaults_and_marks_active_preset(self):
        self.config.crop.top_ratio = .1
        self.config.crop.bottom_ratio = .85
        self.store.save('Defaults', self.config)
        settings = Mock()
        settings.value.side_effect = lambda key, default=None: default
        with patch('root_tracker.gui.main_window.QSettings', return_value=settings), patch.object(MainWindow, '_get_preset_store', return_value=self.store), patch.object(MainWindow, '_reload_images'):
            window = MainWindow()
            self.addCleanup(window.close)
            self.assertEqual(window._active_preset_name, 'Defaults')
            self.assertEqual(window._config.crop.top_ratio, .1)
            self.assertEqual(window._config.crop.bottom_ratio, .85)
            window._refresh_presets_menu()
            checked = [action.text() for action in window._presets_menu.actions() if action.isChecked()]
            self.assertEqual(checked, ['Defaults'])
        dialog = PresetsDialog(self.store, self.config, active_name='Defaults')
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.list.currentItem().text(), 'Defaults')
        self.assertFalse(dialog.list.currentItem().icon().isNull())
        self.assertEqual(dialog._loaded_name, 'Defaults')

    def test_decimal_fields_hide_trailing_zeros_without_losing_precision(self):
        from PySide6.QtCore import QLocale
        from root_tracker.gui.dialogs.presets_dialog import CompactDoubleSpinBox
        widget = CompactDoubleSpinBox()
        self.addCleanup(widget.deleteLater)
        widget.setDecimals(6)
        widget.setRange(-360, 360)
        for locale, separator in ((QLocale('en_US'), '.'), (QLocale('de_DE'), ',')):
            widget.setLocale(locale)
            for value, expected in ((0., '0'), (.03, '0'+separator+'03'), (180., '180'), (.123456, '0'+separator+'123456')):
                widget.setValue(value)
                self.assertEqual(widget.text(), expected)
                self.assertEqual(widget.value(), value)

    def test_defaults_is_the_in_vitro_yaml_and_migration_preserves_edits(self):
        bundled = Path(self.temp.name)/'configs'
        bundled.mkdir()
        self.config.crop.top_ratio = .1
        self.config.crop.bottom_ratio = .85
        self.config.to_yaml(bundled/'in_vitro.yaml')
        self.store.initialize(bundled)
        self.assertEqual(self.store.names(), ['Defaults'])
        self.assertEqual(self.store.load('Defaults').crop.top_ratio, .1)
        self.assertEqual(self.store.load('Defaults').crop.bottom_ratio, .85)
        self.assertTrue(self.store.path('Defaults').is_file())
        # Simulate an installation created by the previous version.
        (self.store.directory/'.defaults-from-in-vitro').unlink()
        self.store.save('Defaults', Config())
        self.config.n_clusters = 8
        self.store.save('in_vitro', self.config)
        self.store.initialize(bundled)
        self.assertEqual(self.store.names(), ['Defaults'])
        self.assertEqual(self.store.load('Defaults').n_clusters, 8)
        self.assertEqual(self.store.load('Defaults').crop.top_ratio, .1)
        self.store.delete('Defaults')
        self.store.initialize(bundled)
        self.assertEqual(self.store.names(), [])

    def test_remembered_in_vitro_selection_follows_renamed_yaml(self):
        self.store.save('Defaults', self.config)
        settings = Mock()
        settings.value.side_effect = lambda key, default=None: str(self.store.path('in_vitro')) if key == 'active_preset' else default
        with patch('root_tracker.gui.main_window.QSettings', return_value=settings), patch.object(MainWindow, '_get_preset_store', return_value=self.store), patch.object(MainWindow, '_reload_images'):
            window = MainWindow()
            self.addCleanup(window.close)
            self.assertEqual(window._active_preset_name, 'Defaults')
            settings.setValue.assert_called_with('active_preset', str(self.store.path('Defaults')))


if __name__ == '__main__':
    unittest.main()
