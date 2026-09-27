"""Metadata comparisons must identify outliers without treating missing tags as equal."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from root_tracker.io import image_metadata as metadata


class MetadataTests(unittest.TestCase):
    def test_majority_missing_and_tied_values(self):
        rows, warnings = metadata.compare_metadata([
            {'EXIF:FNumber': '4', 'EXIF:ISO': '100'},
            {'EXIF:FNumber': '4', 'EXIF:ISO': '200'},
            {'EXIF:FNumber': '8'},
        ])
        self.assertFalse(next(row for row in rows if row.key == 'Aperture').same)
        self.assertEqual(warnings[0], ['ISO sensitivity'])
        self.assertEqual(warnings[1], ['ISO sensitivity'])
        self.assertEqual(warnings[2], ['Aperture', 'ISO sensitivity'])

    def test_only_readable_photo_settings_are_compared(self):
        irrelevant = {'Sony:Sony_0x2025': '42', 'System:FileName': 'a.jpg',
                      'File:ImageWidth': '100', 'ExifIFD:DateTimeOriginal': 'today',
                      'IFD1:ThumbnailImage': 'binary', 'System:FilePermissions': 'rw',
                      'Sony:BatteryLevel': '50%', 'Sony:FocusPosition2': '121'}
        other = {key: 'different' for key in irrelevant}
        rows, warnings = metadata.compare_metadata([irrelevant, other])
        self.assertEqual(rows, [])
        self.assertEqual(warnings, [[], []])

    def test_friendly_fields_merge_duplicates_and_describe_units(self):
        rows, warnings = metadata.compare_metadata([{
            'ExifIFD:FNumber': '2.8', 'Composite:Aperture': '2.8',
            'Sony:SonyFNumber': '2.9', 'ExifIFD:ExposureTime': '1/200',
            'ExifIFD:WhiteBalance': 'Auto', 'Sony:FocusMode': 'Manual',
            'Sony:UnknownFocusThing': '42', 'IFD1:FNumber': '99',
        }])
        values = {row.key: row.values[0] for row in rows}
        self.assertEqual(values, {'Aperture': 'f/2.8', 'Exposure time': '1/200 s',
                                  'White balance': 'Auto', 'Focus mode': 'Manual'})
        self.assertEqual(warnings, [[]])

    def test_fallback_codes_become_words_and_unknown_values_are_hidden(self):
        rows, _ = metadata.compare_metadata([{'EXIF:WhiteBalance': '0',
                                             'EXIF:MeteringMode': '5',
                                             'Sony:FocusMode': 'Unknown (99)'}])
        self.assertEqual({r.key: r.values[0] for r in rows},
                         {'White balance': 'Auto', 'Light metering': 'Multi-segment'})

    def test_equal_and_empty_groups(self):
        self.assertEqual(metadata.compare_metadata([]), ([], []))
        rows, warnings = metadata.compare_metadata([{'EXIF:ISO': '100'}] * 2)
        self.assertTrue(rows[0].same)
        self.assertEqual(warnings, [[], []])

    def test_auto_white_balance_gains_still_reveal_color_changes(self):
        rows, warnings = metadata.compare_metadata([
            {'ExifIFD:WhiteBalance': 'Auto', 'Composite:RedBalance': '2', 'Composite:BlueBalance': '1.5'},
            {'ExifIFD:WhiteBalance': 'Auto', 'Composite:RedBalance': '2', 'Composite:BlueBalance': '1.5'},
            {'ExifIFD:WhiteBalance': 'Auto', 'Sony:WB_RGBLevels': '600 200 400'},
        ])
        values = {row.key: row for row in rows}
        self.assertEqual(values['White balance red gain'].values, ['2 ×', '2 ×', '3 ×'])
        self.assertEqual(values['White balance blue gain'].values, ['1.5 ×', '1.5 ×', '2 ×'])
        self.assertTrue(values['White balance'].same)
        self.assertEqual(warnings[:2], [[], []])
        self.assertEqual(warnings[2], ['White balance blue gain', 'White balance red gain'])

    def test_white_balance_shifts_have_named_directions(self):
        rows, _ = metadata.compare_metadata([{
            'Sony:WBShiftAB_GM': '1 1', 'Sony:WBShiftAB_GM_Precise': '2.5 -1.25',
            'Sony:ColorCompensationFilter': '-2', 'Sony:WhiteBalanceFineTune': '1',
            'ExifIFD:Contrast': 'Normal', 'Sony:Contrast': '2',
            'Sony:WB_RGBLevels': '100 0 200',
        }])
        values = {row.key: row.values[0] for row in rows}
        self.assertEqual(values['White balance warmth'], '2.5 toward amber')
        self.assertEqual(values['White balance tint'], '1.25 toward green')
        self.assertEqual(values['Color tint filter'], '2 toward green')
        self.assertEqual(values['White balance fine adjustment'], '1')
        self.assertEqual(values['Contrast adjustment'], '2')
        self.assertNotIn('White balance red gain', values)

    def test_focus_distance_at_infinity_is_readable(self):
        rows, _ = metadata.compare_metadata([{'Composite:FocusDistance2': 'inf'}])
        self.assertEqual(rows[0].values, ['Infinity'])

    def test_binary_values_compare_content_even_when_size_matches(self):
        self.assertNotEqual(metadata._text('base64:YWJj'), metadata._text('base64:eHl6'))
        self.assertEqual(metadata._text('base64:YWJj'), metadata._text(b'abc'))

    def test_pillow_fallback_reads_exif(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photo.jpg'
            exif = Image.Exif()
            exif[271] = 'Fallback camera'
            exif[33434] = 0.01
            Image.new('RGB', (16, 12)).save(path, exif=exif)
            with patch.object(metadata.shutil, 'which', return_value=None):
                values, error = metadata.read_metadata([str(path)])[str(path)]
            self.assertFalse(error)
            self.assertEqual(values['EXIF:Make'], 'Fallback camera')
            self.assertEqual(values['EXIF:ExposureTime'], '0.01')

    def test_real_exif_and_unreadable_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photo.jpg'
            exif = Image.Exif()
            exif[271] = 'Test camera'
            exif[33434] = 0.01
            Image.new('RGB', (16, 12)).save(path, exif=exif)
            result = metadata.read_metadata([str(path), str(Path(folder) / 'missing.jpg')])
            values, error = result[str(path)]
            self.assertFalse(error)
            self.assertIn('Test camera', values.values())
            self.assertTrue(any('ExposureTime' in key for key in values))
            self.assertTrue(result[str(Path(folder) / 'missing.jpg')][1])


class MetadataGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_warnings_toggle_and_panel_survives_steps(self):
        from datetime import datetime
        from root_tracker.config import Config
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.gui.image_metadata import MetadataController
        from root_tracker.gui.image_tree import ImageTree
        from root_tracker.gui.settings_panel import SettingsPanel
        from root_tracker.gui.workflow_bar import WorkflowStep
        images = [ImageData(datetime(2026, 1, i + 1), f'{i}.jpg', 'group',
                            camera_metadata={'EXIF:FNumber': value})
                  for i, value in enumerate(['4', '4', '8'])]
        series = ImageSeries('group', images)
        config = Config()
        config.data.detect_barcodes = False
        tree = ImageTree()
        tree.set_step(WorkflowStep.LOAD, config)
        tree.set_series({'group': series})
        panel = SettingsPanel(config)
        self.addCleanup(panel.close)
        self.addCleanup(tree.close)
        control = MetadataController()
        control.changed.connect(lambda: tree.set_metadata_warnings(control.warnings()))
        panel.metadata_panel.options_changed.connect(control.set_options)
        control.changed.connect(panel.metadata_panel.refresh)
        panel.metadata_panel.enabled.setChecked(True)
        control.set_series({'group': series})
        panel.metadata_panel.set_selection(series, images[2])
        group = tree._tree.topLevelItem(0)
        self.assertEqual([group.child(i).text(1) for i in range(3)], ['✓', '✓', '⚠'])
        self.assertIn('Aperture', group.child(2).toolTip(1))
        row = panel.metadata_panel.table.topLevelItem(0)
        self.assertEqual((row.text(1), row.text(2)), ('✗', 'f/8'))
        self.assertEqual(row.childCount(), 3)
        panel.set_step(WorkflowStep.TRACK)
        self.app.processEvents()
        panel.set_step(WorkflowStep.LOAD)
        self.assertEqual(panel.metadata_panel.table.topLevelItem(0).text(2), 'f/8')
        panel.metadata_panel.enabled.setChecked(False)
        self.assertEqual([group.child(i).text(1) for i in range(3)], ['✓', '✓', '✓'])
        panel.metadata_panel.enabled.setChecked(True)
        self.assertEqual(group.child(2).text(1), '⚠')
        images[2].rsml_document = object()
        tree.refresh_status()
        self.assertEqual(group.child(2).text(1), 'RSML')
        self.assertFalse(group.child(2).icon(0).isNull())
        panel.metadata_panel.enabled.setChecked(False)
        self.assertTrue(group.child(2).icon(0).isNull())

    def test_metadata_list_is_compact_and_collapses_when_disabled(self):
        from datetime import datetime
        from PySide6.QtWidgets import QWidget, QVBoxLayout
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.gui.image_metadata import MetadataPanel
        parent = QWidget()
        layout = QVBoxLayout(parent)
        panel = MetadataPanel()
        panel.enabled.setChecked(True)
        layout.addWidget(panel)
        image = ImageData(datetime(2026, 1, 1), 'a.jpg', 'group',
                          camera_metadata={'EXIF:ISO': '100', 'EXIF:FNumber': '4'})
        panel.set_selection(ImageSeries('group', [image]))
        parent.resize(450, 500)
        parent.show()
        self.addCleanup(parent.close)
        self.app.processEvents()
        height = panel.height()
        self.assertLess(panel.table.height(), 300)
        parent.resize(450, 1400)
        self.app.processEvents()
        self.assertEqual(panel.height(), height)
        panel.enabled.setChecked(False)
        self.app.processEvents()
        self.assertFalse(panel.table.isVisible())
        self.assertFalse(panel.summary.isVisible())
        self.assertLess(panel.height(), height)
        panel.enabled.setChecked(True)
        self.app.processEvents()
        self.assertTrue(panel.table.isVisible())
        self.assertEqual(panel.table.topLevelItemCount(), 2)

    def test_background_extraction_populates_images(self):
        from datetime import datetime
        from PySide6.QtTest import QTest
        from root_tracker.models import ImageData, ImageSeries
        from root_tracker.gui.image_metadata import MetadataController
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photo.jpg'
            Image.new('RGB', (16, 12)).save(path)
            image = ImageData(datetime(2026, 1, 1), str(path), 'group')
            controller = MetadataController()
            controller.set_options(True)
            controller.set_series({'group': ImageSeries('group', [image])})
            for _ in range(200):
                if image.camera_metadata is not None:
                    break
                QTest.qWait(10)
            self.assertIsNotNone(image.camera_metadata)
            self.assertFalse(image.camera_metadata_error)
            self.assertFalse(controller._jobs)
