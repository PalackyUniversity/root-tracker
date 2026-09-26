import unittest
from PySide6.QtWidgets import QApplication
from root_tracker.gui.color_range import ColorRangeControl


class ColorPickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_live_hsv_edit_and_cancel_restore_exact_achromatic_range(self):
        control = ColorRangeControl('Blue', (70, 0, 0), (140, 255, 255))
        self.addCleanup(control.close)
        original = control.values()
        changes = []
        control.range_changed.connect(lambda *values: changes.append(values))
        control.swatches[0].click()
        picker = control._picker
        self.assertEqual(picker.values(), original)
        self.assertTrue(control.preview.isChecked())
        picker.editors[0].controls[0].setValue(80)
        self.assertEqual(control.values()[0], (80, 0, 0))
        self.assertTrue(changes)
        picker.reject()
        self.assertEqual(control.values(), original)
        self.assertFalse(control.preview.isChecked())

    def test_accept_and_context_change_keep_live_edits(self):
        control = ColorRangeControl('Green', (28, 166, 50), (42, 255, 200))
        self.addCleanup(control.close)
        control.swatch.click()
        control._picker.editors[1].controls[2].setValue(180)
        control.finish_picker()
        self.assertIsNone(control._picker)
        self.assertEqual(control.values()[1], (42, 255, 180))

    def test_adjust_range_has_both_limits_and_no_panel_hsv_fields(self):
        from PySide6.QtWidgets import QSpinBox
        control = ColorRangeControl('Background color', (70, 0, 0), (140, 255, 255))
        self.addCleanup(control.close)
        self.assertEqual(control.findChildren(QSpinBox), [])
        control.swatch.click()
        picker = control._picker
        self.assertEqual(len(picker.editors), 2)
        picker.editors[0].controls[0].setValue(90)
        picker.editors[1].controls[0].setValue(130)
        self.assertEqual(control.values(), ((90, 0, 0), (130, 255, 255)))
        picker.reject()
        self.assertEqual(control.values(), ((70, 0, 0), (140, 255, 255)))

    def test_selection_ring_fits_at_all_four_extremes(self):
        from PySide6.QtCore import QRectF
        from root_tracker.gui.color_picker import SaturationValuePlane
        plane = SaturationValuePlane()
        plane.resize(280, 180)
        for saturation, value in ((0, 0), (0, 255), (255, 0), (255, 255)):
            plane.hsv = (110, saturation, value)
            center = plane.selection_position()
            ring = QRectF(center.x()-7, center.y()-7, 14, 14)
            self.assertTrue(QRectF(plane.rect()).contains(ring))

    def test_each_limit_requests_its_own_sample_and_does_not_commit_dialog(self):
        control = ColorRangeControl('Background', (70, 0, 0), (140, 255, 255))
        self.addCleanup(control.close)
        control.swatch.click()
        picker = control._picker
        picker.editors[1].sample_button.click()
        self.assertEqual(control.sample_index, 1)
        self.assertIs(control._picker, picker)
        self.assertFalse(picker.isVisible())
        control.apply_sample((115, 210, 220))
        self.assertEqual(control.values(), ((70, 0, 0), (115, 210, 220)))
        self.assertTrue(picker.isVisible())
        picker.reject()
        self.assertEqual(control.values(), ((70, 0, 0), (140, 255, 255)))

    def test_combined_control_supports_drag_typing_and_keyboard(self):
        from PySide6.QtCore import Qt, QPoint
        from PySide6.QtTest import QTest
        from root_tracker.gui.color_picker import HsvValueControl
        control = HsvValueControl(0, 'Hue')
        control.setRange(0, 179)
        control.resize(280, 48)
        control.show()
        self.addCleanup(control.close)
        QTest.mouseClick(control, Qt.MouseButton.LeftButton, pos=QPoint(140, 36))
        self.assertAlmostEqual(control.value(), 90, delta=2)
        control.entry.setFocus()
        control.entry.selectAll()
        QTest.keyClicks(control.entry, '125')
        QTest.keyClick(control.entry, Qt.Key.Key_Return)
        self.assertEqual(control.value(), 125)
        control.setFocus()
        QTest.keyClick(control, Qt.Key.Key_Right)
        self.assertEqual(control.value(), 126)

    def test_simplified_dialog_labels_and_text_only_buttons(self):
        from PySide6.QtWidgets import QLabel, QDialogButtonBox
        from root_tracker.gui.color_picker import HsvPicker
        picker = HsvPicker('Background color', (70, 0, 0), (140, 255, 255))
        self.addCleanup(picker.close)
        self.assertEqual(picker.findChildren(QLabel), [])
        self.assertEqual([c.accessibleName() for c in picker.editors[0].controls], ['Hue', 'Saturation', 'Value'])
        for button in picker.findChild(QDialogButtonBox).buttons():
            self.assertTrue(button.icon().isNull())

    def test_typing_upper_limit_preserves_partial_input_and_return_keeps_dialog_open(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from root_tracker.gui.color_picker import HsvPicker
        picker = HsvPicker('Plant color', (28, 166, 50), (42, 255, 200))
        picker.show()
        self.app.processEvents()
        self.addCleanup(picker.close)
        entry = picker.editors[1].controls[1].entry
        entry.setFocus()
        entry.selectAll()
        QTest.keyClicks(entry, '220')
        QTest.keyClick(entry, Qt.Key.Key_Return)
        self.assertEqual(picker.values()[1][1], 220)
        self.assertEqual(entry.text(), '220')
        self.assertTrue(picker.isVisible())
