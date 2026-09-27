"""Mask tool choices must send the correct editing operation to the viewer."""
import unittest

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QAbstractButton, QSpinBox
from root_tracker.config import Config
from root_tracker.gui.settings_panel import SettingsPanel
from root_tracker.gui.workflow_bar import WorkflowStep


class MaskControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.panel = SettingsPanel(Config())
        self.panel.set_step(WorkflowStep.TRACK)
        self.addCleanup(self.panel.close)
        self.events = []
        self.panel.mask_tool_changed.connect(lambda tool, size: self.events.append((tool, size)))

    def button(self, text):
        matches = [b for b in self.panel.findChildren(QAbstractButton) if b.text() == text]
        self.assertEqual(len(matches), 1, f'Expected one {text} control')
        return matches[0]

    def diameter(self):
        matches = [s for s in self.panel.findChildren(QSpinBox) if s.suffix() == ' px']
        self.assertEqual(len(matches), 1, 'Brush and eraser should share one diameter control')
        return matches[0]

    def test_shapes_and_operations_deliver_the_correct_tool_and_size(self):
        self.panel.set_mask_available(True)
        self.button('Brush').click()
        self.assertEqual(self.events[-1], ('brush', 100))
        self.diameter().setValue(47)
        self.assertEqual(self.events[-1], ('brush', 47))
        self.button('Restore').click()
        self.assertEqual(self.events[-1], ('brush_eraser', 47))
        self.diameter().setValue(63)
        self.assertEqual(self.events[-1], ('brush_eraser', 63))
        self.button('Rectangle').click()
        self.assertEqual(self.events[-1], ('rect_eraser', 0))
        self.assertFalse(self.diameter().isEnabled())
        self.button('Exclude').click()
        self.assertEqual(self.events[-1], ('rectangle', 0))
        self.button('Brush').click()
        self.assertEqual(self.events[-1], ('brush', 63))
        self.assertFalse(self.panel._is_dirty, 'Selecting tools must not modify the mask')

    def test_group_reset_selects_pan_and_disables_editing_options(self):
        self.button('Brush').click()
        self.panel.deselect_mask_tools()
        self.assertTrue(self.button('Pan').isChecked())
        self.assertIn(self.events[-1][0], ('none', 'move'))
        self.assertEqual(self.events[-1][1], 0)
        self.assertFalse(self.diameter().isEnabled())
        self.assertFalse(self.button('Exclude').isEnabled())
        self.assertFalse(self.button('Restore').isEnabled())
        self.panel.set_step(WorkflowStep.LOAD)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.panel.deselect_mask_tools()  # Must not access deleted Track widgets.
        self.panel.set_step(WorkflowStep.TRACK)
        self.assertTrue(self.button('Pan').isChecked())

    def test_action_and_diameter_do_not_overlap_in_narrow_sidebar(self):
        self.panel.resize(280, 480)
        self.panel.show()
        self.app.processEvents()
        diameter = self.diameter()
        for name in ('Exclude', 'Restore'):
            button = self.button(name)
            bottom = button.mapTo(self.panel, button.rect().bottomLeft()).y()
            top = diameter.mapTo(self.panel, diameter.rect().topLeft()).y()
            self.assertGreater(top, bottom)

    def test_clear_mask_keeps_the_existing_draft_action(self):
        self.panel.set_mask_available(True)
        clears = []
        self.panel.mask_erase_all_requested.connect(lambda: clears.append(True))
        self.button('Clear mask').click()
        self.assertEqual(clears, [True])

    def test_empty_mask_disables_clear_and_restore(self):
        self.button('Brush').click()
        self.assertFalse(self.button('Clear mask').isEnabled())
        self.assertFalse(self.button('Restore').isEnabled())
        self.assertTrue(self.button('Exclude').isEnabled())
        self.panel.set_mask_available(True)
        self.assertTrue(self.button('Clear mask').isEnabled())
        self.assertTrue(self.button('Restore').isEnabled())
        self.panel.set_mask_available(False)
        self.assertFalse(self.button('Clear mask').isEnabled())
        self.assertFalse(self.button('Restore').isEnabled())

    def test_mask_availability_survives_controls_rebuild(self):
        self.panel.set_mask_available(True)
        self.panel.set_step(WorkflowStep.LOAD)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.panel.set_step(WorkflowStep.TRACK)
        self.button('Brush').click()
        self.assertTrue(self.button('Clear mask').isEnabled())
        self.assertTrue(self.button('Restore').isEnabled())

    def test_pan_clears_action_and_select_uses_brush_or_rectangle(self):
        self.assertFalse(self.button('Exclude').isChecked())
        self.assertFalse(self.button('Restore').isChecked())
        self.assertFalse(self.button('Select').isChecked())
        self.button('Brush').click()
        self.button('Select').click()
        self.assertEqual(self.events[-1], ('brush_select', 100))
        self.button('Rectangle').click()
        self.assertEqual(self.events[-1], ('rect_select', 0))
        self.button('Pan').click()
        self.assertTrue(all(not self.button(name).isChecked() for name in ('Exclude', 'Restore', 'Select')))
        self.button('Brush').click()
        self.assertTrue(self.button('Select').isChecked())
        self.panel.set_temporary_mask_pan(True)
        self.assertTrue(all(not self.button(name).isChecked() for name in ('Exclude', 'Restore', 'Select')))
        self.panel.set_temporary_mask_pan(False)
        self.assertTrue(self.button('Select').isChecked())

    def test_pan_state_change_disables_and_clears_all_actions(self):
        self.button('Brush').click()
        self.button('Select').click()
        self.button('Pan').setChecked(True)
        self.panel.set_mask_available(True)
        self.panel.set_processing(True, allow_mask=True)
        self.panel.set_processing(False)
        for name in ('Select', 'Exclude', 'Restore'):
            self.assertFalse(self.button(name).isChecked())
            self.assertFalse(self.button(name).isEnabled())
        self.assertEqual(self.events[-1], ('move', 0))

    def test_select_is_first_action(self):
        layout = self.panel._mask_controls._operation_row.layout()
        self.assertEqual([layout.itemAt(i).widget().text() for i in range(layout.count())],
                         ['Select', 'Exclude', 'Restore'])
