"""Exclusion overlays preserve holes and coverage without Python pixel loops."""
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from root_tracker.gui.masking_tools import MaskOverlay


class MaskOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_fill_and_restored_hole_keep_their_alpha(self):
        mask = np.zeros((100, 120), np.uint8)
        mask[10:90, 10:110] = 255
        mask[40:60, 40:80] = 0
        mask[20:30, 20:30] = 1  # Antialiased brush edges are still exclusions.
        overlay = MaskOverlay(mask.shape)
        overlay.set_masks(None, mask)
        rendered = overlay.render().toImage()
        self.assertEqual((rendered.width(), rendered.height()), (120, 100))
        self.assertEqual(rendered.pixelColor(5, 5).alpha(), 0)
        self.assertEqual(rendered.pixelColor(25, 25).alpha(), 230)
        self.assertEqual(rendered.pixelColor(60, 50).alpha(), 0)
        self.assertEqual(rendered.pixelColor(90, 70).alpha(), 230)

    def test_cropped_overlay_has_scene_offset_and_clears_when_empty(self):
        mask = np.zeros((3000, 4000), np.uint8)
        mask[1200:1230, 2200:2250] = 255
        overlay = MaskOverlay(mask.shape)
        overlay.set_masks(None, mask)
        pixmap = overlay.render(cropped=True)
        self.assertEqual(overlay.offset, (2198, 1198))
        self.assertEqual((pixmap.width(), pixmap.height()), (54, 34))
        self.assertEqual(pixmap.toImage().pixelColor(25, 15).alpha(), 230)
        mask.fill(0)
        self.assertIsNone(overlay.render(cropped=True))

    def test_viewer_positions_cropped_overlay_and_removes_it_after_clear(self):
        from root_tracker.gui.image_viewer import ImageViewer
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.set_image(np.zeros((100, 200, 3), np.uint8))
        mask = np.zeros((100, 200), np.uint8)
        mask[20:40, 70:100] = 255
        viewer.set_mask_data(None, mask)
        item = viewer._mask_overlay_item
        self.assertEqual((item.pos().x(), item.pos().y()), (68, 18))
        self.assertEqual(item.pixmap().toImage().pixelColor(12, 12).alpha(), 230)
        viewer.set_mask_data(None, np.zeros_like(mask))
        self.assertIsNone(viewer._mask_overlay_item)
