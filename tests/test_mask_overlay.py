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

    def test_restore_reveals_image_during_drag_without_committing(self):
        from PySide6.QtCore import QPointF, QRectF, Qt
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtTest import QTest
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.gui.masking_tools import MaskTool

        for tool in (MaskTool.BRUSH_ERASER, MaskTool.RECT_ERASER):
            with self.subTest(tool=tool):
                viewer = ImageViewer()
                self.addCleanup(viewer.close)
                viewer.resize(500, 500)
                viewer.show()
                viewer.set_image(np.full((100, 100, 3), 255, np.uint8))
                mask = np.zeros((100, 100), np.uint8)
                mask[20:80, 20:80] = 255
                viewer.set_mask_data(mask.copy(), mask.copy())
                viewer.set_mask_tool(tool, 12)
                self.app.processEvents()
                viewer._set_zoom(3)
                changes = []
                viewer.mask_modified.connect(lambda: changes.append(True))

                def pixel(x, y):
                    image = QImage(100, 100, QImage.Format.Format_ARGB32)
                    image.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(image)
                    viewer._scene.render(painter, QRectF(0, 0, 100, 100), QRectF(0, 0, 100, 100))
                    painter.end()
                    return image.pixelColor(x, y)

                def position(x, y):
                    return viewer._view.mapFromScene(QPointF(x, y))

                self.assertLess(pixel(40, 40).red(), 50)
                QTest.mousePress(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position(40, 40))
                if tool == MaskTool.BRUSH_ERASER:
                    self.assertEqual(pixel(40, 40).red(), 255)
                QTest.mouseMove(viewer._view.viewport(), position(60, 60))
                self.assertEqual(pixel(50, 50).red(), 255)
                self.assertEqual(pixel(50, 50).green(), 255)
                self.assertLess(pixel(30, 60).red(), 50)
                np.testing.assert_array_equal(viewer.get_working_mask(), mask)
                self.assertEqual(changes, [])
                if tool == MaskTool.RECT_ERASER:
                    QTest.mouseMove(viewer._view.viewport(), position(45, 45))
                    self.assertLess(pixel(50, 50).red(), 50)
                    QTest.mouseMove(viewer._view.viewport(), position(60, 60))
                QTest.mouseRelease(viewer._view.viewport(), Qt.MouseButton.LeftButton, pos=position(60, 60))
                self.assertEqual(viewer.get_working_mask()[50, 50], 0)
                self.assertEqual(changes, [True])
                self.assertEqual(pixel(50, 50).red(), 255)
