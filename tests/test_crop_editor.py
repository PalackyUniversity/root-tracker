import unittest
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from root_tracker.gui.crop_overlay import AngleSnap, CropOverlay
from root_tracker.preprocessing.roi import pixel_box


class CropEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def test_snap_release_ignores_same_angle_until_another_snap(self):
        snap = AngleSnap()
        self.assertEqual(snap.update(14), 15)
        self.assertEqual(snap.update(17), 15)
        self.assertEqual(snap.update(21), 21)
        self.assertEqual(snap.update(16), 16)
        self.assertEqual(snap.update(14), 14)
        self.assertEqual(snap.update(29), 30)
        self.assertEqual(snap.update(15), 15)
        snap.reset()
        self.assertEqual(snap.update(14), 15)

    def test_passing_a_stop_between_mouse_events_does_not_recapture_on_return(self):
        snap = AngleSnap()
        self.assertEqual(snap.update(10), 10)
        self.assertEqual(snap.update(20), 20)
        self.assertEqual(snap.update(16), 16)
        self.assertEqual(snap.update(29), 30)
        self.assertEqual(snap.update(16), 15)

    def test_snap_wraparound_and_negative_angles(self):
        snap = AngleSnap()
        self.assertEqual(snap.update(-1), 0)
        self.assertEqual(snap.update(7), 7)
        self.assertEqual(snap.update(359), 359)
        self.assertEqual(snap.update(344), 345)
        self.assertEqual(snap.update(359), 360)

    def test_resize_keeps_opposite_corner_fixed_on_rotated_box(self):
        from root_tracker.preprocessing.roi import corners
        item = CropOverlay((400, 600), (.5, .5, .4, .4, 25))
        before = corners(item.box, item.image_shape)
        item.begin_drag('se', QPointF(*before[2]))
        item.drag_to(QPointF(*(before[2] + [15, 20])))
        after = corners(item.box, item.image_shape)
        np.testing.assert_allclose(after[0], before[0], atol=1e-8)
        self.assertGreater(pixel_box(item.box, item.image_shape)[2], 240)
        item.end_drag()

    def test_drag_just_outside_each_corner_rotates_instead_of_resizing(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.preprocessing.roi import corners
        import math
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(800, 600)
        viewer.show()
        self.app.processEvents()
        viewer.set_image(np.zeros((400, 600, 3), np.uint8))
        for index, direction in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            viewer.set_crop((400, 600), (.5, .5, .4, .4, 0.))
            item = viewer._crop_overlay
            scale = viewer._view.transform().m11()
            start = corners(item.box, item.image_shape)[index] + np.array(direction)*13/scale
            center = np.array([300, 200])
            angle = math.radians(22)
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            end = center + rotation @ (start-center)
            QTest.mousePress(viewer._view.viewport(), Qt.MouseButton.LeftButton,
                             pos=viewer._view.mapFromScene(QPointF(*start)))
            QTest.mouseMove(viewer._view.viewport(), viewer._view.mapFromScene(QPointF(*end)), delay=10)
            QTest.mouseRelease(viewer._view.viewport(), Qt.MouseButton.LeftButton,
                               pos=viewer._view.mapFromScene(QPointF(*end)))
            self.assertAlmostEqual(item.box[4], 22, delta=1)
            np.testing.assert_allclose(item.box[:4], (.5, .5, .4, .4), atol=1e-8)

    def test_corner_hover_cursors_are_directional_and_follow_box_rotation(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from root_tracker.gui.image_viewer import ImageViewer
        from root_tracker.preprocessing.roi import corners
        import math
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(800, 600)
        viewer.show()
        self.app.processEvents()
        viewer.set_image(np.zeros((400, 600, 3), np.uint8))
        cursors = []
        for angle in (0, 30):
            viewer.set_crop((400, 600), (.5, .5, .4, .4, angle))
            scale = viewer._view.transform().m11()
            a = math.radians(angle)
            basis = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            for i, direction in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
                point = corners(viewer._crop_overlay.box, (400, 600))[i] + basis @ np.array(direction)*13/scale
                QTest.mouseMove(viewer._view.viewport(), viewer._view.mapFromScene(QPointF(*point)), delay=5)
                cursor = viewer._view.viewport().cursor()
                self.assertEqual(cursor.shape(), Qt.CursorShape.BitmapCursor)
                cursors.append(cursor.pixmap().toImage())
        for i in range(4):
            self.assertNotEqual(cursors[i], cursors[(i+1)%4])
            self.assertNotEqual(cursors[i], cursors[i+4])

    def test_move_clamps_box_and_emits_on_release(self):
        item = CropOverlay((400, 600), (.5, .5, .4, .4, 0))
        changes = []
        item.changed.connect(changes.append)
        item.begin_drag('move', QPointF(300, 200))
        item.drag_to(QPointF(900, 800))
        self.assertEqual(changes, [])
        item.end_drag()
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(item.box[0], .8)
        self.assertAlmostEqual(item.box[1], .8)

    def test_angle_label_adapts_to_background_when_crop_moves(self):
        from root_tracker.gui.image_viewer import ImageViewer
        viewer = ImageViewer()
        self.addCleanup(viewer.close)
        viewer.resize(800, 600)
        viewer.show()
        image = np.zeros((400, 600, 3), np.uint8)
        image[:, 300:] = 255
        viewer.set_image(image)
        viewer.set_crop(image.shape, (.2, .6, .15, .2, 0))
        self.app.processEvents()
        viewer.grab()
        overlay = viewer._crop_overlay
        self.assertEqual(overlay.label.brush().color().name(), '#ffffff')
        overlay.box = (.7, .6, .15, .2, 0)
        overlay._layout_handles()
        viewer.grab()
        self.assertEqual(overlay.label.brush().color().name(), '#000000')
        viewer._set_zoom(2)
        viewer.grab()
        self.assertEqual(overlay.label.brush().color().name(), '#000000')

if __name__ == '__main__':
    unittest.main()
