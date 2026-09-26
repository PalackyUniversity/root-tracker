"""Cursor anchoring and view preservation, including images smaller than the view."""
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from root_tracker.gui.image_viewer import ImageViewer


class ViewNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.viewer = ImageViewer()
        self.viewer.resize(900, 650)
        self.viewer.show()
        self.addCleanup(self.viewer.close)
        self.viewer.set_image(np.zeros((400, 600, 3), np.uint8))
        self.app.processEvents()

    def test_wheel_keeps_cursor_point_when_image_is_smaller_than_viewport(self):
        viewer = self.viewer
        viewer._set_zoom(.2)
        view = viewer._view
        for point, delta in [(QPoint(390, 280), 120), (QPoint(470, 330), -120),
                             (QPoint(180, 160), 120)]:
            before = view.mapToScene(point)
            event = QWheelEvent(QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
                                QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                                Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            view.wheelEvent(event)
            after = view.mapFromScene(before)
            self.assertLessEqual((after - point).manhattanLength(), 2)

    def test_cursor_anchor_remains_stable_across_fit_boundary(self):
        viewer = self.viewer
        viewer.set_image(np.zeros((2000, 3000, 3), np.uint8))
        viewer._set_zoom(.2)
        view = viewer._view
        point = QPoint(550, 380)
        for delta in [120] * 18 + [-120] * 18:
            before = view.mapToScene(point)
            event = QWheelEvent(QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
                                QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                                Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            view.wheelEvent(event)
            self.assertLessEqual((view.mapFromScene(before) - point).manhattanLength(), 2)

    def test_refresh_preserves_zoom_and_scene_center(self):
        viewer = self.viewer
        viewer._set_zoom(2)
        viewer._view.centerOn(220, 175)
        before_scale = viewer._view.transform().m11()
        before = viewer._view.mapToScene(viewer._view.viewport().rect().center())
        viewer.set_image(np.ones((400, 600, 3), np.uint8), preserve_view=True)
        after = viewer._view.mapToScene(viewer._view.viewport().rect().center())
        self.assertAlmostEqual(viewer._view.transform().m11(), before_scale)
        self.assertLessEqual((after - before).manhattanLength(), 1)

    def test_alt_drag_over_crop_and_handles_pans_without_editing(self):
        viewer = self.viewer
        view = viewer._view
        box = (.5, .5, .4, .4, 0.)
        viewer.set_crop((400, 600), box)
        crop = viewer._crop_overlay
        points = [QPointF(300, 200), crop.handles['se'].scenePos(),
                  crop.handles['rotate'].scenePos(),
                  crop.handles['nw'].scenePos() - QPointF(13, 13) / view.transform().m11()]
        changes = []
        viewer.crop_changed.connect(changes.append)
        for point in points:
            with self.subTest(point=point):
                start = view.mapFromScene(point)
                end = start + QPoint(40, 25)
                before = view.mapToScene(view.viewport().rect().center())
                QTest.mouseMove(view.viewport(), start)
                QTest.keyPress(view, Qt.Key.Key_Alt)
                QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.AltModifier, start)
                QTest.mouseMove(view.viewport(), end, delay=10)
                QTest.keyRelease(view, Qt.Key.Key_Alt)
                end += QPoint(10, 5)
                QTest.mouseMove(view.viewport(), end, delay=10)
                QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
                self.assertEqual(crop.box, box)
                after = view.mapToScene(view.viewport().rect().center())
                self.assertGreater((after - before).manhattanLength(), 10)
        self.assertEqual(changes, [])
        # A normal drag still edits the crop after releasing Alt.
        start = view.mapFromScene(QPointF(300, 200))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), start + QPoint(30, 20), delay=10)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start + QPoint(30, 20))
        self.assertNotEqual(crop.box, box)
