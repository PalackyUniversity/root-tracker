"""Cursor anchoring and view preservation, including images smaller than the view."""
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
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
