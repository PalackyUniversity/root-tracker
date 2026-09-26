"""Window construction must not issue an early native maximize request."""
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication

from root_tracker.gui.main_window import MainWindow


class WindowStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([
            'window-startup-test', '-platform', 'offscreen'])

    def setUp(self):
        # Avoid loading the user's last dataset or starting processing workers.
        with patch.object(MainWindow, '_load_last_folder'):
            self.window = MainWindow()
        self.addCleanup(self.window.close)

    def test_constructor_keeps_window_hidden_and_normal(self):
        self.assertFalse(self.window.isVisible())
        self.assertFalse(self.window.windowState() & Qt.WindowState.WindowMaximized)

    def test_first_show_and_move_preserve_normal_size(self):
        self.window.show()
        self.app.processEvents()
        self.assertFalse(self.window.isMaximized())
        size = self.window.size()
        self.window.move(self.window.pos() + QPoint(80, 50))
        self.app.processEvents()
        self.assertFalse(self.window.isMaximized())
        self.assertEqual(self.window.size(), size)


if __name__ == '__main__':
    unittest.main()
