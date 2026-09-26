"""Selecting an unchecked image must display it when barcode work finishes."""
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow, ProcessingState
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline


class PreviewCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        config = Config()
        config.data.input = folder.name
        config.data.output = folder.name
        config.data.detect_barcodes = True
        with patch.object(MainWindow, '_load_last_folder'):
            self.window = MainWindow(config)
        self.addCleanup(self.window.close)
        self.window._pipeline = RootTrackingPipeline(config)
        self.window._auto_preview_action.setChecked(True)
        images = []
        for i, color in enumerate(((0, 0, 255), (255, 0, 0))):
            path = str(Path(folder.name) / f'{i}.png')
            cv2.imwrite(path, np.full((40, 60, 3), color, dtype=np.uint8))
            images.append(ImageData(datetime(2026, 1, i + 1), path, 'group'))
        self.series = ImageSeries('group', images)
        self.window._series_dict = {'group': self.series}
        self.window._image_tree.set_series(self.window._series_dict)

    def click_and_wait(self):
        tree = self.window._image_tree._tree
        tree.setCurrentItem(tree.topLevelItem(0).child(1))
        worker = self.window._worker  # Retain until the underlying QThread exits.
        initial_item = self.window._image_viewer._pixmap_item
        initial_color = (initial_item.pixmap().toImage().pixelColor(20, 20)
                         if initial_item is not None else None)
        try:
            for _ in range(300):
                self.app.processEvents()
                if self.window._state == ProcessingState.IDLE:
                    break
                QTest.qWait(10)
            self.assertEqual(self.window._state, ProcessingState.IDLE)
        finally:
            if worker is not None:
                worker.cancel()
                worker.wait()
        self.assertIsNotNone(initial_color, 'Show the selected photo while scanning')
        self.assertEqual((initial_color.red(), initial_color.green(), initial_color.blue()), (0, 0, 255))
        self.assertTrue(all(image.barcode_detected for image in self.series.images))
        self.assertIs(self.window._image_tree.get_selected_data(), self.series.images[1])

    def assert_blue_preview(self):
        item = self.window._image_viewer._pixmap_item
        self.assertIsNotNone(item, 'Selected image must be shown after processing')
        color = item.pixmap().toImage().pixelColor(20, 20)
        self.assertEqual((color.red(), color.green(), color.blue()), (0, 0, 255))

    def test_first_selection_populates_preview_after_barcode_detection(self):
        self.click_and_wait()
        self.assert_blue_preview()
        self.assertTrue(self.window._image_viewer._barcode_items)
        self.assertIn('NO BARCODE DETECTED', self.window._image_viewer._barcode_items[0].text())

    def test_new_selection_replaces_previous_preview_after_barcode_detection(self):
        self.window._image_viewer.set_image(np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8))
        self.click_and_wait()
        self.assert_blue_preview()


if __name__ == '__main__':
    unittest.main()
