"""Barcode file loading, fallback, and original-coordinate regression checks."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from pyzbar import pyzbar
from root_tracker.io.barcode import BarcodeReader
from root_tracker.config import Config
from root_tracker.models import ImageData
from root_tracker.pipeline import RootTrackingPipeline


class BarcodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # pyzbar distributes its own real CODE128 fixture with the package.
        fixture = Path(pyzbar.__file__).parent / 'tests' / 'code128.png'
        cls.symbol = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        if cls.symbol is not None:
            # Isolate one of the fixture's two independent barcodes.
            cls.symbol = cls.symbol[530:646, 17:381]
        cls.expected = 'Foramenifera'

    def make_image(self, bottom=True):
        if self.symbol is None:
            self.skipTest('Installed pyzbar omits its CODE128 test fixture')
        symbol = cv2.resize(self.symbol, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        height, width = symbol.shape
        image = np.full((height * 4, width + 200), 255, np.uint8)
        y = height * 2 if bottom else 20
        image[y:y+height, 100:100+width] = symbol
        return image, (100, y, width, height)

    def test_zero_area_fast_detection_uses_full_reader(self):
        from types import SimpleNamespace
        image, _ = self.make_image()
        decode = pyzbar.decode
        first = True

        def fast_zero_box_then_real(image, **kwargs):
            nonlocal first
            if first:
                first = False
                return [SimpleNamespace(data=b'invalid-fast-result', rect=(3, 5, 0, 0))]
            return decode(image, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sample.jpg')
            self.assertTrue(cv2.imwrite(path, image))
            with patch('root_tracker.io.barcode.pyzbar.decode', side_effect=fast_zero_box_then_real):
                text, rect = BarcodeReader().read_file(path)
            self.assertEqual(text, self.expected)
            self.assertGreater(rect[2], 0)
            self.assertGreater(rect[3], 0)

    def test_zero_area_decode_is_retained_if_full_reader_fails(self):
        from types import SimpleNamespace
        image, _ = self.make_image()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sample.jpg')
            self.assertTrue(cv2.imwrite(path, image))
            reader = BarcodeReader()
            decoded = SimpleNamespace(data=self.expected.encode(), rect=(3, 5, 0, 0))
            with patch('root_tracker.io.barcode.pyzbar.decode', return_value=[decoded]), \
                    patch.object(reader, 'read_fast', return_value=('', None)):
                text, rect = reader.read_file(path)
            self.assertEqual(text, self.expected)
            self.assertEqual(rect[2:], (0, 0))

    def test_file_reader_bottom_barcode_returns_original_coordinates(self):
        image, expected_box = self.make_image()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sample.JPG')
            self.assertTrue(cv2.imwrite(path, image))
            reader = BarcodeReader()
            with patch.object(reader, 'read_fast', side_effect=AssertionError('Expected fast path')):
                text, rect = reader.read_file(path)
        self.assertEqual(text, self.expected)
        x, y, w, h = rect
        sx, sy, sw, sh = expected_box
        self.assertTrue(sx <= x < x+w <= sx+sw)
        self.assertTrue(sy <= y < y+h <= sy+sh)

    def test_large_odd_jpeg_preserves_original_coordinates(self):
        small, (sx, sy, sw, sh) = self.make_image()
        image = np.full((4001, 6001), 255, np.uint8)
        x0, y0 = 2300, 3100
        image[y0:y0+sh, x0:x0+sw] = small[sy:sy+sh, sx:sx+sw]
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'odd.jpeg')
            self.assertTrue(cv2.imwrite(path, image))
            reader = BarcodeReader()
            with patch.object(reader, 'read_fast', side_effect=AssertionError('Expected fast path')):
                text, (x, y, w, h) = reader.read_file(path)
            self.assertEqual(text, self.expected)
            self.assertTrue(x0 <= x < x+w <= x0+sw)
            self.assertTrue(y0 <= y < y+h <= y0+sh)

    def test_top_barcode_survives_fast_path_failure(self):
        image, _ = self.make_image(bottom=False)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'top.jpg')
            self.assertTrue(cv2.imwrite(path, image))
            self.assertEqual(BarcodeReader().read_file(path)[0], self.expected)

    def test_non_jpeg_uses_original_reader(self):
        image, _ = self.make_image()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sample.png')
            self.assertTrue(cv2.imwrite(path, image))
            reader = BarcodeReader()
            expected = reader.read_fast(cv2.imread(path))
            self.assertEqual(reader.read_file(path), expected)

    def test_pipeline_reports_actual_code_and_filename_mismatch(self):
        image, _ = self.make_image()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'wrong-label.jpg')
            self.assertTrue(cv2.imwrite(path, image))
            config = Config()
            config.data.output = folder
            pipeline = RootTrackingPipeline(config)
            item = ImageData(datetime(2026, 1, 1), path, 'wrong-label')
            self.assertTrue(pipeline.detect_barcode_in_image(item))
            self.assertEqual(item.barcode_read, self.expected)
            self.assertTrue(item.barcode_detected)
            self.assertTrue(item.barcode_mismatch)
            self.assertFalse(item.barcode_not_found)
            item.barcode = self.expected
            self.assertFalse(pipeline.detect_barcode_in_image(item))
            self.assertFalse(item.barcode_mismatch)

    def test_parallel_reads_overlap_and_preserve_individual_results(self):
        from threading import Barrier
        # Both readers must be active before either returns. A serial
        # implementation breaks the barrier and cannot produce these results.
        barrier = Barrier(2, timeout=3)
        items = [ImageData(datetime(2026, 1, 1), '/first.jpg', 'first'),
                 ImageData(datetime(2026, 1, 1), '/second.jpg', 'wrong')]
        with tempfile.TemporaryDirectory() as folder:
            config = Config()
            config.data.output = folder
            pipeline = RootTrackingPipeline(config)
            def read(path):
                barrier.wait()
                return Path(path).stem, (1, 2, 30, 40)
            with patch.object(pipeline.barcode_reader, 'read_file', side_effect=read), \
                    patch('root_tracker.pipeline.os.cpu_count', return_value=2):
                completed = list(pipeline.iter_detect_barcodes(items, max_workers=2))
        self.assertEqual({id(item) for item in completed}, {id(item) for item in items})
        self.assertEqual([item.barcode_read for item in items], ['first', 'second'])
        self.assertEqual([item.barcode_mismatch for item in items], [False, True])
        self.assertTrue(all(item.barcode_detected for item in items))

    def test_parallel_reader_preserves_cached_and_real_barcode_results(self):
        image, _ = self.make_image()
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'label.jpg')
            self.assertTrue(cv2.imwrite(path, image))
            config = Config()
            config.data.output = folder
            pipeline = RootTrackingPipeline(config)
            items = [ImageData(datetime(2026, 1, 1), path, self.expected) for _ in range(3)]
            items[0].barcode_detected = True
            items[0].barcode_read = 'cached value'
            self.assertEqual(len(list(pipeline.iter_detect_barcodes(items))), 3)
            self.assertEqual([item.barcode_read for item in items], ['cached value', self.expected, self.expected])

    def test_cancelled_parallel_scan_does_not_schedule_remaining_images(self):
        from threading import Event
        stop = Event()
        with tempfile.TemporaryDirectory() as folder:
            config = Config()
            config.data.output = folder
            pipeline = RootTrackingPipeline(config)
            items = [ImageData(datetime(2026, 1, 1), f'/{i}.jpg', 'code') for i in range(8)]
            with patch.object(pipeline.barcode_reader, 'read_file', return_value=('code', (1, 2, 3, 4))):
                scan = pipeline.iter_detect_barcodes(items, cancelled=stop.is_set, max_workers=2)
                next(scan)
                stop.set()
                list(scan)
            self.assertLessEqual(sum(item.barcode_detected for item in items), 2)
            stop.set()
            remaining = [item for item in items if not item.barcode_detected]
            self.assertEqual(list(pipeline.iter_detect_barcodes(remaining, cancelled=stop.is_set)), [])
            self.assertFalse(any(item.barcode_detected for item in remaining))

    def test_blank_and_missing_file_never_invent_a_barcode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'blank.jpg')
            self.assertTrue(cv2.imwrite(path, np.full((200, 300), 255, np.uint8)))
            self.assertEqual(BarcodeReader().read_file(path), ('', None))
            with patch('root_tracker.io.barcode.cv2.imread', return_value=None):
                self.assertEqual(BarcodeReader().read_file('missing.jpg'), ('', None))


if __name__ == '__main__':
    unittest.main()
