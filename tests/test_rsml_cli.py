from copy import deepcopy
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from root_tracker.config import Config
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline
from test_rsml import EXTENDED


class RSMLCliTests(unittest.TestCase):
    def test_inspection_copy_and_bad_input_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.rsml'
            copy = Path(directory) / 'copy.rsml'
            source.write_bytes(EXTENDED)
            result = subprocess.run([sys.executable, 'scripts/rsml.py', str(source), '--output', str(copy)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(copy.read_bytes(), EXTENDED)
            self.assertIn('2 roots', result.stdout)
            source.write_bytes(b'not xml')
            result = subprocess.run([sys.executable, 'scripts/rsml.py', str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_pipeline_cli_advertises_opt_in_export(self):
        result = subprocess.run([sys.executable, 'scripts/run_pipeline.py', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--rsml-output', result.stdout)

    def test_tracking_without_detected_plants_exports_empty_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(n_clusters=1)
            config.data.output = directory
            image = ImageData(datetime(2026, 1, 1), 'empty.png', 'empty')
            image.image = np.zeros((10, 10, 3), np.uint8)
            image.process = np.zeros((10, 10), np.uint8)
            series = ImageSeries('empty', [image])
            pipeline = RootTrackingPipeline(config)
            pipeline.rsml_output = str(Path(directory) / 'rsml')
            with patch.object(RootTrackingPipeline, 'preprocess_series'), patch.object(RootTrackingPipeline, 'register_series'):
                self.assertEqual(pipeline.process_series_wrapper(series), [])
            from root_tracker.io.rsml import read_rsml
            files = list(Path(pipeline.rsml_output).rglob('*.rsml'))
            self.assertEqual(len(files), 1)
            self.assertEqual(read_rsml(files[0]).roots, ())

    def test_real_tracking_statistics_unchanged_by_export_and_worker_owns_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(n_clusters=1)
            config.data.output = directory
            image = ImageData(datetime(2026, 1, 1), 'root.png', 'plant')
            image.image = np.zeros((180, 180, 3), np.uint8)
            image.process = np.zeros((180, 180), np.uint8)
            cv2.line(image.process, (80, 35), (80, 160), 255, 5)
            image.positions_x, image.positions_y, image.green_areas = [80], [20], [100]
            original = ImageSeries('plant', [image])
            pipeline = RootTrackingPipeline(config)
            with patch.object(RootTrackingPipeline, 'preprocess_series'), patch.object(RootTrackingPipeline, 'register_series'):
                baseline = pipeline.process_series_wrapper(deepcopy(original))
                self.assertTrue(baseline)
                pipeline.rsml_output = str(Path(directory) / 'rsml')
                actual = pipeline._process_series_for_parallel(deepcopy(original))
                self.assertEqual([s.to_dict() for s in actual], [s.to_dict() for s in baseline])
                self.assertEqual(len(list(Path(pipeline.rsml_output).rglob('*.rsml'))), 1)
                # Explicitly requested exports cannot disappear behind the legacy
                # wrapper's print-and-return-empty failure behavior.
                with self.assertRaises(FileExistsError):
                    pipeline._process_series_for_parallel(deepcopy(original))


if __name__ == '__main__':
    unittest.main()
