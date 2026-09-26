"""Both export entry points must prepare the whole dataset before writing CSV."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from PySide6.QtWidgets import QApplication

from root_tracker.config import Config
from root_tracker.gui.main_window import MainWindow
from root_tracker.gui.workflow_bar import WorkflowStep
from root_tracker.models import ImageData, ImageSeries
from root_tracker.pipeline import RootTrackingPipeline


class ResultsExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['test', '-platform', 'offscreen'])

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = Config()
        config.data.input = self.directory.name
        config.data.output = self.directory.name
        with patch.object(MainWindow, '_load_last_folder'):
            self.window = MainWindow(config)
        self.addCleanup(self.window.close)
        self.window._pipeline = RootTrackingPipeline(config)
        self.window._series_dict = {
            group: ImageSeries(group, [ImageData(datetime(2026, 1, 1), f'{group}.png', group)])
            for group in ('ready', 'unprocessed')
        }
        self.complete_series(self.window._series_dict['ready'])
        self.window._export_action.setEnabled(True)
        self.path = Path(self.directory.name) / 'statistics.csv'

    def complete_series(self, series):
        state = series.pipeline_state
        state.tracked = True
        state.preprocessed = True
        state.preprocess_config_hash = self.window._config.preprocess_config_hash()
        state.tracking_config_hash = self.window._config.tracking_config_hash()
        state.last_statistics = [{'group': series.group, 'length': 123}]

    def run_export(self, entry, batch_result=True, destination=None):
        def finish_batch():
            if batch_result:
                for series in self.window._series_dict.values():
                    self.complete_series(series)
            return batch_result

        # Replace multiprocessing and modal dialogs; exercise real export and CSV writing.
        with patch.object(self.window, '_track_all_groups', side_effect=finish_batch), \
                patch('root_tracker.gui.main_window.QFileDialog.getSaveFileName',
                      return_value=(str(self.path) if destination is None else destination, '')), \
                patch('root_tracker.gui.main_window.QMessageBox.information'), \
                patch('root_tracker.gui.main_window.QMessageBox.critical') as error:
            if entry == 'menu':
                self.window._export_action.trigger()
            else:
                self.window._workflow_bar.set_current_step(WorkflowStep.TRACK)
                self.window._on_next_step_clicked()
            error.assert_not_called()

    def test_menu_exports_rows_from_unprocessed_groups(self):
        self.run_export('menu')
        self.assertEqual(pd.read_csv(self.path)['group'].tolist(), ['ready', 'unprocessed'])

    def test_button_exports_rows_from_unprocessed_groups(self):
        self.run_export('button')
        self.assertEqual(pd.read_csv(self.path)['group'].tolist(), ['ready', 'unprocessed'])

    def test_menu_does_not_write_partial_csv_after_batch_cancellation(self):
        self.run_export('menu', batch_result=False)
        self.assertFalse(self.path.exists())

    def test_menu_save_dialog_cancellation_does_not_write_csv(self):
        self.run_export('menu', destination='')
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
