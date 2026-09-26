"""Filename dates use year-month-day in the default experiment."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from root_tracker.config import Config
from root_tracker.gui.presets import PresetStore
from root_tracker.io.loader import ImageLoader


class DateDefaultsTests(unittest.TestCase):
    def test_default_configs_parse_consecutive_days_in_one_year(self):
        bundled = Path(__file__).resolve().parents[1] / 'configs/in_vitro.yaml'
        for config in (Config(), Config.from_yaml(bundled)):
            loader = ImageLoader(config)
            for day in (27, 28, 29, 30):
                with self.subTest(format=config.data.date_format, day=day):
                    self.assertEqual(
                        loader.parse_filename(f'RT_26_2-1.26-04-{day}.JPG'),
                        ('RT_26_2-1', datetime(2026, 4, day)),
                    )

    def test_existing_defaults_migrate_once_without_changing_other_settings(self):
        bundled = Path(__file__).resolve().parents[1] / 'configs'
        with TemporaryDirectory() as directory:
            store = PresetStore(directory)
            config = Config()
            config.data.date_format = '%d-%m-%y'
            config.n_clusters = 9
            store.save('Defaults', config)
            store.save('Custom', config)
            (store.directory / '.initialized').touch()
            (store.directory / '.defaults-from-in-vitro').touch()
            store.initialize(bundled)
            updated = store.load('Defaults')
            self.assertEqual(updated.data.date_format, '%y-%m-%d')
            self.assertEqual(updated.n_clusters, 9)
            self.assertEqual(store.load('Custom').data.date_format, '%d-%m-%y')
            # A later deliberate edit must survive subsequent startups.
            store.save('Defaults', config)
            store.initialize(bundled)
            self.assertEqual(store.load('Defaults').data.date_format, '%d-%m-%y')

    def test_date_format_change_invalidates_chronological_processing(self):
        config = Config()
        config.data.date_format = '%d-%m-%y'
        old = (config.preprocess_config_hash(), config.tracking_config_hash())
        config.data.date_format = '%y-%m-%d'
        self.assertNotEqual(config.preprocess_config_hash(), old[0])
        self.assertNotEqual(config.tracking_config_hash(), old[1])


if __name__ == '__main__':
    unittest.main()
