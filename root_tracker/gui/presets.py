"""User-owned YAML presets; bundled configurations are copied only on first use."""
from copy import deepcopy
from pathlib import Path
import os
import tempfile

from ..config import Config


class PresetStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def names(self):
        return sorted(path.stem for path in self.directory.glob('*.yaml'))

    def path(self, name):
        if not name.strip() or name != name.strip() or name in ('.', '..') or any(c in name for c in '/\\\0'):
            raise ValueError('Enter a preset name without slashes or surrounding spaces')
        return self.directory / (name + '.yaml')

    def load(self, name):
        return Config.from_yaml(self.path(name))

    def save(self, name, config):
        path = self.path(name)
        config = deepcopy(config)
        config._validate()
        config.data.resolve_paths(config.base_path)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.preset-', suffix='.yaml', dir=self.directory)
        os.close(fd)
        try:
            config.to_yaml(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def delete(self, name):
        self.path(name).unlink()

    def initialize(self, bundled_directory):
        marker = self.directory / '.initialized'
        migration = self.directory / '.defaults-from-in-vitro'
        bundled_directory = Path(bundled_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            for path in sorted(bundled_directory.glob('*.yaml')):
                name = 'Defaults' if path.stem == 'in_vitro' else path.stem
                if not self.path(name).exists():
                    self.save(name, Config.from_yaml(path))
            marker.touch()
            migration.touch()
        elif not migration.exists():
            # Replace the old generic Defaults with the existing In vitro YAML,
            # preserving any edits made to that preset. Do this only once so a
            # later user deletion is respected and never recreated on startup.
            previous = self.path('in_vitro')
            if previous.exists():
                os.replace(previous, self.path('Defaults'))
            elif self.path('Defaults').exists() and (bundled_directory / 'in_vitro.yaml').exists():
                self.save('Defaults', Config.from_yaml(bundled_directory / 'in_vitro.yaml'))
            migration.touch()
        date_migration = self.directory / '.defaults-year-month-day'
        if not date_migration.exists():
            # Repair the shipped date order without replacing other preset edits.
            if self.path('Defaults').exists():
                config = self.load('Defaults')
                if (config.data.filename_template == '{group}.{date}'
                        and config.data.date_format == '%d-%m-%y'):
                    config.data.date_format = '%y-%m-%d'
                    self.save('Defaults', config)
            date_migration.touch()
