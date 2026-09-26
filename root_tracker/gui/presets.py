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
        if marker.exists():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.path('Defaults').exists():
            self.save('Defaults', Config())
        for path in sorted(Path(bundled_directory).glob('*.yaml')):
            if not self.path(path.stem).exists():
                self.save(path.stem, Config.from_yaml(path))
        marker.touch()
