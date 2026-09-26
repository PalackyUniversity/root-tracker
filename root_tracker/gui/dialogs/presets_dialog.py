"""Preset management and a complete editor for the YAML configuration schema."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from ..combo_box import RoundedComboBox as QComboBox
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QListWidget, QPushButton, QTabWidget,
    QWidget, QFormLayout, QScrollArea, QLineEdit, QSpinBox, QDoubleSpinBox,
    QLabel, QFileDialog, QInputDialog, QMessageBox, QGroupBox, QStyle, QFrame,
)
from ...config import Config
from ..color_range import ColorRangeControl
from ..right_checkbox import RightAlignedCheckBox
from ..theme import presets_stylesheet


LABELS = {
    'n_clusters': 'Origin counts', 'rotation': 'Initial rotation (degrees)',
    'load_roi': 'Manual Load crop', 'preprocess_roi': 'Crop within plate',
    'margin_top': 'Top margin', 'margin_bottom': 'Bottom margin',
    'margin_left': 'Left margin', 'margin_right': 'Right margin',
    'min_count': 'Minimum contour points', 'min_area': 'Minimum leaf area (pixels)',
    'top_ratio': 'Initial crop start (fraction)', 'bottom_ratio': 'Initial crop end (fraction)',
    'background_enabled': 'Enable plate search by color', 'background_region': 'Matching regions',
    'auto_apply': 'Auto-apply settings', 'load_crop_editing': 'Start Load with crop editing enabled',
    'preprocess_crop_editing': 'Start Preprocess with crop editing enabled',
    'margin_ratio': 'Registration search margin', 'enabled': 'Enable registration',
    'low': 'Low root threshold', 'high': 'High root threshold',
    'min_contour_area': 'Minimum contour area', 'min_contour_length': 'Minimum contour length',
    'filename_template': 'Filename template', 'date_format': 'Date format',
    'input': 'Input folder', 'output': 'Output folder', 'statistics': 'Statistics file',
    'detect_barcodes': 'Detect barcodes', 'cache_compressed': 'Compress disk cache',
}


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep editing precision without displaying insignificant trailing zeros."""
    def textFromValue(self, value):
        text = super().textFromValue(value)
        separator = self.locale().decimalPoint()
        if separator in text:
            zero = self.locale().zeroDigit()
            while text.endswith(zero):
                text = text[:-len(zero)]
            if text.endswith(separator):
                text = text[:-len(separator)]
        return text


class CropDefaults(QGroupBox):
    def __init__(self, value, parent=None):
        super().__init__('Use custom box', parent)
        self.setCheckable(True)
        self.setChecked(value is not None)
        layout = QFormLayout(self)
        self.controls = []
        for label, number in zip(('Center X', 'Center Y', 'Width', 'Height', 'Rotation'), value or (.5, .5, 1., 1., 0.)):
            widget = CompactDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(-36000 if label == 'Rotation' else .000001 if label in ('Width', 'Height') else 0,
                            36000 if label == 'Rotation' else 1000 if label in ('Width', 'Height') else 1)
            widget.setSingleStep(15 if label == 'Rotation' else .01)
            widget.setValue(number)
            layout.addRow(label, widget)
            self.controls.append(widget)
        self.setStatusTip('Center and size are fractions of the source image in Load, or the detected plate in Preprocess. Rotation is in degrees.')

    def value(self):
        return tuple(widget.value() for widget in self.controls) if self.isChecked() else None


class ConfigEditor(QTabWidget):
    """Every public Config field is represented; no separate preset schema."""
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setObjectName('presetEditor')
        self.setStyleSheet(presets_stylesheet(self.palette()))
        self.set_config(config)

    def set_config(self, config):
        while self.count():
            widget = self.widget(0)
            self.removeTab(0)
            widget.deleteLater()
        self._config = deepcopy(config)
        self._readers = {}
        groups = [('Geometry', '', config), ('Plate search', 'crop', config.crop),
                  ('Plants', 'green', config.green), ('Registration', 'registration', config.registration),
                  ('Tracking', 'threshold', config.threshold), ('Files', 'data', config.data),
                  ('Workflow', 'gui', config.gui)]
        for title, prefix, section in groups:
            page = QWidget()
            page.setObjectName('presetPage')
            form = QFormLayout(page)
            form.setContentsMargins(16, 16, 16, 16)
            form.setVerticalSpacing(12)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            entries = [(field, section, prefix) for field in fields(section)]
            if not prefix:
                entries += [(field, config.crop, 'crop') for field in fields(config.crop)
                            if field.name in ('top_ratio', 'bottom_ratio')]
            elif prefix == 'green':
                entries += [(field, config, '') for field in fields(config) if field.name == 'n_clusters']
            for field, owner, field_prefix in entries:
                name, value = field.name, getattr(owner, field.name)
                if name.startswith('_') or hasattr(value, '__dataclass_fields__') or 'hsv_' in name:
                    continue
                if (not prefix and name == 'n_clusters') or (prefix == 'crop' and name in ('top_ratio', 'bottom_ratio')):
                    continue
                key = f'{field_prefix}.{name}' if field_prefix else name
                label = LABELS.get(name, name.replace('_', ' ').capitalize())
                if isinstance(value, bool):
                    widget = RightAlignedCheckBox(label)
                    widget.setChecked(value)
                    form.addRow(widget)
                    reader = widget.isChecked
                elif name == 'background_region':
                    widget = QComboBox()
                    widget.addItem('Largest matching', 'largest')
                    widget.addItem('All matching', 'all')
                    widget.setCurrentIndex(0 if value == 'largest' else 1)
                    form.addRow(label, widget)
                    reader = widget.currentData
                elif name.endswith('_roi'):
                    widget = CropDefaults(value)
                    form.addRow(label, widget)
                    reader = widget.value
                elif isinstance(value, int) and name != 'rotation':
                    widget = QSpinBox()
                    widget.setRange(1 if name == 'n_clusters' else 0, 255 if name in ('low', 'high') else 1000000)
                    widget.setValue(value)
                    form.addRow(label, widget)
                    reader = widget.value
                elif isinstance(value, float) or name == 'rotation':
                    widget = CompactDoubleSpinBox()
                    widget.setDecimals(6)
                    widget.setRange(-36000 if name == 'rotation' else 0, 36000 if name == 'rotation' else 1)
                    widget.setSingleStep(15 if name == 'rotation' else .01)
                    widget.setValue(value)
                    form.addRow(label, widget)
                    reader = widget.value
                else:
                    widget = QLineEdit('' if value is None else str(list(value)) if isinstance(value, (tuple, list)) else str(value))
                    widget.setProperty('presetField', True)
                    reader = widget.text
                    form.addRow(label, widget)
                if key in ('crop.top_ratio', 'crop.bottom_ratio'):
                    widget.setStatusTip('Vertical start/end positions in the plate, before margins. Use 0 and 1 for the full plate; margins are then applied inside this range.')
                elif key in ('margin_top', 'margin_bottom'):
                    widget.setStatusTip('Fraction removed from this edge after the initial crop. With initial crop start 0 and end 1, this is a fraction of the full plate height.')
                widget.setObjectName(key)
                self._readers[key] = reader
            if prefix in ('green', 'crop'):
                lower, upper = ('hsv_lower', 'hsv_upper') if prefix == 'green' else ('blue_hsv_lower', 'blue_hsv_upper')
                color = ColorRangeControl('Leaves color' if prefix == 'green' else 'Background color', getattr(section, lower), getattr(section, upper))
                color.preview.hide()
                color.setStatusTip('Click the color range to edit both HSV limits.')
                # Presets have no image canvas to sample from.
                color.swatch.clicked.connect(lambda checked=False, c=color: self._disable_sampling(c))
                form.addRow(color)
                self._readers[f'{prefix}.{lower}'] = lambda c=color: c.values()[0]
                self._readers[f'{prefix}.{upper}'] = lambda c=color: c.values()[1]
            if prefix == '':
                note = QLabel('Margins and crop ratios apply when Crop within plate is empty. Crop coordinates are normalized; 0.5 is the center.')
                note.setWordWrap(True)
                form.addRow(note)
            scroll = QScrollArea()
            scroll.setObjectName('presetScroll')
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll, title)

    @staticmethod
    def _disable_sampling(control):
        if control._picker is not None:
            for button in control._picker.findChildren(QPushButton):
                if 'Sample' in button.text():
                    button.hide()

    def value(self):
        raw = self._config.to_dict()
        for key, reader in self._readers.items():
            if '.' in key:
                section, name = key.split('.')
                raw[section][name] = reader()
            else:
                raw[key] = reader()
        config = Config.from_dict(raw)
        config._base_path = self._config._base_path
        return config


class PresetsDialog(QDialog):
    def __init__(self, store, current, parent=None, *, active_name=None):
        super().__init__(parent)
        self.setWindowTitle('Presets')
        self.setStyleSheet(presets_stylesheet(self.palette()))
        self.resize(960, 650)
        self.store = store
        self.active_name = active_name
        self.selected_config = None
        self.selected_name = None
        self._loaded_name = None
        self._loaded_values = current.to_dict()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'Active preset: {active_name}' if active_name else 'Active configuration: custom settings'))
        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setObjectName('presetList')
        self.list.setMouseTracking(True)
        self.list.setMaximumWidth(200)
        body.addWidget(self.list)
        self.editor = ConfigEditor(current)
        body.addWidget(self.editor, 1)
        layout.addLayout(body)
        row = QHBoxLayout()
        for label, callback in (('New…', self.new), ('Save', self.save), ('Delete', self.delete),
                                ('Import YAML…', self.import_yaml), ('Export YAML…', self.export_yaml)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        row.addStretch()
        use = QPushButton('Use preset')
        use.clicked.connect(self.use)
        row.addWidget(use)
        close = QPushButton('Close')
        close.clicked.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)
        self.list.currentTextChanged.connect(self._select)
        self._refresh(active_name)
        if active_name in self.store.names():
            self._select(active_name)

    def _refresh(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(self.store.names())
        for index in range(self.list.count()):
            item = self.list.item(index)
            if item.text() == self.active_name:
                item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        matches = self.list.findItems(select or '', Qt.MatchFlag.MatchExactly)
        if matches:
            self.list.setCurrentItem(matches[0])
        else:
            self.list.setCurrentRow(-1)
        self.list.blockSignals(False)

    def _discard_edits(self):
        try:
            dirty = self.editor.value().to_dict() != self._loaded_values
        except Exception:
            dirty = True
        return not dirty or QMessageBox.question(self, 'Unsaved preset edits', 'Discard the unsaved preset edits?', QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def _select(self, name):
        if not name or name == self._loaded_name:
            return
        if not self._discard_edits():
            self._refresh(self._loaded_name)
            return
        try:
            config = self.store.load(name)
            self.editor.set_config(config)
            self._loaded_name = name
            self._loaded_values = config.to_dict()
        except Exception as error:
            self._error(error)
            self._refresh(self._loaded_name)

    def _error(self, error):
        QMessageBox.warning(self, 'Preset could not be updated', str(error))

    def new(self):
        name, ok = QInputDialog.getText(self, 'New preset', 'Name')
        if not ok:
            return
        try:
            if name in self.store.names():
                raise ValueError('A preset with this name already exists')
            config = self.editor.value()
            self.store.save(name, config)
            self._loaded_name = name
            self._loaded_values = config.to_dict()
            self._refresh(name)
        except Exception as error:
            self._error(error)

    def save(self):
        if self._loaded_name is None:
            self.new()
            return
        try:
            config = self.editor.value()
            self.store.save(self._loaded_name, config)
            self._loaded_values = config.to_dict()
        except Exception as error:
            self._error(error)

    def delete(self):
        if self._loaded_name is None:
            return
        if QMessageBox.question(self, 'Delete preset', f'Delete “{self._loaded_name}”?') != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete(self._loaded_name)
            self._loaded_name = None
            self._refresh()
        except Exception as error:
            self._error(error)

    def import_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import preset', '', 'YAML (*.yaml *.yml)')
        if not path or not self._discard_edits():
            return
        try:
            config = Config.from_yaml(path)
            name, ok = QInputDialog.getText(self, 'Import preset', 'Name', text=Path(path).stem)
            if not ok:
                return
            if name in self.store.names():
                raise ValueError('A preset with this name already exists')
            self.store.save(name, config)
            self.editor.set_config(config)
            self._loaded_name = name
            self._loaded_values = config.to_dict()
            self._refresh(name)
        except Exception as error:
            self._error(error)

    def export_yaml(self):
        try:
            config = self.editor.value()
            path, _ = QFileDialog.getSaveFileName(self, 'Export preset', (self._loaded_name or 'preset')+'.yaml', 'YAML (*.yaml)')
            if path:
                config.data.resolve_paths(config.base_path)
                config.to_yaml(path)
        except Exception as error:
            self._error(error)

    def use(self):
        try:
            config = self.editor.value()
            if self._loaded_name is None:
                self.new()
                if self._loaded_name is None:
                    return
            self.store.save(self._loaded_name, config)
            self.selected_config = self.store.load(self._loaded_name)
            self.selected_name = self._loaded_name
            self.accept()
        except Exception as error:
            self._error(error)

    def reject(self):
        if self._discard_edits():
            super().reject()
