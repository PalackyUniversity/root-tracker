"""Compact hue-range summary with a single dialog for editing both HSV limits."""
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel
from .color_picker import HsvPicker
from .right_checkbox import RightAlignedCheckBox


class ColorRangeControl(QGroupBox):
    range_changed = Signal(object, object)
    preview_changed = Signal(bool)
    pick_requested = Signal(bool)

    def __init__(self, title, lower, upper, parent=None, *, selector_label=None):
        super().__init__(title, parent)
        self._picker = None
        self.sample_index = None
        self._values = (tuple(lower), tuple(upper))
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.swatch = QPushButton()
        self.swatch.setFixedSize(64, 28)
        self.swatch.setAccessibleName(f'{title}: adjust HSV range')
        self.swatch.setStatusTip('Selected hue range, shown at full saturation and brightness. Click to edit all HSV limits.')
        self.swatch.clicked.connect(self._open_picker)
        self.swatches = [self.swatch]
        if selector_label:
            label = QLabel(selector_label)
            label.setBuddy(self.swatch)
            label.setStatusTip(self.swatch.statusTip())
            row.addWidget(label)
            row.addStretch()
        row.addWidget(self.swatch)
        layout.addLayout(row)
        # Sampling remains available from the range dialog; this hidden toggle
        # holds its state while the user interacts with the image canvas.
        self.pick_button = QPushButton('Sample from image', self)
        self.pick_button.setCheckable(True)
        self.pick_button.toggled.connect(self.pick_requested)
        self.pick_button.hide()
        self.preview = RightAlignedCheckBox('Show segmentation')
        self.preview.setStatusTip('Highlight pixels selected by this range, with the same cleanup used during processing.')
        self.preview.toggled.connect(self.preview_changed)
        if selector_label:
            layout.addWidget(self.preview)
        else:
            row.addWidget(self.preview)
            row.addStretch()
        self._update_swatch()

    def values(self):
        return self._values

    def set_range(self, lower, upper):
        self._values = (tuple(map(int, lower)), tuple(map(int, upper)))
        self._update_swatch()

    def _update_swatch(self):
        lower, upper = self.values()
        span = (upper[0]-lower[0]) % 180
        stops = []
        for index in range(13):
            fraction = index / 12
            hue = round((lower[0]+span*fraction)*2) % 360
            stops.append(f'stop:{fraction:.3f} {QColor.fromHsv(hue, 255, 255).name()}')
        self.swatch.setStyleSheet('background: qlineargradient(x1:0,y1:0,x2:1,y2:0, '
                                 + ', '.join(stops) + '); border: 1px solid #888; border-radius: 4px;')

    def finish_picker(self):
        if self._picker is not None:
            self._picker.accept()

    def _open_picker(self, checked=False):
        if self._picker is not None:
            self._picker.raise_()
            self._picker.activateWindow()
            return
        self.pick_button.setChecked(False)
        self._original_range = self.values()
        self._original_preview = self.preview.isChecked()
        picker = HsvPicker(self.title(), *self._original_range, self)
        self._picker = picker
        picker.range_changed.connect(self._picker_update)
        picker.sample_requested.connect(self._picker_sample)
        picker.finished.connect(self._picker_finished)
        self.preview.setChecked(True)
        picker.show()

    def _picker_update(self, lower, upper):
        self.set_range(lower, upper)
        self.range_changed.emit(lower, upper)

    def _picker_finished(self, result):
        picker, self._picker = self._picker, None
        self.sample_index = None
        self.pick_button.setChecked(False)
        if not result:
            self._picker_update(*self._original_range)
            self.preview.setChecked(self._original_preview)
        self.preview_changed.emit(self.preview.isChecked())
        picker.deleteLater()

    def _picker_sample(self, index):
        self.sample_index = index
        self._picker.hide()
        self.pick_button.setChecked(True)

    def end_sampling(self):
        self.sample_index = None
        if self._picker is not None:
            self._picker.show()
            self._picker.raise_()

    def apply_sample(self, hsv):
        index = self.sample_index
        if self._picker is not None and index is not None:
            self._picker.editors[index].set_values(hsv)
            self._picker._changed(index)
        self.pick_button.setChecked(False)
        self.end_sampling()
