"""Non-modal HSV editor preserving hue even for black and gray limits."""
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QLinearGradient, QPen, QIntValidator, QIcon
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QSlider, QDialogButtonBox,
                              QWidget, QHBoxLayout, QGroupBox, QPushButton, QLineEdit,
                              QSizePolicy)


class ValueEntry(QLineEdit):
    """Consume Return so typing a channel value does not accept the dialog."""
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.editingFinished.emit()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.setText(str(self.parent().value()))
            self.parent().setFocus()
            event.accept()
        else:
            super().keyPressEvent(event)


class HsvValueControl(QSlider):
    """One color field: full-width drag track and an inline editable value."""
    def __init__(self, channel, name, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.channel, self.name = channel, name
        self.hsv = (0, 255, 255)
        self.setAccessibleName(name)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStatusTip(f'Drag the {name.lower()} color bar or type a value on the right. Arrow keys adjust by one.')
        self.entry = ValueEntry(self)
        self.entry.setAccessibleName(f'{name} value')
        self.entry.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.entry.setFrame(False)
        self.entry.setStyleSheet('QLineEdit { background: transparent; border: none; padding: 0px; }')
        self.entry.setValidator(QIntValidator(0, 179 if channel == 0 else 255, self.entry))
        self.entry.setStatusTip(self.statusTip())
        self.entry.setText(str(self.value()))
        self.entry.textEdited.connect(self._typed)
        self.entry.editingFinished.connect(self._commit_text)
        self.valueChanged.connect(self._sync_text)

    def _sync_text(self, value):
        # Preserve partially typed numbers while range constraints update the value.
        if not (self.entry.hasFocus() and self.entry.isModified()):
            self.entry.setText(str(value))
        self.update()

    def _typed(self, text):
        if self.entry.hasAcceptableInput():
            self.setValue(int(text))

    def _commit_text(self):
        if self.entry.text().isdigit():
            self.setValue(int(self.entry.text()))
        self.entry.setModified(False)
        self._sync_text(self.value())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.entry.setGeometry(self.width()-58, 4, 46, 22)

    def track_rect(self):
        return QRectF(12, 32, max(1, self.width()-24), 8)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        border = self.palette().highlight().color() if self.hasFocus() else self.palette().mid().color()
        painter.setPen(QPen(border, 1))
        painter.setBrush(self.palette().base())
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 5, 5)
        painter.setPen(self.palette().text().color())
        painter.drawText(QRectF(12, 4, self.width()-76, 22), Qt.AlignmentFlag.AlignVCenter, self.name)
        rect = self.track_rect()
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        for index in range(13):
            fraction = index/12
            h, saturation, value = self.hsv
            if self.channel == 0:
                h, saturation, value = round(179*fraction), 255, 255
            elif self.channel == 1:
                saturation, value = round(255*fraction), 255
            else:
                value = round(255*fraction)
            gradient.setColorAt(fraction, QColor.fromHsv(h*2, saturation, value))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 3, 3)
        x = rect.left()+(self.value()-self.minimum())/max(1, self.maximum()-self.minimum())*rect.width()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for color, width in (('black', 3), ('white', 1.5)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawRoundedRect(QRectF(x-2, rect.top()-2, 4, rect.height()+4), 2, 2)

    def _drag(self, event):
        rect = self.track_rect()
        fraction = max(0., min(1., (event.position().x()-rect.left())/rect.width()))
        self.setValue(round(self.minimum()+fraction*(self.maximum()-self.minimum())))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.setSliderDown(True)
            self._drag(event)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self._drag(event)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSliderDown():
            self._drag(event)
            self.setSliderDown(False)
            event.accept()


class SaturationValuePlane(QWidget):
    changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hsv = (0, 255, 255)
        self.setMinimumSize(260, 180)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setStatusTip('Drag to choose saturation horizontally and brightness vertically.')

    def color_rect(self):
        return QRectF(self.rect()).adjusted(8, 8, -8, -8)

    def selection_position(self):
        rect = self.color_rect()
        return QPointF(rect.left()+self.hsv[1]/255*rect.width(),
                       rect.top()+(1-self.hsv[2]/255)*rect.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        hue, saturation, value = self.hsv
        rect = self.color_rect()
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0, QColor('white'))
        gradient.setColorAt(1, QColor.fromHsv(hue * 2, 255, 255))
        painter.fillRect(rect, gradient)
        shade = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        shade.setColorAt(0, QColor(0, 0, 0, 0))
        shade.setColorAt(1, QColor('black'))
        painter.fillRect(rect, shade)
        position = self.selection_position()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for color, width in (('black', 3), ('white', 1)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawEllipse(position, 5, 5)

    def _pick(self, event):
        rect = self.color_rect()
        saturation = round(255 * max(0, min(1, (event.position().x()-rect.left()) / rect.width())))
        value = round(255 * (1-max(0, min(1, (event.position().y()-rect.top()) / rect.height()))))
        self.changed.emit(saturation, value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick(event)


class HsvEditor(QGroupBox):
    hsv_changed = Signal(object)
    sample_requested = Signal()

    def __init__(self, title, hsv, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self.plane = SaturationValuePlane()
        layout.addWidget(self.plane)
        self.controls = []
        for index, (name, maximum) in enumerate((('Hue', 179), ('Saturation', 255), ('Value', 255))):
            control = HsvValueControl(index, name)
            control.setRange(0, maximum)
            control.setValue(hsv[index])
            control.valueChanged.connect(self._changed)
            self.controls.append(control)
            layout.addWidget(control)
        self.plane.hsv = tuple(hsv)
        self.plane.changed.connect(self._plane_changed)
        self.sample_button = QPushButton('Sample from image')
        self.sample_button.setStatusTip(f'Click one pixel area to set the {title.lower()}. Press Escape to return without sampling.')
        self.sample_button.clicked.connect(self.sample_requested)
        layout.addWidget(self.sample_button)
        self._update_gradients()

    def _update_gradients(self):
        for slider in self.controls:
            slider.hsv = self.values()
            slider.update()

    def values(self):
        return tuple(spin.value() for spin in self.controls)

    def set_values(self, hsv):
        for control, value in zip(self.controls, hsv):
            control.blockSignals(True)
            control.setValue(value)
            control._sync_text(value)
            control.blockSignals(False)
        self.plane.hsv = tuple(hsv)
        self.plane.update()
        self._update_gradients()

    def _plane_changed(self, saturation, value):
        self.set_values((self.controls[0].value(), saturation, value))
        self._changed()

    def _changed(self):
        self.plane.hsv = self.values()
        self.plane.update()
        self._update_gradients()
        self.hsv_changed.emit(self.values())


class HsvPicker(QDialog):
    range_changed = Signal(object, object)
    sample_requested = Signal(int)

    def __init__(self, title, lower, upper, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'{title} — Adjust range')
        self.setWindowModality(Qt.WindowModality.NonModal)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.editors = []
        for index, (name, hsv) in enumerate((('Lower limit', lower), ('Upper limit', upper))):
            editor = HsvEditor(name, hsv)
            editor.hsv_changed.connect(self._editor_changed)
            editor.sample_requested.connect(self._request_sample)
            self.editors.append(editor)
            row.addWidget(editor)
        layout.addLayout(row)
        footer = QHBoxLayout()
        footer.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        for button in buttons.buttons():
            button.setIcon(QIcon())
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)

    def values(self):
        return tuple(editor.values() for editor in self.editors)

    def _request_sample(self):
        self.sample_requested.emit(self.editors.index(self.sender()))

    def _editor_changed(self, hsv):
        self._changed(self.editors.index(self.sender()))

    def _changed(self, index):
        limits = [list(values) for values in self.values()]
        for channel in (1, 2):
            if limits[0][channel] > limits[1][channel]:
                limits[index][channel] = limits[1-index][channel]
        self.editors[index].set_values(limits[index])
        self.range_changed.emit(*self.values())
