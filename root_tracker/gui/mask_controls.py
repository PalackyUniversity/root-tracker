"""Compact, mutually exclusive tools for editing a group's exclusion mask."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup, QGridLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QVBoxLayout,
)

from .masking_tools import MaskTool


class MaskControls(QGroupBox):
    tool_changed = Signal(str, int)
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__('Exclusion mask', parent)
        self.setObjectName('maskControls')
        self._restore_action_before_drag = None
        # Palette roles follow the application's light/dark theme. Scope the
        # checked style to these controls, leaving other application buttons alone.
        self.setStyleSheet('''
            QGroupBox#maskControls QPushButton:checked {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                border: 1px solid palette(highlight);
            }
            QGroupBox#maskControls QPushButton:focus {
                border: 2px solid palette(text);
            }
        ''')
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        self._tools = QButtonGroup(self)
        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        for index, (name, tip) in enumerate((
            ('Pan', 'Drag to move the image. Scroll to zoom.'),
            ('Brush', 'Paint an exclusion or restore an area with a round brush.'),
            ('Rectangle', 'Drag a rectangle to exclude or restore an area.'),
        )):
            button = self._toggle(name, tip)
            self._tools.addButton(button, index)
            tool_row.addWidget(button, 1)
        self._tools.button(0).setChecked(True)
        layout.addLayout(tool_row)

        self._operations = QButtonGroup(self)
        self._operation_row = QFrame()
        operation_layout = QHBoxLayout(self._operation_row)
        operation_layout.setContentsMargins(0, 0, 0, 0)
        operation_layout.setSpacing(4)
        for index, (name, tip) in enumerate((
            ('Exclude', 'Add to the mask: these areas will be ignored by tracking.'),
            ('Restore', 'Remove from the mask: allow tracking in these areas again. Right-drag the image to restore temporarily.'),
        )):
            button = self._toggle(name, tip)
            self._operations.addButton(button, index)
            operation_layout.addWidget(button, 1)
        self._operations.button(0).setChecked(True)

        form = QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)
        action_label = QLabel('Action')
        action_label.setStatusTip('Exclude adds areas to the mask; Restore removes them.')
        form.addWidget(action_label, 0, 0)
        form.addWidget(self._operation_row, 0, 1)
        self._diameter = QSpinBox()
        self._diameter.setRange(1, 999)
        self._diameter.setValue(100)
        self._diameter.setSuffix(' px')
        self._diameter.setMinimumHeight(30)
        self._diameter.setAccessibleName('Brush diameter')
        self._diameter.setStatusTip('Brush diameter in image pixels, shared by Exclude and Restore. Alt + scroll adjusts by 5 px.')
        self._diameter_label = QLabel('Diameter')
        self._diameter_label.setBuddy(self._diameter)
        self._diameter_label.setStatusTip(self._diameter.statusTip())
        form.addWidget(self._diameter_label, 1, 0)
        form.addWidget(self._diameter, 1, 1)
        layout.addLayout(form)

        footer = QHBoxLayout()
        footer.addStretch()
        clear = QPushButton('Clear mask')
        clear.setMinimumHeight(32)
        clear.setToolTip('Remove all exclusions for this group.')
        clear.setStatusTip(clear.toolTip())
        clear.clicked.connect(self.clear_requested.emit)
        footer.addWidget(clear)
        layout.addLayout(footer)

        self._tools.buttonClicked.connect(self._update_tool)
        self._operations.buttonClicked.connect(self._update_tool)
        self._diameter.valueChanged.connect(self._update_tool)
        self._update_tool()

    @staticmethod
    def _toggle(text, tip):
        button = QPushButton(text)
        button.setCheckable(True)
        button.setMinimumHeight(32)
        button.setToolTip(tip)
        button.setStatusTip(tip)
        return button

    def set_temporary_restore(self, active):
        """Right-drag overrides the action only until the gesture is released."""
        if active:
            if self._restore_action_before_drag is None:
                self._restore_action_before_drag = self._operations.checkedId()
            self._operations.button(1).setChecked(True)
        elif self._restore_action_before_drag is not None:
            self._operations.button(self._restore_action_before_drag).setChecked(True)
            self._restore_action_before_drag = None
        self._update_tool()

    def adjust_diameter(self, steps):
        self._diameter.setValue(self._diameter.value() + steps * 5)

    def reset_to_pan(self):
        """Keep the selected tool and the viewer in sync on group changes."""
        self._tools.button(0).setChecked(True)
        self._update_tool()

    def _update_tool(self, *_):
        shape = self._tools.checkedId()
        restore = self._operations.checkedId() == 1
        self._operation_row.setEnabled(shape != 0)
        self._diameter.setEnabled(shape == 1)
        self._diameter_label.setEnabled(shape == 1)
        if shape == 0:
            tool, size = MaskTool.MOVE, 0
        else:
            if shape == 1:
                tool = MaskTool.BRUSH_ERASER if restore else MaskTool.BRUSH
            else:
                tool = MaskTool.RECT_ERASER if restore else MaskTool.RECTANGLE
            size = self._diameter.value() if shape == 1 else 0
        self.tool_changed.emit(tool.value, size)
