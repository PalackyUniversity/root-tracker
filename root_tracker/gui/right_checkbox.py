"""Full-width settings checkbox with a left label and a right indicator."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QCheckBox, QStyle, QStyleOptionButton


class RightAlignedCheckBox(QCheckBox):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        text = option.text
        option.text = ''
        painter = QPainter(self)
        self.style().drawControl(QStyle.ControlElement.CE_CheckBox, option, painter, self)
        indicator = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
        label = self.rect()
        label.setRight(indicator.left() - 6)
        self.style().drawItemText(
            painter, label, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            option.palette, self.isEnabled(), text, QPalette.ColorRole.WindowText)

    def hitButton(self, pos):
        return self.rect().contains(pos)
