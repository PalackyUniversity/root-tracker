"""Dropdowns with a transparent backing behind the rounded list border."""
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QComboBox, QFrame


class RoundedComboBox(QComboBox):
    def showPopup(self):
        popup = self.view().window()
        popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if isinstance(popup, QFrame):
            popup.setFrameShape(QFrame.Shape.NoFrame)
        super().showPopup()
        gap = 4
        below = self.mapToGlobal(QPoint(0, self.height())).y() + gap
        available = self.screen().availableGeometry()
        if below + popup.height() <= available.bottom() + 1:
            y = below
        else:
            # Preserve access near the screen edge by opening above the field.
            above = self.mapToGlobal(QPoint(0, 0)).y() - gap - popup.height()
            y = max(available.top(), above)
        popup.move(popup.x(), y)
