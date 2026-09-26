"""Plain, compact application information dialog."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('About Root Tracker')
        self.setMinimumWidth(380)
        font = self.font()
        font.setWeight(QFont.Weight.Normal)
        self.setFont(font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(14)

        title = QLabel('Root Tracker')
        title_font = QFont(font)
        title_font.setPointSize(18)
        title.setFont(title_font)
        layout.addWidget(title)
        for text in (
            'Version 2.5',
            'A tool for tracking and analyzing plant root growth.',
        ):
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addSpacing(6)
        authors = QVBoxLayout()
        authors.setSpacing(6)
        heading = QLabel('AUTHORS')
        heading_font = QFont(font)
        heading_font.setPointSizeF(max(8, font.pointSizeF() - 1))
        heading_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(heading_font)
        palette = heading.palette()
        foreground = palette.color(QPalette.ColorRole.WindowText)
        background = palette.color(QPalette.ColorRole.Window)
        muted = QColor(*(round(f * 0.75 + b * 0.25) for f, b in zip(
            foreground.getRgb()[:3], background.getRgb()[:3])))
        palette.setColor(QPalette.ColorRole.WindowText, muted)
        heading.setPalette(palette)
        authors.addWidget(heading)
        names = QLabel('Tadeáš Fryčák, Petr Ivan')
        names.setWordWrap(True)
        authors.addWidget(names)
        layout.addLayout(authors)

        layout.addSpacing(6)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText('OK')
        ok.setIcon(QIcon())
        ok.setDefault(True)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
