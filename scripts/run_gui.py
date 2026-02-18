#!/usr/bin/env python3
"""
Run the Root Tracker GUI.

Usage:
    python run_gui.py [--config configs/in_vitro.yaml]

Remember to activate your virtual environment before running!
"""

import sys
import os
import atexit
import tempfile
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from root_tracker import Config
from root_tracker.gui import MainWindow

# ---------------------------------------------------------------------------
# SVG icons – written to temp files at startup so Qt CSS can reference them
# ---------------------------------------------------------------------------

_CHECKMARK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12">'
    '<polyline points="2,6 5,9 10,3" stroke="#cccccc" stroke-width="2.5"'
    ' fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
_RADIO_DOT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12">'
    '<circle cx="6" cy="6" r="3.5" fill="#cccccc"/>'
    '</svg>'
)
_ARROW_UP_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 5">'
    '<polyline points="0.5,4.5 4,0.5 7.5,4.5" stroke="#aaaaaa"'
    ' stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)
_ARROW_DOWN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 5">'
    '<polyline points="0.5,0.5 4,4.5 7.5,0.5" stroke="#aaaaaa"'
    ' stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)


def _svg_url(svg: str) -> str:
    """Write SVG content to a temp file and return a CSS url() string."""
    f = tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w", encoding="utf-8")
    f.write(svg)
    f.close()
    atexit.register(os.unlink, f.name)
    return 'url("' + f.name.replace("\\", "/") + '")'


def _build_stylesheet() -> str:
    checkmark  = _svg_url(_CHECKMARK_SVG)
    radio_dot  = _svg_url(_RADIO_DOT_SVG)
    arrow_up   = _svg_url(_ARROW_UP_SVG)
    arrow_down = _svg_url(_ARROW_DOWN_SVG)

    return """
        /* ── Buttons ── */
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 3px;
            padding: 4px 8px;
            color: #ccc;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
            border: 1px solid #5a5a5a;
            color: #fff;
        }
        QPushButton:pressed  { background-color: #2a2a2a; }
        QPushButton:disabled {
            background-color: #2e2e2e;
            border: 1px solid #3a3a3a;
            color: #555;
        }

        /* ── Labels ── */
        QLabel { color: #ccc; }

        /* ── Number fields and text inputs ── */
        QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 3px;
            padding: 3px 6px;
            color: #ccc;
            selection-background-color: #4a90d9;
            selection-color: #fff;
        }
        QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
            border-color: #5a5a5a;
        }
        QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
            background-color: #2e2e2e;
            color: #555;
            border-color: #3a3a3a;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            background-color: #3a3a3a;
            border: none;
            border-left: 1px solid #4a4a4a;
            border-bottom: 1px solid #4a4a4a;
            width: 18px;
            border-top-right-radius: 3px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #3a3a3a;
            border: none;
            border-left: 1px solid #4a4a4a;
            width: 18px;
            border-bottom-right-radius: 3px;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4a4a4a;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: """ + arrow_up + """;
            width: 7px; height: 5px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: """ + arrow_down + """;
            width: 7px; height: 5px;
        }

        /* ── Combo box ── */
        QComboBox {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 3px;
            padding: 3px 6px;
            color: #ccc;
        }
        QComboBox:hover   { border-color: #5a5a5a; }
        QComboBox:disabled { background-color: #2e2e2e; color: #555; border-color: #3a3a3a; }
        QComboBox::drop-down {
            border: none;
            border-left: 1px solid #4a4a4a;
            width: 20px;
        }
        QComboBox::down-arrow {
            image: """ + arrow_down + """;
            width: 7px; height: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            color: #ccc;
            outline: none;
        }
        QComboBox QAbstractItemView::item {
            padding: 3px 8px;
            min-height: 22px;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: #4a4a4a;
            color: #fff;
        }
        QComboBox QAbstractItemView::item:selected {
            background-color: #4a90d9;
            color: #fff;
        }

        /* ── Checkboxes ── */
        QCheckBox          { color: #ccc; spacing: 6px; }
        QCheckBox:disabled { color: #555; }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #5a5a5a;
            border-radius: 3px;
            background-color: #2b2b2b;
        }
        QCheckBox::indicator:hover   { border-color: #7a7a7a; }
        QCheckBox::indicator:checked {
            image: """ + checkmark + """;
        }
        QCheckBox::indicator:disabled {
            background-color: #2e2e2e;
            border-color: #3a3a3a;
        }

        /* ── Radio buttons ── */
        QRadioButton          { color: #ccc; spacing: 6px; }
        QRadioButton:disabled { color: #555; }
        QRadioButton::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #5a5a5a;
            border-radius: 7px;
            background-color: #2b2b2b;
        }
        QRadioButton::indicator:hover   { border-color: #7a7a7a; }
        QRadioButton::indicator:checked {
            image: """ + radio_dot + """;
        }

        /* ── Group boxes ── */
        QGroupBox {
            color: #888;
            border: 1px solid #404040;
            border-radius: 4px;
            margin-top: 10px;
            margin-bottom: 16px;
            padding-top: 6px;
            font-size: 11px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 4px 4px;
        }

        /* ── Scroll bars ── */
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 10px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #4a4a4a;
            border-radius: 5px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background-color: #5a5a5a; }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical     { height: 0; border: none; }

        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 10px;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background-color: #4a4a4a;
            border-radius: 5px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover { background-color: #5a5a5a; }
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal     { width: 0; border: none; }

        /* ── Menus ── */
        QMenu {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 4px 0px;
            color: #ccc;
        }
        QMenu::item {
            padding: 5px 14px 5px 14px;
            border-radius: 3px;
            margin: 1px 4px;
        }
        QMenu::item:selected  { background-color: #4a4a4a; color: #fff; }
        QMenu::item:disabled  { color: #555; }
        QMenu::separator {
            height: 1px;
            background-color: #4a4a4a;
            margin: 4px 8px;
        }
        QMenu::indicator {
            width: 13px;
            height: 13px;
            margin-left: 4px;
            border: 1px solid #5a5a5a;
            border-radius: 2px;
            background-color: #3a3a3a;
        }
        QMenu::indicator:checked {
            image: """ + checkmark + """;
        }
    """


def main() -> None:
    """Main entry point for GUI."""
    parser = argparse.ArgumentParser(description="Root Tracker GUI")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = Config.from_yaml(args.config)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Root Tracker")
    app.setOrganizationName("Palacky University")

    app.setStyle("Fusion")
    app.setStyleSheet(_build_stylesheet())

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
