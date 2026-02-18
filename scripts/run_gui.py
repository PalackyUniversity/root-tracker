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


def _checkmark_url() -> str:
    """
    Write a checkmark SVG to a temp file and return a CSS url() string.
    The file is deleted automatically when the process exits.
    CSS image: properties in Qt don't support data: URIs, so a real file is needed.
    """
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12">'
        '<polyline points="2,6 5,9 10,3" stroke="#cccccc" stroke-width="2.5"'
        ' fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )
    f = tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w", encoding="utf-8")
    f.write(svg)
    f.close()
    atexit.register(os.unlink, f.name)
    return f'url("{f.name.replace(chr(92), "/")}")'


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

    checkmark = _checkmark_url()
    app.setStyleSheet("""
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
        QMenu::item:selected {
            background-color: #4a4a4a;
            color: #fff;
        }
        QMenu::item:disabled { color: #555; }
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
    """)

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
