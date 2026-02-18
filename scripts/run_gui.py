#!/usr/bin/env python3
"""
Run the Root Tracker GUI.

Usage:
    python run_gui.py [--config configs/in_vitro.yaml]
    
Remember to activate your virtual environment before running!
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from root_tracker import Config
from root_tracker.gui import MainWindow


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
    
    # Set application style
    app.setStyle("Fusion")

    # Global button style matching the "Open..." button in the image tree header
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
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
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
    """)
    
    # Create and show main window
    window = MainWindow(config)
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
