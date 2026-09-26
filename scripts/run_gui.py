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

    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
