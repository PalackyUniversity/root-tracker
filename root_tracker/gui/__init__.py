"""
PySide6 GUI for Root Tracker.

Provides a graphical interface for the root tracking pipeline with:
- 4-step workflow (Load, Preprocess, Track, Export)
- Zoomable image preview
- Group/image navigation
- Configuration editing
"""

from .main_window import MainWindow

__all__ = ["MainWindow"]
