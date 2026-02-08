"""
Main application window for Root Tracker GUI.

Integrates all components into the main window layout.
"""

import os
import cv2
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QStatusBar, QMessageBox,
    QProgressDialog, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence

from ..config import Config
from ..models import ImageData, ImageSeries
from ..pipeline import RootTrackingPipeline
from ..io import ImageLoader

from .workflow_bar import WorkflowBar, WorkflowStep
from .image_tree import ImageTree
from .image_viewer import ImageViewer
from .settings_panel import SettingsPanel
from .dialogs import LoadDialog, ExportDialog


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Layout:
    - Top: Workflow step bar
    - Left: Image tree + Settings panel
    - Right: Zoomable image preview
    - Bottom: Status bar
    """
    
    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        
        # Initialize with default config if none provided
        if config is None:
            config = Config()
        self._config = config
        
        # Pipeline and data state
        self._pipeline: RootTrackingPipeline | None = None
        self._series_dict: dict[str, ImageSeries] = {}
        self._current_image: ImageData | None = None
        self._current_series: ImageSeries | None = None
        
        self.setWindowTitle("Root Tracker")
        self.setMinimumSize(1200, 800)
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        
        # Start on Load step with settings panel hidden
        self._settings_panel.hide()
    
    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with workflow steps aligned right
        top_bar = QWidget()
        top_bar.setFixedHeight(46)  # Minimal height for workflow bar
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 2, 5, 2)
        top_layout.addStretch()  # Push workflow bar to right
        self._workflow_bar = WorkflowBar()
        top_layout.addWidget(self._workflow_bar)
        main_layout.addWidget(top_bar)
        
        # Main content area with splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Image tree only
        self._image_tree = ImageTree()
        self._image_tree.setMinimumWidth(180)
        self._image_tree.setMaximumWidth(250)
        self._splitter.addWidget(self._image_tree)
        
        # Center: Image viewer
        self._image_viewer = ImageViewer()
        self._splitter.addWidget(self._image_viewer)
        
        # Right panel: Settings
        self._settings_panel = SettingsPanel(self._config)
        self._settings_panel.setMinimumWidth(280)
        self._settings_panel.setMaximumWidth(350)
        self._splitter.addWidget(self._settings_panel)
        
        # Set splitter proportions
        self._splitter.setSizes([200, 800, 300])
        
        main_layout.addWidget(self._splitter)
        
        # Status bar containing all controls
        from PySide6.QtWidgets import QPushButton, QLabel, QProgressBar
        
        self._status_bar = QStatusBar()
        self._status_bar.setSizeGripEnabled(False)  # Remove resize grip
        self.setStatusBar(self._status_bar)
        
        # Left: Zoom controls (as regular widgets - stay on left)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setFixedSize(40, 26)
        self._fit_btn.clicked.connect(self._image_viewer.fit_in_view)
        self._status_bar.addWidget(self._fit_btn, 0)
        
        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setFixedSize(26, 26)
        self._zoom_out_btn.clicked.connect(self._image_viewer.zoom_out)
        self._status_bar.addWidget(self._zoom_out_btn, 0)
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(45)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_bar.addWidget(self._zoom_label, 0)
        
        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFixedSize(26, 26)
        self._zoom_in_btn.clicked.connect(self._image_viewer.zoom_in)
        self._status_bar.addWidget(self._zoom_in_btn, 0)
        
        # Spacer to push right-side widgets
        spacer = QLabel()
        self._status_bar.addWidget(spacer, 1)  # stretch=1 to fill space
        
        # Right side: Status label, Progress bar, Preview, Back, Next
        self._status_label = QLabel("Ready")
        self._status_label.setMinimumWidth(150)
        self._status_bar.addPermanentWidget(self._status_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedSize(120, 18)
        self._progress_bar.setMaximum(100)
        self._progress_bar.hide()
        self._status_bar.addPermanentWidget(self._progress_bar)
        
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setFixedHeight(26)
        self._preview_btn.setEnabled(False)
        self._preview_btn.setToolTip("Auto Preview is enabled - processing happens automatically")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        self._status_bar.addPermanentWidget(self._preview_btn)
        
        self._back_step_btn = QPushButton("← Back")
        self._back_step_btn.setFixedHeight(26)
        self._back_step_btn.clicked.connect(self._on_step_back)
        self._status_bar.addPermanentWidget(self._back_step_btn)
        
        self._next_step_btn = QPushButton("Next →")
        self._next_step_btn.setFixedHeight(26)
        self._next_step_btn.clicked.connect(self._on_step_next)
        self._status_bar.addPermanentWidget(self._next_step_btn)
    
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_action = QAction("&Load Images...", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self._on_load_images)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        fit_action = QAction("&Fit Image", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self._image_viewer.fit_in_view)
        view_menu.addAction(fit_action)
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self._image_viewer.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self._image_viewer.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Edit menu with Auto Preview toggle
        edit_menu = menubar.addMenu("&Edit")
        
        self._auto_preview_action = QAction("&Auto Preview", self)
        self._auto_preview_action.setCheckable(True)
        self._auto_preview_action.setChecked(True)  # Default on
        self._auto_preview_action.setToolTip("Automatically preview preprocessing when group is selected")
        self._auto_preview_action.toggled.connect(self._on_auto_preview_toggled)
        edit_menu.addAction(self._auto_preview_action)
        
        edit_menu.addSeparator()
        
        track_action = QAction("&Track Roots", self)
        track_action.setShortcut(QKeySequence("Ctrl+T"))
        track_action.triggered.connect(self._on_track_roots)
        edit_menu.addAction(track_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Workflow bar
        self._workflow_bar.step_changed.connect(self._on_step_changed)
        
        # Image tree
        self._image_tree.image_selected.connect(self._on_image_selected)
        self._image_tree.group_selected.connect(self._on_group_selected)
        self._image_tree.load_requested.connect(self._on_load_images)
        
        # Settings panel
        self._settings_panel.apply_requested.connect(self._on_apply_settings)
        self._settings_panel.redetect_requested.connect(self._on_redetect_plants)
        self._settings_panel.track_requested.connect(self._on_track_roots)
        self._settings_panel.export_requested.connect(self._on_export_results)
        
        # Image viewer - update zoom label in bottom bar
        self._image_viewer.zoom_changed.connect(self._on_zoom_changed)
    
    def _on_load_images(self) -> None:
        """Handle load images action (Ctrl+O)."""
        dialog = LoadDialog(
            self, 
            self._config.data.input,
            self._config.data.filename_template,
            self._config.data.date_format
        )
        
        if dialog.exec() == LoadDialog.DialogCode.Accepted:
            # Update config
            self._config.data.input = dialog.input_path
            self._config.data.filename_template = dialog.filename_template
            self._config.data.date_format = dialog.date_format
            
            # Create pipeline and load images
            self._pipeline = RootTrackingPipeline(self._config)
            
            self._status_label.setText("Loading images...")
            QApplication.processEvents()
            
            try:
                self._series_dict = self._pipeline.load_images()
                
                if not self._series_dict:
                    QMessageBox.warning(
                        self, "No Images Found",
                        f"No images found in '{self._config.data.input}' matching template '{self._config.data.filename_template}'.\n\n"
                        "Check that the folder path and filename template are correct."
                    )
                    self._status_label.setText("No images found.")
                    return
                
                self._image_tree.set_series(self._series_dict)
                
                # Select first image
                self._image_tree.select_first_image()
                
                # Mark step as complete but DON'T auto-advance
                self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
                
                total_images = sum(len(s.images) for s in self._series_dict.values())
                self._status_label.setText(
                    f"Loaded {len(self._series_dict)} groups, {total_images} images. Click Next to continue."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load images:\n{e}")
                self._status_label.setText("Load failed.")
    

    def _on_image_selected(self, image_data: ImageData) -> None:
        """Handle image selection in tree."""
        self._current_image = image_data
        self._current_series = self._image_tree.get_selected_series()
        
        # Load and display the image
        self._display_image(image_data)
        
        self._status_label.setText(f"Selected: {image_data.barcode or image_data.path}")
    
    def _on_group_selected(self, series: ImageSeries) -> None:
        """Handle group selection in tree."""
        # Reset settings for new group (discard unsaved changes)
        self._settings_panel.reset_for_group()
        
        self._current_series = series
        self._current_image = series.images[0] if series.images else None
        
        if self._current_image:
            self._display_image(self._current_image)
        
        # Auto-preview if enabled and on Preprocess step
        step = self._workflow_bar.get_current_step()
        if (step == WorkflowStep.PREPROCESS and 
            self._auto_preview_action.isChecked() and 
            self._pipeline is not None):
            self._preprocess_group(series)
        
        self._status_label.setText(f"Selected group: {series.group} ({len(series.images)} images)")
    
    def _display_image(self, image_data: ImageData) -> None:
        """Display an image in the viewer."""
        # Determine which image to show based on workflow step
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Show original image
            image = cv2.imread(image_data.path)
        elif step == WorkflowStep.PREPROCESS:
            # Show processed image if available, else original
            if image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
        else:
            # Show annotated image if available
            if image_data.image is not None:
                image = image_data.image
            elif image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
        
        if image is not None:
            self._image_viewer.set_image(image)
    
    def _on_step_changed(self, step: WorkflowStep) -> None:
        """Handle workflow step change."""
        # Validate step prerequisites
        if step > WorkflowStep.LOAD and not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
            QMessageBox.warning(
                self, "Step Not Available",
                "Please load images first (Step 1) before proceeding."
            )
            # Reset workflow bar to current step
            self._workflow_bar.set_current_step(WorkflowStep.LOAD)
            return
        
        # Hide settings panel on Load step (use Ctrl+O instead)
        if step == WorkflowStep.LOAD:
            self._settings_panel.hide()
        else:
            self._settings_panel.show()
            self._settings_panel.set_step(step)
        
        # Refresh image display for new step
        if self._current_image:
            self._display_image(self._current_image)
        
        # Auto-preview when switching to Preprocess step
        if (step == WorkflowStep.PREPROCESS and 
            self._auto_preview_action.isChecked() and 
            self._current_series is not None and
            self._pipeline is not None):
            self._preprocess_group(self._current_series)
        
        self._status_label.setText(f"Step: {step.name}")
    
    def _on_apply_settings(self) -> None:
        """Handle Apply button - reprocess current group with new settings."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return
        
        # Update config from settings panel
        values = self._settings_panel.get_current_values()
        for key, value in values.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Reprocess current group
        self._preprocess_group(self._current_series)
    
    def _on_redetect_plants(self) -> None:
        """Re-run plant centroid detection (after user moved centroids)."""
        if self._pipeline is None or self._current_series is None:
            return
        
        self._status_label.setText("Re-detecting plant centroids...")
        QApplication.processEvents()
        
        try:
            # TODO: Re-run centroid detection on current group
            # For now, just reprocess
            for image_data in self._current_series.images:
                self._pipeline.preprocess_image(image_data)
            
            if self._current_image:
                self._display_image(self._current_image)
            
            self._status_label.setText("Plant detection complete.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed:\n{e}")
    
    def _on_auto_preview_toggled(self, enabled: bool) -> None:
        """Handle Auto Preview menu toggle."""
        if enabled:
            self._preview_btn.setEnabled(False)
            self._preview_btn.setToolTip("Auto Preview is enabled - processing happens automatically")
        else:
            self._preview_btn.setEnabled(True)
            self._preview_btn.setToolTip("Click to preview preprocessing for current group")
    
    def _on_preview_clicked(self) -> None:
        """Handle Preview button click (when Auto Preview is disabled)."""
        if self._current_series is not None:
            self._preprocess_group(self._current_series)
    
    def _preprocess_group(self, series: 'ImageSeries') -> None:
        """Preprocess and register all images in a group with progress."""
        if self._pipeline is None:
            return
        
        # Check if already processed (first image has process data)
        if series.images and series.images[0].process is not None:
            self._status_label.setText(f"Group '{series.group}' already preprocessed.")
            if self._current_image:
                self._display_image(self._current_image)
            return
        
        # Lock UI
        self._set_ui_locked(True)
        
        # Show progress bar in status bar
        self._progress_bar.setMaximum(len(series.images))
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        
        try:
            for i, image_data in enumerate(series.images):
                self._status_label.setText(f"Preprocessing {i + 1}/{len(series.images)}...")
                self._pipeline.preprocess_image(image_data)
                self._progress_bar.setValue(i + 1)
                QApplication.processEvents()
            
            # Register the series
            self._pipeline.register_series(series)
            
            # Only mark complete if not already completed
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)
            
            if self._current_image:
                self._display_image(self._current_image)
            
            self._status_label.setText(
                f"Preprocessed {len(series.images)} images."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._set_ui_locked(False)
    
    def _set_ui_locked(self, locked: bool) -> None:
        """Lock/unlock UI during processing."""
        self._workflow_bar.setEnabled(not locked)
        self._settings_panel.setEnabled(not locked)
        self._image_tree.setEnabled(not locked)  # Freeze image selection
        self._back_step_btn.setEnabled(not locked)
        self._next_step_btn.setEnabled(not locked)
        self._fit_btn.setEnabled(not locked)
        self._zoom_in_btn.setEnabled(not locked)
        self._zoom_out_btn.setEnabled(not locked)
        if not self._auto_preview_action.isChecked():
            self._preview_btn.setEnabled(not locked)
    
    def _on_track_roots(self) -> None:
        """Run root tracking on current group."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please preprocess images first.")
            return
        
        self._status_label.setText("Tracking roots...")
        QApplication.processEvents()
        
        try:
            stats = self._pipeline.track_and_analyze_series(self._current_series)
            
            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)
            self._workflow_bar.set_current_step(WorkflowStep.EXPORT)
            self._settings_panel.set_step(WorkflowStep.EXPORT)
            
            if self._current_image:
                self._display_image(self._current_image)
            
            self._status_label.setText(f"Tracking complete. {len(stats)} records generated.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
    
    def _on_export_results(self) -> None:
        """Export results to CSV."""
        if self._pipeline is None:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        dialog = ExportDialog(self, self._config.data.output)
        
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            self._config.data.output = dialog.output_path
            
            self._status_label.setText("Exporting...")
            QApplication.processEvents()
            
            try:
                # Get all statistics
                all_stats = []
                for series in self._series_dict.values():
                    stats = self._pipeline.track_and_analyze_series(series)
                    all_stats.extend(stats)
                
                csv_path = self._pipeline.export_results(all_stats)
                
                self._workflow_bar.mark_step_completed(WorkflowStep.EXPORT)
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Results exported to:\n{csv_path}"
                )
                self._status_label.setText(f"Exported to {csv_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
    
    def _on_zoom_changed(self, percentage: int) -> None:
        """Handle zoom level change - update bottom bar label."""
        self._zoom_label.setText(f"{percentage}%")
    
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Root Tracker",
            "<h3>Root Tracker</h3>"
            "<p>A tool for tracking and analyzing plant root growth.</p>"
            "<p>Version 1.0</p>"
        )
    
    def _on_step_back(self) -> None:
        """Navigate to previous workflow step."""
        current = self._workflow_bar.get_current_step()
        if current > WorkflowStep.LOAD:
            new_step = WorkflowStep(current - 1)
            self._workflow_bar.set_current_step(new_step)
            self._on_step_changed(new_step)
    
    def _on_step_next(self) -> None:
        """Navigate to next workflow step."""
        current = self._workflow_bar.get_current_step()
        if current < WorkflowStep.EXPORT:
            new_step = WorkflowStep(current + 1)
            # Use the same validation as clicking on step
            self._workflow_bar.set_current_step(new_step)
            self._on_step_changed(new_step)
