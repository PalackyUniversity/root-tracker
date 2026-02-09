"""
Main application window for Root Tracker GUI.

Integrates all components into the main window layout.
"""

import os
import time
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
        
        # Set initial button states (no images loaded yet)
        self._update_initial_ui_state()
    
    def _setup_ui(self) -> None:
        """Set up the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        
        # Initialize workflow bar (will be added to menu bar)
        self._workflow_bar = WorkflowBar()
        
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
        
        # Right side: Processing label + progress bar (hidden by default), groups progress, buttons
        self._processing_label = QLabel("")
        self._processing_label.hide()
        self._status_bar.addPermanentWidget(self._processing_label)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedSize(150, 18)
        self._progress_bar.setMaximum(100)
        self._progress_bar.hide()
        self._status_bar.addPermanentWidget(self._progress_bar)
        
        self._groups_progress_label = QLabel("0/0 groups processed")
        self._groups_progress_label.setMinimumWidth(150)
        self._status_bar.addPermanentWidget(self._groups_progress_label)
        
        # Track warning count for groups
        self._warning_count = 0
        
        self._process_group_btn = QPushButton("Process this group")
        self._process_group_btn.setFixedHeight(26)
        self._process_group_btn.setToolTip("Process the currently selected group")
        self._process_group_btn.clicked.connect(self._on_process_group_clicked)
        self._status_bar.addPermanentWidget(self._process_group_btn)
        
        self._process_all_btn = QPushButton("Process all groups")
        self._process_all_btn.setFixedHeight(26)
        self._process_all_btn.setToolTip("Process all groups")
        self._process_all_btn.clicked.connect(self._on_process_all_clicked)
        self._status_bar.addPermanentWidget(self._process_all_btn)
        
        # Next step button (rightmost)
        self._next_step_btn = QPushButton("Next")
        self._next_step_btn.setFixedHeight(26)
        self._next_step_btn.setToolTip("Go to next step (auto-processes current step if needed)")
        self._next_step_btn.clicked.connect(self._on_next_step_clicked)
        self._status_bar.addPermanentWidget(self._next_step_btn)
        
        # Small spacer after buttons
        spacer2 = QWidget()
        spacer2.setFixedWidth(0.25)
        self._status_bar.addPermanentWidget(spacer2)
    
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # Style the menu bar to match workflow bar height and center items
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: transparent;
                border-bottom: 1px solid #333;
                min-height: 38px;
            }
            QMenuBar::item {
                height: 38px;
                background: transparent;
                font-size: 14px;
                vertical-align: middle;
            }
            QMenuBar::item:selected {
                background: #444;
            }
        """)
        
        # Add workflow bar to right side of menu bar
        menubar.setCornerWidget(self._workflow_bar, Qt.Corner.TopRightCorner)
        
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
        self._settings_panel.apply_all_requested.connect(self._on_apply_all_settings)
        self._settings_panel.redetect_requested.connect(self._on_redetect_plants)
        self._settings_panel.track_requested.connect(self._on_track_roots)
        self._settings_panel.export_requested.connect(self._on_export_results)
        
        # Image viewer - centroid dragging
        self._image_viewer.centroid_moved.connect(self._on_centroid_moved)
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
            
            try:
                self._series_dict = self._pipeline.load_images()
                
                if not self._series_dict:
                    QMessageBox.warning(
                        self, "No Images Found",
                        f"No images found in '{self._config.data.input}' matching template '{self._config.data.filename_template}'.\n\n"
                        "Check that the folder path and filename template are correct."
                    )
                    return
                
                # Initialize warning count (barcode detection is lazy - done when viewing)
                self._warning_count = 0
                
                self._image_tree.set_series(self._series_dict)
                
                # Select first image
                self._image_tree.select_first_image()
                
                # Mark step as complete but DON'T auto-advance
                self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
                
                # Update groups progress label
                self._update_groups_progress()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load images:\n{e}")
    

    def _on_image_selected(self, image_data: ImageData) -> None:
        """Handle image selection in tree."""
        self._current_image = image_data
        self._current_series = self._image_tree.get_selected_series()
        
        step = self._workflow_bar.get_current_step()
        
        # Auto-detect barcodes for this group if on LOAD step and auto-preview enabled
        if (step == WorkflowStep.LOAD and 
            self._auto_preview_action.isChecked() and 
            self._pipeline is not None and
            self._current_series is not None):
            # Only detect if group has undetected barcodes
            if not all(img.barcode_detected for img in self._current_series.images):
                self._detect_barcodes_in_group(self._current_series)
                return  # _detect_barcodes_in_group already displays the image
        
        # Auto-preprocess if enabled and on PREPROCESS step
        if (step == WorkflowStep.PREPROCESS and 
            self._auto_preview_action.isChecked() and 
            self._pipeline is not None and
            self._current_series is not None):
            self._preprocess_group(self._current_series)
            return
        
        # Load and display the image
        self._display_image(image_data)
        
        # Update button states for new selection
        self._update_process_button_states()
    
    def _on_group_selected(self, series: ImageSeries) -> None:
        """Handle group selection in tree."""
        # Reset settings for new group (discard unsaved changes)
        self._settings_panel.reset_for_group()
        
        self._current_series = series
        self._current_image = series.images[0] if series.images else None
        
        step = self._workflow_bar.get_current_step()
        
        # Auto-detect barcodes if on LOAD step and auto-preview enabled
        if (step == WorkflowStep.LOAD and 
            self._auto_preview_action.isChecked() and 
            self._pipeline is not None):
            self._detect_barcodes_in_group(series)
        
        if self._current_image:
            self._display_image(self._current_image)
        
        # Auto-preprocess if enabled and on PREPROCESS step
        if (step == WorkflowStep.PREPROCESS and 
            self._auto_preview_action.isChecked() and 
            self._pipeline is not None):
            self._preprocess_group(series)
        
        # Update button states for new selection
        self._update_process_button_states()
    
    def _display_image(self, image_data: ImageData) -> None:
        """Display an image in the viewer."""
        # Determine which image to show based on workflow step
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Show original image with barcode overlay
            image = cv2.imread(image_data.path)
            
            # Rotate if needed (to match barcode detection)
            if self._config.rotation:
                from ..preprocessing import ImageCropper
                cropper = ImageCropper(self._config)
                image = cropper.rotate(image)
            
            # Draw barcode overlay if detected (detection is done at group level)
            if image_data.barcode_rect is not None:
                x, y, w, h = image_data.barcode_rect
                # Green for match, red for mismatch
                color = (0, 0, 255) if image_data.barcode_mismatch else (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                
                # Draw detected barcode text
                label = image_data.barcode_read
                if image_data.barcode_mismatch:
                    label = f"X {label} (expected: {image_data.barcode})"
                cv2.putText(
                    image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
                )
            
            self._image_viewer.clear_centroids()
        elif step == WorkflowStep.PREPROCESS:
            # Show processed image (RGB) if available, else original (rotated + cropped)
            if image_data.image is not None:
                image = image_data.image
                # Show centroids if we have position data
                if image_data.positions_x and image_data.positions_y:
                    positions = list(zip(image_data.positions_x, image_data.positions_y))
                    self._image_viewer.set_centroids(positions)
                else:
                    self._image_viewer.clear_centroids()
            else:
                # Preview: Rotate and auto-crop (to avoid "flash" of raw image)
                image = cv2.imread(image_data.path)
                
                from ..preprocessing import ImageCropper
                cropper = ImageCropper(self._config)
                
                # Rotate
                image = cropper.rotate(image)
                
                # Auto-crop (try to simulate what processing will do)
                try:
                    image = cropper.auto_crop_to_blue_background(image)
                except Exception:
                    pass
                
                self._image_viewer.clear_centroids()
        else:
            # Show annotated image if available
            if image_data.image is not None:
                image = image_data.image
            elif image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
            self._image_viewer.clear_centroids()
        
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
        
        # Auto-preview when switching to Preprocess step
        if (step == WorkflowStep.PREPROCESS and 
            self._auto_preview_action.isChecked() and 
            self._current_series is not None and
            self._pipeline is not None):
            # Don't display image yet - wait for processing to finish
            # The _preprocess_group content will handle display at the end
            self._preprocess_group(self._current_series)
        else:
            # Refresh image display for new step immediately
            if self._current_image:
                self._display_image(self._current_image)
        
        # Update button states for new step
        self._update_process_button_states()
        self._update_groups_progress()
    
    def _update_config_from_panel(self) -> None:
        """Update config object from settings panel values."""
        values = self._settings_panel.get_current_values()
        for key, value in values.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            
            # Handle nested registration settings
            if key == "reg_enabled":
                self._config.registration.enabled = value
            elif key == "reg_margin":
                self._config.registration.margin_ratio = value
    
    def _on_apply_settings(self) -> None:
        """Handle Apply button - reprocess current group with new settings."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return
        
        # Update config from settings panel
        self._update_config_from_panel()
        
        # Reprocess current group
        self._preprocess_group(self._current_series)
        
    def _on_apply_all_settings(self) -> None:
        """Handle Apply All button - reprocess ALL groups with new settings."""
        if self._pipeline is None or not self._series_dict:
            return
        
        # Update config from settings panel
        self._update_config_from_panel()
        
        # Process all groups
        self._preprocess_all_groups()
    
    def _on_centroid_moved(self, index: int, x: float, y: float) -> None:
        """Handle centroid drag - update image data and enable Re-detect."""
        if self._current_image is None:
            return
        
        # Update the position in image data
        if self._current_image.positions_x and index < len(self._current_image.positions_x):
            self._current_image.positions_x[index] = x
        if self._current_image.positions_y and index < len(self._current_image.positions_y):
            self._current_image.positions_y[index] = y
        
        # Notify settings panel that centroids were modified
        self._settings_panel.mark_centroids_modified()
    
    def _on_redetect_plants(self) -> None:
        """Re-run plant centroid detection (after user moved centroids)."""
        if self._pipeline is None or self._current_series is None:
            return
        
        self._processing_label.setText("Re-detecting...")
        self._processing_label.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            # TODO: Re-run centroid detection on current group
            # For now, just reprocess
            for image_data in self._current_series.images:
                self._pipeline.preprocess_image(image_data)
            
            if self._current_image:
                self._display_image(self._current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed:\n{e}")
        finally:
            self._processing_label.hide()
            QApplication.restoreOverrideCursor()
    
    def _on_auto_preview_toggled(self, enabled: bool) -> None:
        """Handle Auto Preview menu toggle."""
        # Auto Preview controls whether preprocessing happens automatically on group selection
        pass
    
    def _on_process_group_clicked(self) -> None:
        """Handle 'Process group' button click - step aware."""
        if self._current_series is None or self._pipeline is None:
            return
        
        step = self._workflow_bar.get_current_step()
        if step == WorkflowStep.LOAD:
            self._detect_barcodes_in_group(self._current_series)
        elif step == WorkflowStep.PREPROCESS:
            self._preprocess_group(self._current_series)
    
    def _on_process_all_clicked(self) -> None:
        """Handle 'Process all groups' button click - step aware."""
        if self._pipeline is None or not self._series_dict:
            return
        
        step = self._workflow_bar.get_current_step()
        if step == WorkflowStep.LOAD:
            self._detect_barcodes_all_groups()
        elif step == WorkflowStep.PREPROCESS:
            self._preprocess_all_groups()
    
    def _detect_barcodes_in_group(self, series: 'ImageSeries') -> None:
        """Detect barcodes for all images in a group with progress bar."""
        if self._pipeline is None:
            return
        
        # Check if barcodes already detected for this group
        if all(img.barcode_detected for img in series.images):
            return
        
        # Lock UI
        self._set_ui_locked(True)
        
        # Show progress bar
        total_images = len(series.images)
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.setText("Detecting barcodes...")
        self._processing_label.show()
        
        start_time = time.time()
        
        try:
            for i, image_data in enumerate(series.images):
                if not image_data.barcode_detected:
                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if i > 0 and elapsed > 0:
                        avg_time = elapsed / i
                        remaining = total_images - i
                        eta_seconds = int(avg_time * remaining)
                        if eta_seconds >= 60:
                            eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                        else:
                            eta_text = f"{eta_seconds}s"
                        self._processing_label.setText(f"ETA: {eta_text}")
                    
                    self._pipeline.detect_barcode_in_image(image_data)
                
                self._progress_bar.setValue(i + 1)
                QApplication.processEvents()
            
            # Update warning count and tree (refresh preserves selection)
            self._recalculate_warning_count()
            self._update_groups_progress()
            self._image_tree.refresh()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Barcode detection failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
        
        # Restore focus (tree was disabled during processing)
        self._image_tree.setFocus()
        
        # Display the current image
        if self._current_image:
            self._display_image(self._current_image)
    
    def _detect_barcodes_all_groups(self) -> None:
        """Detect barcodes for all groups with progress bar."""
        if self._pipeline is None or not self._series_dict:
            return
        
        # Find groups with undetected barcodes
        undetected_groups = [
            s for s in self._series_dict.values()
            if not all(img.barcode_detected for img in s.images)
        ]
        
        if not undetected_groups:
            return
        
        total_images = sum(len(s.images) for s in undetected_groups)
        
        # Lock UI
        self._set_ui_locked(True)
        
        # Setup cumulative progress bar
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.setText("Detecting barcodes...")
        self._processing_label.show()
        
        current_image_count = 0
        start_time = time.time()
        
        try:
            for series in undetected_groups:
                for image_data in series.images:
                    if not image_data.barcode_detected:
                        current_image_count += 1
                        
                        # Calculate ETA
                        elapsed = time.time() - start_time
                        if current_image_count > 0 and elapsed > 0:
                            avg_time = elapsed / current_image_count
                            remaining = total_images - current_image_count
                            eta_seconds = int(avg_time * remaining)
                            if eta_seconds >= 60:
                                eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                            else:
                                eta_text = f"{eta_seconds}s"
                            self._processing_label.setText(f"ETA: {eta_text}")
                        
                        self._pipeline.detect_barcode_in_image(image_data)
                    else:
                        current_image_count += 1
                    
                    self._progress_bar.setValue(current_image_count)
                    QApplication.processEvents()
                
                # Update warning count after each group
                self._recalculate_warning_count()
                self._update_groups_progress()
            
            # Update tree to show warning icons (refresh preserves selection)
            self._image_tree.refresh()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Barcode detection failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
        
        # Restore focus (tree was disabled during processing)
        self._image_tree.setFocus()
        
        # Display current image if we have one
        if self._current_image:
            self._display_image(self._current_image)
    
    def _preprocess_all_groups(self) -> None:
        """Preprocess all unprocessed groups with progress bar."""
        if self._pipeline is None or not self._series_dict:
            return
        
        # Calculate total images across all unprocessed groups
        unprocessed_groups = [
            s for s in self._series_dict.values()
            if not s.images or s.images[0].process is None
        ]
        
        if not unprocessed_groups:
            return
        
        total_images = sum(len(s.images) for s in unprocessed_groups)
        
        # Lock UI
        self._set_ui_locked(True)
        
        # Setup cumulative progress bar
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.show()
        
        current_image_count = 0
        start_time = time.time()
        
        try:
            for group_idx, series in enumerate(unprocessed_groups):
                for i, image_data in enumerate(series.images):
                    current_image_count += 1
                    
                    # Calculate ETA
                    elapsed = time.time() - start_time
                    if current_image_count > 0 and elapsed > 0:
                        avg_time_per_image = elapsed / current_image_count
                        remaining_images = total_images - current_image_count
                        eta_seconds = int(avg_time_per_image * remaining_images)
                        if eta_seconds >= 60:
                            eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                        else:
                            eta_text = f"{eta_seconds}s"
                        self._processing_label.setText(f"ETA: {eta_text}")
                    else:
                        self._processing_label.setText("Processing...")
                    
                    self._pipeline.preprocess_image(image_data)
                    self._progress_bar.setValue(current_image_count)
                    QApplication.processEvents()
                
                # Register the series
                self._pipeline.register_series(series)
                
                # Update groups progress after each group
                self._update_groups_progress()
            
            # Mark step complete
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)
            
            if self._current_image:
                self._display_image(self._current_image)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
        
        # Restore focus to tree
        self._image_tree.setFocus()
    
    def _update_groups_progress(self, warning_text: str = "") -> None:
        """Update the groups processed label in status bar (step-aware)."""
        total = len(self._series_dict) if self._series_dict else 0
        
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Count groups with barcodes detected
            processed = sum(
                1 for series in self._series_dict.values()
                if series.images and all(img.barcode_detected for img in series.images)
            ) if self._series_dict else 0
            text = f"{processed}/{total} groups scanned"
        else:
            # Count groups with preprocessing done
            processed = sum(
                1 for series in self._series_dict.values()
                if series.images and series.images[0].process is not None
            ) if self._series_dict else 0
            text = f"{processed}/{total} groups processed"
        
        if self._warning_count > 0:
            text += f" ({self._warning_count} warnings)"
        self._groups_progress_label.setText(text)
    
    def _recalculate_warning_count(self) -> None:
        """Recalculate total warning count from all images."""
        if not self._series_dict:
            self._warning_count = 0
            return
        
        self._warning_count = sum(
            1 for series in self._series_dict.values()
            for img in series.images
            if img.barcode_mismatch
        )
    
    def _update_tree_warnings(self) -> None:
        """Refresh tree to show updated warning icons."""
        # Only refresh if there's a mismatch in the current series
        if self._current_series and self._current_series.has_barcode_warning:
            self._image_tree.refresh()
    
    def _preprocess_group(self, series: 'ImageSeries') -> None:
        """Preprocess and register all images in a group with progress."""
        if self._pipeline is None:
            return
        
        # Check if already processed (first image has process data)
        if series.images and series.images[0].process is not None:
            if self._current_image:
                self._display_image(self._current_image)
            return
        
        # Lock UI
        self._set_ui_locked(True)
        
        # Show progress bar in status bar
        total_images = len(series.images)
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.show()
        
        start_time = time.time()
        
        try:
            for i, image_data in enumerate(series.images):
                # Calculate ETA
                elapsed = time.time() - start_time
                if i > 0 and elapsed > 0:
                    avg_time_per_image = elapsed / i
                    remaining_images = total_images - i
                    eta_seconds = int(avg_time_per_image * remaining_images)
                    if eta_seconds >= 60:
                        eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                    else:
                        eta_text = f"{eta_seconds}s"
                    self._processing_label.setText(f"ETA: {eta_text}")
                else:
                    self._processing_label.setText("Processing...")
                
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
            
            self._update_groups_progress()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
        
        # Restore focus to tree
        self._image_tree.setFocus()
    
    def _set_ui_locked(self, locked: bool) -> None:
        """Lock/unlock UI during processing."""
        if locked:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

        self._workflow_bar.setEnabled(not locked)
        self._settings_panel.setEnabled(not locked)
        self._image_tree.setEnabled(not locked)  # Freeze image selection
        self._fit_btn.setEnabled(not locked)
        self._zoom_in_btn.setEnabled(not locked)
        self._zoom_out_btn.setEnabled(not locked)
        
        if locked:
            self._process_group_btn.setEnabled(False)
            self._process_all_btn.setEnabled(False)
            self._next_step_btn.setEnabled(False)
        else:
            self._update_process_button_states()
    
    def _update_process_button_states(self) -> None:
        """Update process button enabled states based on current step and processing status."""
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Check if current group has barcodes detected
            group_done = False
            if self._current_series and self._current_series.images:
                group_done = all(img.barcode_detected for img in self._current_series.images)
            
            # Check if all groups have barcodes detected
            all_done = False
            if self._series_dict:
                all_done = all(
                    all(img.barcode_detected for img in series.images)
                    for series in self._series_dict.values()
                    if series.images
                )
            
            self._process_group_btn.setEnabled(not group_done and self._current_series is not None)
            self._process_all_btn.setEnabled(not all_done and bool(self._series_dict))
            
        elif step == WorkflowStep.PREPROCESS:
            # Check if current group is preprocessed
            group_done = False
            if self._current_series and self._current_series.images:
                group_done = self._current_series.images[0].process is not None
            
            # Check if all groups are preprocessed
            all_done = False
            if self._series_dict:
                all_done = all(
                    series.images and series.images[0].process is not None
                    for series in self._series_dict.values()
                )
            
            self._process_group_btn.setEnabled(not group_done and self._current_series is not None)
            self._process_all_btn.setEnabled(not all_done and bool(self._series_dict))
        else:
            # Other steps - disable both buttons
            self._process_group_btn.setEnabled(False)
            self._process_all_btn.setEnabled(False)
        
        # Next button: enabled unless on Export step or no images
        on_export = (step == WorkflowStep.EXPORT)
        self._next_step_btn.setEnabled(not on_export and bool(self._series_dict))
        
        # Zoom controls: enabled only when images are loaded
        has_images = bool(self._series_dict)
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)
        self._zoom_label.setEnabled(has_images)
        
        # Groups progress label: visible only when images are loaded
        self._groups_progress_label.setVisible(has_images)
    
    def _update_initial_ui_state(self) -> None:
        """Set initial UI state when no images are loaded."""
        # Disable zoom controls
        self._fit_btn.setEnabled(False)
        self._zoom_in_btn.setEnabled(False)
        self._zoom_out_btn.setEnabled(False)
        self._zoom_label.setEnabled(False)
        
        # Disable process buttons
        self._process_group_btn.setEnabled(False)
        self._process_all_btn.setEnabled(False)
        self._next_step_btn.setEnabled(False)
        
        # Hide groups progress label
        self._groups_progress_label.hide()
    
    def _on_track_roots(self) -> None:
        """Run root tracking on current group."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please preprocess images first.")
            return
        
        self._processing_label.setText("Tracking roots...")
        self._processing_label.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            stats = self._pipeline.track_and_analyze_series(self._current_series)
            
            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)
            self._workflow_bar.set_current_step(WorkflowStep.EXPORT)
            self._settings_panel.set_step(WorkflowStep.EXPORT)
            
            if self._current_image:
                self._display_image(self._current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
        finally:
            self._processing_label.hide()
            QApplication.restoreOverrideCursor()
    
    def _on_export_results(self) -> None:
        """Export results to CSV."""
        if self._pipeline is None:
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        dialog = ExportDialog(self, self._config.data.output)
        
        if dialog.exec() == ExportDialog.DialogCode.Accepted:
            self._config.data.output = dialog.output_path
            
            self._processing_label.setText("Exporting...")
            self._processing_label.show()
            QApplication.setOverrideCursor(Qt.WaitCursor)
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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
            finally:
                self._processing_label.hide()
                QApplication.restoreOverrideCursor()
    
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
    
    def _on_next_step_clicked(self) -> None:
        """
        Handle Next button click.
        
        Auto-processes the current step if not complete, then advances to next step.
        """
        current = self._workflow_bar.get_current_step()
        
        # Can't go past export
        if current >= WorkflowStep.EXPORT:
            return
        
        # Check if we have images loaded (required for all steps)
        if not self._series_dict:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return
        
        # Auto-process current step if needed
        if current == WorkflowStep.LOAD:
            # Ensure all barcodes are detected
            if not all(
                all(img.barcode_detected for img in series.images)
                for series in self._series_dict.values()
                if series.images
            ):
                self._detect_barcodes_all_groups()
            
            # Mark as completed and move to next
            if not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
                self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
            
        elif current == WorkflowStep.PREPROCESS:
            # Ensure barcodes were detected first (previous step)
            if not all(
                all(img.barcode_detected for img in series.images)
                for series in self._series_dict.values()
                if series.images
            ):
                self._detect_barcodes_all_groups()
                if not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
                    self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
            
            # Ensure all groups are preprocessed
            if not all(
                series.images and series.images[0].process is not None
                for series in self._series_dict.values()
            ):
                self._preprocess_all_groups()
            
            # Mark as completed
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)
                
        elif current == WorkflowStep.TRACK:
            # Ensure previous steps are done
            if not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
                self._detect_barcodes_all_groups()
                self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
            
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._preprocess_all_groups()
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)
            
            # Track is handled differently - just move to export
            if not self._workflow_bar.is_step_completed(WorkflowStep.TRACK):
                self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)
        
        # Advance to next step
        new_step = WorkflowStep(current + 1)
        self._workflow_bar.set_current_step(new_step)
        self._on_step_changed(new_step)
