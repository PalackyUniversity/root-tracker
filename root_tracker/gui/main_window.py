"""
Main application window for Root Tracker GUI.

Integrates all components into the main window layout.
"""

import os
import time
import cv2
import numpy as np
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
from ..io import ImageLoader, mask_io

from .workflow_bar import WorkflowBar, WorkflowStep
from .image_tree import ImageTree
from .image_viewer import ImageViewer
from .settings_panel import SettingsPanel
from .dialogs import LoadDialog, ExportDialog
from .masking_tools import MaskTool

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
        # self._settings_panel.track_requested.connect(self._on_track_roots)
        self._settings_panel.export_requested.connect(self._on_export_results)

        # Masking signals
        self._settings_panel.mask_tool_changed.connect(self._on_mask_tool_changed)
        self._settings_panel.mask_erase_all_requested.connect(self._on_mask_erase_all)

        # Image viewer - centroid dragging
        self._image_viewer.centroid_moved.connect(self._on_centroid_moved)
        # Image viewer - update zoom label in bottom bar
        self._image_viewer.zoom_changed.connect(self._on_zoom_changed)
        # Image viewer - mask modification
        self._image_viewer.mask_modified.connect(self._on_mask_modified)
    
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
            # First ensure barcodes are detected
            if not all(img.barcode_detected for img in self._current_series.images):
                self._detect_barcodes_in_group(self._current_series, continue_to_next_step=True)
                return
            self._preprocess_group(self._current_series)
            return

        # Auto-process if enabled and on TRACK step
        if (step == WorkflowStep.TRACK and
            self._auto_preview_action.isChecked() and
            self._pipeline is not None and
            self._current_series is not None):
            # First ensure barcodes are detected
            if not all(img.barcode_detected for img in self._current_series.images):
                self._detect_barcodes_in_group(self._current_series, continue_to_next_step=True)
                return
            self._auto_process_for_tracking(self._current_series)
            return

        # Load and display the image
        self._display_image(image_data)

        # Update button states for new selection
        self._update_process_button_states()
    
    def _on_group_selected(self, series: ImageSeries) -> None:
        """Handle group selection in tree."""
        # Reset settings for new group (discard unsaved changes)
        self._settings_panel.reset_for_group()

        # Deselect mask tools when switching groups
        self._settings_panel.deselect_mask_tools()

        self._current_series = series
        self._current_image = series.images[0] if series.images else None

        # Initialize working_mask if needed
        if series.working_mask is None and series.user_mask is not None:
            series.working_mask = series.user_mask.copy()

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
            # First ensure barcodes are detected
            if not all(img.barcode_detected for img in series.images):
                self._detect_barcodes_in_group(series, continue_to_next_step=True)
                return
            self._preprocess_group(series)

        # Auto-process if enabled and on TRACK step
        if (step == WorkflowStep.TRACK and
            self._auto_preview_action.isChecked() and
            self._pipeline is not None):
            # First ensure barcodes are detected
            if not all(img.barcode_detected for img in series.images):
                self._detect_barcodes_in_group(series, continue_to_next_step=True)
                return
            self._auto_process_for_tracking(series)

        # Update button states for new selection
        self._update_process_button_states()
    
    def _display_image(self, image_data: ImageData) -> None:
        """Display an image in the viewer."""
        # Determine which image to show based on workflow step
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Show original image with barcode overlay
            image = cv2.imread(image_data.path)
            
            # No rotation in LOAD step - show raw image
            pass
            
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
                # Don't show centroids in Preprocess step (requested)
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
            # TRACK/EXPORT: prefer annotated image, fall back to preprocessed
            if image_data.image_annotated is not None:
                image = image_data.image_annotated
            elif image_data.image is not None:
                image = image_data.image
            elif image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
            self._image_viewer.clear_centroids()

        if image is not None:
            self._image_viewer.set_image(image)

            # Set mask data if in TRACK step
            if step == WorkflowStep.TRACK and self._current_series is not None:
                self._image_viewer.set_mask_data(
                    self._current_series.user_mask,
                    self._current_series.working_mask
                )
            else:
                # Clear mask data in other steps
                self._image_viewer.set_mask_data(None, None)
    
    def _on_step_changed(self, step: WorkflowStep) -> None:
        """Handle workflow step change."""
        # Validate step prerequisites
        if step > WorkflowStep.LOAD and not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
            QMessageBox.warning(
                self, "Step Not Available",
                "Please load images first (Step 1) before proceeding."
            )
            self._workflow_bar.set_current_step(WorkflowStep.LOAD)
            return

        # Show/hide settings panel
        if step == WorkflowStep.LOAD:
            self._settings_panel.hide()
        else:
            self._settings_panel.show()
            self._settings_panel.set_step(step)

        # Step-specific auto-processing
        if step == WorkflowStep.PREPROCESS:
            self._handle_enter_preprocess()
        elif step == WorkflowStep.TRACK:
            self._handle_enter_track()
        else:
            # LOAD and EXPORT: just refresh display
            if self._current_image:
                self._display_image(self._current_image)

        self._update_process_button_states()
        self._update_groups_progress()

    def _handle_enter_preprocess(self) -> None:
        """Handle entering the PREPROCESS step."""
        if (self._auto_preview_action.isChecked()
                and self._current_series is not None
                and self._pipeline is not None):
            state = self._current_series.pipeline_state
            current_hash = self._config.preprocess_config_hash()
            if state.preprocessed and state.preprocess_config_hash == current_hash:
                # Already done and current — just display
                if self._current_image:
                    self._display_image(self._current_image)
            else:
                self._preprocess_group(self._current_series)
        elif self._current_image:
            self._display_image(self._current_image)

    def _handle_enter_track(self) -> None:
        """Handle entering the TRACK step."""
        if self._current_series is None or self._pipeline is None:
            if self._current_image:
                self._display_image(self._current_image)
            return

        state = self._current_series.pipeline_state

        # Ensure preprocessing is done first
        preprocess_hash = self._config.preprocess_config_hash()
        if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
            self._processing_label.setText("Preprocessing...")
            self._processing_label.show()
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            QApplication.processEvents()
            self._preprocess_group(self._current_series, force=True, hide_progress=False)
            QApplication.processEvents()

        # Check if tracking is already done and current
        if self._pipeline.is_tracking_current(self._current_series):
            # Tracking already done — just display the annotated image
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
            if self._current_image:
                self._display_image(self._current_image)
            return

        # Auto-run tracking if enabled
        if self._auto_preview_action.isChecked():
            self._processing_label.setText("Tracking...")
            self._processing_label.show()
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            QApplication.processEvents()
            self._on_track_roots()
            self._processing_label.hide()
            self._progress_bar.hide()
        else:
            self._force_unlock_ui()
            self._processing_label.hide()
            self._progress_bar.hide()
            if self._current_image:
                self._display_image(self._current_image)
    
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
            # Handle nested threshold settings
            elif key == "min_contour_area":
                self._config.threshold.min_contour_area = value
            elif key == "min_contour_length":
                self._config.threshold.min_contour_length = value
    
    def _on_apply_settings(self) -> None:
        """Handle Apply button - reprocess current group with new settings and apply mask."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return

        step = self._workflow_bar.get_current_step()

        # Handle mask application in Track step
        mask_changed = False
        if step == WorkflowStep.TRACK:
            # Check if mask changed
            mask_changed = self._current_series.has_pending_mask_changes()
            if mask_changed:
                # Apply mask
                working_mask = self._image_viewer.get_working_mask()
                self._current_series.user_mask = working_mask.copy() if working_mask is not None else None
                self._current_series.working_mask = working_mask.copy() if working_mask is not None else None

                # Save to disk
                try:
                    if working_mask is not None and np.any(working_mask > 0):
                        mask_io.save_mask(self._current_series, self._config)
                    else:
                        # All mask erased - delete file
                        mask_io.delete_mask(self._current_series, self._config)
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to save mask:\n{e}")

                # Clear tracking results (mask changed)
                self._current_series.clear_tracking_results()

        # Snapshot config hashes BEFORE updating
        old_preprocess_hash = self._config.preprocess_config_hash()
        old_tracking_hash = self._config.tracking_config_hash()

        # Update config from settings panel
        self._update_config_from_panel()

        # Determine what changed
        preprocess_changed = (old_preprocess_hash != self._config.preprocess_config_hash())
        tracking_changed = (old_tracking_hash != self._config.tracking_config_hash())

        if preprocess_changed:
            # Preprocessing params changed — invalidate everything and reprocess
            self._current_series.clear_preprocessing_results()
            self._preprocess_group(self._current_series, force=True)
        elif tracking_changed or mask_changed:
            # Tracking params or mask changed — invalidate tracking only and retrack
            if not mask_changed:  # Already cleared above if mask changed
                self._current_series.clear_tracking_results()
            if self._auto_preview_action.isChecked():
                self._on_track_roots()
            elif self._current_image:
                self._display_image(self._current_image)

    def _on_apply_all_settings(self) -> None:
        """Handle Apply All button - reprocess ALL groups with new settings."""
        if self._pipeline is None or not self._series_dict:
            return

        # Snapshot config hashes BEFORE updating
        old_preprocess_hash = self._config.preprocess_config_hash()
        old_tracking_hash = self._config.tracking_config_hash()

        # Update config from settings panel
        self._update_config_from_panel()

        preprocess_changed = (old_preprocess_hash != self._config.preprocess_config_hash())
        tracking_changed = (old_tracking_hash != self._config.tracking_config_hash())

        if preprocess_changed:
            for series in self._series_dict.values():
                series.clear_preprocessing_results()
            self._preprocess_all_groups(force=True)
        elif tracking_changed:
            for series in self._series_dict.values():
                series.clear_tracking_results()

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
            # Re-run preprocessing on current group
            self._current_series.clear_tracking_results()
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
    
    def _detect_barcodes_in_group(self, series: 'ImageSeries', continue_to_next_step: bool = False) -> None:
        """
        Detect barcodes for all images in a group with progress bar.

        Args:
            series: ImageSeries to detect barcodes in.
            continue_to_next_step: If True, automatically continue to preprocessing/tracking after detection.
        """
        if self._pipeline is None:
            return

        # Check if barcodes already detected for this group
        if all(img.barcode_detected for img in series.images):
            # Already detected - continue if requested
            if continue_to_next_step:
                self._continue_after_barcode_detection(series)
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
            current_image_count = 0
            for i, image_data in enumerate(series.images):
                if not image_data.barcode_detected:
                    self._update_progress_label("Detecting barcodes", start_time, current_image_count, total_images)
                    self._pipeline.detect_barcode_in_image(image_data)
                    current_image_count += 1
                else:
                    self._update_progress_label("Detecting barcodes", start_time, current_image_count, total_images)
                    current_image_count += 1

                self._progress_bar.setValue(current_image_count)
                QApplication.processEvents()

            # Update warning count and tree (refresh preserves selection)
            self._recalculate_warning_count()
            self._update_groups_progress()
            self._image_tree.refresh()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Barcode detection failed:\n{e}")
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
            return
        finally:
            pass  # Don't hide progress/unlock here if continuing to next step

        # Continue to next step if requested
        if continue_to_next_step:
            self._continue_after_barcode_detection(series)
        else:
            # Normal completion - hide progress and unlock
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)

            # Restore focus (tree was disabled during processing)
            self._image_tree.setFocus()

            if self._current_image:
                self._display_image(self._current_image)

    def _update_progress_label(self, action_name: str, start_time: float, current: int, total: int) -> None:
        """
        Update processing label with action name and ETA.

        Args:
            action_name: Name of action (e.g., "Detecting barcodes").
            start_time: Time when processing started.
            current: Number of items processed so far.
            total: Total number of items to process.
        """
        if current > 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                avg_time = elapsed / current
                remaining = total - current
                eta_seconds = int(avg_time * remaining)
                if eta_seconds >= 60:
                    eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
                else:
                    eta_text = f"{eta_seconds}s"
                self._processing_label.setText(f"{action_name}... ETA {eta_text}")
                return

        self._processing_label.setText(f"{action_name}...")

    def _continue_after_barcode_detection(self, series: 'ImageSeries') -> None:
        """
        Continue to next processing step after barcode detection completes.

        Called from _detect_barcodes_in_group when continue_to_next_step=True.

        Args:
            series: ImageSeries that was just processed.
        """
        step = self._workflow_bar.get_current_step()

        if step == WorkflowStep.PREPROCESS:
            # Continue to preprocessing
            self._preprocess_group(series, hide_progress=True)
            # Force unlock UI to clear any stacked wait cursors
            self._force_unlock_ui()
            # Restore focus and display
            self._image_tree.setFocus()
            if self._current_image:
                self._display_image(self._current_image)
        elif step == WorkflowStep.TRACK:
            # Continue with full tracking pipeline (will preprocess if needed, then track)
            # Check if preprocessing is needed
            preprocess_hash = self._config.preprocess_config_hash()
            state = series.pipeline_state
            if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
                self._processing_label.setText("Preprocessing...")
                QApplication.processEvents()
                self._preprocess_group(series, force=True, hide_progress=False)

            # Check if tracking is needed
            if not self._pipeline.is_tracking_current(series):
                self._processing_label.setText("Tracking...")
                self._progress_bar.setValue(0)
                QApplication.processEvents()

                try:
                    start_time = time.time()
                    # Define progress callback
                    def update_progress(current, total):
                        if total > 0:
                            percent = int((current / total) * 100)
                            self._progress_bar.setValue(percent)
                            self._update_progress_label("Tracking", start_time, current, total)
                        QApplication.processEvents()

                    stats = self._pipeline.track_and_analyze_series(
                        series,
                        progress_callback=update_progress
                    )

                    # Update pipeline state
                    series.pipeline_state.tracked = True
                    series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
                    series.pipeline_state.last_statistics = stats

                    self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")

            # All done - hide progress and unlock
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
            if self._current_image:
                self._display_image(self._current_image)
            self._image_tree.setFocus()
        else:
            # LOAD or EXPORT step - just finish normally
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)
            if self._current_image:
                self._display_image(self._current_image)
            self._image_tree.setFocus()
    
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
                        self._pipeline.detect_barcode_in_image(image_data)
                    
                    current_image_count += 1
                    self._update_progress_label("Detecting barcodes", start_time, current_image_count, total_images)
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
    
    def _preprocess_all_groups(self, force: bool = False) -> None:
        """Preprocess all unprocessed groups with progress bar."""
        if self._pipeline is None or not self._series_dict:
            return
        
        # Calculate total images across all unprocessed groups (or all if forced)
        current_hash = self._config.preprocess_config_hash()
        if force:
            unprocessed_groups = list(self._series_dict.values())
        else:
            unprocessed_groups = [
                s for s in self._series_dict.values()
                if not s.pipeline_state.preprocessed
                or s.pipeline_state.preprocess_config_hash != current_hash
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
                    self._update_progress_label("Preprocessing", start_time, current_image_count, total_images)
                    self._pipeline.preprocess_image(image_data)
                    self._progress_bar.setValue(current_image_count)
                    QApplication.processEvents()
                
                # Register the series
                self._pipeline.register_series(series)

                # Update pipeline state
                series.pipeline_state.preprocessed = True
                series.pipeline_state.preprocess_config_hash = self._config.preprocess_config_hash()
                series.pipeline_state.invalidate_from('track')

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
                if series.pipeline_state.preprocessed
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

    def _preprocess_group(self, series: 'ImageSeries', force: bool = False, hide_progress: bool = True) -> None:
        """
        Preprocess the given group (series).

        Args:
            series: ImageSeries to process.
            force: If True, re-process even if data exists.
            hide_progress: If True, hide progress bar after completion.
        """
        if self._pipeline is None:
            return

        # Check if already processed with current config
        current_hash = self._config.preprocess_config_hash()
        already_done = (
            series.pipeline_state.preprocessed
            and series.pipeline_state.preprocess_config_hash == current_hash
        )
        if not force and already_done:
            if self._current_image:
                self._display_image(self._current_image)
            return

        # Show progress bar in status bar
        total_images = len(series.images)
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.setText(f"Preprocessing...")
        self._processing_label.show()
        self._set_ui_locked(True)
        QApplication.processEvents()

        start_time = time.time()

        try:
            for i, image_data in enumerate(series.images):
                self._update_progress_label("Preprocessing", start_time, i + 1, total_images)
                self._pipeline.preprocess_image(image_data)
                self._progress_bar.setValue(i + 1)
                QApplication.processEvents()

            # Register the series
            self._pipeline.register_series(series)

            # Update pipeline state
            series.pipeline_state.preprocessed = True
            series.pipeline_state.preprocess_config_hash = self._config.preprocess_config_hash()
            # Invalidate tracking since preprocessing changed
            series.pipeline_state.invalidate_from('track')

            # Only mark complete if not already completed
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)

            if self._current_image:
                self._display_image(self._current_image)

            self._update_groups_progress()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{e}")
        finally:
            if hide_progress:
                self._progress_bar.hide()
                self._processing_label.hide()
                self._set_ui_locked(False)
            else:
                pass

        # Restore focus to tree
        self._image_tree.setFocus()

    def _auto_process_for_tracking(self, series: 'ImageSeries') -> None:
        """
        Auto-process a series for tracking (barcode detection → preprocessing → tracking).

        Called when selecting an image/group in TRACK step with auto-preview enabled.

        Args:
            series: ImageSeries to process.
        """
        if self._pipeline is None:
            return

        # Step 1: Ensure barcodes are detected
        if not all(img.barcode_detected for img in series.images):
            self._processing_label.setText("Detecting barcodes...")
            self._processing_label.show()
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            QApplication.processEvents()
            self._detect_barcodes_in_group(series)

        # Step 2: Ensure preprocessing is done
        preprocess_hash = self._config.preprocess_config_hash()
        state = series.pipeline_state
        if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
            self._processing_label.setText("Preprocessing...")
            self._processing_label.show()
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            QApplication.processEvents()
            self._preprocess_group(series, force=True, hide_progress=False)

        # Step 3: Check if tracking is needed
        if self._pipeline.is_tracking_current(series):
            # Already tracked with current config - just display
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
            if self._current_image:
                self._display_image(self._current_image)
            return

        # Step 4: Run tracking
        self._processing_label.setText("Tracking...")
        self._processing_label.show()
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        QApplication.processEvents()

        try:
            start_time = time.time()
            # Define progress callback
            def update_progress(current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    self._progress_bar.setValue(percent)
                    self._update_progress_label("Tracking", start_time, current, total)
                QApplication.processEvents()

            stats = self._pipeline.track_and_analyze_series(
                series,
                progress_callback=update_progress
            )

            # Update pipeline state
            series.pipeline_state.tracked = True
            series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
            series.pipeline_state.last_statistics = stats

            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)

            if self._current_image:
                self._display_image(self._current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
        finally:
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
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
            
    def _force_unlock_ui(self) -> None:
        """Force unlock UI and reset cursor stack."""
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
            
        self._workflow_bar.setEnabled(True)
        self._settings_panel.setEnabled(True)
        self._image_tree.setEnabled(True)
        self._fit_btn.setEnabled(True)
        self._zoom_in_btn.setEnabled(True)
        self._zoom_out_btn.setEnabled(True)
        
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
            if self._current_series:
                group_done = self._current_series.pipeline_state.preprocessed

            # Check if all groups are preprocessed
            all_done = False
            if self._series_dict:
                all_done = all(
                    series.pipeline_state.preprocessed
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
        
        # Prepare progress bar (always reset for tracking phase)
        self._processing_label.setText("Tracking...")
        self._processing_label.show()
        
        # Reset to percentage mode
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            start_time = time.time()
            # Define progress callback
            def update_progress(current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    self._progress_bar.setValue(percent)
                    self._update_progress_label("Tracking", start_time, current, total)
                QApplication.processEvents()
            
            stats = self._pipeline.track_and_analyze_series(
                self._current_series, 
                progress_callback=update_progress
            )
            
            # Update pipeline state
            self._current_series.pipeline_state.tracked = True
            self._current_series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
            self._current_series.pipeline_state.last_statistics = stats

            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)

            if self._current_image:
                self._display_image(self._current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
        finally:
            # Only hide if we showed it (logic slightly complex with auto-run, 
            # but usually hiding here is safe as auto-run will also hide or rely on this)
            # Actually, auto-run hides it at end of its block. 
            # If we hide here, auto-run's hide is redundant which is fine.
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
    
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
                # Collect statistics (use cached when available)
                all_stats = []
                for series in self._series_dict.values():
                    state = series.pipeline_state
                    if state.tracked and state.last_statistics:
                        all_stats.extend(state.last_statistics)
                    else:
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

    # ========== Masking Handlers ==========

    def _on_mask_tool_changed(self, tool_name: str, size: int) -> None:
        """Handle mask tool selection change."""
        # Convert string to MaskTool enum
        try:
            tool = MaskTool(tool_name)
        except ValueError:
            tool = MaskTool.NONE

        self._image_viewer.set_mask_tool(tool, size)

    def _on_mask_modified(self) -> None:
        """Handle mask modification - mark settings as dirty."""
        if self._current_series is not None:
            # Sync working mask from viewer to series
            working_mask = self._image_viewer.get_working_mask()
            self._current_series.working_mask = working_mask.copy() if working_mask is not None else None

            # Mark settings as dirty so Apply buttons enable
            self._settings_panel._mark_dirty()

    def _on_mask_erase_all(self) -> None:
        """Erase all masks (both working and applied)."""
        if self._current_series is None:
            return

        # Clear masks in series
        self._current_series.user_mask = None
        self._current_series.working_mask = None

        # Clear in viewer
        self._image_viewer.clear_all_masks()

        # Delete mask file
        try:
            mask_io.delete_mask(self._current_series, self._config)
        except Exception:
            pass

        # Mark as dirty so Apply button enables (to commit the erasure)
        self._settings_panel._mark_dirty()

        # Clear tracking results
        self._current_series.clear_tracking_results()

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
        """Handle Next button click — advance to next step."""
        current = self._workflow_bar.get_current_step()

        if current >= WorkflowStep.EXPORT:
            return

        if not self._series_dict:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return

        new_step = WorkflowStep(current + 1)
        self._workflow_bar.set_current_step(new_step)
        self._on_step_changed(new_step)
