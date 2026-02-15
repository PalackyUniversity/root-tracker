"""
Main application window for Root Tracker GUI.

Integrates all components into the main window layout.
"""

import os
import multiprocessing as mp
from multiprocessing import Manager
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QSplitter, QStatusBar, QMessageBox,
    QApplication, QFileDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence

from ..config import Config
from ..models import ImageData, ImageSeries
from ..pipeline import RootTrackingPipeline, preprocess_and_cache_worker, track_and_cache_worker
from ..io import mask_io, series_cache

from .workflow_bar import WorkflowBar, WorkflowStep
from .image_tree import ImageTree
from .image_viewer import ImageViewer
from .settings_panel import SettingsPanel
from .dialogs import LoadDialog
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
        self._processing: bool = False  # Reentrancy guard for processEvents
        self._cancel_requested = mp.Event()

        
        self.setWindowTitle("Root Tracker")
        self.setMinimumSize(1200, 800)
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        
        # Start on Load step with settings panel hidden
        self._settings_panel.hide()
        
        # Set initial button states (no images loaded yet)
        self._update_initial_ui_state()
        
        # Start maximized
        self.showMaximized()
    
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
        
        load_action = QAction("&Open Folder...", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self._on_load_images)
        file_menu.addAction(load_action)
        
        self._clear_cache_action = QAction("&Clear Cache", self)
        self._clear_cache_action.triggered.connect(self._on_clear_cache)
        file_menu.addAction(self._clear_cache_action)
        
        file_menu.addSeparator()
        
        self._export_action = QAction("&Export Results...", self)
        self._export_action.setShortcut(QKeySequence("Ctrl+E"))
        self._export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(self._export_action)
        
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

        view_menu.addSeparator()

        self._auto_preview_action = QAction("&Auto Preview", self)
        self._auto_preview_action.setCheckable(True)
        self._auto_preview_action.setChecked(True)  # Default on
        self._auto_preview_action.setToolTip("Automatically preview preprocessing when group is selected")
        self._auto_preview_action.toggled.connect(self._on_auto_preview_toggled)
        view_menu.addAction(self._auto_preview_action)
        
        # Process menu
        process_menu = menubar.addMenu("&Process")

        self._cancel_prediction_action = QAction("&Cancel Prediction", self)
        self._cancel_prediction_action.setShortcut(QKeySequence("Ctrl+."))
        self._cancel_prediction_action.triggered.connect(self._on_cancel_prediction)
        self._cancel_prediction_action.setEnabled(False) # Default disabled
        process_menu.addAction(self._cancel_prediction_action)
        
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
        self._image_tree.set_aside_requested.connect(self._on_set_aside_requested)
        self._image_tree.unset_aside_requested.connect(self._on_unset_aside_requested)
        self._image_tree.delete_requested.connect(self._on_delete_requested)
        
        # Settings panel
        self._settings_panel.apply_requested.connect(self._on_apply_settings)
        self._settings_panel.apply_all_requested.connect(self._on_apply_all_settings)
        self._settings_panel.redetect_requested.connect(self._on_redetect_plants)

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
            
            self._reload_images()

    def _reload_images(self) -> None:
        """Reload images from the current input folder."""
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
                # Clear UI if reload failed to find images
                self._image_tree.clear()
                self._update_initial_ui_state()
                return
            
            self._image_tree.set_series(self._series_dict)
            
            # Select first image
            self._image_tree.select_first_image()
            
            # Mark step as complete but DON'T auto-advance
            self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
            
            # Update cache action state
            self._update_cache_action_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load images:\n{e}")

    

    def _on_image_selected(self, image_data: ImageData) -> None:
        """Handle image selection in tree."""
        step = self._workflow_bar.get_current_step()
        if self._processing:
            return
        self._current_image = image_data
        self._current_series = self._image_tree.get_selected_series()

        # Reload from cache if arrays were freed
        if self._current_series is not None:
            self._ensure_series_loaded(self._current_series)

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
        if self._processing:
            return
        # Reset settings for new group (discard unsaved changes)
        self._settings_panel.reset_for_group()

        # Deselect mask tools when switching groups
        self._settings_panel.deselect_mask_tools()

        self._current_series = series
        self._current_image = series.images[0] if series.images else None

        # Reload from cache if arrays were freed
        self._ensure_series_loaded(series)

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
        # Reload from cache if arrays were freed
        if self._current_series is not None:
            self._ensure_series_loaded(self._current_series)

        # Determine which image to show based on workflow step
        step = self._workflow_bar.get_current_step()
        
        if step == WorkflowStep.LOAD:
            # Show original image with barcode overlay
            image = cv2.imread(image_data.path)
            
            # No rotation in LOAD step - show raw image
            self._image_viewer.set_image(image)
            
            # Draw barcode overlay if detected
            if image_data.barcode_rect is not None:
                label = image_data.barcode_read
                if image_data.barcode_mismatch:
                    label = f"X {label} (expected: {image_data.barcode})"
                
                self._image_viewer.set_barcode_overlay(
                    image_data.barcode_rect, 
                    label, 
                    image_data.barcode_mismatch
                )
            elif image_data.barcode_not_found and image_data.barcode_detected:
                # Show generic warning overlay
                self._image_viewer.set_barcode_overlay(
                    None,
                    "",
                    is_mismatch=False,
                    is_missing=True
                )
            else:
                self._image_viewer.clear_barcode_overlay()
            
            self._image_viewer.clear_centroids()
        elif step == WorkflowStep.PREPROCESS:
            # Show processed image (RGB) if available, else original (rotated + cropped)
            if image_data.image is not None:
                image = image_data.image
                # Don't show centroids in Preprocess step (requested)
                self._image_viewer.clear_centroids()
                self._image_viewer.clear_barcode_overlay()
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
                self._image_viewer.clear_barcode_overlay()
        else:
            # TRACK: prefer annotated image, fall back to preprocessed
            if image_data.image_annotated is not None:
                image = image_data.image_annotated
            elif image_data.image is not None:
                image = image_data.image
            elif image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
            self._image_viewer.clear_centroids()
            self._image_viewer.clear_barcode_overlay()

        if image is not None:
            self._image_viewer.set_image(image)

            # Set mask data if in TRACK step
            if step == WorkflowStep.TRACK and self._current_series is not None:
                # Erase mask if dimensions don't match (preprocessing changed the crop)
                user_mask = self._current_series.user_mask
                if user_mask is not None and user_mask.shape[:2] != image.shape[:2]:
                    self._current_series.user_mask = None
                    self._current_series.working_mask = None
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
            self._settings_panel.show()
            self._settings_panel.set_step(step)
            
        # Update tree filtering: show aside items ONLY in LOAD step
        self._image_tree.set_filter_aside(step != WorkflowStep.LOAD)

        # Step-specific auto-processing
        if step == WorkflowStep.PREPROCESS:
            self._handle_enter_preprocess()
        elif step == WorkflowStep.TRACK:
            self._handle_enter_track()
        else:
            # LOAD: just refresh display
            if self._current_image:
                self._display_image(self._current_image)

        self._update_process_button_states()

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

        self._processing = True  # Guard before any processEvents() calls

        state = self._current_series.pipeline_state

        # Ensure preprocessing is done first
        preprocess_hash = self._config.preprocess_config_hash()
        if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
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

            # Re-register the series with updated centroid positions
            self._pipeline.register_series(self._current_series)

            # Save to cache (preprocessing results updated with new centroids)
            series_cache.save_series(self._current_series, self._config)

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
    
    def _on_set_aside_requested(self, item: ImageData | ImageSeries) -> None:
        """Handle request to set an item aside (move to aside/ folder)."""
        base_dir = self._config.data.input
        aside_base = os.path.join(base_dir, "aside")
        
        # Collect items to move
        items_to_move = []
        if isinstance(item, ImageData):
            items_to_move.append(item)
        else:
            # It's a series - move all images
            items_to_move.extend(item.images)
            
        if not items_to_move:
            return

        try:
            for image_data in items_to_move:
                src_path = image_data.path
                
                try:
                    rel_path = os.path.relpath(src_path, base_dir)
                except ValueError:
                     # If file is not in base_dir, maybe it's already in aside or elsewhere?
                     # But we allow moving FROM base TO aside.
                     continue
                     
                target_path = os.path.join(aside_base, rel_path)
                
                # Ensure target directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                shutil.move(src_path, target_path)
                image_data.path = target_path
            
            # If it was a series, we might want to update the group identifier if it contained path info?
            # But group is logical, so we leave it.
            # Actually, if we move ALL images, the series is now effectively "aside".
            
            self._image_tree.refresh()
            
            # Clear selection if needed
            self._check_selection_visibility()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move item aside:\n{e}")

    def _check_selection_visibility(self) -> None:
        """Clear selection if the selected item is no longer visible in current step."""
        step = self._workflow_bar.get_current_step()
        selected = self._image_tree.get_selected_data()
        
        # If we have a selection, checks if it should be hidden
        if selected and step != WorkflowStep.LOAD:
            # Check if selected item is now aside
            # If selected is series, checking it is_set_aside
            is_aside = False
            if isinstance(selected, ImageData):
                is_aside = selected.is_set_aside
            elif isinstance(selected, ImageSeries):
                is_aside = selected.is_set_aside
            
            if is_aside:
                self._current_image = None
                self._current_series = None
                # Refesh triggers selection restore, but if it's hidden it won't be selected?
                # ImageTree.refresh tries to restore.
                # If we want to force clear:
                self._image_tree.clear_selection()
                self._image_viewer.clear()

    def _on_unset_aside_requested(self, item: ImageData | ImageSeries) -> None:
        """Handle request to restore an item from aside."""
        base_dir = self._config.data.input
        aside_base = os.path.join(base_dir, "aside")
        
        items_to_move = []
        if isinstance(item, ImageData):
            items_to_move.append(item)
        else:
            items_to_move.extend(item.images)
            
        try:
            for image_data in items_to_move:
                src_path = image_data.path
                
                try:
                    rel_path = os.path.relpath(src_path, aside_base)
                except ValueError:
                    # Not in aside folder?
                    continue
                    
                target_path = os.path.join(base_dir, rel_path)
                
                # Ensure target directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                shutil.move(src_path, target_path)
                image_data.path = target_path

            self._image_tree.refresh()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to restore item:\n{e}")

    def _on_delete_requested(self, item: ImageData | ImageSeries) -> None:
        """Handle request to delete an item."""
        if isinstance(item, ImageData):
            name = item.filename
            items_to_delete = [item]
        else:
            name = f"Group {item.barcode}"
            items_to_delete = item.images[:]  # Copy list
            
        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to permanently delete {name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                for image_data in items_to_delete:
                    if os.path.exists(image_data.path):
                        os.remove(image_data.path)
                    
                    # update data structures
                    # If dealing with single image delete
                    if isinstance(item, ImageData):
                        # Find series
                        for series in self._series_dict.values():
                            if image_data in series.images:
                                series.images.remove(image_data)
                                break
                
                # If deleted whole series
                if isinstance(item, ImageSeries):
                    # Remove from dict
                    keys_to_remove = [k for k, v in self._series_dict.items() if v is item]
                    for k in keys_to_remove:
                        del self._series_dict[k]
                
                self._image_tree.refresh()
                self._current_image = None
                self._current_series = None
                self._image_viewer.clear()
                        
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete item:\n{e}")
    
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
        self._processing_label.show()

        start_time = time.time()

        try:
            current_image_count = 0
            for i, image_data in enumerate(series.images):
                if not image_data.barcode_detected:
                    self._update_progress_label(start_time, current_image_count, total_images)
                    self._pipeline.detect_barcode_in_image(image_data)
                    current_image_count += 1
                else:
                    self._update_progress_label(start_time, current_image_count, total_images)
                    current_image_count += 1

                self._progress_bar.setValue(current_image_count)
                QApplication.processEvents()

            # Update tree to show warning icons (refresh preserves selection)
            self._image_tree.refresh()

            # Save barcode detection results to cache
            series_cache.save_series(series, self._config)

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

    def _update_progress_label(self, start_time: float, current: int, total: int) -> None:
        """
        Update processing label with smoothed ETA.

        Uses a coasting algorithm to count down smoothly between updates,
        avoiding jumps due to parallel processing batches.

        Args:
            start_time: Time when processing started.
            current: Number of items processed so far.
            total: Total number of items to process.
        """
        if current <= 0 or current >= total:
            if current >= total:
                self._processing_label.setText("Finishing up...")
            else:
                self._processing_label.setText("Starting...")
            self._processing_label.show()
            return

        elapsed = time.time() - start_time
        
        # Calculate raw ETA based on average speed
        avg_time = elapsed / current
        remaining_items = total - current
        raw_eta = avg_time * remaining_items
        
        # Smooth the ETA
        # If we have a previous ETA estimate, don't jump wildly.
        # instead, check if the raw ETA is significantly different.
        
        current_time = time.time()
        
        if not hasattr(self, '_last_eta_update_time'):
            # First valid estimate
            self._last_eta_estimate = raw_eta
            self._last_eta_update_time = current_time
        else:
            time_since_last = current_time - self._last_eta_update_time
            
            # Coasting: ideally, the new ETA should be old_eta - time_since_last
            projected_eta = max(0, self._last_eta_estimate - time_since_last)
            
            # Deviation check: is the raw_eta very different from projected?
            # If within threshold (e.g. 5 seconds or 20%), stick to projected to appear smooth.
            diff = abs(raw_eta - projected_eta)
            threshold = max(5.0, projected_eta * 0.2)
            
            if diff < threshold:
                # Coasting feels better
                self._last_eta_estimate = projected_eta
            else:
                # Significant change (speed up or slow down)
                # Blend it to avoid instant jump
                self._last_eta_estimate = (projected_eta * 0.7) + (raw_eta * 0.3)
            
            self._last_eta_update_time = current_time

        eta_seconds = int(self._last_eta_estimate)
        
        if eta_seconds >= 60:
            eta_text = f"{eta_seconds // 60}m {eta_seconds % 60}s"
        else:
            eta_text = f"{eta_seconds}s"
            
        self._processing_label.setText(f"ETA: {eta_text}")
        self._processing_label.show()

    def _continue_after_barcode_detection(self, series: 'ImageSeries') -> None:
        """
        Continue to next processing step after barcode detection completes.

        Called from _detect_barcodes_in_group when continue_to_next_step=True.

        Args:
            series: ImageSeries that was just processed.
        """
        self._processing = True  # Guard before any processEvents() calls
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
                QApplication.processEvents()
                self._preprocess_group(series, force=True, hide_progress=False)

            # Check if tracking is needed
            if not self._pipeline.is_tracking_current(series):
                self._progress_bar.setValue(0)
                QApplication.processEvents()

                try:
                    start_time = time.time()
                    # Define progress callback
                    def update_progress(current, total):
                        if total > 0:
                            percent = int((current / total) * 100)
                            self._progress_bar.setValue(percent)
                            self._update_progress_label(start_time, current, total)
                        QApplication.processEvents()

                    stats = self._pipeline.track_and_analyze_series(
                        series,
                        progress_callback=update_progress
                    )

                    # Update pipeline state
                    series.pipeline_state.tracked = True
                    series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
                    series.pipeline_state.last_statistics = stats

                    # Save to cache
                    series_cache.save_series(series, self._config)

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
            # LOAD step - just finish normally
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
        self._processing_label.show()

        current_image_count = 0
        start_time = time.time()

        try:
            for series in undetected_groups:
                for image_data in series.images:
                    if not image_data.barcode_detected:
                        self._pipeline.detect_barcode_in_image(image_data)

                    current_image_count += 1
                    self._update_progress_label(start_time, current_image_count, total_images)
                    self._progress_bar.setValue(current_image_count)
                    QApplication.processEvents()
                
                # Save this series to cache after all its barcodes are detected
                series_cache.save_series(series, self._config)
                
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
        """Preprocess all unprocessed groups in parallel using all CPU cores.

        Submits each group to a ProcessPoolExecutor. Workers save results
        to disk cache. Progress bar tracks completed groups.
        """
        if self._pipeline is None or not self._series_dict:
            return

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

        total_groups = len(unprocessed_groups)

        # Lock UI
        self._set_ui_locked(True)

        # Count total images for progress bar
        total_images = sum(len(s.images) for s in unprocessed_groups)

        # Progress bar tracks completed images
        self._progress_bar.setRange(0, total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        start_time = time.time()
        # Reset ETA smoother
        if hasattr(self, '_last_eta_update_time'):
            del self._last_eta_update_time

        completed_images = 0
        self._cancel_requested.clear()

        try:
            # Create a manager and queue for progress updates
            manager = Manager()
            progress_queue = manager.Queue()

            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as executor:
                # Pass queue to workers
                futures = {
                    executor.submit(preprocess_and_cache_worker, (self._config, s, progress_queue)): s
                    for s in unprocessed_groups
                }

                while futures:
                    # Check cancellation
                    if self._cancel_requested.is_set():
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                    # Check for progress updates from queue
                    while not progress_queue.empty():
                        try:
                            _ = progress_queue.get_nowait()
                            completed_images += 1
                            self._progress_bar.setValue(completed_images)
                            self._update_progress_label(start_time, completed_images, total_images)
                        except Exception:
                            pass
                    
                    QApplication.processEvents()

                    done = [f for f in futures if f.done()]
                    for f in done:
                        series = futures.pop(f)
                        try:
                            f.result()
                            # Update pipeline state in main process
                            series.pipeline_state.preprocessed = True
                            series.pipeline_state.preprocess_config_hash = current_hash
                            series.pipeline_state.invalidate_from('track')
                        except Exception as e:
                            print(f"Error preprocessing {series.group}: {e}")
                    
                    # Update ETA even if no progress, to handle coasting
                    if completed_images > 0:
                        self._update_progress_label(start_time, completed_images, total_images)

                    if futures:
                        time.sleep(0.05)

            # Reload current series from cache if it was in the batch
            if self._current_series in unprocessed_groups:
                self._ensure_series_loaded(self._current_series)

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
        self._processing_label.show()
        self._set_ui_locked(True)
        QApplication.processEvents()

        start_time = time.time()

        try:
            for i, image_data in enumerate(series.images):
                self._update_progress_label(start_time, i + 1, total_images)
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

            # Save to cache
            series_cache.save_series(series, self._config)

            # Only mark complete if not already completed
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)

            if self._current_image:
                self._display_image(self._current_image)

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

        self._processing = True  # Guard before any processEvents() calls

        # Step 1: Ensure barcodes are detected
        if not all(img.barcode_detected for img in series.images):
            self._processing_label.show()
            self._progress_bar.setValue(0)
            self._progress_bar.show()
            QApplication.processEvents()
            self._detect_barcodes_in_group(series)

        # Step 2: Ensure preprocessing is done
        preprocess_hash = self._config.preprocess_config_hash()
        state = series.pipeline_state
        if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
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
                    self._update_progress_label(start_time, current, total)
                QApplication.processEvents()

            stats = self._pipeline.track_and_analyze_series(
                series,
                progress_callback=update_progress
            )

            # Update pipeline state
            series.pipeline_state.tracked = True
            series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
            series.pipeline_state.last_statistics = stats

            # Save to cache
            series_cache.save_series(series, self._config)

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
        self._processing = locked
        
        # Disable main interactive elements
        self._image_tree.setEnabled(not locked)
        self._settings_panel.setEnabled(not locked)
        self._workflow_bar.setEnabled(not locked)
        
        # Buttons
        self._next_step_btn.setEnabled(not locked)
        
        # Zoom controls: always enabled when images are loaded (independent of lock state)
        has_images = bool(self._series_dict)
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)

        # Menu items
        if hasattr(self, '_export_action'):
            self._export_action.setEnabled(not locked)
        if hasattr(self, '_cancel_prediction_action'):
            self._cancel_prediction_action.setEnabled(locked)
        
        # Cursor
        if locked:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _on_clear_cache(self) -> None:
        """Clear the cache directory for the current folder."""
        if not self._config.data.input:
            QMessageBox.warning(self, "No Folder Open", "Please open a folder first.")
            return

        cache_dir = os.path.join(self._config.data.input, ".root_tracker_cache")
        if not os.path.exists(cache_dir):
            QMessageBox.information(self, "Cache Empty", "No cache directory found.")
            return

        reply = QMessageBox.question(
            self, "Clear Cache",
            "Are you sure you want to delete all cached data?\n"
            "This will remove all preprocessing and tracking results.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(cache_dir)
                # Reload to reflect cleared state
                self._reload_images()
                QMessageBox.information(self, "Success", "Cache cleared and reloaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache:\n{e}")
        
        self._update_cache_action_state()

    def _update_cache_action_state(self) -> None:
        """Enable/disable Clear Cache action based on cache existence."""
        if not self._config.data.input:
            self._clear_cache_action.setEnabled(False)
            return
            
        cache_dir = os.path.join(self._config.data.input, ".root_tracker_cache")
        has_cache = os.path.exists(cache_dir)
        self._clear_cache_action.setEnabled(has_cache)

    def _on_cancel_prediction(self) -> None:
        """Request cancellation of current processing."""
        self._cancel_requested.set()
            
    def _force_unlock_ui(self) -> None:
        """Force unlock UI and reset cursor stack."""
        self._processing = False
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
            
        self._workflow_bar.setEnabled(True)
        self._settings_panel.setEnabled(True)
        self._image_tree.setEnabled(True)
        
        # Zoom controls: enabled only if images are loaded
        has_images = bool(self._series_dict)
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)
        
        self._update_process_button_states()
    
    def _ensure_series_loaded(self, series: 'ImageSeries') -> None:
        """Reload series arrays from disk cache if they were freed."""
        if series is None:
            return
        sample = series.images[0] if series.images else None
        if sample and sample.image is None and series.pipeline_state.preprocessed:
            series_cache.load_series(series, self._config)

    @staticmethod
    def _free_series_arrays(series: 'ImageSeries') -> None:
        """Free heavy image arrays from a series to reduce RAM usage."""
        for img in series.images:
            img.image = None
            img.process = None
            img.canny = None
            img.diff = None
            img.image_annotated = None
            img.colored_samples = {}

    def _update_process_button_states(self) -> None:
        """Update UI states based on current step and processing status."""
        step = self._workflow_bar.get_current_step()
        
        # Next/Export button: on Track step becomes green "Export" button
        has_images = bool(self._series_dict)
        if step == WorkflowStep.TRACK:
            self._next_step_btn.setText("Export")
            self._next_step_btn.setStyleSheet("""
                QPushButton {
                    background-color: #5cb85c;
                    color: white;
                    font-weight: bold;
                    padding: 2px 12px;
                }
                QPushButton:hover {
                    background-color: #449d44;
                }
                QPushButton:disabled {
                    background-color: #88c888;
                    color: #ccc;
                }
            """)
            self._next_step_btn.setToolTip("Export results to CSV")
        else:
            self._next_step_btn.setText("Next")
            self._next_step_btn.setStyleSheet("")
            self._next_step_btn.setToolTip("Go to next step")
        self._next_step_btn.setEnabled(has_images)

        # Zoom controls: enabled only when images are loaded
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)
        self._zoom_label.setEnabled(has_images)
        
        # Menu items state
        if hasattr(self, '_export_action'):
            self._export_action.setEnabled(has_images)
            
        # Ensure cancel is disabled when not processing (safety check)
        if hasattr(self, '_cancel_prediction_action') and not self._processing:
            self._cancel_prediction_action.setEnabled(False)
            
        # Update cache action state as well (processing might have created cache)
        self._update_cache_action_state()
    
    def _update_initial_ui_state(self) -> None:
        """Set initial UI state when no images are loaded."""
        # Disable zoom controls
        self._fit_btn.setEnabled(False)
        self._zoom_in_btn.setEnabled(False)
        self._zoom_out_btn.setEnabled(False)
        self._zoom_label.setEnabled(False)
        
        # Disable next button
        self._next_step_btn.setEnabled(False)
        
        # Disable menu items
        if hasattr(self, '_clear_cache_action'):
            self._clear_cache_action.setEnabled(False)
        if hasattr(self, '_export_action'):
            self._export_action.setEnabled(False)
        if hasattr(self, '_cancel_prediction_action'):
            self._cancel_prediction_action.setEnabled(False)
    
    def _track_all_groups(self) -> None:
        """Track all untracked groups in parallel using all CPU cores.

        Saves caches and frees arrays before submitting, so workers
        reload from disk. Progress bar tracks completed groups.
        """
        if self._pipeline is None or not self._series_dict:
            return

        untracked = [
            s for s in self._series_dict.values()
            if not self._pipeline.is_tracking_current(s)
        ]
        if not untracked:
            return

        total_groups = len(untracked)
        total_images = sum(len(s.images) for s in untracked)

        # Prepare: save cache + free arrays so pickling is lightweight
        for series in untracked:
            if series.images and series.images[0].image is not None:
                series_cache.save_series(series, self._config)
            self._free_series_arrays(series)

        self._set_ui_locked(True)
        self._progress_bar.setRange(0, total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        start_time = time.time()
        # Reset ETA smoother
        if hasattr(self, '_last_eta_update_time'):
            del self._last_eta_update_time

        completed_images = 0
        self._cancel_requested.clear()

        try:
            # Create a manager and queue for progress updates
            manager = Manager()
            progress_queue = manager.Queue()

            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as executor:
                # Pass queue to workers
                futures = {
                    executor.submit(track_and_cache_worker, (self._config, s, progress_queue)): s
                    for s in untracked
                }

                while futures:
                    # Check cancellation
                    if self._cancel_requested.is_set():
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                    # Check for progress updates from queue
                    while not progress_queue.empty():
                        try:
                            _ = progress_queue.get_nowait()
                            completed_images += 1
                            self._progress_bar.setValue(completed_images)
                            self._update_progress_label(start_time, completed_images, total_images)
                        except Exception:
                            pass
                    
                    QApplication.processEvents()

                    done = [f for f in futures if f.done()]
                    for f in done:
                        series = futures.pop(f)
                        try:
                            group, stats_dicts, state_dict = f.result()
                            # Update pipeline state in main process
                            series.pipeline_state.preprocessed = state_dict['preprocessed']
                            series.pipeline_state.preprocess_config_hash = state_dict['preprocess_config_hash']
                            series.pipeline_state.tracked = state_dict['tracked']
                            series.pipeline_state.tracking_config_hash = state_dict['tracking_config_hash']
                            series.pipeline_state.last_statistics = stats_dicts
                        except Exception as e:
                            print(f"Error tracking {series.group}: {e}")

                    # Update ETA even if no progress, to handle coasting
                    if completed_images > 0:
                        self._update_progress_label(start_time, completed_images, total_images)

                    if futures:
                        time.sleep(0.05)

            # Reload current series from cache for display
            if self._current_series is not None:
                self._ensure_series_loaded(self._current_series)

            if not self._workflow_bar.is_step_completed(WorkflowStep.TRACK):
                self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)

            if self._current_image:
                self._display_image(self._current_image)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
        finally:
            self._progress_bar.hide()
            self._processing_label.hide()
            self._set_ui_locked(False)

        self._image_tree.setFocus()

    def _on_track_roots(self) -> None:
        """Run root tracking on current group."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please preprocess images first.")
            return
        
        # Prepare progress bar (always reset for tracking phase)
        self._processing_label.show()

        # Reset to percentage mode
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        self._processing = True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            start_time = time.time()
            # Define progress callback
            def update_progress(current, total):
                if total > 0:
                    percent = int((current / total) * 100)
                    self._progress_bar.setValue(percent)
                    self._update_progress_label(start_time, current, total)
                QApplication.processEvents()

            stats = self._pipeline.track_and_analyze_series(
                self._current_series,
                progress_callback=update_progress
            )

            # Update pipeline state
            self._current_series.pipeline_state.tracked = True
            self._current_series.pipeline_state.tracking_config_hash = self._config.tracking_config_hash()
            self._current_series.pipeline_state.last_statistics = stats

            # Save to cache
            series_cache.save_series(self._current_series, self._config)

            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)

            if self._current_image:
                self._display_image(self._current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{e}")
        finally:
            self._processing_label.hide()
            self._progress_bar.hide()
            self._force_unlock_ui()
    
    def _on_export_results(self) -> None:
        """Handle export results action."""
        if self._pipeline is None or not self._series_dict:
            QMessageBox.warning(self, "No Data", "No data loaded to export.")
            return

        # Default filename
        default_name = "statistics.csv"
        default_path = os.path.join(self._config.data.input, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            default_path,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return

        try:
            # Export statistics
            stats_df = self._pipeline.export_statistics(self._series_dict)
            stats_df.to_csv(file_path, index=False)
            
            QMessageBox.information(
                self, 
                "Export Complete",
                f"Results exported to:\n{file_path}"
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

        # Save to cache (tracking results are now cleared)
        series_cache.save_series(self._current_series, self._config)

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
        """Handle Next/Export button click."""
        current = self._workflow_bar.get_current_step()

        if not self._series_dict:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return

        if current == WorkflowStep.TRACK:
            # On Track step, the button is "Export" — track all then export
            self._track_all_and_export()
            return

        if current >= WorkflowStep.TRACK:
            return

        new_step = WorkflowStep(current + 1)
        self._workflow_bar.set_current_step(new_step)
        self._on_step_changed(new_step)

    def _track_all_and_export(self) -> None:
        """Show export options first, then track all and export."""
        if self._pipeline is None or not self._series_dict:
            return

        # Ask export path before any computation
        default_name = "statistics.csv"
        default_path = os.path.join(self._config.data.input, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            default_path,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        # Track all untracked groups
        untracked = [
            s for s in self._series_dict.values()
            if not self._pipeline.is_tracking_current(s)
        ]
        if untracked:
            self._track_all_groups()

        # Export with the already-chosen options
        self._processing_label.setText("Exporting...")
        self._processing_label.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            # Export statistics
            stats_df = self._pipeline.export_statistics(self._series_dict)
            stats_df.to_csv(file_path, index=False)
            
            QMessageBox.information(
                self, 
                "Export Complete",
                f"Results exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
        finally:
            self._processing_label.hide()
            QApplication.restoreOverrideCursor()
