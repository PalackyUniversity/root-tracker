"""
Main application window for Root Tracker GUI.

Integrates all components into the main window layout.
"""

import os
from copy import deepcopy
import math
import multiprocessing as mp
from multiprocessing import Manager
import shutil
import time
import threading
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QSplitter, QStatusBar, QMessageBox,
    QApplication, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, QSettings, QThread, Signal, QStandardPaths
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
from .menus import MenuBarPopup
from .operation_progress import OperationProgress
from .theme import button_stylesheet


class ProcessingState(Enum):
    """Enum representing the current processing state of the application."""
    IDLE = "idle"
    BARCODE_DETECTION = "barcode_detection"
    PREPROCESSING = "preprocessing"
    TRACKING = "tracking"
    BATCH_PREPROCESS = "batch_preprocess"
    BATCH_TRACK = "batch_track"
    RSML_EXPORT = "rsml_export"


class ProcessingContext:
    """
    Context manager for processing operations.
    
    Ensures UI is locked during processing and properly unlocked on completion,
    even if exceptions occur. Handles worker cleanup and state transitions.
    """
    
    def __init__(self, main_window, state: ProcessingState, series=None):
        self.main_window = main_window
        self.state = state
        self.series = series
        self.next_operation = None  # Set by operations to chain processing
        self.was_cancelled = False
        self._worker_started = False
        
    def __enter__(self):
        """Lock UI and set processing state."""
        self.main_window._state = self.state
        self.main_window._context = self
        
        # Save current step for potential cancel restoration
        if self.main_window._step_before_processing is None:
            self.main_window._step_before_processing = self.main_window._workflow_bar.get_current_step()
        
        # Lock UI
        self.main_window._lock_ui()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unlock UI and clean up, even if exception occurred."""
        # Cancel ongoing operations if exception occurred
        if exc_type is not None:
            self.was_cancelled = True
            if self.main_window._worker is not None and self.main_window._worker.isRunning():
                self.main_window._worker.cancel()
            if self.main_window._executor is not None:
                self.main_window._executor.shutdown(wait=False, cancel_futures=True)
                self.main_window._executor = None
        
        # Clear continuation flags if cancelled to prevent restart
        if self.was_cancelled:
            if hasattr(self.main_window, '_auto_process_pending_preprocess'):
                self.main_window._auto_process_pending_preprocess = False
            if hasattr(self.main_window, '_auto_process_pending_tracking'):
                self.main_window._auto_process_pending_tracking = False
        
        # Clear worker reference
        if self.main_window._worker is not None:
            self.main_window._worker = None
        
        # Clear executor reference
        if self.main_window._executor is not None:
            self.main_window._executor = None
        
        continuing = bool(not self.was_cancelled and self.next_operation)
        self.main_window._finish_operation_progress(not self.was_cancelled, continuing)
        if continuing:
            # Keep one visible bar and a locked UI across worker transitions.
            self.main_window._context = None
            self.next_operation()
            return False

        # Hide progress indicators
        self.main_window._progress_bar.hide()
        self.main_window._processing_label.hide()
        
        # Clear cancel flag
        self.main_window._cancel_requested.clear()
        
        # Reset state
        self.main_window._state = ProcessingState.IDLE
        self.main_window._context = None
        
        # Unlock UI
        self.main_window._unlock_ui(restore_step=self.was_cancelled)
        
        # Don't steal focus from an ongoing mask stroke.
        if not self.main_window._mask_refresh_active:
            self.main_window._image_tree.setFocus()
        
        # Don't suppress exceptions
        return False


class ProcessWorker(QThread):
    """Worker thread for processing operations to keep UI responsive."""
    
    image_ready = Signal(object)
    phase = Signal(str)
    progress = Signal(int, int)  # percentage within this operation
    finished = Signal(bool, str)  # success, error_msg
    
    def __init__(self, pipeline, series, config, operation='preprocess'):
        super().__init__()
        self.pipeline = pipeline
        self.series = series
        self.config = config
        self.operation = operation
        self.persist_mask = False
        self._cancelled = False
    
    def cancel(self):
        """Request cancellation of the operation."""
        self._cancelled = True
    
    def run(self):
        """Execute the processing operation in a background thread."""
        try:
            if self.operation == 'preprocess':
                self._run_preprocess()
            elif self.operation == 'track':
                self._run_track()
            elif self.operation == 'barcode':
                self._run_barcode_detection()
            
            if not self._cancelled:
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, "Cancelled")
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def _run_preprocess(self):
        """Run preprocessing on the series."""
        if self._cancelled:
            return
        self.series.pipeline_state.invalidate_from('preprocess')
        def progress_callback(current, total):
            if self._cancelled:
                raise InterruptedError("Cancelled")
            self.progress.emit(int(75 * current / max(1, total)), 100)

        self.phase.emit("Preparing images")
        self.pipeline.preprocess_series(self.series, progress_callback=progress_callback)

        if not self._cancelled:
            # Register the series
            self.phase.emit("Aligning images")
            self.progress.emit(75, 100)
            self.pipeline.register_series(self.series)
            self.progress.emit(92, 100)
            
            # Update pipeline state
            self.series.pipeline_state.preprocessed = True
            self.series.pipeline_state.preprocess_config_hash = self.config.preprocess_config_hash()
            self.series.pipeline_state.invalidate_from('track')
            
            # Save to cache
            from ..io import series_cache
            self.phase.emit("Saving results")
            series_cache.save_series(self.series, self.config)
    
    def _run_track(self):
        """Run tracking on the series."""
        if self.persist_mask:
            mask = self.series.user_mask
            if mask is not None and np.any(mask):
                mask_io.save_mask(self.series, self.config)
            else:
                mask_io.delete_mask(self.series, self.config)

        if self._cancelled:
            return
        self.series.pipeline_state.invalidate_from('track')

        def progress_callback(current, total):
            if self._cancelled:
                raise InterruptedError("Cancelled")
            self.progress.emit(int(85 * current / max(1, total)), 100)
        
        options = {}
        if self.persist_mask:
            options = dict(image_callback=self.image_ready.emit, save_images=False)
        stats = self.pipeline.track_and_analyze_series(
            self.series, progress_callback=progress_callback, **options
        )
        
        if not self._cancelled:
            # Update pipeline state
            self.series.pipeline_state.tracked = True
            self.series.pipeline_state.tracking_config_hash = self.config.tracking_config_hash()
            self.series.pipeline_state.last_statistics = stats
            
            # Keep image encoding and cache writes after the calculated preview.
            self.phase.emit("Saving results")
            if self.persist_mask:
                for image in self.series.images:
                    if self._cancelled:
                        return
                    if image.image_annotated is not None:
                        self.pipeline.exporter.save_image(image.image_annotated, os.path.basename(image.path))
            series_cache.save_series(self.series, self.config)

    def _run_barcode_detection(self):
        """Run barcode detection on the series."""
        total_images = len(self.series.images)
        for i, image_data in enumerate(self.pipeline.iter_detect_barcodes(
                self.series.images, cancelled=lambda: self._cancelled)):
            self.progress.emit(int(90 * (i + 1) / max(1, total_images)), 100)
        
        if not self._cancelled:
            # Save to cache
            from ..io import series_cache
            self.phase.emit("Saving results")
            series_cache.save_series(self.series, self.config)


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
        
        # An explicit CLI configuration takes precedence over the last preset.
        self._startup_preset = False
        self._active_preset_name = None
        if config is None:
            from pathlib import Path
            store = self._get_preset_store()
            preferences = QSettings("RootTracker", "RootTracker")
            saved = preferences.value('active_preset', '')
            if saved and Path(saved) == store.path('in_vitro') and store.path('Defaults').exists():
                saved = str(store.path('Defaults'))
                preferences.setValue('active_preset', saved)
            if saved:
                try:
                    config = Config.from_yaml(saved)
                    from pathlib import Path
                    self._active_preset_name = Path(saved).stem
                    self._startup_preset = os.path.isdir(config.data.input)
                except Exception:
                    config = None
            if config is None:
                if 'Defaults' in store.names():
                    config = store.load('Defaults')
                    self._active_preset_name = 'Defaults'
                    self._startup_preset = os.path.isdir(config.data.input)
                else:
                    config = Config()
                    self._active_preset_name = None
        self._config = config
        
        # Settings for persisting application state
        self._settings = QSettings("RootTracker", "RootTracker")
        
        # Pipeline and data state
        self._pipeline: RootTrackingPipeline | None = None
        self._series_dict: dict[str, ImageSeries] = {}
        self._evaluation_timer = QTimer(self)
        self._evaluation_timer.setSingleShot(True)
        self._evaluation_timer.timeout.connect(self._evaluate_selected_step)
        self._auto_apply_baselines = {}
        self._auto_mask_baselines = {}
        self._mask_refresh_active = False
        self._auto_apply_timer = QTimer(self)
        self._auto_apply_timer.setSingleShot(True)
        self._auto_apply_timer.setInterval(300)
        self._auto_apply_timer.timeout.connect(self._apply_auto_settings)
        self._settings_drafts = {}
        self._mask_drafts = set()
        self._restoring_settings = False
        self._current_image: ImageData | None = None
        self._current_series: ImageSeries | None = None
        
        # Processing state management (new unified system)
        self._state: ProcessingState = ProcessingState.IDLE
        self._context: ProcessingContext | None = None
        self._cancel_requested = mp.Event()
        self._operation_progress = None
        self._worker: ProcessWorker | None = None  # Current worker thread
        self._executor: ProcessPoolExecutor | None = None  # Current batch executor
        self._step_before_processing: WorkflowStep | None = None  # Track step to restore on cancel
        
        # Cache synchronization
        self._cache_lock = threading.Lock()
        self._series_in_use: set[str] = set()  # Track series being processed

        
        self.setWindowTitle("Root Tracker")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(button_stylesheet(self.palette()))
        
        self._setup_ui()
        self._setup_menu()
        from .roi_editor import RoiEditor
        self._roi_editor = RoiEditor(self)
        self._connect_signals()
        
        # Start on Load step with settings panel and image viewer hidden
        self._settings_panel.hide()
        self._image_viewer.hide()
        
        # Set initial button states (no images loaded yet)
        self._update_initial_ui_state()
        
        # Finish startup while hidden. The launcher shows the normal window
        # once, without an early maximize request racing the first titlebar drag.
        # Load last folder if available
        if self._startup_preset:
            self._reload_images()
        else:
            self._load_last_folder()
    
    def closeEvent(self, event):
        """Handle application close - cleanup any running workers."""
        self._settings_panel.finish_color_picker()
        self._auto_apply_timer.stop()
        self._evaluation_timer.stop()
        self._roi_editor.close()
        # Cancel any running operations
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.cancel()
                # Wait for thread to finish gracefully
                if not self._worker.wait(2000):  # Wait up to 2 seconds
                    self._worker.terminate()
                    self._worker.wait()  # Wait for termination to complete
            self._worker = None
        
        # Shutdown executor if running
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        
        event.accept()
    
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
        self._splitter.setHandleWidth(1)
        
        # Left panel: Image tree only
        self._image_tree = ImageTree()
        self._image_tree.set_step(WorkflowStep.LOAD, self._config)
        self._image_tree.setMinimumWidth(180)
        # Max width will be set dynamically based on whether images are loaded
        self._splitter.addWidget(self._image_tree)
        
        # Center: Image viewer
        self._image_viewer = ImageViewer()
        self._splitter.addWidget(self._image_viewer)
        
        # Right panel: Settings
        self._settings_panel = SettingsPanel(self._config)
        self._settings_panel.setMinimumWidth(360)
        self._settings_panel.setMaximumWidth(480)
        self._splitter.addWidget(self._settings_panel)
        
        # Set splitter proportions
        self._splitter.setSizes([200, 800, 360])
        
        main_layout.addWidget(self._splitter)
        
        # Status bar containing all controls
        from PySide6.QtWidgets import QPushButton, QLabel, QProgressBar
        
        self._status_bar = QStatusBar()
        self._status_bar.setSizeGripEnabled(False)  # Remove resize grip
        # Match the status bar's two-pixel bottom inset at the right edge.
        self._status_bar.setContentsMargins(0, 0, 2, 0)
        self.setStatusBar(self._status_bar)
        # Left: Zoom controls (as regular widgets - stay on left)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setMinimumSize(44, 32)
        self._fit_btn.clicked.connect(self._image_viewer.fit_in_view)
        self._status_bar.addWidget(self._fit_btn, 0)
        
        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setMinimumSize(32, 32)
        self._zoom_out_btn.clicked.connect(self._image_viewer.zoom_out)
        self._status_bar.addWidget(self._zoom_out_btn, 0)
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(45)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_bar.addWidget(self._zoom_label, 0)
        
        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setMinimumSize(32, 32)
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
        self._progress_bar.setFixedWidth(150)
        self._progress_bar.setFixedHeight(self._fit_btn.sizeHint().height())
        self._progress_bar.setMaximum(100)
        self._progress_bar.hide()
        self._status_bar.addPermanentWidget(self._progress_bar)
        
        # Next step button (rightmost)
        self._next_step_btn = QPushButton("Next")
        self._next_step_btn.setMinimumHeight(32)
        self._next_step_btn.setToolTip("Go to next step (auto-processes current step if needed)")
        self._next_step_btn.clicked.connect(self._on_next_step_clicked)
        self._status_bar.addPermanentWidget(self._next_step_btn)
        
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # Center compact menu items beside the 42 px workflow bar.
        menubar.setFixedHeight(42)
        menubar.setStyleSheet("""
            QMenuBar::item {
                background: transparent;
                font-size: 14px;
                padding: 6px 8px;
                margin: 4px 0px;
            }
            QMenuBar::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
                border-radius: 3px;
            }
        """)
        
        # Add workflow bar to right side of menu bar
        menubar.setCornerWidget(self._workflow_bar, Qt.Corner.TopRightCorner)
        
        # File menu
        file_menu = MenuBarPopup(menubar, "&File")
        menubar.addMenu(file_menu)
        
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
        
        self._presets_menu = MenuBarPopup(menubar, '&Presets')
        menubar.addMenu(self._presets_menu)
        self._presets_menu.addAction('Manage presets…', self._manage_presets)
        self._presets_menu.aboutToShow.connect(self._refresh_presets_menu)

        # View menu
        view_menu = MenuBarPopup(menubar, "&View")
        view_menu.setToolTipsVisible(False)
        menubar.addMenu(view_menu)
        
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

        self._auto_preview_action = QAction("&Auto-apply settings", self)
        self._auto_preview_action.setCheckable(True)
        self._auto_preview_action.setChecked(self._config.gui.auto_apply)
        self._auto_preview_action.setStatusTip("Apply edits automatically in Load and Preprocess. Reset restores settings from before automatic edits. Tracking settings use Apply.")
        self._auto_preview_action.toggled.connect(self._on_auto_preview_toggled)
        view_menu.addAction(self._auto_preview_action)
        
        self._detect_barcodes_action = QAction("Detect &Barcodes", self)
        self._detect_barcodes_action.setCheckable(True)
        self._detect_barcodes_action.setChecked(self._config.data.detect_barcodes)
        self._detect_barcodes_action.setStatusTip("Read barcodes from photographs and compare them with the identifiers in their filenames. Turn off for images without barcodes.")
        self._detect_barcodes_action.toggled.connect(self._on_detect_barcodes_toggled)
        view_menu.addAction(self._detect_barcodes_action)
        
        # Help menu
        help_menu = MenuBarPopup(menubar, "&Help")
        menubar.addMenu(help_menu)
        
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
        self._image_tree.rsml_export_requested.connect(self._on_export_rsml)
        self._image_tree.rsml_replace_requested.connect(self._on_replace_rsml)
        
        # Settings panel
        self._settings_panel.dirty_changed.connect(self._on_settings_dirty_changed)
        self._settings_panel.discard_requested.connect(self._discard_settings_changes)
        self._settings_panel.reset_auto_requested.connect(self._reset_auto_settings)
        self._sync_auto_apply_controls()
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
        self._image_viewer.mask_restore_toggled.connect(self._settings_panel.set_temporary_mask_restore)
        self._image_viewer.mask_diameter_steps.connect(self._settings_panel.adjust_mask_diameter)
    
    def _get_preset_store(self):
        from pathlib import Path
        from .presets import PresetStore
        directory = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)) / 'presets'
        store = PresetStore(directory)
        store.initialize(Path(__file__).resolve().parents[2] / 'configs')
        return store

    def _refresh_presets_menu(self):
        self._presets_menu.clear()
        manage = self._presets_menu.addAction('Manage presets…')
        manage.triggered.connect(self._manage_presets)
        manage.setEnabled(self._state == ProcessingState.IDLE)
        try:
            store = self._get_preset_store()
            self._presets_menu.addSeparator()
            for name in store.names():
                action = self._presets_menu.addAction(name)
                action.setCheckable(True)
                action.setChecked(name == self._active_preset_name)
                action.setEnabled(self._state == ProcessingState.IDLE)
                action.triggered.connect(lambda checked=False, n=name: self._select_preset(n))
        except Exception as error:
            self.statusBar().showMessage(f'Cannot load presets: {error}')

    def _manage_presets(self):
        if self._state != ProcessingState.IDLE:
            return
        from .dialogs.presets_dialog import PresetsDialog
        try:
            dialog = PresetsDialog(self._get_preset_store(), self._config, self, active_name=self._active_preset_name)
            if dialog.exec() == dialog.DialogCode.Accepted:
                self._apply_preset(dialog.selected_config, dialog.selected_name)
        except Exception as error:
            QMessageBox.warning(self, 'Preset error', str(error))

    def _select_preset(self, name):
        if self._state != ProcessingState.IDLE:
            return
        try:
            self._apply_preset(self._get_preset_store().load(name), name)
        except Exception as error:
            QMessageBox.warning(self, 'Preset error', str(error))

    def _apply_preset(self, config, name):
        """Switch dataset and processing defaults as one configuration change."""
        from dataclasses import fields
        from pathlib import Path
        config = deepcopy(config)
        config._validate()
        if not Path(config.data.input).is_dir():
            raise ValueError(f'Input folder does not exist: {config.data.input}')
        self._settings_panel.finish_color_picker()
        self._auto_apply_timer.stop()
        self._evaluation_timer.stop()
        self._roi_editor.reset()
        if (config.data.input == self._config.data.input and
                config.preprocess_config_hash() != self._config.preprocess_config_hash()):
            for series in self._series_dict.values():
                self._invalidate_preprocessing(series)
        # Keep references held by SettingsPanel and other widgets valid.
        for field in fields(Config):
            setattr(self._config, field.name, deepcopy(getattr(config, field.name)))
        self._auto_apply_baselines.clear()
        self._auto_mask_baselines.clear()
        self._settings_drafts.clear()
        self._mask_drafts.clear()
        self._current_series = self._current_image = None
        self._workflow_bar.blockSignals(True)
        self._workflow_bar._completed_steps.clear()
        self._workflow_bar.set_current_step(WorkflowStep.LOAD)
        self._workflow_bar.blockSignals(False)
        self._settings_panel.set_step(WorkflowStep.LOAD)
        for action, value in ((self._auto_preview_action, config.gui.auto_apply),
                              (self._detect_barcodes_action, config.data.detect_barcodes)):
            action.blockSignals(True)
            action.setChecked(value)
            action.blockSignals(False)
        self._sync_auto_apply_controls()
        self._series_dict = {}
        self._image_tree.set_series({})
        self._image_viewer.set_image(None)
        self._image_viewer.clear_centroids()
        self._image_viewer.clear_barcode_overlay()
        self._image_tree.set_filter_aside(False)
        self._image_tree.set_step(WorkflowStep.LOAD, self._config)
        self._reload_images()
        self._active_preset_name = name
        self._settings.setValue('active_preset', str(self._get_preset_store().path(name)))
        self.statusBar().showMessage(f'Preset applied: {name}', 5000)

    def _on_load_images(self) -> None:
        """Handle load images action (Ctrl+O)."""
        dialog = LoadDialog(
            self, 
            self._config.data.input,
            self._config.data.filename_template,
            self._config.data.date_format,
            self._config.data.detect_barcodes
        )
        
        if dialog.exec() == LoadDialog.DialogCode.Accepted:
            # Update config
            self._config.data.input = dialog.input_path
            self._config.data.filename_template = dialog.filename_template
            self._config.data.date_format = dialog.date_format
            self._config.data.detect_barcodes = dialog.detect_barcodes
            
            self._reload_images()

    def _reload_images(self) -> None:
        """Reload images from the current input folder."""
        # Create pipeline and load images
        self._pipeline = RootTrackingPipeline(self._config)
        
        try:
            self._series_dict = self._pipeline.load_images()
            self._settings_panel.finish_color_picker()
            self._roi_editor.reset()
            self._auto_apply_baselines.clear()
            self._auto_mask_baselines.clear()
            self._auto_apply_timer.stop()
            self._settings_drafts.clear()
            self._mask_drafts.clear()
            self._current_series = None
            self._current_image = None
            self._restore_settings_draft()
            
            if not self._series_dict:
                QMessageBox.warning(
                    self, "No Images Found",
                    f"No images found in '{self._config.data.input}' matching template '{self._config.data.filename_template}'.\n\n"
                    "Check that the folder path and filename template are correct."
                )
                return
            
            self._image_tree.set_series(self._series_dict)
            self._image_tree.set_folder_path(self._config.data.input)
            
            # Show image viewer now that we have images
            self._image_viewer.show()
            self._settings_panel.show()
            
            # Select first image
            self._image_tree.select_first_image()
            
            # Fit image to view after UI updates
            QTimer.singleShot(0, self._image_viewer, self._image_viewer.fit_in_view)
            
            # Mark step as complete but DON'T auto-advance
            self._workflow_bar.mark_step_completed(WorkflowStep.LOAD)
            
            # Update cache action state
            self._update_cache_action_state()
            
            # Save this folder path and settings for next time
            self._settings.setValue("last_folder", self._config.data.input)
            self._settings.setValue("filename_template", self._config.data.filename_template)
            self._settings.setValue("date_format", self._config.data.date_format)
            self._settings.setValue("detect_barcodes", self._config.data.detect_barcodes)
            
            # Update menu action to match current setting
            self._detect_barcodes_action.setChecked(self._config.data.detect_barcodes)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load images:\n{e}")

    

    def _on_image_selected(self, image_data: ImageData) -> None:
        """Handle image selection in tree."""
        step = self._workflow_bar.get_current_step()
        if self._state != ProcessingState.IDLE:
            return
        selected_series = self._image_tree.get_selected_series()
        if selected_series is not self._current_series:
            self._activate_group_settings(selected_series)
        self._current_image = image_data
        self._current_series = selected_series

        # Reload from cache if arrays were freed
        if self._current_series is not None and step != WorkflowStep.LOAD:
            self._ensure_series_loaded(self._current_series)

        # Auto-detect barcodes for this group if on LOAD step and auto-preview enabled
        if (step == WorkflowStep.LOAD and
            self._config.data.detect_barcodes and
            self._pipeline is not None and
            self._current_series is not None):
            # Only detect if group has undetected barcodes
            if not all(img.barcode_detected for img in self._current_series.images):
                # Show the selected photo immediately; the completion handler
                # refreshes its overlay after the asynchronous barcode scan.
                self._display_image(image_data)
                self._detect_barcodes_in_group(self._current_series)
                return

        # Auto-preprocess if enabled and on PREPROCESS step
        if (step == WorkflowStep.PREPROCESS and
            self._pipeline is not None and
            self._current_series is not None):
            # First ensure barcodes are detected (if enabled)
            if (self._config.data.detect_barcodes and
                not all(img.barcode_detected for img in self._current_series.images)):
                self._detect_barcodes_in_group(self._current_series, continue_to_next_step=True)
                return
            self._preprocess_group(self._current_series)
            return

        # Auto-process if enabled and on TRACK step
        if (step == WorkflowStep.TRACK and
            self._pipeline is not None and
            self._current_series is not None):
            # First ensure barcodes are detected (if enabled)
            if (self._config.data.detect_barcodes and
                not all(img.barcode_detected for img in self._current_series.images)):
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
        if self._state != ProcessingState.IDLE:
            return
        self._activate_group_settings(series)

        # Deselect mask tools when switching groups
        self._settings_panel.deselect_mask_tools()

        self._current_series = series
        self._current_image = series.images[0] if series.images else None

        # Reload from cache if arrays were freed
        if self._workflow_bar.get_current_step() != WorkflowStep.LOAD:
            self._ensure_series_loaded(series)

        # Initialize working_mask if needed
        if series.working_mask is None and series.user_mask is not None:
            series.working_mask = series.user_mask.copy()

        step = self._workflow_bar.get_current_step()

        # Auto-detect barcodes if on LOAD step and auto-preview enabled
        if (step == WorkflowStep.LOAD and
            self._config.data.detect_barcodes and
            self._pipeline is not None):
            self._detect_barcodes_in_group(series)

        if self._current_image:
            self._display_image(self._current_image)

        # Auto-preprocess if enabled and on PREPROCESS step
        if (step == WorkflowStep.PREPROCESS and
            self._pipeline is not None):
            # First ensure barcodes are detected (if enabled)
            if (self._config.data.detect_barcodes and
                not all(img.barcode_detected for img in series.images)):
                self._detect_barcodes_in_group(series, continue_to_next_step=True)
                return
            self._preprocess_group(series)

        # Auto-process if enabled and on TRACK step
        if (step == WorkflowStep.TRACK and
            self._pipeline is not None):
            # First ensure barcodes are detected (if enabled)
            if (self._config.data.detect_barcodes and
                not all(img.barcode_detected for img in series.images)):
                self._detect_barcodes_in_group(series, continue_to_next_step=True)
                return
            self._auto_process_for_tracking(series)

        # Update button states for new selection
        self._update_process_button_states()
    
    def _display_image(self, image_data: ImageData, *, preserve_view: bool = False) -> None:
        """Display an image in the viewer."""
        step = self._workflow_bar.get_current_step()
        # Reload from cache if arrays were freed
        if self._current_series is not None and step != WorkflowStep.LOAD:
            self._ensure_series_loaded(self._current_series)

        # Determine which image to show based on workflow step
        step = self._workflow_bar.get_current_step()
        editor_presented = self._roi_editor.present(image_data, step, preserve_view=preserve_view)
        
        if step == WorkflowStep.LOAD:
            # Show original image with barcode overlay
            image = None if editor_presented else cv2.imread(image_data.path)
            
            # Draw barcode overlay if detected
            if not self._config.data.detect_barcodes:
                self._image_viewer.clear_barcode_overlay()
            elif image_data.barcode_rect is not None:
                label = image_data.barcode_read
                if image_data.barcode_mismatch:
                    label = f"X {label} (expected: {image_data.barcode})"
                
                rectangle = (self._roi_editor.load_preview_rect(image_data.barcode_rect)
                             if editor_presented else image_data.barcode_rect)
                if rectangle is None:
                    self._image_viewer.clear_barcode_overlay()
                else:
                    self._image_viewer.set_barcode_overlay(
                        rectangle, label, image_data.barcode_mismatch,
                        label_position=(self._roi_editor.load_preview_label_position(image_data.barcode_rect)
                                        if editor_presented else None))
            else:
                self._image_viewer.clear_barcode_overlay()
            
            self._image_viewer.clear_centroids()
        elif step == WorkflowStep.PREPROCESS:
            # Show processed image (RGB) if available, else original (rotated + cropped)
            if editor_presented:
                image = None
            elif image_data.image is not None:
                image = image_data.image
                self._image_viewer.clear_centroids()
                self._image_viewer.clear_barcode_overlay()
            else:
                # Preview: Rotate and auto-crop (to avoid "flash" of raw image)
                image = cv2.imread(image_data.path)
                
                from ..preprocessing import ImageCropper
                cropper = ImageCropper(self._config)
                
                try:
                    image = cropper.process(image)
                except (ValueError, cv2.error):
                    pass

                self._image_viewer.clear_centroids()
                self._image_viewer.clear_barcode_overlay()
        else:
            # TRACK: prefer annotated image, fall back to preprocessed
            if image_data.rsml_document is not None:
                from ..io.rsml_replacement import render_replacement
                image = render_replacement(image_data)
            elif image_data.image_annotated is not None:
                image = image_data.image_annotated
            elif image_data.image is not None:
                image = image_data.image
            elif image_data.process is not None:
                image = image_data.process
            else:
                image = cv2.imread(image_data.path)
            self._image_viewer.clear_centroids()
            self._image_viewer.clear_barcode_overlay()

        if editor_presented:
            self._image_viewer.set_mask_data(None, None)
            return

        if image is not None:
            self._image_viewer.set_image(image, preserve_view=preserve_view)

            if step == WorkflowStep.PREPROCESS and image_data.image is not None:
                self._image_viewer.set_centroids(list(zip(
                    image_data.positions_x, image_data.positions_y)))

            # Set mask data if in TRACK step
            if (step == WorkflowStep.TRACK and self._current_series is not None
                    and image_data.rsml_document is None):
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
        # Don't allow step changes while processing
        if self._state != ProcessingState.IDLE:
            return
        
        # Validate step prerequisites
        if step > WorkflowStep.LOAD and not self._workflow_bar.is_step_completed(WorkflowStep.LOAD):
            QMessageBox.warning(
                self, "Step Not Available",
                "Please load images first (Step 1) before proceeding."
            )
            self._workflow_bar.set_current_step(WorkflowStep.LOAD)
            return

        self._settings_panel.finish_color_picker()
        self._apply_auto_settings(evaluate=False)
        self._roi_editor.reset()
        self._save_settings_draft()
        self._restore_settings_draft(step)
        self._settings_panel.show()

        # Update tree filtering: show aside items ONLY in LOAD step
        self._image_tree.set_step(step, self._config)
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

        # Reset zoom to fit when switching steps
        QTimer.singleShot(0, self._image_viewer, self._image_viewer.fit_in_view)

        self._update_process_button_states()

    def _evaluate_selected_step(self):
        if self._state != ProcessingState.IDLE or self._current_series is None or self._pipeline is None:
            return
        step = self._workflow_bar.get_current_step()
        if step == WorkflowStep.PREPROCESS:
            self._preprocess_group(self._current_series)
        elif step == WorkflowStep.TRACK:
            self._run_track_step_processing()

    def _handle_enter_preprocess(self) -> None:
        """Handle entering the PREPROCESS step."""
        # Display current image immediately for smooth transition
        if self._current_image:
            self._display_image(self._current_image)
        
        # Defer auto-processing until after UI transition completes
        if (self._current_series is not None
                and self._pipeline is not None):
            state = self._current_series.pipeline_state
            current_hash = self._config.preprocess_config_hash()
            if not (state.preprocessed and state.preprocess_config_hash == current_hash):
                # Schedule preprocessing to run after UI updates
                self._evaluation_timer.start(0)

    def _handle_enter_track(self) -> None:
        """Handle entering the TRACK step."""
        if self._current_series is None or self._pipeline is None:
            if self._current_image:
                self._display_image(self._current_image)
            return
        
        # Display current image immediately for smooth transition
        if self._current_image:
            self._display_image(self._current_image)
        
        # Check if processing will be needed
        state = self._current_series.pipeline_state
        preprocess_hash = self._config.preprocess_config_hash()
        needs_preprocessing = not state.preprocessed or state.preprocess_config_hash != preprocess_hash
        needs_tracking = not self._pipeline.is_tracking_current(self._current_series)
        
        # Defer heavy processing until after UI transition completes
        if needs_preprocessing or needs_tracking:
            self._evaluation_timer.start(0)
    
    def _run_track_step_processing(self) -> None:
        """Execute tracking step processing (called after UI transition)."""
        if self._current_series is None or self._pipeline is None:
            return

        # Check if tracking is already done and current
        if self._pipeline.is_tracking_current(self._current_series):
            # Tracking already done — just display the annotated image
            if self._current_image:
                self._display_image(self._current_image)
            return

        # Auto-run tracking if enabled, using the proper chain
        self._auto_process_for_tracking(self._current_series)
    
    def _save_settings_draft(self):
        if self._restoring_settings or self._current_series is None:
            return
        panel = self._settings_panel
        key = (self._current_series.group, panel._current_step)
        edits = {k: v for k, v in panel.get_current_values().items()
                 if v != panel._original_values.get(k)}
        if edits:
            self._settings_drafts[key] = edits
        else:
            self._settings_drafts.pop(key, None)
        if panel._current_step == WorkflowStep.TRACK:
            if self._current_series.has_pending_mask_changes():
                self._mask_drafts.add(self._current_series.group)
            else:
                self._mask_drafts.discard(self._current_series.group)

    def _refresh_pending_settings(self):
        groups = {key[0] for key in self._settings_drafts} | self._mask_drafts
        self._image_tree.set_pending_groups([s for s in self._series_dict.values() if s.group in groups])

    def _on_settings_dirty_changed(self, dirty):
        if self._restoring_settings:
            return
        self._save_settings_draft()
        self._refresh_pending_settings()
        self._sync_auto_apply_controls()
        if dirty and self._auto_apply_enabled():
            self._auto_apply_timer.start()

    def _restore_settings_draft(self, step=None):
        panel = self._settings_panel
        self._restoring_settings = True
        try:
            if step is not None and step != panel._current_step:
                panel.set_step(step)
            panel.reset_for_group()
            if self._current_series is not None:
                key = (self._current_series.group, panel._current_step)
                panel.set_pending_values(self._settings_drafts.get(key, {}))
                if panel._current_step == WorkflowStep.TRACK and self._current_series.has_pending_mask_changes():
                    panel._mark_dirty()
        finally:
            self._restoring_settings = False
        self._save_settings_draft()
        self._refresh_pending_settings()
        self._roi_editor.schedule()
        self._sync_auto_apply_controls()
        if panel._is_dirty and self._auto_apply_enabled():
            self._auto_apply_timer.start()

    def _activate_group_settings(self, series):
        self._settings_panel.finish_color_picker()
        self._apply_auto_settings(evaluate=False)
        self._save_settings_draft()
        self._current_series = series
        self._restore_settings_draft()

    def _discard_settings_changes(self):
        self._settings_panel.finish_color_picker()
        series = self._current_series
        if series is None:
            return
        step = self._settings_panel._current_step
        self._settings_drafts.pop((series.group, step), None)
        if step == WorkflowStep.TRACK:
            series.working_mask = None if series.user_mask is None else series.user_mask.copy()
            self._mask_drafts.discard(series.group)
            self._image_viewer.set_mask_data(series.user_mask, series.working_mask)
        self._restore_settings_draft()

    def _update_config_from_panel(self) -> None:
        """Update config object from settings panel values."""
        values = self._settings_panel.get_current_values()
        from .roi_editor import apply_editor_values
        apply_editor_values(self._config, values)
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
    
    def _invalidate_preprocessing(self, series):
        # A same-sized rotated crop still changes mask coordinates.
        series.clear_preprocessing_results()
        self._auto_mask_baselines.pop(series.group, None)
        self._mask_drafts.discard(series.group)
        mask_io.delete_mask(series, self._config)

    def _on_apply_settings(self, *, evaluate=True) -> None:
        """Handle Apply button - reprocess current group with new settings and apply mask."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please load images first.")
            return

        step = self._settings_panel._current_step

        # Handle mask application in Track step
        mask_changed = False
        mask_refresh = False
        if step == WorkflowStep.TRACK:
            # Check if mask changed
            mask_changed = self._current_series.has_pending_mask_changes()
            if mask_changed:
                mask_refresh = evaluate and self._auto_apply_enabled()
                # Apply mask
                working_mask = self._current_series.working_mask
                self._current_series.user_mask = working_mask.copy() if working_mask is not None else None
                self._current_series.working_mask = working_mask.copy() if working_mask is not None else None

                # Automatic refreshes persist in the worker, keeping PNG
                # encoding off the drawing path. Navigation/manual apply saves here.
                if not mask_refresh:
                    try:
                        if working_mask is not None and np.any(working_mask > 0):
                            mask_io.save_mask(self._current_series, self._config)
                        else:
                            # All mask erased - delete file
                            mask_io.delete_mask(self._current_series, self._config)
                    except Exception as e:
                        QMessageBox.warning(self, "Warning", f"Failed to save mask:\n{e}")

                # Keep the last annotation visible during an automatic mask
                # refresh; the live overlay already hides excluded pixels.
                self._current_series.pipeline_state.invalidate_from('track')

        # Snapshot config hashes BEFORE updating
        old_preprocess_hash = self._config.preprocess_config_hash()
        old_tracking_hash = self._config.tracking_config_hash()

        # Update config from settings panel
        self._update_config_from_panel()

        # Determine what changed
        preprocess_changed = (old_preprocess_hash != self._config.preprocess_config_hash())
        tracking_changed = (old_tracking_hash != self._config.tracking_config_hash())

        if preprocess_changed:
            # Settings are shared. Every affected mask uses the old geometry,
            # even when the new crop has exactly the same pixel dimensions.
            for series in self._series_dict.values():
                self._invalidate_preprocessing(series)
            if not any(series is self._current_series for series in self._series_dict.values()):
                self._invalidate_preprocessing(self._current_series)
            if not evaluate:
                pass  # The newly selected group/step will evaluate these settings.
            elif step == WorkflowStep.LOAD:
                if self._current_image:
                    self._display_image(self._current_image, preserve_view=True)
            else:
                self._preprocess_group(self._current_series, force=True)
        elif tracking_changed or mask_changed:
            # Tracking params or mask changed — invalidate tracking only and retrack
            if not mask_changed:  # Already cleared above if mask changed
                self._current_series.clear_tracking_results()
            if evaluate and self._current_series is not None:
                self._mask_refresh_active = mask_refresh
                self._preserve_tracking_view_for = self._current_image
                self._on_track_roots()
            elif evaluate and self._current_image:
                self._display_image(self._current_image, preserve_view=True)

        self._restore_settings_draft()
        self._refresh_tree_status()

    def _on_apply_all_settings(self) -> None:
        """Handle Apply All button - propagate settings to ALL groups.

        This only changes the parameters and marks every already-evaluated
        group dirty; it does NOT eagerly recompute all groups. Dirty groups
        are recomputed lazily when the final "track all and export"
        computation runs (their stale pipeline state forces a rerun there).
        Only the currently-viewed group is recomputed now, for immediate
        visual feedback.
        """
        if self._pipeline is None or not self._series_dict:
            return

        step = self._workflow_bar.get_current_step()
        # Snapshot config hashes BEFORE updating
        old_preprocess_hash = self._config.preprocess_config_hash()
        old_tracking_hash = self._config.tracking_config_hash()

        # Update config from settings panel
        self._update_config_from_panel()

        preprocess_changed = (old_preprocess_hash != self._config.preprocess_config_hash())
        tracking_changed = (old_tracking_hash != self._config.tracking_config_hash())

        if preprocess_changed:
            # Mark every group dirty (frees arrays + invalidates pipeline state)
            # so the final all-groups computation reprocesses them.
            for series in self._series_dict.values():
                self._invalidate_preprocessing(series)
            # Recompute only the current group so the user sees the new result.
            if self._current_series is not None:
                if step == WorkflowStep.LOAD:
                    if self._current_image:
                        self._display_image(self._current_image, preserve_view=True)
                else:
                    self._preprocess_group(self._current_series, force=True)
        elif tracking_changed:
            for series in self._series_dict.values():
                series.clear_tracking_results()
            # Retrack only the current group for immediate feedback.
            if self._current_series is not None:
                self._preserve_tracking_view_for = self._current_image
                self._on_track_roots()
            elif self._current_image:
                self._display_image(self._current_image, preserve_view=True)

        self._restore_settings_draft()
        self._refresh_tree_status()

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
            self._pipeline.redetect_centroids(self._current_series)

            # Save to cache (preprocessing results updated with new centroids)
            series_cache.save_series(self._current_series, self._config)

            if self._current_image:
                self._display_image(self._current_image, preserve_view=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed:\n{e}")
        finally:
            self._refresh_tree_status()
            self._processing_label.hide()
            QApplication.restoreOverrideCursor()
    
    def _auto_apply_enabled(self):
        return (self._auto_preview_action.isChecked() and
                self._settings_panel._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS, WorkflowStep.TRACK))

    def _auto_mask_changed(self):
        series = self._current_series
        if (series is None or self._settings_panel._current_step != WorkflowStep.TRACK
                or series.group not in self._auto_mask_baselines):
            return False
        baseline = self._auto_mask_baselines[series.group]
        mask = series.working_mask
        if mask is None or baseline is None:
            other = baseline if mask is None else mask
            return other is not None and bool(np.any(other))
        return not np.array_equal(mask, baseline)

    def _sync_auto_apply_controls(self):
        panel = self._settings_panel
        key = (self._current_series.group, panel._current_step) if self._current_series else None
        baseline = self._auto_apply_baselines.get(key)
        panel.set_auto_apply_mode(self._auto_apply_enabled(),
                                  (baseline is not None and panel.get_current_values() != baseline)
                                  or self._auto_mask_changed())

    def _apply_auto_settings(self, *, evaluate=True):
        if not self._auto_apply_enabled() or self._current_series is None or self._pipeline is None:
            return
        panel = self._settings_panel
        if not panel._is_dirty:
            return
        # Coalesce a drag into one processing run. Color changes already have a
        # live segmentation preview; commit them when the range dialog closes.
        if (self._state != ProcessingState.IDLE or (evaluate and QApplication.mouseButtons() != Qt.MouseButton.NoButton)
                or (panel._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS)
                    and panel._color_control._picker is not None)):
            self._auto_apply_timer.start()
            return
        key = (self._current_series.group, panel._current_step)
        self._auto_apply_baselines.setdefault(key, deepcopy(panel._original_values))
        if (panel._current_step == WorkflowStep.TRACK
                and self._current_series.group not in self._auto_mask_baselines):
            self._auto_mask_baselines[self._current_series.group] = deepcopy(self._current_series.user_mask)
        self._settings_drafts.pop(key, None)
        panel._store_original_values()
        # Use the regular invalidation/processing path, without closing crop
        # editing or switching off segmentation during automatic application.
        self._on_apply_settings(evaluate=evaluate)
        self._sync_auto_apply_controls()

    def _reset_auto_settings(self):
        if not self._auto_apply_enabled() or self._current_series is None:
            return
        panel = self._settings_panel
        key = (self._current_series.group, panel._current_step)
        baseline = self._auto_apply_baselines.get(key)
        if baseline is None:
            return
        panel.finish_color_picker()
        series = self._current_series
        if panel._current_step == WorkflowStep.TRACK and series.group in self._auto_mask_baselines:
            mask = self._auto_mask_baselines[series.group]
            # An empty array stages removal of an applied mask; None means no draft.
            if mask is None and series.user_mask is not None:
                mask = np.zeros_like(series.user_mask)
            series.working_mask = deepcopy(mask)
            self._image_viewer.set_mask_data(series.user_mask, series.working_mask)
            panel._mask_dirty = series.has_pending_mask_changes()
        panel.set_pending_values(deepcopy(baseline))
        self._auto_apply_timer.stop()
        self._apply_auto_settings()
        self._sync_auto_apply_controls()

    def _on_auto_preview_toggled(self, enabled: bool) -> None:
        self._config.gui.auto_apply = enabled
        self._auto_apply_timer.stop()
        self._sync_auto_apply_controls()
        if self._auto_apply_enabled() and self._settings_panel._is_dirty:
            self._auto_apply_timer.start()

    def _on_detect_barcodes_toggled(self, enabled: bool) -> None:
        """Handle Detect Barcodes menu toggle."""
        self._config.data.detect_barcodes = enabled
        self._refresh_tree_status()
        # Save to settings for persistence
        self._settings.setValue("detect_barcodes", enabled)
        # Reuse stored detections immediately; toggling visibility never rescans.
        if self._current_image is not None:
            self._display_image(self._current_image, preserve_view=True)
    
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
                
                from ..io.rsml_replacement import move_image_with_replacement
                move_image_with_replacement(src_path, target_path)
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
                
                from ..io.rsml_replacement import move_image_with_replacement
                move_image_with_replacement(src_path, target_path)
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
                    from ..io.rsml_replacement import replacement_path
                    replacement_path(image_data.path).unlink(missing_ok=True)
                    
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
        Detect barcodes for all images in a group.

        Args:
            series: ImageSeries to detect barcodes in.
            continue_to_next_step: If True, set up continuation to preprocessing/tracking after detection.
        """
        if self._pipeline is None:
            return

        # Store series for continuation
        self._processing_series = series

        # Check if barcodes already detected for this group
        if all(img.barcode_detected for img in series.images):
            # Already detected - manually trigger continuation if needed
            if continue_to_next_step:
                step = self._workflow_bar.get_current_step()
                if step == WorkflowStep.PREPROCESS:
                    self._preprocess_group(series)
                elif step == WorkflowStep.TRACK:
                    self._auto_process_for_tracking(series)
            return

        # Set up continuation flags for auto-processing chain
        if continue_to_next_step:
            step = self._workflow_bar.get_current_step()
            if step == WorkflowStep.PREPROCESS:
                # After barcode detection, continue to preprocessing only
                self._auto_process_pending_preprocess = True
                self._auto_process_pending_tracking = False
            elif step == WorkflowStep.TRACK:
                # After barcode detection, continue to preprocessing then tracking
                self._auto_process_pending_preprocess = True
                self._auto_process_pending_tracking = True

        # Use worker thread for barcode detection
        self._start_barcode_detection(series)

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

    # Removed _continue_after_barcode_detection - replaced by continuation pattern in finish handlers
    
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
        
        # Create processing context
        self._context = ProcessingContext(self, ProcessingState.BARCODE_DETECTION, undetected_groups)
        self._context.__enter__()
        
        # Setup cumulative progress bar
        self._progress_bar.setMaximum(total_images)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._processing_label.show()

        current_image_count = 0
        start_time = time.time()

        try:
            for series in undetected_groups:
                if self._cancel_requested.is_set():
                    break
                for image_data in self._pipeline.iter_detect_barcodes(
                        series.images, cancelled=self._cancel_requested.is_set):
                    current_image_count += 1
                    self._update_progress_label(start_time, current_image_count, total_images)
                    self._progress_bar.setValue(current_image_count)
                    self._image_tree.refresh_status()
                    QApplication.processEvents()
                
                # Persist completed reads, including partial progress on cancellation.
                series_cache.save_series(series, self._config)
                
            # Update tree to show warning icons (refresh preserves selection)
            self._image_tree.refresh()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Barcode detection failed:\n{e}")
        finally:
            # Exit processing context (handles cleanup)
            if self._context:
                self._context.__exit__(None, None, None)
        
        # Display current image if we have one
        if self._current_image:
            self._display_image(self._current_image)
    
    def _preprocess_all_groups(self, force: bool = False) -> bool:
        """Preprocess all unprocessed groups in parallel using all CPU cores.

        Submits each group to a ProcessPoolExecutor. Workers save results
        to disk cache. Progress bar tracks completed groups.
        
        Returns:
            True if completed successfully, False if cancelled.
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

        # Count total images for progress bar
        total_images = sum(len(s.images) for s in unprocessed_groups)

        # Create processing context for batch operation
        for series in unprocessed_groups:
            series.pipeline_state.invalidate_from('preprocess')
        self._context = ProcessingContext(self, ProcessingState.BATCH_PREPROCESS, unprocessed_groups)
        self._context.__enter__()

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
            self._executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx)
            executor = self._executor
            
            try:
                # Pass queue to workers
                futures = {
                    executor.submit(preprocess_and_cache_worker, (self._config, s, progress_queue)): s
                    for s in unprocessed_groups
                }

                while futures:
                    # Check cancellation
                    if self._cancel_requested.is_set():
                        # Shutdown executor immediately without waiting
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._processing_label.setText("Cancelled")
                        self._progress_bar.hide()  # Immediately hide to avoid "Finishing up..." flash
                        QApplication.processEvents()
                        time.sleep(0.3)
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
                            # Read only lightweight per-image metadata for status;
                            # processed arrays remain on disk until selected.
                            series_cache.load_series_state(series, self._config)
                            self._image_tree.refresh_status()
                        except Exception as e:
                            print(f"Error preprocessing {series.group}: {e}")
                        self._context.series = list(futures.values())
                        self._refresh_tree_status()
                    
                    # Update ETA even if no progress, to handle coasting
                    if completed_images > 0:
                        self._update_progress_label(start_time, completed_images, total_images)

                    if futures:
                        time.sleep(0.05)
            finally:
                self._executor = None
                executor.shutdown(wait=False)

            # Only proceed if not cancelled
            if not self._cancel_requested.is_set():
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
            # Exit processing context (handles cleanup)
            was_cancelled = self._cancel_requested.is_set()
            if self._context:
                self._context.was_cancelled = was_cancelled
                self._context.__exit__(None, None, None)

        return not was_cancelled
    
    def _preprocess_group(self, series: 'ImageSeries', force: bool = False, hide_progress: bool = True) -> None:
        """
        Preprocess the given group (series) using a worker thread.

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

        self._begin_operation_progress('preprocess')
        self._hide_progress_on_complete = hide_progress

        # Create and enter processing context
        self._context = ProcessingContext(self, ProcessingState.PREPROCESSING, series)
        self._context.__enter__()

        # Create and start worker thread
        self._worker = ProcessWorker(self._pipeline, series, self._config, 'preprocess')
        self._worker.phase.connect(self._on_worker_phase)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_preprocess_finished)
        self._worker.start()

    def _begin_operation_progress(self, operation):
        if self._operation_progress is None:
            operations = [operation]
            if operation == 'barcode' and getattr(self, '_auto_process_pending_preprocess', False):
                operations.append('preprocess')
            if operation != 'track' and getattr(self, '_auto_process_pending_tracking', False):
                operations.append('track')
            self._operation_progress = OperationProgress(operations)
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
        self._operation_progress.begin(operation)
        self._progress_bar.show()
        self._render_operation_progress()

    def _render_operation_progress(self):
        progress = self._operation_progress
        if progress is None:
            return
        self._progress_bar.setValue(int(progress.value * 100))
        remaining = progress.remaining()
        if remaining is None:
            eta = 'Estimating remaining time…'
        else:
            seconds = max(1, math.ceil(remaining))
            eta = f'{seconds // 60}m {seconds % 60}s remaining' if seconds >= 60 else f'{seconds}s remaining'
        self._processing_label.setText(f'{progress.label} · {eta}')
        self._processing_label.show()

    def _on_worker_phase(self, label):
        if self._operation_progress is not None:
            self._operation_progress.label = label
            self._render_operation_progress()

    def _on_worker_progress(self, current: int, total: int) -> None:
        self._image_tree.refresh_status()
        if self._operation_progress is not None:
            self._operation_progress.update(current, total)
            self._render_operation_progress()

    def _finish_operation_progress(self, success, continuing=False):
        if self._operation_progress is None:
            return
        if continuing:
            return
        if success:
            self._progress_bar.setValue(100)
        self._operation_progress = None
        self._auto_process_pending_preprocess = False
        self._auto_process_pending_tracking = False

    def _on_preprocess_finished(self, success: bool, error_msg: str) -> None:
        """Handle preprocessing completion."""
        # Determine if we should continue with next operation
        should_continue_to_tracking = (
            success and 
            hasattr(self, '_auto_process_pending_tracking') and 
            self._auto_process_pending_tracking
        )
        
        if success:
            # Mark step complete
            if not self._workflow_bar.is_step_completed(WorkflowStep.PREPROCESS):
                self._workflow_bar.mark_step_completed(WorkflowStep.PREPROCESS)

            if self._current_image:
                self._display_image(self._current_image, preserve_view=True)
            
            # Set up continuation if needed
            if should_continue_to_tracking:
                self._auto_process_pending_tracking = False
                if self._context:
                    self._context.next_operation = self._continue_auto_track
        elif error_msg and error_msg != "Cancelled":
            QMessageBox.critical(self, "Error", f"Preprocessing failed:\n{error_msg}")
        
        # Cleanup and exit context (handles UI unlock, progress hide, etc.)
        if self._context:
            was_cancelled = not success
            self._context.was_cancelled = was_cancelled
            self._context.__exit__(None, None, None)

    def _on_tracking_image_ready(self, image):
        """Show the actual calculated image without waiting for group disk I/O."""
        viewer = self._image_viewer
        if (self._mask_refresh_active and image is self._current_image
                and self._current_series is not None
                and not self._current_series.has_pending_mask_changes()
                and not viewer._drawing and viewer._rect_start_point is None):
            self._display_image(image, preserve_view=True)
            self._mask_image_published = image

    def _on_tracking_finished(self, success: bool, error_msg: str) -> None:
        """Handle tracking completion."""
        preserve_view = self._current_image is getattr(self, '_preserve_tracking_view_for', None)
        self._preserve_tracking_view_for = None
        if success:
            # Mark step complete
            self._workflow_bar.mark_step_completed(WorkflowStep.TRACK)

            # A newer stroke must remain visible; don't replace the scene
            # underneath a brush drag or redraw obsolete mask results.
            viewer = self._image_viewer
            newer_mask = self._current_series is not None and self._current_series.has_pending_mask_changes()
            editing = viewer._drawing or viewer._rect_start_point is not None
            already_shown = self._mask_refresh_active and self._current_image is getattr(self, '_mask_image_published', None)
            if self._current_image and not already_shown and not (self._mask_refresh_active and (newer_mask or editing)):
                self._display_image(self._current_image, preserve_view=preserve_view)
        elif error_msg and error_msg != "Cancelled":
            QMessageBox.critical(self, "Error", f"Tracking failed:\n{error_msg}")
        
        # Cleanup and exit context (handles UI unlock, progress hide, etc.)
        if self._context:
            was_cancelled = not success
            self._context.was_cancelled = was_cancelled
            self._context.__exit__(None, None, None)

        self._mask_refresh_active = False
        if success and self._current_series is not None and self._current_series.has_pending_mask_changes():
            self._schedule_mask_refresh()

    def _on_barcode_finished(self, success: bool, error_msg: str) -> None:
        """Handle barcode detection completion."""
        # Determine if we should continue with next operation
        should_continue_to_preprocess = (
            success and 
            hasattr(self, '_auto_process_pending_preprocess') and 
            self._auto_process_pending_preprocess
        )
        
        if success:
            # Tree refresh preserves selection without emitting a new selection
            # signal, so explicitly refresh the selected photo and barcode overlay.
            self._image_tree.refresh()
            if (self._workflow_bar.get_current_step() == WorkflowStep.LOAD
                    and self._current_image is not None):
                self._display_image(self._current_image)

            # Set up continuation if needed
            if should_continue_to_preprocess:
                self._auto_process_pending_preprocess = False
                # Note: _auto_process_pending_tracking already set correctly in _detect_barcodes_in_group
                if self._context:
                    self._context.next_operation = self._continue_auto_preprocess
        elif error_msg and error_msg != "Cancelled":
            QMessageBox.critical(self, "Error", f"Barcode detection failed:\n{error_msg}")
        
        # Cleanup and exit context (handles UI unlock, progress hide, etc.)
        if self._context:
            was_cancelled = not success
            self._context.was_cancelled = was_cancelled
            self._context.__exit__(None, None, None)

    def _auto_process_for_tracking(self, series: 'ImageSeries') -> None:
        """
        Auto-process a series for tracking (barcode detection → preprocessing → tracking).

        Called when selecting an image/group in TRACK step with auto-preview enabled.
        Uses continuation pattern with completion handlers to chain operations.

        Args:
            series: ImageSeries to process.
        """
        if self._pipeline is None:
            return

        self._processing_series = series  # Store for continuations
        self._auto_process_pending_preprocess = False
        self._auto_process_pending_tracking = False

        # Step 1: Check if barcodes need detection (if enabled)
        if (self._config.data.detect_barcodes and
            not all(img.barcode_detected for img in series.images)):
            # Need barcode detection, then preprocessing, then tracking
            self._auto_process_pending_preprocess = True
            self._auto_process_pending_tracking = True
            self._start_barcode_detection(series)
            return

        # Step 2: Check if preprocessing is needed
        preprocess_hash = self._config.preprocess_config_hash()
        state = series.pipeline_state
        if not state.preprocessed or state.preprocess_config_hash != preprocess_hash:
            # Need preprocessing, then tracking
            self._auto_process_pending_tracking = True
            self._preprocess_group(series, force=True, hide_progress=False)
            return

        # Step 3: Check if tracking is needed
        if self._pipeline.is_tracking_current(series):
            # Already tracked with current config - just display
            if self._current_image:
                self._display_image(self._current_image)
            return

        # Step 4: Run tracking
        self._start_tracking(series)

    def _start_barcode_detection(self, series: 'ImageSeries') -> None:
        """Start barcode detection in worker thread."""
        self._begin_operation_progress('barcode')
        self._hide_progress_on_complete = False

        # Create and enter processing context
        self._context = ProcessingContext(self, ProcessingState.BARCODE_DETECTION, series)
        self._context.__enter__()

        self._worker = ProcessWorker(self._pipeline, series, self._config, 'barcode')
        self._worker.phase.connect(self._on_worker_phase)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_barcode_finished)
        self._worker.start()

    def _continue_auto_preprocess(self) -> None:
        """Continue auto-processing with preprocessing step."""
        if hasattr(self, '_processing_series'):
            self._preprocess_group(self._processing_series, force=True, hide_progress=False)

    def _continue_auto_track(self) -> None:
        """Continue auto-processing with tracking step."""
        if hasattr(self, '_processing_series'):
            self._start_tracking(self._processing_series)

    def _start_tracking(self, series: 'ImageSeries') -> None:
        """Start tracking in worker thread."""
        self._mask_image_published = None
        self._begin_operation_progress('track')
        self._hide_progress_on_complete = True

        # Create and enter processing context
        self._context = ProcessingContext(self, ProcessingState.TRACKING, series)
        self._context.__enter__()

        self._worker = ProcessWorker(self._pipeline, series, self._config, 'track')
        self._worker.persist_mask = self._mask_refresh_active
        self._worker.image_ready.connect(self._on_tracking_image_ready)
        self._worker.phase.connect(self._on_worker_phase)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_tracking_finished)
        self._worker.start()
    
    def _lock_ui(self) -> None:
        """Lock UI during processing. Called by ProcessingContext.__enter__."""
        self._roi_editor.set_locked(True)
        allow_mask = self._mask_refresh_active and self._state == ProcessingState.TRACKING
        self._image_viewer.set_mask_editing_enabled(allow_mask)
        self._refresh_tree_status()
        # Disable main interactive elements
        self._image_tree.setEnabled(False)
        self._settings_panel.set_processing(True, allow_mask=allow_mask)
        self._workflow_bar.setEnabled(False)
        
        # Next button: change to Cancel during processing
        self._next_step_btn.setText("Cancel")
        self._next_step_btn.setEnabled(True)
        self._next_step_btn.setToolTip("Cancel current processing")
        
        # Zoom controls: always enabled when images are loaded (independent of lock state)
        has_images = bool(self._series_dict)
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)

        # Menu items
        if hasattr(self, '_export_action'):
            self._export_action.setEnabled(False)
        
        # Set wait cursor
        if not allow_mask and QApplication.overrideCursor() is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    
    def _unlock_ui(self, restore_step: bool = False) -> None:
        """Unlock UI after processing. Called by ProcessingContext.__exit__."""
        # Restore all cursor overrides
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        
        # Re-enable main interactive elements
        self._image_tree.setEnabled(True)
        self._settings_panel.set_processing(False)
        self._workflow_bar.setEnabled(True)
        self._roi_editor.set_locked(False)
        self._image_viewer.set_mask_editing_enabled(True)
        
        # Handle step restoration if cancelled
        if restore_step and self._step_before_processing is not None:
            # Temporarily disconnect signal to avoid double _on_step_changed call
            self._workflow_bar.step_changed.disconnect(self._on_step_changed)
            self._workflow_bar.set_current_step(self._step_before_processing)
            self._workflow_bar.step_changed.connect(self._on_step_changed)
            
            # Update UI panels manually without triggering auto-processing
            step = self._step_before_processing
            self._settings_panel.show()
            self._restore_settings_draft(step)
            
            # Update tree filtering
            self._image_tree.set_filter_aside(step != WorkflowStep.LOAD)
            
            # Display current image if available
            if self._current_image:
                self._display_image(self._current_image)
        
        # Clear saved step
        self._step_before_processing = None
        
        # Restore button state based on current step
        self._update_process_button_states()
        
        # Zoom controls: enabled if images are loaded
        has_images = bool(self._series_dict)
        self._fit_btn.setEnabled(has_images)
        self._zoom_in_btn.setEnabled(has_images)
        self._zoom_out_btn.setEnabled(has_images)
        
        # Menu items
        if hasattr(self, '_export_action'):
            self._export_action.setEnabled(True)

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
        # Set cancellation flags
        self._cancel_requested.set()
        if self._context:
            self._context.was_cancelled = True
        
        # Show cancelling status
        self._progress_bar.hide()
        self._processing_label.setText("Cancelling...")
        self._processing_label.show()
        QApplication.processEvents()
        
        # Cancel worker thread if one is running
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        
        # Cancel batch executor if one is running
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            
    # Removed _force_unlock_ui - replaced by unified _unlock_ui
    
    def _ensure_series_loaded(self, series: 'ImageSeries') -> None:
        """Reload series arrays from disk cache if they were freed."""
        if series is None:
            return
        with self._cache_lock:
            sample = series.images[0] if series.images else None
            if sample and sample.image is None and series.pipeline_state.preprocessed:
                series_cache.load_series(series, self._config)

    @staticmethod
    def _free_series_arrays(series: 'ImageSeries') -> None:
        """Free heavy image arrays from a series to reduce RAM usage."""
        # Note: Caller must hold _cache_lock when calling this method
        for img in series.images:
            img.image = None
            img.process = None
            img.canny = None
            img.diff = None
            img.image_annotated = None
            img.colored_samples = {}
            img.rsml_samples = None
            img.rsml_background = None

    # Removed _is_operation_running - replaced by checking _state != ProcessingState.IDLE
    
    def _refresh_tree_status(self) -> None:
        """Keep the tree in sync without rebuilding navigation rows."""
        self._image_tree.set_step(self._workflow_bar.get_current_step(), self._config)
        operation_step = {
            ProcessingState.BARCODE_DETECTION: WorkflowStep.LOAD,
            ProcessingState.PREPROCESSING: WorkflowStep.PREPROCESS,
            ProcessingState.TRACKING: WorkflowStep.TRACK,
            ProcessingState.BATCH_PREPROCESS: WorkflowStep.PREPROCESS,
            ProcessingState.BATCH_TRACK: WorkflowStep.TRACK,
        }.get(self._state)
        series = self._context.series if self._context else None
        self._image_tree.set_processing(series, operation_step)

    def _update_process_button_states(self) -> None:
        """Update UI states based on current step and processing status."""
        self._refresh_tree_status()
        step = self._workflow_bar.get_current_step()
        
        # Next/Export button: Don't override if any operation is running
        has_images = bool(self._series_dict)
        if self._state == ProcessingState.IDLE:
            if step == WorkflowStep.TRACK:
                self._next_step_btn.setText("Export")
                self._next_step_btn.setStyleSheet("")
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
            
        # Update cache action state as well (processing might have created cache)
        self._update_cache_action_state()

        # Refresh the "Apply to all groups" button enabled state
        self._refresh_apply_all_state()

    def _refresh_apply_all_state(self) -> None:
        """Refresh the settings panel's 'Apply to all groups' button.

        Enabled whenever applying the current settings would change at least one
        other group; disabled only when every other group already matches the
        current settings for the active step.
        """
        if self._pipeline is None or not self._series_dict:
            self._settings_panel.set_groups_out_of_sync(False)
            return

        step = self._workflow_bar.get_current_step()
        others = [
            s for s in self._series_dict.values()
            if s is not self._current_series
        ]

        if step == WorkflowStep.PREPROCESS:
            current_hash = self._config.preprocess_config_hash()
            out_of_sync = any(
                not (s.pipeline_state.preprocessed
                     and s.pipeline_state.preprocess_config_hash == current_hash)
                for s in others
            )
        elif step == WorkflowStep.TRACK:
            out_of_sync = any(
                not self._pipeline.is_tracking_current(s) for s in others
            )
        else:
            out_of_sync = False

        self._settings_panel.set_groups_out_of_sync(out_of_sync)

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
    
    def _track_all_groups(self) -> bool:
        """Track all untracked groups in parallel using all CPU cores.

        Saves caches and frees arrays before submitting, so workers
        reload from disk. Progress bar tracks completed groups.
        
        Returns:
            True if completed successfully, False if cancelled.
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
        with self._cache_lock:
            for series in untracked:
                if series.images and series.images[0].image is not None:
                    series_cache.save_series(series, self._config)
                self._free_series_arrays(series)

        # Create processing context for batch operation
        for series in untracked:
            series.pipeline_state.invalidate_from('track')
        self._context = ProcessingContext(self, ProcessingState.BATCH_TRACK, untracked)
        self._context.__enter__()

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
            self._executor = ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx)
            executor = self._executor
            
            try:
                # Pass queue to workers
                futures = {
                    executor.submit(track_and_cache_worker, (self._config, s, progress_queue)): s
                    for s in untracked
                }

                while futures:
                    # Check cancellation
                    if self._cancel_requested.is_set():
                        # Shutdown executor immediately without waiting
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._processing_label.setText("Cancelled")
                        self._progress_bar.hide()  # Immediately hide to avoid "Finishing up..." flash
                        QApplication.processEvents()
                        time.sleep(0.3)
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
                            series_cache.load_series_state(series, self._config)
                            self._image_tree.refresh_status()
                        except Exception as e:
                            print(f"Error tracking {series.group}: {e}")
                        self._context.series = list(futures.values())
                        self._refresh_tree_status()

                    # Update ETA even if no progress, to handle coasting
                    if completed_images > 0:
                        self._update_progress_label(start_time, completed_images, total_images)

                    if futures:
                        time.sleep(0.05)
            finally:
                self._executor = None
                executor.shutdown(wait=False)

            # Only proceed if not cancelled
            if not self._cancel_requested.is_set():
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
            # Exit processing context (handles cleanup)
            was_cancelled = self._cancel_requested.is_set()
            if self._context:
                self._context.was_cancelled = was_cancelled
                self._context.__exit__(None, None, None)
        
        return not was_cancelled

    def _on_track_roots(self) -> None:
        """Run root tracking on current group using worker thread."""
        if self._pipeline is None or self._current_series is None:
            QMessageBox.warning(self, "Warning", "Please preprocess images first.")
            return
        
        # Delegate to _start_tracking which uses ProcessWorker
        self._start_tracking(self._current_series)
    
    def _on_replace_rsml(self, image: ImageData) -> None:
        if self._state != ProcessingState.IDLE:
            return
        series = next((series for series in self._series_dict.values()
                       if any(candidate is image for candidate in series)), None)
        if series is None:
            return
        from ..io.rsml import read_rsml
        from ..io.rsml_replacement import replace_roots, restore_measurements
        filename, _ = QFileDialog.getOpenFileName(
            self, f"Replace roots for {image.filename} with RSML", "", "RSML Files (*.rsml);;All Files (*)")
        if not filename:
            return
        try:
            document = read_rsml(filename)
            self._ensure_series_loaded(series)
            replace_roots(image, document)
            restore_measurements(series)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "RSML Replacement Failed", str(exc))
            return
        self._image_tree.refresh_status()
        if self._current_image is image:
            self._display_image(image, preserve_view=True)
        self._status_bar.showMessage(f"{image.filename}: roots and measurements replaced with RSML", 6000)

    def _on_export_rsml(self, image: ImageData) -> None:
        if self._state != ProcessingState.IDLE or self._pipeline is None:
            return
        from .dialogs.rsml_dialog import RSMLExportDialog
        match = next(((series, index)
                      for series in self._series_dict.values()
                      for index, candidate in enumerate(series.images)
                      if candidate is image), None)
        if match is None:
            return
        series, index = match
        if image.rsml_document is None and not self._pipeline.is_tracking_current(series):
            QMessageBox.warning(self, "Tracking Required", "Track this image's series before exporting RSML.")
            return
        destination = QFileDialog.getExistingDirectory(self, f"Export to RSML: {image.filename}")
        if destination:
            with ProcessingContext(self, ProcessingState.RSML_EXPORT):
                RSMLExportDialog([series], deepcopy(self._config), destination,
                                 self, image_index=index).exec()

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

            self._settings_panel._mask_dirty = self._current_series.has_pending_mask_changes()
            self._settings_panel._on_setting_changed()
            self._schedule_mask_refresh()

    def _on_mask_erase_all(self) -> None:
        """Stage clearing the mask; Apply commits it and Discard restores it."""
        series = self._current_series
        if series is None:
            return
        mask = series.working_mask if series.working_mask is not None else series.user_mask
        if mask is None:
            return
        series.working_mask = np.zeros_like(mask)
        self._image_viewer.set_mask_data(series.user_mask, series.working_mask)
        self._settings_panel._mask_dirty = series.has_pending_mask_changes()
        self._settings_panel._on_setting_changed()
        self._schedule_mask_refresh()

    def _schedule_mask_refresh(self):
        if self._auto_apply_enabled():
            # Strokes notify on release, so there is no ongoing slider/drag to
            # debounce. Keep the normal delay only when a worker is already busy.
            self._auto_apply_timer.stop()
            QTimer.singleShot(0, self._apply_auto_settings)

    def _load_last_folder(self) -> None:
        """Load the last opened folder if it exists."""
        last_folder = self._settings.value("last_folder", None)
        if last_folder and os.path.exists(last_folder):
            # Check if folder contains images
            try:
                files = os.listdir(last_folder)
                has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
                if has_images:
                    # Restore folder path and settings
                    self._config.data.input = last_folder
                    
                    # Restore filename template if saved
                    saved_template = self._settings.value("filename_template", None)
                    if saved_template:
                        self._config.data.filename_template = saved_template
                    
                    # Restore date format if saved
                    saved_format = self._settings.value("date_format", None)
                    if saved_format:
                        self._config.data.date_format = saved_format
                    
                    # Restore detect_barcodes if saved
                    saved_detect = self._settings.value("detect_barcodes", None)
                    if saved_detect is not None:
                        # QSettings may return string "true"/"false", convert to bool
                        if isinstance(saved_detect, str):
                            self._config.data.detect_barcodes = saved_detect.lower() == "true"
                        else:
                            self._config.data.detect_barcodes = bool(saved_detect)
                        # Update menu action to match loaded setting
                        self._detect_barcodes_action.setChecked(self._config.data.detect_barcodes)
                    
                    self._reload_images()
            except Exception:
                pass  # Silently ignore if we can't load the last folder
    
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
        """Handle Next/Export/Cancel button click."""
        # If processing, cancel it
        if self._state != ProcessingState.IDLE:
            self._on_cancel_prediction()
            return
            
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

        # Save current step before navigating (in case we need to cancel back)
        self._step_before_processing = current
        
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
            success = self._track_all_groups()
            
            # If cancelled, don't proceed to export
            if not success:
                return
        
        # Export with ProcessingContext
        self._context = ProcessingContext(self, ProcessingState.IDLE)  # Use IDLE since this is just export
        self._context.__enter__()
        
        self._processing_label.setText("Exporting...")
        self._processing_label.show()
        self._progress_bar.setRange(0, 0)  # Indeterminate progress
        self._progress_bar.show()
        QApplication.processEvents()

        try:
            # Export statistics
            stats_df = self._pipeline.export_statistics(self._series_dict)
            
            # Check if cancelled during export
            if self._cancel_requested.is_set():
                self._processing_label.setText("Cancelled")
                self._progress_bar.hide()
                QApplication.processEvents()
                time.sleep(0.3)
                return
            
            stats_df.to_csv(file_path, index=False)
            
            self._processing_label.hide()
            self._progress_bar.hide()
            
            QMessageBox.information(
                self, 
                "Export Complete",
                f"Results exported to:\n{file_path}"
            )
        except Exception as e:
            self._processing_label.hide()
            self._progress_bar.hide()
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")
        finally:
            # Exit processing context (handles cleanup)
            was_cancelled = self._cancel_requested.is_set()
            if self._context:
                self._context.was_cancelled = was_cancelled
                self._context.__exit__(None, None, None)
