"""
Settings panel for configuration editing.

Context-sensitive panel that shows settings relevant to the current workflow step.
Settings are applied to the entire group on Apply button click.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QLineEdit
)
from PySide6.QtCore import Signal

from ..config import Config
from .workflow_bar import WorkflowStep


class SettingsPanel(QWidget):
    """
    Context-sensitive settings panel.
    
    Shows different settings based on the current workflow step.
    Settings are applied to the entire group when Apply is clicked.
    Changes are discarded when switching groups without applying.
    
    Signals:
        apply_requested: Emitted when user clicks Apply.
        redetect_requested: Emitted when user clicks Re-detect Plants.
        track_requested: Emitted for Track step action.
        export_requested: Emitted for Export step action.
    """
    
    apply_requested = Signal()  # Apply settings to current group
    redetect_requested = Signal()  # Re-run centroid detection
    track_requested = Signal()
    export_requested = Signal()
    
    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._config = config
        self._current_step = WorkflowStep.LOAD
        self._original_values: dict = {}  # Stored values at group selection
        self._is_dirty = False
        self._centroids_modified = False  # True if user moved centroids
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        header_label = QLabel("Settings")
        header_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        self._status_indicator = QLabel("")
        self._status_indicator.setStyleSheet("font-size: 11px;")
        header_layout.addWidget(self._status_indicator)
        
        header_widget.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
        """)
        self._main_layout.addWidget(header_widget)
        
        # Settings container (rebuilt per step)
        self._settings_container = QWidget()
        self._settings_layout = QVBoxLayout(self._settings_container)
        self._main_layout.addWidget(self._settings_container)
        
        # Bottom buttons (Apply and Re-detect)
        self._buttons_widget = QWidget()
        self._buttons_layout = QVBoxLayout(self._buttons_widget)
        self._buttons_layout.setContentsMargins(0, 10, 0, 0)
        
        # Re-detect Plants button (only for Preprocess step)
        self._redetect_btn = QPushButton("Re-detect Plants")
        self._redetect_btn.setEnabled(False)
        self._redetect_btn.setToolTip("Click after moving centroids to re-run auto-detection")
        self._redetect_btn.clicked.connect(self._on_redetect_clicked)
        self._buttons_layout.addWidget(self._redetect_btn)
        
        # Apply button
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip("No changes to apply")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        self._buttons_layout.addWidget(self._apply_btn)
        
        self._main_layout.addWidget(self._buttons_widget)
        self._main_layout.addStretch()
    
    def set_step(self, step: WorkflowStep) -> None:
        """Update panel for the given workflow step."""
        self._current_step = step
        self._rebuild_for_step(step)
    
    def _rebuild_for_step(self, step: WorkflowStep) -> None:
        """Rebuild settings widgets for the given step."""
        # Clear existing settings
        while self._settings_layout.count():
            item = self._settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if step == WorkflowStep.PREPROCESS:
            self._create_preprocess_settings()
            self._buttons_widget.show()
            self._redetect_btn.show()
        elif step == WorkflowStep.TRACK:
            self._create_track_settings()
            self._buttons_widget.show()
            self._redetect_btn.hide()
        elif step == WorkflowStep.EXPORT:
            self._create_export_settings()
            self._buttons_widget.hide()
        else:
            self._buttons_widget.hide()
        
        # Reset state
        self._mark_clean()
    
    def _create_preprocess_settings(self) -> None:
        """Create settings for Preprocess step."""
        group = QGroupBox("Group Settings")
        layout = QFormLayout(group)
        
        self._rotation_spin = QSpinBox()
        self._rotation_spin.setRange(0, 360)
        self._rotation_spin.setSingleStep(90)
        self._rotation_spin.setValue(self._config.rotation)
        self._rotation_spin.valueChanged.connect(self._on_setting_changed)
        layout.addRow("Rotation (°):", self._rotation_spin)
        
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(1, 20)
        self._n_clusters_spin.setValue(self._config.n_clusters)
        self._n_clusters_spin.valueChanged.connect(self._on_setting_changed)
        layout.addRow("Number of plants:", self._n_clusters_spin)
        
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.0, 0.5)
        self._margin_spin.setSingleStep(0.01)
        self._margin_spin.setValue(self._config.margin)
        self._margin_spin.valueChanged.connect(self._on_setting_changed)
        layout.addRow("Side margin:", self._margin_spin)
        
        self._settings_layout.addWidget(group)
        
        # Store original values
        self._store_original_values()
    
    def _create_track_settings(self) -> None:
        """Create settings for Track step."""
        group = QGroupBox("Tracking Settings")
        layout = QFormLayout(group)
        
        self._min_contour_area_spin = QSpinBox()
        self._min_contour_area_spin.setRange(1, 1000)
        self._min_contour_area_spin.setValue(self._config.threshold.min_contour_area)
        self._min_contour_area_spin.valueChanged.connect(self._on_setting_changed)
        layout.addRow("Min contour area:", self._min_contour_area_spin)
        
        self._min_contour_length_spin = QSpinBox()
        self._min_contour_length_spin.setRange(1, 500)
        self._min_contour_length_spin.setValue(self._config.threshold.min_contour_length)
        self._min_contour_length_spin.valueChanged.connect(self._on_setting_changed)
        layout.addRow("Min contour length:", self._min_contour_length_spin)
        
        self._settings_layout.addWidget(group)
        
        # Track button
        self._track_btn = QPushButton("Run Root Tracking")
        self._track_btn.clicked.connect(self.track_requested.emit)
        self._settings_layout.addWidget(self._track_btn)
        
        self._store_original_values()
    
    def _create_export_settings(self) -> None:
        """Create settings for Export step."""
        group = QGroupBox("Export Settings")
        layout = QFormLayout(group)
        
        self._output_path = QLineEdit(self._config.data.output)
        layout.addRow("Output folder:", self._output_path)
        
        self._settings_layout.addWidget(group)
        
        # Export button
        export_btn = QPushButton("Export to CSV")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #5cb85c;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #449d44;
            }
        """)
        export_btn.clicked.connect(self.export_requested.emit)
        self._settings_layout.addWidget(export_btn)
    
    def _store_original_values(self) -> None:
        """Store current values as original (for dirty detection)."""
        self._original_values = self.get_current_values()
        self._mark_clean()
    
    def _on_setting_changed(self) -> None:
        """Handle any setting value change."""
        current = self.get_current_values()
        self._is_dirty = current != self._original_values
        self._update_apply_button()
    
    def _update_apply_button(self) -> None:
        """Update Apply button enabled state and tooltip."""
        if self._is_dirty:
            self._apply_btn.setEnabled(True)
            self._apply_btn.setToolTip("Apply changes to current group")
            self._status_indicator.setText("● Modified")
            self._status_indicator.setStyleSheet("color: #f0ad4e; font-size: 11px;")
        else:
            self._apply_btn.setEnabled(False)
            self._apply_btn.setToolTip("No changes to apply")
            self._status_indicator.setText("")
    
    def _mark_clean(self) -> None:
        """Mark settings as clean (no pending changes)."""
        self._is_dirty = False
        self._update_apply_button()
    
    def _on_apply_clicked(self) -> None:
        """Handle Apply button click."""
        self._store_original_values()  # New baseline
        self.apply_requested.emit()
    
    def _on_redetect_clicked(self) -> None:
        """Handle Re-detect Plants button click."""
        self._centroids_modified = False
        self._redetect_btn.setEnabled(False)
        self._redetect_btn.setToolTip("Click after moving centroids to re-run auto-detection")
        self.redetect_requested.emit()
    
    def enable_redetect(self) -> None:
        """Enable Re-detect button (called when user moves a centroid)."""
        self._centroids_modified = True
        self._redetect_btn.setEnabled(True)
        self._redetect_btn.setToolTip("Re-run plant centroid detection")
    
    def mark_centroids_modified(self) -> None:
        """Mark that centroids have been modified (enable Re-detect button)."""
        self.enable_redetect()
    
    def reset_for_group(self) -> None:
        """Reset settings for a new group (discard unsaved changes)."""
        if self._current_step == WorkflowStep.PREPROCESS and hasattr(self, '_rotation_spin'):
            self._rotation_spin.blockSignals(True)
            self._rotation_spin.setValue(self._config.rotation)
            self._rotation_spin.blockSignals(False)
            
            self._n_clusters_spin.blockSignals(True)
            self._n_clusters_spin.setValue(self._config.n_clusters)
            self._n_clusters_spin.blockSignals(False)
            
            self._margin_spin.blockSignals(True)
            self._margin_spin.setValue(self._config.margin)
            self._margin_spin.blockSignals(False)
        
        self._store_original_values()
        self._centroids_modified = False
        self._redetect_btn.setEnabled(False)
    
    def get_current_values(self) -> dict:
        """Get current setting values."""
        values = {}
        
        if self._current_step == WorkflowStep.PREPROCESS:
            if hasattr(self, '_rotation_spin'):
                values["rotation"] = self._rotation_spin.value()
                values["n_clusters"] = self._n_clusters_spin.value()
                values["margin"] = self._margin_spin.value()
        
        elif self._current_step == WorkflowStep.TRACK:
            if hasattr(self, '_min_contour_area_spin'):
                values["min_contour_area"] = self._min_contour_area_spin.value()
                values["min_contour_length"] = self._min_contour_length_spin.value()
        
        elif self._current_step == WorkflowStep.EXPORT:
            if hasattr(self, '_output_path'):
                values["output"] = self._output_path.text()
        
        return values
    
    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._is_dirty
