"""
Settings panel for configuration editing.

Context-sensitive panel that shows settings relevant to the current workflow step.
"""

from enum import Enum
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QCheckBox, QLineEdit
)
from PySide6.QtCore import Signal, Qt

from ..config import Config
from .workflow_bar import WorkflowStep


class SettingsScope(Enum):
    """Scope for applying settings."""
    THIS_IMAGE = "This Image"
    THIS_GROUP = "This Group"
    ALL = "All Images"


class SettingsPanel(QWidget):
    """
    Context-sensitive settings panel.
    
    Shows different settings based on the current workflow step.
    Settings can be applied to individual images, groups, or all.
    
    Signals:
        settings_changed: Emitted when any setting changes.
        apply_requested: Emitted when user clicks Apply (scope: SettingsScope).
    """
    
    settings_changed = Signal()
    apply_requested = Signal(SettingsScope)
    load_requested = Signal()  # Emitted when user clicks Load in step 1
    preprocess_requested = Signal()  # Emitted when user clicks Preprocess in step 2
    track_requested = Signal()  # Emitted when user clicks Track in step 3
    export_requested = Signal()  # Emitted when user clicks Export in step 4
    
    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._config = config
        self._current_step = WorkflowStep.LOAD
        
        self._setup_ui()
        self._update_for_step(WorkflowStep.LOAD)
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header = QLabel("Settings")
        header.setStyleSheet("""
            QLabel {
                font-weight: bold;
                padding: 5px;
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
        """)
        self._main_layout.addWidget(header)
        
        # Settings container (will be replaced per step)
        self._settings_container = QWidget()
        self._settings_layout = QVBoxLayout(self._settings_container)
        self._main_layout.addWidget(self._settings_container)
        
        # Scope selector and apply button
        self._scope_widget = QWidget()
        scope_layout = QHBoxLayout(self._scope_widget)
        scope_layout.setContentsMargins(0, 10, 0, 0)
        
        scope_layout.addWidget(QLabel("Apply to:"))
        
        self._scope_combo = QComboBox()
        for scope in SettingsScope:
            self._scope_combo.addItem(scope.value, scope)
        scope_layout.addWidget(self._scope_combo)
        
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        scope_layout.addWidget(self._apply_btn)
        
        scope_layout.addStretch()
        self._main_layout.addWidget(self._scope_widget)
        
        self._main_layout.addStretch()
    
    def set_step(self, step: WorkflowStep) -> None:
        """
        Update panel for the given workflow step.
        
        Args:
            step: The current workflow step.
        """
        self._current_step = step
        self._update_for_step(step)
    
    def _update_for_step(self, step: WorkflowStep) -> None:
        """Rebuild settings widgets for the given step."""
        # Clear existing settings
        while self._settings_layout.count():
            item = self._settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        if step == WorkflowStep.LOAD:
            self._create_load_settings()
        elif step == WorkflowStep.PREPROCESS:
            self._create_preprocess_settings()
        elif step == WorkflowStep.TRACK:
            self._create_track_settings()
        elif step == WorkflowStep.EXPORT:
            self._create_export_settings()
    
    def _create_load_settings(self) -> None:
        """Create settings for Load step."""
        group = QGroupBox("Data Settings")
        layout = QFormLayout(group)
        
        self._input_path = QLineEdit(self._config.data.input)
        layout.addRow("Input folder:", self._input_path)
        
        self._filename_template = QLineEdit(self._config.data.filename_template)
        layout.addRow("Filename template:", self._filename_template)
        
        self._date_format = QLineEdit(self._config.data.date_format)
        layout.addRow("Date format:", self._date_format)
        
        self._settings_layout.addWidget(group)
        
        # Load button
        self._load_btn = QPushButton("Load Images")
        self._load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a80c9;
            }
        """)
        self._load_btn.clicked.connect(self.load_requested.emit)
        self._settings_layout.addWidget(self._load_btn)
        
        # Hide scope selector for load step
        self._scope_widget.hide()
    
    def _create_preprocess_settings(self) -> None:
        """Create settings for Preprocess step."""
        # Image settings (per-image)
        img_group = QGroupBox("Image Settings (per image)")
        img_layout = QFormLayout(img_group)
        
        self._rotation_spin = QSpinBox()
        self._rotation_spin.setRange(0, 360)
        self._rotation_spin.setSingleStep(90)
        self._rotation_spin.setValue(self._config.rotation)
        self._rotation_spin.valueChanged.connect(self.settings_changed.emit)
        img_layout.addRow("Rotation (°):", self._rotation_spin)
        
        self._settings_layout.addWidget(img_group)
        
        # Group settings (per-group)
        grp_group = QGroupBox("Group Settings (per group)")
        grp_layout = QFormLayout(grp_group)
        
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(1, 20)
        self._n_clusters_spin.setValue(self._config.n_clusters)
        self._n_clusters_spin.valueChanged.connect(self.settings_changed.emit)
        grp_layout.addRow("Number of plants:", self._n_clusters_spin)
        
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.0, 0.5)
        self._margin_spin.setSingleStep(0.01)
        self._margin_spin.setValue(self._config.margin)
        self._margin_spin.valueChanged.connect(self.settings_changed.emit)
        grp_layout.addRow("Side margin:", self._margin_spin)
        
        self._settings_layout.addWidget(grp_group)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self._preprocess_current_btn = QPushButton("Preprocess This Image")
        self._preprocess_current_btn.clicked.connect(lambda: self.preprocess_requested.emit())
        btn_layout.addWidget(self._preprocess_current_btn)
        
        self._preprocess_all_btn = QPushButton("Preprocess All in Group")
        self._preprocess_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a80c9;
            }
        """)
        self._preprocess_all_btn.clicked.connect(lambda: self.apply_requested.emit(SettingsScope.THIS_GROUP))
        btn_layout.addWidget(self._preprocess_all_btn)
        
        self._settings_layout.addLayout(btn_layout)
        
        self._scope_widget.show()
    
    def _create_track_settings(self) -> None:
        """Create settings for Track step."""
        group = QGroupBox("Tracking Settings")
        layout = QFormLayout(group)
        
        self._contour_filter_spin = QSpinBox()
        self._contour_filter_spin.setRange(1, 1000)
        self._contour_filter_spin.setValue(self._config.contour_filter)
        self._contour_filter_spin.valueChanged.connect(self.settings_changed.emit)
        layout.addRow("Min contour size:", self._contour_filter_spin)
        
        self._distance_filter_spin = QSpinBox()
        self._distance_filter_spin.setRange(1, 500)
        self._distance_filter_spin.setValue(self._config.distance_filter)
        self._distance_filter_spin.valueChanged.connect(self.settings_changed.emit)
        layout.addRow("Max link distance:", self._distance_filter_spin)
        
        self._settings_layout.addWidget(group)
        
        # Track button
        self._track_btn = QPushButton("Run Root Tracking")
        self._track_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a80c9;
            }
        """)
        self._track_btn.clicked.connect(self.track_requested.emit)
        self._settings_layout.addWidget(self._track_btn)
        
        self._scope_widget.show()
    
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
        
        self._scope_widget.hide()
    
    def _on_apply_clicked(self) -> None:
        """Handle Apply button click."""
        scope = self._scope_combo.currentData()
        self.apply_requested.emit(scope)
    
    def get_current_values(self) -> dict:
        """
        Get current setting values.
        
        Returns:
            Dictionary of setting name to value.
        """
        values = {}
        
        if self._current_step == WorkflowStep.LOAD:
            values["input"] = self._input_path.text()
            values["filename_template"] = self._filename_template.text()
            values["date_format"] = self._date_format.text()
        
        elif self._current_step == WorkflowStep.PREPROCESS:
            values["rotation"] = self._rotation_spin.value()
            values["n_clusters"] = self._n_clusters_spin.value()
            values["margin"] = self._margin_spin.value()
        
        elif self._current_step == WorkflowStep.TRACK:
            values["contour_filter"] = self._contour_filter_spin.value()
            values["distance_filter"] = self._distance_filter_spin.value()
        
        elif self._current_step == WorkflowStep.EXPORT:
            values["output"] = self._output_path.text()
        
        return values
    
    def set_rotation(self, value: int) -> None:
        """Set rotation value (for per-image override display)."""
        if hasattr(self, "_rotation_spin"):
            self._rotation_spin.blockSignals(True)
            self._rotation_spin.setValue(value)
            self._rotation_spin.blockSignals(False)
