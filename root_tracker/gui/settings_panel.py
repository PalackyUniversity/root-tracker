"""
Settings panel for configuration editing.

Context-sensitive panel that shows settings relevant to the current workflow step.
Settings are applied to the entire group on Apply button click.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QGroupBox, QLineEdit, QComboBox, QCheckBox, QButtonGroup, QRadioButton, QScrollArea, QFrame
)
from PySide6.QtCore import Signal, Qt

from ..config import Config
from .workflow_bar import WorkflowStep
from .masking_tools import MaskTool
from .color_range import ColorRangeControl
from .right_checkbox import RightAlignedCheckBox


class SettingsPanel(QWidget):
    """
    Context-sensitive settings panel.
    
    Shows different settings based on the current workflow step.
    Settings are applied to the entire group when Apply is clicked.
    The main window retains per-group drafts until applied or discarded.
    
    Signals:
        apply_requested: Emitted when user clicks Apply.
        redetect_requested: Emitted when user clicks Re-detect Plants.
        track_requested: Emitted for Track step action.
    """

    crop_edit_toggled = Signal(bool)
    crop_preview_changed = Signal()
    color_preview_toggled = Signal(bool)
    color_pick_toggled = Signal(bool)
    discard_requested = Signal()
    dirty_changed = Signal(bool)
    apply_requested = Signal()  # Apply settings to current group
    apply_all_requested = Signal()  # Apply settings to all groups
    redetect_requested = Signal()  # Re-run centroid detection
    track_requested = Signal()
    reset_auto_requested = Signal()

    # Masking signals
    mask_tool_changed = Signal(str, int)  # tool name, brush size
    mask_erase_all_requested = Signal()
    
    def __init__(self, config: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._config = config
        self._current_step = WorkflowStep.LOAD
        self._original_values: dict = {}  # Stored values at group selection
        self._is_dirty = False
        self._mask_dirty = False
        self._groups_out_of_sync = False  # True if some other group has different settings
        self._centroids_modified = False  # True if user moved centroids

        # Masking tool widgets (created in _create_track_settings)
        self._mask_tool_group: QButtonGroup | None = None
        self._brush_radio: QRadioButton | None = None
        self._brush_size_spin: QSpinBox | None = None
        self._brush_eraser_radio: QRadioButton | None = None
        self._brush_eraser_size_spin: QSpinBox | None = None
        self._move_radio: QRadioButton | None = None
        self._rect_radio: QRadioButton | None = None
        self._rect_eraser_radio: QRadioButton | None = None
        self._mask_erase_all_btn: QPushButton | None = None

        self._setup_ui()
        self.set_step(WorkflowStep.LOAD)
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Settings container (rebuilt per step)
        self._settings_container = QWidget()
        self._settings_layout = QVBoxLayout(self._settings_container)
        self._settings_scroll = QScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._settings_scroll.setWidget(self._settings_container)
        self._settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._main_layout.addWidget(self._settings_scroll, 1)
        
        # Bottom buttons (Apply and Re-detect)
        self._buttons_widget = QWidget()
        self._buttons_layout = QVBoxLayout(self._buttons_widget)
        self._buttons_layout.setContentsMargins(0, 10, 0, 0)
        
        # Re-detect Plants button is now part of Preprocess settings
        
        # Apply buttons layout
        apply_layout = QHBoxLayout()
        
        # Apply to this group button
        self._apply_btn = QPushButton("Apply settings")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setStatusTip("No changes to apply. Settings are shared: Apply previews this group; other groups use them when next processed. Masks affect this group only.")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        apply_layout.addWidget(self._apply_btn)
        
        # Apply to all groups button
        self._apply_all_btn = QPushButton("Update all groups")
        self._apply_all_btn.setEnabled(False)
        self._apply_all_btn.setStatusTip("No changes to apply")
        self._apply_all_btn.clicked.connect(self._on_apply_all_clicked)
        apply_layout.addWidget(self._apply_all_btn)
        
        self._buttons_layout.addLayout(apply_layout)
        self._discard_btn = QPushButton('Discard changes')
        self._discard_btn.setEnabled(False)
        self._discard_btn.setStatusTip('Discard pending edits for this group and step, including its tracking mask. Restore the applied settings without reprocessing.')
        self._discard_btn.clicked.connect(self.discard_requested.emit)
        self._buttons_layout.addWidget(self._discard_btn)

        self._auto_reset_btn = QPushButton('Reset')
        self._auto_reset_btn.setStatusTip('Restore the settings from before automatic edits for this group and step.')
        self._auto_reset_btn.clicked.connect(self.reset_auto_requested.emit)
        self._auto_reset_btn.hide()
        apply_layout.addWidget(self._auto_reset_btn)
        self._main_layout.addWidget(self._buttons_widget)

    
    def set_auto_apply_mode(self, enabled, can_reset=False):
        enabled = enabled and self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS)
        self._apply_btn.setVisible(not enabled)
        self._discard_btn.setVisible(not enabled)
        self._auto_reset_btn.setVisible(enabled)
        self._auto_reset_btn.setEnabled(can_reset)

    def finish_color_picker(self):
        control = getattr(self, '_color_control', None)
        if control is not None and self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            control.finish_picker()

    def set_step(self, step: WorkflowStep) -> None:
        """Update panel for the given workflow step."""
        self.finish_color_picker()
        self._current_step = step
        self._rebuild_for_step(step)
    
    def _rebuild_for_step(self, step: WorkflowStep) -> None:
        """Rebuild settings widgets for the given step."""
        # Clear existing settings
        while self._settings_layout.count():
            item = self._settings_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        
        if step == WorkflowStep.LOAD:
            self._create_crop_settings(load=True)
            self._buttons_widget.show()
            self._store_original_values()
        elif step == WorkflowStep.PREPROCESS:
            self._create_preprocess_settings()
            self._buttons_widget.show()
            # self._redetect_btn is now inside the settings layout, not buttons layout
        elif step == WorkflowStep.TRACK:
            self._create_track_settings()
            self._buttons_widget.show()
        else:
            self._buttons_widget.hide()
        
        self._install_setting_help()
        # Reset state
        self._mark_clean()
    
    def _install_setting_help(self):
        tips = {
            '_n_clusters_spin': 'Expected number of plants in each image. Green stem detections are grouped into this many plant centers. Match this to the plants in the photograph.',
            '_reg_enabled_cb': 'Align photographs in a group to the first image so roots can be compared at consistent positions over time.',
            '_reg_margin_spin': 'Search border on each side as a fraction of image size (0.25 = 25%). Larger values allow larger shifts but take longer. Smaller images are padded enough to fit the reference.',
            '_min_contour_area_spin': 'Minimum enclosed root-contour area in square pixels. Increase to reject small specks; too high can remove small roots.',
            '_min_contour_length_spin': 'Minimum number of sampled points in a root contour (not root length in millimetres). Increase to reject short contours; too high can remove fine roots.',
            '_move_radio': 'Pan the image without changing the exclusion mask. Scroll to zoom.',
            '_brush_radio': 'Paint areas to exclude from root tracking. Apply settings to use the edited mask for this group.',
            '_brush_size_spin': 'Diameter of the exclusion brush in image pixels. Larger values cover a wider area.',
            '_rect_radio': 'Drag a rectangle to exclude that area from root tracking.',
            '_brush_eraser_radio': 'Erase painted exclusions to include those areas in tracking again.',
            '_brush_eraser_size_spin': 'Diameter of the mask eraser in image pixels.',
            '_rect_eraser_radio': 'Drag a rectangle to remove exclusions inside it.',
            '_mask_erase_all_btn': 'Clear the entire exclusion mask for this group. Apply settings to use the cleared mask.',
            '_redetect_btn': 'Re-run automatic plant detection for this group, replacing manually moved centroid positions.',
        }
        active = self._settings_container.findChildren(QWidget)
        forms = self._settings_container.findChildren(QFormLayout)
        for name, tip in tips.items():
            widget = getattr(self, name, None)
            if widget not in active:
                continue
            widget.setStatusTip(tip)
            for form in forms:
                label = form.labelForField(widget)
                if label is not None:
                    label.setStatusTip(tip)

    def _create_crop_settings(self, *, load):
        group = QGroupBox('Manual crop' if load else 'Crop within plate')
        layout = QVBoxLayout(group)
        self._crop_value = self._config.load_roi if load else self._config.preprocess_roi
        self._crop_summary = QLabel()
        self._crop_summary.setMinimumHeight(20)
        layout.addWidget(self._crop_summary)
        self._crop_summary.setVisible(not load)
        self._crop_edit_btn = QPushButton('Edit crop')
        self._crop_edit_btn.setCheckable(True)
        self._crop_edit_btn.setChecked(load)
        self._crop_edit_btn.setStatusTip('Resize or move the box in the image. Drag just outside a corner or use the round handle to rotate. Rotation snaps every 15°; continue dragging to release a snap.')
        self._crop_edit_btn.toggled.connect(self._crop_edit_changed)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        buttons.addWidget(self._crop_edit_btn)
        self._auto_crop_btn = QPushButton('Reset' if load else 'Reset crop')
        self._auto_crop_btn.setStatusTip('Reset the manual crop to the whole image.' if load else 'Restore the fixed crop margins relative to the detected plate.')
        self._auto_crop_btn.clicked.connect(lambda: self.set_crop(None))
        buttons.addWidget(self._auto_crop_btn)
        self._settings_layout.addWidget(group)
        lower, upper = ((self._config.crop.blue_hsv_lower, self._config.crop.blue_hsv_upper) if load else
                        (self._config.green.hsv_lower, self._config.green.hsv_upper))
        self._color_control = ColorRangeControl('Plate search by color' if load else 'Plants', lower, upper,
                                                selector_label='Background color' if load else 'Leaves color')
        self._color_control.setStatusTip('Detect the plate by color inside the Load search box using the selected region method.' if load else 'Select the green plant parts used to locate plant centers and exclude leaves from root detection.')
        self._color_control.range_changed.connect(self._color_range_changed)
        self._color_control.preview_changed.connect(self.color_preview_toggled)
        self._color_control.pick_requested.connect(self.color_pick_toggled)
        if load:
            self._background_enabled = RightAlignedCheckBox("Enable search")
            self._background_enabled.setChecked(self._config.crop.background_enabled)
            self._background_enabled.setStatusTip('Use color to refine the crop inside the box, or use only the box itself.')
            self._color_control.layout().insertWidget(0, self._background_enabled)
            form = QFormLayout()
            self._background_region = QComboBox()
            self._background_region.addItems(['Largest matching', 'All matching'])
            self._background_region.setCurrentIndex(0 if self._config.crop.background_region == 'largest' else 1)
            self._background_region.setStatusTip('Choose the largest connected color region, or enclose all matching patches inside the box.')
            form.addRow('Region', self._background_region)
            form.labelForField(self._background_region).setStatusTip(self._background_region.statusTip())
            self._color_control.layout().insertLayout(1, form)
        self._settings_layout.addWidget(self._color_control)
        if load:
            self._background_enabled.toggled.connect(self._background_mode_changed)
            self._background_region.currentIndexChanged.connect(self._background_mode_changed)
            self._set_background_controls_enabled()
        self._update_crop_summary()
        self._crop_edit_btn.setText('Finish editing' if load else 'Edit crop')

    def _set_background_controls_enabled(self):
        enabled = self._background_enabled.isChecked()
        self._background_region.setEnabled(enabled)
        if not enabled:
            self._color_control.finish_picker()
            self._color_control.pick_button.setChecked(False)
            self._color_control.preview.setChecked(False)
        self._color_control.swatch.setEnabled(enabled)
        self._color_control.preview.setEnabled(enabled)
        self._color_control.pick_button.setEnabled(enabled)

    def _background_mode_changed(self):
        self._set_background_controls_enabled()
        self._on_setting_changed()
        self.crop_preview_changed.emit()

    def _crop_edit_changed(self, enabled):
        self._crop_edit_btn.setText('Finish editing' if enabled else 'Edit crop')
        self.crop_edit_toggled.emit(enabled)

    def _update_crop_summary(self):
        self._auto_crop_btn.setEnabled(self._crop_value is not None)
        self._crop_summary.setText(('Whole image' if self._current_step == WorkflowStep.LOAD else 'Fixed plate margins') if self._crop_value is None else
                                   f'Manual crop · {self._crop_value[4] % 360:.1f}°')

    def set_crop(self, box):
        self._crop_value = None if box is None else tuple(box)
        self._update_crop_summary()
        self._on_setting_changed()
        self.crop_preview_changed.emit()

    def _color_range_changed(self, lower, upper):
        self._color_control.preview.setChecked(True)
        self._on_setting_changed()
        self.crop_preview_changed.emit()

    def crop_editing(self):
        return self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS) and self._crop_edit_btn.isChecked()

    def _create_preprocess_settings(self) -> None:
        """Create settings for Preprocess step."""
        self._create_crop_settings(load=False)

        # Plant Attributes Group
        attr_layout = QFormLayout()
        self._color_control.layout().addLayout(attr_layout)
        
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(1, 20)
        self._n_clusters_spin.setValue(self._config.n_clusters)
        self._n_clusters_spin.valueChanged.connect(self._on_setting_changed)
        attr_layout.addRow("Origin counts:", self._n_clusters_spin)
        
        # Re-detect button inside Plant Attributes
        self._redetect_btn = QPushButton("Reset origins")
        self._redetect_btn.setEnabled(False)
        self._redetect_btn.setStatusTip("Click after moving centroids to re-run auto-detection")
        self._redetect_btn.clicked.connect(self._on_redetect_clicked)
        
        attr_layout.addRow(self._redetect_btn)
        self._redetect_btn.show()
        
        
        # Registration Group
        reg_group = QGroupBox("Registration of Images in a Group")
        reg_layout = QFormLayout(reg_group)
        
        self._reg_enabled_cb = RightAlignedCheckBox("Enable Registration")
        self._reg_enabled_cb.setChecked(self._config.registration.enabled)
        self._reg_enabled_cb.stateChanged.connect(self._on_setting_changed)
        self._reg_enabled_cb.stateChanged.connect(self._on_reg_enabled_changed)
        reg_layout.addRow(self._reg_enabled_cb)
        
        self._reg_margin_spin = QDoubleSpinBox()
        self._reg_margin_spin.setRange(0.0, 0.5)
        self._reg_margin_spin.setSingleStep(0.05)
        self._reg_margin_spin.setValue(self._config.registration.margin_ratio)
        self._reg_margin_spin.valueChanged.connect(self._on_setting_changed)
        self._reg_margin_spin.setEnabled(self._reg_enabled_cb.isChecked())
        reg_layout.addRow("Margin Ratio:", self._reg_margin_spin)
        
        self._settings_layout.addWidget(reg_group)
        
        # Store original values
        self._store_original_values()
    
    def _create_track_settings(self) -> None:
        """Create settings for Track step."""
        # Tracking settings group
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

        # Masking tools group
        mask_group = QGroupBox("Masking Tools")
        mask_layout = QVBoxLayout(mask_group)

        # Tool selection (radio buttons)
        self._mask_tool_group = QButtonGroup(self)
        self._mask_tool_group.buttonClicked.connect(self._on_mask_tool_selected)

        # Move tool (Default)
        self._move_radio = QRadioButton("Move")
        self._move_radio.setChecked(True)
        self._mask_tool_group.addButton(self._move_radio, 0)  # ID = 0
        mask_layout.addWidget(self._move_radio)

        # Brush tool
        brush_row = QHBoxLayout()
        self._brush_radio = QRadioButton("Brush")
        self._mask_tool_group.addButton(self._brush_radio, 1)  # ID = 1
        brush_row.addWidget(self._brush_radio)
        self._brush_size_spin = QSpinBox()
        self._brush_size_spin.setRange(1, 999)
        self._brush_size_spin.setValue(100)
        self._brush_size_spin.setSuffix(" px")
        self._brush_size_spin.setFixedWidth(80)
        self._brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        brush_row.addWidget(self._brush_size_spin)
        brush_row.addStretch()
        mask_layout.addLayout(brush_row)

        # Rectangle tool
        self._rect_radio = QRadioButton("Rectangle")
        self._mask_tool_group.addButton(self._rect_radio, 2)  # ID = 2
        mask_layout.addWidget(self._rect_radio)

        # Brush eraser tool
        brush_eraser_row = QHBoxLayout()
        self._brush_eraser_radio = QRadioButton("Brush Eraser")
        self._mask_tool_group.addButton(self._brush_eraser_radio, 3)  # ID = 3
        brush_eraser_row.addWidget(self._brush_eraser_radio)
        self._brush_eraser_size_spin = QSpinBox()
        self._brush_eraser_size_spin.setRange(1, 999)
        self._brush_eraser_size_spin.setValue(100)
        self._brush_eraser_size_spin.setSuffix(" px")
        self._brush_eraser_size_spin.setFixedWidth(80)
        self._brush_eraser_size_spin.valueChanged.connect(self._on_brush_eraser_size_changed)
        brush_eraser_row.addWidget(self._brush_eraser_size_spin)
        brush_eraser_row.addStretch()
        mask_layout.addLayout(brush_eraser_row)

        # Rectangle eraser tool
        self._rect_eraser_radio = QRadioButton("Rectangle Eraser")
        self._mask_tool_group.addButton(self._rect_eraser_radio, 4)  # ID = 4
        mask_layout.addWidget(self._rect_eraser_radio)

        # Erase All button
        self._mask_erase_all_btn = QPushButton("Erase All")
        self._mask_erase_all_btn.clicked.connect(self.mask_erase_all_requested.emit)
        mask_layout.addWidget(self._mask_erase_all_btn)

        self._settings_layout.addWidget(mask_group)

        self._store_original_values()

    def _on_mask_tool_selected(self, button: QRadioButton) -> None:
        """Handle mask tool selection."""
        # Determine which tool was selected based on the button
        if button == self._move_radio:
            tool = MaskTool.MOVE
            size = 0
        elif button == self._brush_radio:
            tool = MaskTool.BRUSH
            size = self._brush_size_spin.value()
        elif button == self._rect_radio:
            tool = MaskTool.RECTANGLE
            size = 0
        elif button == self._brush_eraser_radio:
            tool = MaskTool.BRUSH_ERASER
            size = self._brush_eraser_size_spin.value()
        elif button == self._rect_eraser_radio:
            tool = MaskTool.RECT_ERASER
            size = 0
        else:
            tool = MaskTool.NONE
            size = 0

        self.mask_tool_changed.emit(tool.value, size)

    def _on_brush_size_changed(self, size: int) -> None:
        """Handle brush size change."""
        # Sync with eraser size
        if self._brush_eraser_size_spin is not None:
            self._brush_eraser_size_spin.blockSignals(True)
            self._brush_eraser_size_spin.setValue(size)
            self._brush_eraser_size_spin.blockSignals(False)

        if self._brush_radio is not None and self._brush_radio.isChecked():
            self.mask_tool_changed.emit(MaskTool.BRUSH.value, size)

    def _on_brush_eraser_size_changed(self, size: int) -> None:
        """Handle brush eraser size change."""
        # Sync with brush size
        if self._brush_size_spin is not None:
            self._brush_size_spin.blockSignals(True)
            self._brush_size_spin.setValue(size)
            self._brush_size_spin.blockSignals(False)

        if self._brush_eraser_radio is not None and self._brush_eraser_radio.isChecked():
            self.mask_tool_changed.emit(MaskTool.BRUSH_ERASER.value, size)

    def has_pending_mask_changes(self) -> bool:
        """
        Check if there are pending mask changes in Track step.

        Returns:
            True if in Track step and should mark dirty, False otherwise.
        """
        # This will be called by main_window to check if Apply buttons should be enabled
        return False  # Placeholder - actual check done in main_window

    def deselect_mask_tools(self) -> None:
        """Deselect all mask tools (go back to pan/zoom mode)."""
        if self._mask_tool_group is not None:
            # Uncheck all radio buttons
            for button in self._mask_tool_group.buttons():
                button.setAutoExclusive(False)
                button.setChecked(False)
                button.setAutoExclusive(True)
            # Emit tool change to NONE
            self.mask_tool_changed.emit(MaskTool.NONE.value, 0)
    
    def _store_original_values(self) -> None:
        """Store current values as original (for dirty detection)."""
        self._original_values = self.get_current_values()
        self._mark_clean()
    
    def _on_setting_changed(self) -> None:
        """Handle any setting value change."""
        current = self.get_current_values()
        self._is_dirty = current != self._original_values or self._mask_dirty
        self._update_apply_button()
    
    def _update_apply_button(self) -> None:
        """Update Apply buttons enabled state."""
        # "Apply to this group" reflects pending edits to the current group.
        if self._is_dirty:
            self._apply_btn.setEnabled(True)
            self._apply_btn.setStatusTip("Use shared settings and preview this group. Other groups use them when next processed. Mask changes affect this group only.")
        else:
            self._apply_btn.setEnabled(False)
            self._apply_btn.setStatusTip("No changes to apply. Settings are shared: Apply previews this group; other groups use them when next processed. Masks affect this group only.")

        self._discard_btn.setEnabled(self._is_dirty)
        self.dirty_changed.emit(self._is_dirty)

        # "Apply to all groups" stays enabled whenever applying would change at
        # least one group — i.e. there are pending edits, or some other group was
        # last evaluated with different settings. It is disabled only when every
        # other group already matches the current settings.
        apply_all_enabled = self._is_dirty or self._groups_out_of_sync
        self._apply_all_btn.setEnabled(apply_all_enabled)
        if apply_all_enabled:
            self._apply_all_btn.setStatusTip("Use shared settings, mark other results for recalculation, and preview this group. Other groups are processed later; masks are not copied.")
        else:
            self._apply_all_btn.setStatusTip("All groups already use these settings. Update all marks other results for recalculation and previews this group; it does not immediately process every group. Masks are not copied.")

    def set_groups_out_of_sync(self, out_of_sync: bool) -> None:
        """Set whether some other group was last evaluated with different settings.

        Controls the "Apply to all groups" button independently of pending edits:
        the button stays enabled while any other group is out of sync, so the user
        can always propagate the current settings to the rest.
        """
        if out_of_sync == self._groups_out_of_sync:
            return
        self._groups_out_of_sync = out_of_sync
        self._update_apply_button()
    
    def _mark_clean(self) -> None:
        """Mark settings as clean (no pending changes)."""
        self._is_dirty = False
        self._mask_dirty = False
        self._update_apply_button()

    def _mark_dirty(self) -> None:
        """Mark settings as dirty (has pending changes)."""
        self._mask_dirty = True
        self._is_dirty = True
        self._update_apply_button()
    
    def _finish_crop_editing(self):
        if self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            self._crop_edit_btn.setChecked(False)
            self._color_control.preview.setChecked(False)
            self._color_control.pick_button.setChecked(False)

    def _on_apply_clicked(self) -> None:
        """Handle Apply to this group button click."""
        self.finish_color_picker()
        self._store_original_values()  # New baseline
        self._finish_crop_editing()
        self.apply_requested.emit()
        
    def _on_apply_all_clicked(self) -> None:
        """Handle Apply to all groups button click."""
        self.finish_color_picker()
        self._store_original_values()  # New baseline
        self._finish_crop_editing()
        self.apply_all_requested.emit()
    
    def _on_redetect_clicked(self) -> None:
        """Handle Re-detect Plants button click."""
        self._centroids_modified = False
        self._redetect_btn.setEnabled(False)
        self._redetect_btn.setStatusTip("Click after moving centroids to re-run auto-detection")
        self.redetect_requested.emit()
    
    def enable_redetect(self) -> None:
        """Enable Re-detect button (called when user moves a centroid)."""
        self._centroids_modified = True
        if self._current_step == WorkflowStep.PREPROCESS and hasattr(self, '_redetect_btn'):
            self._redetect_btn.setEnabled(True)
            self._redetect_btn.setStatusTip("Re-run plant centroid detection")
    
    def mark_centroids_modified(self) -> None:
        """Mark that centroids have been modified (enable Re-detect button)."""
        self.enable_redetect()
    
    def set_pending_values(self, values):
        """Restore a draft without treating intermediate widget updates as edits."""
        bindings = {
            'n_clusters': ('_n_clusters_spin', 'setValue'),
            'reg_enabled': ('_reg_enabled_cb', 'setChecked'),
            'reg_margin': ('_reg_margin_spin', 'setValue'),
            'min_contour_area': ('_min_contour_area_spin', 'setValue'),
            'min_contour_length': ('_min_contour_length_spin', 'setValue'),
        }
        if self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            load = self._current_step == WorkflowStep.LOAD
            key, prefix = ('load_roi', 'blue') if load else ('preprocess_roi', 'green')
            if key in values:
                self._crop_value = None if values[key] is None else tuple(values[key])
                self._update_crop_summary()
            lower, upper = self._color_control.values()
            self._color_control.set_range(values.get(prefix+'_lower', lower), values.get(prefix+'_upper', upper))
        if self._current_step == WorkflowStep.LOAD:
            for widget, setter, value in (
                (self._background_enabled, 'setChecked', values.get('background_enabled', self._background_enabled.isChecked())),
                (self._background_region, 'setCurrentIndex', 0 if values.get('background_region', 'largest' if self._background_region.currentIndex() == 0 else 'all') == 'largest' else 1),
            ):
                blocked = widget.blockSignals(True)
                getattr(widget, setter)(value)
                widget.blockSignals(blocked)
            self._set_background_controls_enabled()
        for key in self.get_current_values():
            if key not in values or key not in bindings:
                continue
            name, setter = bindings[key]
            widget = getattr(self, name)
            was_blocked = widget.blockSignals(True)
            getattr(widget, setter)(values[key])
            widget.blockSignals(was_blocked)
        if self._current_step == WorkflowStep.PREPROCESS:
            self._on_reg_enabled_changed(self._reg_enabled_cb.isChecked())
        self._on_setting_changed()

    def reset_for_group(self) -> None:
        """Load the current applied settings; the window restores any saved draft."""
        values = dict(n_clusters=self._config.n_clusters,
                      background_enabled=self._config.crop.background_enabled, background_region=self._config.crop.background_region,
                      load_roi=self._config.load_roi, preprocess_roi=self._config.preprocess_roi,
                      blue_lower=self._config.crop.blue_hsv_lower, blue_upper=self._config.crop.blue_hsv_upper,
                      green_lower=self._config.green.hsv_lower, green_upper=self._config.green.hsv_upper,
                      reg_enabled=self._config.registration.enabled,
                      reg_margin=self._config.registration.margin_ratio,
                      min_contour_area=self._config.threshold.min_contour_area,
                      min_contour_length=self._config.threshold.min_contour_length)
        self.set_pending_values(values)
        self._store_original_values()
        self._centroids_modified = False
        if self._current_step == WorkflowStep.PREPROCESS and hasattr(self, '_redetect_btn'):
            self._redetect_btn.setEnabled(False)

    def get_current_values(self) -> dict:
        """Get current setting values."""
        values = {}
        
        if self._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            load = self._current_step == WorkflowStep.LOAD
            key, prefix = ('load_roi', 'blue') if load else ('preprocess_roi', 'green')
            values[key] = self._crop_value
            values[prefix+'_lower'], values[prefix+'_upper'] = self._color_control.values()
            if load:
                values['background_enabled'] = self._background_enabled.isChecked()
                values['background_region'] = 'largest' if self._background_region.currentIndex() == 0 else 'all'
            if not load:
                values['n_clusters'] = self._n_clusters_spin.value()
                values['reg_enabled'] = self._reg_enabled_cb.isChecked()
                values['reg_margin'] = self._reg_margin_spin.value()

        elif self._current_step == WorkflowStep.TRACK:
            if hasattr(self, '_min_contour_area_spin'):
                values["min_contour_area"] = self._min_contour_area_spin.value()
                values["min_contour_length"] = self._min_contour_length_spin.value()
        
        return values
    
    def _on_reg_enabled_changed(self, state: int) -> None:
        """Handle enabling/disabling registration parameters."""
        enabled = bool(state)
        if hasattr(self, '_reg_margin_spin'):
            self._reg_margin_spin.setEnabled(enabled)

    def is_dirty(self) -> bool:
        """Check if there are unsaved changes."""
        return self._is_dirty
