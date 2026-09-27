"""
Load dialog for selecting input folder.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QFormLayout
)
from ..combo_box import RoundedComboBox as QComboBox
from PySide6.QtCore import Qt
from datetime import datetime


class LoadDialog(QDialog):
    """
    Dialog for selecting input folder and data settings.
    
    Attributes:
        input_path: Selected input folder path.
        filename_template: Template for parsing filenames.
        date_format: Date format string.
        detect_barcodes: Enable/disable barcode detection.
    """
    
    def __init__(
        self, 
        parent=None, 
        initial_path: str = "",
        initial_template: str = "{group}.{date}",
        initial_date_format: str = "%y-%m-%d",
        initial_detect_barcodes: bool = True
    ) -> None:
        super().__init__(parent)
        
        self.input_path = initial_path
        self.filename_template = initial_template
        self.date_format = initial_date_format
        self.detect_barcodes = initial_detect_barcodes
        
        # Common filename templates with examples
        self._filename_templates = [
            ("{group}.{date}", "{group}.{date} (e.g., RT_25.25-09-22)"),
            ("{group}_{date}", "{group}_{date} (e.g., RT_25_25-09-22)"),
            ("{date}_{group}", "{date}_{group} (e.g., 25-09-22_RT_25)"),
            ("{group}-{date}", "{group}-{date} (e.g., RT_25-25-09-22)"),
            ("{date}-{group}", "{date}-{group} (e.g., 25-09-22-RT_25)"),
            ("custom", "Custom template..."),
        ]
        
        # Common date formats with examples
        self._date_formats = [
            ("%y-%m-%d", "%y-%m-%d (e.g., 26-04-27 = 27 April 2026)"),
            ("%Y-%m-%d", "%Y-%m-%d (e.g., 2022-09-25)"),
            ("%Y%m%d", "%Y%m%d (e.g., 20220925)"),
            ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M-%S (e.g., 2022-09-25_14-30-45)"),
            ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M%S (e.g., 20220925_143045)"),
            ("%Y-%m-%dT%H-%M-%SZ", "%Y-%m-%dT%H-%M-%SZ (e.g., 2022-09-25T14-30-45Z)"),
            ("%d-%m-%y", "%d-%m-%y (e.g., 25-09-22)"),
            ("custom", "Custom format..."),
        ]
        
        self.setWindowTitle("Load Images")
        self.setMinimumWidth(500)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Form layout for inputs
        form = QFormLayout()
        
        # Input path
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit(self.input_path)
        path_layout.addWidget(self._path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        path_layout.addWidget(browse_btn)
        
        form.addRow("Input folder:", path_layout)
        
        # Filename template dropdown
        self._template_combo = QComboBox()
        for template, display in self._filename_templates:
            self._template_combo.addItem(display, template)
        
        # Find initial selection for template
        initial_template_index = 0
        for i, (template, _) in enumerate(self._filename_templates):
            if template == self.filename_template:
                initial_template_index = i
                break
        else:
            # Not in list - select custom
            initial_template_index = len(self._filename_templates) - 1
        
        self._template_combo.setCurrentIndex(initial_template_index)
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        form.addRow("Filename template:", self._template_combo)
        
        # Custom template input (hidden by default)
        self._custom_template_label = QLabel("Custom template:")
        self._custom_template_edit = QLineEdit(self.filename_template if initial_template_index == len(self._filename_templates) - 1 else "")
        self._custom_template_edit.setPlaceholderText("e.g., {group}.{date}")
        self._custom_template_edit.setToolTip(
            "Template for parsing filenames.\n"
            "Use {group} for group name and {date} for date."
        )
        form.addRow(self._custom_template_label, self._custom_template_edit)
        
        # Show/hide custom template input based on selection
        is_custom_template = initial_template_index == len(self._filename_templates) - 1
        self._custom_template_label.setVisible(is_custom_template)
        self._custom_template_edit.setVisible(is_custom_template)
        
        # Date format dropdown
        self._date_combo = QComboBox()
        for fmt, display in self._date_formats:
            self._date_combo.addItem(display, fmt)
        
        # Find initial selection
        initial_index = 0
        for i, (fmt, _) in enumerate(self._date_formats):
            if fmt == self.date_format:
                initial_index = i
                break
        else:
            # Not in list - select custom
            initial_index = len(self._date_formats) - 1
        
        self._date_combo.setCurrentIndex(initial_index)
        self._date_combo.currentIndexChanged.connect(self._on_date_format_changed)
        form.addRow("Date format:", self._date_combo)
        
        # Custom date format input (hidden by default)
        self._custom_date_label = QLabel("Custom format:")
        self._custom_date_edit = QLineEdit(self.date_format if initial_index == len(self._date_formats) - 1 else "")
        self._custom_date_edit.setPlaceholderText("e.g., %d-%m-%Y")
        self._custom_date_edit.setToolTip(
            "Python strptime format for dates.\n"
            "Example: %d-%m-%Y for day-month-year"
        )
        form.addRow(self._custom_date_label, self._custom_date_edit)
        
        # Show/hide custom input based on selection
        is_custom_date = initial_index == len(self._date_formats) - 1
        self._custom_date_label.setVisible(is_custom_date)
        self._custom_date_edit.setVisible(is_custom_date)
        
        tips = {
            self._path_edit: 'Folder containing the image dataset. Images are grouped using the filename template.',
            self._template_combo: 'How filenames encode the group, barcode and date. Choose a matching pattern or enter a custom template.',
            self._date_combo: 'How the date in each filename is interpreted. This controls chronological ordering and growth calculations.',
        }
        for widget, tip in tips.items():
            widget.setToolTip(tip)
        for widget in (self._template_combo, self._date_combo, self._custom_template_edit, self._custom_date_edit):
            label = form.labelForField(widget)
            if label is not None:
                label.setToolTip(widget.toolTip())
        form.labelForField(path_layout).setToolTip(self._path_edit.toolTip())
        layout.addLayout(form)
        
        # Help text
        help_label = QLabel(
            "<i>Tip: Images should be organized in folders or have consistent naming.</i>"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        load_btn = QPushButton("Load")
        load_btn.setDefault(True)
        load_btn.clicked.connect(self._accept)
        btn_layout.addWidget(load_btn)
        
        layout.addLayout(btn_layout)
    
    def _browse_folder(self) -> None:
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self._path_edit.text()
        )
        if folder:
            self._path_edit.setText(folder)
    
    def _on_template_changed(self, index: int) -> None:
        """Handle filename template selection change."""
        # Show custom input only when "Custom" is selected (last item)
        is_custom = (index == len(self._filename_templates) - 1)
        self._custom_template_label.setVisible(is_custom)
        self._custom_template_edit.setVisible(is_custom)
        
        # Adjust dialog size to accommodate custom field
        self.adjustSize()
    
    def _on_date_format_changed(self, index: int) -> None:
        """Handle date format selection change."""
        # Show custom input only when "Custom" is selected (last item)
        is_custom = (index == len(self._date_formats) - 1)
        self._custom_date_label.setVisible(is_custom)
        self._custom_date_edit.setVisible(is_custom)
        
        # Adjust dialog size to accommodate custom field
        self.adjustSize()
    
    def _accept(self) -> None:
        """Accept dialog and store values."""
        self.input_path = self._path_edit.text()
        
        # Get filename template from dropdown or custom field
        current_template = self._template_combo.currentData()
        if current_template == "custom":
            self.filename_template = self._custom_template_edit.text().strip()
            if not self.filename_template:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Invalid Input", "Please enter a custom filename template.")
                return
        else:
            self.filename_template = current_template
        
        # Get date format from dropdown or custom field
        current_fmt = self._date_combo.currentData()
        if current_fmt == "custom":
            self.date_format = self._custom_date_edit.text().strip()
            if not self.date_format:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Invalid Input", "Please enter a custom date format.")
                return
        else:
            self.date_format = current_fmt
        
        self.accept()
