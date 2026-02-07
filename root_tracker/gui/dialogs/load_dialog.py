"""
Load dialog for selecting input folder.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QFormLayout
)
from PySide6.QtCore import Qt


class LoadDialog(QDialog):
    """
    Dialog for selecting input folder and data settings.
    
    Attributes:
        input_path: Selected input folder path.
        filename_template: Template for parsing filenames.
        date_format: Date format string.
    """
    
    def __init__(
        self, 
        parent=None, 
        initial_path: str = "",
        initial_template: str = "{group}.{date}",
        initial_date_format: str = "%d-%m-%y"
    ) -> None:
        super().__init__(parent)
        
        self.input_path = initial_path
        self.filename_template = initial_template
        self.date_format = initial_date_format
        
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
        
        # Filename template
        self._template_edit = QLineEdit(self.filename_template)
        self._template_edit.setToolTip(
            "Template for parsing filenames.\n"
            "Use {group} for group name and {date} for date."
        )
        form.addRow("Filename template:", self._template_edit)
        
        # Date format
        self._date_edit = QLineEdit(self.date_format)
        self._date_edit.setToolTip(
            "Python strptime format for dates.\n"
            "Example: %d-%m-%Y for 01-02-2024"
        )
        form.addRow("Date format:", self._date_edit)
        
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
    
    def _accept(self) -> None:
        """Accept dialog and store values."""
        self.input_path = self._path_edit.text()
        self.filename_template = self._template_edit.text()
        self.date_format = self._date_edit.text()
        self.accept()
