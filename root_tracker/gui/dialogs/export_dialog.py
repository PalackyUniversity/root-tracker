"""
Export dialog for configuring CSV export.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QFormLayout,
    QCheckBox
)
from PySide6.QtCore import Qt


class ExportDialog(QDialog):
    """
    Dialog for configuring export options.
    
    Attributes:
        output_path: Selected output folder path.
        filename: Output CSV filename.
        include_images: Whether to save annotated images.
    """
    
    def __init__(self, parent=None, initial_path: str = "") -> None:
        super().__init__(parent)
        
        self.output_path = initial_path
        self.filename = "statistics.csv"
        self.include_images = True
        
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(500)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Form layout for inputs
        form = QFormLayout()
        
        # Output path
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit(self.output_path)
        path_layout.addWidget(self._path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        path_layout.addWidget(browse_btn)
        
        form.addRow("Output folder:", path_layout)
        
        # Filename
        self._filename_edit = QLineEdit(self.filename)
        form.addRow("CSV filename:", self._filename_edit)
        
        # Include images checkbox
        self._images_check = QCheckBox("Save annotated images")
        self._images_check.setChecked(self.include_images)
        form.addRow("", self._images_check)
        
        layout.addLayout(form)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        export_btn = QPushButton("Export")
        export_btn.setDefault(True)
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #5cb85c;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #449d44;
            }
        """)
        export_btn.clicked.connect(self._accept)
        btn_layout.addWidget(export_btn)
        
        layout.addLayout(btn_layout)
    
    def _browse_folder(self) -> None:
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._path_edit.text()
        )
        if folder:
            self._path_edit.setText(folder)
    
    def _accept(self) -> None:
        """Accept dialog and store values."""
        self.output_path = self._path_edit.text()
        self.filename = self._filename_edit.text()
        self.include_images = self._images_check.isChecked()
        self.accept()
