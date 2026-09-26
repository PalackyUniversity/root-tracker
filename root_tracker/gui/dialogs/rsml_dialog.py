"""RSML document browsing and export, isolated from the tracking workspace."""
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QGraphicsScene, QGraphicsView, QSplitter, QDialogButtonBox,
    QFileDialog, QMessageBox, QProgressBar,
)

from ...io.rsml import RSMLDocument, save_rsml
from ...io.rsml_export import export_series_rsml


class RSMLDialog(QDialog):
    def __init__(self, document: RSMLDocument, parent=None):
        super().__init__(parent)
        self.document = document
        self.setWindowTitle('RSML document')
        self.resize(900, 650)
        layout = QVBoxLayout(self)
        is_3d = any(len(point) == 3 for root in document.roots for point in root.points)
        self.summary = QLabel(
            f'{len(document.plant_ids)} plants · {len(document.roots)} roots\n'
            f'Image reference: {document.image_name or "not supplied"}\n'
            f'Scale: {document.resolution:g} pixels per {document.unit}\n'
            + ('XY projection of 3D geometry. Lengths include Z.' if is_3d else '2D geometry.')
            + '\nLengths below are polyline lengths in pixels, separate from tracker statistics.'
            + '\nSave Copy preserves all original XML, including information not displayed here.')
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        splitter = QSplitter()
        layout.addWidget(splitter)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Plant / root', 'Length (px)'])
        splitter.addWidget(self.tree)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        splitter.addWidget(self.view)
        plants = [QTreeWidgetItem(self.tree, [pid or '(unnamed plant)']) for pid in document.plant_ids]
        items = []
        for root in document.roots:
            parent_item = items[root.parent_index] if root.parent_index is not None else plants[root.plant_index]
            item = QTreeWidgetItem(parent_item, [root.id or '(unnamed root)',
                                               f'{root.length:.3f}' if root.points else 'unsupported geometry'])
            items.append(item)
            if root.points:
                path = QPainterPath()
                path.moveTo(*root.points[0][:2])
                for point in root.points[1:]:
                    path.lineTo(*point[:2])
                pen = QPen(QColor.fromHsv((root.plant_index * 97) % 360, 200, 210))
                pen.setCosmetic(True)
                pen.setWidth(2)
                self.scene.addPath(path, pen)
        self.tree.expandAll()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton('Save Copy…', QDialogButtonBox.ButtonRole.ActionRole).clicked.connect(self.save_copy)
        buttons.addButton('Fit View', QDialogButtonBox.ButtonRole.ActionRole).clicked.connect(self.fit_view)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        QTimer.singleShot(0, self.fit_view)

    def fit_view(self):
        bounds = self.scene.itemsBoundingRect()
        if not bounds.isEmpty():
            self.view.fitInView(bounds.adjusted(-1, -1, 1, 1), Qt.AspectRatioMode.KeepAspectRatio)

    def save_copy(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save RSML Copy', 'copy.rsml', 'RSML Files (*.rsml)')
        if filename:
            try:
                save_rsml(self.document, filename)
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, 'RSML Save Failed', str(exc))


class RSMLExportWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(bool, str)

    def __init__(self, series, config, destination, parent=None, *, image_index=None):
        super().__init__(parent)
        self.series = series
        self.config = config
        self.destination = Path(destination)
        self.image_index = image_index

    def run(self):
        count = 0
        try:
            for index, series in enumerate(self.series):
                count += len(export_series_rsml(series, self.config, self.destination, image_index=self.image_index))
                self.progress.emit(index + 1, len(self.series))
            self.completed.emit(True, f'Exported {count} RSML files and matching images to:\n{self.destination}')
        except Exception as exc:
            self.completed.emit(False, f'{exc}\nPreviously completed series, if any, remain in {self.destination}.')


class RSMLExportDialog(QDialog):
    """Modal export holds the workspace stable until the worker exits."""
    def __init__(self, series, config, destination, parent=None, *, image_index=None):
        super().__init__(parent)
        self.setWindowTitle('Export to RSML')
        self.resize(560, 160)
        layout = QVBoxLayout(self)
        self.status = QLabel('Exporting RSML and matching processed images…')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, len(series))
        layout.addWidget(self.progress)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.worker = RSMLExportWorker(series, config, destination, self, image_index=image_index)
        self.worker.progress.connect(lambda value, total: self.progress.setValue(value))
        self.worker.completed.connect(self._completed)
        self.worker.finished.connect(lambda: self.buttons.setEnabled(True))
        QTimer.singleShot(0, self.worker.start)
        self._started = False
        self.worker.started.connect(self._mark_started)

    def _mark_started(self):
        self._started = True

    def _completed(self, success, message):
        self.status.setText(('Export complete.\n' if success else 'Export failed.\n') + message)

    def reject(self):
        if self._started and not self.worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if not self._started or self.worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)
