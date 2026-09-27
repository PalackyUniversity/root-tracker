"""Load-step metadata inspector and asynchronous folder metadata reader."""
from collections import Counter

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QEvent
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QSizePolicy,
)
from ..io.image_metadata import compare_metadata, read_metadata
from ..io.photo_settings import FIELD_HELP
from .image_tree import ImageNavigationTree
from .right_checkbox import RightAlignedCheckBox
from .theme import metadata_tree_stylesheet, is_light


class _ReadSignals(QObject):
    finished = Signal(object)


class _ReadMetadata(QRunnable):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.paths = [image.path for image in images]
        self.signals = _ReadSignals()

    def run(self):
        try:
            results = read_metadata(self.paths)
        except Exception as error:
            results = {path: ({}, str(error)) for path in self.paths}
        try:
            self.signals.finished.emit([(image, *results[path])
                                        for image, path in zip(self.images, self.paths)])
        except RuntimeError:
            # QApplication may have destroyed the receiver while a read finishes.
            pass


class MetadataController(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = False
        self.series = {}
        self._jobs = set()
        self._pending = set()

    def set_series(self, series):
        self.series = series
        self._start()
        self.changed.emit()

    def set_options(self, enabled):
        self.enabled = enabled
        self._start()
        self.changed.emit()

    def _start(self):
        if not self.enabled:
            return
        images = [image for series in self.series.values() for image in series.images
                  if image.camera_metadata is None and id(image) not in self._pending]
        if not images:
            return
        job = _ReadMetadata(images)
        self._jobs.add(job)
        self._pending.update(id(image) for image in images)
        job.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(job)

    def _finished(self, results):
        for image, values, error in results:
            image.camera_metadata = values
            image.camera_metadata_error = error
            self._pending.discard(id(image))
        completed = {id(image) for image, _, _ in results}
        self._jobs = {job for job in self._jobs
                      if any(id(image) not in completed for image in job.images)}
        self.changed.emit()

    def warnings(self):
        warnings = {}
        if not self.enabled:
            return warnings
        for series in self.series.values():
            if any(image.camera_metadata is None for image in series.images):
                continue
            _, differences = compare_metadata(
                [image.camera_metadata for image in series.images])
            for image, fields in zip(series.images, differences):
                details = []
                if fields:
                    details.append('Metadata differs: ' + ', '.join(fields))
                if image.camera_metadata_error:
                    details.append('Metadata read warning: ' + image.camera_metadata_error)
                if details:
                    warnings[id(image)] = '; '.join(details)
        return warnings


class MetadataTree(ImageNavigationTree):
    """Share navigation-tree visuals, with normal navigation through field rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(metadata_tree_stylesheet(self.palette()))

    def moveCursor(self, action, modifiers):
        return QTreeWidget.moveCursor(self, action, modifiers)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and not getattr(self, '_updating_style', False):
            self._updating_style = True
            try:
                self.setStyleSheet(metadata_tree_stylesheet(self.palette()))
            finally:
                self._updating_style = False


class MetadataPanel(QGroupBox):
    options_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__('Image metadata', parent)
        self._series = None
        self._image = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        self.enabled = RightAlignedCheckBox('Enable metadata comparison')
        self.enabled.setChecked(False)
        self.enabled.setStatusTip('Read metadata for all loaded images and show differences within each group. Applies to all groups.')
        layout.addWidget(self.enabled)
        self.summary = QLabel('Select a group to inspect its metadata.')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = MetadataTree()
        self.table.setColumnCount(3)
        self.table.setHeaderLabels(['Field', 'Match', 'Value'])
        self.table.setRootIsDecorated(True)
        self.table.setUniformRowHeights(True)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.table.setIndentation(16)
        self.table.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)
        self.table.setHorizontalScrollMode(QTreeWidget.ScrollMode.ScrollPerPixel)
        self.table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 46)
        self.table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.header().setStretchLastSection(True)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(2, 80)
        layout.addWidget(self.table)
        self.enabled.toggled.connect(self._options_changed)
        self.table.itemExpanded.connect(self._fit_height)
        self.table.itemCollapsed.connect(self._fit_height)
        self.refresh()

    def _options_changed(self):
        self.refresh()
        self.options_changed.emit(self.enabled.isChecked())

    def set_selection(self, series, image=None):
        self._series, self._image = series, image
        self.refresh()

    def refresh(self):
        expanded = {self.table.topLevelItem(i).text(0)
                    for i in range(self.table.topLevelItemCount())
                    if self.table.topLevelItem(i).isExpanded()}
        scroll = self.table.verticalScrollBar().value()
        self.table.clear()
        self.summary.show()
        enabled = self.enabled.isChecked()
        self.table.setVisible(enabled)
        if not enabled:
            self.summary.hide()
            return
        if self._series is None:
            self.summary.setText('Select a group to inspect its metadata.')
            self.table.hide()
            return
        images = self._series.images
        if any(image.camera_metadata is None for image in images):
            self.summary.setText('Reading image metadata…')
            self.table.hide()
            return
        rows, _ = compare_metadata([image.camera_metadata for image in images])
        errors = [f'{image.filename}: {image.camera_metadata_error}' for image in images if image.camera_metadata_error]
        self.summary.setText(f'{len(errors)} images could not be read.' if errors
                             else 'No photo settings found.' if not rows else '')
        self.summary.setVisible(bool(errors) or not rows)
        self.table.setVisible(bool(rows))
        self.summary.setToolTip('\n'.join(errors))
        selected = next((index for index, image in enumerate(images) if image is self._image), None)
        for row in rows:
            counts = Counter(row.values)
            summary = '; '.join(f'{value if value is not None else "(missing)"} ({count})'
                                for value, count in counts.items())
            value = row.values[selected] if selected is not None else summary
            item = QTreeWidgetItem([row.key, '✓' if row.same else '✗',
                                   '(missing)' if value is None else value])
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            light = is_light(self.palette())
            item.setForeground(1, QColor(('#237a45' if light else '#74c69d') if row.same
                                        else ('#9a5b00' if light else '#ffad42')))
            item.setToolTip(0, row.key + '\n' + FIELD_HELP[row.key])
            item.setToolTip(1, 'Identical in every image' if row.same else 'Different or missing in some images')
            item.setToolTip(2, summary)
            self.table.addTopLevelItem(item)
            for image, value in zip(images, row.values):
                child = QTreeWidgetItem([image.filename, '', value if value is not None else '(missing)'])
                child.setToolTip(0, image.path)
                child.setToolTip(2, child.text(2))
                item.addChild(child)
            item.setExpanded(row.key in expanded)
        self._fit_height()
        self.table.verticalScrollBar().setValue(scroll)

    def _fit_height(self):
        visible_rows = self.table.topLevelItemCount()
        for index in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(index)
            if item.isExpanded():
                visible_rows += item.childCount()
        row_height = max(28, self.table.fontMetrics().height() + 10)
        # At most eight rows, with scrolling for the rest. The outer settings
        # panel must never give this group unused vertical window space.
        height = (min(8, max(1, visible_rows)) * row_height
                  + self.table.header().sizeHint().height() + 24)
        self.table.setFixedHeight(height)
        self.updateGeometry()
