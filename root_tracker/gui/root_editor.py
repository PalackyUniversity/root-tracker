"""Coordinate selection controls and durable per-image detection edits."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from .workflow_bar import WorkflowStep
from ..io.root_editing import editable_document, edit_roots
from ..io.rsml_replacement import restore_measurements


class RootEditor(QObject):
    save_completed = Signal(str, object)

    def __init__(self, window):
        super().__init__(window)
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='root-save')
        self._saves = {}
        self.save_completed.connect(self._report_save)
        self.window = window
        self.image = None
        self.source = None
        self.document = None
        viewer = window._image_viewer
        panel = window._settings_panel
        viewer.root_selection_changed.connect(self.selection_changed)
        viewer.root_assignment_requested.connect(self.assign)
        viewer.root_delete_requested.connect(self.delete)
        panel.root_assignment_requested.connect(self.assign)
        panel.root_delete_requested.connect(self.delete)

    def present(self, image, step):
        viewer = self.window._image_viewer
        source = (image.rsml_document if image.rsml_document is not None else image.rsml_samples) if image else None
        if step != WorkflowStep.TRACK:
            image = source = None
        if image is not self.image or source is not self.source or viewer._root_document is not self.document:
            self.image, self.source = image, source
            self.document = None
            viewer.set_root_document_loader(self._load_document if source is not None else None)
        else:
            self.selection_changed()

    def _load_document(self):
        series = self.window._current_series
        index = next((i for i, candidate in enumerate(series.images) if candidate is self.image), 0) if series else 0
        self.document = editable_document(self.image, group=series.group if series else None, image_index=index)
        return self.document

    def selection_changed(self):
        viewer = self.window._image_viewer
        self.window._settings_panel.set_root_selection(viewer._root_document, viewer.selected_roots)

    def assign(self, plant_index):
        self._edit(plant_index=plant_index)

    def delete(self):
        self._edit(delete=True)

    def _edit(self, **change):
        window = self.window
        viewer = window._image_viewer
        # Workers and image navigation must never write through stale selection.
        from .main_window import ProcessingState
        if (window._state != ProcessingState.IDLE or self.image is not window._current_image
                or window._workflow_bar.get_current_step() != WorkflowStep.TRACK
                or self.document is None or not viewer.selected_roots):
            return
        if not change.get('delete') and all(
                self.document.roots[i].plant_index == change['plant_index'] for i in viewer.selected_roots):
            return
        try:
            selection = edit_roots(self.image, self.document, viewer.selected_roots, persist=False,
                                   mask=window._current_series.user_mask if window._current_series else None, **change)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(window, "Root edit failed", str(exc))
            return
        if window._current_series is not None:
            restore_measurements(window._current_series)
        window._display_image(self.image, preserve_view=True)
        viewer.select_roots({index for index, source in enumerate(self.image.rsml_root_sources)
                             if source in selection})
        window._image_tree.refresh_status()
        window._sync_auto_apply_controls()
        self._queue_save(self.image)
        window.statusBar().showMessage("Root corrections applied. Saving…", 5000)

    def _queue_save(self, image):
        from ..io.rsml_replacement import replace_roots
        snapshot = copy(image)
        document = snapshot.rsml_unmasked_document
        future = self._save_executor.submit(replace_roots, snapshot, document, manual=True)
        self._saves[image.path] = (future, snapshot)
        future.add_done_callback(lambda done, path=image.path: self.save_completed.emit(path, done))

    @Slot(str, object)
    def _report_save(self, path, future):
        pending = self._saves.get(path)
        if pending is None or pending[0] is not future:
            return
        error = future.exception()
        if error is not None:
            self.window.statusBar().showMessage(f"Root corrections could not be saved: {error}. Reset or closing will offer a retry.")
        else:
            self._saves.pop(path, None)
            self.window.statusBar().showMessage("Root corrections saved.", 3000)

    def flush(self):
        """Finish ordered writes before resetting, moving files, or closing."""
        for path, (future, snapshot) in list(self._saves.items()):
            while True:
                try:
                    future.result()
                    break
                except Exception as error:
                    answer = QMessageBox.critical(
                        self.window, "Root corrections not saved",
                        f"Could not save {snapshot.filename}: {error}\nRetry saving, or cancel to keep this image open with its corrections.",
                        QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
                    if answer != QMessageBox.StandardButton.Retry:
                        return False
                    self._queue_save(snapshot)
                    future, snapshot = self._saves[path]
            self._saves.pop(path, None)
        return True

    def close(self):
        if not self.flush():
            return False
        self._save_executor.shutdown(wait=True)
        return True

    def has_manual_edits(self):
        series = self.window._current_series
        return (series is not None and self.window._settings_panel._current_step == WorkflowStep.TRACK
                and any(image.rsml_unmasked_document is not None for image in series))

    def reset(self):
        from ..io.rsml_replacement import reset_manual_roots
        if not self.flush():
            return False
        series = self.window._current_series
        if series is None:
            return False
        changed = False
        for image in series:
            changed = reset_manual_roots(image) or changed
        if changed:
            series.pipeline_state.invalidate_from('track')
            self.present(None, WorkflowStep.LOAD)
            self.window._image_tree.refresh_status()
        return changed
