"""Workflow adapter for crop geometry and color previews, separate from processing."""
from copy import deepcopy
import math
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtGui import QShortcut, QKeySequence
from .workflow_bar import WorkflowStep
from .preview_cache import PreviewCache
from ..preprocessing.cropper import ImageCropper
from ..preprocessing.green_detector import GreenAreaDetector
from ..preprocessing.colors import sample_hsv
from ..preprocessing import roi


def apply_editor_values(config, values):
    """Used for both preview copies and the committed processing configuration."""
    for key in ('background_enabled', 'background_region'):
        if key in values:
            setattr(config.crop, key, values[key])
    for key in ('load_roi', 'preprocess_roi'):
        if key in values:
            setattr(config, key, values[key])
    for prefix, target, lower, upper in (
        ('blue', config.crop, 'blue_hsv_lower', 'blue_hsv_upper'),
        ('green', config.green, 'hsv_lower', 'hsv_upper'),
    ):
        if prefix+'_lower' in values:
            for field, key in ((lower, prefix+'_lower'), (upper, prefix+'_upper')):
                selected = tuple(values[key])
                if tuple(getattr(target, field)) != selected:
                    setattr(target, field, selected)


class RoiEditor(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.panel = window._settings_panel
        self.viewer = window._image_viewer
        self._canvas = None
        self._source_key = None
        self._hsv = None
        self._source_canvas = None
        self._preview_cache = PreviewCache()
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.setInterval(40)
        self._prefetch_timer.timeout.connect(self._prefetch_neighbors)
        self._escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), window)
        self._escape.setEnabled(False)
        self._escape.activated.connect(self._cancel_pick)
        self._sample_scope = None
        self._locked = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self.refresh)
        self.panel.crop_edit_toggled.connect(self.schedule)
        self.panel.crop_preview_changed.connect(self.schedule)
        self.panel.color_preview_toggled.connect(self.schedule)
        self.panel.color_pick_toggled.connect(self._pick_toggled)
        self.viewer.crop_changed.connect(self.panel.set_crop)
        self.viewer.color_picked.connect(self._sample_color)
        self.viewer.color_pick_cancelled.connect(self._cancel_pick)
        self.viewer.editor_help.connect(window.statusBar().showMessage)

    def schedule(self, *args):
        if not self._locked and not self._timer.isActive():
            self._timer.start()

    def refresh(self):
        self._timer.stop()
        if self._locked:
            return
        image = self.window._current_image
        if image is not None:
            self.window._display_image(image, preserve_view=True)

    def _sampling_scope(self):
        return id(self.window._current_series), self.panel._current_step

    def _pick_toggled(self, enabled):
        if not enabled or self._locked:
            self._cancel_pick()
            return
        self._sample_scope = self._sampling_scope()
        self._escape.setEnabled(True)
        # Prepare exactly the displayed pixels before accepting either sample.
        self.refresh()
        self.viewer.set_color_picking(True)
        limit = 'lower' if self.panel._color_control.sample_index == 0 else 'upper'
        self.window.statusBar().showMessage(f'Click an image area to set the {limit} color limit. Press Escape to return without sampling.')

    def _cancel_pick(self):
        self._sample_scope = None
        self._escape.setEnabled(False)
        self.viewer.set_color_picking(False)
        if self.panel._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            button = self.panel._color_control.pick_button
            previous = button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(previous)
            self.panel._color_control.end_sampling()
        self.window.statusBar().clearMessage()

    def _sample_color(self, x, y):
        if self._canvas is None or self._locked or self._sample_scope != self._sampling_scope():
            return
        if not (0 <= x < self._canvas.shape[1] and 0 <= y < self._canvas.shape[0]):
            return
        sample = sample_hsv(self._canvas, x, y)
        control = self.panel._color_control
        control.apply_sample(sample)
        self._cancel_pick()

    def set_locked(self, locked):
        self._locked = locked
        if locked:
            self._timer.stop()
            self.viewer.set_color_picking(False)
            if self.panel._current_step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
                self.panel._color_control.pick_button.setChecked(False)
        if self.viewer._crop_overlay is not None:
            self.viewer._crop_overlay.setEnabled(not locked)

    def reset(self):
        self._prefetch_timer.stop()
        self._preview_cache.clear()
        self._cancel_pick()
        self._timer.stop()
        self._source_key = None
        self._canvas = self._hsv = self._source_canvas = None
        self._live_key = self._displayed_scope = self._plate_key = None
        self._plate_points = None
        self._load_crop_key = self._load_crop_canvas = self._load_crop_matrix = None
        self._frame_scope = None
        self._frame_matrix = None
        self._live_canvas = self._live_dimmed = None
        self._escape.setEnabled(False)
        self.viewer.clear_crop()
        self.viewer.clear_plate_outline()
        self.viewer.set_color_picking(False)

    def close(self):
        self.reset()
        self._preview_cache.close()

    def _source_request(self, image_data, step, config):
        path = Path(image_data.path)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size, step, config.rotation,
               config.load_roi, tuple(config.crop.blue_hsv_lower), tuple(config.crop.blue_hsv_upper), config.crop.background_enabled, config.crop.background_region)
        # Reuse the current decoded photo when only its color/crop settings change.
        original = self._source_canvas if (step == WorkflowStep.LOAD and self._source_key is not None
                   and self._source_key[:4] == key[:4]) else None
        snapshot = deepcopy(config)
        def prepare():
            source = original if original is not None else cv2.imread(str(path))
            if source is None:
                return None
            cropper = ImageCropper(snapshot)
            return ((source, cropper.plate_outline(source)) if step == WorkflowStep.LOAD else
                    (cropper.process(source), None))
        return key, prepare

    def _source(self, image_data, step, config):
        request = self._source_request(image_data, step, config)
        if request is None:
            return None
        result = self._preview_cache.get(*request)
        if result is None:
            return None
        self._source_key = request[0]
        self._source_canvas, self._source_plate_points = result
        return self._source_canvas

    def _prefetch_neighbors(self):
        series, current = self.window._current_series, self.window._current_image
        step = self.panel._current_step
        if self._locked or series is None or current is None or step not in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS):
            return
        index = next((i for i, item in enumerate(series.images) if item is current), None)
        if index is None:
            return
        config = deepcopy(self.window._config)
        apply_editor_values(config, self.panel.get_current_values())
        requests = []
        for neighbor in (index+1, index-1, index+2):
            if 0 <= neighbor < len(series.images):
                item = series.images[neighbor]
                if step == WorkflowStep.PREPROCESS and item.image is not None and not self.panel.crop_editing():
                    continue
                request = self._source_request(item, step, config)
                if request is not None:
                    requests.append(request)
                if len(requests) == 2:
                    break
        self._preview_cache.prefetch(requests)

    @staticmethod
    def _analysis_box(canvas, config):
        return ImageCropper(config).analysis_roi(canvas.shape)

    def _present_live_color(self, config, step, preserve_view, image_data):
        """Threshold original pixels, but composite only the visible resolution.

        Keep crop geometry stationary during the gesture; the ordinary refresh
        recomputes automatic geometry when the range dialog closes.
        """
        canvas = self._canvas
        crop = self.viewer._crop_overlay
        box = crop.box if crop is not None else None
        scale = min(1., max(1600 / max(canvas.shape[:2]),
                           math.hypot(self.viewer._view.transform().m11(), self.viewer._view.transform().m12()) * self.viewer.devicePixelRatioF()))
        size = (max(1, round(canvas.shape[1]*scale)), max(1, round(canvas.shape[0]*scale)))
        key = (id(canvas), size)
        if getattr(self, '_live_key', None) != key:
            self._live_key = key
            self._live_canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA) if scale < 1 else canvas
            self._live_dimmed = cv2.convertScaleAbs(self._live_canvas, alpha=.3)
        if self._hsv is None:
            self._hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        mask = ((ImageCropper(config).search_mask(canvas, hsv=self._hsv) if self.panel.crop_editing() else ImageCropper(config).blue_mask(canvas, hsv=self._hsv)) if step == WorkflowStep.LOAD else
                GreenAreaDetector(config).detect_green_mask(canvas, hsv=self._hsv))
        if scale < 1:
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        display = self._live_dimmed.copy()
        cv2.copyTo(self._live_canvas, mask, display)
        outline = cv2.subtract(cv2.dilate(mask, None), cv2.erode(mask, None))
        display[outline > 0] = (255, 200, 60) if step == WorkflowStep.LOAD else (80, 255, 120)
        self.viewer.set_image(display, preserve_view=preserve_view, source_shape=canvas.shape)
        if box is not None:
            self.viewer.set_crop(canvas.shape, box)
        if step == WorkflowStep.LOAD and config.crop.background_enabled and getattr(self, '_plate_points', None) is not None:
            self._show_plate_outline(image_data)
        self.viewer.set_color_picking(False)
        return True

    def _show_plate_outline(self, image_data):
        if self._plate_points is not None:
            points = roi.transform_points(np.asarray(self._plate_points)-.5, self._frame_matrix[:2])+.5
            color = '#22c55e'
            if self.window._config.data.detect_barcodes:
                if image_data.barcode_mismatch or (image_data.barcode_detected and image_data.barcode_not_found):
                    color = '#ff9800'
                elif not image_data.barcode_detected or not image_data.barcode_read:
                    color = '#9ca3af'  # Not checked yet; do not imply success.
            self.viewer.set_plate_outline(points, color=color)

    def load_preview_label_position(self, rectangle):
        # Transform the original label anchor, not the axis-aligned/clipped
        # rectangle: its top-left changes when the preview is rotated.
        x, y, _, _ = rectangle
        return tuple(roi.transform_points(np.array([[x, y-20]])-.5,
                                          self._frame_matrix[:2])[0]+.5)

    def load_preview_rect(self, rectangle):
        """Map original-photo overlays into the currently displayed Load crop."""
        matrix = getattr(self, '_frame_matrix', None)
        if matrix is None:
            return rectangle
        x, y, width, height = rectangle
        points = roi.transform_points(np.array([[x, y], [x+width, y], [x+width, y+height], [x, y+height]])-.5, matrix[:2])+.5
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        height, width = self._canvas.shape[:2]
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            return None
        return tuple(map(round, (left, top, right-left, bottom-top)))

    def present(self, image_data, step, *, preserve_view=False):
        """Preview drafts cheaply; masks and sampling stay on the visible crop."""
        if step not in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS) or self.panel._current_step != step:
            self.viewer.set_color_picking(False)
            return False
        control = self.panel._color_control
        if self._sample_scope is not None and self._sample_scope != self._sampling_scope():
            self._cancel_pick()
        editing = self.panel.crop_editing()
        config = deepcopy(self.window._config)
        apply_editor_values(config, self.panel.get_current_values())
        scope = (id(image_data), step)
        if (control._picker is not None and not control.pick_button.isChecked() and control.preview.isChecked() and
                self._canvas is not None and getattr(self, '_displayed_scope', None) == scope):
            return self._present_live_color(config, step, preserve_view, image_data)
        self.viewer.clear_crop()
        applied = self.window._config
        draft_analysis = (step == WorkflowStep.PREPROCESS and
                          config.preprocess_roi != applied.preprocess_roi)
        preview_analysis = step == WorkflowStep.PREPROCESS and (draft_analysis or image_data.image is None)
        if not (editing or control.preview.isChecked() or control.pick_button.isChecked() or preview_analysis or step in (WorkflowStep.LOAD, WorkflowStep.PREPROCESS)):
            self.viewer.set_color_picking(False)
            return False
        cropper = ImageCropper(config)
        frame_matrix = np.eye(3)
        old_viewport = self.viewer._view.viewportTransform()
        box = None
        using_result = step == WorkflowStep.PREPROCESS and not editing and not draft_analysis and image_data.image is not None
        if using_result:
            # Includes the applied crop AND registration. Never jump back to the
            # source just because the user toggles a mask or samples a color.
            canvas = image_data.image
            if image_data.plate_transform:
                frame_matrix[:2] = np.asarray(image_data.plate_transform).reshape(2, 3)
            else:
                # Results created before transform metadata was introduced.
                source = self._source(image_data, step, config)
                if source is not None:
                    _, matrix = roi.extract(source, ImageCropper(applied).analysis_roi(source.shape))
                    frame_matrix[:2] = matrix
        else:
            canvas = self._source(image_data, step, config)
            if canvas is None:
                return False
            if step == WorkflowStep.LOAD:
                box = cropper.search_roi(canvas)
                if not editing:
                    # Color settings affect detection, not the visible search area.
                    crop_key = self._source_key[:6]
                    if getattr(self, '_load_crop_key', None) != crop_key:
                        cropped, matrix = cropper.search_image(canvas)
                        self._load_crop_canvas = cropped.copy()
                        self._load_crop_matrix = matrix
                        self._load_crop_key = crop_key
                    canvas = self._load_crop_canvas
                    frame_matrix[:2] = self._load_crop_matrix
            elif editing or preview_analysis:
                box = self._analysis_box(canvas, config)
                if not editing:
                    canvas, matrix = roi.extract(canvas, box)
                    frame_matrix[:2] = matrix
        if canvas is not self._canvas:
            self._canvas = canvas
            self._hsv = None
        mask = None
        if control.preview.isChecked():
            if self._hsv is None:
                self._hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
            mask = ((cropper.search_mask(canvas, hsv=self._hsv) if editing else cropper.blue_mask(canvas, hsv=self._hsv)) if step == WorkflowStep.LOAD else
                    GreenAreaDetector(config).detect_green_mask(canvas, hsv=self._hsv))
        if step == WorkflowStep.PREPROCESS:
            self.viewer.clear_centroids()
            self.viewer.clear_barcode_overlay()
        display = canvas
        if mask is not None:
            display = cv2.convertScaleAbs(canvas, alpha=.3)
            cv2.copyTo(canvas, mask, display)
            outline = cv2.subtract(cv2.dilate(mask, None), cv2.erode(mask, None))
            color = (255, 200, 60) if step == WorkflowStep.LOAD else (80, 255, 120)
            display[outline > 0] = color
        self.viewer.set_image(display, preserve_view=preserve_view)
        if (preserve_view and
                getattr(self, '_frame_scope', None) == scope and self._frame_matrix is not None and
                not np.allclose(self._frame_matrix, frame_matrix)):
            self.viewer.preserve_frame_position(old_viewport, self._frame_matrix @ np.linalg.inv(frame_matrix))
        if not editing:
            self.viewer.straighten_view()
        self._frame_scope, self._frame_matrix = scope, frame_matrix
        if editing:
            self.viewer.set_crop(canvas.shape, box)
            self.viewer._crop_overlay.setEnabled(not self._locked)
            if not preserve_view:
                self.viewer.fit_in_view()
        elif using_result:
            series = self.window._current_series
            self.viewer.set_centroids(
                series.plant_origins(applied.n_clusters) if series is not None else [])
        if step == WorkflowStep.LOAD:
            self._plate_points = self._source_plate_points
            if config.crop.background_enabled:
                self._show_plate_outline(image_data)
        self._prefetch_timer.start()
        self._displayed_scope = scope
        self.viewer.set_color_picking(control.pick_button.isChecked() and not self._locked)
        return True
