"""
Zoomable image viewer widget.

Provides pan/zoom functionality using QGraphicsView.
"""

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsSimpleTextItem, QGraphicsPolygonItem, QApplication
)
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QEvent
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QPen, QBrush, QColor, QMouseEvent, QTransform, QPolygonF, QPalette
import math
import numpy as np
import cv2

from .crop_overlay import CropOverlay
from .theme import scrollbar_stylesheet, is_light

from .masking_tools import (
    MaskTool, MaskOverlay, MaskOverlayItem, BrushCursor, RectanglePreview, BrushStrokePreview,
    draw_brush_stroke, draw_rectangle
)


class DraggableCentroid(QGraphicsEllipseItem):
    """
    A draggable circle representing a plant centroid.
    
    Args:
        x: Center X coordinate.
        y: Center Y coordinate.
        radius: Circle radius.
        index: Index of this centroid in the list.
        callback: Function to call when moved (takes index, new_x, new_y).
    """
    
    def __init__(
        self, 
        x: float, 
        y: float, 
        radius: float = 15, 
        index: int = 0,
        callback = None
    ) -> None:
        # Create ellipse centered at (x, y)
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        
        self._center_x = x
        self._center_y = y
        self._radius = radius
        self._index = index
        self._callback = callback
        
        # Styling - bright green with semi-transparent fill
        self.setPen(QPen(QColor(0, 255, 0), 2))
        self.setBrush(QBrush(QColor(0, 255, 0, 80)))
        
        # Enable dragging
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setToolTip(f"Plant {index + 1}: drag to correct its center")
        self.setZValue(100)  # Above the image
    
    def itemChange(self, change, value):
        """Handle position changes to notify parent."""
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            # Calculate new center from bounding rect
            rect = self.rect()
            new_x = self.scenePos().x() + rect.x() + self._radius
            new_y = self.scenePos().y() + rect.y() + self._radius
            if self._callback:
                self._callback(self._index, new_x, new_y)
        return super().itemChange(change, value)
    
    def get_center(self) -> tuple[float, float]:
        """Get current center coordinates."""
        rect = self.rect()
        return (
            self.scenePos().x() + rect.x() + self._radius,
            self.scenePos().y() + rect.y() + self._radius
        )


class ImageViewer(QWidget):
    """
    Zoomable image preview widget.
    
    Features:
    - Mouse wheel zoom
    - Click and drag pan
    - Draggable centroid markers
    
    Signals:
        zoom_changed: Emitted when zoom level changes (int: percentage).
        centroid_moved: Emitted when a centroid is dragged (index, x, y).
    """
    
    zoom_changed = Signal(int)
    centroid_moved = Signal(int, float, float)  # index, x, y
    crop_changed = Signal(object)
    color_picked = Signal(int, int)
    color_pick_cancelled = Signal()
    editor_help = Signal(str)
    mask_restore_toggled = Signal(bool)
    mask_pan_toggled = Signal(bool)
    mask_diameter_steps = Signal(int)
    mask_modified = Signal()  # Emitted when working mask changes
    mask_available_changed = Signal(bool)

    # Zoom limits (10% to 500%)
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._crop_overlay = None
        self._plate_outline = None
        self._color_picking = False
        self._zoom_factor = 1.0
        self._centroid_items: list[DraggableCentroid] = []
        self._barcode_items: list[QGraphicsRectItem | QGraphicsSimpleTextItem] = []

        # Masking state
        self._mask_tool = MaskTool.NONE
        self._mask_editing_enabled = True
        self._brush_size = 10
        self._applied_mask: np.ndarray | None = None
        self._working_mask: np.ndarray | None = None
        self._mask_overlay: MaskOverlay | None = None
        self._mask_overlay_item: QGraphicsPixmapItem | None = None

        # Drawing state
        self._drawing = False
        self._mask_draw_button = None
        self._last_draw_point: tuple[int, int] | None = None
        self._rect_start_point: QPointF | None = None

        # Visual feedback items
        self._brush_cursor: BrushCursor | None = None
        self._rect_preview: RectanglePreview | None = None
        self._brush_stroke_preview: BrushStrokePreview | None = None  # Lightweight graphics preview

        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Graphics view for image display
        self._scene = QGraphicsScene(self)
        self._view = ZoomableGraphicsView(self._scene, self)
        self._view.zoom_changed.connect(self._on_view_zoom_changed)
        layout.addWidget(self._view)
        
        # Pixmap item for displaying images
        self._pixmap_item: QGraphicsPixmapItem | None = None
    
    def set_crop(self, shape, box):
        self.clear_crop()
        self._crop_overlay = CropOverlay(shape, box, self._pixmap_item)
        self._crop_overlay.changed.connect(self.crop_changed)
        self._crop_overlay.help_requested.connect(self.editor_help)
        self._scene.addItem(self._crop_overlay)
        self._crop_overlay.set_view_scale(math.hypot(self._view.transform().m11(), self._view.transform().m12()))

    def clear_crop(self):
        if self._crop_overlay is not None:
            self._scene.removeItem(self._crop_overlay)
            self._crop_overlay.deleteLater()
            self._crop_overlay = None

    def set_plate_outline(self, points, *, color="#22c55e"):
        self.clear_plate_outline()
        self._plate_outline = QGraphicsPolygonItem(QPolygonF([QPointF(float(x), float(y)) for x, y in points]))
        pen = QPen(QColor(color), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self._plate_outline.setPen(pen)
        self._plate_outline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._plate_outline.setZValue(1)
        self._scene.addItem(self._plate_outline)

    def clear_plate_outline(self):
        if self._plate_outline is not None:
            self._scene.removeItem(self._plate_outline)
            self._plate_outline = None

    def set_color_picking(self, enabled):
        self._color_picking = enabled
        # Item hover cursors and ScrollHandDrag otherwise hide the eyedropper
        # cursor and make the picker look like the ordinary move tool.
        self._view.setInteractive(not enabled)
        drawing = self._mask_tool not in (MaskTool.NONE, MaskTool.MOVE)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag if enabled or drawing else QGraphicsView.DragMode.ScrollHandDrag)
        self._view.viewport().setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_image(self, image: np.ndarray | None, *, preserve_view: bool = False, source_shape=None) -> None:
        """
        Set the displayed image.
        
        Args:
            image: OpenCV image (BGR format) or None to clear.
            preserve_view: Keep zoom and scene center when refreshing an existing image.
        """
        self.clear_crop()
        self.clear_plate_outline()
        previous_transform = self._view.transform()
        previous_scene_rect = self._view.sceneRect()
        previous_scroll = (self._view.horizontalScrollBar().value(), self._view.verticalScrollBar().value())
        preserve_view = preserve_view and self._pixmap_item is not None

        if image is None:
            if self._pixmap_item is not None:
                self._scene.removeItem(self._pixmap_item)
                self._pixmap_item = None
            return
        
        # Convert OpenCV image to QPixmap
        # Ensure array is contiguous for QImage
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Qt accepts BGR directly; avoid a full-frame channel-swap copy.
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        
        pixmap = QPixmap.fromImage(q_image)
        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)
        logical_height, logical_width = source_shape[:2] if source_shape is not None else image.shape[:2]
        self._pixmap_item.setTransform(QTransform.fromScale(logical_width / width, logical_height / height))
        bounds = self._pixmap_item.sceneBoundingRect()
        self._scene.setSceneRect(bounds)

        if preserve_view:
            # Preserve the exact viewport translation. Round-tripping through
            # mapToScene(center)/centerOn accumulates a pixel on every redraw.
            self._view.setSceneRect(previous_scene_rect.united(bounds))
            self._view.setTransform(previous_transform)
            self._view.horizontalScrollBar().setValue(previous_scroll[0])
            self._view.verticalScrollBar().setValue(previous_scroll[1])
            self._update_zoom_from_view()
        else:
            self.fit_in_view()
    
    def preserve_frame_position(self, old_viewport, new_to_old):
        """Keep corresponding image pixels fixed across an affine frame change."""
        matrix = np.asarray(new_to_old, dtype=float).copy()
        # CV transforms address pixel centers; graphics scenes address pixel edges.
        matrix[:2, 2] += .5 - matrix[:2, :2] @ np.array([.5, .5])
        mapping = QTransform(matrix[0, 0], matrix[1, 0], matrix[0, 1],
                             matrix[1, 1], matrix[0, 2], matrix[1, 2])
        self._set_viewport_transform(mapping * old_viewport)

    def straighten_view(self):
        """Align image axes to the screen, retaining zoom and the center pixel."""
        current = self._view.viewportTransform()
        if abs(current.m12()) < 1e-10 and abs(current.m21()) < 1e-10 and current.m11() > 0 and current.m22() > 0:
            return
        inverse, valid = current.inverted()
        if not valid:
            return
        center = QRectF(self._view.viewport().rect()).center()
        pixel = inverse.map(center)
        scale = math.hypot(current.m11(), current.m12())
        self._set_viewport_transform(QTransform(scale, 0, 0, scale,
                                              center.x()-scale*pixel.x(),
                                              center.y()-scale*pixel.y()))

    def _set_viewport_transform(self, desired):
        inverse, valid = desired.inverted()
        if not valid:
            return
        visible = inverse.mapRect(QRectF(self._view.viewport().rect()))
        bounds = self._scene.sceneRect().united(visible)
        self._view.setSceneRect(bounds.adjusted(-visible.width(), -visible.height(), visible.width(), visible.height()))
        self._view.setTransform(desired)
        self._view.horizontalScrollBar().setValue(0)
        self._view.verticalScrollBar().setValue(0)
        self._update_zoom_from_view()

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        """
        Set the displayed image from a QPixmap.
        
        Args:
            pixmap: QPixmap to display or None to clear.
        """
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        
        if pixmap is None:
            return
        
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.sceneBoundingRect())
    
    def fit_in_view(self) -> None:
        """Fit the image to the view."""
        if self._pixmap_item is not None:
            self._view.ensure_pan_space(self._pixmap_item.sceneBoundingRect().center())
            bounds = self._pixmap_item.sceneBoundingRect()
            if self._crop_overlay is not None:
                # Leave screen space for corner and rotation handles at the edges.
                scale = min(self._view.viewport().width()/max(1, bounds.width()),
                            self._view.viewport().height()/max(1, bounds.height()))
                pad = 48/max(scale, .0001)
                bounds = bounds.adjusted(-pad, -pad, pad, pad)
            self._view.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
            self._view.ensure_pan_space(self._pixmap_item.sceneBoundingRect().center())
            self._view.centerOn(self._pixmap_item.sceneBoundingRect().center())
            self._update_zoom_from_view()
    
    def zoom_in(self) -> None:
        """Zoom in by 20%."""
        if self._pixmap_item is not None:
            self._set_zoom(self._zoom_factor * 1.2)
    
    def zoom_out(self) -> None:
        """Zoom out by 20%."""
        if self._pixmap_item is not None:
            self._set_zoom(self._zoom_factor / 1.2)
    
    def _set_zoom(self, factor: float) -> None:
        """Set zoom to a specific factor."""
        factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, factor))
        
        # Calculate scale change
        scale = factor / self._zoom_factor
        self._view.zoom_at(self._view.viewport().rect().center(), scale)
        self._zoom_factor = factor
        
        # Update UI
        self._update_zoom_ui()
    
    def _update_zoom_ui(self) -> None:
        """Update zoom signal."""
        if self._crop_overlay is not None:
            self._crop_overlay.set_view_scale(math.hypot(self._view.transform().m11(), self._view.transform().m12()))
        percentage = int(self._zoom_factor * 100)
        self.zoom_changed.emit(percentage)
    
    def _update_zoom_from_view(self) -> None:
        """Update zoom factor from view transform."""
        transform = self._view.transform()
        self._zoom_factor = math.hypot(transform.m11(), transform.m12())
        self._update_zoom_ui()
    
    def _on_view_zoom_changed(self) -> None:
        """Handle zoom changed from view (mouse wheel)."""
        self._update_zoom_from_view()
    
    def set_centroids(self, positions: list[tuple[float, float]]) -> None:
        """
        Display draggable centroid markers at the given positions.
        
        Args:
            positions: List of (x, y) tuples for centroid centers.
        """
        self.clear_centroids()
        
        for i, (x, y) in enumerate(positions):
            centroid = DraggableCentroid(
                x, y, 
                radius=7,
                index=i,
                callback=self._on_centroid_moved
            )
            self._scene.addItem(centroid)
            self._centroid_items.append(centroid)
    
    def clear_centroids(self) -> None:
        """Remove all centroid markers."""
        for item in self._centroid_items:
            self._scene.removeItem(item)
        self._centroid_items.clear()
    
    def get_centroids(self) -> list[tuple[float, float]]:
        """Get current centroid positions."""
        return [item.get_center() for item in self._centroid_items]
    
    def _on_centroid_moved(self, index: int, x: float, y: float) -> None:
        """Handle centroid drag - emit signal."""
        self.centroid_moved.emit(index, x, y)

    # ========== Barcode Overlay Methods ==========

    def set_barcode_overlay(
        self, 
        rect: tuple[int, int, int, int] | None, 
        text: str, 
        is_mismatch: bool,
        label_position: tuple[float, float] | None = None,
    ) -> None:
        """
        Display a barcode overlay that remains legible at any zoom.
        
        Args:
            rect: (x, y, w, h) tuple or None.
            text: Text to display.
            is_mismatch: True if barcode does NOT match expected (Red).
        """
        self.clear_barcode_overlay()
        
        if rect is None:
            return
            
        x, y, w, h = rect
        color = QColor(255, 0, 0) if is_mismatch else QColor(0, 255, 0)
        
        # 1. (Removed) Rectangle - User requested text only
        
        # 2. Text label (ignore transformations for constant size)
        text_item = QGraphicsSimpleTextItem(text)
        text_item.setBrush(QBrush(color))
        # Position above the box
        # Since we use ItemIgnoresTransformations, the position is in scene coords
        # but the drawing of text happens at 1:1 screen scale.
        text_item.setPos(*(label_position if label_position is not None else (x, y - 20)))
        text_item.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        text_item.setZValue(50)  # Ensure text is on top of image
        
        self._scene.addItem(text_item)
        self._barcode_items.append(text_item)

    def clear_barcode_overlay(self) -> None:
        """Remove barcode overlay items."""
        if hasattr(self, '_barcode_items'):
            for item in self._barcode_items:
                self._scene.removeItem(item)
            self._barcode_items.clear()
        else:
            self._barcode_items = []

    # ========== Masking Methods ==========

    def set_mask_editing_enabled(self, enabled: bool) -> None:
        """Prevent mask writes while a worker reads the current group."""
        self._mask_editing_enabled = enabled

    def _finish_mask_gesture(self):
        if self._mask_overlay_item is not None:
            from PySide6.QtGui import QPainterPath
            self._mask_overlay_item.set_restore_path(QPainterPath())
        button = self._mask_draw_button
        self._mask_draw_button = None
        if button == Qt.MouseButton.RightButton:
            self.mask_restore_toggled.emit(False)

    def set_mask_tool(self, tool: MaskTool, size: int = 10) -> None:
        """
        Activate a masking tool.

        Args:
            tool: The tool to activate.
            size: Brush size in pixels (for brush tools).
        """
        self._mask_tool = tool
        self._brush_size = size

        # Update brush cursor size if active
        if self._brush_cursor is not None:
            self._brush_cursor.set_size(size)
        if self._brush_stroke_preview is not None and tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            self._brush_stroke_preview.set_size(size)
            self._update_restore_preview()

        # Show/hide visual feedback items
        if tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            if self._brush_cursor is None:
                self._brush_cursor = BrushCursor(size)
                self._scene.addItem(self._brush_cursor)
                self._brush_cursor.hide()  # Hidden until mouse enters view
        else:
            if self._brush_cursor is not None:
                self._scene.removeItem(self._brush_cursor)
                self._brush_cursor = None

        # Update view mouse tracking and drag mode
        if tool != MaskTool.NONE and tool != MaskTool.MOVE:
            self._view.setMouseTracking(True)
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self._view.setMouseTracking(False)
            self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def set_mask_data(self, applied_mask: np.ndarray | None, working_mask: np.ndarray | None) -> None:
        """
        Set the mask data for display and editing.

        Args:
            applied_mask: The committed/saved mask.
            working_mask: The currently being edited mask.
        """
        self._applied_mask = applied_mask

        # If no working_mask but there is an applied_mask, initialize working_mask
        if working_mask is None and applied_mask is not None:
            self._working_mask = applied_mask.copy()
        else:
            self._working_mask = working_mask

        # Update overlay
        self._update_mask_overlay()

    def get_working_mask(self) -> np.ndarray | None:
        """Get the current working mask."""
        return self._working_mask

    def clear_working_mask(self) -> None:
        """Reset working mask to applied mask."""
        if self._applied_mask is not None:
            self._working_mask = self._applied_mask.copy()
        else:
            self._working_mask = None

        self._update_mask_overlay()
        self.mask_modified.emit()

    def clear_all_masks(self) -> None:
        """Remove all mask data."""
        self._applied_mask = None
        self._working_mask = None
        self._update_mask_overlay()
        self.mask_modified.emit()

    def _update_restore_preview(self) -> None:
        if self._mask_overlay_item is None:
            return
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        if self._mask_tool == MaskTool.BRUSH_ERASER and self._brush_stroke_preview is not None:
            path = self._brush_stroke_preview.restore_path()
        elif self._mask_tool == MaskTool.RECT_ERASER and self._rect_start_point is not None:
            path.addRect(self._rect_preview.rect())
        self._mask_overlay_item.set_restore_path(path)

    def _update_mask_overlay(self) -> None:
        """Update the mask overlay visualization."""
        self.mask_available_changed.emit(
            self._working_mask is not None and bool(np.any(self._working_mask)))
        if self._working_mask is None or self._pixmap_item is None:
            if self._mask_overlay_item is not None:
                self._scene.removeItem(self._mask_overlay_item)
                self._mask_overlay_item = None
            return

        # Create new overlay if we have mask data
        if self._working_mask is not None and self._pixmap_item is not None:
            if self._mask_overlay is None:
                image_rect = self._pixmap_item.sceneBoundingRect()
                self._mask_overlay = MaskOverlay((int(image_rect.height()), int(image_rect.width())))

            # Just show working_mask (brush preview is now a graphics item overlay)
            self._mask_overlay.set_masks(self._applied_mask, self._working_mask)
            overlay_pixmap = self._mask_overlay.render(cropped=True)

            if overlay_pixmap is not None:
                if self._mask_overlay_item is None:
                    self._mask_overlay_item = MaskOverlayItem(overlay_pixmap)
                    self._mask_overlay_item.setZValue(10)  # Above image, below centroids
                    self._scene.addItem(self._mask_overlay_item)
                else:
                    self._mask_overlay_item.setPixmap(overlay_pixmap)
                self._mask_overlay_item.setPos(*self._mask_overlay.offset)
            elif self._mask_overlay_item is not None:
                self._scene.removeItem(self._mask_overlay_item)
                self._mask_overlay_item = None

    def _scene_to_image_coords(self, scene_pos: QPointF) -> tuple[int, int] | None:
        """
        Convert scene coordinates to image pixel coordinates.

        Args:
            scene_pos: Position in scene coordinates.

        Returns:
            (x, y) in image coordinates, or None if outside image.
        """
        if self._pixmap_item is None:
            return None

        # Scene coordinates are same as image coordinates (pixmap item at origin)
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        # Check bounds
        rect = self._pixmap_item.sceneBoundingRect()
        if 0 <= x < rect.width() and 0 <= y < rect.height():
            return (x, y)

        return None


class ZoomableGraphicsView(QGraphicsView):
    """
    QGraphicsView with mouse wheel zoom and drag pan.
    
    Signals:
        zoom_changed: Emitted when zoom changes.
    """
    
    zoom_changed = Signal()
    
    # Same limits as ImageViewer
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        
        self._current_zoom = 1.0
        self._pan_previous = None
        self._alt_held = False
        self._pan_dragging = False
        QApplication.instance().installEventFilter(self)
        
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints().TextAntialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Pan space is always scrollable. Reserve scrollbar space up front so
        # deferred appearance cannot resize/recenter the viewport on a click.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        palette = self.palette()
        background = (QColor("#e5e5e5")
                      if is_light(palette) else QColor("#202020"))
        self.setBackgroundBrush(background)
        self.setStyleSheet(scrollbar_stylesheet(palette) + f"""
            QGraphicsView {{ border: none; background-color: {background.name()}; }}
            QScrollBar:vertical, QScrollBar:horizontal,
            QAbstractScrollArea::corner {{ background-color: {background.name()}; }}
        """)
    
    def ensure_pan_space(self, center=None):
        """Leave scrollable space around the image even below fit-to-view zoom.

        QGraphicsView otherwise centers a small scene and ignores cursor
        anchoring. Keep the scene's own bounds equal to the image for overlays;
        only this view gets an expanded navigable rectangle.
        """
        if center is None:
            center = self.mapToScene(self.viewport().rect().center())
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        visible.moveCenter(center)
        bounds = self.scene().sceneRect().united(visible)
        self.setSceneRect(bounds.adjusted(-visible.width(), -visible.height(),
                                         visible.width(), visible.height()))
        self.centerOn(center)

    def zoom_at(self, position, factor):
        anchor = self.mapToScene(position)
        self.scale(factor, factor)
        self.ensure_pan_space(anchor)
        # Scrollbar rounding can introduce at most about one screen pixel.
        center_offset = self.mapToScene(self.viewport().rect().center()) - self.mapToScene(position)
        self.centerOn(anchor + center_offset)

    def _temporary_pan(self, active):
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            return
        if active and self._pan_previous is None:
            if (not viewer._mask_editing_enabled or viewer._color_picking
                    or viewer._mask_draw_button is not None
                    or viewer._mask_tool in (MaskTool.NONE, MaskTool.MOVE)):
                return
            self._pan_previous = (viewer._mask_tool, viewer._brush_size)
            viewer.mask_pan_toggled.emit(True)
            viewer.set_mask_tool(MaskTool.MOVE, viewer._brush_size)
        elif not active and self._pan_previous is not None:
            tool, size = self._pan_previous
            self._pan_previous = None
            viewer.mask_pan_toggled.emit(False)
            viewer.set_mask_tool(tool, size)

    def eventFilter(self, watched, event):
        if isinstance(watched, QWidget) and watched.window() == self.window():
            kind = event.type()
            if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.ShortcutOverride):
                if event.key() == Qt.Key.Key_Alt:
                    viewer = self.parent()
                    crop_active = isinstance(viewer, ImageViewer) and viewer._crop_overlay is not None
                    if kind == QEvent.Type.ShortcutOverride:
                        if crop_active or self._pan_previous is not None or (isinstance(viewer, ImageViewer)
                                and viewer._mask_tool not in (MaskTool.NONE, MaskTool.MOVE)):
                            event.accept()
                            return True
                    elif not event.isAutoRepeat():
                        was_panning = self._pan_previous is not None
                        self._alt_held = kind == QEvent.Type.KeyPress
                        if self._alt_held or not self._pan_dragging:
                            self._temporary_pan(self._alt_held)
                        if crop_active or self._pan_previous is not None or was_panning:
                            event.accept()
                            return True
            elif kind == QEvent.Type.WindowDeactivate:
                self._alt_held = False
                self._pan_dragging = False
                self._temporary_pan(False)
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Keep the scene point beneath the cursor fixed at every zoom level."""
        viewer = self.parent()
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                and isinstance(viewer, ImageViewer) and viewer._mask_editing_enabled
                and viewer._mask_tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER)):
            delta = event.angleDelta().y()
            remainder = getattr(self, '_diameter_wheel_remainder', 0) + delta
            steps = int(remainder / 120)
            self._diameter_wheel_remainder = remainder - steps * 120
            if steps:
                viewer.mask_diameter_steps.emit(steps)
            event.accept()
            return
        self._diameter_wheel_remainder = 0
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        current = math.hypot(self.transform().m11(), self.transform().m12())
        requested = current * (1.15 if delta > 0 else 1 / 1.15)
        # Fit can be below MIN_ZOOM for very large photographs: let wheel-up
        # recover smoothly rather than jumping directly to the manual minimum.
        if delta > 0:
            target = max(current, min(self.MAX_ZOOM, requested))
        else:
            target = max(min(current, self.MIN_ZOOM), requested)
        if target == current:
            event.accept()
            return
        self.zoom_at(event.position().toPoint(), target / current)
        self.zoom_changed.emit()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press - start drawing if tool active."""
        # Get parent ImageViewer
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            super().mousePressEvent(event)
            return

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self._alt_held = True
            self._temporary_pan(True)
        if self._pan_previous is not None:
            self._pan_dragging = event.button() == Qt.MouseButton.LeftButton
            super().mousePressEvent(event)
            return

        if viewer._mask_draw_button is not None:
            event.accept()
            return

        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.RightButton and viewer._color_picking:
            super().mousePressEvent(event)
            return

        if viewer._color_picking:
            coords = viewer._scene_to_image_coords(self.mapToScene(event.position().toPoint()))
            if coords is not None:
                viewer.color_picked.emit(*coords)
            event.accept()
            return

        if not viewer._mask_editing_enabled:
            super().mousePressEvent(event)
            return

        # Check if a masking tool is active
        if viewer._mask_tool == MaskTool.NONE or viewer._mask_tool == MaskTool.MOVE:
            super().mousePressEvent(event)
            return

        # Initialize working mask if needed
        if viewer._working_mask is None and viewer._pixmap_item is not None:
            rect = viewer._pixmap_item.sceneBoundingRect()
            viewer._working_mask = np.zeros((int(rect.height()), int(rect.width())), dtype=np.uint8)
            # If there's an applied mask, copy it to working mask
            if viewer._applied_mask is not None:
                viewer._working_mask = viewer._applied_mask.copy()

        # Get scene position
        scene_pos = self.mapToScene(event.position().toPoint())
        img_coords = viewer._scene_to_image_coords(scene_pos)

        if img_coords is None:
            super().mousePressEvent(event)
            return

        viewer._mask_draw_button = event.button()
        if event.button() == Qt.MouseButton.RightButton:
            viewer.mask_restore_toggled.emit(True)

        # Handle tool-specific actions
        if viewer._mask_tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            viewer._drawing = True
            viewer._last_draw_point = img_coords
            # Create lightweight graphics preview
            is_eraser = (viewer._mask_tool == MaskTool.BRUSH_ERASER)
            viewer._brush_stroke_preview = BrushStrokePreview(viewer._brush_size, is_eraser)
            viewer._brush_stroke_preview.add_point(scene_pos)
            viewer._scene.addItem(viewer._brush_stroke_preview)
            viewer._update_restore_preview()

        elif viewer._mask_tool in (MaskTool.RECTANGLE, MaskTool.RECT_ERASER):
            viewer._rect_start_point = scene_pos
            # Create rectangle preview
            if viewer._rect_preview is None:
                viewer._rect_preview = RectanglePreview()
                viewer._scene.addItem(viewer._rect_preview)
            viewer._rect_preview.set_rectangle(scene_pos, scene_pos)
            viewer._rect_preview.setBrush(Qt.BrushStyle.NoBrush if viewer._mask_tool == MaskTool.RECT_ERASER
                                          else QBrush(QColor(255, 255, 255, 50)))
            viewer._rect_preview.show()
            viewer._update_restore_preview()

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move - draw or update preview."""
        # Get parent ImageViewer
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())

        # Update brush cursor position
        if viewer._brush_cursor is not None:
            viewer._brush_cursor.setPos(scene_pos)
            viewer._brush_cursor.show()

        # Handle drawing - update preview for brush (just add points to graphics path)
        if viewer._drawing and viewer._mask_tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            if viewer._brush_stroke_preview is not None:
                viewer._brush_stroke_preview.add_point(scene_pos)
                viewer._update_restore_preview()
            event.accept()
            return

        # Update rectangle preview
        if viewer._rect_start_point is not None and viewer._rect_preview is not None:
            viewer._rect_preview.set_rectangle(viewer._rect_start_point, scene_pos)
            viewer._update_restore_preview()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release - finalize drawing."""
        # Get parent ImageViewer
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            super().mouseReleaseEvent(event)
            return

        if self._pan_previous is not None:
            super().mouseReleaseEvent(event)
            self._pan_dragging = False
            if not self._alt_held:
                self._temporary_pan(False)
            return

        if event.button() != viewer._mask_draw_button:
            super().mouseReleaseEvent(event)
            return

        # Finalize brush drawing - convert graphics preview to mask
        if viewer._drawing:
            viewer._drawing = False

            # Convert graphics preview to numpy mask
            if viewer._brush_stroke_preview is not None:
                # Create temporary mask for this stroke
                rect = viewer._pixmap_item.sceneBoundingRect()
                stroke_mask = np.zeros((int(rect.height()), int(rect.width())), dtype=np.uint8)

                # Rasterize the stroke path to the mask
                # Draw each segment of the stroke
                if viewer._last_draw_point is not None:
                    # Get all points from the path and draw them
                    from PySide6.QtGui import QPainterPath
                    path = viewer._brush_stroke_preview._path

                    # Sample points along the path and draw lines
                    prev_point = None
                    for i in range(path.elementCount()):
                        elem = path.elementAt(i)
                        curr_point = (int(elem.x), int(elem.y))

                        if prev_point is not None:
                            draw_brush_stroke(stroke_mask, prev_point, curr_point, viewer._brush_size, erase=False)
                        else:
                            # Draw first point
                            draw_brush_stroke(stroke_mask, curr_point, curr_point, viewer._brush_size, erase=False)

                        prev_point = curr_point

                # Apply to working_mask
                is_eraser = (viewer._mask_tool == MaskTool.BRUSH_ERASER)
                if is_eraser:
                    # Erase: Remove stroke areas from working_mask
                    viewer._working_mask = cv2.bitwise_and(
                        viewer._working_mask,
                        cv2.bitwise_not(stroke_mask)
                    )
                else:
                    # Draw: Add stroke areas to working_mask
                    viewer._working_mask = cv2.bitwise_or(viewer._working_mask, stroke_mask)

                # Remove graphics preview
                viewer._scene.removeItem(viewer._brush_stroke_preview)
                viewer._brush_stroke_preview = None
                viewer._last_draw_point = None

                # Update overlay and notify
                viewer._update_mask_overlay()
                viewer.mask_modified.emit()

            viewer._finish_mask_gesture()
            event.accept()
            return

        # Finalize rectangle
        if viewer._rect_start_point is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            start_coords = viewer._scene_to_image_coords(viewer._rect_start_point)
            end_coords = viewer._scene_to_image_coords(scene_pos)

            if start_coords is not None and end_coords is not None:
                is_eraser = (viewer._mask_tool == MaskTool.RECT_ERASER)
                draw_rectangle(viewer._working_mask, start_coords, end_coords, erase=is_eraser)
                viewer._update_mask_overlay()
                viewer.mask_modified.emit()

            # Hide rectangle preview
            if viewer._rect_preview is not None:
                viewer._rect_preview.hide()

            viewer._rect_start_point = None
            viewer._finish_mask_gesture()
            event.accept()
            return

        viewer._finish_mask_gesture()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        viewer = self.parent()
        if event.key() == Qt.Key.Key_Escape and isinstance(viewer, ImageViewer) and viewer._color_picking:
            viewer.set_color_picking(False)
            viewer.color_pick_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving view - hide brush cursor."""
        viewer = self.parent()
        if isinstance(viewer, ImageViewer) and viewer._brush_cursor is not None:
            viewer._brush_cursor.hide()
        super().leaveEvent(event)
