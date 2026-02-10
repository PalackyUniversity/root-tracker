"""
Zoomable image viewer widget.

Provides pan/zoom functionality using QGraphicsView.
"""

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QGraphicsEllipseItem
)
from PySide6.QtCore import Qt, Signal, QPointF
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QPen, QBrush, QColor, QMouseEvent
import numpy as np
import cv2

from .masking_tools import (
    MaskTool, MaskOverlay, BrushCursor, RectanglePreview, BrushStrokePreview,
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
        super().__init__(x - radius, y - radius, radius * 2, radius * 2)
        
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
    mask_modified = Signal()  # Emitted when working mask changes

    # Zoom limits (10% to 500%)
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._zoom_factor = 1.0
        self._centroid_items: list[DraggableCentroid] = []

        # Masking state
        self._mask_tool = MaskTool.NONE
        self._brush_size = 10
        self._applied_mask: np.ndarray | None = None
        self._working_mask: np.ndarray | None = None
        self._mask_overlay: MaskOverlay | None = None
        self._mask_overlay_item: QGraphicsPixmapItem | None = None

        # Drawing state
        self._drawing = False
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
    
    def set_image(self, image: np.ndarray | None) -> None:
        """
        Set the displayed image.
        
        Args:
            image: OpenCV image (BGR format) or None to clear.
        """
        # Clear existing
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        
        if image is None:
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
            # Color (BGR to RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        
        # Fit to view on first load
        self.fit_in_view()
    
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
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
    
    def fit_in_view(self) -> None:
        """Fit the image to the view."""
        if self._pixmap_item is not None:
            self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._update_zoom_from_view()
    
    def zoom_in(self) -> None:
        """Zoom in by 20%."""
        self._set_zoom(self._zoom_factor * 1.2)
    
    def zoom_out(self) -> None:
        """Zoom out by 20%."""
        self._set_zoom(self._zoom_factor / 1.2)
    
    def _set_zoom(self, factor: float) -> None:
        """Set zoom to a specific factor."""
        factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, factor))
        
        # Calculate scale change
        scale = factor / self._zoom_factor
        self._view.scale(scale, scale)
        self._zoom_factor = factor
        
        # Update UI
        self._update_zoom_ui()
    
    def _update_zoom_ui(self) -> None:
        """Update zoom signal."""
        percentage = int(self._zoom_factor * 100)
        self.zoom_changed.emit(percentage)
    
    def _update_zoom_from_view(self) -> None:
        """Update zoom factor from view transform."""
        transform = self._view.transform()
        self._zoom_factor = transform.m11()  # Horizontal scale
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
                radius=15, 
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

    # ========== Masking Methods ==========

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

    def _update_mask_overlay(self) -> None:
        """Update the mask overlay visualization."""
        # Remove existing overlay
        if self._mask_overlay_item is not None:
            self._scene.removeItem(self._mask_overlay_item)
            self._mask_overlay_item = None

        # Create new overlay if we have mask data
        if self._working_mask is not None and self._pixmap_item is not None:
            if self._mask_overlay is None:
                image_rect = self._pixmap_item.boundingRect()
                self._mask_overlay = MaskOverlay((int(image_rect.height()), int(image_rect.width())))

            # Just show working_mask (brush preview is now a graphics item overlay)
            self._mask_overlay.set_masks(self._applied_mask, self._working_mask)
            overlay_pixmap = self._mask_overlay.render()

            if overlay_pixmap is not None:
                self._mask_overlay_item = QGraphicsPixmapItem(overlay_pixmap)
                self._mask_overlay_item.setZValue(10)  # Above image, below centroids
                self._scene.addItem(self._mask_overlay_item)

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
        rect = self._pixmap_item.boundingRect()
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
        
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints().TextAntialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming with limits."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Get current zoom from transform
        current_zoom = self.transform().m11()

        if event.angleDelta().y() > 0:
            # Zoom in - check max limit
            if current_zoom * zoom_in_factor <= self.MAX_ZOOM:
                self.scale(zoom_in_factor, zoom_in_factor)
        else:
            # Zoom out - check min limit
            if current_zoom * zoom_out_factor >= self.MIN_ZOOM:
                self.scale(zoom_out_factor, zoom_out_factor)

        self.zoom_changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press - start drawing if tool active."""
        # Get parent ImageViewer
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            super().mousePressEvent(event)
            return

        # Only handle left button
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # Check if a masking tool is active
        if viewer._mask_tool == MaskTool.NONE or viewer._mask_tool == MaskTool.MOVE:
            super().mousePressEvent(event)
            return

        # Initialize working mask if needed
        if viewer._working_mask is None and viewer._pixmap_item is not None:
            rect = viewer._pixmap_item.boundingRect()
            viewer._working_mask = np.zeros((int(rect.height()), int(rect.width())), dtype=np.uint8)
            # If there's an applied mask, copy it to working mask
            if viewer._applied_mask is not None:
                viewer._working_mask = viewer._applied_mask.copy()

        # Get scene position
        scene_pos = self.mapToScene(event.pos())
        img_coords = viewer._scene_to_image_coords(scene_pos)

        if img_coords is None:
            super().mousePressEvent(event)
            return

        # Handle tool-specific actions
        if viewer._mask_tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            viewer._drawing = True
            viewer._last_draw_point = img_coords
            # Create lightweight graphics preview
            is_eraser = (viewer._mask_tool == MaskTool.BRUSH_ERASER)
            viewer._brush_stroke_preview = BrushStrokePreview(viewer._brush_size, is_eraser)
            viewer._brush_stroke_preview.add_point(scene_pos)
            viewer._scene.addItem(viewer._brush_stroke_preview)

        elif viewer._mask_tool in (MaskTool.RECTANGLE, MaskTool.RECT_ERASER):
            viewer._rect_start_point = scene_pos
            # Create rectangle preview
            if viewer._rect_preview is None:
                viewer._rect_preview = RectanglePreview()
                viewer._scene.addItem(viewer._rect_preview)
            viewer._rect_preview.set_rectangle(scene_pos, scene_pos)
            viewer._rect_preview.show()

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move - draw or update preview."""
        # Get parent ImageViewer
        viewer = self.parent()
        if not isinstance(viewer, ImageViewer):
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        # Update brush cursor position
        if viewer._brush_cursor is not None:
            viewer._brush_cursor.setPos(scene_pos)
            viewer._brush_cursor.show()

        # Handle drawing - update preview for brush (just add points to graphics path)
        if viewer._drawing and viewer._mask_tool in (MaskTool.BRUSH, MaskTool.BRUSH_ERASER):
            if viewer._brush_stroke_preview is not None:
                viewer._brush_stroke_preview.add_point(scene_pos)
            event.accept()
            return

        # Update rectangle preview
        if viewer._rect_start_point is not None and viewer._rect_preview is not None:
            viewer._rect_preview.set_rectangle(viewer._rect_start_point, scene_pos)
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

        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        # Finalize brush drawing - convert graphics preview to mask
        if viewer._drawing:
            viewer._drawing = False

            # Convert graphics preview to numpy mask
            if viewer._brush_stroke_preview is not None:
                # Create temporary mask for this stroke
                rect = viewer._pixmap_item.boundingRect()
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

            event.accept()
            return

        # Finalize rectangle
        if viewer._rect_start_point is not None:
            scene_pos = self.mapToScene(event.pos())
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
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        """Handle mouse leaving view - hide brush cursor."""
        viewer = self.parent()
        if isinstance(viewer, ImageViewer) and viewer._brush_cursor is not None:
            viewer._brush_cursor.hide()
        super().leaveEvent(event)
