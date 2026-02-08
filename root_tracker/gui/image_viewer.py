"""
Zoomable image viewer widget.

Provides pan/zoom functionality using QGraphicsView.
"""

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QWheelEvent
import numpy as np
import cv2


class ImageViewer(QWidget):
    """
    Zoomable image preview widget.
    
    Features:
    - Mouse wheel zoom
    - Click and drag pan
    
    Signals:
        zoom_changed: Emitted when zoom level changes (int: percentage).
    """
    
    zoom_changed = Signal(int)
    
    # Zoom limits (10% to 500%)
    MIN_ZOOM = 0.1
    MAX_ZOOM = 5.0
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        self._zoom_factor = 1.0
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
