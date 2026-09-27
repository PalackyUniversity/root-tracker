"""
Masking tools for interactive root removal in Track tab.

Provides drawing tools for creating and editing masks to remove detected roots.
"""

from enum import Enum
from typing import Optional
import numpy as np
import cv2
from PySide6.QtWidgets import QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPixmapItem
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QImage, QPixmap, QPainter, QPolygon, QPainterPath, QPainterPathStroker


class MaskTool(Enum):
    """Available masking tools."""
    NONE = "none"
    BRUSH = "brush"
    RECTANGLE = "rectangle"
    BRUSH_ERASER = "brush_eraser"
    RECT_ERASER = "rect_eraser"
    MOVE = "move"
    BRUSH_SELECT = "brush_select"
    RECT_SELECT = "rect_select"


class MaskOverlay:
    """Render exclusions with a translucent fill and dashed outline."""

    def __init__(self, image_shape: tuple[int, int]) -> None:
        """
        Initialize mask overlay.

        Args:
            image_shape: (height, width) of the image.
        """
        self.image_shape = image_shape
        self.offset = (0, 0)
        self._applied_mask: Optional[np.ndarray] = None
        self._working_mask: Optional[np.ndarray] = None

    def set_masks(self, applied_mask: Optional[np.ndarray], working_mask: Optional[np.ndarray]) -> None:
        """
        Update the mask layers.

        Args:
            applied_mask: The committed/saved mask.
            working_mask: The currently being edited mask.
        """
        self._applied_mask = applied_mask
        self._working_mask = working_mask

    def render(self, *, cropped: bool = False) -> Optional[QPixmap]:
        """
        Render the mask overlay as a QPixmap.

        Returns:
            QPixmap with colored overlay, or None if no masks.
        """
        self.offset = (0, 0)
        if self._working_mask is None:
            return None

        mask = self._working_mask
        if cropped:
            # Small edits need a small scene item, not a full-photo RGBA upload.
            x, y, width, height = cv2.boundingRect(mask)
            if not width or not height:
                return None
            left, top = max(0, x - 2), max(0, y - 2)
            right, bottom = min(mask.shape[1], x + width + 2), min(mask.shape[0], y + height + 2)
            self.offset = (left, top)
            mask = mask[top:bottom, left:right]
        mask = np.ascontiguousarray(mask)
        height, width = mask.shape
        # An indexed image avoids allocating/filling a four-channel NumPy image
        # for every stroke. Every nonzero brush value is an exclusion.
        qimage = QImage(mask.data, width, height, mask.strides[0], QImage.Format.Format_Indexed8)
        qimage.setColorTable([QColor(0, 0, 0, 0).rgba()] + [QColor(0, 0, 0, 230).rgba()] * 255)
        pixmap = QPixmap.fromImage(qimage)

        # Approximate only collinear contour points, preserving the mask shape
        # and holes. Qt draws complete dashed paths in native code.
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 255, 255, 200), 2)
        pen.setDashPattern([5, 5])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for contour in contours:
            polygon = QPolygon([QPoint(int(x), int(y)) for x, y in contour[:, 0]])
            painter.drawPolygon(polygon)
        painter.end()
        return pixmap


class MaskOverlayItem(QGraphicsPixmapItem):
    """Reveal the image during restore without rerasterizing the mask."""

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self._restore_path = QPainterPath()

    def set_restore_path(self, scene_path):
        path = self.mapFromScene(scene_path)
        dirty = self._restore_path.boundingRect().united(path.boundingRect())
        self._restore_path = path
        self.update(dirty.adjusted(-2, -2, 2, 2))

    def paint(self, painter, option, widget=None):
        painter.save()
        if not self._restore_path.isEmpty():
            visible = QPainterPath()
            visible.addRect(self.boundingRect())
            painter.setClipPath(visible.subtracted(self._restore_path), Qt.ClipOperation.IntersectClip)
        super().paint(painter, option, widget)
        painter.restore()


class BrushCursor(QGraphicsEllipseItem):
    """Visual cursor showing brush size."""

    def __init__(self, size: int) -> None:
        """
        Initialize brush cursor.

        Args:
            size: Brush diameter in pixels.
        """
        super().__init__(-size / 2, -size / 2, size, size)

        # Semi-transparent circle outline
        pen = QPen(QColor(255, 255, 255, 200))
        pen.setWidth(1)
        pen.setCosmetic(True)  # Width doesn't scale with zoom
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        self.setZValue(1000)  # Always on top
        # Don't use ItemIgnoresTransformations - we want it to scale with zoom

    def set_size(self, size: int) -> None:
        """Update brush size."""
        self.setRect(-size / 2, -size / 2, size, size)


class RectanglePreview(QGraphicsRectItem):
    """Preview rectangle during drag operation."""

    def __init__(self) -> None:
        """Initialize rectangle preview."""
        super().__init__()

        # Semi-transparent white outline
        pen = QPen(QColor(255, 255, 255, 200))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

        brush = QBrush(QColor(255, 255, 255, 50))
        self.setBrush(brush)

        self.setZValue(999)  # Just below brush cursor

    def set_rectangle(self, start: QPointF, end: QPointF) -> None:
        """
        Update rectangle from two corner points.

        Args:
            start: Starting corner in scene coordinates.
            end: Ending corner in scene coordinates.
        """
        x = min(start.x(), end.x())
        y = min(start.y(), end.y())
        w = abs(end.x() - start.x())
        h = abs(end.y() - start.y())

        self.setRect(QRectF(x, y, w, h))


class BrushStrokePreview(QGraphicsItem):
    """Preview brush stroke during drag operation - lightweight graphics item."""

    def __init__(self, brush_size: int, is_eraser: bool = False) -> None:
        """Initialize brush stroke preview."""
        super().__init__()
        from PySide6.QtGui import QPainterPath, QPainter

        self._path = QPainterPath()
        self._brush_size = brush_size
        self._is_eraser = is_eraser
        self.setZValue(999)  # Just below brush cursor

    def set_size(self, size: int) -> None:
        self.prepareGeometryChange()
        self._brush_size = size
        self.update()

    def add_point(self, point: QPointF) -> None:
        """Add a point to the stroke path."""
        self.prepareGeometryChange()
        if self._path.elementCount() == 0:
            self._path.moveTo(point)
        else:
            self._path.lineTo(point)
        self.update()

    def restore_path(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(self._brush_size)
        stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        path = stroker.createStroke(self._path)
        if self._path.elementCount() == 1:
            point = self._path.elementAt(0)
            path.addEllipse(QPointF(point.x, point.y), self._brush_size / 2, self._brush_size / 2)
        return path

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle for this item."""
        from PySide6.QtCore import QMarginsF
        margin = self._brush_size / 2 + 2
        return self._path.boundingRect().marginsAdded(QMarginsF(margin, margin, margin, margin))

    def paint(self, painter, option, widget=None) -> None:
        """Paint the brush stroke preview."""
        from PySide6.QtGui import QPainter

        if self._is_eraser:
            return  # The exclusion overlay clips out the restore stroke.
        pen = QPen(QColor(0, 0, 0, 200))

        pen.setWidth(self._brush_size)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPath(self._path)


def draw_brush_stroke(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    size: int,
    erase: bool = False
) -> None:
    """
    Draw a brush stroke on the mask.

    Modifies mask in-place by drawing a line between start and end points.

    Args:
        mask: Binary mask array (uint8).
        start: (x, y) starting point in image coordinates.
        end: (x, y) ending point in image coordinates.
        size: Brush diameter in pixels.
        erase: If True, erase (set to 0), otherwise draw (set to 255).
    """
    color = 0 if erase else 255
    thickness = max(1, size)

    cv2.line(mask, start, end, color, thickness, cv2.LINE_AA)


def draw_rectangle(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    erase: bool = False
) -> None:
    """
    Draw a filled rectangle on the mask.

    Modifies mask in-place.

    Args:
        mask: Binary mask array (uint8).
        start: (x, y) top-left corner in image coordinates.
        end: (x, y) bottom-right corner in image coordinates.
        erase: If True, erase (set to 0), otherwise draw (set to 255).
    """
    color = 0 if erase else 255

    # Ensure start is top-left, end is bottom-right
    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
    x2, y2 = max(start[0], end[0]), max(start[1], end[1])

    # Draw filled rectangle
    cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)  # -1 = filled
