"""
Masking tools for interactive root removal in Track tab.

Provides drawing tools for creating and editing masks to remove detected roots.
"""

from enum import Enum
from typing import Optional
import numpy as np
import cv2
from PySide6.QtWidgets import QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QBrush, QColor, QImage, QPixmap


class MaskTool(Enum):
    """Available masking tools."""
    NONE = "none"
    BRUSH = "brush"
    RECTANGLE = "rectangle"
    BRUSH_ERASER = "brush_eraser"
    RECT_ERASER = "rect_eraser"


class MaskOverlay:
    """
    Manages mask visualization with two layers.

    The overlay shows:
    - Black (50% opacity): New masked areas in working_mask
    - Green (50% opacity): Areas where applied_mask is being removed
    """

    def __init__(self, image_shape: tuple[int, int]) -> None:
        """
        Initialize mask overlay.

        Args:
            image_shape: (height, width) of the image.
        """
        self.image_shape = image_shape
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

    def render(self) -> Optional[QPixmap]:
        """
        Render the mask overlay as a QPixmap.

        Returns:
            QPixmap with colored overlay, or None if no masks.
        """
        if self._working_mask is None:
            return None

        height, width = self.image_shape

        # Create RGBA overlay image
        overlay = np.zeros((height, width, 4), dtype=np.uint8)

        # Show all working_mask areas as black (no green for removed areas)
        overlay[self._working_mask > 0] = [0, 0, 0, 230]  # Black, 90% opacity

        # Draw white dashed borderline around working_mask areas (both outer and inner contours)
        if np.any(self._working_mask > 0):
            # Find all contours (including holes/inner contours)
            contours, _ = cv2.findContours(self._working_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # Draw dashed contours
            for contour in contours:
                # Draw dashed line by sampling points along the contour
                dash_length = 10
                gap_length = 10
                total_length = dash_length + gap_length

                for i in range(0, len(contour), total_length):
                    # Draw dash
                    end_idx = min(i + dash_length, len(contour))
                    if end_idx > i:
                        pts = contour[i:end_idx]
                        for pt_idx in range(len(pts) - 1):
                            pt1 = tuple(pts[pt_idx][0])
                            pt2 = tuple(pts[pt_idx + 1][0])
                            cv2.line(overlay, pt1, pt2, (255, 255, 255, 200), 2)

        # Convert to QImage then QPixmap
        qimage = QImage(
            overlay.data,
            width,
            height,
            4 * width,  # bytes per line
            QImage.Format.Format_RGBA8888
        )

        return QPixmap.fromImage(qimage)


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

    def add_point(self, point: QPointF) -> None:
        """Add a point to the stroke path."""
        if self._path.elementCount() == 0:
            self._path.moveTo(point)
        else:
            self._path.lineTo(point)
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:
        """Return bounding rectangle for this item."""
        from PySide6.QtCore import QMarginsF
        margin = self._brush_size / 2 + 2
        return self._path.boundingRect().marginsAdded(QMarginsF(margin, margin, margin, margin))

    def paint(self, painter, option, widget=None) -> None:
        """Paint the brush stroke preview."""
        from PySide6.QtGui import QPainter

        # Set up pen for stroke
        if self._is_eraser:
            # Green for eraser preview
            pen = QPen(QColor(0, 255, 0, 200))
        else:
            # Black for brush preview
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
