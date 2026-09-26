"""Movable crop rectangle with screen-sized handles and gesture-local angle snaps."""
import math
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QPen, QBrush, QPainterPath, QPolygonF, QTransform, QCursor, QPixmap, QPainter, QFontMetricsF
from PySide6.QtWidgets import QGraphicsObject, QGraphicsRectItem, QGraphicsSimpleTextItem, QGraphicsPathItem
from ..preprocessing import roi


class AngleSnap:
    """Latch near 15° stops; do not recapture a released stop in the same gesture."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.locked = None
        self.ignored = set()
        self.previous = None

    def update(self, angle):
        previous, self.previous = self.previous, angle
        if self.locked is not None:
            delta = (angle-self.locked+180) % 360-180
            if abs(delta) <= 4:
                return angle-delta
            self.ignored.add(self.locked % 360)
            self.locked = None
        target = round(angle/15)*15
        if abs(target-angle) <= 2 and target % 360 not in self.ignored:
            self.locked = target
            self.ignored.clear()
            return target
        if previous is not None:
            # Mouse events can jump completely over the capture band. Passing
            # those stops also suppresses capture when the user reverses.
            end = previous + (angle-previous+180) % 360-180
            low, high = sorted((previous, end))
            for step in range(math.floor(low/15)+1, math.ceil(high/15)):
                self.ignored.add((step*15) % 360)
        return angle


class CropHandle(QGraphicsRectItem):
    def __init__(self, owner, kind):
        super().__init__(-5, -5, 10, 10, owner)
        self.owner, self.kind = owner, kind
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setBrush(QBrush(QColor('#ffffff')))
        self.setPen(QPen(QColor('#2563eb'), 1.5))
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor if kind == 'rotate' else
                       Qt.CursorShape.SizeFDiagCursor if kind in ('nw', 'se') else
                       Qt.CursorShape.SizeBDiagCursor if kind in ('ne', 'sw') else
                       Qt.CursorShape.SizeHorCursor if kind in ('w', 'e') else Qt.CursorShape.SizeVerCursor)

    def paint(self, painter, option, widget=None):
        if self.kind == 'rotate':
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            painter.drawEllipse(self.rect())
        else:
            super().paint(painter, option, widget)

    def hoverEnterEvent(self, event):
        self.owner.help_requested.emit('Drag to rotate. Snaps every 15°. Drag past a snap to release it; returning skips that angle until another snap or a new drag.' if self.kind == 'rotate' else 'Drag to resize the crop. Drag inside the box to move it.')
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.owner.help_requested.emit('')
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        # Let the view handle Alt-drag as panning, including over handles.
        if (event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.owner.begin_drag(self.kind, event.scenePos())
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        self.owner.drag_to(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.owner.end_drag()
        event.accept()


class CornerRotationZone(QGraphicsPathItem):
    """Invisible screen-sized hit region outside a corner's resize handle."""
    def __init__(self, owner, corner):
        super().__init__(owner)
        self.owner, self.corner = owner, corner
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(1)  # Resize handles retain priority in their own hit area.
        self.setAcceptHoverEvents(True)

    def _set_rotation_cursor(self, angle):
        pixmap = QPixmap(40, 40)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.translate(20, 20)
        painter.rotate(angle)
        painter.translate(-16, -16)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for color, width in ((QColor('white'), 4), (QColor('#222222'), 2)):
            painter.setPen(QPen(color, width))
            painter.drawArc(QRectF(6, 6, 20, 20), 50*16, 270*16)
        painter.setBrush(QColor('#222222'))
        painter.drawPolygon(QPolygonF([QPointF(26, 4), QPointF(25, 13), QPointF(18, 8)]))
        painter.end()
        self.setCursor(QCursor(pixmap, 20, 20))

    def set_angle(self, angle):
        self._set_rotation_cursor(angle + {"nw": -90, "ne": 0, "se": 90, "sw": 180}[self.corner])
        region = QPainterPath()
        region.addEllipse(QRectF(-24, -24, 48, 48))
        hole = QPainterPath()
        hole.addEllipse(QRectF(-9, -9, 18, 18))
        inside = QPainterPath()
        x = 0 if self.corner in ('nw', 'sw') else -24
        y = 0 if self.corner in ('nw', 'ne') else -24
        inside.addRect(QRectF(x, y, 24, 24))
        self.setPath(QTransform().rotate(angle).map(region.subtracted(hole).subtracted(inside)))

    def hoverEnterEvent(self, event):
        self.owner.help_requested.emit('Drag outside the corner to rotate. Snaps every 15°; keep dragging to release a snap.')
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.owner.help_requested.emit('')
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        # Let the view handle Alt-drag as panning, including over handles.
        if (event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.owner.begin_drag('rotate', event.scenePos())
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        self.owner.drag_to(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.owner.end_drag()
        event.accept()


class AngleLabel(QGraphicsSimpleTextItem):
    """Choose text contrast from the actual image underneath the label."""

    def paint(self, painter, option, widget=None):
        owner = self.parentItem()
        source = owner.image_item
        if source is not None:
            image = source.pixmap().toImage()
            views = self.scene().views()
            view = next((v for v in views if v.viewport() is widget), views[0] if views else None)
            if view is not None:
                device = self.deviceTransform(view.viewportTransform())
                inverse, _ = view.viewportTransform().inverted()
                to_scene = lambda point: inverse.map(device.map(point))
                background = view.backgroundBrush().color()
            else:
                to_scene = lambda point: self.mapToScene(point / owner._scale)
                background = QColor('#808080')
            rect = self.boundingRect()
            crop = owner.shape()
            luminances = []
            for y in (.2, .5, .8):
                for x in (.1, .3, .5, .7, .9):
                    scene_point = to_scene(QPointF(rect.width() * x, rect.height() * y))
                    pixel = source.mapFromScene(scene_point)
                    if 0 <= pixel.x() < image.width() and 0 <= pixel.y() < image.height():
                        color = image.pixelColor(int(pixel.x()), int(pixel.y()))
                        shade = 1 if crop.contains(owner.mapFromScene(scene_point)) else 140 / 255
                    else:
                        color, shade = background, 1
                    rgb = [channel * shade / 255 for channel in color.getRgb()[:3]]
                    linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
                    luminances.append(sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722))))
            # Equal black/white contrast occurs at relative luminance 0.179.
            dark_text = float(np.median(luminances)) > .179
            foreground = QColor('#000000' if dark_text else '#ffffff')
            outline = QPen(QColor('#ffffff' if dark_text else '#000000'), .75)
            outline.setCosmetic(True)
            if self.brush().color() != foreground:
                self.setBrush(foreground)
            if self.pen() != outline:
                self.setPen(outline)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = QPainterPath()
        text.addText(QPointF(0, QFontMetricsF(self.font()).ascent()), self.font(), self.text())
        # Fill after outlining so the halo cannot cover the thin glyph strokes.
        painter.strokePath(text, self.pen())
        painter.fillPath(text, self.brush())
        painter.restore()


class CropOverlay(QGraphicsObject):
    changed = Signal(object)
    help_requested = Signal(str)

    def __init__(self, image_shape, box, image_item=None):
        super().__init__()
        self.image_shape = image_shape
        self.image_item = image_item
        self.box = tuple(box)
        self._scale = 1.
        self._drag = None
        self._snap = AngleSnap()
        self.setZValue(30)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.handles = {name: CropHandle(self, name) for name in ('nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'rotate')}
        self.rotation_zones = {name: CornerRotationZone(self, name) for name in ('nw', 'ne', 'se', 'sw')}
        self.label = AngleLabel(self)
        self.label.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.label.setBrush(QColor('#ffffff'))
        self.label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._layout_handles()

    def boundingRect(self):
        h, w = self.image_shape[:2]
        pad = 90/max(self._scale, .0001)
        return QRectF(0, 0, w, h).united(self.shape().boundingRect()).adjusted(-pad, -pad, pad, pad)

    def shape(self):
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(*p) for p in roi.corners(self.box, self.image_shape)]))
        path.closeSubpath()
        return path

    def paint(self, painter, option, widget=None):
        h, w = self.image_shape[:2]
        outside = QPainterPath()
        outside.addRect(QRectF(0, 0, w, h))
        painter.fillPath(outside.subtracted(self.shape()), QColor(0, 0, 0, 115))
        pen = QPen(QColor('#60a5fa'), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.shape())
        painter.drawLine(self.handles['n'].pos(), self.handles['rotate'].pos())

    def set_view_scale(self, scale):
        self.prepareGeometryChange()
        self._scale = max(scale, .0001)
        self._layout_handles()
        self.update()

    def _layout_handles(self):
        points = roi.corners(self.box, self.image_shape)
        positions = dict(zip(('nw', 'ne', 'se', 'sw'), points))
        positions.update(n=(points[0]+points[1])/2, e=(points[1]+points[2])/2,
                         s=(points[2]+points[3])/2, w=(points[3]+points[0])/2)
        angle = math.radians(self.box[4])
        positions['rotate'] = positions['n'] + np.array([math.sin(angle), -math.cos(angle)])*32/self._scale
        for name, position in positions.items():
            self.handles[name].setPos(QPointF(*position))
        for name, zone in self.rotation_zones.items():
            zone.setPos(QPointF(*positions[name]))
            zone.set_angle(self.box[4])
        self.handles['rotate'].setCursor(self.rotation_zones['ne'].cursor())
        self.label.setText(f'{self.box[4] % 360:.1f}°')
        self.label.setPos(self.handles['rotate'].pos() + QPointF(12/self._scale, -9/self._scale))

    def begin_drag(self, kind, position):
        self._drag = kind
        self._start = QPointF(position)
        self._initial = self.box
        self._snap.reset()
        cx, cy, *_ = roi.pixel_box(self.box, self.image_shape)
        self._pointer_angle = math.degrees(math.atan2(position.y()-cy, position.x()-cx))
        self._raw_angle = self.box[4]

    def drag_to(self, position):
        if self._drag is None:
            return
        h, w = self.image_shape[:2]
        cx, cy, width, height, angle = roi.pixel_box(self._initial, self.image_shape)
        dx, dy = position.x()-self._start.x(), position.y()-self._start.y()
        if self._drag == 'move':
            box = ((cx+dx)/w, (cy+dy)/h, width/w, height/h, angle)
            # Translation can be outside before being clamped.
            box = (min(1., max(0., box[0])), min(1., max(0., box[1])), *box[2:])
        elif self._drag == 'rotate':
            pointer = math.degrees(math.atan2(position.y()-cy, position.x()-cx))
            self._raw_angle += (pointer-self._pointer_angle+180) % 360-180
            self._pointer_angle = pointer
            box = (*self._initial[:4], self._snap.update(self._raw_angle))
        else:
            a = math.radians(angle)
            c, s = math.cos(a), math.sin(a)
            ux, uy = c*dx+s*dy, -s*dx+c*dy
            xsign = -1 if 'w' in self._drag else 1 if 'e' in self._drag else 0
            ysign = -1 if 'n' in self._drag else 1 if 's' in self._drag else 0
            nw = max(8., width+xsign*ux)
            nh = max(8., height+ysign*uy)
            shift_x, shift_y = xsign*(nw-width)/2, ysign*(nh-height)/2
            cx += c*shift_x-s*shift_y
            cy += s*shift_x+c*shift_y
            box = (min(1., max(0., cx/w)), min(1., max(0., cy/h)), nw/w, nh/h, angle)
        self.prepareGeometryChange()
        self.box = roi.fit_inside(box, self.image_shape)
        self._layout_handles()
        self.update()

    def end_drag(self):
        if self._drag is not None:
            self._drag = None
            self._snap.reset()
            if self.box != self._initial:
                self.changed.emit(self.box)

    def mousePressEvent(self, event):
        # Let the view handle Alt-drag as panning, including over handles.
        if (event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.begin_drag('move', event.scenePos())
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        self.drag_to(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event):
        self.end_drag()
        event.accept()
