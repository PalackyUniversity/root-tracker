"""Normalized rotated rectangles shared by the pipeline and crop editor.

(cx, cy, width, height, clockwise_degrees) uses image-edge coordinates.
Extraction maps the rectangle's local axes into an upright output image.
"""
import math
import cv2
import numpy as np


def validate(box):
    if box is None:
        return None
    if len(box) != 5 or not all(math.isfinite(v) for v in box):
        raise ValueError('A crop needs five finite values: center, size and angle')
    cx, cy, w, h, angle = map(float, box)
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0):
        raise ValueError('Crop center must be inside the image and its size must be positive')
    return cx, cy, w, h, angle


def from_bounds(x, y, width, height, shape):
    h, w = shape[:2]
    return ((x+width/2)/w, (y+height/2)/h, width/w, height/h, 0.)


def pixel_box(box, shape):
    cx, cy, width, height, angle = validate(box)
    h, w = shape[:2]
    return cx*w, cy*h, width*w, height*h, angle


def corners(box, shape):
    cx, cy, width, height, angle = pixel_box(box, shape)
    a = math.radians(angle)
    basis = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    points = np.array([[-width/2, -height/2], [width/2, -height/2],
                       [width/2, height/2], [-width/2, height/2]])
    return points @ basis.T + [cx, cy]


def fit_inside(box, shape):
    """Shrink only as necessary when rotating; then clamp translation to the image."""
    cx, cy, width, height, angle = pixel_box(box, shape)
    h, w = shape[:2]
    a = math.radians(angle)
    ex = abs(math.cos(a))*width + abs(math.sin(a))*height
    ey = abs(math.sin(a))*width + abs(math.cos(a))*height
    factor = min(1., w/max(ex, 1e-9), h/max(ey, 1e-9))
    width *= factor
    height *= factor
    ex *= factor/2
    ey *= factor/2
    cx = min(max(cx, ex), w-ex)
    cy = min(max(cy, ey), h-ey)
    return cx/w, cy/h, width/w, height/h, angle


def extraction_geometry(shape, box):
    """Return the crop transforms and output size without allocating pixels."""
    cx, cy, width, height, angle = pixel_box(box, shape)
    diagonal = math.hypot(*shape[:2])
    if max(width, height) > diagonal + 1:
        raise ValueError('Crop dimensions exceed the image diagonal')
    ow, oh = max(1, round(width)), max(1, round(height))
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    inverse = np.array([[c, -s, cx-.5-c*(ow-1)/2+s*(oh-1)/2],
                        [s, c, cy-.5-s*(ow-1)/2-c*(oh-1)/2]], np.float64)
    matrix = cv2.invertAffineTransform(inverse)
    return inverse, matrix, (ow, oh)


def extract(image, box, interpolation=cv2.INTER_LINEAR):
    """Return cropped pixels and the source-pixel → output-pixel affine matrix."""
    inverse, matrix, (ow, oh) = extraction_geometry(image.shape, box)
    cx, cy, width, height, angle = pixel_box(box, image.shape)
    # Integer, unrotated crops can be copied without interpolation.
    x, y = cx-width/2, cy-height/2
    if angle % 360 == 0 and abs(x-round(x)) < 1e-7 and abs(y-round(y)) < 1e-7:
        ix, iy = round(x), round(y)
        if ix >= 0 and iy >= 0 and ix+ow <= image.shape[1] and iy+oh <= image.shape[0]:
            return image[iy:iy+oh, ix:ix+ow].copy(), matrix
    return cv2.warpAffine(image, inverse, (ow, oh),
                          flags=interpolation | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT), matrix


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def rotation_matrix(shape, clockwise_degrees):
    """Image rotation with expanded bounds, in pixel-center coordinates."""
    h, w = shape[:2]
    a = math.radians(clockwise_degrees % 360)
    c, s = math.cos(a), math.sin(a)
    ow = int(math.ceil(round(abs(c)*w + abs(s)*h, 10)))
    oh = int(math.ceil(round(abs(s)*w + abs(c)*h, 10)))
    matrix = np.array([[c, -s, (ow-1)/2-c*(w-1)/2+s*(h-1)/2],
                       [s, c, (oh-1)/2-s*(w-1)/2-c*(h-1)/2]])
    return matrix, (ow, oh)


def unrotate_box(box, oriented_shape, source_shape, rotation):
    cx, cy, width, height, angle = pixel_box(box, oriented_shape)
    matrix, _ = rotation_matrix(source_shape, rotation)
    point = transform_points([(cx-.5, cy-.5)], cv2.invertAffineTransform(matrix))[0] + .5
    h, w = source_shape[:2]
    return point[0]/w, point[1]/h, width/w, height/h, angle-rotation
