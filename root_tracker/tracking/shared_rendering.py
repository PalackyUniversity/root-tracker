"""Render shared root paths as adjacent plant-color bands."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def draw_shared_segments(image, uppers, colored, main_segments, linker):
    """Paint ordered color bands around each shared segment's centerline."""
    for upper in uppers:
        owners = colored.get(upper.get('lower_point'))
        if not isinstance(owners, tuple) or len(owners) < 2 or 'contour' not in upper:
            continue
        contour = upper['contour']
        points = contour[:, 0]
        start = np.flatnonzero(np.all(points == upper['point'], axis=1))
        if not len(start):
            continue
        ordered = np.roll(points, -int(start[0]), axis=0)
        end = np.flatnonzero(np.all(ordered == upper['lower_point'], axis=1))
        if not len(end) or end[0] < 1:
            continue
        path = ordered[:int(end[0]) + 1]
        tangent = np.gradient(path.astype(float), axis=0)
        norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)
        normal = np.column_stack((tangent[:, 1], -tangent[:, 0])) / norm[:, None]
        main_plants = main_segments.get(upper.get('contour_index'), set())
        thickness = 12 if main_plants else 6
        padding = thickness + 1
        x0, y0 = np.maximum(points.min(axis=0) - padding, 0)
        x1, y1 = np.minimum(points.max(axis=0) + padding + 1, image.shape[1::-1])
        shape = (y1 - y0, x1 - x0)
        mask = np.zeros(shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness, offset=(-int(x0), -int(y0)))
        local = path - (x0, y0)
        centers = np.ones(shape, np.uint8)
        centers[local[:, 1], local[:, 0]] = 0
        indices = distance_transform_edt(centers, return_distances=False, return_indices=True)
        normals = np.zeros((*shape, 2), float)
        normals[local[:, 1], local[:, 0]] = normal
        ys, xs = np.nonzero(mask)
        cy, cx = indices[:, ys, xs]
        nearest_normal = normals[cy, cx]
        signed = (xs - cx) * nearest_normal[:, 0] + (ys - cy) * nearest_normal[:, 1]
        band = np.clip(((signed / thickness + .5) * len(owners)).astype(int), 0, len(owners) - 1)
        for i, pid in enumerate(owners):
            color = linker.get_color(pid)
            if pid in main_plants:
                color = tuple(min(c + 170, 255) for c in color)
            selected = band == i
            image[ys[selected] + y0, xs[selected] + x0] = color
