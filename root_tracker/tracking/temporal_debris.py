"""Conservative rejection of stationary, disconnected foreground objects."""

import cv2
import numpy as np
from scipy import ndimage


def filter_static_islands(masks, origins):
    """Remove persistent islands with no resolved extension across a series.

    Inputs are registered binary masks, before user exclusions. Components are
    associated through their union, with a one-pixel neighbourhood for raster
    jitter and tiny segmentation gaps. A connection in ANY frame protects the
    entire component, including old stationary parts of a growing root.

    A robust median of persistent-object displacements compensates residual
    translation for comparison only; output pixels stay in the original frame.
    Reject only objects present in every frame whose later observations stay
    inside the first observation plus its measured foreground radius. Shrinking
    due to fading/threshold changes is not evidence of root growth. Origins,
    new objects and uncertain comparisons are protected. An isolated real root
    that never extends or connects during the recording is indistinguishable
    from debris using this evidence alone.

    Missing or incompatible frames disable rejection. Arrays are never mutated.
    """
    result = list(masks)
    if (len(masks) < 2 or any(mask is None for mask in masks)
            or any(mask.shape != masks[0].shape for mask in masks)):
        return result

    union = np.zeros(masks[0].shape, np.uint8)
    for mask in masks:
        union |= (mask != 0).astype(np.uint8)
    neighbourhood = np.ones((3, 3), np.uint8)
    labels, _ = ndimage.label(cv2.dilate(union, neighbourhood), neighbourhood)
    regions = ndimage.find_objects(labels)
    protected = {int(labels[y, x]) for x, y in origins
                 if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]}
    # Green stem contours are erased in preprocessing, leaving origins in
    # background. Protect the nearest foreground below each stem as well.
    foreground_y, foreground_x = np.nonzero(union)
    for x, y in origins:
        # Also retain the first foreground directly below the stem: a nearer
        # off-axis speck must not steal protection from the stationary root.
        if 0 <= x < union.shape[1] and 0 <= y < union.shape[0]:
            direct = np.flatnonzero(union[y:, x])
            if len(direct):
                protected.add(int(labels[y + direct[0], x]))
        below = foreground_y >= y
        if np.any(below):
            ys, xs = foreground_y[below], foreground_x[below]
            nearest = np.argmin((xs - x)**2 + (ys - y)**2)
            protected.add(int(labels[ys[nearest], xs[nearest]]))

    # Only identical silhouettes up to translation can vote on camera motion:
    # centroid movement alone could be caused by root extension or fading.
    # Fewer than three votes cannot establish a majority.
    displacements = [[] for _ in masks]
    persistent = []
    for label, region in enumerate(regions, start=1):
        if region is None or label in protected:
            continue
        component = labels[region] == label
        points = [np.column_stack(np.nonzero((mask[region] != 0) & component))
                  for mask in masks]
        if any(not len(p) for p in points):
            continue
        first = points[0] - points[0].min(axis=0)
        for i, p in enumerate(points):
            if np.array_equal(p - p.min(axis=0), first):
                displacements[i].append(p.min(axis=0) - points[0].min(axis=0))
        persistent.append(label)
    shifts = np.array([np.rint(np.median(votes, axis=0)).astype(int)
                       if len(votes) >= 3 else np.zeros(2, int)
                       for votes in displacements])
    padding = int(np.abs(shifts).max()) + 1

    rejected = np.zeros(len(regions) + 1, bool)
    for label in persistent:
        region = regions[label - 1]
        component = labels[region] == label
        first = (masks[0][region] != 0) & component
        # Padding makes thickness finite even at an image boundary.
        radius = float(cv2.distanceTransform(np.pad(first.astype(np.uint8), 1),
                                            cv2.DIST_L2, cv2.DIST_MASK_PRECISE).max())
        distance = cv2.distanceTransform((~np.pad(first, padding)).astype(np.uint8),
                                        cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for mask, (dy, dx) in zip(masks[1:], shifts[1:]):
            y, x = np.nonzero((mask[region] != 0) & component)
            if np.any(distance[y - dy + padding, x - dx + padding] > radius):
                break
        else:
            rejected[label] = True

    removal = rejected[labels]
    for i, mask in enumerate(masks):
        result[i] = mask.copy()
        result[i][removal] = 0
    return result
