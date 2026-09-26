"""Shared HSV selection so segmentation previews use the pipeline's thresholds."""
import cv2
import numpy as np


def color_mask(image, lower, upper, *, hsv=None):
    if hsv is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = tuple(map(int, lower)), tuple(map(int, upper))
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    # Hue is circular: red selections can cross the 179/0 boundary.
    return cv2.bitwise_or(cv2.inRange(hsv, (0, *lower[1:]), upper),
                         cv2.inRange(hsv, lower, (179, *upper[1:])))


def sample_hsv(image, x, y):
    """Sample a 3×3 patch to reduce sensitivity to individual noisy pixels."""
    h, w = image.shape[:2]
    patch = image[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Circular mean avoids picking cyan when a red patch straddles hue zero.
    angles = hsv[:, 0].astype(float)*np.pi/90
    hue = int(round(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())*90/np.pi)) % 180
    saturation, value = np.median(hsv[:, 1:], axis=0).astype(int)
    return hue, int(saturation), int(value)


def range_from_samples(first, second):
    """Two sampled shades bound S/V and the shorter circular hue interval."""
    hue_low, hue_high = sorted((first[0], second[0]))
    if hue_high-hue_low > 90:
        hue_low, hue_high = hue_high, hue_low
    return ((hue_low, min(first[1], second[1]), min(first[2], second[2])),
            (hue_high, max(first[1], second[1]), max(first[2], second[2])))
