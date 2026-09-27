"""
Image calibration helpers: tonal adjustment, dataset matching, and HSV
auto-detection.

These are pure, GUI-independent functions used by the Load step (brightness /
shadows / highlights + "match to reference dataset") and by the Preprocess step
colour pickers (deriving HSV ranges from a picked colour or automatically from
the image).
"""

from pathlib import Path

import cv2
import numpy as np


# Percentiles used to characterise a dataset's tone: shadows / midtones / highlights.
_SHADOW_PCT = 10
_MID_PCT = 50
_HIGHLIGHT_PCT = 90

# Image extensions scanned when computing reference statistics.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Tonal adjustment (brightness / shadows / highlights)
# ---------------------------------------------------------------------------

def apply_adjustments(
    image: np.ndarray,
    brightness: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
) -> np.ndarray:
    """
    Apply brightness / shadows / highlights to a BGR (or grayscale) image.

    All three parameters are in the range [-100, 100] where 0 means "no change".
    - brightness shifts every tone uniformly,
    - shadows lifts (or lowers) dark tones,
    - highlights lifts (or lowers) bright tones,
    using smooth tone-dependent weights so mid-tones are affected least by the
    shadow/highlight controls.

    Returns a new uint8 array; the input is never modified. A no-op (all zeros)
    returns the original array unchanged.
    """
    if brightness == 0 and shadows == 0 and highlights == 0:
        return image

    x = image.astype(np.float32) / 255.0
    # Tone-dependent weights: shadows act on darks, highlights on brights.
    shadow_w = (1.0 - x) ** 2
    highlight_w = x ** 2

    # 0.5 scale keeps a full ±100 slider within roughly ±0.5 of normalised range.
    out = (
        x
        + (brightness / 100.0) * 0.5
        + (shadows / 100.0) * 0.5 * shadow_w
        + (highlights / 100.0) * 0.5 * highlight_w
    )
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def tone_stats(image: np.ndarray) -> tuple[float, float, float]:
    """Return (shadow, midtone, highlight) luminance percentiles of an image."""
    if image.ndim == 3:
        lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        lum = image
    p = np.percentile(lum, [_SHADOW_PCT, _MID_PCT, _HIGHLIGHT_PCT])
    return float(p[0]), float(p[1]), float(p[2])


def _read_reduced(path: str | Path) -> np.ndarray | None:
    """
    Decode an image at 1/8 resolution for fast exposure statistics.

    Tone percentiles are virtually unchanged when downscaled, but decoding at
    reduced resolution (libjpeg DCT scaling) is dramatically faster than reading
    full-resolution camera JPEGs — the difference between seconds and minutes
    when scanning many images.
    """
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        img = cv2.imread(str(path))  # fallback for formats without reduced decode
    return img


def tone_stats_for_path(path: str | Path) -> tuple[float, float, float] | None:
    """Read an image (reduced resolution) and return its tone stats, or None."""
    img = _read_reduced(path)
    if img is None:
        return None
    return tone_stats(img)


# ---------------------------------------------------------------------------
# Brightness-only matching (fast exposure match)
# ---------------------------------------------------------------------------

def mean_luma(image: np.ndarray) -> float:
    """Mean luminance of an image (0..255)."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(image.mean())


def mean_luma_for_path(path: str | Path) -> float | None:
    """Mean luminance of an image read at reduced resolution, or None."""
    img = _read_reduced(path)
    if img is None:
        return None
    return mean_luma(img)


def reference_brightness_from_folder(
    folder: str | Path, max_images: int = 8
) -> float | None:
    """Average mean-luminance of a sample of images in a reference folder."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in _IMAGE_EXTS and "aside" not in p.parts
    )
    if not paths:
        return None
    if len(paths) > max_images:
        step = len(paths) / max_images
        paths = [paths[int(i * step)] for i in range(max_images)]

    values = [v for v in (mean_luma_for_path(p) for p in paths) if v is not None]
    if not values:
        return None
    return float(np.mean(values))


def compute_brightness_match(current_mean: float, target_mean: float) -> float:
    """Brightness slider value (clamped to [-100, 100]) to match a target mean luma.

    Inverts the brightness gain of :func:`apply_adjustments`
    (Δluma = gain · brightness, gain = 0.5·255/100).
    """
    gain = 0.5 * 255.0 / 100.0
    return float(max(-100.0, min(100.0, (target_mean - current_mean) / gain)))


def reference_stats_from_folder(
    folder: str | Path, max_images: int = 12
) -> dict | None:
    """
    Compute aggregate tone statistics for a reference dataset.

    Samples up to ``max_images`` images from ``folder`` (recursively), averaging
    their shadow/midtone/highlight percentiles. Returns a dict with keys
    'shadow', 'mid', 'highlight', or None if no images were found.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None

    paths = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in _IMAGE_EXTS and "aside" not in p.parts
    )
    if not paths:
        return None

    # Evenly sample across the folder rather than just the first N.
    if len(paths) > max_images:
        step = len(paths) / max_images
        paths = [paths[int(i * step)] for i in range(max_images)]

    shadows, mids, highlights = [], [], []
    for p in paths:
        stats = tone_stats_for_path(p)
        if stats is None:
            continue
        s, m, h = stats
        shadows.append(s)
        mids.append(m)
        highlights.append(h)

    if not mids:
        return None

    return {
        "shadow": float(np.mean(shadows)),
        "mid": float(np.mean(mids)),
        "highlight": float(np.mean(highlights)),
    }


def compute_match_adjustments(
    current: tuple[float, float, float],
    target: dict,
) -> tuple[float, float, float]:
    """
    Compute (brightness, shadows, highlights) slider values (each clamped to
    [-100, 100]) that move a group's tone statistics toward the reference target.

    ``current`` is (shadow, mid, highlight) of the current group; ``target`` is a
    dict with 'shadow'/'mid'/'highlight'.

    :func:`apply_adjustments` adds, at a normalised tone ``x``::

        Δ(x) = gain * (brightness + shadows·(1-x)² + highlights·x²)

    with ``gain = 0.5·255/100``. The three controls are fitted sequentially —
    brightness from the mid-tone, then shadows from the (post-brightness) shadow
    tone, then highlights from the (post-brightness+shadows) highlight tone —
    each clamped to [-100, 100]. Sequential fitting stays stable when the tones
    are clustered (where a joint solve would be ill-conditioned). The result is a
    good first approximation; the user can fine-tune afterwards.
    """
    cur_shadow, cur_mid, cur_highlight = (float(c) for c in current)
    gain = 0.5 * 255.0 / 100.0
    clamp = lambda v: float(max(-100.0, min(100.0, v)))

    # 1) Brightness from the mid-tone (uniform shift).
    brightness = clamp((target["mid"] - cur_mid) / gain)
    bshift = gain * brightness

    # 2) Shadows from the shadow tone, after brightness.
    xs = min(1.0, max(0.0, (cur_shadow + bshift) / 255.0))
    ws = max((1.0 - xs) ** 2, 0.05)
    shadows = clamp((target["shadow"] - (cur_shadow + bshift)) / (gain * ws))

    # 3) Highlights from the highlight tone, after brightness + shadows.
    xh = min(1.0, max(0.0, (cur_highlight + bshift) / 255.0))
    wh = max(xh ** 2, 0.05)
    highlight_now = cur_highlight + bshift + gain * shadows * (1.0 - xh) ** 2
    highlights = clamp((target["highlight"] - highlight_now) / (gain * wh))

    return brightness, shadows, highlights


# ---------------------------------------------------------------------------
# HSV range helpers (Preprocess colour pickers)
# ---------------------------------------------------------------------------

def hsv_window(
    bgr: tuple[int, int, int],
    h_tol: int = 12,
    s_tol: int = 70,
    v_tol: int = 70,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Build an (lower, upper) HSV range around a single picked BGR colour.

    Hue wraps at 180 (OpenCV); we clamp rather than wrap to keep ranges simple.
    """
    px = np.uint8([[list(bgr)]])
    h, s, v = (int(c) for c in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0])
    lower = (
        max(0, h - h_tol),
        max(0, s - s_tol),
        max(0, v - v_tol),
    )
    upper = (
        min(179, h + h_tol),
        min(255, s + s_tol),
        min(255, v + v_tol),
    )
    return lower, upper


def _range_from_hue_gate(
    hsv: np.ndarray,
    hue_lo: int,
    hue_hi: int,
    min_sat: int = 40,
    min_val: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    """Derive an HSV range from pixels falling inside a coarse hue gate."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= hue_lo) & (h <= hue_hi) & (s >= min_sat) & (v >= min_val)
    if np.count_nonzero(mask) < 50:
        return None
    sel = hsv[mask]
    lo = np.percentile(sel, 5, axis=0)
    hi = np.percentile(sel, 95, axis=0)
    lower = tuple(int(max(0, x)) for x in lo)
    upper = tuple(int(min(255 if i else 179, x)) for i, x in enumerate(hi))
    return lower, upper


def auto_hsv_ranges(image: np.ndarray) -> dict:
    """
    Auto-detect green (plant stems) and blue (tray background) HSV ranges.

    Returns a dict possibly containing 'green' and 'blue' keys, each mapping to
    an (lower, upper) tuple. Missing keys mean that colour couldn't be found
    confidently (caller should keep the existing range).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = {}
    green = _range_from_hue_gate(hsv, 30, 90)
    if green is not None:
        result["green"] = green
    blue = _range_from_hue_gate(hsv, 90, 140)
    if blue is not None:
        result["blue"] = blue
    return result


def auto_hsv_ranges_for_path(path: str | Path) -> dict:
    """Auto-detect green/blue HSV ranges from an image read at reduced resolution.

    Hue/saturation distributions are essentially scale-invariant, so working at
    1/8 resolution gives the same ranges far faster and with a fraction of the
    memory (no full-resolution HSV copy + percentile), avoiding UI lag.
    """
    img = _read_reduced(path)
    if img is None:
        return {}
    return auto_hsv_ranges(img)
