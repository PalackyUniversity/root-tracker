"""
Template matching based image registration.

Aligns time series images using template matching on edge features.
"""

import cv2
import numpy as np
import os
from multiprocessing import current_process
from scipy import fft

from ..config import Config
from ..models import ImageData


class ImageRegistrator:
    """
    Registers (aligns) images in a time series.
    
    Uses template matching on Canny edge images to align subsequent
    images to maintain consistent geometry across the time series.
    
    Args:
        config: Configuration object with registration settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def _coarse_match_location(
        self, target: np.ndarray, template: np.ndarray
    ) -> tuple[int, int] | None:
        """Propose a full-resolution peak, or decline uncertain coarse evidence.

        Area averaging keeps thin edges visible. Integer reduction with trimmed
        trailing pixels preserves the coordinate scale for differently sized
        inputs. The full-resolution score still chooses the final pixel; coarse
        ambiguity, weak matches and search-boundary peaks use the full search.
        """
        h, w = template.shape[:2]
        th, tw = target.shape[:2]
        if (target.ndim != 2 or template.ndim != 2 or
                target.dtype != np.uint8 or template.dtype != np.uint8 or
                target.size < 1_000_000 or target.size >= 2**31 or min(h, w) < 256 or th < h or tw < w):
            return None
        count = cv2.countNonZero(template)
        if count < 64 or count > 262144:
            return None
        if (np.any((template != 0) & (template != 255)) or
                np.any((target != 0) & (target != 255))):
            return None
        factor = max(4, min(16, 2 ** int(np.log2(max(th, tw) / 600))))
        def reduce(image):
            ih, iw = image.shape
            small = cv2.resize(image[:ih // factor * factor, :iw // factor * factor],
                               (iw // factor, ih // factor), interpolation=cv2.INTER_AREA)
            # Smooth on the small grid in floating point before judging
            # uniqueness. Without this, sub-cell translation can weaken an
            # exact match enough for a cell-aligned damaged copy to dominate.
            return cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.5)

        small_target, small_template = reduce(target), reduce(template)
        scores = cv2.matchTemplate(small_target, small_template, cv2.TM_CCOEFF)
        _, peak, _, (cx, cy) = cv2.minMaxLoc(scores)
        if peak <= 0 or cx <= 1 or cy <= 1 or cx >= scores.shape[1]-2 or cy >= scores.shape[0]-2:
            return None
        # Distinct distant peaks indicate repetition or a broad, poorly located
        # feature. Preserve the existing full-search winner in those cases.
        # The smoothing widens one peak over several coarse pixels; exclude
        # that neighborhood, which is covered by the refinement window below.
        other = scores.copy()
        other[max(0, cy-4):cy+5, max(0, cx-4):cx+5] = -np.inf
        if np.max(other) >= peak * .90:
            return None
        coarse_patch = small_target[cy:cy+small_template.shape[0], cx:cx+small_template.shape[1]]
        quality = cv2.matchTemplate(coarse_patch, small_template, cv2.TM_CCOEFF_NORMED)[0, 0]
        if quality < .35:
            return None

        # Refine once at half resolution, then evaluate only a few full-size
        # shifts. Sparse edge coordinates avoid FFTs over a multi-megapixel
        # template for that tiny final search.
        half_template = cv2.resize(template[:h//2*2, :w//2*2], (w//2, h//2),
                                   interpolation=cv2.INTER_AREA)
        radius = factor * 5
        x0, y0 = max(0, cx*factor-radius), max(0, cy*factor-radius)
        x1, y1 = min(tw-w, cx*factor+radius), min(th-h, cy*factor+radius)
        x0, y0 = x0//2*2, y0//2*2
        half_roi = target[y0:(y1+h)//2*2, x0:(x1+w)//2*2]
        half_target = cv2.resize(half_roi, (half_roi.shape[1]//2, half_roi.shape[0]//2),
                                 interpolation=cv2.INTER_AREA)
        scores = cv2.matchTemplate(half_target, half_template, cv2.TM_CCOEFF)
        _, _, _, (hx, hy) = cv2.minMaxLoc(scores)
        if hx == 0 or hy == 0 or hx == scores.shape[1]-1 or hy == scores.shape[0]-1:
            return None
        center_x, center_y = x0+hx*2, y0+hy*2
        x0, y0 = max(0, center_x-8), max(0, center_y-8)
        x1, y1 = min(tw-w, center_x+8), min(th-h, center_y+8)
        roi = target[y0:y1+h, x0:x1+w]
        # Use 0/1 values so integral counts cannot overflow for supported sizes.
        integral = cv2.integral(roi // 255, sdepth=cv2.CV_32S)
        sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        ey, ex = np.nonzero(template)
        edge_indices = (ey+y0)*tw + ex+x0
        flat_target = target.ravel()
        scores = np.empty(sums.shape, np.float64)
        for dy in range(scores.shape[0]):
            for dx in range(scores.shape[1]):
                overlap = np.count_nonzero(flat_target[edge_indices+dy*tw+dx])
                scores[dy, dx] = overlap - (count / template.size) * sums[dy, dx]
        best = int(np.argmax(scores))
        y, x = divmod(best, scores.shape[1])
        peak = scores[y, x]
        if x == 0 or y == 0 or x == scores.shape[1]-1 or y == scores.shape[0]-1:
            return None
        scores[y, x] = -np.inf
        if peak - np.max(scores) <= 1:
            return None
        target_count = sums[y, x]
        energy = np.sqrt(count * (1-count/template.size) *
                         target_count * (1-target_count/template.size))
        if energy <= 0 or peak / energy < .10:
            return None
        return x0+x, y0+y

    def _match_location(self, target: np.ndarray, template: np.ndarray) -> tuple[int, int]:
        """Locate sparse Canny edges, falling back to the existing full search.

        Confident coarse proposals use exact sparse local CCOEFF scores.
        Otherwise, a single circular FFT covers all valid template positions when padded to
        at least the target size. Binary edges permit correlation in units of
        edge pixels; integral sums supply the same CCOEFF mean correction.
        Keep OpenCV for small/dense/general images, process-pool workers, and
        close peaks where floating-point rounding could affect the chosen offset.
        """
        def original():
            return cv2.minMaxLoc(cv2.matchTemplate(target, template, cv2.TM_CCOEFF))[3]

        coarse = self._coarse_match_location(target, template)
        if coarse is not None:
            return coarse
        h, w = template.shape[:2]
        th, tw = target.shape[:2]
        if (target.ndim != 2 or template.ndim != 2 or
                target.dtype != np.uint8 or template.dtype != np.uint8 or
                th < h or tw < w or target.size < 1_000_000 or target.size >= 2**31 or
                current_process().name != 'MainProcess'):
            return original()
        count = cv2.countNonZero(template)
        if not count or count > 262144:
            return original()
        if (np.any((template != 0) & (template != 255)) or
                np.any((target != 0) & (target != 255))):
            return original()

        workers = min(4, os.cpu_count() or 1)
        shape = (fft.next_fast_len(th), fft.next_fast_len(tw))
        spectrum = fft.rfft2(target.astype(np.float32) / 255, s=shape, workers=workers)
        reference = fft.rfft2(template.astype(np.float32) / 255, s=shape, workers=workers)
        reference.imag *= -1
        spectrum *= reference
        del reference
        correlation = fft.irfft2(spectrum, s=shape, workers=workers)[:th-h+1, :tw-w+1]
        del spectrum
        integral = cv2.integral(target // 255, sdepth=cv2.CV_32S)
        sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        del integral
        scores = correlation.astype(np.float64) - (count / template.size) * sums
        best = int(np.argmax(scores))
        peak = scores.flat[best]
        scores.flat[best] = -np.inf
        # One edge-pixel of separation is conservative for single-precision FFT
        # error on sparse maps. Preserve OpenCV's tie/near-tie choice otherwise.
        if peak - np.max(scores) <= 1:
            return original()
        y, x = divmod(best, scores.shape[1])
        return x, y

    def align_to_template(
        self, 
        template_canny: np.ndarray,
        target_image: np.ndarray,
        target_canny: np.ndarray,
        target_process: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Align a target image to a template using template matching.
        
        Args:
            template_canny: Canny edges of the template (reference) image.
            target_image: BGR image to align.
            target_canny: Canny edges of the target image.
            target_process: Processed grayscale of the target image.
            
        Returns:
            Tuple of (aligned_image, aligned_canny, aligned_process, offset_x, offset_y).
        """
        h, w = template_canny.shape[:2]
        
        # Respect the configured search border on each side. Smaller targets
        # still need enough padding to fit the reference at least once.
        ratio = self.config.registration.margin_ratio
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError("Registration margin_ratio must be finite and non-negative")
        th, tw = target_canny.shape[:2]
        mx = max(int(tw * ratio), (w - tw + 1) // 2, 0)
        my = max(int(th * ratio), (h - th + 1) // 2, 0)
        
        # Add border to target images
        target_canny_padded = cv2.copyMakeBorder(
            target_canny, my, my, mx, mx, cv2.BORDER_CONSTANT, value=0
        )
        # Perform template matching
        left, top = self._match_location(target_canny_padded, template_canny)
        
        # Crop aligned images to template size
        def crop_with_border(source):
            # Allocate only the requested output, not a 2.25x padded color or
            # process image whose backing allocation survives with its view.
            result = np.zeros((h, w) + source.shape[2:], dtype=source.dtype)
            sx, sy = left - mx, top - my
            x0, y0 = max(0, sx), max(0, sy)
            x1, y1 = min(source.shape[1], sx+w), min(source.shape[0], sy+h)
            if x1 > x0 and y1 > y0:
                result[y0-sy:y1-sy, x0-sx:x1-sx] = source[y0:y1, x0:x1]
            return result

        aligned_image = crop_with_border(target_image)
        # A view would retain the entire padded search image for every frame.
        aligned_canny = target_canny_padded[top:top + h, left:left + w].copy()
        aligned_process = crop_with_border(target_process)
        
        # Calculate position offsets for updating coordinates
        offset_x = mx - left
        offset_y = my - top
        
        return aligned_image, aligned_canny, aligned_process, offset_x, offset_y
    
    def register_series(self, images: list[ImageData]) -> None:
        """
        Register all images in a series to the first image.
        
        Modifies ImageData objects in place with aligned images and
        updated position coordinates.
        
        Args:
            images: List of ImageData objects with canny and process attributes set.
        """
        if len(images) < 2:
            return
            
        # If registration is disabled, just compute difference images without alignment
        if not self.config.registration.enabled:
            for n in range(len(images) - 1):
                if images[n].process is None or images[n + 1].process is None:
                    continue
                    
                # Compute difference image (simple subtraction)
                images[n + 1].diff = cv2.subtract(images[n + 1].process, images[n].process)
            return
        
        for n in range(len(images) - 1):
            template = images[n].canny
            
            if template is None or images[n + 1].canny is None:
                continue
            
            # Align next image to current
            aligned_image, aligned_canny, aligned_process, offset_x, offset_y = (
                self.align_to_template(
                    template,
                    images[n + 1].image,
                    images[n + 1].canny,
                    images[n + 1].process
                )
            )
            
            # Update image data
            images[n + 1].image = aligned_image
            images[n + 1].canny = aligned_canny
            images[n + 1].process = aligned_process
            
            if images[n + 1].plate_transform:
                images[n + 1].plate_transform[2] += offset_x
                images[n + 1].plate_transform[5] += offset_y

            # Update position coordinates
            images[n + 1].positions_x = [x + offset_x for x in images[n + 1].positions_x]
            images[n + 1].positions_y = [y + offset_y for y in images[n + 1].positions_y]
            
            # Compute difference image for new growth detection
            images[n + 1].diff = cv2.subtract(images[n + 1].process, images[n].process)
