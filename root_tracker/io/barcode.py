"""
Barcode reading utilities.

Reads barcodes from images using pyzbar.
"""

import cv2
import numpy as np
from pathlib import Path
from pyzbar import pyzbar


class BarcodeReader:
    """
    Reads barcodes from images.
    
    Uses pyzbar library to detect and decode CODE128 barcodes.
    Falls back to thresholding if initial detection fails.
    """
    
    def __init__(self) -> None:
        self._symbol_type = pyzbar.ZBarSymbol.CODE128
        self._max_width = 2000  # Downscale to this width for faster processing
        self._roi_fraction = 0.50  # Scan top 50% of image

    def read_file(self, path: str) -> tuple[str, tuple[int, int, int, int] | None]:
        """Read a photograph, trying its bottom label without full JPEG decoding.

        JPEG's half-resolution grayscale decode avoids loading a full BGR image.
        The fast path prefers bottom labels (the acquisition setup places them
        there); all failures and other file formats use the original reader.
        Coordinates always refer to the original, unrotated photograph.
        """
        path = str(path)
        fast_result = None
        if Path(path).suffix.lower() in {'.jpg', '.jpeg', '.jpe'}:
            reduced = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if reduced is not None and reduced.shape[0] >= 2:
                offset_y = reduced.shape[0] // 2
                roi = reduced[offset_y:]
                scale = min(1.0, self._max_width / roi.shape[1])
                scale_y = 1.0
                if scale < 1:
                    resized_height = max(1, int(roi.shape[0] * scale))
                    scale_y = resized_height / roi.shape[0]
                    roi = cv2.resize(roi, (self._max_width, resized_height),
                                     interpolation=cv2.INTER_AREA)
                decoded = pyzbar.decode(roi, symbols=[self._symbol_type])
                if decoded:
                    barcode = decoded[0]
                    x, y, w, h = barcode.rect
                    fast_result = (barcode.data.decode('utf-8'), (
                        int(2 * x / scale), int(2 * (y / scale_y + offset_y)),
                        int(2 * w / scale), int(2 * h / scale_y)))
                    # A single successful scanline may have a zero-area box.
                    # Prefer the original overlay if that path can read it too.
                    if w > 0 and h > 0:
                        return fast_result
        result = self.read_fast(cv2.imread(path))
        return result if result[0] or fast_result is None else fast_result
    
    def read_fast(self, image: np.ndarray) -> tuple[str, tuple[int, int, int, int] | None]:
        """
        Robust multi-stage barcode detection.
        
        Strategies tried in order:
        1. ROI (Top 50%) + Downscale (Fastest)
        2. Full Image + Downscale
        3. Full Image + Original Resolution (slower)
        4. CLAHE Contrast Enhancement + Thresholding (Slowest, most robust)
        
        Args:
            image: Input BGR image.
            
        Returns:
            Tuple of (barcode_text, rect) where rect is (x, y, w, h) in original coordinates.
        """
        if image is None:
            return "", None
        
        h, w = image.shape[:2]
        
        # --- Helper for detection ---
        def try_detect(img, scale_factor=1.0, offset_y=0):
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            decoded = pyzbar.decode(gray, symbols=[self._symbol_type])
            if decoded:
                bcd = decoded[0]
                bx, by, bw, bh = bcd.rect
                text = bcd.data.decode("utf-8")
                
                # Rescale coordinates to original image
                final_x = int(bx / scale_factor)
                final_y = int(by / scale_factor) + offset_y
                final_w = int(bw / scale_factor)
                final_h = int(bh / scale_factor)
                
                return text, (final_x, final_y, final_w, final_h)
            return None

        # --- Stage 1: Fast ROI (Top 50%) + Downscale ---
        roi_height = int(h * self._roi_fraction)
        roi = image[:roi_height]
        
        scale_1 = 1.0
        if w > self._max_width:
            scale_1 = self._max_width / w
            new_w = self._max_width
            new_h = int(roi_height * scale_1)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        result = try_detect(roi, scale_1)
        if result:
            return result
            
        # --- Stage 2: Full Image + Downscale ---
        # Only if Stage 1 failed (maybe barcode is at bottom?)
        scale_2 = 1.0
        full_downscaled = image
        if w > self._max_width:
            scale_2 = self._max_width / w
            new_w = self._max_width
            new_h = int(h * scale_2)
            full_downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        result = try_detect(full_downscaled, scale_2)
        if result:
            return result

        # --- Stage 3: Full Image @ Original Resolution ---
        # Only if image isn't MASSIVE (cap at 4000px width to avoid freezing)
        if w <= 4000 and scale_2 != 1.0: # If scale_2 == 1.0, we already did this in Stage 2
            result = try_detect(image, 1.0)
            if result:
                return result

        # --- Stage 4: Enhanced Contrast (CLAHE) + Thresholding ---
        # Work on the downscaled full image from Stage 2 for speed
        if len(full_downscaled.shape) == 3:
            gray_stage4 = cv2.cvtColor(full_downscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray_stage4 = full_downscaled
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_stage4)
        
        # Try pure enhanced
        decoded_enhanced = pyzbar.decode(enhanced, symbols=[self._symbol_type])
        if decoded_enhanced:
             bcd = decoded_enhanced[0]
             bx, by, bw, bh = bcd.rect
             text = bcd.data.decode("utf-8")
             return text, (int(bx/scale_2), int(by/scale_2), int(bw/scale_2), int(bh/scale_2))

        # Try Otsu thresholding on enhanced
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        decoded_thresh = pyzbar.decode(thresh, symbols=[self._symbol_type])
        if decoded_thresh:
             bcd = decoded_thresh[0]
             bx, by, bw, bh = bcd.rect
             text = bcd.data.decode("utf-8")
             return text, (int(bx/scale_2), int(by/scale_2), int(bw/scale_2), int(bh/scale_2))

        return "", None
    
    def read(self, image: np.ndarray) -> tuple[str, np.ndarray]:
        """
        Read barcode from an image.
        
        Attempts to read barcodes from the image. If initial detection
        fails, tries again after applying Otsu thresholding.
        
        Args:
            image: Input BGR or grayscale image.
            
        Returns:
            Tuple of (barcode_text, annotated_image).
            If no barcode found, returns ("", image).
        """
        working_image = image.copy()
        
        # Try initial detection
        barcodes = pyzbar.decode(working_image, symbols=[self._symbol_type])
        
        # If failed, try with thresholding
        if not barcodes:
            if len(working_image.shape) == 3:
                working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            
            _, working_image = cv2.threshold(
                working_image, 
                0, 
                255, 
                cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            barcodes = pyzbar.decode(working_image, symbols=[self._symbol_type])
        
        # Process detected barcodes
        for barcode in barcodes:
            x, y, w, h = barcode.rect
            barcode_text = barcode.data.decode("utf-8")
            
            # Annotate image
            cv2.rectangle(working_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                working_image, 
                barcode_text, 
                (x + 6, y - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2.0, 
                (255, 255, 255), 
                2
            )
            
            return barcode_text, working_image
        
        return "", working_image
    
    def read_text_only(self, image: np.ndarray) -> str:
        """
        Read barcode text only (no annotation).
        
        Args:
            image: Input image.
            
        Returns:
            Barcode text, or empty string if not found.
        """
        text, _ = self.read(image)
        return text
    
    def read_with_rect(self, image: np.ndarray) -> tuple[str, tuple[int, int, int, int] | None]:
        """
        Read barcode text and return bounding box.
        
        Args:
            image: Input image.
            
        Returns:
            Tuple of (barcode_text, rect) where rect is (x, y, w, h) or None.
        """
        working_image = image.copy()
        
        # Try initial detection
        barcodes = pyzbar.decode(working_image, symbols=[self._symbol_type])
        
        # If failed, try with thresholding
        if not barcodes:
            if len(working_image.shape) == 3:
                working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
            
            _, working_image = cv2.threshold(
                working_image, 
                0, 
                255, 
                cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            barcodes = pyzbar.decode(working_image, symbols=[self._symbol_type])
        
        if barcodes:
            barcode = barcodes[0]
            x, y, w, h = barcode.rect
            barcode_text = barcode.data.decode("utf-8")
            return barcode_text, (x, y, w, h)
        
        return "", None
