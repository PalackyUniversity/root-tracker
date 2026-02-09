"""
Barcode reading utilities.

Reads barcodes from images using pyzbar.
"""

import cv2
import numpy as np
from pyzbar import pyzbar


class BarcodeReader:
    """
    Reads barcodes from images.
    
    Uses pyzbar library to detect and decode CODE128 barcodes.
    Falls back to thresholding if initial detection fails.
    """
    
    def __init__(self) -> None:
        self._symbol_type = pyzbar.ZBarSymbol.CODE128
        self._max_width = 1000  # Downscale to this width for faster processing
        self._roi_fraction = 0.35  # Only scan top 35% of image
    
    def read_fast(self, image: np.ndarray) -> tuple[str, tuple[int, int, int, int] | None]:
        """
        Fast barcode detection with optimizations.
        
        Uses ROI detection (top portion only) and downscaling for speed.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Tuple of (barcode_text, rect) where rect is (x, y, w, h) in original coordinates.
        """
        if image is None:
            return "", None
        
        h, w = image.shape[:2]
        
        # Only scan top portion where barcodes are typically located
        roi_height = int(h * self._roi_fraction)
        roi = image[:roi_height]
        
        # Calculate scale factor for downscaling
        scale = 1.0
        if w > self._max_width:
            scale = self._max_width / w
            new_w = self._max_width
            new_h = int(roi_height * scale)
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for faster processing
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Try detection on grayscale
        barcodes = pyzbar.decode(gray, symbols=[self._symbol_type])
        
        # If failed, try with Otsu thresholding
        if not barcodes:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            barcodes = pyzbar.decode(thresh, symbols=[self._symbol_type])
        
        if barcodes:
            barcode = barcodes[0]
            x, y, bw, bh = barcode.rect
            barcode_text = barcode.data.decode("utf-8")
            
            # Scale rect back to original image coordinates
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                bw = int(bw / scale)
                bh = int(bh / scale)
            
            return barcode_text, (x, y, bw, bh)
        
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
