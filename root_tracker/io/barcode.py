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
