from pyzbar import pyzbar
from typing import Union
import numpy as np
import cv2


def read_barcodes(frame: np.ndarray) -> tuple[str, Union[np.ndarray, None]]:
    """
    Read barcodes from an image.
    - TODO try cv2.barcode.BarcodeDetector() in the future

    :param frame: frame to read barcodes from
    :return: barcode text, frame with barcode drawn on it
    """

    barcodes = pyzbar.decode(frame, symbols=[pyzbar.ZBarSymbol.CODE128])

    # If pyzbar fails, try to threshold the image and try again
    if not barcodes:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, None, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        barcodes = pyzbar.decode(frame, symbols=[pyzbar.ZBarSymbol.CODE128])

    # Draw barcode on frame
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        barcode_info = barcode.data.decode("utf-8")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, barcode_info, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

        return barcode_info, frame

    return "", frame
