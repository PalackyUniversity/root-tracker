"""
Image cropping and rotation utilities.

Handles automatic cropping based on blue background detection and image rotation.
"""

import cv2
import numpy as np

from ..config import Config
from . import roi
from .colors import color_mask


# Rotation mapping for OpenCV
ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class ImageCropper:
    """
    Handles image rotation and automatic cropping.
    
    Uses blue background detection to find the region of interest and
    crops the image accordingly.
    
    Args:
        config: Configuration object with crop settings.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def rotate(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image according to configuration.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Rotated image, or original if rotation is 0.
        """
        if self.config.rotation == 0:
            return image
        
        rotation_code = ROTATION_MAP.get(self.config.rotation)
        if rotation_code is None:
            matrix, size = roi.rotation_matrix(image.shape, self.config.rotation)
            return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
        
        return cv2.rotate(image, rotation_code)
    
    def auto_crop_to_blue_background(self, image: np.ndarray) -> np.ndarray:
        """
        Automatically crop image based on blue background detection.
        
        Finds the bounding box of the dominant background region and crops to that area.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Cropped image containing the blue background region.
        """
        x, y, w, h = self.blue_bounds(image)
        return image[y:y+h, x:x+w]

    def blue_mask(self, image, *, hsv=None):
        mask = color_mask(image, self.config.crop.blue_hsv_lower,
                          self.config.crop.blue_hsv_upper, hsv=hsv)
        return cv2.erode(mask, None, iterations=3)

    def blue_bounds(self, image):
        if not self.config.crop.background_enabled:
            return 0, 0, image.shape[1], image.shape[0]
        thresh = self.blue_mask(image)
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # A rough search area may still contain blue noise or unrelated objects.
        # The plate is the dominant connected region, not their combined extent.
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        if not contours:
            return 0, 0, image.shape[1], image.shape[0]
        if self.config.crop.background_region == 'all':
            bounds = [cv2.boundingRect(c) for c in contours]
            left, top = min(r[0] for r in bounds), min(r[1] for r in bounds)
            right = max(r[0]+r[2] for r in bounds)
            bottom = max(r[1]+r[3] for r in bounds)
            return left, top, right-left, bottom-top
        return cv2.boundingRect(max(contours, key=cv2.contourArea))

    def analysis_roi(self, shape):
        """Fixed normalized crop inside the detected plate; never follows plants."""
        if self.config.preprocess_roi is not None:
            return self.config.preprocess_roi
        height, width = shape[:2]
        start = round(height*self.config.crop.top_ratio)
        end = round(height*self.config.crop.bottom_ratio)
        top, bottom, left, right = self.margin_offsets(end-start, width)
        return roi.from_bounds(left, start+top, max(1, width-left-right),
                               max(1, end-start-top-bottom), shape)

    def search_image(self, image):
        if self.config.load_roi is not None:
            return roi.extract(image, self.config.load_roi)
        matrix, _ = roi.rotation_matrix(image.shape, self.config.rotation)
        return self.rotate(image), matrix

    def search_roi(self, image):
        if self.config.load_roi is not None:
            return self.config.load_roi
        oriented_shape = roi.rotation_matrix(image.shape, self.config.rotation)[1][::-1]
        return roi.unrotate_box((.5, .5, 1., 1., 0.), oriented_shape, image.shape, self.config.rotation)

    def plate_outline(self, image):
        search, matrix = self.search_image(image)
        x, y, width, height = self.blue_bounds(search)
        points = np.array([[x-.5, y-.5], [x+width-.5, y-.5],
                           [x+width-.5, y+height-.5], [x-.5, y+height-.5]])
        return roi.transform_points(points, cv2.invertAffineTransform(matrix)) + .5

    def crop_result(self, image, outline=None):
        """Crop plus original-image → cropped-image pixel transform."""
        search, matrix = self.search_image(image)
        if outline is None:
            x, y, width, height = self.blue_bounds(search)
        else:
            points = roi.transform_points(np.asarray(outline)-.5, matrix)+.5
            x, y = np.rint(points.min(axis=0)).astype(int)
            right, bottom = np.rint(points.max(axis=0)).astype(int)
            width, height = right-x, bottom-y
        matrix = matrix.copy()
        matrix[:, 2] -= (x, y)
        return search[y:y+height, x:x+width], matrix

    def search_mask(self, image, *, hsv=None):
        mask = self.blue_mask(image, hsv=hsv)
        if self.config.load_roi is not None:
            region = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(region, np.rint(roi.corners(self.config.load_roi, image.shape)).astype(np.int32), 255)
            cv2.bitwise_and(mask, region, dst=mask)
        return mask

    def automatic_roi(self, image):
        oriented = self.rotate(image)
        box = roi.from_bounds(*self.blue_bounds(oriented), oriented.shape)
        return roi.unrotate_box(box, oriented.shape, image.shape, self.config.rotation)

    def margin_offsets(self, height: int, width: int) -> tuple[int, int, int, int]:
        """
        Convert the configured margin fractions to pixel sizes.

        Args:
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Tuple of (top, bottom, left, right) margins in pixels.
        """
        return (
            round(self.config.margin_top * height),
            round(self.config.margin_bottom * height),
            round(self.config.margin_left * width),
            round(self.config.margin_right * width),
        )

    def crop_sides(self, image: np.ndarray) -> np.ndarray:
        """
        Apply side margin cropping (removes box edges).
        
        Args:
            image: Input image.
            
        Returns:
            Image with side margins removed.
        """
        margin = round(self.config.margin * image.shape[1])
        result = image.copy()
        result[:, :margin] = 0
        result[:, -margin:] = 0
        return result
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full cropping pipeline: rotate and auto-crop.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Rotated and cropped image.
        """
        search, _ = self.search_image(image)
        return self.auto_crop_to_blue_background(search)
