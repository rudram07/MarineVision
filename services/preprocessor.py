import os
import cv2
import numpy as np
from config import Config

class Preprocessor:
    """
    Side-Scan Sonar (SSS) Image Preprocessing Pipeline.
    Implements:
    - Grayscale normalization & channel standardization
    - Edge-preserving acoustic speckle noise reduction (Bilateral filter or Median filter)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE) for sonar backscatter enhancement
    - Intensity stretching / dynamic range normalization
    - Optional standard sizing
    """
    def __init__(self, target_size=None, denoise_method='bilateral', contrast_method='clahe',
                 clahe_clip_limit=2.5, clahe_tile_grid=(8, 8), normalize=True):
        self.target_size = target_size
        self.denoise_method = denoise_method
        self.contrast_method = contrast_method
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.normalize_intensity = normalize

    def to_grayscale(self, img):
        """Converts multi-channel sonar images to standard single-channel grayscale."""
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                # RGBA to BGR then Gray
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def reduce_noise(self, gray_img):
        """
        Suppresses acoustic speckle noise while preserving sharp object-shadow boundaries.
        Bilateral filter is optimal for sonar backscatter reflections.
        """
        if self.denoise_method == 'bilateral':
            # d=7, sigmaColor=50, sigmaSpace=50
            return cv2.bilateralFilter(gray_img, d=7, sigmaColor=50, sigmaSpace=50)
        elif self.denoise_method == 'median':
            return cv2.medianBlur(gray_img, 3)
        elif self.denoise_method == 'gaussian':
            return cv2.GaussianBlur(gray_img, (5, 5), 0)
        return gray_img

    def enhance_contrast(self, gray_img):
        """
        Enhances local acoustic contrast using CLAHE.
        Prevents over-amplification of noise while revealing subtle seabed textures.
        """
        if self.contrast_method == 'clahe':
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid
            )
            return clahe.apply(gray_img)
        elif self.contrast_method == 'hist_eq':
            return cv2.equalizeHist(gray_img)
        return gray_img

    def normalize(self, img):
        """Min-max intensity stretching to utilize full [0, 255] dynamic range."""
        if not self.normalize_intensity:
            return img
        norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm_img.astype(np.uint8)

    def resize(self, img):
        """Resizes to target dimensions if configured."""
        if self.target_size and len(self.target_size) == 2:
            return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img

    def process(self, input_image_path, output_image_path):
        """
        Full preprocessing execution pipeline.
        Reads input image, applies transformations, saves preprocessed image,
        and returns preprocessing parameters and metadata.
        """
        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read input sonar image from {input_image_path}")

        # 1. Grayscale
        gray = self.to_grayscale(img)

        # 2. Acoustic speckle denoising
        denoised = self.reduce_noise(gray)

        # 3. Contrast enhancement
        enhanced = self.enhance_contrast(denoised)

        # 4. Normalization
        normalized = self.normalize(enhanced)

        # 5. Optional resize
        final_img = self.resize(normalized)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Save processed image as high-quality PNG
        success = cv2.imwrite(output_image_path, final_img)
        if not success:
            raise IOError(f"Failed to write preprocessed image to {output_image_path}")

        params = {
            'denoise_method': self.denoise_method,
            'contrast_enhancement': self.contrast_method,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_grid': list(self.clahe_tile_grid),
            'normalized': self.normalize_intensity,
            'output_dimensions': f"{final_img.shape[1]}x{final_img.shape[0]}"
        }

        return output_image_path, params, final_img
