import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from config import Config

class ValidationResult:
    def __init__(self, is_valid, error_message=None, file_format=None, file_size_kb=0.0, width_px=0, height_px=0, original_filename=""):
        self.is_valid = is_valid
        self.error_message = error_message
        self.file_format = file_format
        self.file_size_kb = file_size_kb
        self.width_px = width_px
        self.height_px = height_px
        self.original_filename = original_filename

    def to_dict(self):
        return {
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'file_format': self.file_format,
            'file_size_kb': round(self.file_size_kb, 2),
            'width_px': self.width_px,
            'height_px': self.height_px,
            'original_filename': self.original_filename
        }


class ImageValidator:
    """
    Validates uploaded Side-Scan Sonar imagery against:
    - Extension / MIME restrictions (JPG, PNG, TIFF)
    - File size limits (Max 10 MB, Non-zero)
    - True image header and byte stream integrity via OpenCV
    - Minimum image dimension thresholds
    """
    def __init__(self, allowed_formats=None, max_size_mb=10, min_width=64, min_height=64):
        self.allowed_formats = allowed_formats or Config.ALLOWED_EXTENSIONS
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.min_width = min_width
        self.min_height = min_height

    def validate_file_object(self, file_storage):
        """
        Validates a Werkzeug FileStorage object directly from Flask request.files.
        """
        if not file_storage or not file_storage.filename or file_storage.filename.strip() == '':
            return ValidationResult(False, "No file was selected for upload.")

        filename = secure_filename(file_storage.filename)
        if not filename:
            # Fallback for filenames that might get stripped completely
            filename = "sonar_image.png"

        # 1. Format / Extension Check
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in self.allowed_formats:
            return ValidationResult(
                False,
                f"Unsupported file format '.{ext}'. Allowed formats: JPG, JPEG, PNG, TIFF.",
                original_filename=file_storage.filename
            )

        # Map to database constraint format: JPG, PNG, TIFF
        fmt_db = 'JPG' if ext in ('jpg', 'jpeg') else ('PNG' if ext == 'png' else 'TIFF')

        # Read file stream to memory
        file_bytes = file_storage.read()
        file_storage.seek(0) # Reset stream pointer

        # 2. Size Check
        file_size_bytes = len(file_bytes)
        if file_size_bytes == 0:
            return ValidationResult(
                False,
                "Uploaded file is empty (0 bytes).",
                original_filename=file_storage.filename
            )

        if file_size_bytes > self.max_size_bytes:
            size_mb = round(file_size_bytes / (1024 * 1024), 2)
            return ValidationResult(
                False,
                f"File size ({size_mb} MB) exceeds maximum allowed limit of 10 MB.",
                original_filename=file_storage.filename
            )

        file_size_kb = file_size_bytes / 1024.0

        # 3. Integrity and Decoding Check via OpenCV
        try:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return ValidationResult(
                    False,
                    "Corrupted or invalid image content. File could not be decoded as a valid image.",
                    original_filename=file_storage.filename
                )

            # Dimensions check
            if len(img.shape) >= 2:
                height_px, width_px = img.shape[:2]
            else:
                return ValidationResult(
                    False,
                    "Invalid image structure or channel dimensionality.",
                    original_filename=file_storage.filename
                )

            if width_px < self.min_width or height_px < self.min_height:
                return ValidationResult(
                    False,
                    f"Image resolution ({width_px}x{height_px}) is below minimum required threshold ({self.min_width}x{self.min_height} px).",
                    file_format=fmt_db,
                    file_size_kb=file_size_kb,
                    width_px=width_px,
                    height_px=height_px,
                    original_filename=file_storage.filename
                )

            return ValidationResult(
                True,
                file_format=fmt_db,
                file_size_kb=file_size_kb,
                width_px=width_px,
                height_px=height_px,
                original_filename=file_storage.filename
            )

        except Exception as e:
            return ValidationResult(
                False,
                f"Image validation error: {str(e)}",
                original_filename=file_storage.filename
            )
