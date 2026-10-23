"""
MarineVision - Services: Debris Detector Interface.
Integrates with detector.py to provide robust YOLO marine debris inference for Flask.
Strictly avoids fabricating detections when trained weights are missing.
"""

import os
from config import Config
from detector import get_detector, SonarDetector


class DebrisDetector:
    """
    Flask service wrapper for YOLO-based Marine Debris Detection.
    Connects Flask analysis pipeline to detector.py.
    """
    def __init__(self, weights_path=None, conf_threshold=0.30):
        self.weights_path = weights_path or Config.YOLO_WEIGHTS_PATH
        self.conf_threshold = conf_threshold
        self.detector = get_detector(weights_path=self.weights_path, conf_threshold=self.conf_threshold)

    @property
    def is_available(self):
        # Refresh detector if weights were recently created
        if not self.detector.is_available and os.path.exists(self.weights_path):
            self.detector.load_model()
        return self.detector.is_available

    @property
    def status_message(self):
        return self.detector.status_message

    @property
    def version(self):
        return self.detector.version

    def detect(self, img):
        """
        Runs real YOLO inference on sonar image.
        Returns: list of dicts:
        [{'class_label': str, 'confidence': float, 'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int}]
        """
        if not self.is_available:
            return []

        result = self.detector.predict(img, conf=self.conf_threshold, save_annotated=False)
        return result.get('detections', [])

    def get_info(self):
        info = self.detector.get_info()
        info['is_available'] = self.is_available
        info['threshold'] = self.conf_threshold
        return info
