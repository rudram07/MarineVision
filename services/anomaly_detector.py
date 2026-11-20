"""
MarineVision - Services: Anomaly Detector Interface.
Integrates with anomaly_detector.py to provide robust CNN Autoencoder anomaly inference for Flask.
Strictly avoids fabricating anomaly scores or regions when trained weights are missing.
"""

import os
from config import Config


class AnomalyDetector:
    """
    Flask service wrapper for Convolutional Autoencoder Sonar Anomaly Detection.
    Connects Flask analysis pipeline to anomaly_detector.py.
    """
    def __init__(self, weights_path=None, score_threshold=None):
        self.weights_path = weights_path or Config.ANOMALY_WEIGHTS_PATH
        self.score_threshold = score_threshold
        from anomaly_detector import get_anomaly_detector
        self.detector = get_anomaly_detector(weights_path=self.weights_path, score_threshold=self.score_threshold)

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
        Runs real autoencoder reconstruction and returns candidate anomalous regions.
        Returns: list of dicts:
        [{'anomaly_score': float, 'x': int, 'y': int, 'width': int, 'height': int, 'severity': 'LOW'|'MEDIUM'|'HIGH'}]
        """
        if not self.is_available:
            return []

        result = self.detector.detect(img, save_visuals=False)
        return result.get('anomalies', [])

    def detect_full(self, img, save_visuals=True, output_dir=None):
        """
        Runs full anomaly detection returning reconstruction, error maps, score, threshold, and regions.
        """
        return self.detector.detect(img, save_visuals=save_visuals, output_dir=output_dir)

    def get_info(self):
        info = self.detector.get_info()
        info['is_available'] = self.is_available
        return info
