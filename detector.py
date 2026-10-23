"""
MarineVision - Side-Scan Sonar Object & Marine Debris Detector (YOLO).
Provides the standard predict_sonar() interface for running inference on trained checkpoints.
Strictly returns real predictions or transparent unavailable status; never fabricates results.
"""

import os
import time
import cv2
import torch
from config import Config

# Default class names used in the trained side-scan sonar debris model
DEFAULT_CLASS_NAMES = {
    0: 'submarine_pipeline',
    1: 'shipwreck',
    2: 'ghost_net',
    3: 'mine_cylinder'
}


class SonarDetector:
    """
    Manages loading and inference for our from-scratch trained YOLO sonar detector.
    """
    def __init__(self, weights_path=None, conf_threshold=0.25):
        self.weights_path = weights_path or Config.YOLO_WEIGHTS_PATH
        self.conf_threshold = conf_threshold
        self.model = None
        self.is_available = False
        self.status_message = ""
        self.version = "YOLO-Sonar-Debris-Scratch-v1.0"
        self.class_names = DEFAULT_CLASS_NAMES
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.load_model()

    def load_model(self):
        """Attempts to load model weights if the file exists."""
        if not self.weights_path or not os.path.exists(self.weights_path):
            self.is_available = False
            self.status_message = (
                f"Model checkpoint '{os.path.basename(self.weights_path)}' not found in models directory. "
                "Marine debris detection is pending model training."
            )
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            self.is_available = True
            if hasattr(self.model, 'names') and isinstance(self.model.names, dict) and self.model.names:
                self.class_names = self.model.names
            self.status_message = f"YOLO model loaded successfully from {self.weights_path} ({self.device})."
        except Exception as e:
            self.is_available = False
            self.status_message = f"Error loading YOLO weights from {self.weights_path}: {str(e)}"

    def predict(self, image_input, conf=None, save_annotated=True, output_dir=None):
        """
        Runs real YOLO inference on a sonar image path or loaded numpy array.
        Returns:
            dict containing detections, count, annotated_path, inference_time_sec, and metadata.
        """
        conf = conf if conf is not None else self.conf_threshold
        start_time = time.time()

        if not self.is_available or self.model is None:
            return {
                'success': False,
                'is_available': False,
                'status': 'MODEL_NOT_AVAILABLE',
                'status_message': self.status_message,
                'detections': [],
                'detection_count': 0,
                'count': 0,
                'inference_time_sec': 0.0,
                'annotated_image_path': None,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

        # Resolve image array and original path
        if isinstance(image_input, str):
            image_path = image_input
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to read sonar image at {image_path}")
        else:
            img = image_input
            image_path = None

        h, w = img.shape[:2]

        try:
            results = self.model(img, conf=conf, device=self.device, verbose=False)
            inference_time = round(time.time() - start_time, 4)

            detections = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self.class_names.get(cls_id, f"debris_class_{cls_id}")

                    # Clip bounding box to image boundaries
                    x1 = max(0, min(int(x1), w - 1))
                    y1 = max(0, min(int(y1), h - 1))
                    x2 = max(0, min(int(x2), w))
                    y2 = max(0, min(int(y2), h))

                    detections.append({
                        'class_id': cls_id,
                        'class_label': name,
                        'confidence': round(score, 4),
                        'x_min': x1,
                        'y_min': y1,
                        'x_max': x2,
                        'y_max': y2
                    })

            # Generate and save annotated image if requested
            annotated_path = None
            if save_annotated:
                out_folder = output_dir or Config.PROCESSED_FOLDER
                os.makedirs(out_folder, exist_ok=True)
                token = int(time.time() * 1000)
                ann_filename = f"yolo_det_{token}.png"
                annotated_path = os.path.join(out_folder, ann_filename)

                # Render annotations using clean OpenCV styling
                annotated_img = img.copy()
                if len(annotated_img.shape) == 2:
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)

                for d in detections:
                    bx1, by1, bx2, by2 = d['x_min'], d['y_min'], d['x_max'], d['y_max']
                    label_text = f"{d['class_label']} {d['confidence']*100:.1f}%"
                    color = (0, 255, 204)  # Teal/Cyan in BGR

                    cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated_img, (bx1, max(0, by1 - 18)), (bx1 + tw + 6, max(18, by1)), color, -1)
                    cv2.putText(
                        annotated_img, label_text,
                        (bx1 + 3, max(14, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
                    )

                cv2.imwrite(annotated_path, annotated_img)

            return {
                'success': True,
                'is_available': True,
                'status': 'COMPLETED',
                'status_message': f"Inference completed in {inference_time}s with {len(detections)} detection(s).",
                'detections': detections,
                'detection_count': len(detections),
                'count': len(detections),
                'inference_time_sec': inference_time,
                'annotated_image_path': annotated_path,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

        except Exception as e:
            inference_time = round(time.time() - start_time, 4)
            return {
                'success': False,
                'is_available': True,
                'status': 'FAILED',
                'status_message': f"YOLO inference runtime error: {str(e)}",
                'detections': [],
                'detection_count': 0,
                'inference_time_sec': inference_time,
                'annotated_image_path': None,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

    def get_info(self):
        return {
            'model_name': 'YOLO Side-Scan Sonar Marine Debris Detector',
            'model_type': 'DEBRIS_DETECTOR',
            'version': self.version,
            'weights_path': self.weights_path,
            'is_available': self.is_available,
            'status_message': self.status_message,
            'threshold': self.conf_threshold,
            'classes': self.class_names
        }


# Singleton detector instance for efficient reuse across Flask requests
_GLOBAL_DETECTOR = None


def get_detector(weights_path=None, conf_threshold=0.25):
    """Returns or lazily initializes the singleton SonarDetector instance."""
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None or (_GLOBAL_DETECTOR.weights_path != (weights_path or Config.YOLO_WEIGHTS_PATH)):
        _GLOBAL_DETECTOR = SonarDetector(weights_path=weights_path, conf_threshold=conf_threshold)
    return _GLOBAL_DETECTOR


def predict_sonar(image_path, weights_path=None, conf_threshold=0.25, save_annotated=True, output_dir=None):
    """
    Standard interface function requested by project specification:
    Run inference using our trained YOLO checkpoint.
    Returns:
        dict with detections, annotated_image_path, inference_time_sec, and model metadata.
    """
    detector = get_detector(weights_path=weights_path, conf_threshold=conf_threshold)
    return detector.predict(image_path, conf=conf_threshold, save_annotated=save_annotated, output_dir=output_dir)
