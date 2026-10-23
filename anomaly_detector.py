"""
MarineVision - Side-Scan Sonar Acoustic Anomaly Detector (Convolutional Autoencoder).
Provides the standard detect_anomalies() interface for running inference on trained checkpoints.
Calculates pixel-level reconstruction-error maps, image-level anomaly scores,
and candidate anomaly regions based on a data-derived validation threshold.
"""

import os
import time
import cv2
import numpy as np
import torch
from config import Config
from models.autoencoder import ConvAutoencoder


class SonarAnomalyDetector:
    """
    Manages loading and inference for our from-scratch trained Autoencoder Sonar Anomaly Detector.
    """
    def __init__(self, weights_path=None, score_threshold=None, input_size=(256, 256)):
        self.weights_path = weights_path or Config.ANOMALY_WEIGHTS_PATH
        self.input_size = input_size
        self.model = None
        self.is_available = False
        self.status_message = ""
        self.version = "Autoencoder-Anomaly-Scratch-v1.0"
        self.score_threshold = score_threshold  # Will be populated from checkpoint if None
        self.mean_normal_score = 0.05
        self.std_normal_score = 0.02
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_model()

    def load_model(self):
        """Attempts to load model weights and validation threshold from checkpoint."""
        if not self.weights_path or not os.path.exists(self.weights_path):
            self.is_available = False
            self.status_message = (
                f"Model checkpoint '{os.path.basename(self.weights_path)}' not found in models directory. "
                "Anomaly detection is pending model training."
            )
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)

            # Support both full checkpoint dict and direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Retrieve validation-derived threshold
                if self.score_threshold is None:
                    self.score_threshold = checkpoint.get('threshold', 0.12)
                self.mean_normal_score = checkpoint.get('mean_normal_score', 0.05)
                self.std_normal_score = checkpoint.get('std_normal_score', 0.02)
                self.version = checkpoint.get('version', self.version)
                if 'input_size' in checkpoint:
                    self.input_size = tuple(checkpoint['input_size'])
            else:
                state_dict = checkpoint
                if self.score_threshold is None:
                    self.score_threshold = 0.12

            self.model = ConvAutoencoder(in_channels=1, input_size=self.input_size)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.is_available = True
            self.status_message = (
                f"Autoencoder Anomaly Detector loaded successfully from {self.weights_path} "
                f"({self.device}, threshold={self.score_threshold:.4f})."
            )
        except Exception as e:
            self.is_available = False
            self.status_message = f"Error loading Autoencoder weights from {self.weights_path}: {str(e)}"

    def detect(self, image_input, save_visuals=True, output_dir=None, min_contour_area=35):
        """
        Runs real autoencoder reconstruction inference and generates anomaly maps.
        Returns:
            dict containing anomaly score, threshold, candidate regions,
            paths to reconstruction and error overlay, inference time, and metadata.
        """
        start_time = time.time()

        if not self.is_available or self.model is None:
            return {
                'success': False,
                'is_available': False,
                'status': 'MODEL_NOT_AVAILABLE',
                'status_message': self.status_message,
                'anomaly_score': 0.0,
                'threshold_used': self.score_threshold or 0.0,
                'is_anomalous': False,
                'anomalies': [],
                'anomaly_count': 0,
                'reconstruction_image_path': None,
                'error_map_path': None,
                'overlay_image_path': None,
                'inference_time_sec': 0.0,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

        # 1. Load image and extract dimensions
        if isinstance(image_input, str):
            image_path = image_input
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                raise ValueError(f"Unable to read sonar image at {image_path}")
        else:
            orig_img = image_input
            image_path = None

        orig_h, orig_w = orig_img.shape[:2]

        # Convert to single-channel grayscale if multi-channel
        if len(orig_img.shape) == 3:
            gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = orig_img.copy()

        # 2. Resize and normalize to [0.0, 1.0] tensor
        target_w, target_h = self.input_size
        resized_gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        norm_gray = resized_gray.astype(np.float32) / 255.0

        tensor_in = torch.from_numpy(norm_gray).unsqueeze(0).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                tensor_recon = self.model(tensor_in)
                tensor_error = torch.abs(tensor_in - tensor_recon)

                # Compute image-level anomaly score
                flat_errors = tensor_error.view(-1)
                mean_err = flat_errors.mean().item()
                top_k = max(1, int(len(flat_errors) * 0.05))
                top_err = flat_errors.topk(top_k)[0].mean().item()
                anomaly_score = round(0.5 * mean_err + 0.5 * top_err, 4)

            inference_time = round(time.time() - start_time, 4)

            # Convert reconstruction and error map to numpy
            recon_np = (tensor_recon.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            error_np = tensor_error.squeeze().cpu().numpy()  # in [0, 1]

            # 3. Candidate Anomaly Region Extraction via Adaptive Thresholding & Morphology
            # Threshold using normal-validation derived threshold
            active_threshold = self.score_threshold if self.score_threshold is not None else 0.12
            is_anomalous_image = anomaly_score > active_threshold

            # Pixel mask for regions exceeding the threshold
            pixel_threshold = max(0.08, active_threshold * 1.25)
            binary_mask = (error_np > pixel_threshold).astype(np.uint8) * 255

            # Morphological post-processing to eliminate isolated speckle noise
            # Opening removes small false-positive acoustic dots
            # Dilation connects fragmented acoustic highlight/shadow regions
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)

            # Find contours
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            scale_x = orig_w / float(target_w)
            scale_y = orig_h / float(target_h)

            anomalies = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_contour_area:
                    continue

                rx, ry, rw, rh = cv2.boundingRect(cnt)

                # Extract local max error in this contour
                mask_roi = cleaned_mask[ry:ry+rh, rx:rx+rw]
                error_roi = error_np[ry:ry+rh, rx:rx+rw]
                roi_score = float(error_roi[mask_roi > 0].mean()) if (mask_roi > 0).any() else float(error_roi.mean())

                # Map coordinates back to original image resolution
                orig_rx = max(0, min(int(rx * scale_x), orig_w - 1))
                orig_ry = max(0, min(int(ry * scale_y), orig_h - 1))
                orig_rw = max(1, min(int(rw * scale_x), orig_w - orig_rx))
                orig_rh = max(1, min(int(rh * scale_y), orig_h - orig_ry))

                # Assign severity based on error magnitude
                if roi_score >= active_threshold * 1.8:
                    severity = 'HIGH'
                elif roi_score >= active_threshold * 1.3:
                    severity = 'MEDIUM'
                else:
                    severity = 'LOW'

                anomalies.append({
                    'anomaly_score': round(roi_score, 4),
                    'x': orig_rx,
                    'y': orig_ry,
                    'width': orig_rw,
                    'height': orig_rh,
                    'severity': severity
                })

            # Sort anomalies by score descending
            anomalies.sort(key=lambda a: a['anomaly_score'], reverse=True)

            # 4. Save Visual Artifacts if requested
            recon_path = None
            error_map_path = None
            overlay_path = None

            if save_visuals:
                out_folder = output_dir or Config.PROCESSED_FOLDER
                os.makedirs(out_folder, exist_ok=True)
                token = int(time.time() * 1000)

                # A. Reconstructed image resized to original size
                recon_orig = cv2.resize(recon_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                recon_filename = f"autoenc_recon_{token}.png"
                recon_path = os.path.join(out_folder, recon_filename)
                cv2.imwrite(recon_path, recon_orig)

                # B. Colorized Error Map (Heatmap)
                error_scaled = (np.clip(error_np / max(0.25, active_threshold * 2.0), 0.0, 1.0) * 255.0).astype(np.uint8)
                error_resized = cv2.resize(error_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                error_heatmap = cv2.applyColorMap(error_resized, cv2.COLORMAP_INFERNO)
                err_filename = f"autoenc_error_{token}.png"
                error_map_path = os.path.join(out_folder, err_filename)
                cv2.imwrite(error_map_path, error_heatmap)

                # C. Anomaly Overlay on Original Sonar Image
                bgr_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(orig_img.shape) == 2 else orig_img.copy()
                # Blend with subtle heatmap
                blended = cv2.addWeighted(bgr_base, 0.75, error_heatmap, 0.25, 0)

                # Draw candidate anomaly region bounding boxes
                severity_colors = {
                    'LOW': (0, 255, 255),     # Yellow in BGR
                    'MEDIUM': (0, 165, 255),  # Orange in BGR
                    'HIGH': (0, 0, 255)       # Red in BGR
                }

                for a in anomalies:
                    ax, ay, aw, ah = a['x'], a['y'], a['width'], a['height']
                    sev = a['severity']
                    color = severity_colors.get(sev, (0, 165, 255))
                    label = f"ANOMALY [{sev}] {a['anomaly_score']:.2f}"

                    cv2.rectangle(blended, (ax, ay), (ax + aw, ay + ah), color, 2)
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(blended, (ax, max(0, ay - 18)), (ax + lw + 6, max(18, ay)), color, -1)
                    cv2.putText(
                        blended, label,
                        (ax + 3, max(14, ay - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
                    )

                overlay_filename = f"autoenc_overlay_{token}.png"
                overlay_path = os.path.join(out_folder, overlay_filename)
                cv2.imwrite(overlay_path, blended)

            return {
                'success': True,
                'is_available': True,
                'status': 'COMPLETED',
                'status_message': f"Anomaly detection completed in {inference_time}s (Score: {anomaly_score:.4f}, Threshold: {active_threshold:.4f}).",
                'anomaly_score': anomaly_score,
                'threshold_used': active_threshold,
                'is_anomalous': is_anomalous_image,
                'anomalies': anomalies,
                'anomaly_count': len(anomalies),
                'reconstruction_image_path': recon_path,
                'error_map_path': error_map_path,
                'overlay_image_path': overlay_path,
                'inference_time_sec': inference_time,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

        except Exception as e:
            inference_time = round(time.time() - start_time, 4)
            return {
                'success': False,
                'is_available': True,
                'status': 'FAILED',
                'status_message': f"Autoencoder inference runtime error: {str(e)}",
                'anomaly_score': 0.0,
                'threshold_used': self.score_threshold or 0.0,
                'is_anomalous': False,
                'anomalies': [],
                'anomaly_count': 0,
                'reconstruction_image_path': None,
                'error_map_path': None,
                'overlay_image_path': None,
                'inference_time_sec': inference_time,
                'model_version': self.version,
                'weights_path': self.weights_path
            }

    def get_info(self):
        return {
            'model_name': 'CNN Convolutional Autoencoder Sonar Anomaly Detector',
            'model_type': 'ANOMALY_DETECTOR',
            'version': self.version,
            'weights_path': self.weights_path,
            'is_available': self.is_available,
            'status_message': self.status_message,
            'threshold': self.score_threshold
        }


# Singleton detector instance for efficient reuse across Flask requests
_GLOBAL_ANOMALY_DETECTOR = None


def get_anomaly_detector(weights_path=None, score_threshold=None):
    """Returns or lazily initializes the singleton SonarAnomalyDetector instance."""
    global _GLOBAL_ANOMALY_DETECTOR
    if _GLOBAL_ANOMALY_DETECTOR is None or (_GLOBAL_ANOMALY_DETECTOR.weights_path != (weights_path or Config.ANOMALY_WEIGHTS_PATH)):
        _GLOBAL_ANOMALY_DETECTOR = SonarAnomalyDetector(weights_path=weights_path, score_threshold=score_threshold)
    return _GLOBAL_ANOMALY_DETECTOR


def detect_anomalies(image_path, weights_path=None, score_threshold=None, save_visuals=True, output_dir=None):
    """
    Standard interface function requested by project specification:
    Run inference using our trained autoencoder checkpoint.
    Returns:
        dict with reconstruction, error map, anomaly score, threshold, highlighted regions, and inference time.
    """
    detector = get_anomaly_detector(weights_path=weights_path, score_threshold=score_threshold)
    return detector.detect(image_path, save_visuals=save_visuals, output_dir=output_dir)
