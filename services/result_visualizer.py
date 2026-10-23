import os
import cv2
import numpy as np

class ResultVisualizer:
    """
    Renders visual bounding boxes, anomaly highlights, and side-by-side comparison views
    for Side-Scan Sonar imagery.
    """
    def __init__(self, box_color=(0, 255, 204), anomaly_color=(0, 165, 255)):
        # BGR format colors (Cyan/Teal for debris, Orange/Amber for anomaly)
        self.box_color = box_color
        self.anomaly_color = anomaly_color

    def draw_detections(self, image, detections):
        """
        Draws bounding boxes and confidence badges on the sonar image.
        """
        if not detections:
            return image

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for d in detections:
            x1, y1 = int(d['x_min']), int(d['y_min'])
            x2, y2 = int(d['x_max']), int(d['y_max'])
            label = f"{d['class_label']} {d['confidence']*100:.1f}%"

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.box_color, 2)

            # Draw label banner
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w + 6, y1), self.box_color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    def highlight_anomalies(self, image, anomalies):
        """
        Draws anomaly bounding regions with severity labels.
        """
        if not anomalies:
            return image

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        severity_colors = {
            'LOW': (0, 255, 255),     # Yellow
            'MEDIUM': (0, 165, 255),  # Orange
            'HIGH': (0, 0, 255)       # Red
        }

        for a in anomalies:
            x, y = int(a['x']), int(a['y'])
            w, h = int(a['width']), int(a['height'])
            sev = a.get('severity', 'MEDIUM').upper()
            color = severity_colors.get(sev, self.anomaly_color)
            label = f"ANOMALY [{sev}] {a.get('anomaly_score', 0):.2f}"

            # Draw dashed or solid rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Draw label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, y - 18), (x + tw + 6, y), color, -1)
            cv2.putText(annotated, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    def build_comparison(self, original_path, processed_path, output_path):
        """
        Generates a composite side-by-side comparison image between original and preprocessed sonar.
        """
        orig = cv2.imread(original_path)
        proc = cv2.imread(processed_path)

        if orig is None or proc is None:
            return None

        # Ensure 3-channels
        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        if len(proc.shape) == 2:
            proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

        # Match height
        target_h = max(orig.shape[0], proc.shape[0], 480)
        orig_w = int(orig.shape[1] * (target_h / orig.shape[0]))
        proc_w = int(proc.shape[1] * (target_h / proc.shape[0]))

        orig_resized = cv2.resize(orig, (orig_w, target_h), interpolation=cv2.INTER_AREA)
        proc_resized = cv2.resize(proc, (proc_w, target_h), interpolation=cv2.INTER_AREA)

        # Separator line
        divider = np.full((target_h, 6, 3), 40, dtype=np.uint8)

        # Add banners
        banner_h = 36
        banner_orig = np.full((banner_h, orig_w, 3), 25, dtype=np.uint8)
        cv2.putText(banner_orig, "ORIGINAL SONAR BACKSCATTER", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        banner_proc = np.full((banner_h, proc_w, 3), 35, dtype=np.uint8)
        cv2.putText(banner_proc, "PREPROCESSED / CLAHE ENHANCED", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 200), 1, cv2.LINE_AA)

        left_side = np.vstack([banner_orig, orig_resized])
        right_side = np.vstack([banner_proc, proc_resized])
        divider_full = np.full((target_h + banner_h, 6, 3), 50, dtype=np.uint8)

        composite = np.hstack([left_side, divider_full, right_side])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, composite)
        return output_path
