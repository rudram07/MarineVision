import os
import cv2
import numpy as np

# Structural shape dictionary mapping each acoustic class to its morphological characteristics
OBJECT_STRUCTURAL_PROFILES = {
    'submarine_pipeline': {
        'display_name': 'Submarine Pipeline / Conduit',
        'shape': 'Linear Continuous Corridor',
        'structure': 'Continuous linear acoustic highlight with parallel acoustic shadow across the seabed sediment.'
    },
    'shipwreck': {
        'display_name': 'Shipwreck / Maritime Wreckage',
        'shape': 'High-Relief Angular Hull Structure',
        'structure': 'Large irregular high-backscatter cluster with prominent structural relief, angular beam geometry, and extensive acoustic shadow field.'
    },
    'ghost_net': {
        'display_name': 'Ghost Net (Lost Fishing Gear)',
        'shape': 'Tangled Diffuse Synthetic Fiber Web',
        'structure': 'Multi-point diffuse acoustic backscatter with tangled web morphology, lack of rigid metallic boundaries, and trailing acoustic filaments.'
    },
    'mine_cylinder': {
        'display_name': 'Cylindrical Ordnance / Contact',
        'shape': 'Symmetrical Compact Cylinder',
        'structure': 'High-reflectance metallic target with symmetrical elliptical/cylindrical geometry and distinct sharp acoustic shadow.'
    }
}


class ResultVisualizer:
    """
    Renders high-visibility RED bounding boxes, translucent target tints,
    structural shape descriptors, and multi-view comparisons for Side-Scan Sonar imagery.
    """
    def __init__(self, box_color=(0, 0, 255), anomaly_color=(0, 140, 255)):
        # BGR format: Pure Red (0, 0, 255) for Debris Bounding Boxes
        self.box_color = box_color
        self.anomaly_color = anomaly_color

    def draw_detections(self, image, detections):
        """
        Draws bright RED bounding boxes, red translucent target tints,
        and shape/structural labels on the sonar image.
        """
        if not detections:
            return image

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        h, w = annotated.shape[:2]

        for d in detections:
            x1 = max(0, min(int(d['x_min']), w - 1))
            y1 = max(0, min(int(d['y_min']), h - 1))
            x2 = max(0, min(int(d['x_max']), w))
            y2 = max(0, min(int(d['y_max']), h))

            if x2 <= x1 or y2 <= y1:
                continue

            raw_label = d.get('class_label', 'debris')
            profile = OBJECT_STRUCTURAL_PROFILES.get(raw_label, {
                'display_name': raw_label.replace('_', ' ').title(),
                'shape': d.get('structural_shape', 'Acoustic Target Structure'),
                'structure': d.get('structural_description', 'High acoustic backscatter signature.')
            })

            display_name = profile['display_name']
            shape_info = d.get('structural_shape') or profile['shape']
            conf_pct = d.get('confidence', 0.85) * 100.0

            # 1. Red Translucent Tint inside target bounding box
            roi = annotated[y1:y2, x1:x2]
            red_overlay = np.full(roi.shape, (0, 0, 240), dtype=np.uint8)
            cv2.addWeighted(red_overlay, 0.22, roi, 0.78, 0, roi)
            annotated[y1:y2, x1:x2] = roi

            # 2. Solid Bold RED Bounding Box (3px)
            red_color = (0, 0, 255)  # Pure Red in BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), red_color, 3)

            # 3. Corner brackets for enhanced tactical clarity
            c_len = max(8, min(24, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), (255, 255, 255), 2)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

            # 4. Text labels: Header line (Object Name + Conf) and Subtitle line (Shape & Structure)
            text_line1 = f"● {display_name.upper()}  [{conf_pct:.1f}%]"
            text_line2 = f"Shape: {shape_info}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1

            (tw1, th1), _ = cv2.getTextSize(text_line1, font, font_scale, thickness)
            (tw2, th2), _ = cv2.getTextSize(text_line2, font, 0.38, thickness)
            badge_w = max(tw1, tw2) + 14
            badge_h = th1 + th2 + 16

            badge_y1 = max(0, y1 - badge_h - 4)
            badge_y2 = badge_y1 + badge_h
            badge_x1 = max(0, min(x1, w - badge_w))
            badge_x2 = badge_x1 + badge_w

            # Solid Red Background badge with thin white border
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), red_color, -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)

            # White Text
            cv2.putText(annotated, text_line1, (badge_x1 + 6, badge_y1 + th1 + 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, text_line2, (badge_x1 + 6, badge_y1 + th1 + th2 + 10), font, 0.38, (220, 240, 255), 1, cv2.LINE_AA)

        return annotated

    def highlight_anomalies(self, image, anomalies):
        """
        Draws high-visibility red/amber anomaly bounding regions with reconstruction error badges.
        """
        if not anomalies:
            return image

        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        h, w = annotated.shape[:2]

        for a in anomalies:
            x = max(0, min(int(a['x']), w - 1))
            y = max(0, min(int(a['y']), h - 1))
            box_w = max(4, min(int(a['width']), w - x))
            box_h = max(4, min(int(a['height']), h - y))
            sev = a.get('severity', 'MEDIUM').upper()
            score = a.get('anomaly_score', 0)

            color = (0, 0, 255) if sev == 'HIGH' else (0, 140, 255)  # Red for High, Amber for Medium

            # Translucent tint
            roi = annotated[y:y+box_h, x:x+box_w]
            tint = np.full(roi.shape, color, dtype=np.uint8)
            cv2.addWeighted(tint, 0.18, roi, 0.82, 0, roi)
            annotated[y:y+box_h, x:x+box_w] = roi

            # Border
            cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, 2)

            label = f"ANOMALY [{sev}] {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x, max(0, y - 18)), (x + tw + 8, max(18, y)), color, -1)
            cv2.putText(annotated, label, (x + 4, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def build_comparison(self, original_path, right_path, output_path, right_title="PREPROCESSED / CLAHE ENHANCED"):
        """
        Generates a composite side-by-side comparison between original and processed/annotated sonar.
        """
        orig = cv2.imread(original_path)
        right_img = cv2.imread(right_path)

        if orig is None or right_img is None:
            return None

        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        if len(right_img.shape) == 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)

        target_h = max(orig.shape[0], right_img.shape[0], 480)
        orig_w = int(orig.shape[1] * (target_h / orig.shape[0]))
        right_w = int(right_img.shape[1] * (target_h / right_img.shape[0]))

        orig_resized = cv2.resize(orig, (orig_w, target_h), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_img, (right_w, target_h), interpolation=cv2.INTER_AREA)

        banner_h = 36
        banner_orig = np.full((banner_h, orig_w, 3), 25, dtype=np.uint8)
        cv2.putText(banner_orig, "ORIGINAL SONAR BACKSCATTER", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        banner_right = np.full((banner_h, right_w, 3), (20, 20, 140) if "RED" in right_title.upper() or "DETECTION" in right_title.upper() else 35, dtype=np.uint8)
        right_text_color = (100, 100, 255) if "RED" in right_title.upper() or "DETECTION" in right_title.upper() else (0, 220, 200)
        cv2.putText(banner_right, right_title, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, right_text_color, 1, cv2.LINE_AA)

        left_side = np.vstack([banner_orig, orig_resized])
        right_side = np.vstack([banner_right, right_resized])
        divider_full = np.full((target_h + banner_h, 6, 3), 50, dtype=np.uint8)

        composite = np.hstack([left_side, divider_full, right_side])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, composite)
        return output_path
