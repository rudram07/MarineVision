"""
MarineVision - Evaluation Pipeline for YOLO Side-Scan Sonar Marine Debris Detector.
Evaluates the trained model checkpoint on the held-out test set.
Computes Precision, Recall, mAP@0.5, mAP@0.5:0.95, and generates visual predictions.
"""

import os
import json
import argparse
import time
from pathlib import Path

from config import Config


def evaluate_yolo(args):
    print("=" * 70)
    print("MARINEVISION - YOLO MARINE DEBRIS DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Checkpoint:   {args.weights}")
    print(f"Dataset YAML: {args.data}")
    print(f"Split:        {args.split}")
    print(f"Device:       {args.device}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint weights not found at {args.weights}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    print(f"Model loaded successfully from {args.weights}.")

    # 1. Run Ultralytics Validation on Test Split
    print(f"\nRunning evaluation on '{args.split}' split...")
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=False,
        verbose=True
    )

    # 2. Extract standard metrics
    metrics = {
        'weights_path': args.weights,
        'split': args.split,
        'precision': round(float(results.box.p.mean()), 4) if hasattr(results.box, 'p') and len(results.box.p) > 0 else 0.0,
        'recall': round(float(results.box.r.mean()), 4) if hasattr(results.box, 'r') and len(results.box.r) > 0 else 0.0,
        'map50': round(float(results.box.map50), 4) if hasattr(results.box, 'map50') else 0.0,
        'map50_95': round(float(results.box.map), 4) if hasattr(results.box, 'map') else 0.0,
        'speed_ms': results.speed if hasattr(results, 'speed') else {},
        'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Per-class metrics
    class_names = model.names
    per_class = {}
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        for cls_id, map_val in enumerate(results.box.maps):
            c_name = class_names.get(cls_id, f"class_{cls_id}")
            per_class[c_name] = round(float(map_val), 4)
    metrics['per_class_map50_95'] = per_class

    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY:")
    print("=" * 50)
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  mAP@0.5:         {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95:    {metrics['map50_95']:.4f}")
    if per_class:
        print("  Per-Class mAP@0.5:0.95:")
        for c, m in per_class.items():
            print(f"    - {c}: {m:.4f}")

    # 3. Save Summary to JSON
    out_dir = os.path.dirname(args.weights)
    json_path = os.path.join(out_dir, 'yolo_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation summary saved to: {json_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained YOLO Debris Detector on Test Split")
    parser.add_argument('--weights', type=str, default=Config.YOLO_WEIGHTS_PATH, help='Trained .pt model weights path')
    parser.add_argument('--data', type=str, default='configs/yolo_data.yaml', help='YOLO dataset yaml path')
    parser.add_argument('--split', type=str, default='test', help='Split to evaluate: test or val')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or 0')

    args = parser.parse_args()
    evaluate_yolo(args)


if __name__ == '__main__':
    main()
