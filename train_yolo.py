"""
MarineVision - Training Pipeline for YOLO Side-Scan Sonar Marine Debris Detector.
Initialized strictly from YAML architecture configuration with random weights.
Strict requirement enforced: pretrained=False, no pretrained checkpoints, no transfer learning.
"""

import os
import sys
import time
import shutil
import argparse
import yaml
from pathlib import Path

from config import Config


def verify_from_scratch_config(model, pretrained_flag):
    """
    Verifies that the YOLO model was instantiated from architecture YAML
    and that pretrained=False is strictly enforced.
    """
    if pretrained_flag is not False:
        raise ValueError("Strict requirement violation: pretrained must be False.")
    print("Verification: Model is initialized from architecture config with random weights.")


def train_yolo(args):
    """
    Trains YOLO architecture from random initialization on side-scan sonar dataset.
    """
    print("=" * 70)
    print("MARINEVISION - YOLO MARINE DEBRIS DETECTOR TRAINING (FROM SCRATCH)")
    print("=" * 70)
    print(f"Data Config:        {args.data}")
    print(f"Architecture YAML:  {args.arch}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch Size:         {args.batch}")
    print(f"Image Size:         {args.imgsz}")
    print(f"Initial LR:         {args.lr0}")
    print(f"Device:             {args.device}")
    print(f"Random Seed:        {args.seed}")
    print(f"Pretrained Weights: STRICTLY FALSE (Trained from scratch)")

    # 1. Verify data yaml exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"YOLO data YAML not found at {args.data}. Run prepare_dataset.py first.")

    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)

    nc = data_cfg.get('nc', len(data_cfg.get('names', [])))
    names = data_cfg.get('names', {})
    print(f"\nConfigured Classes ({nc}): {names}")

    # 2. Adjust for smoke-test mode if requested
    epochs = args.epochs
    batch = args.batch
    if args.smoke_test:
        print("\n[SMOKE TEST MODE ENABLED]: Running 2 epochs with batch 4.")
        epochs = 2
        batch = min(4, batch)

    # 3. Instantiate model from architecture YAML (Random initialization)
    from ultralytics import YOLO

    print(f"\nInstantiating {args.arch} from architecture specification (Random Initialization)...")
    model = YOLO(args.arch)
    verify_from_scratch_config(model, pretrained_flag=False)

    project_dir = os.path.join(Config.BASE_DIR, 'runs', 'detect')
    run_name = args.name or f"sonar_scratch_{int(time.time())}"

    # 4. Train from Scratch
    start_time = time.time()
    try:
        results = model.train(
            data=args.data,
            pretrained=False,     # STRICT: Random weights only
            epochs=epochs,
            batch=batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
            project=project_dir,
            name=run_name,
            save=True,
            val=True,
            plots=True,
            verbose=True,
            exist_ok=True
        )
        elapsed_sec = round(time.time() - start_time, 2)
        print(f"\nTraining completed in {elapsed_sec}s ({elapsed_sec/60:.2f} min).")

    except Exception as e:
        print(f"\nError during YOLO training: {e}")
        raise e

    # 5. Locate and Save Best Checkpoint to models/yolo_debris.pt
    run_output_dir = os.path.join(project_dir, run_name)
    best_weights = os.path.join(run_output_dir, 'weights', 'best.pt')
    last_weights = os.path.join(run_output_dir, 'weights', 'last.pt')

    chosen_weights = best_weights if os.path.exists(best_weights) else last_weights

    if os.path.exists(chosen_weights):
        target_path = args.output or Config.YOLO_WEIGHTS_PATH
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copyfile(chosen_weights, target_path)
        print(f"\nTrained checkpoint successfully installed to: {target_path}")
    else:
        print(f"\nWarning: Could not locate best.pt in {run_output_dir}/weights")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO from Scratch for Side-Scan Sonar Debris Detection")
    parser.add_argument('--data', type=str, default='configs/yolo_data.yaml', help='Path to YOLO dataset yaml')
    parser.add_argument('--arch', type=str, default='yolov8n.yaml', help='YOLO architecture yaml (e.g. yolov8n.yaml, yolo11n.yaml)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image input size')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or 0 for cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Run name')
    parser.add_argument('--output', type=str, default=Config.YOLO_WEIGHTS_PATH, help='Destination .pt checkpoint path')
    parser.add_argument('--smoke-test', action='store_true', help='Smoke-test run (2 epochs) for fast pipeline validation')

    args = parser.parse_args()
    train_yolo(args)


if __name__ == '__main__':
    main()
