"""
MarineVision - Evaluation Pipeline for Convolutional Autoencoder Sonar Anomaly Detector.
Evaluates the trained autoencoder on held-out normal and anomalous sonar test images.
Computes ROC-AUC, Precision, Recall, F1-Score, and anomaly separation metrics.
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix

from models.autoencoder import ConvAutoencoder
from config import Config


def evaluate_autoencoder(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print("=" * 70)
    print("MARINEVISION - AUTOENCODER ANOMALY DETECTOR EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Directory: {args.data_dir}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    # 1. Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    threshold = checkpoint.get('threshold', 0.12)
    input_size = tuple(checkpoint.get('input_size', (256, 256)))

    model = ConvAutoencoder(in_channels=1, input_size=input_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Validation-derived threshold: {threshold:.6f}")

    # 2. Discover normal and anomalous test images
    normal_dir = os.path.join(args.data_dir, 'test', 'normal')
    anom_dir = os.path.join(args.data_dir, 'test', 'anomalous')

    normal_files = sorted(glob.glob(os.path.join(normal_dir, '**', '*.jpg'), recursive=True) +
                          glob.glob(os.path.join(normal_dir, '**', '*.png'), recursive=True))
    anom_files = sorted(glob.glob(os.path.join(anom_dir, '**', '*.jpg'), recursive=True) +
                        glob.glob(os.path.join(anom_dir, '**', '*.png'), recursive=True))

    if not normal_files and not anom_files:
        # Fallback to search in test/
        test_dir = os.path.join(args.data_dir, 'test')
        all_test = sorted(glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True) +
                           glob.glob(os.path.join(test_dir, '**', '*.png'), recursive=True))
        normal_files = [f for f in all_test if 'bg_' in os.path.basename(f)]
        anom_files = [f for f in all_test if 'bg_' not in os.path.basename(f)]

    print(f"Found {len(normal_files)} normal test samples and {len(anom_files)} anomalous test samples.")

    if not normal_files and not anom_files:
        raise FileNotFoundError(f"No test images found in {args.data_dir}/test")

    # 3. Compute Anomaly Scores
    def score_images(file_list):
        scores = []
        for path in file_list:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            t_in = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                t_recon = model(t_in)
                t_err = torch.abs(t_in - t_recon)
                flat_err = t_err.view(-1)
                mean_e = flat_err.mean().item()
                top_k = max(1, int(len(flat_err) * 0.05))
                top_e = flat_err.topk(top_k)[0].mean().item()
                score = 0.5 * mean_e + 0.5 * top_e
                scores.append(score)
        return scores

    normal_scores = score_images(normal_files)
    anom_scores = score_images(anom_files)

    print(f"\nScored {len(normal_scores)} normal and {len(anom_scores)} anomalous test images.")

    y_true = [0] * len(normal_scores) + [1] * len(anom_scores)
    y_scores = normal_scores + anom_scores
    y_pred = [1 if s > threshold else 0 for s in y_scores]

    # 4. Compute Metrics
    results = {
        'threshold_used': float(threshold),
        'normal_count': len(normal_scores),
        'anomalous_count': len(anom_scores),
        'normal_mean_score': float(np.mean(normal_scores)) if normal_scores else 0.0,
        'normal_std_score': float(np.std(normal_scores)) if normal_scores else 0.0,
        'anomalous_mean_score': float(np.mean(anom_scores)) if anom_scores else 0.0,
        'anomalous_std_score': float(np.std(anom_scores)) if anom_scores else 0.0,
    }

    if len(normal_scores) > 0 and len(anom_scores) > 0:
        auc = roc_auc_score(y_true, y_scores)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        results['roc_auc'] = float(auc)
        results['precision'] = float(prec)
        results['recall'] = float(rec)
        results['f1_score'] = float(f1)
        results['confusion_matrix'] = cm

        print(f"\nTest Evaluation Metrics:")
        print(f"  ROC-AUC:               {auc:.4f}")
        print(f"  Precision:             {prec:.4f}")
        print(f"  Recall:                {rec:.4f}")
        print(f"  F1-Score:              {f1:.4f}")
        print(f"  Confusion Matrix:      TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

        # Plot ROC and score distribution
        out_dir = os.path.dirname(args.checkpoint)
        plot_path = os.path.join(out_dir, 'autoencoder_evaluation_plots.png')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Subplot 1: Score Distribution
        ax1.hist(normal_scores, bins=25, alpha=0.7, label='Normal Seabed', color='#00ADB5', density=True)
        ax1.hist(anom_scores, bins=25, alpha=0.7, label='Anomalous / Target', color='#FF6B6B', density=True)
        ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        ax1.set_title('Reconstruction Anomaly Score Distribution', fontweight='bold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Subplot 2: ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax2.plot(fpr, tpr, color='#008170', linewidth=2.5, label=f'Autoencoder (AUC = {auc:.4f})')
        ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax2.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Evaluation plots saved to: {plot_path}")

    # Save JSON summary
    out_dir = os.path.dirname(args.checkpoint)
    json_path = os.path.join(out_dir, 'autoencoder_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation summary JSON saved to: {json_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN Autoencoder Sonar Anomaly Detector")
    parser.add_argument('--checkpoint', type=str, default=Config.ANOMALY_WEIGHTS_PATH, help='Path to trained .pth checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/sonar_autoencoder', help='Directory containing test set')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU execution')

    args = parser.parse_args()
    evaluate_autoencoder(args)


if __name__ == '__main__':
    main()
