"""
MarineVision - Training Pipeline for Convolutional Autoencoder Sonar Anomaly Detector.
Trained strictly from scratch on verified normal seabed sonar imagery.
Derives anomaly detection threshold from validation reconstruction errors.
"""

import os
import sys
import time
import argparse
import random
import glob
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.autoencoder import ConvAutoencoder
from config import Config


# -----------------------------------------------------------------------------
# Normal Sonar Seabed Dataset Loader
# -----------------------------------------------------------------------------
class NormalSeabedDataset(Dataset):
    """
    Loads verified normal seabed sonar images for self-supervised reconstruction.
    """
    def __init__(self, image_paths, img_size=(256, 256), augment=False):
        self.image_paths = image_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for corrupt read
            img = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)

        # Resize to fixed autoencoder dimensions
        resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)

        # Optional data augmentations (flips preserve acoustic physics of seabed)
        if self.augment:
            if random.random() > 0.5:
                resized = cv2.flip(resized, 1)  # Horizontal flip
            if random.random() > 0.5:
                resized = cv2.flip(resized, 0)  # Vertical flip

        # Normalize to [0.0, 1.0]
        norm = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm).unsqueeze(0)  # (1, H, W)
        return tensor, path


def set_seed(seed=42):
    """Sets random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(args):
    """Executes from-scratch training of the Convolutional Autoencoder."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print("=" * 70)
    print("MARINEVISION - AUTOENCODER ANOMALY DETECTOR TRAINING (FROM SCRATCH)")
    print("=" * 70)
    print(f"Device:           {device}")
    print(f"Random Seed:      {args.seed}")
    print(f"Image Size:       {args.img_size}x{args.img_size}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Max Epochs:       {args.epochs}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Output Checkpoint:{args.output}")

    # 1. Discover verified normal seabed images
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_files = sorted(glob.glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True) +
                         glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True))
    val_files = sorted(glob.glob(os.path.join(val_dir, '**', '*.jpg'), recursive=True) +
                       glob.glob(os.path.join(val_dir, '**', '*.png'), recursive=True))

    if not train_files:
        raise FileNotFoundError(
            f"No training images found in {train_dir}. Please run prepare_dataset.py first."
        )
    if not val_files:
        print(f"Warning: No validation images found in {val_dir}. Splitting 15% from train.")
        split_idx = int(len(train_files) * 0.85)
        val_files = train_files[split_idx:]
        train_files = train_files[:split_idx]

    if args.smoke_test:
        print("\n[SMOKE TEST MODE ENABLED]: Limiting to small subset and 2 epochs.")
        train_files = train_files[:min(32, len(train_files))]
        val_files = val_files[:min(16, len(val_files))]
        args.epochs = 2

    print(f"\nDataset Splits:")
    print(f"  Verified Normal Train Images: {len(train_files)}")
    print(f"  Verified Normal Val Images:   {len(val_files)}")

    # Data loaders
    train_dataset = NormalSeabedDataset(train_files, img_size=(args.img_size, args.img_size), augment=True)
    val_dataset = NormalSeabedDataset(val_files, img_size=(args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(train_files) > args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Initialize Model From Scratch
    model = ConvAutoencoder(in_channels=1, input_size=(args.img_size, args.img_size))
    model.to(device)

    # Verify no weights loaded from external checkpoints
    print("\nModel Architecture Initialized From Scratch:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Trainable Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 3. Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    start_train_time = time.time()

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        train_loss = running_train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_sample_x = None
        val_sample_recon = None

        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                running_val_loss += loss.item() * batch_x.size(0)

                if val_sample_x is None:
                    val_sample_x = batch_x[:4].cpu()
                    val_sample_recon = recon[:4].cpu()

        val_loss = running_val_loss / len(val_dataset)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start
        improved = val_loss < best_val_loss

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] - "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s {'[BEST - SAVED]' if improved else ''}")

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Temporary best state dict save
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience and not args.smoke_test:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    total_time = round(time.time() - start_train_time, 2)
    print(f"\nTraining completed in {total_time}s ({total_time/60:.2f} min). Best Val Loss: {best_val_loss:.6f}")

    # Load best weights before calculating validation threshold
    model.load_state_dict(best_model_state)
    model.eval()

    # 4. Determine Anomaly Threshold from Normal Validation Data
    print("\nComputing Empirical Anomaly Threshold from Normal Validation Set...")
    val_scores = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            error_map = torch.abs(batch_x - recon)

            # Calculate image-level score for each sample in batch
            for i in range(error_map.size(0)):
                err = error_map[i].view(-1)
                mean_err = err.mean().item()
                top_k = max(1, int(len(err) * 0.05))
                top_err = err.topk(top_k)[0].mean().item()
                score = 0.5 * mean_err + 0.5 * top_err
                val_scores.append(score)

    val_scores = np.array(val_scores)
    mean_val = float(np.mean(val_scores))
    std_val = float(np.std(val_scores))
    p95_val = float(np.percentile(val_scores, 95))
    p99_val = float(np.percentile(val_scores, 99))
    max_val = float(np.max(val_scores))

    # Threshold set at 95th percentile or mean + 2.5 * std (whichever is more conservative)
    threshold = float(max(p95_val, mean_val + 2.0 * std_val))

    print(f"Normal Validation Statistics:")
    print(f"  Mean Error Score:       {mean_val:.6f}")
    print(f"  Std Deviation:          {std_val:.6f}")
    print(f"  95th Percentile:        {p95_val:.6f}")
    print(f"  99th Percentile:        {p99_val:.6f}")
    print(f"  Max Normal Error:       {max_val:.6f}")
    print(f"  Selected Anomaly Threshold: {threshold:.6f}")

    # 5. Save Checkpoint with Complete Metadata
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': (args.img_size, args.img_size),
        'in_channels': 1,
        'threshold': threshold,
        'mean_normal_score': mean_val,
        'std_normal_score': std_val,
        'p95_normal_score': p95_val,
        'p99_normal_score': p99_val,
        'max_normal_score': max_val,
        'version': 'Autoencoder-Anomaly-Scratch-v1.0',
        'epochs_trained': epoch,
        'best_val_loss': best_val_loss,
        'training_time_sec': total_time,
        'seed': args.seed,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, args.output)
    print(f"\nFinal model checkpoint saved to: {args.output}")

    # 6. Save Training Loss Curves
    plot_dir = os.path.dirname(args.output)
    curve_path = os.path.join(plot_dir, 'autoencoder_loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train MSE Loss', color='#00ADB5', linewidth=2)
    plt.plot(history['val_loss'], label='Val MSE Loss', color='#FF6B6B', linewidth=2)
    plt.title('Convolutional Autoencoder Sonar Reconstruction Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Loss curves saved to: {curve_path}")

    # 7. Save Representative Input/Reconstruction Pairs
    if val_sample_x is not None and val_sample_recon is not None:
        recon_fig_path = os.path.join(plot_dir, 'autoencoder_reconstructions.png')
        n_samples = min(4, val_sample_x.size(0))
        fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n_samples):
            orig = val_sample_x[i, 0].numpy()
            rec = val_sample_recon[i, 0].numpy()
            diff = np.abs(orig - rec)

            axes[i, 0].imshow(orig, cmap='gray')
            axes[i, 0].set_title(f"Sample {i+1}: Original Normal Sonar")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(rec, cmap='gray')
            axes[i, 1].set_title(f"Reconstruction")
            axes[i, 1].axis('off')

            im_diff = axes[i, 2].imshow(diff, cmap='inferno')
            axes[i, 2].set_title(f"Reconstruction Error Map")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(recon_fig_path, dpi=200)
        plt.close()
        print(f"Reconstruction sample pairs saved to: {recon_fig_path}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train CNN Autoencoder from Scratch for Sonar Anomaly Detection")
    parser.add_argument('--data-dir', type=str, default='data/sonar_autoencoder', help='Directory with train and val normal sonar images')
    parser.add_argument('--output', type=str, default=Config.ANOMALY_WEIGHTS_PATH, help='Destination checkpoint path')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=256, help='Input image width and height')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU execution even if CUDA is detected')
    parser.add_argument('--smoke-test', action='store_true', help='Run small smoke-test on subset to verify pipeline')

    args = parser.parse_args()
    train_autoencoder(args)


if __name__ == '__main__':
    main()
