"""
MarineVision - Convolutional Autoencoder for Underwater Sonar Anomaly Detection.
Trained from scratch (no pretrained weights, no transfer learning).
Reconstructs verified normal seabed sonar backscatter and flags regions with high
reconstruction error as acoustic seabed anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    Symmetric Convolutional Encoder-Decoder for Side-Scan Sonar Anomaly Detection.
    All trainable layers are initialized from scratch using Kaiming normal initialization.
    """
    def __init__(self, in_channels=1, latent_channels=64, input_size=(256, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.input_size = input_size

        # -------------------------------------------------------------
        # Encoder: Progressively downsamples 256x256 -> 16x16
        # -------------------------------------------------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (32, 128, 128)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (64, 64, 64)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (128, 32, 32)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (256, 16, 16)

        # Latent bottleneck representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (latent_channels, 16, 16)

        # -------------------------------------------------------------
        # Decoder: Progressively reconstructs 16x16 -> 256x256
        # -------------------------------------------------------------
        self.dec_bottleneck = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (256, 16, 16)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (128, 32, 32)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (64, 64, 64)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (32, 128, 128)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()  # Normalizes output pixel intensities to [0, 1]
        )  # -> (in_channels, 256, 256)

        # Explicitly initialize all weights from scratch
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes every trainable weight from scratch using Kaiming normal
        initialization. Guarantees no pretrained weights are present.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def encode(self, x):
        """Passes input through encoder to produce latent bottleneck."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.bottleneck(e4)
        return z

    def decode(self, z):
        """Passes latent code through decoder to produce reconstruction."""
        d0 = self.dec_bottleneck(z)
        d1 = self.dec1(d0)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        reconstruction = self.dec4(d3)
        return reconstruction

    def forward(self, x):
        """Full autoencoder forward pass returning reconstructed sonar image."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    @staticmethod
    def compute_error_map(x, x_recon):
        """
        Computes absolute pixel-wise reconstruction error map in [0, 1].
        x, x_recon: torch.Tensor of shape (B, C, H, W)
        """
        return torch.abs(x - x_recon)

    @staticmethod
    def compute_image_anomaly_score(error_map, top_percentile=95):
        """
        Calculates a robust image-level anomaly score from the error map.
        Combines mean error with top-percentile error to be sensitive to localized anomalies.
        """
        b = error_map.size(0)
        flat_errors = error_map.view(b, -1)
        mean_err = flat_errors.mean(dim=1)
        k = max(1, int(flat_errors.size(1) * (1.0 - top_percentile / 100.0)))
        top_k_err, _ = torch.topk(flat_errors, k=k, dim=1)
        top_err = top_k_err.mean(dim=1)
        score = 0.5 * mean_err + 0.5 * top_err
        return score
