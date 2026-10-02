"""AIDPro networks: the frozen face manipulation model (FMM) and the WI/WD modules.

Reference Code: https://github.com/daizigege/AIDPro
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    """Per-instance, per-channel normalization used by the released FMM."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - torch.mean(inputs, dim=(2, 3), keepdim=True)
        inv_std = torch.rsqrt(
            torch.mean(centered * centered, dim=(2, 3), keepdim=True)
            + self.epsilon
        )
        return centered * inv_std


class ApplyStyle(nn.Module):
    """Map a 512-D identity to AdaIN scale and bias parameters."""

    def __init__(self, latent_size: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(
        self, inputs: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        style = self.linear(identity).view(
            -1, 2, inputs.shape[1], 1, 1
        )
        return inputs * (style[:, 0] + 1.0) + style[:, 1]


class ResnetBlockAdain(nn.Module):
    """Identity-conditioned (AdaIN) residual block of the FMM."""

    def __init__(
        self,
        channels: int,
        latent_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            InstanceNorm(),
        )
        self.style1 = ApplyStyle(latent_size, channels)
        self.act1 = activation
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            InstanceNorm(),
        )
        self.style2 = ApplyStyle(latent_size, channels)

    def forward(
        self, inputs: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        residual = self.conv1(inputs)
        residual = self.style1(residual, identity)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        residual = self.style2(residual, identity)
        return inputs + residual


class Generator_Adain_Upsample(nn.Module):
    """Frozen 224-pixel FMM with checkpoint-compatible parameter names."""

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        latent_size: int = 512,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")

        activation = nn.ReLU(inplace=True)
        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.BatchNorm2d(64),
            activation,
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            activation,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            activation,
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            activation,
        )
        self.BottleNeck = nn.Sequential(
            *[
                ResnetBlockAdain(
                    channels=512,
                    latent_size=latent_size,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ]
        )
        self.up3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            activation,
        )
        self.up2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        # No output activation, as in the released checkpoint.
        self.last_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
        )

    def forward(
        self, inputs: torch.Tensor, identity: torch.Tensor
    ) -> torch.Tensor:
        features = self.first_layer(inputs)
        features = self.down1(features)
        features = self.down2(features)
        features = self.down3(features)
        for block in self.BottleNeck:
            features = block(features, identity)
        features = self.up3(features)
        features = self.up2(features)
        features = self.up1(features)
        return self.last_layer(features)


class WatermarkInsert(nn.Module):
    """Two-layer identity/message transformer (WI)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        output_size: int = 512,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.leaky_relu(self.fc1(inputs)))


class WatermarkDecoder(nn.Module):
    """Two-layer message decoder (WD), returning logits."""

    def __init__(
        self,
        output_size: int,
        input_size: int = 512,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.leaky_relu(self.fc1(inputs)))


__all__ = [
    "Generator_Adain_Upsample",
    "WatermarkDecoder",
    "WatermarkInsert",
]
