"""Trainable APF networks: the gradient-transfer U-Net ``T`` and the enhancement module ``E``.

Reference Code: https://github.com/adversarial-for-goodness/APF (GPL-3.0; see LICENSE)
"""
from __future__ import annotations

import torch
from torch import nn


class _DownBlock(nn.Module):
    """3x3 stride-2 conv + LeakyReLU (one encoder step)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _UpBlock(nn.Module):
    """3x3 stride-2 transposed conv + LeakyReLU (one decoder step)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UNetTransfer(nn.Module):
    """Gradient-transfer U-Net ``T``: ``[B, 3, 112, 112]`` probe noise to an unbounded perturbation
    for the server models.
    """

    def __init__(self) -> None:
        super().__init__()
        self.down1 = _DownBlock(3, 64)
        self.down2 = _DownBlock(64, 128)
        self.down3 = _DownBlock(128, 256)
        self.down4 = _DownBlock(256, 512)

        self.up1 = _UpBlock(512, 256)
        self.up2 = _UpBlock(512, 128)   # in: up1(256) + skip x3(256)
        self.up3 = _UpBlock(256, 64)    # in: up2(128) + skip x2(128)
        self.up4 = _UpBlock(128, 3)     # in: up3(64)  + skip x1(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x3_ = self.up1(x4)
        x2_ = self.up2(torch.cat([x3_, x3], dim=1))
        x1_ = self.up3(torch.cat([x2_, x2], dim=1))
        return self.up4(torch.cat([x1_, x1], dim=1))


class Enhancement(nn.Module):
    """Enhancement module ``E(u) = conv1x1(beta * u + gamma)`` over a frozen universal
    perturbation ``u`` (paper Eqn. 8).
    """

    def __init__(self, uap: torch.Tensor) -> None:
        super().__init__()
        if uap.dim() != 4 or uap.shape[1] != 3:
            raise ValueError(f"uap must be [1, 3, H, W], got {tuple(uap.shape)}")
        # Frozen prior (not a trainable parameter); persisted via save_run.
        self.register_buffer("uap", uap.detach().clone())
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return the enhanced universal perturbation broadcast to ``batch_size``."""
        enhanced = self.conv(self.uap * self.beta + self.gamma)
        return enhanced.expand(batch_size, -1, -1, -1)
