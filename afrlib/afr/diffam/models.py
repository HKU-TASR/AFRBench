"""CelebA-HQ 256x256 DDPM U-Net used by DiffAM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DDPMConfig:
    """Fixed architecture of the published ``celeba_hq.ckpt``."""

    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    residual_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.0
    resample_with_conv: bool = True


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    """Build the sinusoidal timestep embedding used by the original DDPM."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a one-dimensional tensor")
    half = dimension // 2
    scale = math.log(10_000) / (half - 1)
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float32) * -scale
    ).to(timesteps.device)
    embedding = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.conv(value) if self.with_conv else value


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(F.pad(value, (0, 1, 0, 1)))
        return F.avg_pool2d(value, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = group_norm(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        if in_channels != self.out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, self.out_channels, 3, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1(swish(self.norm1(value)))
        hidden = hidden + self.temb_proj(swish(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(swish(self.norm2(hidden))))
        if self.in_channels != self.out_channels:
            value = (
                self.conv_shortcut(value)
                if self.use_conv_shortcut
                else self.nin_shortcut(value)
            )
        return value + hidden


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(value)
        query, key, vector = self.q(hidden), self.k(hidden), self.v(hidden)
        batch, channels, height, width = query.shape
        query = query.reshape(batch, channels, height * width).permute(0, 2, 1)
        key = key.reshape(batch, channels, height * width)
        weights = torch.bmm(query, key) * (int(channels) ** -0.5)
        weights = weights.softmax(dim=2).permute(0, 2, 1)
        hidden = torch.bmm(vector.reshape(batch, channels, height * width), weights)
        hidden = self.proj_out(hidden.reshape(batch, channels, height, width))
        return value + hidden


class DDPM(nn.Module):
    """State-dict-compatible reimplementation of DiffAM's 256-pixel U-Net."""

    def __init__(self, config: DDPMConfig | None = None) -> None:
        super().__init__()
        config = config or DDPMConfig()
        channels = config.channels
        multipliers = config.channel_multipliers
        self.config = config
        self.ch = channels
        self.temb_ch = channels * 4
        self.num_resolutions = len(multipliers)
        self.num_res_blocks = config.residual_blocks
        self.resolution = config.image_size
        self.in_channels = config.in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            (nn.Linear(channels, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch))
        )
        self.conv_in = nn.Conv2d(config.in_channels, channels, 3, padding=1)

        current_resolution = config.image_size
        in_multipliers = (1,) + multipliers
        self.down = nn.ModuleList()
        block_in = channels
        for level in range(self.num_resolutions):
            blocks = nn.ModuleList()
            attention = nn.ModuleList()
            block_in = channels * in_multipliers[level]
            block_out = channels * multipliers[level]
            for _ in range(config.residual_blocks):
                blocks.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if current_resolution in config.attention_resolutions:
                    attention.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = blocks
            down.attn = attention
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, config.resample_with_conv)
                current_resolution //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=config.dropout,
        )

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            blocks = nn.ModuleList()
            attention = nn.ModuleList()
            block_out = channels * multipliers[level]
            skip_in = channels * multipliers[level]
            for block_index in range(config.residual_blocks + 1):
                if block_index == config.residual_blocks:
                    skip_in = channels * in_multipliers[level]
                blocks.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if current_resolution in config.attention_resolutions:
                    attention.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = blocks
            up.attn = attention
            if level != 0:
                up.upsample = Upsample(block_in, config.resample_with_conv)
                current_resolution *= 2
            self.up.insert(0, up)

        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, config.out_channels, 3, padding=1)

    def forward(
        self, value: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if value.shape[2:] != (self.resolution, self.resolution):
            raise ValueError(
                f"DDPM expects {self.resolution}x{self.resolution}, "
                f"got {tuple(value.shape[2:])}"
            )
        time_embedding = timestep_embedding(timesteps, self.ch)
        time_embedding = self.temb.dense[1](
            swish(self.temb.dense[0](time_embedding))
        )

        history = [self.conv_in(value)]
        for level in range(self.num_resolutions):
            for block_index in range(self.num_res_blocks):
                hidden = self.down[level].block[block_index](
                    history[-1], time_embedding
                )
                if self.down[level].attn:
                    hidden = self.down[level].attn[block_index](hidden)
                history.append(hidden)
            if level != self.num_resolutions - 1:
                history.append(self.down[level].downsample(history[-1]))

        hidden = self.mid.block_1(history[-1], time_embedding)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden, time_embedding)

        for level in reversed(range(self.num_resolutions)):
            for block_index in range(self.num_res_blocks + 1):
                hidden = self.up[level].block[block_index](
                    torch.cat((hidden, history.pop()), dim=1), time_embedding
                )
                if self.up[level].attn:
                    hidden = self.up[level].attn[block_index](hidden)
            if level != 0:
                hidden = self.up[level].upsample(hidden)

        if history:
            raise RuntimeError("DDPM skip-connection stack did not balance")
        return self.conv_out(swish(self.norm_out(hidden)))


__all__ = ["DDPM", "DDPMConfig", "timestep_embedding"]
