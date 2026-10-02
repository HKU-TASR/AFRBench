"""Differentiable sticker projection and spatial transform for AdvHat.

Reference Code: https://github.com/papermsucode/advhat
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _integral(value: torch.Tensor, constant: torch.Tensor) -> torch.Tensor:
    root = torch.sqrt(value.square() + constant)
    return 0.5 * (
        value * root
        + constant * torch.log(torch.abs(value + root))
    )


def _pre_parabola(x: torch.Tensor, parabola: torch.Tensor) -> torch.Tensor:
    """Map arc-length coordinates onto the parabolic cylinder."""
    centered = x - 450.0
    constant = 0.25 / parabola.square()
    mapped = 2.0 * parabola * (
        _integral(centered.abs(), constant)
        - _integral(torch.zeros_like(centered), constant)
    )
    return mapped + 450.0


def project_sticker(
    sticker: torch.Tensor,
    parabola: torch.Tensor,
    pitch_degrees: torch.Tensor,
) -> torch.Tensor:
    """Bend and pitch a batch of stickers.

    Args:
        sticker: NCHW float tensor ``[B,3,400,900]`` in ``[0,1]``.
        parabola: ``[B,1]`` parabolic rates.
        pitch_degrees: ``[B,1]`` pitch angles in degrees.

    Returns:
        NCHW float tensor ``[B,3,900,900]``.
    """
    if sticker.ndim != 4 or tuple(sticker.shape[1:]) != (3, 400, 900):
        raise ValueError(
            "sticker must have shape [B,3,400,900], got "
            f"{tuple(sticker.shape)}"
        )
    batch = sticker.shape[0]
    if tuple(parabola.shape) != (batch, 1):
        raise ValueError(
            f"parabola must have shape [{batch},1], got {tuple(parabola.shape)}"
        )
    if tuple(pitch_degrees.shape) != (batch, 1):
        raise ValueError(
            "pitch_degrees must have shape "
            f"[{batch},1], got {tuple(pitch_degrees.shape)}"
        )

    # NHWC, as in the TensorFlow reference code, for an op-by-op port; the result is NCHW.
    logo = sticker.permute(0, 2, 3, 1)
    right_cumsum = F.pad(
        torch.cumsum(logo[:, :, 450:], dim=2),
        (0, 0, 1, 0),
    ).permute(0, 2, 1, 3)
    left_cumsum = F.pad(
        torch.cumsum(logo[:, :, :450].flip(2), dim=2),
        (0, 0, 1, 0),
    ).permute(0, 2, 1, 3)

    coordinates = torch.arange(
        450, 901, device=sticker.device, dtype=sticker.dtype
    ).unsqueeze(0)
    anchors = torch.round(
        (_pre_parabola(coordinates, parabola) - 450.0).clamp(0.0, 450.0)
    ).to(torch.long)
    batch_indices = torch.arange(batch, device=sticker.device)[:, None]
    anchor_widths = (
        anchors[:, 1:] - anchors[:, :-1]
    ).clamp(1, 900).to(sticker.dtype)[:, :, None, None]

    right_values = right_cumsum[batch_indices, anchors]
    left_values = left_cumsum[batch_indices, anchors]
    right = (right_values[:, 1:] - right_values[:, :-1]) / anchor_widths
    left = (left_values[:, 1:] - left_values[:, :-1]) / anchor_widths
    compressed = torch.cat((left.flip(1), right), dim=1).permute(0, 2, 1, 3)
    vertical_cumsum = F.pad(
        torch.cumsum(compressed, dim=1),
        (0, 0, 0, 0, 1, 0),
    )

    angle = torch.deg2rad(pitch_degrees).unsqueeze(2)
    x = torch.arange(900, device=sticker.device, dtype=sticker.dtype)
    z = parabola * (x - 449.5).square()
    z = z[:, None, :].expand(-1, 901, -1)
    y = torch.arange(
        -250, 651, device=sticker.device, dtype=sticker.dtype
    )[None, :, None].expand(batch, -1, 900)
    source_y = (y + z * torch.sin(-angle)) / torch.cos(angle)
    source_y = torch.round(source_y.clamp(0.0, 400.0)).to(torch.long)
    source_heights = (
        source_y[:, 1:] - source_y[:, :-1]
    ).clamp(1, 900).to(sticker.dtype)

    x_indices = torch.arange(900, device=sticker.device)[None, None, :]
    x_indices = x_indices.expand(batch, 901, -1)
    batch_grid = torch.arange(batch, device=sticker.device)[:, None, None]
    batch_grid = batch_grid.expand(-1, 901, 900)
    chosen = vertical_cumsum[batch_grid, source_y, x_indices]
    result = (chosen[:, 1:] - chosen[:, :-1]) / source_heights[..., None]
    return result.clamp(0.0, 1.0).permute(0, 3, 1, 2)


def _pixel_value(
    image: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    batch, _, height, width = image.shape
    flat = image.permute(0, 2, 3, 1)
    batch_grid = torch.arange(batch, device=image.device)[:, None, None]
    batch_grid = batch_grid.expand(-1, x.shape[1], x.shape[2])
    return flat[batch_grid, y, x]


def spatial_transform(
    image: torch.Tensor,
    theta: torch.Tensor,
    output_size: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """The reference code's affine grid and bilinear sampler, on NCHW images."""
    if image.ndim != 4:
        raise ValueError(f"image must be NCHW, got {tuple(image.shape)}")
    batch, _, height, width = image.shape
    if tuple(theta.shape) != (batch, 6):
        raise ValueError(
            f"theta must have shape [{batch},6], got {tuple(theta.shape)}"
        )
    if output_size is None:
        output_height, output_width = height, width
    else:
        output_height, output_width = int(output_size[0]), int(output_size[1])

    x = torch.linspace(
        -1.0, 1.0, output_width, device=image.device, dtype=image.dtype
    )
    y = torch.linspace(
        -1.0, 1.0, output_height, device=image.device, dtype=image.dtype
    )
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack(
        (x_grid.reshape(-1), y_grid.reshape(-1), torch.ones_like(x_grid).reshape(-1))
    )
    grid = theta.reshape(batch, 2, 3) @ base_grid.unsqueeze(0)
    x_source = grid[:, 0].reshape(batch, output_height, output_width)
    y_source = grid[:, 1].reshape(batch, output_height, output_width)

    max_y = height - 1
    max_x = width - 1
    # As upstream, scaling by max_{x,y} - 1 excludes the last source row and
    # column even for an identity transform.
    x_source = 0.5 * (x_source + 1.0) * float(max_x - 1)
    y_source = 0.5 * (y_source + 1.0) * float(max_y - 1)

    x0 = torch.floor(x_source).to(torch.long)
    x1 = x0 + 1
    y0 = torch.floor(y_source).to(torch.long)
    y1 = y0 + 1
    x0 = x0.clamp(0, max_x)
    x1 = x1.clamp(0, max_x)
    y0 = y0.clamp(0, max_y)
    y1 = y1.clamp(0, max_y)

    top_left = _pixel_value(image, x0, y0)
    bottom_left = _pixel_value(image, x0, y1)
    top_right = _pixel_value(image, x1, y0)
    bottom_right = _pixel_value(image, x1, y1)

    x0_float, x1_float = x0.to(image.dtype), x1.to(image.dtype)
    y0_float, y1_float = y0.to(image.dtype), y1.to(image.dtype)
    weight_tl = (x1_float - x_source) * (y1_float - y_source)
    weight_bl = (x1_float - x_source) * (y_source - y0_float)
    weight_tr = (x_source - x0_float) * (y1_float - y_source)
    weight_br = (x_source - x0_float) * (y_source - y0_float)
    output = (
        weight_tl[..., None] * top_left
        + weight_bl[..., None] * bottom_left
        + weight_tr[..., None] * top_right
        + weight_br[..., None] * bottom_right
    )
    return output.permute(0, 3, 1, 2)


def total_variation(sticker: torch.Tensor, weight: float = 1e-4) -> torch.Tensor:
    """Weighted isotropic total variation, as in the reference code."""
    vertical = sticker[:, :, 1:] - sticker[:, :, :-1]
    horizontal = sticker[:, :, :, 1:] - sticker[:, :, :, :-1]
    vertical_sq = F.pad(vertical.square(), (0, 0, 1, 0))
    horizontal_sq = F.pad(horizontal.square(), (1, 0, 0, 0))
    return float(weight) * torch.sqrt(
        vertical_sq + horizontal_sq + 1e-5
    ).sum()


def composite_sticker(
    faces: torch.Tensor,
    sticker: torch.Tensor,
    parabola: torch.Tensor,
    pitch_degrees: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render and composite the sticker onto 600x600 aligned faces; return (composite, alpha)."""
    projected = project_sticker(sticker, parabola, pitch_degrees)
    prepared = spatial_transform(projected, theta)
    mask = spatial_transform(
        project_sticker(torch.ones_like(sticker), parabola, pitch_degrees),
        theta,
    )
    rendered = prepared[:, :, 300:, 150:750]
    alpha = mask[:, :, 300:, 150:750]
    return rendered * alpha + faces * (1.0 - alpha), alpha


__all__ = [
    "composite_sticker",
    "project_sticker",
    "spatial_transform",
    "total_variation",
]
