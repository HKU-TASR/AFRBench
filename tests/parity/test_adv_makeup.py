"""Adv-Makeup losses against the released inline formulas.

Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (``model.py``)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import models

from afrlib.afr.adv_makeup.losses import (
    MeanShift,
    VGG16Features,
    discriminator_mse,
    gradient_loss,
    style_loss,
    total_variation_loss,
)


def _random_vgg(tmp_path: Path) -> VGG16Features:
    network = models.vgg16(weights=None)
    path = tmp_path / "vgg16.pth"
    torch.save(network.state_dict(), path)
    return VGG16Features(path)


def test_discriminator_mse_matches_formula():
    torch.manual_seed(0)
    logits = torch.randn(4, 1)
    ours = discriminator_mse(logits, real=True)
    expected = torch.nn.functional.mse_loss(torch.sigmoid(logits), torch.ones_like(logits))
    assert torch.allclose(ours, expected, atol=1e-5)


def test_tv_and_gradient_match_released_formulas():
    """Total variation matches the released formula; the gradient loss is finite."""
    torch.manual_seed(0)
    original = torch.rand(1, 3, 16, 16)
    composite = torch.rand(1, 3, 16, 16)
    ours_tv = total_variation_loss([composite])
    horiz = torch.mean(torch.abs(composite[:, :, :, :-1] - composite[:, :, :, 1:]))
    vert = torch.mean(torch.abs(composite[:, :, :-1, :] - composite[:, :, 1:, :]))
    assert torch.allclose(ours_tv, horiz + vert, atol=1e-5)

    ours_grad = gradient_loss([original], [composite])
    assert torch.isfinite(ours_grad)


def test_style_loss_mean_gram_mse_times_1e5(tmp_path):
    torch.manual_seed(0)
    vgg = _random_vgg(tmp_path)
    shift = MeanShift()
    original = torch.rand(1, 3, 32, 32)
    composite = torch.rand(1, 3, 32, 32)
    ours = style_loss(vgg, shift, [original], [composite])
    o_grams = [afr_gram(feat) for feat in vgg(shift(original))]
    c_grams = [afr_gram(feat) for feat in vgg(shift(composite))]
    expected = torch.stack([
        torch.nn.functional.mse_loss(actual, target)
        for actual, target in zip(c_grams, o_grams)
    ]).mean() * 100_000.0
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


def afr_gram(features: torch.Tensor) -> torch.Tensor:
    """Reference Gram matrix, normalized by ``C * H * W``."""
    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    return flat.bmm(flat.transpose(1, 2)) / (channels * height * width)
