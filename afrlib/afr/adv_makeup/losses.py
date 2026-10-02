"""Adv-Makeup training losses and Poisson blending.

Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (commit 359af7e; academic research only, see LICENSE)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import factorized
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
).view(1, 1, 3, 3)


class VGG16Features(nn.Module):
    """Frozen VGG16 feature slices (relu1_2 to relu4_3) for the style loss."""

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        network = models.vgg16(weights=None)
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        network.load_state_dict(state, strict=True)
        features = network.features
        self.slice1 = nn.Sequential(*features[:4])
        self.slice2 = nn.Sequential(*features[4:9])
        self.slice3 = nn.Sequential(*features[9:16])
        self.slice4 = nn.Sequential(*features[16:23])
        self.requires_grad_(False)
        self.eval()

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relu1_2 = self.slice1(inputs)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        return relu1_2, relu2_2, relu3_3, relu4_3


class MeanShift(nn.Conv2d):
    """RGB mean subtraction of the released code, applied directly to ``[-1, 1]`` crops."""

    def __init__(self) -> None:
        super().__init__(3, 3, kernel_size=1)
        means = torch.tensor((0.4488, 0.4371, 0.4040))
        with torch.no_grad():
            self.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            self.bias.copy_(-means)
        self.requires_grad_(False)


def cosine_distance(
    embeddings: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Impersonation loss ``1 - mean cosine similarity``."""

    return 1.0 - F.cosine_similarity(
        embeddings, targets, dim=1
    ).mean()


def discriminator_mse(
    logits: torch.Tensor, *, real: bool
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    labels = torch.ones_like(probabilities) if real else torch.zeros_like(
        probabilities
    )
    return F.mse_loss(probabilities, labels)


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    return flattened.bmm(flattened.transpose(1, 2)) / (
        channels * height * width
    )


def style_loss(
    vgg: VGG16Features,
    mean_shift: MeanShift,
    originals: Sequence[torch.Tensor],
    composites: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean Gram-matrix style loss, scaled by 1e5 as in the released code."""

    if len(originals) != len(composites) or not originals:
        raise ValueError("Style loss needs matching non-empty crop sequences")
    values = []
    for original, composite in zip(originals, composites):
        original_grams = [
            gram_matrix(value)
            for value in vgg(mean_shift(original))
        ]
        composite_grams = [
            gram_matrix(value)
            for value in vgg(mean_shift(composite))
        ]
        value = torch.stack(
            [
                F.mse_loss(actual, expected)
                for actual, expected in zip(
                    composite_grams, original_grams
                )
            ]
        ).mean()
        values.append(value * 100_000.0)
    return torch.stack(values).mean()


def laplacian_channels(
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel = _LAPLACIAN_KERNEL.to(
        device=images.device, dtype=images.dtype
    )
    return tuple(
        F.conv2d(images[:, channel : channel + 1], kernel, padding=1)
        .squeeze(1)
        for channel in range(3)
    )


def gradient_loss(
    originals: Sequence[torch.Tensor],
    composites: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(originals) != len(composites) or not originals:
        raise ValueError("Gradient loss needs matching non-empty crops")
    values = []
    for original, composite in zip(originals, composites):
        expected = laplacian_channels(original)
        actual = laplacian_channels(composite)
        values.extend(
            F.mse_loss(value, target)
            for value, target in zip(actual, expected)
        )
    return torch.stack(values).mean()


def total_variation_loss(
    composites: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not composites:
        raise ValueError("TV loss needs at least one crop")
    values = []
    for image in composites:
        horizontal = torch.mean(
            torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        )
        vertical = torch.mean(
            torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        )
        values.append(horizontal + vertical)
    return torch.stack(values).mean()


def _poisson_matrix(height: int, width: int) -> scipy.sparse.lil_matrix:
    row = scipy.sparse.lil_matrix((width, width))
    row.setdiag(-1, -1)
    row.setdiag(4)
    row.setdiag(-1, 1)
    matrix = scipy.sparse.block_diag([row] * height).tolil()
    matrix.setdiag(-1, width)
    matrix.setdiag(-1, -width)
    return matrix


def poisson_blend(
    generated_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Poisson-blend uint8 ``generated_rgb`` into ``original_rgb`` inside ``mask``, per channel as
    in the reference code (the outer pixel ring keeps its Laplacian rows).
    """

    source = np.asarray(generated_rgb, dtype=np.uint8)
    target = np.asarray(original_rgb, dtype=np.uint8).copy()
    binary = np.asarray(mask).copy()
    if source.shape != target.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Poisson source and target must match HxWx3")
    if binary.shape != source.shape[:2]:
        raise ValueError("Poisson mask must match the image height and width")
    binary[binary != 0] = 1

    height, width = binary.shape
    matrix = _poisson_matrix(height, width)
    laplacian = matrix.tocoo()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if binary[y, x] == 0:
                index = x + y * width
                matrix[index, index] = 1
                matrix[index, index + 1] = 0
                matrix[index, index - 1] = 0
                matrix[index, index + width] = 0
                matrix[index, index - width] = 0
    matrix = matrix.tocsc()
    solve = factorized(matrix)

    mask_flat = binary.reshape(-1)
    for channel in range(3):
        source_flat = source[:, :, channel].reshape(-1)
        target_flat = target[:, :, channel].reshape(-1)
        rhs = laplacian.dot(source_flat)
        rhs[mask_flat == 0] = target_flat[mask_flat == 0]
        solved = solve(rhs).reshape(height, width)
        target[:, :, channel] = np.clip(solved, 0, 255).astype(np.uint8)
    return target
