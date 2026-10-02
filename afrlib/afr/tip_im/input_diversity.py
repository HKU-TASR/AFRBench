"""Random affine and rotation input diversity for TIP-IM.

Reference Code: https://github.com/ShawnXYang/TIP-IM (Apache-2.0; see LICENSE)
"""
import numpy as np
import torch
import torch.nn.functional as F


def affine(x: torch.Tensor, vgrid: torch.Tensor) -> torch.Tensor:
    """grid_sample with a hard validity mask (out-of-frame pixels zeroed)."""
    # The released code targets PyTorch >= 1.4, whose default is
    # align_corners=False; set it explicitly.
    output = F.grid_sample(x, vgrid, align_corners=False)
    mask = F.grid_sample(torch.ones_like(x), vgrid, align_corners=False)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    return output * mask


def rotate(x: torch.Tensor, theta: np.ndarray) -> torch.Tensor:
    rotation = np.zeros((2, 3, x.size(0)))
    cos = np.cos(theta).ravel()
    sin = np.sin(theta).ravel()
    rotation[0, 0] = cos
    rotation[0, 1] = sin
    rotation[1, 0] = -sin
    rotation[1, 1] = cos
    rotation = torch.tensor(rotation.transpose((2, 0, 1)), dtype=torch.float32, device=x.device)
    grid = F.affine_grid(rotation, size=x.size(), align_corners=False)
    return affine(x, grid)


def input_diversity(
    x: torch.Tensor,
    std_proj: float,
    std_rotate: float,
    rng=None,
) -> torch.Tensor:
    """Random small projective jitter followed by a random small rotation."""
    rng = np.random if rng is None else rng
    n = x.size(0)
    M = np.tile(np.array([[1, 0, 0], [0, 1, 0]]), (n, 1, 1)) + rng.normal(
        scale=std_proj,
        size=(n, 2, 3),
    )
    grid = F.affine_grid(
        torch.tensor(M, dtype=torch.float32, device=x.device),
        x.size(),
        align_corners=False,
    )
    x = affine(x, grid)
    theta = rng.normal(scale=std_rotate, size=(n, 1))
    return rotate(x, theta)
