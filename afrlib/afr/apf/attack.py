"""APF stage A: I2-FGSM noise from the probe model, with the reference input diversity.

Reference Code: https://github.com/adversarial-for-goodness/APF (GPL-3.0; see LICENSE)
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

# The reference constants on the [0, 255] scale (the TF code works in [-1, 1]).
EPSILON = 8.0
STEP_SIZE = 2.0
N_ITER = 10
DIVERSITY_PROB = 0.5


def squared_l2_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance ``2 - 2 * cos(a, b)`` between L2-normalized embeddings, per row."""
    a = F.normalize(emb_a, p=2, dim=1)
    b = F.normalize(emb_b, p=2, dim=1)
    return ((a - b) ** 2).sum(dim=1)


def _rand_apply(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], prob: float) -> torch.Tensor:
    """Apply ``fn`` to the whole batch with probability ``prob`` (one coin flip per batch)."""
    if torch.rand(()) < prob:
        return fn(x)
    return x


def _pixel_enhance(x: torch.Tensor) -> torch.Tensor:
    """The reference ``input_enhance``, reproduced as released: its double rescaling shifts every
    pixel by about -128.5 levels (not a +-5 jitter) before clamping.
    """
    max_pixel = 5.0 / 127.5 - 1.0
    min_pixel = -5.0 / 127.5 - 1.0
    random_pixel = torch.empty_like(x).uniform_(min_pixel, max_pixel)
    random_pixel = random_pixel / 127.5 - 1.0
    # ``random_pixel`` is in the [-1, 1] scale; convert the jitter to [0, 255].
    jitter = random_pixel * 127.5
    return (x + jitter).clamp(0.0, 255.0)


def _scale_pad(x: torch.Tensor) -> torch.Tensor:
    """The reference ``input_scaled``: nearest-neighbour downscale and random zero padding."""
    b, _, h, w = x.shape
    rnd = int(torch.randint(h - 2, h + 1, (1,)).item())  # [h-2, h]
    rescaled = F.interpolate(x, size=(rnd, rnd), mode="nearest")
    h_rem = h - rnd
    w_rem = w - rnd
    pad_top = int(torch.randint(0, h_rem + 1, (1,)).item()) if h_rem > 0 else 0
    pad_left = int(torch.randint(0, w_rem + 1, (1,)).item()) if w_rem > 0 else 0
    pad_bottom = h_rem - pad_top
    pad_right = w_rem - pad_left
    # .contiguous(): downstream FR backbones reshape with .view(), which needs a
    # contiguous tensor.
    return F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom)).contiguous()


def _rotate(x: torch.Tensor) -> torch.Tensor:
    """The reference ``input_rotate``: a random rotation per image within +-15 degrees about the
    centre, with zeros outside.
    """
    b, _, h, w = x.shape
    angles = (torch.rand(b, device=x.device, dtype=x.dtype) * 2 - 1) * (math.pi / 12.0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zeros = torch.zeros_like(cos)
    # Rotation about the centre; out-of-frame samples are zero (align_corners=False).
    theta = torch.stack(
        [
            torch.stack([cos, -sin, zeros], dim=1),
            torch.stack([sin, cos, zeros], dim=1),
        ],
        dim=1,
    )  # [B, 2, 3]
    grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def input_diversity(x: torch.Tensor, prob: float = DIVERSITY_PROB) -> torch.Tensor:
    """The reference ``input_diversity``: pixel enhance, then scale/pad, then rotation."""
    x = _rand_apply(x, _pixel_enhance, prob)
    x = _rand_apply(x, _scale_pad, prob)
    x = _rand_apply(x, _rotate, prob)
    # .contiguous(): grid_sample / pad can return non-contiguous tensors, and the
    # FR backbones reshape with .view().
    return x.contiguous()


def i2fgsm(
    x: torch.Tensor,
    enrolled: torch.Tensor,
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    epsilon: float = EPSILON,
    step_size: float = STEP_SIZE,
    n_iter: int = N_ITER,
    prob: float = DIVERSITY_PROB,
) -> torch.Tensor:
    """I2-FGSM dodging: ``n_iter`` sign steps up the probe's distance between ``x`` and ``enrolled``,
    within the L-inf ``epsilon`` ball and ``[0, 255]``.

    Args:
        x: Source images, ``[B, 3, H, W]`` in ``[0, 255]``.
        enrolled: Enrolled same-identity images, same shape/range as ``x``.
        embed_fn: Maps a ``[0, 255]`` batch to probe embeddings ``[B, D]``
            (typically ``lambda imgs: probe.embed_crops(imgs / 255.0, input_space="01")``).

    Returns:
        The adversarial images ``x_adv`` (same shape/range as ``x``).
    """
    clip_min = (x - epsilon).clamp(min=0.0)
    clip_max = (x + epsilon).clamp(max=255.0)
    adv = x.clamp(clip_min, clip_max)

    with torch.no_grad():
        enrolled_emb = embed_fn(enrolled)

    for _ in range(n_iter):
        adv = adv.detach().requires_grad_(True)
        diverse = input_diversity(adv, prob)
        emb = embed_fn(diverse)
        dist = squared_l2_distance(emb, enrolled_emb).sum()
        grad = torch.autograd.grad(dist, adv)[0]
        adv = adv.detach() + step_size * torch.sign(grad)
        adv = torch.min(torch.max(adv, clip_min), clip_max)
    return adv.detach()
