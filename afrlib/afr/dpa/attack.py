"""Hard model aggregation: DPA's impersonation I-FGSM over the snapshot ensemble.

Reference Code: https://github.com/fengfanzhou/DPA
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .models import HardModelConv

# The reference constants (config.py, hma.py), on the [0, 255] scale.
EPSILON = 10.0
STEP_SIZE = 1.0
N_ITER = 200
ADV_FEAT_STEP = 8e-4
# Input diversity: the reference samples both stds from U(0.01, 0.1) each iter.
TRANSFORM_STD_LOW = 0.01
TRANSFORM_STD_HIGH = 0.1


def squared_l2_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance ``2 - 2 cos`` between L2-normalized embeddings, per row."""
    a = F.normalize(emb_a, p=2, dim=1)
    b = F.normalize(emb_b, p=2, dim=1)
    return ((a - b) ** 2).sum(dim=1)


def _masked_grid_sample(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Affine-warp ``x`` by ``theta``, zeroing samples that reach outside the frame, as the
    reference ``image_transform.affine`` does.
    """
    grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(
        torch.ones_like(x), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    mask = (mask > 0.9999).to(x.dtype)
    return out * mask


def input_diversity(x: torch.Tensor) -> torch.Tensor:
    """The reference input transform ``T``: a random affine warp and rotation, each with a
    standard deviation drawn per call.
    """
    b, _, h, w = x.shape
    device, dtype = x.device, x.dtype

    std_proj = torch.empty(1).uniform_(TRANSFORM_STD_LOW, TRANSFORM_STD_HIGH).item()
    theta_p = torch.eye(2, 3, dtype=dtype, device=device).unsqueeze(0).repeat(b, 1, 1)
    theta_p = theta_p + torch.randn(b, 2, 3, dtype=dtype, device=device) * std_proj
    x = _masked_grid_sample(x, theta_p)

    std_rotate = torch.empty(1).uniform_(TRANSFORM_STD_LOW, TRANSFORM_STD_HIGH).item()
    ang = torch.randn(b, dtype=dtype, device=device) * std_rotate
    cos, sin = torch.cos(ang), torch.sin(ang)
    zero = torch.zeros_like(cos)
    theta_r = torch.stack(
        [
            torch.stack([cos, sin, zero], dim=1),
            torch.stack([-sin, cos, zero], dim=1),
        ],
        dim=1,
    )
    x = _masked_grid_sample(x, theta_r)
    return x.contiguous()


def hma_attack(
    surrogates: Sequence[torch.nn.Module],
    wrappers: Sequence[HardModelConv],
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = EPSILON,
    step_size: float = STEP_SIZE,
    n_iter: int = N_ITER,
) -> torch.Tensor:
    """Craft an impersonation adversarial example for ``source`` toward ``target``.

    Args:
        surrogates: The hard-model surrogates (``SurrogateForward`` with wrapped
            convs), each mapping ``[B, 3, H, W]`` RGB in ``[0, 255]`` to ``[B, D]``.
        wrappers: The ``HardModelConv`` wrappers across all surrogates; reset here
            so each attack starts with no stored gradient.
        source: Source image(s), ``[B, 3, H, W]`` RGB floats in ``[0, 255]``.
        target: Target-identity image(s), broadcastable to ``source``'s batch.

    Returns:
        The adversarial image(s), same shape/range as ``source``.
    """
    if target.shape[0] == 1 and source.shape[0] > 1:
        target = target.expand(source.shape[0], -1, -1, -1)

    for wrapper in wrappers:
        wrapper.reset()
    for model in surrogates:
        model.eval()

    clip_min = (source - epsilon).clamp(min=0.0)
    clip_max = (source + epsilon).clamp(max=255.0)
    # Contiguous even if ``source`` is a permuted view.
    adv = source.detach().clone().contiguous().requires_grad_(True)

    device = source.device
    n_models = len(surrogates)
    if n_models == 0:
        raise ValueError("HMA requires at least one surrogate")

    # Target embeddings are fixed. Each model visits the device only while in
    # use, so the ensemble stays on the CPU between steps.
    target_embs = []
    with torch.no_grad():
        for model in surrogates:
            model = model.to(device)
            model.eval()
            target_embs.append(model(target).detach())
            model.cpu()

    for _ in range(n_iter):
        grad_acc = torch.zeros_like(adv)
        for model, target_emb in zip(surrogates, target_embs):
            model = model.to(device)
            model.eval()
            adv_emb = model(input_diversity(adv))
            loss = squared_l2_distance(adv_emb, target_emb.to(device)).mean() / n_models
            grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]
            grad_acc = grad_acc + grad
            for wrapper in model.modules():
                if isinstance(wrapper, HardModelConv) and wrapper.grad_prev is not None:
                    wrapper.grad_prev = wrapper.grad_prev.detach().cpu()
            model.cpu()
        adv = adv.detach() - step_size * torch.sign(grad_acc)
        adv = torch.min(torch.max(adv, clip_min), clip_max)
        adv = adv.detach().requires_grad_(True)
    return adv.detach()
