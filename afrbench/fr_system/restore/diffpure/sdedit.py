"""CelebA-HQ DDPM SDEdit step used by DiffPure (no file I/O, no ImageNet/CIFAR).

Adapted from ``DiffPure/runners/diffpure_ddpm.py`` (NVIDIA Source Code License).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def celeba_config() -> SimpleNamespace:
    """The ``configs/celeba.yml`` model / diffusion block as namespaces."""
    return SimpleNamespace(
        data=SimpleNamespace(dataset="CelebA_HQ", image_size=256, channels=3),
        model=SimpleNamespace(
            type="simple",
            in_channels=3,
            out_ch=3,
            ch=128,
            ch_mult=[1, 1, 2, 2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
            var_type="fixedsmall",
            ema_rate=0.999,
            ema=True,
            resamp_with_conv=True,
        ),
        diffusion=SimpleNamespace(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        ),
    )


def get_beta_schedule(*, beta_start: float, beta_end: float, num_diffusion_timesteps: int):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    return betas


def extract(values, t: torch.Tensor, x_shape):
    bs, = t.shape
    if torch.is_tensor(values):
        coeff = values.to(device=t.device, dtype=torch.float)
    else:
        coeff = torch.as_tensor(values, dtype=torch.float, device=t.device)
    out = torch.gather(coeff, 0, t.long())
    return out.reshape((bs,) + (1,) * (len(x_shape) - 1))


def image_editing_denoising_step_flexible_mask(x, t, *, model, logvar, betas):
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (
        x - extract(weighted_score, t, x.shape) * model_output
    )
    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    return mean + mask * torch.exp(0.5 * logvar) * noise


def diffusion_buffers(config: Any = None):
    cfg = config or celeba_config()
    betas = get_beta_schedule(
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps,
    )
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    if cfg.model.var_type == "fixedlarge":
        logvar = np.log(np.append(posterior_variance[1], betas[1:]))
    else:
        logvar = np.log(np.maximum(posterior_variance, 1e-20))
    return torch.from_numpy(betas).float(), logvar


def image_editing_sample(
    model,
    img: torch.Tensor,
    *,
    betas: torch.Tensor,
    logvar,
    t: int = 500,
    sample_step: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """SDEdit purify. ``img`` is NCHW in ``[-1, 1]``. Seeds the global RNG for ε."""
    if img.ndim != 4:
        raise ValueError(f"DiffPure expected [N,3,H,W], got {tuple(img.shape)}")
    batch_size = img.shape[0]
    torch.manual_seed(int(seed))
    if img.is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    x0 = img
    device_betas = betas.to(device=img.device, dtype=img.dtype)
    with torch.no_grad():
        for _ in range(int(sample_step)):
            e = torch.randn_like(x0)
            a = (1 - device_betas).cumprod(dim=0)
            total = int(t)
            x = x0 * a[total - 1].sqrt() + e * (1.0 - a[total - 1]).sqrt()
            for step in reversed(range(total)):
                time = torch.tensor([step] * batch_size, device=img.device)
                x = image_editing_denoising_step_flexible_mask(
                    x, t=time, model=model, logvar=logvar, betas=device_betas
                )
            x0 = x
    return x0
