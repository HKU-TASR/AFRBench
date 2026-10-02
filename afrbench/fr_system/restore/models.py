"""Restorer models (GFPGAN v1.3, CodeFormer, DiffPure), loaded lazily once per device."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from afrlib.utils.assets import get_asset

from .align import restorer_canvas_size

_MODULES: Dict[Tuple[str, str], Any] = {}
_DIFFPURE_STATE: Dict[str, Any] = {}


def _device_key(device: "torch.device | str") -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None and torch.cuda.is_available():
        torch_device = torch.device("cuda", torch.cuda.current_device())
    return str(torch_device)


def _load_gfpgan(device: torch.device):
    from .gfpgan.gfpganv1_clean_arch import GFPGANv1Clean

    path = get_asset("bench/gfpgan_v1.3.pth")
    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key], strict=True)
    return model.to(device).eval()


def _load_codeformer(device: torch.device):
    from .codeformer.codeformer_arch import CodeFormer

    path = get_asset("bench/codeformer.pth")
    model = CodeFormer(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=["32", "64", "128", "256"],
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval()


def _load_diffpure(device: torch.device):
    from .diffpure.sdedit import celeba_config, diffusion_buffers
    from .diffpure.unet_ddpm import Model

    path = get_asset("bench/diffpure_celeba_hq.ckpt")
    config = celeba_config()
    model = Model(config)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    betas, logvar = diffusion_buffers(config)
    return model.to(device).eval(), betas, logvar


def get_restorer(name: str, device: "torch.device | str"):
    """The restorer ``name`` on ``device``, loaded once and cached."""
    torch_device = torch.device(device)
    key = (name, _device_key(torch_device))
    cached = _MODULES.get(key)
    if cached is not None:
        return cached
    if name == "gfpgan":
        module = _load_gfpgan(torch_device)
    elif name == "codeformer":
        module = _load_codeformer(torch_device)
    elif name == "diffpure":
        module, betas, logvar = _load_diffpure(torch_device)
        _DIFFPURE_STATE[_device_key(torch_device)] = (betas, logvar)
    else:
        raise ValueError(f"unknown restorer {name!r}")
    _MODULES[key] = module
    return module


def diffpure_buffers(device: "torch.device | str"):
    """DiffPure's ``(betas, logvar)`` diffusion buffers on ``device``."""
    get_restorer("diffpure", device)
    return _DIFFPURE_STATE[_device_key(device)]


def _as_nchw(image: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim == 4:
        return image, False
    raise ValueError(f"expected [3,H,W] or [N,3,H,W], got {tuple(image.shape)}")


def _require_canvas(image: torch.Tensor, method: str) -> None:
    side = restorer_canvas_size(method)
    height, width = image.shape[-2:]
    if height != side or width != side:
        raise ValueError(
            f"{method} expects an already-aligned {side}x{side} canvas; "
            f"got {height}x{width}. The FFHQ hop lives in FRSystem.embed."
        )


def run_gfpgan(image: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
    """Restore FFHQ-512 canvases with GFPGAN v1.3; ``weight`` / ``w`` are ignored."""
    batch, squeezed = _as_nchw(image)
    _require_canvas(batch, "gfpgan")
    model = get_restorer("gfpgan", batch.device)
    x = batch.to(dtype=torch.float32).clamp(0, 255).div(255.0)
    x = x.sub(0.5).div(0.5)
    with torch.inference_mode():
        out = model(x, return_rgb=False, randomize_noise=False)[0]
    restored = out.clamp(-1, 1).add(1).mul(127.5)
    return restored[0] if squeezed else restored


def run_codeformer(
    image: torch.Tensor,
    w: float = 0.5,
    adain: bool = True,
    **_kwargs: Any,
) -> torch.Tensor:
    """Restore an FFHQ-512 canvas with CodeFormer (``w=0.5``, ``adain=True``)."""
    batch, squeezed = _as_nchw(image)
    _require_canvas(batch, "codeformer")
    model = get_restorer("codeformer", batch.device)
    x = batch.to(dtype=torch.float32).clamp(0, 255).div(255.0)
    x = x.sub(0.5).div(0.5)
    with torch.inference_mode():
        out = model(x, w=float(w), adain=bool(adain))[0]
    restored = out.clamp(-1, 1).add(1).mul(127.5)
    return restored[0] if squeezed else restored


def run_diffpure(
    image: torch.Tensor,
    t: int = 500,
    sample_step: int = 1,
    seed: int = 0,
    **_kwargs: Any,
) -> torch.Tensor:
    """Purify 256 canvases with the CelebA-HQ DiffPure model (SDEdit from step ``t``)."""
    from .diffpure.sdedit import image_editing_sample

    batch, squeezed = _as_nchw(image)
    _require_canvas(batch, "diffpure")
    model = get_restorer("diffpure", batch.device)
    betas, logvar = diffpure_buffers(batch.device)
    x = batch.to(dtype=torch.float32).clamp(0, 255).div(255.0)
    x = x.mul(2.0).sub(1.0)
    out = image_editing_sample(
        model, x, betas=betas, logvar=logvar, t=int(t),
        sample_step=int(sample_step), seed=int(seed),
    )
    restored = out.clamp(-1, 1).add(1).mul(127.5)
    return restored[0] if squeezed else restored
