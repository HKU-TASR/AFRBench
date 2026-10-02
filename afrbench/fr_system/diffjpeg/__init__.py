"""Dynamic-size DiffJPEG wrapper.

Upstream ``DiffJPEG`` is constructed for a fixed ``H×W``. Eval pads to a
multiple of 8, reuses one module per ``(H, W, Q, device)``, and crops back.
``differentiable=False`` uses ``torch.round``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .codec import DiffJPEG

_CACHE: Dict[Tuple[int, int, int, str], DiffJPEG] = {}


def _pad_to_multiple(image: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, int, int]:
    # 4:2:0 chroma is H/2 × W/2, so both sides must be multiples of 16.
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return image, height, width


def _module(height: int, width: int, quality: int, device: torch.device) -> DiffJPEG:
    key = (int(height), int(width), int(quality), str(device))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    module = DiffJPEG(
        height=int(height),
        width=int(width),
        differentiable=False,
        quality=int(quality),
    ).to(device)
    module.eval()
    _CACHE[key] = module
    return module


def apply_diffjpeg(image: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """JPEG-compress ``[0, 255]`` NCHW or CHW RGB; return float ``[0, 255]``."""
    squeezed = image.ndim == 3
    if squeezed:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"jpeg expected [3,H,W] or [N,3,H,W], got {tuple(image.shape)}")
    padded, height, width = _pad_to_multiple(image.to(dtype=torch.float32))
    module = _module(padded.shape[-2], padded.shape[-1], int(quality), padded.device)
    # DiffJPEG compresses ``image * 255`` internally; feed [0, 1].
    with torch.inference_mode():
        recovered = module(padded.clamp(0, 255) / 255.0)
    recovered = recovered[..., :height, :width] * 255.0
    return recovered[0] if squeezed else recovered
