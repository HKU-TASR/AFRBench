"""The image-space operations an FR system's pipeline is built from."""
from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

from .diffjpeg import apply_diffjpeg

def _map_nchw(image: torch.Tensor, apply_one: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Apply a CHW-only op to each image of an NCHW batch."""
    if image.ndim == 3:
        return apply_one(image)
    if image.ndim != 4:
        raise ValueError(f"expected [3,H,W] or [N,3,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        return apply_one(image[0]).unsqueeze(0)
    return torch.stack([apply_one(image[index]) for index in range(image.shape[0])])


def _odd_kernel(size: float) -> int:
    kernel = int(round(float(size)))
    if kernel < 1:
        kernel = 1
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def quantize(image: torch.Tensor, precision: str = "uint8") -> torch.Tensor:
    """Reduce precision: round to the 8-bit grid (``uint8``) or through ``float16``."""
    if precision == "uint8":
        return image.clamp(0, 255).round()
    if precision == "float16":
        return image.to(torch.float16).to(torch.float32)
    raise ValueError(f"Unknown quantize precision {precision!r}")


def jpeg(image: torch.Tensor, quality: int = 75, impl: str = "diffjpeg") -> torch.Tensor:
    """DiffJPEG encode and decode at a fixed ``quality``, which is not scaled with face width,
    so the output may be non-integer and a smaller face is filtered harder.
    """
    if impl != "diffjpeg":
        raise ValueError(f"jpeg.impl must be 'diffjpeg'; got {impl!r}")
    return apply_diffjpeg(image, quality=int(quality))


def resize(image: torch.Tensor, factor: float = 0.4, mode: str = "bicubic") -> torch.Tensor:
    """Downscale then upscale by ``factor`` (platform rescaling)."""
    from torchvision.transforms.functional import InterpolationMode, resize as _resize

    interpolation = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }[mode]
    height, width = image.shape[-2:]
    small = _resize(
        image, [max(int(height * factor), 8), max(int(width * factor), 8)],
        interpolation=interpolation, antialias=True,
    )
    return _resize(small, [height, width], interpolation=interpolation, antialias=True)


def median(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Median filter; ``kernel_size`` is rounded to an odd size >= 1."""
    k = _odd_kernel(kernel_size)

    def _one(crop: torch.Tensor) -> torch.Tensor:
        pad = k // 2
        padded = F.pad(crop, (pad, pad, pad, pad), mode="reflect")
        patches = padded.unfold(1, k, 1).unfold(2, k, 1)
        height, width = crop.shape[-2:]
        return patches.reshape(crop.shape[0], height, width, k * k).median(dim=-1).values

    return _map_nchw(image, _one)


def gfpgan(image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    from .restore.models import run_gfpgan

    return run_gfpgan(image, **kwargs)


def codeformer(image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    from .restore.models import run_codeformer

    return run_codeformer(image, **kwargs)


def diffpure(image: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    from .restore.models import run_diffpure

    return run_diffpure(image, **kwargs)


#: The op registry. Keys are the names a pipeline config uses.
OPS: Dict[str, Callable[..., torch.Tensor]] = {
    "quantize": quantize,
    "jpeg": jpeg,
    "resize": resize,
    "median": median,
    "diffpure": diffpure,
    "gfpgan": gfpgan,
    "codeformer": codeformer,
}

#: Ops that are purifiers (the ``purify`` stage draws from these).
PURIFIERS = ("median", "diffpure", "gfpgan", "codeformer")
