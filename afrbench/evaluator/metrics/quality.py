"""Visual quality of protected frames against their clean originals."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from afrlib import get_logger

logger = get_logger(__name__)

_MODELS: Dict[Tuple[str, str], torch.nn.Module] = {}


def psnr(protected: torch.Tensor, clean: torch.Tensor) -> float:
    """PSNR between two ``[3, H, W]`` images in ``[0, 255]`` (99.0 when identical)."""
    mse = ((protected - clean) ** 2).mean().item()
    if mse == 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(255.0**2 / mse)).item())


def ssim(protected: torch.Tensor, clean: torch.Tensor) -> float:
    from pytorch_msssim import ssim as _ssim

    return float(_ssim(_batch(protected), _batch(clean), data_range=255.0).item())


def ms_ssim(protected: torch.Tensor, clean: torch.Tensor) -> float:
    from pytorch_msssim import ms_ssim as _ms_ssim

    if min(protected.shape[-2:]) < 160:
        # MS-SSIM's 4 downsamplings need >= 160px; smaller frames report NaN
        # and are excluded from the mean.
        return float("nan")
    return float(_ms_ssim(_batch(protected), _batch(clean), data_range=255.0).item())


def lpips_distance(protected: torch.Tensor, clean: torch.Tensor, device) -> float:
    model = _model("lpips", device)
    return float(model(_batch_lpips(protected, device), _batch_lpips(clean, device)).item())


def _metric_device(device) -> torch.device:
    """Keep CUDA devices; run everything else on CPU (MPS lacks float64)."""
    target = torch.device(device) if not isinstance(device, torch.device) else device
    if target.type == "cuda":
        return target
    return torch.device("cpu")


def no_reference(image: torch.Tensor, metric: str, device) -> float:
    """BRISQUE or NIQE of one image via ``pyiqa``, lower being better for both; on CUDA where
    available, since crops are native resolution.
    """
    target = _metric_device(device)
    model = _model(metric, target)
    return float(model(_batch01(image, target)).item())


def fid(protected_dir, clean_dir, device, batch_size: int = 32) -> float:
    """FID between two directories of frames via ``torch_fidelity``, which takes a CUDA flag
    rather than a device, so other devices fall back to the CPU.
    """
    from torch_fidelity import calculate_metrics

    metrics = calculate_metrics(
        input1=str(protected_dir),
        input2=str(clean_dir),
        fid=True,
        batch_size=batch_size,
        cuda=str(device).startswith("cuda"),
        save_cpu_ram=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def valid_region(image: torch.Tensor, valid_rect: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Restrict a crop to its real-pixel rectangle, dropping repeated padding."""
    if not valid_rect:
        return image
    x, y, width, height = (int(v) for v in valid_rect[:4])
    if width <= 0 or height <= 0:
        return image
    return image[..., y : y + height, x : x + width]


def per_image(
    protected: torch.Tensor,
    clean: torch.Tensor,
    device,
    *,
    no_reference_metrics: Sequence[str] = ("brisque", "niqe"),
    valid_rect: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Every per-image quality metric for one ``(protected, clean)`` pair, resizing the clean
    frame if rounding left it a pixel off and excluding padding outside ``valid_rect``.
    """
    target = _metric_device(device)
    protected = valid_region(protected, valid_rect).to(target, non_blocking=True)
    clean = valid_region(clean, valid_rect).to(target, non_blocking=True)
    if protected.shape[-2:] != clean.shape[-2:]:
        from torchvision.transforms.functional import InterpolationMode, resize

        clean = resize(
            clean.unsqueeze(0), list(protected.shape[-2:]),
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        )[0]
    values = {
        "psnr": psnr(protected, clean),
        "ssim": ssim(protected, clean),
        "ms_ssim": ms_ssim(protected, clean),
        "lpips": lpips_distance(protected, clean, target),
        **perturbation_norms(protected, clean),
    }
    for metric in no_reference_metrics:
        values[metric] = no_reference(protected, metric, target)
    return values


def perturbation_norms(protected: torch.Tensor, clean: torch.Tensor) -> Dict[str, float]:
    """Perturbation size per subpixel on the 0-255 scale: mean and root-mean-square deviation,
    the fraction changed by more than half a level, and the peak deviation.
    """
    delta = (protected - clean).abs()
    return {
        "changed_fraction": float((delta > 0.5).to(torch.float32).mean().item()),
        "mae": float(delta.mean().item()),
        "rmse": float(delta.pow(2).mean().sqrt().item()),
        "max_abs": float(delta.max().item()),
    }


def mean_of(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Column means over per-image rows, skipping NaN (e.g. MS-SSIM under 160px)."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and value == value:
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _batch(image: torch.Tensor) -> torch.Tensor:
    return image.unsqueeze(0) if image.dim() == 3 else image


def _batch01(image: torch.Tensor, device) -> torch.Tensor:
    return (image.to(device, torch.float32) / 255.0).clamp(0, 1).unsqueeze(0)


def _batch_lpips(image: torch.Tensor, device) -> torch.Tensor:
    """LPIPS expects RGB in ``[-1, 1]``."""
    return (image.to(device, torch.float32) / 127.5 - 1.0).clamp(-1, 1).unsqueeze(0)


def _model(name: str, device) -> torch.nn.Module:
    key = (name, str(device))
    if key not in _MODELS:
        if name == "lpips":
            import lpips as _lpips

            module = _lpips.LPIPS(net="alex")
        else:
            import pyiqa

            module = pyiqa.create_metric(name, device=device)
        _MODELS[key] = module.to(device).eval().requires_grad_(False)
    return _MODELS[key]
