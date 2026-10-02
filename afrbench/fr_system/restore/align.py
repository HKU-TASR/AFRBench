"""5-point FFHQ templates for the restorers, and the warp from their canvas to ArcFace-112."""
from __future__ import annotations

from typing import Tuple

import torch

from frbench import ARCFACE_112_TEMPLATE, align

#: GFPGAN / InsightFace 5-point template on a 512 canvas (x, y).
FFHQ_512_TEMPLATE = torch.tensor(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=torch.float32,
)

RESTORER_CANVAS = {
    "gfpgan": 512,
    "codeformer": 512,
    "diffpure": 256,
}

MODEL_RESTORERS = tuple(RESTORER_CANVAS)


def ffhq_template(size: int = 512) -> torch.Tensor:
    """5-point FFHQ template scaled to a square ``size`` canvas."""
    if int(size) == 512:
        return FFHQ_512_TEMPLATE.clone()
    scale = float(size) / 512.0
    return FFHQ_512_TEMPLATE * scale


def compose_to_arcface(canvas: torch.Tensor, canvas_size: int) -> torch.Tensor:
    """Warp an already-aligned restorer canvas, ``[3, H, W]`` or ``[N, 3, H, W]`` in ``[0, 255]``,
    onto the ArcFace-112 template, using the FFHQ template as the source landmarks.
    """
    template = ffhq_template(int(canvas_size)).to(device=canvas.device, dtype=canvas.dtype)
    dest = ARCFACE_112_TEMPLATE.to(device=canvas.device, dtype=canvas.dtype)
    if canvas.ndim == 3:
        return align(canvas, template.unsqueeze(0), dest, (112, 112))[0]
    if canvas.ndim != 4:
        raise ValueError(f"expected [3,H,W] or [N,3,H,W], got {tuple(canvas.shape)}")
    rows = [
        align(canvas[index], template.unsqueeze(0), dest, (112, 112))[0]
        for index in range(canvas.shape[0])
    ]
    return torch.stack(rows, dim=0)


def restorer_canvas_size(method: str) -> int:
    """Side of the square canvas a restorer works on (512, or 256 for DiffPure)."""
    try:
        return RESTORER_CANVAS[method]
    except KeyError as exc:
        raise ValueError(f"unknown restorer {method!r}") from exc


def restorer_output_size(method: str) -> Tuple[int, int]:
    side = restorer_canvas_size(method)
    return (side, side)
