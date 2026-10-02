"""Eyeglass mask, frame movement and printable palette for Accessorize."""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from frbench import (
    arcface_template,
    estimate_similarity_transform,
)

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_SIZE = 224


def asset_path(name: str) -> str:
    return os.path.join(_ASSET_DIR, name)


def load_ref_marks() -> np.ndarray:
    """Load the released 68-point VGG reference landmarks ``(68, 2)``."""
    path = asset_path("ref_marks.csv")
    marks = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if marks.shape != (68, 2):
        raise ValueError(f"Expected ref_marks shape (68, 2), got {marks.shape}")
    return marks


def vgg68_to_5(ref_marks: np.ndarray) -> np.ndarray:
    """Reduce dlib-68 VGG reference landmarks to the ArcFace 5-point layout."""
    left_eye = ref_marks[36:42].mean(axis=0)
    right_eye = ref_marks[42:48].mean(axis=0)
    nose = ref_marks[30]
    mouth_left = ref_marks[48]
    mouth_right = ref_marks[54]
    return np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)


def vgg_to_arcface_matrix(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Similarity ``(2, 3)`` mapping VGG-aligned pixels -> ArcFace ``size x size``."""
    vgg5 = torch.from_numpy(vgg68_to_5(load_ref_marks()).astype(np.float32)).unsqueeze(0)
    arc5 = arcface_template((size, size))
    return estimate_similarity_transform(vgg5, arc5)[0].cpu().numpy()


def load_arcface_frame_mask(device: torch.device, size: int = DEFAULT_SIZE) -> torch.Tensor:
    """Binary eyeglass mask ``(1, 3, size, size)`` in ArcFace coordinates, warped from the
    reference silhouette's VGG pose (cached for 224 px).
    """
    cached = asset_path("silhouette_arcface224.png")
    if size == DEFAULT_SIZE and os.path.isfile(cached):
        mask_np = np.array(Image.open(cached).convert("L"), dtype=np.float32) / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
    else:
        sil = np.array(Image.open(asset_path("silhouette.png")).convert("RGB"))
        mask_np = (sil != 0).any(axis=2).astype(np.float32)
        matrix = vgg_to_arcface_matrix(size)
        a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
        c, d, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
        det = a * d - b * c
        inv = torch.tensor(
            [[d / det, -b / det, (-d * tx + b * ty) / det],
             [-c / det, a / det, (c * tx - a * ty) / det]],
            dtype=torch.float32,
        )
        A = inv[:, :2]
        t = inv[:, 2]
        s = float(size - 1)
        A_n = A.clone()
        t_n = (2.0 / s) * t + A.sum(dim=1) - 1.0
        theta_n = torch.zeros(1, 2, 3)
        theta_n[0, :, :2] = A_n
        theta_n[0, :, 2] = t_n
        src = torch.from_numpy(mask_np).view(1, 1, *mask_np.shape).float()
        if mask_np.shape != (size, size):
            src = F.interpolate(src, size=(size, size), mode="nearest")
        grid = F.affine_grid(theta_n, size=(1, 1, size, size), align_corners=True)
        warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        mask_np = (warped[0, 0].numpy() > 0.5).astype(np.float32)
    mask = torch.from_numpy(mask_np).view(1, 1, size, size).repeat(1, 3, 1, 1)
    return mask.to(device)


def affine_translate_rotate(
    img: torch.Tensor,
    dx: float,
    dy: float,
    angle_deg: float,
    fill: float = 0.0,
) -> torch.Tensor:
    """Rotate ``img`` ``(N, C, H, W)`` about its center, then shift it by ``(dx, dy)`` pixels;
    uncovered pixels are set to ``fill``.
    """
    n, c, h, w = img.shape
    device, dtype = img.device, img.dtype
    angle = angle_deg * np.pi / 180.0
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # The grid inverts x' = cos*(x-cx) - sin*(y-cy) + cx + dx,
    #                 y' = sin*(x-cx) + cos*(y-cy) + cy + dy.
    a, b = cos_a, sin_a
    A00, A01, A10, A11 = a, b, -b, a
    tx = -a * (cx + dx) - b * (cy + dy) + cx
    ty = b * (cx + dx) - a * (cy + dy) + cy
    s_x, s_y = float(w - 1), float(h - 1)
    theta = torch.zeros(n, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = A00
    theta[:, 0, 1] = A01 * (s_y / s_x)
    theta[:, 0, 2] = A00 + A01 * (s_y / s_x) + 2.0 * tx / s_x - 1.0
    theta[:, 1, 0] = A10 * (s_x / s_y)
    theta[:, 1, 1] = A11
    theta[:, 1, 2] = A10 * (s_x / s_y) + A11 + 2.0 * ty / s_y - 1.0
    grid = F.affine_grid(theta, size=img.shape, align_corners=True)
    if fill == 0.0:
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    ones = torch.ones(n, 1, h, w, device=device, dtype=dtype)
    coverage = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return torch.where(coverage > 0.5, sampled, torch.full_like(img, fill))


def reverse_affine_translate_rotate(
    img: torch.Tensor,
    dx: float,
    dy: float,
    angle_deg: float,
) -> torch.Tensor:
    """Approximate inverse of :func:`affine_translate_rotate`: negate shift and angle, as the
    reference code does (exact when either is zero).
    """
    return affine_translate_rotate(img, -dx, -dy, -angle_deg, fill=0.0)


def load_palette(num_colors: int = 30) -> torch.Tensor:
    """Printable RGB palette ``(num_colors, 3)`` in ``[0, 1]``, quantized from the photo of
    the printed palette with 1.5% borders trimmed.
    """
    npy = asset_path("printed-palette-30.npy")
    if num_colors == 30 and os.path.isfile(npy):
        palette = np.load(npy).astype(np.float32) / 255.0
        return torch.from_numpy(palette)
    pal_img = Image.open(asset_path("printed-palette.png")).convert("RGB")
    arr = np.array(pal_img)
    h, w, _ = arr.shape
    ch, cw = round(0.015 * h), round(0.015 * w)
    crop = arr[ch:h - ch, cw:w - cw]
    quantized = Image.fromarray(crop).quantize(colors=num_colors)
    palette = np.array(quantized.getpalette()[: num_colors * 3]).reshape(-1, 3)
    palette = palette[np.lexsort(palette.T[::-1])]
    return torch.from_numpy(palette.astype(np.float32) / 255.0)
