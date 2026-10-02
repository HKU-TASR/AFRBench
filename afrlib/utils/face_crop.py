"""Crop faces into a method's input format, and paste results back."""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize

from afrlib.contracts import AlignType, CropType, ImageSpec

#: Supersample cap for antialiasing, mirroring frbench.align.
_MAX_SUPERSAMPLE = 4

Box = Tuple[float, float, float, float]
Window = Tuple[int, int, int]


# Frame geometry: the square window an original image is cut from


def face_square(box: Sequence[float]) -> Tuple[float, float, float]:
    """The square ``(x, y, s)`` centered on a tight detector box, side max(w, h)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    side = max(x2 - x1, y2 - y1)
    return (x1 + x2 - side) / 2.0, (y1 + y2 - side) / 2.0, side


@dataclasses.dataclass(frozen=True)
class FramePlan:
    """Integer square window around a face, with the padding it needs.

    Attributes:
        window: ``(x, y, side)`` in photo pixels; ``x`` or ``y`` is negative when the
            window starts off the photo.
        photo_size: The photo's ``(W, H)``.
        pad: Border-repeat padding ``(left, top, right, bottom)``.
        valid_rect: The ``(x, y, w, h)`` rectangle of real photo pixels, in window pixels.
        padding_fraction: The share of the window that is padding.
        original_ratio: The ratio of window side to face side the window was planned with.
        face_side: The face square's side in photo pixels.
    """

    window: Window
    photo_size: Tuple[int, int]  # (W, H)
    pad: Tuple[int, int, int, int]  # left, top, right, bottom
    valid_rect: Tuple[int, int, int, int]
    padding_fraction: float
    original_ratio: float
    face_side: float

    def to_json(self) -> dict:
        return {
            "original_box": [int(v) for v in self.window],
            "padding": [int(v) for v in self.pad],
            "valid_rect": [int(v) for v in self.valid_rect],
            "padding_fraction": float(self.padding_fraction),
            "original_ratio": float(self.original_ratio),
            "face_side": float(self.face_side),
        }


def _place_axis(face_start: float, face_len: float, win: int, photo: int) -> int:
    """Integer origin that contains the face and minimizes out-of-bounds extent."""
    lo = int(math.ceil(face_start + face_len - win))
    hi = int(math.floor(face_start))
    if lo > hi:
        mid = int(round(face_start + face_len / 2.0 - win / 2.0))
        lo = hi = mid
    center = face_start + face_len / 2.0 - win / 2.0
    if win <= photo:
        pref_lo, pref_hi = 0, photo - win
    else:
        pref_lo, pref_hi = photo - win, 0
    inter_lo, inter_hi = max(lo, pref_lo), min(hi, pref_hi)
    if inter_lo <= inter_hi:
        return int(min(max(round(center), inter_lo), inter_hi))
    if hi < pref_lo:
        return hi
    if lo > pref_hi:
        return lo
    return int(min(max(round((pref_lo + pref_hi) / 2.0), lo), hi))


def plan_frame(
    face: Tuple[float, float, float],
    image_size: Tuple[int, int],
    ratio: float = 2.0,
) -> FramePlan:
    """Plan the ``ceil(ratio * side)`` square window containing ``face``, centered on it
    where possible and shifted to minimize padding.

    Args:
        face: The face square ``(x, y, side)`` in photo pixels.
        image_size: The photo's ``(W, H)``.
        ratio: Window side over face side.

    Raises:
        ValueError: If the face or the photo has no area.
    """
    face_x, face_y, face_side = (float(v) for v in face)
    width, height = int(image_size[0]), int(image_size[1])
    if face_side <= 0 or width <= 0 or height <= 0:
        raise ValueError("face and photo must have positive size")
    side = max(1, math.ceil(float(ratio) * face_side))
    x = _place_axis(face_x, face_side, side, width)
    y = _place_axis(face_y, face_side, side, height)
    pad_left = max(0, -x)
    pad_top = max(0, -y)
    pad_right = max(0, x + side - width)
    pad_bottom = max(0, y + side - height)
    valid_w = side - pad_left - pad_right
    valid_h = side - pad_top - pad_bottom
    area = side * side
    padding_fraction = 1.0 - (valid_w * valid_h) / float(area) if area else 1.0
    return FramePlan(
        window=(x, y, side),
        photo_size=(width, height),
        pad=(pad_left, pad_top, pad_right, pad_bottom),
        valid_rect=(pad_left, pad_top, valid_w, valid_h),
        padding_fraction=float(padding_fraction),
        original_ratio=float(ratio),
        face_side=face_side,
    )


def plan_from_stored(
    window: Sequence[int],
    photo_size: Tuple[int, int],
    *,
    ratio: float = 2.0,
    face_side: float = 0.0,
    pad: Optional[Sequence[int]] = None,
    valid_rect: Optional[Sequence[int]] = None,
    padding_fraction: Optional[float] = None,
) -> FramePlan:
    """Rebuild a :class:`FramePlan` from stored annotations; ``photo_size`` is ``(W, H)``,
    and ``pad``, ``valid_rect`` and ``padding_fraction`` are recomputed when not given.
    """
    x, y, side = (int(v) for v in window[:3])
    width, height = int(photo_size[0]), int(photo_size[1])
    if pad is None:
        pad = (
            max(0, -x),
            max(0, -y),
            max(0, x + side - width),
            max(0, y + side - height),
        )
    left, top, right, bottom = (int(v) for v in pad[:4])
    if valid_rect is None:
        valid_rect = (left, top, side - left - right, side - top - bottom)
    vx, vy, vw, vh = (int(v) for v in valid_rect[:4])
    if padding_fraction is None:
        area = side * side
        padding_fraction = 1.0 - (vw * vh) / float(area) if area else 1.0
    return FramePlan(
        window=(x, y, side),
        photo_size=(width, height),
        pad=(left, top, right, bottom),
        valid_rect=(vx, vy, vw, vh),
        padding_fraction=float(padding_fraction),
        original_ratio=float(ratio),
        face_side=float(face_side),
    )


def frame_window(
    face: Tuple[float, float, float],
    image_size: Tuple[int, int],
    ratio: float = 2.0,
) -> Window:
    """The ``(x, y, side)`` window of :func:`plan_frame`; may start off-image."""
    return plan_frame(face, image_size, ratio).window


def face_fraction(face_side: float, frame_side: float) -> float:
    """The face's share of the frame area, about ``1 / ratio**2`` for :func:`plan_frame`."""
    return (face_side / frame_side) ** 2


def crop_window(image, window: Window):
    """The ``(x, y, side)`` window of a ``[..., H, W]`` tensor; it must lie inside the
    image (otherwise use :func:`extract_frame`).
    """
    x, y, side = window
    return image[..., y : y + side, x : x + side]


def extract_frame(image, window: Window, photo_size: Optional[Tuple[int, int]] = None):
    """Crop ``window`` from a ``[3, H, W]`` tensor, padding with border repeat.

    Args:
        image: The photo.
        window: ``(x, y, side)`` in photo pixels.
        photo_size: ``(W, H)`` to compute the padding from, instead of the tensor's size.
    """
    x, y, side = (int(v) for v in window[:3])
    height = int(image.shape[-2])
    width = int(image.shape[-1])
    if photo_size is not None:
        width, height = int(photo_size[0]), int(photo_size[1])
    pad_left = max(0, -x)
    pad_top = max(0, -y)
    pad_right = max(0, x + side - width)
    pad_bottom = max(0, y + side - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        x += pad_left
        y += pad_top
    return image[..., y : y + side, x : x + side]


def extract_frame_numpy(image: np.ndarray, window: Window) -> np.ndarray:
    """Crop ``window`` from an ``(H, W[, C])`` array, padding with edge repeat."""
    x, y, side = (int(v) for v in window[:3])
    height, width = int(image.shape[0]), int(image.shape[1])
    pad_left = max(0, -x)
    pad_top = max(0, -y)
    pad_right = max(0, x + side - width)
    pad_bottom = max(0, y + side - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        if image.ndim == 3:
            pad_width = pad_width + ((0, 0),)
        image = np.pad(image, pad_width, mode="edge")
        x += pad_left
        y += pad_top
    return image[y : y + side, x : x + side]


def valid_region(image, valid_rect: Sequence[int]):
    """The real-pixel rectangle ``(x, y, w, h)`` of a ``[3, H, W]`` crop; the whole crop
    when the rectangle is empty.
    """
    x, y, width, height = (int(v) for v in valid_rect[:4])
    if width <= 0 or height <= 0:
        return image
    return image[..., y : y + height, x : x + width]


def ffhq_natural_size(landmarks: Sequence[Sequence[float]]) -> int:
    """The FFHQ parallelogram's pre-resize side from 5-point landmarks."""
    lm = np.asarray(landmarks, dtype=np.float64)
    _affine, quad = ffhq_affine(lm)
    side = float(
        max(np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[1] - quad[0])))
    )
    return max(int(round(side)), 1)


# Crop plans


@dataclasses.dataclass
class CropPlan:
    """How one method input was cropped from a photo, and how to undo it.

    Attributes:
        matrix: The 2x3 affine from photo pixels to natural-crop pixels.
        natural_size: Side of the warped crop, before the final resize.
        native_size: The spec's side, after the final resize.
        photo_size: The photo's ``(H, W)``.
        range_min: Lower end of the spec's value range.
        range_max: Upper end of the spec's value range.
    """

    matrix: torch.Tensor  # (2, 3) photo -> natural crop
    natural_size: int
    native_size: int
    photo_size: Tuple[int, int]  # (H, W)
    range_min: float = 0.0
    range_max: float = 255.0

    @property
    def resize_factor(self) -> float:
        """Native side over natural side; below 1 the final resize downsamples."""
        return self.native_size / self.natural_size

    def to_json(self) -> dict:
        return {
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "natural_size": self.natural_size,
            "native_size": self.native_size,
            "photo_size": list(self.photo_size),
            "resize_factor": self.resize_factor,
        }


# Affine fitting (float64 numpy; warps run in float32 torch)


def _full(matrix: np.ndarray) -> np.ndarray:
    """Embed a 2x3 affine in a 3x3 homogeneous matrix."""
    full = np.eye(3)
    full[:2] = matrix
    return full


def _fit_affine(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Least-squares 2x3 affine mapping ``source`` points to ``destination``."""
    design = np.column_stack((source, np.ones(source.shape[0])))
    coefficients, _res, rank, _sv = np.linalg.lstsq(design, destination, rcond=None)
    if rank != 3:
        raise ValueError("degenerate point set cannot determine an affine map")
    return coefficients.T


def _box_affine(box: Sequence[float]) -> np.ndarray:
    """The photo->unit-square affine of an axis-aligned box."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array(
        [[1.0 / (x2 - x1), 0.0, -x1 / (x2 - x1)],
         [0.0, 1.0 / (y2 - y1), -y1 / (y2 - y1)]]
    )


def ffhq_affine(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The FFHQ parallelogram from 5-point landmarks (StyleGAN aligner formula).

    Returns:
        The 2x3 photo -> unit-square affine, and the quad's four corners
        (NW, SW, SE, NE) in photo coordinates.
    """
    eye_left, eye_right, _nose, mouth_left, mouth_right = landmarks
    eye_avg = (eye_left + eye_right) * 0.5
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - eye_to_mouth[::-1] * [-1.0, 1.0]
    x = x / np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = x[::-1] * [-1.0, 1.0]
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([center - x - y, center - x + y, center + x + y, center + x - y])
    unit = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    return _fit_affine(quad, unit), quad


# Warping


def _theta(
    matrix: torch.Tensor,
    out_size: Tuple[int, int],
    in_size: Tuple[int, int],
    *,
    invert: bool,
) -> torch.Tensor:
    """``grid_sample`` theta of photo->crop ``matrix``; ``invert=True`` samples the crop."""
    oh, ow = out_size
    ih, iw = in_size
    full = torch.eye(3, dtype=torch.float64)
    full[:2] = matrix.to(torch.float64)
    if invert:
        full = torch.linalg.inv(full)
    denorm_out = torch.tensor(
        [[ow / 2, 0, ow / 2 - 0.5], [0, oh / 2, oh / 2 - 0.5], [0, 0, 1]],
        dtype=torch.float64,
    )
    norm_in = torch.tensor(
        [[2 / iw, 0, 1 / iw - 1], [0, 2 / ih, 1 / ih - 1], [0, 0, 1]],
        dtype=torch.float64,
    )
    return (norm_in @ full @ denorm_out)[:2].to(torch.float32)


def _warp(
    img: torch.Tensor,
    theta: torch.Tensor,
    out_size: Tuple[int, int],
    padding: str,
) -> torch.Tensor:
    """Sample ``img`` (``[3,H,W]``) at ``theta`` into ``out_size``."""
    grid = F.affine_grid(
        theta.unsqueeze(0).to(img.device), (1, 3, *out_size), align_corners=False
    )
    return F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode=padding,
        align_corners=False,
    )[0]


def _warp_antialiased(
    img: torch.Tensor,
    theta: torch.Tensor,
    out_size: Tuple[int, int],
    scale: float,
    padding: str,
) -> torch.Tensor:
    """Warp, supersampling and box-averaging when the transform downsamples."""
    factor = int(min(_MAX_SUPERSAMPLE, max(1, math.ceil(1.0 / max(scale, 1e-6)))))
    if factor == 1:
        return _warp(img, theta, out_size, padding)
    hi = (out_size[0] * factor, out_size[1] * factor)
    return F.avg_pool2d(_warp(img, theta, hi, padding).unsqueeze(0), factor)[0]


def _resize_image(crop: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bicubic resize of a ``[3,H,W]`` image/face crop."""
    if crop.shape[-1] == size and crop.shape[-2] == size:
        return crop
    return resize(
        crop.unsqueeze(0), [size, size],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    )[0]


def _resize_delta(crop: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bilinear resize of an additive crop-space delta."""
    if crop.shape[-1] == size and crop.shape[-2] == size:
        return crop
    return resize(
        crop.unsqueeze(0), [size, size],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )[0]


def _resize(crop: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased bicubic resize of a ``[3,H,W]`` crop to ``size`` x ``size``."""
    return _resize_image(crop, size)


def _scale_of(matrix: torch.Tensor) -> float:
    """The transform's pixel scale (crop pixels per photo pixel)."""
    linear = matrix.to(torch.float64)[:, :2]
    return float(torch.sqrt(torch.abs(torch.det(linear))))


# Public API


def plan_for(
    spec: ImageSpec,
    landmarks: Sequence[Sequence[float]],
    face_box: Sequence[float],
    photo_size: Tuple[int, int],
) -> Optional[CropPlan]:
    """Plan the crop of one image into ``spec``; ``None`` for an Original spec.

    Args:
        spec: The format to crop into, typically the method's ``attack_spec()``.
        landmarks: 5-point landmarks (left eye, right eye, nose, left and right
            mouth corners) in photo pixels.
        face_box: Detector box ``(x1, y1, x2, y2)`` in photo pixels.
        photo_size: The photo's ``(H, W)``.

    Raises:
        ValueError: If ``spec`` crops but has no ``size``, or its
            ``(crop_type, align_type)`` is not supported.
    """
    if spec.crop_type is CropType.Original:
        return None
    native = spec.size
    if native is None:
        raise ValueError(f"Spec {spec} declares a crop but no size")

    lm = np.asarray(landmarks, dtype=np.float64)
    if spec.crop_type is CropType.FaceBox and spec.align_type is AlignType.ArcFace5:
        from frbench import ARCFACE_112_TEMPLATE, estimate_similarity_transform

        template = ARCFACE_112_TEMPLATE.to(torch.float64) * (native / 112.0)
        matrix = estimate_similarity_transform(
            torch.from_numpy(lm[None]).to(torch.float64), template
        )[0]
        # The similarity already carries the resize: natural == native.
        natural = native
    elif spec.crop_type in (CropType.FFHQ, CropType.LADN600):
        unit_affine, quad = ffhq_affine(lm)
        if spec.align_type in (AlignType.FFHQQuad, AlignType.LADN600):
            side = float(
                max(np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[1] - quad[0])))
            )
        else:  # FFHQ framing without the warp: the quad's bounding square
            x1, y1 = quad.min(axis=0)
            x2, y2 = quad.max(axis=0)
            side = float(max(x2 - x1, y2 - y1))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            unit_affine = _box_affine(
                (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
            )
        natural = max(int(round(side)), 1)
        matrix = torch.from_numpy(
            np.array([[natural, 0.0], [0.0, natural]]) @ unit_affine
        )
    elif spec.crop_type is CropType.FaceBox:  # NoAlign
        x1, y1, x2, y2 = (float(v) for v in face_box)
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        natural = max(int(round(side)), 1)
        matrix = torch.from_numpy(
            np.array([[natural, 0.0], [0.0, natural]])
            @ _box_affine((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        )
    else:
        raise ValueError(f"Unsupported spec combination: {spec}")

    return CropPlan(
        matrix=matrix.to(torch.float32),
        natural_size=natural,
        native_size=native,
        photo_size=photo_size,
        range_min=spec.range_min,
        range_max=spec.range_max,
    )


def apply_plan(photo: torch.Tensor, plan: CropPlan) -> torch.Tensor:
    """Crop a ``[3, H, W]`` photo as planned: warp to natural size, then resize, clamped to
    the spec's value range.
    """
    crop = natural_crop(photo, plan)
    return _resize(crop, plan.native_size).clamp(plan.range_min, plan.range_max)


def natural_crop(photo: torch.Tensor, plan: CropPlan) -> torch.Tensor:
    """The planned crop at its natural size, before :func:`apply_plan`'s final resize."""
    natural = (plan.natural_size, plan.natural_size)
    theta = _theta(plan.matrix, natural, plan.photo_size, invert=True)
    return _warp_antialiased(
        photo, theta, natural, scale=_scale_of(plan.matrix), padding="border"
    )


def paste_back(
    photo: torch.Tensor, protected: torch.Tensor, plan: CropPlan
) -> torch.Tensor:
    """Replace the photo's crop region with the protected crop."""
    crop = _resize(protected, plan.natural_size)
    natural = (plan.natural_size, plan.natural_size)
    theta = _theta(plan.matrix, plan.photo_size, natural, invert=False)
    warped = _warp(crop, theta, plan.photo_size, padding="zeros")
    ones = torch.ones(3, *natural, device=crop.device, dtype=crop.dtype)
    mask = _warp(ones, theta, plan.photo_size, padding="zeros").clamp(0, 1)
    # The zero-padded warp is already premultiplied by its coverage ``mask``.
    return (photo * (1.0 - mask) + warped).clamp(plan.range_min, plan.range_max)


def paste_delta(
    photo: torch.Tensor, delta: torch.Tensor, plan: Optional[CropPlan]
) -> torch.Tensor:
    """Warp a crop-space delta (``adv - clean``) onto the photo; pixels outside the crop are
    unchanged, and with ``plan=None`` the delta is added as is.
    """
    if plan is None:
        return (photo + delta).clamp(0.0, 255.0)
    natural = (plan.natural_size, plan.natural_size)
    theta = _theta(plan.matrix, plan.photo_size, natural, invert=False)
    warped = _warp(_resize_delta(delta, plan.natural_size), theta, plan.photo_size, "zeros")
    # The zero-padded warp is already weighted by its coverage, so it is added as is.
    return (photo + warped).clamp(plan.range_min, plan.range_max)


def prepare_input(
    photo: torch.Tensor,
    spec: ImageSpec,
    landmarks: Sequence[Sequence[float]],
    face_box: Sequence[float],
) -> Tuple[torch.Tensor, Optional[CropPlan]]:
    """One image in the spec's format and the plan that produced it; for an Original spec,
    the photo itself and ``None``.
    """
    plan = plan_for(
        spec, landmarks, face_box, (int(photo.shape[-2]), int(photo.shape[-1]))
    )
    if plan is None:
        return photo, None
    return apply_plan(photo, plan), plan


# Batch helpers for protect()


def prepare_inputs(
    photos: Sequence[torch.Tensor],
    geometry: Sequence[dict],
    spec: ImageSpec,
) -> Tuple[List[torch.Tensor], List[Optional[CropPlan]]]:
    """Crop each original to ``spec``; return the crops and their plans.

    Args:
        photos: Original images, each ``[3, Hi, Wi]``.
        geometry: ``{"landmarks", "face_box"}`` per photo, in that photo's
            pixel coordinates (as in ``EvalData.eval_geometry``).
        spec: The format to crop into.

    Returns:
        The crops, and the plans :func:`paste_deltas` or :func:`paste_faces`
        need to put results back.

    Raises:
        ValueError: If ``geometry`` does not have one entry per photo.
    """
    if geometry is None or len(geometry) != len(photos):
        raise ValueError(
            "prepare_inputs needs one geometry entry per image; got "
            f"{0 if geometry is None else len(geometry)} for {len(photos)} images."
        )
    crops, plans = [], []
    for photo, geom in zip(photos, geometry):
        crop, plan = prepare_input(photo, spec, geom["landmarks"], geom["face_box"])
        crops.append(crop)
        plans.append(plan)
    return crops, plans


def paste_deltas(
    photos: Sequence[torch.Tensor],
    deltas: Sequence[torch.Tensor],
    plans: Sequence[Optional[CropPlan]],
) -> List[torch.Tensor]:
    """:func:`paste_delta` over a batch of photos, deltas and plans."""
    return [
        paste_delta(photo, delta.to(photo.device), plan)
        for photo, delta, plan in zip(photos, deltas, plans)
    ]


def paste_faces(
    photos: Sequence[torch.Tensor],
    faces: Sequence[torch.Tensor],
    plans: Sequence[Optional[CropPlan]],
) -> List[torch.Tensor]:
    """:func:`paste_back` over a batch, for a method that regenerates the face; with plan
    ``None`` the face replaces the whole photo.
    """
    return [
        face.to(photo.device) if plan is None
        else paste_back(photo, face.to(photo.device), plan)
        for photo, face, plan in zip(photos, faces, plans)
    ]
