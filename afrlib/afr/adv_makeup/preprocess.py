"""106-point orbital geometry, eye compositing and landmark detection for Adv-Makeup.

Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (commit 359af7e; academic research only, see LICENSE)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from .losses import poisson_blend


EYE_INPUT_SIZE = (160, 420)  # (height, width)
REFERENCE_FACE_SIZE = 600
# Face box used when the detector misses on an (already aligned) LADN600 crop:
# the canvas inset by this fraction on each side.
_DEFAULT_FACE_INSET = 0.12


def default_aligned_bbox(
    height: int,
    width: int,
    inset: float = _DEFAULT_FACE_INSET,
) -> np.ndarray:
    """Canvas box ``[x1, y1, x2, y2]`` inset by ``inset`` on each side."""
    return np.array(
        [
            width * inset,
            height * inset,
            width * (1.0 - inset),
            height * (1.0 - inset),
        ],
        dtype=np.float32,
    )


def largest_detection_bbox(detection: torch.Tensor | None) -> np.ndarray | None:
    """Box with the largest area times face probability, or ``None`` if nothing was detected."""
    if detection is None or detection.shape[0] == 0:
        return None
    areas = (
        (detection[:, 2] - detection[:, 0])
        * (detection[:, 3] - detection[:, 1])
        * detection[:, 5]
    )
    return (
        detection[areas.argmax(), :4]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
EYE_BOX_INDICES = (9, 10, 11, 19, 84, 29, 79, 28, 24, 73, 70, 75, 74, 13, 15, 14, 22)
LEFT_EYE_INDICES = (94, 1, 34, 53, 59, 67, 3, 12)
RIGHT_EYE_INDICES = (27, 104, 41, 85, 20, 47, 43, 51)
LEFT_ORBIT_INDICES = (19, 84, 29, 79, 28, 35, 59, 67, 3, 12, 94)
RIGHT_ORBIT_INDICES = (91, 24, 73, 70, 75, 74, 20, 47, 43, 51, 27)
REQUIRED_LANDMARK_INDICES = tuple(
    sorted(
        set(
            EYE_BOX_INDICES
            + LEFT_EYE_INDICES
            + RIGHT_EYE_INDICES
            + LEFT_ORBIT_INDICES
            + RIGHT_ORBIT_INDICES
        )
    )
)

# Face++ index -> InsightFace 2d106det index for the required points above,
# derived from the eight aligned images shipped upstream (all eight agree).
FACEPP_TO_INSIGHTFACE = {
    1: 41,
    3: 33,
    9: 10,
    10: 9,
    11: 1,
    12: 36,
    13: 17,
    14: 26,
    15: 25,
    19: 43,
    20: 93,
    22: 74,
    24: 102,
    27: 89,
    28: 50,
    29: 49,
    34: 40,
    35: 75,
    41: 94,
    43: 87,
    47: 91,
    51: 90,
    53: 42,
    59: 39,
    67: 37,
    70: 104,
    73: 103,
    74: 101,
    75: 105,
    79: 51,
    84: 48,
    85: 96,
    91: 81,
    94: 35,
    104: 95,
}


@dataclass(frozen=True)
class OrbitalGeometry:
    """One source image's eye box and hard orbital mask."""

    box: tuple[int, int, int, int]  # left, top, right, bottom
    mask: torch.Tensor  # [1, H, W], binary float32

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]


@dataclass
class CompositeBatch:
    """Composited full images plus the per-image crops and geometry behind them."""

    full_images: torch.Tensor
    original_crops: list[torch.Tensor]
    generated_crops: list[torch.Tensor]
    composite_crops: list[torch.Tensor]
    geometries: list[OrbitalGeometry]


def validate_landmarks(landmarks: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not isinstance(landmarks, torch.Tensor):
        raise TypeError("landmarks must be a torch.Tensor")
    if landmarks.ndim == 2:
        landmarks = landmarks.unsqueeze(0)
    if tuple(landmarks.shape[1:]) != (106, 2):
        raise ValueError("landmarks must have shape [B,106,2]")
    if landmarks.shape[0] != batch_size:
        raise ValueError(
            f"Expected {batch_size} landmark sets, got {landmarks.shape[0]}"
        )
    required = landmarks[:, REQUIRED_LANDMARK_INDICES]
    if not torch.isfinite(required).all():
        raise ValueError("Required eye landmarks contain non-finite values")
    return landmarks.to(torch.float32)


def build_orbital_geometry(
    landmarks: torch.Tensor | np.ndarray,
    image_size: tuple[int, int],
) -> OrbitalGeometry:
    """Create the released end-exclusive eye box and binary orbital mask."""

    points = np.asarray(
        landmarks.detach().cpu() if isinstance(landmarks, torch.Tensor) else landmarks
    )
    if points.shape != (106, 2):
        raise ValueError(f"Expected 106x2 landmarks, got {points.shape}")
    points = points.astype(np.int64)
    height, width = image_size
    selected = points[list(EYE_BOX_INDICES)]
    left = int(selected[:, 0].min())
    top = int(selected[:, 1].min())
    right = int(selected[:, 0].max())
    bottom = int(selected[:, 1].max())
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"Orbital crop {(left, top, right, bottom)} leaves image {(height, width)}"
        )
    if right <= left or bottom <= top:
        raise ValueError("Degenerate orbital crop")

    lower_margin = int(round(40 * height / REFERENCE_FACE_SIZE))

    def relative(indices: tuple[int, ...]) -> list[tuple[int, int]]:
        return [
            (int(points[index, 0] - left), int(points[index, 1] - top))
            for index in indices
        ]

    def orbit(indices: tuple[int, ...]) -> list[tuple[int, int]]:
        values = []
        for position, index in enumerate(indices):
            y_offset = lower_margin if position >= 6 else 0
            values.append(
                (
                    int(points[index, 0] - left),
                    int(points[index, 1] - top + y_offset),
                )
            )
        return values

    mask_image = Image.new("L", (right - left, bottom - top), 0)
    drawer = ImageDraw.Draw(mask_image)
    drawer.polygon(orbit(LEFT_ORBIT_INDICES), outline=1, fill=1)
    drawer.polygon(orbit(RIGHT_ORBIT_INDICES), outline=1, fill=1)
    drawer.polygon(relative(LEFT_EYE_INDICES), outline=0, fill=0)
    drawer.polygon(relative(RIGHT_EYE_INDICES), outline=0, fill=0)
    mask_array = np.asarray(mask_image, dtype=np.uint8).copy()
    if mask_array.max(initial=0) != 1 or mask_array.sum() == 0:
        raise ValueError("Orbital mask is empty or non-binary")
    return OrbitalGeometry(
        box=(left, top, right, bottom),
        mask=torch.from_numpy(mask_array).unsqueeze(0).to(torch.float32),
    )


def eye_crop_to_internal(
    image_rgb255: torch.Tensor, geometry: OrbitalGeometry
) -> torch.Tensor:
    """Crop the eye box, resize it to 160x420 with PIL's default filter, map to ``[-1, 1]``."""

    if image_rgb255.ndim != 3 or image_rgb255.shape[0] != 3:
        raise ValueError("image_rgb255 must be CHW RGB")
    left, top, right, bottom = geometry.box
    crop = (
        image_rgb255[:, top:bottom, left:right]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    resized = Image.fromarray(crop).resize(
        (EYE_INPUT_SIZE[1], EYE_INPUT_SIZE[0]), resample=Image.Resampling.BICUBIC
    )
    array = np.asarray(resized, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32)
    return tensor.div(127.5).sub(1.0)


def composite_generated_eye(
    full_internal: torch.Tensor,
    generated_eyes: torch.Tensor,
    landmarks: torch.Tensor,
) -> CompositeBatch:
    """Resize generated eye strips to each eye box and paste them through the orbital masks."""

    if full_internal.ndim != 4 or full_internal.shape[1] != 3:
        raise ValueError("full_internal must have shape [B,3,H,W]")
    if tuple(generated_eyes.shape[1:]) != (3, *EYE_INPUT_SIZE):
        raise ValueError("generated_eyes must have shape [B,3,160,420]")
    if generated_eyes.shape[0] != full_internal.shape[0]:
        raise ValueError("Generated and full-image batch sizes differ")
    landmarks = validate_landmarks(landmarks, full_internal.shape[0])

    pasted_images = []
    original_crops: list[torch.Tensor] = []
    generated_crops: list[torch.Tensor] = []
    composite_crops: list[torch.Tensor] = []
    geometries: list[OrbitalGeometry] = []
    image_size = tuple(map(int, full_internal.shape[-2:]))
    for index in range(full_internal.shape[0]):
        geometry = build_orbital_geometry(landmarks[index], image_size)
        left, top, right, bottom = geometry.box
        original = full_internal[
            index : index + 1, :, top:bottom, left:right
        ]
        generated = F.interpolate(
            generated_eyes[index : index + 1],
            size=(geometry.height, geometry.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = geometry.mask.to(
            device=full_internal.device, dtype=full_internal.dtype
        ).unsqueeze(0)
        composite = generated * mask + original * (1.0 - mask)
        pasted = full_internal[index : index + 1].clone()
        pasted[:, :, top:bottom, left:right] = composite

        pasted_images.append(pasted)
        original_crops.append(original)
        generated_crops.append(generated)
        composite_crops.append(composite)
        geometries.append(geometry)

    return CompositeBatch(
        full_images=torch.cat(pasted_images, dim=0),
        original_crops=original_crops,
        generated_crops=generated_crops,
        composite_crops=composite_crops,
        geometries=geometries,
    )


def poisson_restore_batch(
    source_rgb255: torch.Tensor,
    original_eyes_internal: torch.Tensor,
    generated_crops_internal: list[torch.Tensor],
    geometries: list[OrbitalGeometry],
) -> torch.Tensor:
    """Apply the upstream Poisson postprocess to each image's eye box, in memory."""

    if (
        source_rgb255.shape[0] != len(geometries)
        or original_eyes_internal.shape[0] != len(geometries)
        or len(geometries) != len(generated_crops_internal)
    ):
        raise ValueError("Poisson batch metadata does not align")
    outputs = []
    for index, (generated, geometry) in enumerate(
        zip(generated_crops_internal, geometries)
    ):
        source = (
            source_rgb255[index]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
            .copy()
        )
        left, top, right, bottom = geometry.box
        # As upstream, the Poisson target is the 160x420 eye crop resized back
        # to the eye box, not the original pixels of the 600px source.
        original_native = F.interpolate(
            original_eyes_internal[index : index + 1],
            size=(geometry.height, geometry.width),
            mode="bilinear",
            align_corners=False,
        )
        original_crop = (
            original_native[0]
            .detach()
            .cpu()
            .add(1.0)
            .mul(127.5)
            .permute(1, 2, 0)
            .add(0.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        generated_crop = (
            generated[0]
            .detach()
            .cpu()
            .add(1.0)
            .mul(127.5)
            .permute(1, 2, 0)
            .add(0.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        blended = poisson_blend(
            generated_crop,
            original_crop,
            geometry.mask[0].numpy(),
        )
        source[top:bottom, left:right] = blended
        outputs.append(
            torch.from_numpy(source.copy()).permute(2, 0, 1).to(torch.float32)
        )
    return torch.stack(outputs, dim=0)


class Landmark106Detector:
    """``[B, 106, 2]`` Face++-ordered landmarks of ``[0, 255]`` RGB images, from an FRBench face
    box and InsightFace ``2d106det``; only the points in :data:`FACEPP_TO_INSIGHTFACE` are set.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
    ) -> None:
        from frbench import FaceDetector
        from insightface.model_zoo import get_model

        self.device = torch.device(device)
        self.face_detector = (
            FaceDetector(eager_load=True).to(self.device).eval()
        )
        if self.device.type == "cuda":
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"device_id": str(device_index)},
                ),
                "CPUExecutionProvider",
            ]
            context_id = device_index
        else:
            providers = ["CPUExecutionProvider"]
            context_id = -1
        self.landmark_model = get_model(
            str(checkpoint), providers=providers
        )
        self.landmark_model.prepare(ctx_id=context_id)

    def __call__(self, images_rgb255: torch.Tensor) -> torch.Tensor:
        from insightface.app.common import Face

        detections = self.face_detector.detect(images_rgb255).detections
        outputs = []
        for index, (image, detection) in enumerate(
            zip(images_rgb255, detections)
        ):
            bbox = largest_detection_bbox(detection)
            if bbox is None:
                height, width = int(image.shape[-2]), int(image.shape[-1])
                bbox = default_aligned_bbox(height, width)
            rgb = (
                image.detach()
                .cpu()
                .permute(1, 2, 0)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )
            bgr = rgb[:, :, ::-1].copy()
            try:
                predicted = np.asarray(
                    self.landmark_model.get(bgr, Face(bbox=bbox)),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"106-landmark model failed on Adv-Makeup source image {index}"
                ) from exc
            if predicted.shape != (106, 2):
                raise RuntimeError(
                    f"106-landmark model returned {predicted.shape}"
                )
            reordered = np.zeros((106, 2), dtype=np.float32)
            for facepp_index, insightface_index in FACEPP_TO_INSIGHTFACE.items():
                reordered[facepp_index] = predicted[insightface_index]
            outputs.append(torch.from_numpy(reordered))
        return torch.stack(outputs, dim=0)
