# Adapted from AMT-GAN (https://github.com/CGCL-codes/AMT-GAN),
# Copyright (c) 2022 CGCL-codes, MIT License. See LICENSE in this package.

"""Released AMT-GAN face preparation and Laplacian postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Union

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .face_parser import FaceParser


Device = Union[str, torch.device]
Pathish = Union[str, PathLike[str]]

_RESAMPLING = getattr(Image, "Resampling", Image)
_LANCZOS = _RESAMPLING.LANCZOS
_BILINEAR = _RESAMPLING.BILINEAR
_NEAREST = _RESAMPLING.NEAREST
# The input is already an FFHQ face crop. If HOG misses on that 256px crop,
# the crop itself is the face box (a 12% inset leaves a little context).
_DEFAULT_FACE_INSET = 0.12


def default_face_rect(
    width: int,
    height: int,
    inset: float = _DEFAULT_FACE_INSET,
) -> dlib.rectangle:
    """A full-crop face box for an already-aligned attack-domain image."""
    inset_x = max(int(round(width * inset)), 0)
    inset_y = max(int(round(height * inset)), 0)
    left = min(inset_x, max(width - 2, 0))
    top = min(inset_y, max(height - 2, 0))
    right = max(width - inset_x, left + 1)
    bottom = max(height - inset_y, top + 1)
    return dlib.rectangle(left, top, right, bottom)


@dataclass(frozen=True)
class PreparedFace:
    """AMT-GAN generator inputs for one face, in the method's 256px domain."""

    image: torch.Tensor  # [1, 3, 256, 256], normalized to [-1, 1]
    masks: torch.Tensor  # [lip/face/eye, 1, 256, 256]
    distances: torch.Tensor  # [lip/face/eye, 136(y then x), 64, 64]


def _resize_by_max(
    image: np.ndarray,
    max_side: int = 512,
    force: bool = False,
) -> np.ndarray:
    """Match the released detector resize, including its rounding."""

    height, width = image.shape[:2]
    if max(height, width) < max_side and not force:
        return image
    ratio = max(height, width) / max_side
    resized_width = int(width / ratio + 0.5)
    resized_height = int(height / ratio + 0.5)
    return cv2.resize(image, (resized_width, resized_height))


def _resize_short_edge(
    image: Image.Image,
    size: int,
    resample: int,
) -> Image.Image:
    """PIL equivalent of the released ``transforms.Resize(256)``."""

    width, height = image.size
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    if width < height:
        resized_width = size
        resized_height = int(size * height / width)
    else:
        resized_height = size
        resized_width = int(size * width / height)
    return image.resize((resized_width, resized_height), resample)


def _rgb_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, copy=True)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("AMT-GAN expects an RGB image")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return tensor.float().div(255.0).sub(0.5).div(0.5)


def _raw_mask_tensor(mask: Image.Image) -> torch.Tensor:
    """Match the release's mask ToTensor: CHW layout without /255 scaling."""

    if mask.mode == "I":
        array = np.array(mask, dtype=np.int32, copy=True)
    elif mask.mode == "I;16":
        array = np.array(mask, dtype=np.int16, copy=True)
    else:
        array = np.array(mask, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class AMTGANPreprocessor:
    """Prepare inference and training inputs with released AMT-GAN semantics."""

    image_size = 256
    diff_size = (64, 64)
    eye_margin = 16
    lip_classes = (7, 9)
    face_classes = (1, 6)

    def __init__(
        self,
        parser_checkpoint: Pathish,
        landmark_predictor: Pathish,
        device: Device,
    ) -> None:
        self.device = torch.device(device)
        # Assets are loaded per instance, not at module import.
        self.face_parser = FaceParser(parser_checkpoint, self.device)
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(landmark_predictor))

        xs, ys = np.meshgrid(
            np.linspace(0, self.image_size - 1, self.image_size),
            np.linspace(0, self.image_size - 1, self.image_size),
        )
        xs = xs[None].repeat(68, axis=0)
        ys = ys[None].repeat(68, axis=0)
        # Channels are all 68 y maps followed by all 68 x maps.
        self._coordinate_grid = torch.Tensor(
            np.concatenate((ys, xs), axis=0)
        )

    def _detect(self, image: Image.Image) -> dlib.rectangles:
        array = np.asarray(image)
        height, width = array.shape[:2]
        small = _resize_by_max(array, 361)
        actual_height, actual_width = small.shape[:2]
        faces_on_small = self._detector(small, 1)
        faces = dlib.rectangles()
        for face in faces_on_small:
            faces.append(
                dlib.rectangle(
                    int(face.left() / actual_width * width + 0.5),
                    int(face.top() / actual_height * height + 0.5),
                    int(face.right() / actual_width * width + 0.5),
                    int(face.bottom() / actual_height * height + 0.5),
                )
            )
        return faces

    def _landmarks(
        self,
        image: Image.Image,
        face: dlib.rectangle,
    ) -> np.ndarray:
        shape = self._predictor(np.asarray(image), face).parts()
        # The release stores every point as (y, x), not (x, y).
        return np.array([[point.y, point.x] for point in shape])

    def _copy_eye_region(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        landmarks: np.ndarray,
    ) -> None:
        rectangle = (
            int(min(landmarks[:, 1])) - self.eye_margin,
            int(min(landmarks[:, 0])) - self.eye_margin,
            int(max(landmarks[:, 1])) + self.eye_margin + 1,
            int(max(landmarks[:, 0])) + self.eye_margin + 1,
        )
        # Negative or out-of-range slice bounds are not clamped, as in the release.
        target[
            :,
            :,
            rectangle[1] : rectangle[3],
            rectangle[0] : rectangle[2],
        ] = source[
            :,
            :,
            rectangle[1] : rectangle[3],
            rectangle[0] : rectangle[2],
        ]
        source[
            :,
            :,
            rectangle[1] : rectangle[3],
            rectangle[0] : rectangle[2],
        ] = 0

    def _process_regions(
        self,
        mask: torch.Tensor,
        landmarks: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if landmarks.shape != (68, 2):
            raise ValueError(
                f"Expected 68 (y, x) landmarks, got {landmarks.shape}"
            )

        mask = mask.to(device=device).float()
        landmark_vector = landmarks.transpose((1, 0)).reshape(-1, 1, 1)
        differences = (
            self._coordinate_grid.to(device).double()
            - torch.tensor(landmark_vector, device=device)
        ).unsqueeze(0).float()

        mask_lip = (mask == self.lip_classes[0]).float()
        mask_lip += (mask == self.lip_classes[1]).float()
        mask_face = (mask == self.face_classes[0]).float()
        mask_face += (mask == self.face_classes[1]).float()

        mask_eyes = torch.zeros_like(mask, device=device)
        self._copy_eye_region(mask_eyes, mask_face, landmarks[42:48])
        self._copy_eye_region(mask_eyes, mask_face, landmarks[36:42])
        masks = torch.cat((mask_lip, mask_face, mask_eyes), dim=0)

        mask_weights = F.interpolate(
            masks,
            size=self.diff_size,
        ).repeat(1, differences.shape[1], 1, 1)
        distances = F.interpolate(
            differences,
            size=self.diff_size,
        ).repeat(3, 1, 1, 1)
        distances *= mask_weights
        norms = torch.norm(distances, dim=1, keepdim=True).repeat(
            1,
            distances.shape[1],
            1,
            1,
        )
        norms = torch.where(
            norms == 0,
            torch.tensor(1e10, device=device),
            norms,
        )
        distances /= norms
        return masks, distances

    def prepare_inference(
        self,
        crop: Image.Image,
    ) -> PreparedFace | None:
        """Generator inputs from one FFHQ-framed crop, or ``None`` if the 68-point predictor
        fails; the crop replaces the reference code's detect-and-crop step.
        """

        crop = crop.convert("RGB")
        faces = self._detect(crop)
        face = faces[0] if faces else default_face_rect(crop.width, crop.height)

        crop_array = np.array(crop)
        parser_input = cv2.resize(crop_array, (512, 512))
        parsed = self.face_parser.parse(parser_input)
        parsed = F.interpolate(
            parsed.view(1, 1, 512, 512),
            (self.image_size, self.image_size),
            mode="nearest",
        )
        parsed = parsed.type(torch.uint8).float()

        try:
            landmarks = self._landmarks(crop, face)
        except Exception:
            return None
        if not np.isfinite(landmarks).all():
            return None
        landmarks = landmarks * self.image_size / crop.width
        landmarks = landmarks.round()
        masks, distances = self._process_regions(
            parsed,
            landmarks,
            self.device,
        )

        resized = crop.resize(
            (self.image_size, self.image_size),
            _LANCZOS,
        )
        internal = _rgb_to_normalized_tensor(resized)
        internal = internal.unsqueeze(0).to(self.device)
        return PreparedFace(internal, masks, distances)

    def prepare_training(
        self,
        image: Image.Image,
        segmentation: Image.Image | np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the unbatched tensors emitted by the released DataLoader."""

        image = image.convert("RGB")
        landmarks = np.asarray(landmarks)
        if landmarks.shape != (68, 2):
            raise ValueError(
                f"Expected 68 (y, x) landmarks, got {landmarks.shape}"
            )

        # (y, x) landmarks are divided by (width, height), as in the release.
        scaled_landmarks = (
            landmarks / np.asarray(image.size) * self.image_size
        )
        resized_image = _resize_short_edge(image, self.image_size, _BILINEAR)
        image_tensor = _rgb_to_normalized_tensor(resized_image)

        mask_image = (
            segmentation
            if isinstance(segmentation, Image.Image)
            else Image.fromarray(np.asarray(segmentation))
        )
        mask_image = _resize_short_edge(mask_image, self.image_size, _NEAREST)
        mask_tensor = _raw_mask_tensor(mask_image).unsqueeze(0)
        masks, distances = self._process_regions(
            mask_tensor,
            scaled_landmarks,
            torch.device("cpu"),
        )
        return image_tensor, masks, distances


def _as_hwc_array(value: Image.Image | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.array(value)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
    ):
        array = np.transpose(array, (1, 2, 0))
    return array


def laplacian_postprocess(
    source_crop: Image.Image | np.ndarray | torch.Tensor,
    generated_256: Image.Image | np.ndarray | torch.Tensor,
) -> Image.Image:
    """Restore source high frequencies using released float64/OpenCV math."""

    source = _as_hwc_array(source_crop)
    generated = _as_hwc_array(generated_256)
    height, width = source.shape[:2]
    small_source = cv2.resize(source, (256, 256))
    # float64 matches the release's ``np.float`` alias, which NumPy has removed.
    laplacian_difference = source.astype(np.float64) - cv2.resize(
        small_source,
        (width, height),
    ).astype(np.float64)
    result = (
        cv2.resize(generated, (width, height)) + laplacian_difference
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(result).convert("RGB")


__all__ = [
    "AMTGANPreprocessor",
    "PreparedFace",
    "laplacian_postprocess",
]
