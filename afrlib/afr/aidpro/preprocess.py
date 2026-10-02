"""SCRFD detection and five-point alignment from AIDPro's upstream preprocessing."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage import transform as sk_transform
import torch

from afrlib.utils.assets import get_asset


_POSE_TEMPLATES = np.asarray(
    [
        [
            [51.642, 50.115],
            [57.617, 49.990],
            [35.740, 69.007],
            [51.157, 89.050],
            [57.025, 89.702],
        ],
        [
            [45.031, 50.118],
            [65.568, 50.872],
            [39.677, 68.111],
            [45.177, 86.190],
            [64.246, 86.758],
        ],
        [
            [39.730, 51.138],
            [72.270, 51.138],
            [56.000, 68.493],
            [42.463, 87.010],
            [69.537, 87.010],
        ],
        [
            [46.845, 50.872],
            [67.382, 50.118],
            [72.737, 68.111],
            [48.167, 86.758],
            [67.236, 86.190],
        ],
        [
            [54.796, 49.990],
            [60.771, 50.115],
            [76.673, 69.007],
            [55.388, 89.702],
            [61.257, 89.050],
        ],
    ],
    dtype=np.float32,
)


def estimate_released_transform(
    landmarks: np.ndarray, image_size: int = 224
) -> np.ndarray:
    """Similarity transform (2x3) onto the best-fitting upstream pose template."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2):
        raise ValueError(
            f"Expected five xy landmarks, got {landmarks.shape}"
        )
    templates = _POSE_TEMPLATES * (float(image_size) / 112.0)
    homogeneous = np.concatenate(
        [landmarks, np.ones((5, 1), dtype=np.float32)], axis=1
    )
    best_matrix: Optional[np.ndarray] = None
    best_error = float("inf")
    for template in templates:
        try:
            transform = sk_transform.SimilarityTransform.from_estimate(
                landmarks, template
            )
        except ValueError:
            continue
        matrix = transform.params[:2].astype(np.float32)
        projected = homogeneous @ matrix.T
        error = float(np.linalg.norm(projected - template, axis=1).sum())
        if error < best_error:
            best_error = error
            best_matrix = matrix
    if best_matrix is None:
        raise RuntimeError("Unable to estimate AIDPro alignment transform")
    return best_matrix


class AIDProAligner:
    """SCRFD detector with AIDPro's upstream alignment of the highest-scoring face."""

    def __init__(
        self,
        detector_path: Optional[str | Path] = None,
        *,
        ctx_id: int = -1,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.1,
    ) -> None:
        # Imported lazily so importing AFRLib does not initialize ONNX Runtime.
        from insightface.model_zoo import model_zoo

        path = Path(
            detector_path or get_asset("aidpro/scrfd_10g_bnkps.onnx")
        ).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.detector_path = str(path)
        self.det_size = tuple(int(value) for value in det_size)
        self.det_thresh = float(det_thresh)
        self.ctx_id = int(ctx_id)
        self.detector = model_zoo.get_model(self.detector_path)
        # The 0.1 default (InsightFace: 0.5) also detects tight CelebA-HQ crops;
        # the upstream code stores 0.6 but does not apply it.
        self.detector.prepare(
            ctx_id=self.ctx_id,
            input_size=self.det_size,
            det_thresh=self.det_thresh,
        )

    def align_bgr(
        self, image: np.ndarray, image_size: int = 224
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align the top-scoring face of an HWC BGR image; return crop and matrix."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a non-empty HWC BGR image")
        boxes, landmarks = self.detector.detect(
            image, max_num=0, metric="default"
        )
        if boxes.shape[0] == 0 or landmarks is None:
            raise RuntimeError("SCRFD did not detect a face with five landmarks")
        best = int(np.argmax(boxes[:, 4]))
        matrix = estimate_released_transform(
            landmarks[best], image_size=image_size
        )
        aligned = cv2.warpAffine(
            image,
            matrix,
            (image_size, image_size),
            borderValue=0.0,
        )
        return aligned, matrix

    @staticmethod
    def _as_images(
        images: torch.Tensor | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            if images.ndim != 4:
                raise ValueError(
                    f"Expected [B,3,H,W], got {tuple(images.shape)}"
                )
            values = list(images.unbind(0))
        else:
            values = list(images)
        result: list[torch.Tensor] = []
        for image in values:
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"Expected [3,H,W], got {tuple(image.shape)}"
                )
            result.append(image)
        if not result:
            raise ValueError("At least one image is required")
        return result

    def align_rgb255(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        *,
        image_size: int = 224,
    ) -> torch.Tensor:
        """Align RGB images in ``[0, 255]`` to CPU float32 ``[B, 3, S, S]`` crops."""
        aligned: list[torch.Tensor] = []
        for index, image in enumerate(self._as_images(images)):
            image = image.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(image).all():
                raise ValueError(f"Image {index} contains NaN or Inf")
            minimum = float(image.min())
            maximum = float(image.max())
            if minimum < 0.0 or maximum > 255.0:
                raise ValueError(
                    f"Image {index} must be RGB in [0,255], "
                    f"got [{minimum:.4f},{maximum:.4f}]"
                )
            rgb = (
                image.round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            try:
                crop_bgr, _ = self.align_bgr(
                    bgr, image_size=image_size
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to align image index {index}"
                ) from exc
            crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
            aligned.append(
                torch.from_numpy(crop_rgb)
                .permute(2, 0, 1)
                .to(torch.float32)
            )
        return torch.stack(aligned)


__all__ = ["AIDProAligner", "estimate_released_transform"]
