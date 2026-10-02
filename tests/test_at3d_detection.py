"""AT3D MTCNN fallbacks: stored landmarks seed POS; POS-mapped crop points seed FR."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from afrlib.afr.at3d.at3d import _fallback_landmarks
from afrlib.afr.at3d.reconstruction import (
    FaceDetectionError,
    FaceReconstructor,
    _as_five_landmarks,
    align_deep3d,
)


def _frontal_landmarks(side: int) -> np.ndarray:
    scale = float(side)
    return np.asarray(
        [
            [0.35 * scale, 0.38 * scale],
            [0.65 * scale, 0.38 * scale],
            [0.50 * scale, 0.55 * scale],
            [0.38 * scale, 0.72 * scale],
            [0.62 * scale, 0.72 * scale],
        ],
        dtype=np.float32,
    )


def _mtcnn_face(box=(40, 40, 140, 140), offset=0.0) -> dict:
    return {
        "box": list(box),
        "keypoints": {
            "left_eye": (70.0 + offset, 80.0),
            "right_eye": (150.0 + offset, 80.0),
            "nose": (110.0 + offset, 120.0),
            "mouth_left": (80.0 + offset, 160.0),
            "mouth_right": (140.0 + offset, 160.0),
        },
    }


class _EmptyThenCropDetector:
    """Miss the original, succeed on the canonical 224 crop."""

    def detect_faces(self, image):
        if image.shape[0] == 224 and image.shape[1] == 224:
            return [_mtcnn_face()]
        return []


class _AlwaysDetector:
    def detect_faces(self, image):
        return [_mtcnn_face(offset=3.0)]


class _NeverDetector:
    def detect_faces(self, image):
        return []


class _OriginalOnlyDetector:
    """Hit the original, miss the canonical 224 crop."""

    def detect_faces(self, image):
        if image.shape[0] == 224 and image.shape[1] == 224:
            return []
        return [_mtcnn_face(offset=3.0)]


class _DummyRegressor:
    def __call__(self, images01: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images01.shape[0], 257, dtype=images01.dtype)


def _reconstructor(detector) -> FaceReconstructor:
    recon = FaceReconstructor.__new__(FaceReconstructor)
    recon.detector = detector
    recon.device = torch.device("cpu")
    recon.regressor = _DummyRegressor()
    return recon


def test_first_stage_falls_back_to_stored_landmarks(caplog):
    recon = _reconstructor(_EmptyThenCropDetector())
    stored = _frontal_landmarks(64)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with caplog.at_level("WARNING"):
        out = recon._first_stage_landmarks(image, 43, stored)
    np.testing.assert_allclose(out, stored)
    assert "eval image 43" in caplog.text
    assert "eval_geometry" in caplog.text


def test_first_stage_prefers_mtcnn_when_it_fires():
    recon = _reconstructor(_AlwaysDetector())
    stored = _frontal_landmarks(64)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    out = recon._first_stage_landmarks(image, 0, stored)
    expected = np.asarray(
        [
            [73.0, 80.0],
            [153.0, 80.0],
            [113.0, 120.0],
            [83.0, 160.0],
            [143.0, 160.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected)


def test_first_stage_still_errors_without_fallback():
    recon = _reconstructor(_NeverDetector())
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(FaceDetectionError, match="batch item 7"):
        recon._first_stage_landmarks(image, 7, None)


def test_prepare_uses_fallback_then_redetects_on_crop():
    recon = _reconstructor(_EmptyThenCropDetector())
    photo = torch.zeros(1, 3, 256, 256)
    stored = _frontal_landmarks(256)
    prepared = recon.prepare(
        photo, fallback_landmarks=[stored], image_ids=[41]
    )
    assert prepared.images.shape == (1, 3, 224, 224)
    assert prepared.landmarks.shape == (1, 5, 2)
    # Second-stage points come from the crop detector, not the stored original.
    np.testing.assert_allclose(
        prepared.landmarks[0].numpy(),
        np.asarray(
            [
                [70.0, 80.0],
                [150.0, 80.0],
                [110.0, 120.0],
                [80.0, 160.0],
                [140.0, 160.0],
            ],
            dtype=np.float32,
        ),
    )


def test_prepare_second_stage_uses_pos_landmarks_when_crop_mtcnn_misses(caplog):
    recon = _reconstructor(_NeverDetector())
    photo = torch.zeros(1, 3, 256, 256)
    stored = _frontal_landmarks(256)
    image_np = (
        photo[0].detach().round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    )
    _, expected, _ = align_deep3d(image_np, stored)
    with caplog.at_level("WARNING"):
        prepared = recon.prepare(
            photo, fallback_landmarks=[stored], image_ids=[41]
        )
    assert prepared.images.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(prepared.landmarks[0].numpy(), expected)
    assert "eval image 41" in caplog.text
    assert "POS-mapped" in caplog.text


def test_prepare_second_stage_uses_pos_when_first_stage_mtcnn_hits(caplog):
    recon = _reconstructor(_OriginalOnlyDetector())
    photo = torch.zeros(1, 3, 256, 256)
    stored = _frontal_landmarks(256)
    image_np = (
        photo[0].detach().round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    )
    first_stage = np.asarray(
        [
            [73.0, 80.0],
            [153.0, 80.0],
            [113.0, 120.0],
            [83.0, 160.0],
            [143.0, 160.0],
        ],
        dtype=np.float32,
    )
    _, expected, _ = align_deep3d(image_np, first_stage)
    with caplog.at_level("WARNING"):
        prepared = recon.prepare(
            photo, fallback_landmarks=[stored], image_ids=[8]
        )
    np.testing.assert_allclose(prepared.landmarks[0].numpy(), expected)
    assert "eval image 8" in caplog.text
    assert "POS-mapped" in caplog.text


def test_fallback_landmarks_helper_reads_eval_geometry():
    stored = _frontal_landmarks(100)
    out = _fallback_landmarks([{"landmarks": stored.tolist(), "face_box": [0, 0, 1, 1]}], 0)
    np.testing.assert_allclose(out, stored)
    assert _fallback_landmarks(None, 0) is None
    with pytest.raises(ValueError, match="five 2D landmarks"):
        _as_five_landmarks([[1.0, 2.0]])
