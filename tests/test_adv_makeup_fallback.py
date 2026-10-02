"""Adv-Makeup fallbacks: default face box on a detector miss, source crop on a per-image failure."""
from __future__ import annotations

import numpy as np
import torch

from afrlib.afr.adv_makeup.adv_makeup import AdvMakeup
from afrlib.afr.adv_makeup.preprocess import (
    Landmark106Detector,
    default_aligned_bbox,
    largest_detection_bbox,
)
from afrlib.contracts import EvalData


def test_default_aligned_bbox_insets_ladn_canvas():
    box = default_aligned_bbox(600, 600, inset=0.12)
    assert box.tolist() == [72.0, 72.0, 528.0, 528.0]


def test_largest_detection_bbox_none_when_empty():
    assert largest_detection_bbox(None) is None
    assert largest_detection_bbox(torch.zeros(0, 6)) is None


def test_largest_detection_bbox_picks_highest_area():
    detection = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.0, 1.0],
            [0.0, 0.0, 40.0, 40.0, 0.0, 1.0],
        ]
    )
    box = largest_detection_bbox(detection)
    assert box.tolist() == [0.0, 0.0, 40.0, 40.0]


def test_detector_uses_default_box_when_frbench_misses():
    captured = {}

    class _Det:
        detections = [None]

        def detect(self, _images):
            return self

    class _Model:
        def get(self, _bgr, face):
            captured["bbox"] = list(face.bbox)
            return np.zeros((106, 2), dtype=np.float32)

    detector = Landmark106Detector.__new__(Landmark106Detector)
    detector.device = torch.device("cpu")
    detector.face_detector = _Det()
    detector.landmark_model = _Model()
    image = torch.zeros(1, 3, 600, 600)
    out = detector(image)
    assert out.shape == (1, 106, 2)
    assert captured["bbox"] == default_aligned_bbox(600, 600).tolist()


def test_protect_returns_source_crop_when_landmarks_fail(monkeypatch):
    method = AdvMakeup.__new__(AdvMakeup)
    method.device = torch.device("cpu")
    method.poisson_blend = False

    class _Detector:
        def __call__(self, _source):
            raise RuntimeError("106-landmark model failed on Adv-Makeup source image 0")

    monkeypatch.setattr(method, "_check_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(method, "_require_ready", lambda: None)
    monkeypatch.setattr(
        method,
        "_validate_external_images",
        lambda photo, **_kw: [photo[0]],
    )
    monkeypatch.setattr(method, "_get_landmark_detector", lambda: _Detector())
    photo = torch.linspace(0, 255, 3 * 16 * 16).reshape(3, 16, 16)
    geometry = {
        "landmarks": [[4, 4], [12, 4], [8, 8], [4, 12], [12, 12]],
        "face_box": [2, 2, 14, 14],
    }
    out = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry]))
    assert len(out) == 1
    assert out[0].shape[0] == 3
    assert torch.isfinite(out[0]).all()


def test_protect_returns_source_crop_when_orbital_geometry_fails(monkeypatch):
    method = AdvMakeup.__new__(AdvMakeup)
    method.device = torch.device("cpu")
    method.poisson_blend = False

    class _Detector:
        def __call__(self, _source):
            return torch.zeros(1, 106, 2)

    monkeypatch.setattr(method, "_check_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(method, "_require_ready", lambda: None)
    monkeypatch.setattr(
        method,
        "_validate_external_images",
        lambda photo, **_kw: [photo[0]],
    )
    monkeypatch.setattr(method, "_get_landmark_detector", lambda: _Detector())
    monkeypatch.setattr(
        "afrlib.afr.adv_makeup.adv_makeup.build_orbital_geometry",
        lambda *_args, **_kw: (_ for _ in ()).throw(ValueError("Degenerate orbital crop")),
    )
    photo = torch.linspace(0, 255, 3 * 16 * 16).reshape(3, 16, 16)
    geometry = {
        "landmarks": [[4, 4], [12, 4], [8, 8], [4, 12], [12, 12]],
        "face_box": [2, 2, 14, 14],
    }
    out = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry]))
    assert len(out) == 1
    assert out[0].shape[0] == 3
    assert torch.isfinite(out[0]).all()
