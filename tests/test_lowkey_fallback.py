"""LowKey RetinaFace miss: stored eval_geometry seeds alignment."""
from __future__ import annotations

import pytest
import torch
from frbench import FRDetectResult

from afrlib.afr.lowkey.lowkey import (
    LowKey,
    _fill_missing_detections,
    _geometry_detection,
)


def _geometry() -> dict:
    return {
        "landmarks": [
            [40.0, 50.0],
            [80.0, 50.0],
            [60.0, 70.0],
            [45.0, 90.0],
            [75.0, 90.0],
        ],
        "face_box": [30.0, 40.0, 90.0, 100.0],
    }


class _StubDetector:
    def align(self, imgs, detections, **_kwargs):
        dets = detections.detections if hasattr(detections, "detections") else detections
        out = []
        for img, det in zip(imgs, dets):
            if det is None or det.shape[0] == 0:
                out.append(img.new_zeros(0, 3, 224, 224))
            else:
                out.append(img.new_zeros(1, 3, 224, 224))
        return out


def test_geometry_detection_uses_face_box():
    det = _geometry_detection(_geometry(), torch.device("cpu"))
    assert det is not None
    assert det.shape == (1, 16)
    assert det[0, :4].tolist() == [30.0, 40.0, 90.0, 100.0]


def test_geometry_detection_none_without_landmarks():
    assert _geometry_detection({"face_box": [0, 0, 1, 1]}, torch.device("cpu")) is None
    assert _geometry_detection(None, torch.device("cpu")) is None


def test_fill_missing_uses_eval_geometry(caplog):
    empty = FRDetectResult(detections=[None, torch.zeros(0, 16)])
    with caplog.at_level("WARNING"):
        filled = _fill_missing_detections(
            empty, [_geometry(), _geometry()], torch.device("cpu")
        )
    assert filled.detections[0] is not None
    assert filled.detections[0].shape == (1, 16)
    assert filled.detections[1].shape == (1, 16)
    assert "eval image 0" in caplog.text
    assert "eval_geometry" in caplog.text


def test_fill_missing_keeps_retinaface_hits():
    hit = FRDetectResult.from_landmarks(
        torch.tensor(
            [[10.0, 12.0], [20.0, 12.0], [15.0, 18.0], [11.0, 24.0], [19.0, 24.0]]
        )
    )
    filled = _fill_missing_detections(hit, [_geometry()], torch.device("cpu"))
    assert torch.equal(filled.detections[0], hit.detections[0])


def test_fill_missing_without_geometry_leaves_empty():
    empty = FRDetectResult(detections=[None])
    filled = _fill_missing_detections(empty, None, torch.device("cpu"))
    assert filled.detections[0] is None


def test_align_largest_succeeds_after_geometry_fill():
    method = LowKey.__new__(LowKey)
    method.device = torch.device("cpu")
    method.fd = _StubDetector()
    imgs = [torch.zeros(3, 128, 128)]
    dets = _fill_missing_detections(
        FRDetectResult(detections=[None]), [_geometry()], torch.device("cpu")
    )
    aligned = method._align_largest(imgs, dets)
    assert aligned.shape == (1, 3, 224, 224)


def test_align_largest_still_errors_without_geometry():
    method = LowKey.__new__(LowKey)
    method.device = torch.device("cpu")
    method.fd = _StubDetector()
    imgs = [torch.zeros(3, 128, 128)]
    dets = _fill_missing_detections(
        FRDetectResult(detections=[None]), None, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="failed indices: \\[0\\]"):
        method._align_largest(imgs, dets)
