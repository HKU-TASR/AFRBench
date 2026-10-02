"""The eval-side face detector: RetinaFace, selected by the FR system's config."""
from __future__ import annotations

from typing import Dict

from frbench import FaceDetector

#: Detector names accepted in a config's ``detect.detector`` field.
DETECTORS = ("retinaface_resnet50", "retinaface_mobilenetv1")

_CACHE: Dict[str, FaceDetector] = {}


def get_detector(
    name: str = "retinaface_resnet50", device: "str" = "cpu"
) -> FaceDetector:
    """The shared detector for ``name`` on ``device`` (built once, then cached)."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown eval detector {name!r}; known: {list(DETECTORS)}")
    key = f"{name}:{device}"
    if key not in _CACHE:
        _CACHE[key] = FaceDetector(name).to(device).eval()
    return _CACHE[key]
