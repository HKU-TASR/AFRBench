"""The intruder's deployed FR pipeline: model, image-space ops, and detector."""
from .detector import DETECTORS, get_detector
from .ops import OPS, PURIFIERS
from .system import PRE_DETECT_OPS, FRSystem

__all__ = [
    "FRSystem",
    "PRE_DETECT_OPS",
    "OPS",
    "PURIFIERS",
    "DETECTORS",
    "get_detector",
]
