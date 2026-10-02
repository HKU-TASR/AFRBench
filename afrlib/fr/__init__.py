"""Face-recognition backends and the backend registry, which also resolves ``package.module:Class``."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import torch
    from .base import FaceRecognizer, ImagesLike

_FR_TARGETS: Dict[str, Tuple[str, str]] = {
    "frbench": ("afrlib.fr.frbench", "FRBenchRecognizer"),
    "facenet_pytorch": ("afrlib.fr.facenet", "FaceNetRecognizer"),
}


class _LazyFRRegistry(Mapping[str, type]):
    """Read-only backend registry that imports implementations on first use."""

    def __init__(self, targets: Dict[str, Tuple[str, str]]):
        self._targets = targets
        self._loaded: Dict[str, type] = {}

    def __getitem__(self, key: str) -> type:
        if key not in self._loaded:
            if key in self._targets:
                module_name, class_name = self._targets[key]
                self._loaded[key] = getattr(import_module(module_name), class_name)
            elif ":" in key:
                self._loaded[key] = _import_plugin(key)
            else:
                raise KeyError(key)
        return self._loaded[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def __len__(self) -> int:
        return len(self._targets)


def _import_plugin(path: str) -> type:
    """Import an out-of-tree backend given as ``"package.module:Class"``."""
    from .base import FaceRecognizer

    module_name, class_name = path.split(":", 1)
    cls = getattr(import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, FaceRecognizer)):
        raise TypeError(f"{path} is not a FaceRecognizer subclass")
    if getattr(cls, "backend_name", None) != path:
        raise ValueError(
            f"{path} must set backend_name = {path!r} so that get_spec() rebuilds it"
        )
    return cls


FR_REGISTRY: Mapping[str, type] = _LazyFRRegistry(_FR_TARGETS)
_CLASS_TO_KEY = {class_name: key for key, (_, class_name) in _FR_TARGETS.items()}

_LAZY_ATTRS = {
    "FaceRecognizer": (".base", "FaceRecognizer"),
    "ImagesLike": (".base", "ImagesLike"),
    "FREmbedResult": ("frbench", "FREmbedResult"),
    "FRDetectResult": ("frbench", "FRDetectResult"),
}


def build_fr(
    spec: Mapping[str, Any],
    device: Optional[Union[str, torch.device]] = None,
) -> FaceRecognizer:
    """Build one frozen, eval-mode recognizer from a spec dict.

    Args:
        spec: ``backend`` selects the backend from :data:`FR_REGISTRY`; the other keys are its
            constructor arguments, e.g. ``{"backend": "frbench", "backbone": "irse-100",
            "loss": "arcface", "dataset": "ms1m"}``.
        device: Device to move the model to.

    Raises:
        ValueError: When the spec names no backend.
    """
    kwargs = dict(spec)
    backend = kwargs.pop("backend", None)
    if backend is None:
        raise ValueError(f"FR spec is missing 'backend': {dict(spec)}")
    try:
        cls = FR_REGISTRY[backend]
    except KeyError:
        raise ValueError(
            f"Unknown face-recognition backend: {backend!r}. "
            f"Available: {sorted(FR_REGISTRY)}"
        ) from None
    recognizer = cls(**kwargs)
    if device is not None:
        recognizer = recognizer.to(device)
    return recognizer.eval().requires_grad_(False)


def build_fr_ensemble(
    specs: Sequence[Mapping[str, Any]],
    device: Optional[Union[str, torch.device]] = None,
) -> List[FaceRecognizer]:
    """Build a list of recognizers from spec dicts; see :func:`build_fr`."""
    return [build_fr(spec, device) for spec in specs]


def __getattr__(name: str):
    """Lazy ``from afrlib.fr import FaceRecognizer``/``FRBenchRecognizer``-style imports."""
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, attr = target
        value = getattr(import_module(module_name, __name__), attr)
        globals()[name] = value
        return value
    try:
        key = _CLASS_TO_KEY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = FR_REGISTRY[key]
    globals()[name] = value
    return value


__all__ = [
    "FR_REGISTRY",
    "FaceRecognizer",
    "ImagesLike",
    "build_fr",
    "build_fr_ensemble",
    "FREmbedResult",
    "FRDetectResult",
    "FRBenchRecognizer",
    "FaceNetRecognizer",
]
