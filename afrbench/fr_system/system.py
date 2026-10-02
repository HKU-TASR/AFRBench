"""The FR system: one FR model plus the intruder's deployed input pipeline."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from afrlib import get_logger
from afrlib.fr import build_fr as _construct_fr
from .ops import OPS, PURIFIERS
from .restore.align import (
    MODEL_RESTORERS,
    compose_to_arcface,
    ffhq_template,
    restorer_canvas_size,
)

logger = get_logger(__name__)

#: Resident FR weights keyed by ``(system.digest, device)``.
_FR_CACHE: Dict[Tuple[str, str], Any] = {}

#: Optional stages that run before detection, in execution order.
PRE_DETECT_OPS = ("quantize", "jpeg", "resize")
ALIGNED_CROP_SIDE = 112
_DISABLED_PURIFY = {"enabled": False, "method": "median", "kernel_size": 3}


@dataclasses.dataclass(frozen=True)
class FRSystem:
    """One (FR model, pipeline) pair, parsed from a config's ``fr_system`` block."""

    name: str
    fr_spec: Dict[str, Any]
    pipeline: Dict[str, Dict[str, Any]]

    # Parsing

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FRSystem":
        """Parse an ``fr_system`` block, ignoring ``noise_emulation``.

        Raises:
            ValueError: On an enabled ``gaussian`` stage, a ``purify.position`` key, or an
                unknown purifier.
        """
        name = raw.get("name")
        fr_spec = raw.get("fr")
        pipeline = raw.get("pipeline")
        if not name or not fr_spec or pipeline is None:
            raise ValueError("fr_system needs 'name', 'fr', and 'pipeline'")
        pipeline = {stage: dict(cfg) for stage, cfg in pipeline.items()}
        pipeline.pop("noise_emulation", None)
        gaussian = pipeline.get("gaussian") or {}
        if gaussian.get("enabled"):
            raise ValueError("gaussian was removed from the FR pipeline")
        pipeline.pop("gaussian", None)
        purify = pipeline.get("purify") or {}
        if "position" in purify:
            raise ValueError(
                "purify.position was removed; purify is post-detect only"
            )
        if purify.get("enabled"):
            method = purify.get("method")
            if method not in PURIFIERS:
                raise ValueError(f"purify.method must be one of {PURIFIERS}; got {method!r}")
        return cls(name=str(name), fr_spec=dict(fr_spec), pipeline=pipeline)

    # Identity

    def to_dict(self) -> Dict[str, Any]:
        """The normalized config; a disabled ``purify`` stage is written out explicitly."""
        pipeline = {stage: dict(cfg) for stage, cfg in self.pipeline.items()}
        pipeline.pop("noise_emulation", None)
        pipeline.pop("gaussian", None)
        purify = pipeline.get("purify") or {}
        if not purify.get("enabled"):
            pipeline["purify"] = dict(_DISABLED_PURIFY)
        return {"name": self.name, "fr": self.fr_spec, "pipeline": pipeline}

    @property
    def digest(self) -> str:
        """16-hex digest of the FR spec and normalized pipeline; ``name`` is only a label."""
        payload = {"fr": self.fr_spec, "pipeline": self.to_dict()["pipeline"]}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def weight_identity(self) -> str:
        """Checksum identity of the FR and detector weights this system uses."""
        detector = (self.pipeline.get("detect") or {}).get("detector", "retinaface_resnet50")
        payload = {
            "fr": self.fr_spec,
            "detector": detector,
            "pins": _weight_pins(self.fr_spec, str(detector)),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def for_gallery(self) -> "FRSystem":
        """The system that embeds the distractor crops: purification off, and quantize too when
        jpeg and tta are off, since the crops are already uint8; ``self`` if nothing changes.
        """
        purify = self.pipeline.get("purify") or {}
        quantize = self.pipeline.get("quantize") or {}
        strip_purify = bool(purify.get("enabled"))
        strip_quantize = bool(quantize.get("enabled")) and not (
            self._enabled("jpeg") or self._enabled("tta")
        )
        if not strip_purify and not strip_quantize:
            return self
        pipeline = {stage: dict(cfg) for stage, cfg in self.pipeline.items()}
        if strip_purify:
            pipeline["purify"] = dict(_DISABLED_PURIFY)
        if strip_quantize:
            disabled = dict(pipeline.get("quantize") or {})
            disabled["enabled"] = False
            pipeline["quantize"] = disabled
        return dataclasses.replace(self, pipeline=pipeline)

    def _enabled(self, stage: str) -> bool:
        return bool(self.pipeline.get(stage, {}).get("enabled", False))

    def _cfg(self, stage: str) -> Dict[str, Any]:
        return {k: v for k, v in self.pipeline.get(stage, {}).items() if k != "enabled"}

    def _purify_method(self) -> Optional[str]:
        purify = self.pipeline.get("purify") or {}
        if not purify.get("enabled"):
            return None
        return str(purify.get("method") or "")

    # Execution

    def build_fr(self, device: "torch.device | str"):
        """The FR model on ``device``, built once per (digest, device) and cached."""
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch_device.index is None and torch.cuda.is_available():
            torch_device = torch.device("cuda", torch.cuda.current_device())
        key = (self.digest, str(torch_device))
        cached = _FR_CACHE.get(key)
        if cached is not None:
            return cached
        recognizer = _construct_fr(self.fr_spec, torch_device)
        _FR_CACHE[key] = recognizer
        return recognizer

    def _apply_pre_detect(self, image: torch.Tensor) -> torch.Tensor:
        for stage in PRE_DETECT_OPS:
            if not self._enabled(stage):
                continue
            image = OPS[stage](image, **dict(self._cfg(stage)))
        return image

    def _apply_face_median(self, crop: torch.Tensor) -> torch.Tensor:
        if self._purify_method() != "median":
            return crop
        cfg = {
            k: v
            for k, v in (self.pipeline.get("purify") or {}).items()
            if k not in ("enabled", "method")
        }
        return OPS["median"](crop, **cfg)

    def embed_aligned_batch(
        self,
        images: torch.Tensor,
        device: "torch.device | str",
    ) -> torch.Tensor:
        """Embed a stacked ``[N, 3, H, W]`` batch of aligned crops in ``[0, 255]``, with the same
        ops as :meth:`embed_aligned`, on the calling thread.
        """
        fr = self.build_fr(device)
        torch_device = torch.device(device)
        tta = self._cfg("tta").get("ops", []) if self._enabled("tta") else []
        out_size = tuple(fr.input_size)
        if images.device != torch_device:
            images = images.to(torch_device, non_blocking=True)
        images = self._apply_pre_detect(images)
        if tuple(images.shape[-2:]) != out_size:
            images = torch.nn.functional.interpolate(
                images, size=out_size, mode="bilinear",
                align_corners=False, antialias=True,
            )
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.div(255.0)
        with torch.inference_mode():
            embeddings = fr.embed_crops(
                images, input_space="01", tta=tta, l2_normalize=True
            )
        return embeddings.detach().cpu()

    def embed_aligned(
        self,
        items: Sequence[Tuple[str, torch.Tensor]],
        device: "torch.device | str",
        *,
        batch_size: int = 32,
    ) -> "Tuple[List[str], torch.Tensor, List[str]]":
        """Embed items that are already aligned crops: no detector, no restorer and no median, so
        the failure list is always empty.
        """
        fr = self.build_fr(device)
        torch_device = torch.device(device)
        ids: List[str] = []
        rows: List[torch.Tensor] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            images = torch.stack(
                [
                    image
                    if image.device == torch_device
                    else image.to(torch_device, non_blocking=True)
                    for _image_id, image in chunk
                ]
            )
            embeddings = self.embed_aligned_batch(images, device)
            ids.extend(image_id for image_id, _img in chunk)
            rows.append(embeddings)
        stacked = torch.cat(rows, dim=0) if rows else torch.zeros((0, fr.embedding_dim))
        return ids, stacked, []

    def embed(
        self,
        items: Sequence[Tuple[str, torch.Tensor]],
        device: "torch.device | str",
        *,
        batch_size: int = 32,
        prefetch: bool = True,
    ) -> "Tuple[List[str], torch.Tensor, List[str]]":
        """Embed ``(image_id, image)`` items through the full pipeline.

        Args:
            items: ``(image_id, image)`` pairs, each image ``[3, H, W]`` in ``[0, 255]``.
            device: Device to embed on.
            batch_size: Images per batch.
            prefetch: Load the next slice on a host thread while the current batch runs, for a
                file-backed lazy loader.

        Returns:
            ``(ids, embeddings, failed)``, where ``failed`` lists the ids whose alignment did
            not yield exactly one face.
        """
        from .detector import get_detector

        fr = self.build_fr(device)
        detect_cfg = self._cfg("detect")
        detector = get_detector(detect_cfg.get("detector", "retinaface_resnet50"), device)
        keep_largest = detect_cfg.get("keep", "largest") == "largest"
        template = self._cfg("align").get("template", "arcface_112")
        if template != "arcface_112":
            raise ValueError(
                f"align.template {template!r} is not implemented; the eval "
                "pipeline aligns to the ArcFace 5-point template."
            )
        out_size = tuple(fr.input_size)
        tta = self._cfg("tta").get("ops", []) if self._enabled("tta") else []
        method = self._purify_method()
        restorer = method in MODEL_RESTORERS
        canvas_size = restorer_canvas_size(method) if restorer else ALIGNED_CROP_SIDE
        ffhq = ffhq_template(canvas_size) if restorer else None
        purify_cfg = {
            k: v
            for k, v in (self.pipeline.get("purify") or {}).items()
            if k not in ("enabled", "method")
        }

        ids: List[str] = []
        rows: List[torch.Tensor] = []
        failed: List[str] = []
        slices = (
            _prefetched_slices(items, batch_size)
            if prefetch
            else (
                (start, _host_materialize(items, start, batch_size))
                for start in range(0, len(items), batch_size)
            )
        )
        for start, chunk in slices:
            images = [self._apply_pre_detect(img.to(device)) for _id, img in chunk]
            detections = detector.detect(images)
            if restorer:
                canvases = detector.align(
                    images, detections, size=(canvas_size, canvas_size),
                    template=ffhq, keep_largest=keep_largest,
                )
            else:
                canvases = detector.align(
                    images, detections, size=out_size, keep_largest=keep_largest,
                )
            pending_ids: List[str] = []
            pending_crops: List[torch.Tensor] = []
            for (image_id, _img), crops in zip(chunk, canvases):
                if crops.shape[0] != 1:
                    failed.append(image_id)
                    continue
                pending_ids.append(image_id)
                pending_crops.append(crops[0])
            if not pending_crops:
                continue
            stacked = torch.stack(pending_crops, dim=0)
            if restorer:
                stacked = OPS[method](stacked, **purify_cfg)
                stacked = compose_to_arcface(stacked, canvas_size)
            else:
                stacked = self._apply_face_median(stacked)
            stacked = stacked.div(255.0)
            with torch.no_grad():
                embeddings = fr.embed_crops(
                    stacked, input_space="01", tta=tta, l2_normalize=True
                )
            ids.extend(pending_ids)
            rows.append(embeddings.detach().cpu())
        embeddings = (
            torch.cat(rows, dim=0) if rows else torch.zeros((0, fr.embedding_dim))
        )
        return ids, embeddings, failed


def _is_lazy_images(items: Sequence) -> bool:
    return callable(getattr(items, "_load", None))


def _allow_host_prefetch(items: Sequence) -> bool:
    """Prefetch only lazy loaders that allow it; loaded tensors and LMDB stay serial."""
    if not _is_lazy_images(items):
        return False
    return bool(getattr(items, "host_prefetch", True))


def _host_materialize(items: Sequence, start: int, n: int):
    """Load one slice on the calling thread, without moving tensors to a device."""
    return list(items[start : start + n])


def _prefetched_slices(items: Sequence, batch_size: int):
    """Yield ``(start, materialized_chunk)``. At most one host lookahead batch."""
    from concurrent.futures import ThreadPoolExecutor

    starts = list(range(0, len(items), batch_size))
    if not starts:
        return
    if not _allow_host_prefetch(items):
        for start in starts:
            yield start, _host_materialize(items, start, batch_size)
        return
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_host_materialize, items, starts[0], batch_size)
        for index, start in enumerate(starts):
            chunk = future.result()
            nxt = starts[index + 1] if index + 1 < len(starts) else None
            if nxt is not None:
                future = executor.submit(_host_materialize, items, nxt, batch_size)
            yield start, chunk
    finally:
        executor.shutdown(wait=True)


def _weight_pins(fr_spec: Dict[str, Any], detector: str) -> Dict[str, str]:
    """Registered asset checksums for the FR spec and eval detector, if known."""
    pins: Dict[str, str] = {"detector": detector}
    try:
        from afrlib.utils.assets import _lookup

        backend = str(fr_spec.get("backend") or "")
        candidates = [
            f"fr/{backend}",
            f"frbench/{fr_spec.get('backbone')}_{fr_spec.get('loss')}_{fr_spec.get('dataset')}",
        ]
        for key in candidates:
            try:
                asset = _lookup(key)
            except Exception:
                continue
            pins[key] = getattr(asset, "sha256", "") or str(asset)
    except Exception:
        pass
    return pins
