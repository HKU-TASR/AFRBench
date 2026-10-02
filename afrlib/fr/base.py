"""Face-recognizer interface implemented by every AFRLib FR backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from frbench import FRDetectResult, FREmbedResult

from afrlib.contracts import ChannelType

ImagesLike = Union[torch.Tensor, Sequence[torch.Tensor]]

_INPUT_SPACES = ("01", "pm1")
_SUPPORTED_TTA = ("flip_horizontal",)


class FaceRecognizer(nn.Module, ABC):
    """Interface of an AFRLib face-recognition backend (detect, align, backbone, embedding);
    `docs/extending.md <../../docs/extending.md>`_ lists what a subclass provides.
    """

    # Registry key, e.g. "frbench"; an out-of-tree backend uses its
    # "package.module:Class" path. Set by every subclass.
    backend_name: ClassVar[str]

    # Model-native normalization on the [0, 1] scale, in the backbone's channel order and
    # broadcastable to (N, 3, H, W); attributes, buffers or properties.
    mean: torch.Tensor
    std: torch.Tensor
    channel: ChannelType

    # The raw embedding network: native normalized input -> (N, D) embedding.
    backbone: nn.Module

    # Optional detector used by detect(); an annotation only, since a class-level default
    # would shadow the submodule registered on assignment.
    detector: Optional[Any]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dev", torch.empty(0), persistent=False)

    # Identity

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique instance name, e.g. ``"facenet_pytorch/vggface2"``."""

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Crop size ``(H, W)`` the backbone consumes."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimension."""

    @property
    def device(self) -> torch.device:
        """Device this recognizer lives on."""
        return self._dev.device

    def get_spec(self) -> Dict[str, Any]:
        """Round-trippable build spec: the identity-defining, JSON-able constructor arguments, from
        which ``build_fr(fr.get_spec())`` rebuilds this recognizer.
        """
        return {"backend": self.backend_name}

    def describe(self) -> Dict[str, Any]:
        """Provenance dict for ``AFRMethod.get_config()``: the metadata plus :meth:`get_spec`."""
        return {
            "backend": self.backend_name,
            "name": self.name,
            "spec": self.get_spec(),
            "input_size": [int(v) for v in self.input_size],
            "embedding_dim": int(self.embedding_dim),
            "channel": self.channel.value,
            "mean": [float(v) for v in self.mean.flatten()],
            "std": [float(v) for v in self.std.flatten()],
        }

    # Low-level API

    def normalize(self, crops: torch.Tensor, *, input_space: str = "01") -> torch.Tensor:
        """Convert pre-cropped RGB faces to the backbone's native input space.

        Args:
            crops: ``(N, 3, H, W)`` RGB float tensor in ``input_space``.
            input_space: ``"01"`` for ``[0, 1]`` pixels, ``"pm1"`` for
                ``[-1, 1]`` pixels.

        Returns:
            ``(N, 3, H, W)`` tensor in the backbone's channel order,
            normalized with ``mean`` and ``std``.
        """
        if input_space == "pm1":
            crops = crops.add(1.0).div(2.0)
        elif input_space != "01":
            raise ValueError(
                f"Unknown input_space {input_space!r}; expected one of {_INPUT_SPACES}."
            )
        if self.channel is ChannelType.BGR:
            crops = crops[:, [2, 1, 0]]
        return (crops - self.mean) / self.std

    def embed_crops(
        self,
        crops: torch.Tensor,
        *,
        input_space: str = "01",
        mode: str = "bilinear",
        antialias: bool = True,
        tta: Sequence[str] = (),
        l2_normalize: bool = False,
    ) -> torch.Tensor:
        """Embed pre-cropped faces (resize, normalize, backbone), with no detection, alignment,
        range validation or clamping.

        Args:
            crops: ``(N, 3, H, W)`` or ``(3, H, W)`` RGB float tensor in the
                declared ``input_space``.
            input_space: ``"01"`` (``[0, 1]``) or ``"pm1"`` (``[-1, 1]``).
            mode: Interpolation mode of the resize to ``input_size`` (applied
                only when the crop size differs).
            antialias: Antialias the resize (recommended for downsampling).
            tta: Test-time augmentations averaged over; supports
                ``"flip_horizontal"``.
            l2_normalize: L2-normalize the returned embeddings.

        Returns:
            ``(N, D)`` embeddings.
        """
        if crops.dim() == 3:
            crops = crops.unsqueeze(0)
        if crops.dim() != 4 or crops.shape[1] != 3:
            raise ValueError(
                f"Expected crops (N,3,H,W) or (3,H,W), got {tuple(crops.shape)}"
            )
        x = crops.to(device=self.device, dtype=torch.float32)
        if tuple(x.shape[-2:]) != tuple(self.input_size):
            x = F.interpolate(
                x,
                size=tuple(self.input_size),
                mode=mode,
                align_corners=False,
                antialias=antialias,
            )
        x = self.normalize(x, input_space=input_space)
        return self._embed_normalized(x, tta=tta, l2_normalize=l2_normalize)

    def _embed_normalized(
        self,
        x: torch.Tensor,
        *,
        tta: Sequence[str],
        l2_normalize: bool,
    ) -> torch.Tensor:
        """Backbone forward over TTA views of already-normalized crops."""
        unsupported = set(tta) - set(_SUPPORTED_TTA)
        if unsupported:
            raise ValueError(
                f"Unsupported TTA {sorted(unsupported)}; "
                f"supported: {list(_SUPPORTED_TTA)}"
            )
        views = [x]
        if "flip_horizontal" in tta:
            views.append(torch.flip(x, dims=(-1,)))
        embeddings = self.backbone(torch.cat(views, dim=0))
        embeddings = embeddings.reshape(len(views), x.shape[0], -1).mean(0)
        if l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    # Pipeline API

    def detect(
        self,
        imgs: ImagesLike,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
    ) -> FRDetectResult:
        """Detect faces with ``self.detector`` without embedding them; pass the result to
        :meth:`forward` as ``detections`` to skip re-detection.

        Raises:
            NotImplementedError: If this backend has no detector.
        """
        detector = getattr(self, "detector", None)
        if detector is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide face detection."
            )
        return detector.detect(
            imgs, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )

    @abstractmethod
    def forward(
        self,
        imgs: ImagesLike,
        need_crop: bool = True,
        need_align: bool = True,
        keep_largest: bool = True,
        discard_invalid: bool = False,
        conf_thresh: float = 0.8,
        iou_thresh: float = 0.4,
        loosen_crop: float = 1.0,
        tta: Sequence[str] = ("flip_horizontal",),
        l2_normalize: bool = False,
        detections: Optional[Union[FRDetectResult, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> FREmbedResult:
        """Full pipeline on RGB ``[0, 255]`` images: detect, align or crop, backbone, embedding.

        Args:
            imgs: ``(3, H, W)``, ``(B, 3, H, W)``, or a list of ``(3, Hi, Wi)``.
            need_crop: ``False`` treats the inputs as pre-cropped faces.
            need_align: Align the detected face before embedding.
            keep_largest: Keep only the largest detection per image.
            discard_invalid: Drop images with no usable detection instead of failing.
            conf_thresh: Detector confidence threshold.
            iou_thresh: Detector NMS threshold.
            loosen_crop: Scale applied to the detected box before cropping.
            tta: Test-time augmentations to average over.
            l2_normalize: L2-normalize the returned embeddings.
            detections: Detections from :meth:`detect`, to skip re-detection.
        """

    def embed(self, imgs: ImagesLike, **kwargs: Any) -> FREmbedResult:
        """Alias for :meth:`forward` (invokes module hooks via ``__call__``)."""
        return self(imgs, **kwargs)
