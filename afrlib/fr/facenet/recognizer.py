"""Example FR backend: facenet-pytorch's InceptionResnetV1.

Reference Code: https://github.com/timesler/facenet-pytorch
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from frbench import FaceDetector, FRDetectResult, FREmbedResult

from afrlib.contracts import ChannelType

from ..base import FaceRecognizer, ImagesLike

_PRETRAINED_KEYS = ("vggface2", "casia-webface")


class FaceNetRecognizer(FaceRecognizer):
    """InceptionResnetV1 (facenet-pytorch) on 160x160 RGB crops, giving 512-D embeddings that the
    model itself standardizes and L2-normalizes.
    """

    backend_name = "facenet_pytorch"
    channel = ChannelType.RGB
    embedding_dim = 512
    input_size = (160, 160)

    def __init__(
        self,
        pretrained: str = "vggface2",
        *,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        """Load the pretrained InceptionResnetV1.

        Args:
            pretrained: facenet-pytorch weights, ``"vggface2"`` or
                ``"casia-webface"``.
            detector: frbench ``FaceDetector`` to share; a default one is
                built when ``None``.
        """
        super().__init__()
        if pretrained not in _PRETRAINED_KEYS:
            raise ValueError(
                f"Unknown facenet-pytorch pretraining {pretrained!r}; "
                f"expected one of {_PRETRAINED_KEYS}."
            )
        from facenet_pytorch import InceptionResnetV1

        self.pretrained = pretrained
        self.backbone = (
            InceptionResnetV1(pretrained=pretrained, classify=False)
            .eval()
            .requires_grad_(False)
        )
        self.detector = detector if detector is not None else FaceDetector()
        self.detector_name = self.detector.name
        # (x / 255 - mean) / std with these constants is (x - 127.5) / 128.
        self.register_buffer(
            "_mean", torch.full((1, 3, 1, 1), 127.5 / 255.0), persistent=False
        )
        self.register_buffer(
            "_std", torch.full((1, 3, 1, 1), 128.0 / 255.0), persistent=False
        )

    # Identity and metadata (steps 2 and 4)

    @property
    def name(self) -> str:
        return f"facenet_pytorch/{self.pretrained}"

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def get_spec(self) -> Dict[str, Any]:
        spec = super().get_spec()
        spec["pretrained"] = self.pretrained
        return spec

    def describe(self) -> Dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "pretrained": self.pretrained,
                "detector": self.detector_name,
                "weights": "facenet-pytorch torch-hub cache (self-managed)",
                "output_normalization": "backbone_l2_normalized",
            }
        )
        return description

    # Pipeline (step 3)

    @staticmethod
    def _as_image_list(imgs: ImagesLike) -> List[torch.Tensor]:
        """Split ``(3,H,W)``, ``(B,3,H,W)`` or a list into ``(3,H,W)`` images."""
        if isinstance(imgs, (list, tuple)):
            images = list(imgs)
        else:
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            if imgs.dim() != 4:
                raise ValueError(
                    f"Expected (B,3,H,W), (3,H,W), or a list, got {tuple(imgs.shape)}"
                )
            images = list(imgs.unbind(0))
        result = []
        for image in images:
            if image.dim() == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"Expected image shape (3,H,W), got {tuple(image.shape)}"
                )
            result.append(image)
        return result

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
        detections: Optional[
            Union[FRDetectResult, Sequence[Optional[torch.Tensor]]]
        ] = None,
    ) -> FREmbedResult:
        """Detect, align or crop, and embed (see :meth:`FaceRecognizer.forward`); an undetected
        image is embedded whole unless ``discard_invalid`` is set.
        """
        images = self._as_image_list(imgs)
        empty_crops = self._dev.new_empty((0, 3, *self.input_size))
        if not images:
            return FREmbedResult(
                embeddings=self._dev.new_empty((0, self.embedding_dim)),
                indices=[],
                crops=empty_crops,
            )

        if not need_crop:
            faces01 = [
                image.unsqueeze(0).to(self.device, torch.float32).div(255.0)
                for image in images
            ]
            indices = list(range(len(images)))
        else:
            dets = (
                detections.detections
                if isinstance(detections, FRDetectResult)
                else detections
            )
            if dets is not None and len(dets) != len(images):
                raise ValueError(
                    f"Expected {len(images)} detections, got {len(dets)}."
                )
            if dets is None:
                dets = self.detect(
                    images, conf_thresh=conf_thresh, iou_thresh=iou_thresh
                ).detections

            faces01 = []
            indices = []
            for index, (image, image_dets) in enumerate(zip(images, dets)):
                if image_dets is None or image_dets.shape[0] == 0:
                    if not discard_invalid:
                        fallback = image.unsqueeze(0).to(self.device, torch.float32).div(255.0)
                        if tuple(fallback.shape[-2:]) != self.input_size:
                            fallback = F.interpolate(
                                fallback, size=self.input_size, mode="bilinear",
                                align_corners=False, antialias=True,
                            )
                        faces01.append(fallback)
                        indices.append(index)
                    continue
                if need_align:
                    groups = self.detector.align(
                        [image],
                        detections=[image_dets],
                        size=self.input_size,
                        keep_largest=keep_largest,
                    )
                else:
                    groups = self.detector.crop(
                        [image],
                        detections=[image_dets],
                        size=self.input_size,
                        loosen=loosen_crop,
                        keep_largest=keep_largest,
                    )
                group = groups[0]
                if group.shape[0]:
                    faces01.append(group.to(self.device, torch.float32).div(255.0))
                    indices.extend([index] * group.shape[0])

        if not faces01:
            return FREmbedResult(
                embeddings=self._dev.new_empty((0, self.embedding_dim)),
                indices=[],
                crops=empty_crops,
            )
        # Same steps as embed_crops, but keeps the normalized crops, which
        # frbench returns as FREmbedResult.crops.
        crops01 = torch.cat(faces01, dim=0)
        if tuple(crops01.shape[-2:]) != self.input_size:
            crops01 = F.interpolate(
                crops01,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        normalized = self.normalize(crops01, input_space="01")
        embeddings = self._embed_normalized(
            normalized, tta=tta, l2_normalize=l2_normalize
        )
        return FREmbedResult(
            embeddings=embeddings,
            indices=indices,
            crops=normalized,
        )
