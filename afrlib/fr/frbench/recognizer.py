"""FRBench backend: any ``frbench`` model as an AFRLib :class:`FaceRecognizer`."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from frbench import FR, FaceDetector, FRDetectResult, FREmbedResult

from afrlib.contracts import ChannelType

from ..base import FaceRecognizer, ImagesLike


class FRBenchRecognizer(FaceRecognizer):
    """One ``frbench.FR`` pipeline, with its RetinaFace detector."""

    backend_name = "frbench"

    def __init__(
        self,
        backbone: str,
        loss: str,
        dataset: str,
        *,
        detector: Optional[Union[str, FaceDetector]] = None,
    ) -> None:
        """Build and load the pipeline for ``(backbone, loss, dataset)``.

        Args:
            backbone: Backbone key, e.g. ``"irse-100"`` or ``"mobilefacenet"``.
            loss: Loss key, e.g. ``"arcface"``.
            dataset: Training dataset key, e.g. ``"ms1m"``.
            detector: RetinaFace asset name (default
                ``"retinaface_mobilenetv1"``), or an existing
                ``frbench.FaceDetector`` instance to share across recognizers.
        """
        super().__init__()
        self.backbone_type = backbone
        self.loss_type = loss
        self.dataset_type = dataset
        self._fr = FR(
            backbone_type=backbone,
            loss_type=loss,
            dataset_type=dataset,
            detector=detector if detector is not None else "retinaface_mobilenetv1",
            eager_load=True,
        )
        self.detector = self._fr.detector

    @property
    def name(self) -> str:
        return f"frbench/{self._fr.model_name}"

    @property
    def input_size(self) -> tuple[int, int]:
        return self._fr.input_size

    @property
    def embedding_dim(self) -> int:
        return self._fr.embedding_dim

    def get_spec(self) -> Dict[str, Any]:
        spec = super().get_spec()
        spec.update(
            {
                "backbone": self.backbone_type,
                "loss": self.loss_type,
                "dataset": self.dataset_type,
            }
        )
        return spec

    def describe(self) -> Dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "backbone": self.backbone_type,
                "loss": self.loss_type,
                "dataset": self.dataset_type,
                "detector": self._fr.detector_name,
            }
        )
        return description

    @property
    def backbone(self) -> torch.nn.Module:
        return self._fr.backbone

    @property
    def mean(self) -> torch.Tensor:
        return self._fr.preprocessor.mean

    @property
    def std(self) -> torch.Tensor:
        return self._fr.preprocessor.std

    @property
    def channel(self) -> ChannelType:
        return (
            ChannelType.BGR
            if self._fr.preprocessor.channel == "bgr"
            else ChannelType.RGB
        )

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
        """Delegate to the wrapped ``frbench.FR.forward`` unchanged."""
        return self._fr.forward(
            imgs,
            need_crop=need_crop,
            need_align=need_align,
            keep_largest=keep_largest,
            discard_invalid=discard_invalid,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            loosen_crop=loosen_crop,
            tta=tta,
            l2_normalize=l2_normalize,
            detections=detections,
        )
