"""AdvHat: a curved adversarial sticker on the forehead, fitted per protectee.

Reference Paper: AdvHat: Real-World Adversarial Attack on ArcFace Face ID System (ICPR 2020)
Reference Code: https://github.com/papermsucode/advhat
"""
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import rescale

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    FitData,
    ImageSpec,
    StageInputs,
)
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.utils.log import get_logger

from .geometry import (
    composite_sticker,
    spatial_transform,
    total_variation,
)


logger = get_logger(__name__)


class AdvHat(AFRMethod):
    """A bent and pitched forehead sticker fitted per protectee, worn in every protected photo.

    Args:
        batch_size: Randomly placed renders per optimization step.
        tv_weight: Weight of the total-variation penalty.
        pitch_degrees: Pitch of the sticker, in degrees.
        parabola: Curvature of the sticker's parabolic bend.
        scale: Scale of the sticker on the 600 px face.
        x_offset: Horizontal offset of the sticker, in pixels of the 600 px face.
        y_offset: Vertical offset of the sticker, in pixels of the 600 px face.
        max_steps_per_stage: Step cap per optimization stage (the reference code has none).
        seed: Seed for the random placements; ``None`` leaves them unseeded.
    """

    name = "AdvHat"
    fit_inputs = StageInputs(required=("protectee_album",))
    STICKER_SHAPE = (1, 3, 400, 900)
    FACE_SIZE = 600
    CROP_SIZE = 112

    def __init__(
        self,
        device: torch.device | str,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: "str | os.PathLike[str] | None" = None,
        batch_size: int = 20,
        tv_weight: float = 1e-4,
        pitch_degrees: float = 17.0,
        parabola: float = 0.0013,
        scale: float = 0.465,
        x_offset: float = 0.0,
        y_offset: float = -15.0,
        max_steps_per_stage: int = 5000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        self.batch_size = int(batch_size)
        self.tv_weight = float(tv_weight)
        self.pitch_degrees = float(pitch_degrees)
        self.parabola = float(parabola)
        self.scale = float(scale)
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        self.max_steps_per_stage = int(max_steps_per_stage)
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.univ_mask: Optional[torch.Tensor] = None
        self._maybe_load_run()


    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=self.FACE_SIZE,
            channel=ChannelType.RGB,
            crop_type=CropType.FaceBox,
            align_type=AlignType.ArcFace5,
        )

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the fitted sticker ``univ_mask``; the method owns no priors."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.univ_mask.detach().cpu(), root / "univ_mask.pt")
        return super().save_run(root, files=["univ_mask.pt"])

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the sticker ``univ_mask`` written by :meth:`save_run`."""
        root = super().load_run(load_path)
        path = root / "univ_mask.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No AdvHat sticker under {root}")
        self.univ_mask = (
            torch.load(path, map_location=self.device, weights_only=True)
            .to(self.device, torch.float32)
        )
        self._fitted = True
        self._mark_ready()
        return root

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "batch_size": self.batch_size,
                "tv_weight": self.tv_weight,
                "pitch_degrees": self.pitch_degrees,
                "parabola": self.parabola,
                "scale": self.scale,
                "x_offset": self.x_offset,
                "y_offset": self.y_offset,
                "max_steps_per_stage": self.max_steps_per_stage,
                "seed": self.seed,
            }
        )
        return config

    def fit(self, data: FitData) -> None:
        """Fit the sticker on ``protectee_album[0]`` against ``protectee_album[1]`` as the anchor."""
        self._check_inputs("fit", data)
        self._require_fr()
        self._begin_fitting()
        imgs = data.protectee_album
        fit = imgs[0:1].to(self.device, torch.float32)
        anchor = imgs[1:2].to(self.device, torch.float32)
        fit01 = fit / 255.0
        anchor_crop255 = self._prepare_anchor(anchor)
        anchor_features = [
            self._embed(recognizer, anchor_crop255).detach()
            for recognizer in self.fr_models
        ]

        sticker = torch.full(
            self.STICKER_SHAPE,
            127.0 / 255.0,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        momentum = torch.zeros_like(sticker)
        # As upstream, one cosine trace spans both stages; the stopping slope is
        # fitted to its last 100 entries.
        cosine_trace: list[float] = []

        stages = (
            (1, 1.0 / 51.0, 0.9, 100),
            (2, 1.0 / 255.0, 0.995, 200),
        )
        for stage, step_size, momentum_value, trend_start in stages:
            for stage_step in range(1, self.max_steps_per_stage + 1):
                eot_crops = self._render_eot(fit01, sticker)
                crop_gradient = self._recognition_crop_gradient(
                    eot_crops, anchor_features
                )
                identity_gradient = torch.autograd.grad(
                    eot_crops,
                    sticker,
                    grad_outputs=crop_gradient,
                    retain_graph=False,
                )[0] / float(self.batch_size)
                tv = total_variation(sticker, self.tv_weight)
                tv_gradient = torch.autograd.grad(tv, sticker)[0]

                fixed_crop = self._fixed_crop(fit01, sticker)
                validation = self._validation_similarity(
                    fixed_crop, anchor_features
                )
                cosine_trace.append(validation)

                gradient = identity_gradient + tv_gradient
                momentum = (
                    momentum * momentum_value
                    + gradient * (1.0 - momentum_value)
                )
                sticker = (
                    sticker - step_size * momentum.sign()
                ).clamp(0.0, 1.0).detach().requires_grad_(True)

                slope: Optional[float] = None
                if stage_step > trend_start:
                    recent = np.asarray(cosine_trace[-100:], dtype=np.float64)
                    slope = float(
                        np.polyfit(
                            np.arange(100, dtype=np.float64), recent, 1
                        )[0]
                    )
                if stage_step % 20 == 0:
                    logger.info(
                        "AdvHat stage %d step %d cosine %.5f slope %s",
                        stage,
                        stage_step,
                        validation,
                        "n/a" if slope is None else f"{slope:.3e}",
                    )
                if slope is not None and slope >= 0.0:
                    break
            else:
                # Step cap reached (upstream has none and loops until the slope
                # rule fires): keep the last sticker; stage 1 goes on to stage 2.
                logger.warning(
                    "AdvHat stage %d hit the %d-step cap without a "
                    "non-negative slope; keeping the last sticker%s.",
                    stage,
                    self.max_steps_per_stage,
                    " and continuing to stage 2" if stage == 1 else "",
                )

        self.univ_mask = sticker.detach()
        self._mark_ready()
        logger.info(
            "Fitted AdvHat sticker in %d optimization steps.",
            len(cosine_trace),
        )

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Render the fitted sticker on each aligned 600 px face and paste the change back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        images = torch.stack(crops).to(self.device, torch.float32)
        batch = images.shape[0]
        with torch.no_grad():
            composite, _ = self._compose(
                images / 255.0,
                self.univ_mask.expand(batch, -1, -1, -1),
                self._scalar_batch(self.parabola, batch),
                self._scalar_batch(self.pitch_degrees, batch),
                self._fixed_theta(batch),
            )
        worn = composite.mul(255.0).clamp(0.0, 255.0)
        return paste_deltas(photos, worn - images, plans)

    def _prepare_anchor(self, anchor255: torch.Tensor) -> torch.Tensor:
        # As upstream, the clean anchor is downscaled with skimage (order 5),
        # not with the attack's spatial transformer.
        image = (
            anchor255[0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            / 255.0
        )
        crop = rescale(
            image,
            self.CROP_SIZE / self.FACE_SIZE,
            order=5,
            channel_axis=-1,
            anti_aliasing=True,
            preserve_range=True,
        )
        if crop.shape != (self.CROP_SIZE, self.CROP_SIZE, 3):
            raise RuntimeError(
                f"Unexpected anchor crop shape from skimage: {crop.shape}"
            )
        return (
            torch.from_numpy(np.asarray(crop, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            .mul(255.0)
        )

    def _embed(
        self,
        recognizer: FaceRecognizer,
        crops255: torch.Tensor,
    ) -> torch.Tensor:
        result = recognizer.forward(
            crops255.contiguous(),
            need_crop=False,
            tta=(),
            l2_normalize=True,
        )
        embeddings = result.embeddings
        if embeddings.ndim != 2 or embeddings.shape[0] != crops255.shape[0]:
            raise RuntimeError(
                "FR recognizer returned incompatible embeddings: "
                f"{tuple(embeddings.shape)}"
            )
        return F.normalize(embeddings, dim=1)

    def _recognition_crop_gradient(
        self,
        crops01: torch.Tensor,
        anchor_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        # Upstream takes dL/d(normalized input) as the crop gradient; with inputs normalized as
        # 2 * x / 255 - 1, that is dL/d(x_255) * 127.5.
        gradient = torch.zeros_like(crops01)
        for recognizer, anchor in zip(self.fr_models, anchor_features):
            crops255 = crops01.detach().mul(255.0).requires_grad_(True)
            embeddings = self._embed(recognizer, crops255)
            cosine = (embeddings * anchor.expand_as(embeddings)).sum(dim=1)
            rgb255_gradient = torch.autograd.grad(
                cosine.sum(), crops255
            )[0]
            gradient.add_(rgb255_gradient * 127.5)
        return gradient / float(len(self.fr_models))

    def _validation_similarity(
        self,
        crop01: torch.Tensor,
        anchor_features: Sequence[torch.Tensor],
    ) -> float:
        similarities = []
        with torch.no_grad():
            for recognizer, anchor in zip(
                self.fr_models, anchor_features
            ):
                embedding = self._embed(recognizer, crop01 * 255.0)
                similarities.append(
                    (embedding * anchor.expand_as(embedding)).sum(dim=1).mean()
                )
        return float(torch.stack(similarities).mean())

    def _render_eot(
        self,
        fit01: torch.Tensor,
        sticker: torch.Tensor,
    ) -> torch.Tensor:
        """Render ``batch_size`` randomly placed, bent and pitched stickers as 112x112 crops."""
        batch = self.batch_size
        alpha1 = self._rng.uniform(-1.0, 1.0, (batch, 1)) / 180.0 * np.pi
        scale1 = self._rng.uniform(
            self.scale - 0.02, self.scale + 0.02, (batch, 1)
        )
        jitter = self.FACE_SIZE / self.CROP_SIZE
        y1 = self._rng.uniform(
            self.y_offset - jitter, self.y_offset + jitter, (batch, 1)
        )
        x1 = self._rng.uniform(
            self.x_offset - jitter, self.x_offset + jitter, (batch, 1)
        )
        alpha2 = self._rng.uniform(-1.0, 1.0, (batch, 1)) / 180.0 * np.pi
        scale2 = self._rng.uniform(1.0 / 1.04, 1.04, (batch, 1))
        y2 = self._rng.uniform(-1.0, 1.0, (batch, 1)) / 66.0
        pitch = self._rng.uniform(
            self.pitch_degrees - 2.0,
            self.pitch_degrees + 2.0,
            (batch, 1),
        )
        parabola = self._rng.uniform(
            self.parabola - 0.0002,
            self.parabola + 0.0002,
            (batch, 1),
        )
        theta1 = (1.0 / scale1) * np.hstack(
            (
                np.cos(alpha1),
                np.sin(alpha1),
                -x1 / 450.0,
                -np.sin(alpha1),
                np.cos(alpha1),
                -y1 / 450.0,
            )
        )
        theta2 = scale2 * np.hstack(
            (
                np.cos(alpha2),
                np.sin(alpha2),
                np.zeros((batch, 1)),
                -np.sin(alpha2),
                np.cos(alpha2),
                y2,
            )
        )
        composite, _ = self._compose(
            fit01.expand(batch, -1, -1, -1),
            sticker.expand(batch, -1, -1, -1),
            self._tensor(parabola),
            self._tensor(pitch),
            self._tensor(theta1),
        )
        return spatial_transform(
            composite, self._tensor(theta2), (self.CROP_SIZE, self.CROP_SIZE)
        ).clamp(0.0, 1.0)

    def _fixed_crop(
        self,
        fit01: torch.Tensor,
        sticker: torch.Tensor,
    ) -> torch.Tensor:
        composite, _ = self._compose(
            fit01,
            sticker,
            self._scalar_batch(self.parabola, 1),
            self._scalar_batch(self.pitch_degrees, 1),
            self._fixed_theta(1),
        )
        identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
        )
        return spatial_transform(
            composite, identity, (self.CROP_SIZE, self.CROP_SIZE)
        ).clamp(0.0, 1.0)

    def _compose(
        self,
        faces01: torch.Tensor,
        sticker: torch.Tensor,
        parabola: torch.Tensor,
        pitch_degrees: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return composite_sticker(
            faces01, sticker, parabola, pitch_degrees, theta
        )

    def _fixed_theta(self, batch: int) -> torch.Tensor:
        theta = torch.tensor(
            [
                1.0,
                0.0,
                -self.x_offset / 450.0,
                0.0,
                1.0,
                -self.y_offset / 450.0,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return theta.mul(1.0 / self.scale).repeat(batch, 1)

    def _scalar_batch(self, value: float, batch: int) -> torch.Tensor:
        return torch.full(
            (batch, 1),
            float(value),
            device=self.device,
            dtype=torch.float32,
        )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            array, device=self.device, dtype=torch.float32
        )


__all__ = ["AdvHat"]
