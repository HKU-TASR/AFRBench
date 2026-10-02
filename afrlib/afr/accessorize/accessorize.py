"""Accessorize to a Crime: adversarial eyeglass frames against face recognition.

Reference Paper: Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition (ACM CCS 2016)
Reference Code: https://github.com/mahmoods01/accessorize-to-a-crime
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

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
from afrlib.utils.log import get_logger, tqdm

from .geometry import (
    DEFAULT_SIZE,
    affine_translate_rotate,
    load_arcface_frame_mask,
    load_palette,
    reverse_affine_translate_rotate,
)

logger = get_logger(__name__)

# Solid starting colors; the one that dodges best initializes the frame.
_START_COLORS = (
    (128, 128, 128),  # grey
    (220, 130, 0),    # orangish
    (160, 105, 55),   # brownish
    (200, 175, 30),   # goldish
    (220, 210, 50),   # yellowish
)


class Accessorize(AFRMethod):
    """An eyeglass frame fitted per protectee, worn in every protected photo to dodge recognition.

    Args:
        step_size: Step size of the identity gradient, in 8-bit levels (divided by the
            number of views).
        momentum_coeff: Momentum of the per-view updates.
        max_iter: Optimization iterations.
        lambda_tv: Weight of the total-variation step, in 8-bit levels.
        lambda_nps: Step size of the non-printability term, in 8-bit levels.
        movement: Largest random shift of the frame per view, in pixels.
        rot_range: Largest random rotation of the frame per view, in degrees.
        num_views: Maximum number of album images used for fitting.
        num_palette_colors: Printable colors in the non-printability palette.
        seed: Seed for view sampling and frame movement; ``None`` leaves them unseeded.
    """

    name = "Accessorize"
    fit_inputs = StageInputs(required=("protectee_album",))
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: "str | os.PathLike[str] | None" = None,
        step_size: float = 20.0,
        momentum_coeff: float = 0.4,
        max_iter: int = 300,
        lambda_tv: float = 3.0,
        lambda_nps: float = 5.0,
        movement: int = 4,
        rot_range: float = 4.0,
        num_views: int = 30,
        num_palette_colors: int = 30,
        seed: Optional[int] = 0,
    ):
        super().__init__(device, fr_models, assets_path=assets_path)

        self.step_size = step_size
        self.momentum_coeff = momentum_coeff
        self.max_iter = max_iter
        self.lambda_tv = lambda_tv
        self.lambda_nps = lambda_nps
        self.movement = movement
        self.rot_range = float(rot_range)
        self.num_views = num_views
        self.num_palette_colors = num_palette_colors
        self.seed = seed
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

        self.mask = load_arcface_frame_mask(self.device, DEFAULT_SIZE)
        self.palette = load_palette(num_palette_colors).to(self.device)
        # The fitted frame, (1, 3, 224, 224) in [0, 1].
        self.univ_mask: Optional[torch.Tensor] = None
        self._maybe_load_run()


    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=DEFAULT_SIZE,
            channel=ChannelType.RGB,
            crop_type=CropType.FaceBox,
            align_type=AlignType.ArcFace5,
        )

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the fitted frame ``univ_mask``; the silhouette and palette ship with the package."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.univ_mask.detach().cpu(), root / "univ_mask.pt")
        return super().save_run(root, files=["univ_mask.pt"])

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the eyeglass frame ``univ_mask`` written by :meth:`save_run`."""
        root = super().load_run(load_path)
        path = root / "univ_mask.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No Accessorize frame under {root}")
        self.univ_mask = torch.load(
            path, map_location=self.device, weights_only=True
        ).to(self.device, torch.float32)
        self._fitted = True
        self._mark_ready()
        return root

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "step_size": self.step_size,
            "momentum_coeff": self.momentum_coeff,
            "max_iter": self.max_iter,
            "lambda_tv": self.lambda_tv,
            "lambda_nps": self.lambda_nps,
            "movement": self.movement,
            "rot_range": self.rot_range,
            "num_views": self.num_views,
            "num_palette_colors": self.num_palette_colors,
            "seed": self.seed,
        })
        return config

    def fit(self, data: FitData) -> None:
        """Fit the universal dodging frame from the protectee's aligned images.

        Args:
            data.protectee_album: ArcFace-aligned 224x224 RGB images of the protectee in
                ``[0, 255]``, shape ``(N, 3, 224, 224)``, ``float32``.
        """
        self._check_inputs("fit", data)
        self._require_fr()
        self._begin_fitting()
        imgs = data.protectee_album
        views255 = imgs.to(self.device, dtype=torch.float32)
        views01 = views255 / 255.0

        n_avail = views01.shape[0]
        n_use = min(self.num_views, n_avail)
        if n_use < n_avail:
            idx = torch.randperm(n_avail, generator=self._generator)[:n_use]
            views01 = views01[idx.to(views01.device)]

        # Clean per-view embeddings are the reference each composite is pushed away from.
        with torch.no_grad():
            clean_feats = self._embed(views01)

        frame = self._best_solid_start(views01, clean_feats)
        n = views01.shape[0]
        r_momentum = [torch.zeros_like(frame) for _ in range(n)]

        for _ in tqdm(range(self.max_iter), desc="Accessorize/fit", leave=False):
            agg = torch.zeros_like(frame)
            for i in range(n):
                dx, dy, angle = self._sample_movement()
                # Shifted-in regions are white, as in the reference code.
                moved_frame = affine_translate_rotate(frame, dx, dy, angle, fill=1.0)
                moved_mask = affine_translate_rotate(self.mask, dx, dy, angle)
                moved_mask = (moved_mask > 0.5).float()
                moved_frame = moved_frame.detach().requires_grad_(True)
                composite = views01[i:i + 1] * (1 - moved_mask) + moved_frame * moved_mask
                id_loss = self._identity_loss(composite, [f[i:i + 1] for f in clean_feats])
                grad = torch.autograd.grad(id_loss, moved_frame)[0] * moved_mask
                id_dir = -grad / (grad.abs().max() + 1e-12)

                tv_dir = torch.zeros_like(id_dir)
                if self.lambda_tv > 0:
                    mf = moved_frame.detach().requires_grad_(True)
                    tv = self._total_variation(
                        views01[i:i + 1] * (1 - moved_mask) + mf * moved_mask
                    )
                    tv_grad = torch.autograd.grad(tv, mf)[0] * moved_mask
                    tv_dir = tv_grad / (tv_grad.abs().max() + 1e-12)

                r = (self.step_size * id_dir - self.lambda_tv * tv_dir) / (255.0 * n)
                r = reverse_affine_translate_rotate(r, dx, dy, angle)
                r = r * self.mask
                r_momentum[i] = self.momentum_coeff * r_momentum[i] + r
                agg = agg + r_momentum[i]

            if self.lambda_nps > 0:
                frame = self._apply_nps_step(frame)

            trial = frame + agg
            agg = agg.masked_fill((trial > 1.0) | (trial < 0.0), 0.0)
            frame = frame + agg
            frame = torch.floor(frame * 255.0) / 255.0
            frame = torch.clamp(frame, 0.0, 1.0)

        self.univ_mask = frame.detach()
        self._mark_ready()
        logger.info("Fitted the Accessorize frame on %d views.", n)

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Composite the fitted frame onto each aligned crop and paste the change back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        imgs255 = torch.stack(crops).to(self.device, dtype=torch.float32)
        imgs01 = imgs255 / 255.0
        frame = self.univ_mask
        if frame.shape[0] == 1 and imgs01.shape[0] > 1:
            frame = frame.expand(imgs01.shape[0], -1, -1, -1)
        with torch.no_grad():
            out = imgs01 * (1 - self.mask) + frame * self.mask
        worn = (out * 255.0).clamp(0.0, 255.0)
        return paste_deltas(photos, worn - imgs255, plans)

    def _best_solid_start(
        self, views01: torch.Tensor, clean_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """Pick the solid starting color with the lowest clean-identity similarity."""
        colors = torch.tensor(_START_COLORS, dtype=torch.float32, device=self.device) / 255.0
        best_frame, best_loss = None, float("inf")
        with torch.no_grad():
            for color in colors:
                frame = color.view(1, 3, 1, 1) * self.mask
                comps = views01 * (1 - self.mask) + frame
                loss = self._identity_loss(comps, clean_feats).item()
                if loss < best_loss:
                    best_loss, best_frame = loss, frame.contiguous()
        return best_frame

    def _identity_loss(
        self, composite: torch.Tensor, clean_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """Mean cosine similarity to the clean embeddings (minimized to dodge)."""
        feats = self._embed(composite)
        losses = [
            F.cosine_similarity(a, t.expand_as(a), dim=1).mean()
            for a, t in zip(feats, clean_feats)
        ]
        return torch.stack(losses).mean()

    def _sample_movement(self) -> Tuple[float, float, float]:
        if self.movement <= 0 and self.rot_range <= 0:
            return 0.0, 0.0, 0.0

        def _sample_shift(rng: float) -> float:
            if rng <= 0:
                return 0.0
            u = torch.rand((), generator=self._generator).item()
            return float(round((2 * u - 1) * rng))

        dx = _sample_shift(float(self.movement))
        dy = _sample_shift(float(self.movement))
        if self.rot_range <= 0:
            angle = 0.0
        else:
            u = torch.rand((), generator=self._generator).item()
            angle = (2 * u - 1) * self.rot_range
        return dx, dy, angle

    def _apply_nps_step(self, frame: torch.Tensor) -> torch.Tensor:
        f = (frame * self.mask).detach().requires_grad_(True)
        nps = self._non_printability_score(f)
        grad = torch.autograd.grad(nps, f)[0] * self.mask
        nps_dir = -grad / (grad.abs().max() + 1e-12)
        step = self.lambda_nps / 255.0 * nps_dir
        trial = frame + step
        step = step.masked_fill((trial > 1.0) | (trial < 0.0), 0.0)
        return torch.clamp(frame + step, 0.0, 1.0)

    def _non_printability_score(self, frame: torch.Tensor) -> torch.Tensor:
        max_norm = 3 * (80.0 / 255.0) ** 2
        pixels = frame.reshape(3, -1).t()
        mask_flat = self.mask.reshape(3, -1)[0] > 0
        pixels = pixels[mask_flat]
        dist2 = ((pixels[:, None, :] - self.palette[None, :, :]) ** 2).sum(-1)
        return (dist2 / max_norm).prod(dim=1).sum()

    @staticmethod
    def _total_variation(img: torch.Tensor) -> torch.Tensor:
        dw = torch.zeros_like(img)
        dh = torch.zeros_like(img)
        dw[:, :, :, :-1] = img[:, :, :, 1:] - img[:, :, :, :-1]
        dh[:, :, :-1, :] = img[:, :, 1:, :] - img[:, :, :-1, :]
        return torch.sqrt(dw ** 2 + dh ** 2 + 1e-8).sum()

    def _embed(self, imgs01: torch.Tensor) -> List[torch.Tensor]:
        imgs255 = imgs01.clamp(0.0, 1.0) * 255.0
        return [
            fr.forward(imgs255, need_crop=False, tta=(), l2_normalize=True).embeddings
            for fr in self.fr_models
        ]
