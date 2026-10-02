"""Chameleon: a personalized universal privacy protection mask (P3-Mask).

Reference Paper: Personalized Privacy Protection Mask Against Unauthorized Facial Recognition (ECCV 2024)
Reference Code: https://github.com/git-disl/Chameleon
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as calc_ssim

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
from afrlib.fr import FaceRecognizer
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.utils.log import tqdm, get_logger

logger = get_logger(__name__)


class Chameleon(AFRMethod):
    """One universal mask per protectee, fitted to lower the surrogates' cosine similarity while
    an SSIM hinge keeps it above ``min_ssim``.

    Args:
        epochs: Fitting epochs over the album.
        batch_size: Album images per update.
        learning_rate: Step size of the signed-gradient update, on the ``[0, 1]`` scale.
        epsilon: L-inf budget of the mask, on the ``[0, 1]`` scale.
        min_ssim: SSIM the protected album images must keep.
    """

    name = "Chameleon"
    fit_inputs = StageInputs(required=("protectee_album",))
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            assets_path: "str | os.PathLike[str] | None" = None,
            epochs: int = 100,
            batch_size: int = 4,
            learning_rate: float = 0.01 * (16 / 255.),
            epsilon: float = 16 / 255.,
            min_ssim: float = 0.95,
            ):
        super().__init__(device, fr_models, assets_path=assets_path)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_ssim = min_ssim

        # Fixed mask init seed (mask_random_seed = 114).
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(114)

        # The fitted P3-Mask, (1, 3, 224, 224) in [-epsilon, epsilon].
        self.univ_mask = None
        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=224, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.NoAlign)

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the fitted P3-Mask ``univ_mask`` (Chameleon has no priors)."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.univ_mask.detach().cpu(), root / "univ_mask.pt")
        return super().save_run(root, files=["univ_mask.pt"])

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the P3-Mask ``univ_mask`` written by :meth:`save_run`."""
        root = super().load_run(load_path)
        path = root / "univ_mask.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No Chameleon mask under {root}")
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
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "min_ssim": self.min_ssim,
        })
        return config

    def fit(self, data: FitData) -> None:
        """Fit the protectee's P3-Mask from their facial images.

        Args:
            data.protectee_album: Images of the same person following ``spec_for``: RGB in
                ``[0, 255]``, shape ``(N, 3, 224, 224)``, ``float32``.
        """
        self._check_inputs("fit", data)
        self._require_fr()
        self._begin_fitting()
        imgs = data.protectee_album
        orig_imgs = imgs.to(self.device) / 255.  # internal range [0, 1]

        self.generator.manual_seed(114)
        univ_mask = (torch.rand(size=(1, 3, 224, 224), device=self.device, generator=self.generator,
                                dtype=torch.float32) - 0.5) * self.epsilon / 0.5
        percep_loss_weight = 1.  # lambda_SSIM, dynamically scheduled below

        pbar = tqdm(range(self.epochs), desc="Chameleon training")
        for epoch in pbar:
            batch_losses, batch_feature_losses, batch_percep_losses = [], [], []
            for start in range(0, orig_imgs.shape[0], self.batch_size):
                orig_faces = orig_imgs[start:start + self.batch_size]
                img_num = orig_faces.shape[0]

                textures = univ_mask.clone().repeat(img_num, 1, 1, 1).requires_grad_(True)
                protected_faces = torch.clamp(orig_faces + textures, 0, 1)

                percep_loss = calc_ssim(protected_faces, orig_faces, data_range=1., size_average=True)
                percep_term = percep_loss_weight * torch.max(self.min_ssim - percep_loss,
                                                             torch.tensor(0., device=self.device))

                ensemble_losses, _feature_loss = [], 0.
                for fr in self.fr_models:
                    protected_features = self._embed_one(fr, protected_faces)
                    orig_features = self._embed_one(fr, orig_faces)
                    feature_loss = F.cosine_similarity(orig_features, protected_features).mean()
                    ensemble_losses.append(feature_loss + percep_term)
                    _feature_loss += feature_loss.detach()

                loss = torch.mean(torch.stack(ensemble_losses, dim=0), dim=0)
                loss.backward()

                # PGD-style update with the batch-averaged signed gradient.
                grad_avg = torch.mean(textures.grad.detach(), dim=0, keepdim=True)
                univ_mask = torch.clamp(univ_mask - self.learning_rate * torch.sign(grad_avg),
                                        -self.epsilon, self.epsilon)
                textures.grad = None

                batch_losses.append(loss.item())
                batch_feature_losses.append(_feature_loss.item() / len(self.fr_models))
                batch_percep_losses.append(percep_loss.item())

                # Reschedule lambda_SSIM: halve it when SSIM is 1% above the target,
                # double it when 1% below, and reset it to 1 inside that band.
                if batch_percep_losses[-1] >= self.min_ssim * 1.01 and percep_loss_weight > 1. / 129:
                    percep_loss_weight /= 2
                elif batch_percep_losses[-1] <= self.min_ssim * 0.99 and percep_loss_weight <= 129.:
                    percep_loss_weight *= 2
                elif self.min_ssim * 0.99 < batch_percep_losses[-1] < self.min_ssim * 1.01:
                    percep_loss_weight = 1.

            pbar.set_description(
                f"Chameleon training | Loss: {sum(batch_losses)/len(batch_losses):.4f} | "
                f"Feature: {sum(batch_feature_losses)/len(batch_feature_losses):.4f} | "
                f"SSIM: {sum(batch_percep_losses)/len(batch_percep_losses):.4f}")

        self.univ_mask = univ_mask.detach()
        self._mark_ready()
        logger.info("Trained the P3-Mask for %d epochs on %d images.", self.epochs, orig_imgs.shape[0])

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Add the fitted mask to each 224 px crop and paste the change back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device)
        with torch.no_grad():
            masked = torch.clamp(clean / 255. + self.univ_mask, 0, 1) * 255.
        return paste_deltas(photos, masked - clean, plans)

    def _embed_one(self, fr: FaceRecognizer, imgs01: torch.Tensor) -> torch.Tensor:
        """Embed ``[0, 1]`` RGB faces with one FR model: detection, alignment and flip TTA, not
        L2-normalized.
        """
        return fr.forward(imgs01 * 255., need_crop=True, need_align=True, tta=('flip_horizontal', ), l2_normalize=False).embeddings
