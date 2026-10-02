"""Protego: a per-user privacy protection texture on the FLAME UV map.

Reference Paper: Protego: User-Centric Pose-Invariant Privacy Protection Against Face Recognition-Induced Digital Footprint Exposure (CVPR 2026)
Reference Code: https://github.com/HKU-TASR/Protego
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
from afrlib.utils.log import tqdm, get_logger, suppress_output
from .uv_mapping import UVGenerator

logger = get_logger(__name__)


class Protego(AFRMethod):
    """One texture per protectee on the canonical FLAME UV map, sampled through each face's
    estimated UV map so the perturbation follows the pose.

    Args:
        epochs: Fitting epochs over the album.
        batch_size: Album images per update.
        learning_rate: Step size of the signed-gradient update, on the ``[0, 1]`` scale.
        epsilon: L-inf budget of the texture, on the ``[0, 1]`` scale.
        min_ssim: SSIM the protected album images must keep.
        uv_batch_size: Images per UV-estimation batch.
    """

    name = "Protego"
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
            uv_batch_size: int = 8,
            ):
        super().__init__(device, fr_models, assets_path=assets_path)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.min_ssim = min_ssim
        self.uv_batch_size = uv_batch_size

        # SMIRK + FLAME + renderer UV pipeline; its build-time prints go to the DEBUG log.
        with suppress_output(logger):
            self.uv_generator = UVGenerator(device=self.device)

        # Fixed seed for the random texture initialization.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(114)

        # The fitted PPT, (1, 3, 224, 224) in [-epsilon, epsilon].
        self.univ_mask = None
        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=224, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.NoAlign)

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the fitted PPT ``univ_mask``; the UV-mapping priors are not saved."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.univ_mask.detach().cpu(), root / "univ_mask.pt")
        return super().save_run(root, files=["univ_mask.pt"])

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the PPT ``univ_mask`` written by :meth:`save_run`."""
        root = super().load_run(load_path)
        path = root / "univ_mask.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No Protego PPT under {root}")
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
            "uv_batch_size": self.uv_batch_size,
        })
        return config

    def fit(self, data: FitData) -> None:
        """Fit the protectee's PPT from their facial images.

        Args:
            data.protectee_album: Images of the same person following ``spec_for``: RGB in
                ``[0, 255]``, shape ``(N, 3, 224, 224)``, ``float32``.
        """
        self._check_inputs("fit", data)
        self._require_fr()
        self._begin_fitting()
        imgs = data.protectee_album
        orig_imgs = imgs.to(self.device) / 255.  # internal range [0, 1]

        # UV maps depend only on the (fixed) input images, so compute them once.
        with torch.no_grad():
            uvs, bin_masks = self.uv_generator.forward(orig_imgs, batch_size=self.uv_batch_size)

        self.generator.manual_seed(114)
        univ_mask = (torch.rand(size=(1, 3, 224, 224), device=self.device, generator=self.generator,
                                dtype=torch.float32) - 0.5) * self.epsilon / 0.5
        constant = 1e-4         # numerical stability of the Gram matrix log-det
        percep_loss_weight = 1.  # lambda_SSIM, dynamically scheduled below

        pbar = tqdm(range(self.epochs), desc="Protego training")
        for epoch in pbar:
            batch_losses, batch_hyper_losses, batch_feature_losses, batch_percep_losses = [], [], [], []
            for start in range(0, orig_imgs.shape[0], self.batch_size):
                orig_faces = orig_imgs[start:start + self.batch_size]
                img_num = orig_faces.shape[0]

                # Deform the PPT to each face's pose/expression and apply it to the face region.
                textures = univ_mask.clone().repeat(img_num, 1, 1, 1).requires_grad_(True)
                perturbations = torch.clamp(
                    F.grid_sample(input=textures, grid=uvs[start:start + img_num], mode='bilinear', align_corners=True),
                    -self.epsilon, self.epsilon)
                protected_faces = torch.clamp(orig_faces + perturbations * bin_masks[start:start + img_num], 0, 1)

                percep_loss = calc_ssim(protected_faces, orig_faces, data_range=1., size_average=True)
                percep_term = percep_loss_weight * torch.max(self.min_ssim - percep_loss,
                                                             torch.tensor(0., device=self.device))

                ensemble_losses, _hyper_loss, _feature_loss = [], 0., 0.
                for fr in self.fr_models:
                    protected_features = self._embed_one(fr, protected_faces)
                    orig_features = self._embed_one(fr, orig_faces)
                    normalized_protected_features = F.normalize(protected_features, p=2, dim=1)
                    # Hypersensitivity: maximize the volume spanned by protected features.
                    gram_matrix = torch.matmul(normalized_protected_features, normalized_protected_features.T)
                    regularized_gram_matrix = gram_matrix + constant * torch.eye(gram_matrix.shape[0], device=gram_matrix.device)
                    hyper_loss = -torch.log(torch.linalg.det(regularized_gram_matrix) + constant)
                    # Feature deviation: penalize similarity between protected and original features.
                    feature_loss = F.cosine_similarity(orig_features, protected_features).mean()

                    ensemble_losses.append(hyper_loss + feature_loss + percep_term)
                    _hyper_loss += hyper_loss.detach()
                    _feature_loss += feature_loss.detach()

                loss = torch.mean(torch.stack(ensemble_losses, dim=0), dim=0)
                loss.backward()

                # PGD-style update with the batch-averaged signed gradient.
                grad_avg = torch.mean(textures.grad.detach(), dim=0, keepdim=True)
                univ_mask = torch.clamp(univ_mask - self.learning_rate * torch.sign(grad_avg),
                                        -self.epsilon, self.epsilon)
                textures.grad = None

                batch_losses.append(loss.item())
                batch_hyper_losses.append(_hyper_loss.item() / len(self.fr_models))
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
                f"Protego training | Loss: {sum(batch_losses)/len(batch_losses):.4f} | "
                f"Hyper: {sum(batch_hyper_losses)/len(batch_hyper_losses):.4f} | "
                f"Feature: {sum(batch_feature_losses)/len(batch_feature_losses):.4f} | "
                f"SSIM: {sum(batch_percep_losses)/len(batch_percep_losses):.4f}")

        self.univ_mask = univ_mask.detach()
        self._mark_ready()
        logger.info("Trained the PPT for %d epochs on %d images.", self.epochs, orig_imgs.shape[0])

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Sample the fitted texture through each face's UV map and paste the change back; the
        perturbation stays inside the visible face.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device)

        protected = []
        with torch.no_grad():
            for start in range(0, clean.shape[0], self.uv_batch_size):
                orig_faces = clean[start:start + self.uv_batch_size] / 255.
                uvs, bin_masks = self.uv_generator.forward(orig_faces)
                perturbations = torch.clamp(
                    F.grid_sample(self.univ_mask.repeat(orig_faces.shape[0], 1, 1, 1), uvs,
                                  mode='bilinear', align_corners=True),
                    -self.epsilon, self.epsilon)
                protected.append(torch.clamp(orig_faces + perturbations * bin_masks, 0, 1) * 255.)
        return paste_deltas(photos, torch.cat(protected, dim=0) - clean, plans)

    def _embed_one(self, fr: FaceRecognizer, imgs01: torch.Tensor) -> torch.Tensor:
        """Embed ``[0, 1]`` RGB faces with one FR model (pre-cropped, no TTA, raw features)."""
        return fr.forward(imgs01 * 255., need_crop=False, tta=(), l2_normalize=False).embeddings
