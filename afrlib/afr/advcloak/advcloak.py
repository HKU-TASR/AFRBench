"""AdvCloak: a two-stage GAN that generates a person-specific adversarial cloak.

Reference Paper: AdvCloak: Customized Adversarial Cloak for Privacy Protection (Pattern Recognition 2025)
Reference Code: https://github.com/liuxuannan/AdvCloak (CC BY-SA 4.0, adapted; see LICENSE)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    FitData,
    ImageSpec,
    StageInputs,
    TrainData,
)
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.afr.datasets import dataset_config
from afrlib.afr.opom.opom import ConvexHullLoss
from afrlib.fr import FaceRecognizer
from afrlib.utils.log import get_logger, tqdm

from .losses import (
    discriminator_loss,
    feature_similarity_loss,
    generator_gan_loss,
    norm_constraint_loss,
)
from .models import Discriminator, Generator

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

logger = get_logger(__name__)

# The generator's tanh output perturbs images in [0, 1]; the reference inference
# scales it to pixel units by this factor.
_MASK_TO_PIXEL = 256.0


class AdvCloak(AFRMethod):
    """A GAN-generated cloak, averaged into one mask per protectee and added to every photo.

    Args:
        stage1_epochs: Epochs of stage 1 (one mask per image).
        stage2_epochs: Epochs of stage 2 (one mask per person).
        stage1_batch_size: Batch size of stage 1.
        learning_rate: Adam learning rate of stage 1.
        stage2_learning_rate: Adam learning rate of stage 2.
        beta1: Adam beta1.
        beta2: Adam beta2.
        lambda_adv_stage1: Weight of the adversarial loss in stage 1.
        lambda_adv_stage2: Weight of the adversarial loss in stage 2.
        lambda_norm: Weight of the perturbation-norm penalty.
        norm_cap: L2 norm, on the ``[0, 1]`` scale, below which the penalty is constant.
        num_shot: Identities per stage-2 batch.
        images_per_subject: Images per identity in stage 2.
        subspace: ``"none"`` (cosine similarity to each clean feature) or ``"convex_hull"``
            (OPOM's convex-hull loss) for the stage-2 adversarial loss.
        hull_warmup_epochs: Stage-2 epochs that use the class center before the hull.
        hull_upper_bound: Upper bound of the convex-hull coefficients.
        hull_lower_bound: Lower bound of the convex-hull coefficients.
        mask_l2_cap: L2 cap of the fitted mask, in pixel units.
    """

    name = "AdvCloak"
    training_corpus = "casia_webface"
    train_inputs = StageInputs(required=("corpus",))
    fit_inputs = StageInputs(required=("protectee_album",))
    eval_inputs = StageInputs(required=("eval_imgs",))
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: "str | os.PathLike[str] | None" = None,
        stage1_epochs: int = 100,
        stage2_epochs: int = 100,
        stage1_batch_size: int = 32,
        learning_rate: float = 1e-4,
        stage2_learning_rate: float = 2e-8,
        beta1: float = 0.5,
        beta2: float = 0.9,
        lambda_adv_stage1: float = 10.0,
        lambda_adv_stage2: float = 15.0,
        lambda_norm: float = 1.0,
        norm_cap: float = 3.0,
        num_shot: int = 2,
        images_per_subject: int = 16,
        subspace: str = "none",
        hull_warmup_epochs: int = 40,
        hull_upper_bound: float = 1.0,
        hull_lower_bound: float = 0.0,
        mask_l2_cap: float = 1200.0,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        self.stage1_epochs = int(stage1_epochs)
        self.stage2_epochs = int(stage2_epochs)
        if stage1_batch_size < 1:
            raise ValueError("stage1_batch_size must be positive")
        self.stage1_batch_size = int(stage1_batch_size)
        self.learning_rate = float(learning_rate)
        self.stage2_learning_rate = float(stage2_learning_rate)
        self.betas = (float(beta1), float(beta2))
        self.lambda_adv_stage1 = float(lambda_adv_stage1)
        self.lambda_adv_stage2 = float(lambda_adv_stage2)
        self.lambda_norm = float(lambda_norm)
        self.norm_cap = float(norm_cap)
        self.num_shot = int(num_shot)
        self.images_per_subject = int(images_per_subject)
        if subspace not in ("none", "convex_hull"):
            raise ValueError(f"subspace must be 'none' or 'convex_hull', got {subspace!r}")
        self.subspace = subspace
        self.hull_warmup_epochs = int(hull_warmup_epochs)
        self.hull_upper_bound = float(hull_upper_bound)
        self.hull_lower_bound = float(hull_lower_bound)
        self.mask_l2_cap = float(mask_l2_cap)

        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)
        self._hull_loss = ConvexHullLoss().to(self.device) if subspace == "convex_hull" else None
        self._train_dataset_config: Optional[Dict[str, Any]] = None
        # Baked by fit(): this protectee's One-Person-One-Mask, in pixels.
        self.person_mask: Optional[torch.Tensor] = None
        self._maybe_load_run()


    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.ArcFace5)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration plus fixed settings and dataset provenance."""
        config = super().get_config()
        config.update(
            {
                "stage1_epochs": self.stage1_epochs,
                "stage2_epochs": self.stage2_epochs,
                "stage1_batch_size": self.stage1_batch_size,
                "learning_rate": self.learning_rate,
                "stage2_learning_rate": self.stage2_learning_rate,
                "beta1": self.betas[0],
                "beta2": self.betas[1],
                "lambda_adv_stage1": self.lambda_adv_stage1,
                "lambda_adv_stage2": self.lambda_adv_stage2,
                "lambda_norm": self.lambda_norm,
                "norm_cap": self.norm_cap,
                "num_shot": self.num_shot,
                "images_per_subject": self.images_per_subject,
                "subspace": self.subspace,
                "hull_warmup_epochs": self.hull_warmup_epochs,
                "hull_upper_bound": self.hull_upper_bound,
                "hull_lower_bound": self.hull_lower_bound,
                "mask_l2_cap": self.mask_l2_cap,
                # Fixed settings and provenance, not constructor arguments.
                "image_size": 112,
                "external_range": [0.0, 255.0],
                "internal_range": [-1.0, 1.0],
                "training_dataset": self._train_dataset_config,
            }
        )
        return config

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the generator, the discriminator, the fitted mask (after :meth:`fit`) and
        ``config.json``.
        """
        self._require_savable()
        root = Path(save_path)
        root.mkdir(parents=True, exist_ok=True)
        weights = []
        tensors = [(self.netG.state_dict(), "netG"), (self.netD.state_dict(), "netD")]
        if self.person_mask is not None:
            tensors.append(({"person_mask": self.person_mask}, "person_mask"))
        for state, name in tensors:
            destination = root / f"{name}.pth"
            temporary = root / f".{name}.pth.tmp"
            torch.save({k: v.detach().cpu() for k, v in state.items()}, temporary)
            os.replace(temporary, destination)
            weights.append(destination.name)
        return super().save_run(root, weights=weights)

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the generator, and the discriminator and fitted mask if saved.

        Args:
            load_path: A :meth:`save_run` folder, or a generator ``.pth`` file such as the
                released ``netG_convex_hull.pth``.
        """
        root = super().load_run(load_path)
        path = Path(load_path).expanduser()
        generator_path = path if path.suffix == ".pth" and path.is_file() else root / "netG.pth"
        if not generator_path.is_file():
            raise FileNotFoundError(f"No AdvCloak generator found at {generator_path}")
        self.netG.load_state_dict(
            torch.load(generator_path, map_location=self.device, weights_only=True), strict=True
        )
        self.netG.eval().requires_grad_(False)
        discriminator_path = root / "netD.pth"
        if discriminator_path.is_file():
            self.netD.load_state_dict(
                torch.load(discriminator_path, map_location=self.device, weights_only=True), strict=True
            )
            self.netD.eval().requires_grad_(False)
        mask_path = root / "person_mask.pth"
        if mask_path.is_file():
            self.person_mask = torch.load(
                mask_path, map_location=self.device, weights_only=True
            )["person_mask"]
            self._fitted = True
        self._trained = True
        # The mask is per protectee: a run without one still needs fit().
        if self.person_mask is not None:
            self._mark_ready()
        return root

    def train(self, data: TrainData) -> None:
        """Run stage 1 then stage 2, leaving a trained generator in ``self.netG``.

        Args:
            data.corpus: A labelled face corpus, read flat by stage 1 and grouped by
                identity by stage 2.
        """
        self._check_inputs("train", data)
        self._require_fr()
        stage1_loader, stage2_loader = self._build_loaders(data.corpus)
        self._begin_training()
        self._train_dataset_config = {
            "stage1": dataset_config(stage1_loader),
            "stage2": dataset_config(stage2_loader),
        }
        self._train_stage1(stage1_loader)
        self._train_stage2(stage2_loader)
        self.netG.eval().requires_grad_(False)
        self.netD.eval().requires_grad_(False)
        logger.info(
            "Trained AdvCloak for %d + %d epochs (subspace=%s).",
            self.stage1_epochs,
            self.stage2_epochs,
            self.subspace,
        )

    def _build_loaders(
        self, corpus: "TrainingCorpus"
    ) -> tuple[DataLoader, DataLoader]:
        """The flat stage-1 and identity-grouped stage-2 loaders; stage 2 is not shuffled, so
        each batch holds whole identity blocks.
        """
        from .data import AdvCloakStage1Dataset, AdvCloakStage2Dataset

        stage1 = DataLoader(
            AdvCloakStage1Dataset.from_corpus(corpus),
            batch_size=self.stage1_batch_size,
            shuffle=True,
            drop_last=True,
        )
        stage2 = DataLoader(
            AdvCloakStage2Dataset.from_corpus(
                corpus, images_per_subject=self.images_per_subject
            ),
            batch_size=self.num_shot * self.images_per_subject,
            shuffle=False,
            drop_last=True,
        )
        for loader, stage in ((stage1, "stage 1"), (stage2, "stage 2")):
            if len(loader) == 0:
                raise ValueError(
                    f"Corpus {corpus.name!r} has too few samples for AdvCloak "
                    f"{stage} after filtering"
                )
        return stage1, stage2

    def _train_stage1(self, loader: DataLoader) -> None:
        """One Image One Mask: per-image perturbations, ensemble-mean cosine adv loss."""
        netG, netD = self.netG.train().requires_grad_(True), self.netD.train().requires_grad_(True)
        netG.apply(self._weights_init)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=self.learning_rate, betas=self.betas)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=self.learning_rate, betas=self.betas)

        for epoch in range(1, self.stage1_epochs + 1):
            progress = tqdm(loader, desc=f"AdvCloak stage 1 epoch {epoch}/{self.stage1_epochs}")
            for images in progress:
                images = images.to(self.device, torch.float32).div(127.5).sub(1.0)

                # Discriminator step.
                optimizer_D.zero_grad(set_to_none=True)
                perturbation = netG(images)
                adv_images = 2 * torch.clamp(perturbation + (images + 1.0) / 2.0, 0, 1) - 1
                loss_D_real = discriminator_loss(netD(images), real=True)
                loss_D_real.backward()
                loss_D_fake = discriminator_loss(netD(adv_images.detach()), real=False)
                loss_D_fake.backward()
                optimizer_D.step()

                # Generator step.
                optimizer_G.zero_grad(set_to_none=True)
                loss_G_gan = generator_gan_loss(netD(adv_images))
                loss_G_gan.backward(retain_graph=True)
                loss_norm = norm_constraint_loss(perturbation, self.norm_cap)
                clean_feats = [fr.embed_crops((images + 1.0) / 2.0, input_space="01") for fr in self.fr_models]
                adv_feats = [fr.embed_crops((adv_images + 1.0) / 2.0, input_space="01") for fr in self.fr_models]
                loss_adv = torch.mean(
                    torch.stack([feature_similarity_loss(c, a) for c, a in zip(clean_feats, adv_feats)])
                )
                loss_G = self.lambda_adv_stage1 * loss_adv + self.lambda_norm * loss_norm
                loss_G.backward()
                optimizer_G.step()
                progress.set_postfix(
                    D=f"{float(loss_D_real.detach() + loss_D_fake.detach()):.3f}",
                    adv=f"{float(loss_adv.detach()):.3f}",
                )

    def _train_stage2(self, loader: DataLoader) -> None:
        """One Person One Mask: aggregate each identity's perturbations into a shared mask."""
        netG, netD = self.netG.train().requires_grad_(True), self.netD.train().requires_grad_(True)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=self.stage2_learning_rate, betas=self.betas)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=self.stage2_learning_rate, betas=self.betas)
        n = self.images_per_subject

        for epoch in range(1, self.stage2_epochs + 1):
            progress = tqdm(loader, desc=f"AdvCloak stage 2 epoch {epoch}/{self.stage2_epochs}")
            for images in progress:
                images = images.to(self.device, torch.float32).div(127.5).sub(1.0)

                # Discriminator step.
                optimizer_D.zero_grad(set_to_none=True)
                perturbation = netG(images)
                masks = torch.stack(
                    [perturbation[s * n:(s + 1) * n].mean(dim=0) for s in range(self.num_shot)]
                )
                adv_images = torch.cat(
                    [
                        2 * torch.clamp(masks[s] + (images[s * n:(s + 1) * n] + 1.0) / 2.0, 0, 1) - 1
                        for s in range(self.num_shot)
                    ]
                )
                loss_D_real = discriminator_loss(netD(images), real=True)
                loss_D_real.backward()
                loss_D_fake = discriminator_loss(netD(adv_images.detach()), real=False)
                loss_D_fake.backward()
                optimizer_D.step()

                # Generator step.
                optimizer_G.zero_grad(set_to_none=True)
                loss_G_gan = generator_gan_loss(netD(adv_images))
                loss_G_gan.backward(retain_graph=True)
                loss_norm = norm_constraint_loss(masks, self.norm_cap)
                clean_feats = [fr.embed_crops((images + 1.0) / 2.0, input_space="01") for fr in self.fr_models]
                adv_feats = [fr.embed_crops((adv_images + 1.0) / 2.0, input_space="01") for fr in self.fr_models]
                loss_adv = torch.mean(
                    torch.stack(
                        [
                            self._stage2_adv_loss(c, a, epoch, n)
                            for c, a in zip(clean_feats, adv_feats)
                        ]
                    )
                )
                loss_G = self.lambda_adv_stage2 * loss_adv + self.lambda_norm * loss_norm
                loss_G.backward()
                optimizer_G.step()
                progress.set_postfix(
                    D=f"{float(loss_D_real.detach() + loss_D_fake.detach()):.3f}",
                    adv=f"{float(loss_adv.detach()):.3f}",
                )

    def _stage2_adv_loss(
        self, clean_feats: torch.Tensor, adv_feats: torch.Tensor, epoch: int, n: int
    ) -> torch.Tensor:
        """Per-model stage-2 loss over the identities: cosine similarity to each clean feature,
        or the distance to the clean features' convex hull after a class-center warm-up.
        """
        if self.subspace == "convex_hull":
            if epoch < self.hull_warmup_epochs:
                upper = lower = 1.0 / n
            else:
                upper, lower = self.hull_upper_bound, self.hull_lower_bound
            terms = [
                self._hull_loss(
                    adv_feats[s * n:(s + 1) * n], clean_feats[s * n:(s + 1) * n], upper=upper, lower=lower
                )
                for s in range(self.num_shot)
            ]
            return torch.stack(terms).mean()
        terms = [
            feature_similarity_loss(clean_feats[s * n:(s + 1) * n], adv_feats[s * n:(s + 1) * n])
            for s in range(self.num_shot)
        ]
        return torch.stack(terms).mean()

    def fit(self, data: FitData) -> None:
        """Fit the protectee's mask: the album's mean generator perturbation, L2-capped.

        Args:
            data.protectee_album: The protectee's aligned RGB crops in ``[0, 255]``,
                ``(B, 3, 112, 112)``.
        """
        self._check_inputs("fit", data)
        self._begin_fitting()
        album = self._validate_images(data.protectee_album, "protectee_album")
        self.netG.eval()
        with torch.no_grad():
            internal = album.to(self.device).div(127.5).sub(1.0)
            perturbation = self.netG(internal)
            # One Person One Mask: aggregate per-image perturbations into one.
            mask = perturbation.mean(dim=0, keepdim=True)
            self.person_mask = self._project_l2(
                mask * _MASK_TO_PIXEL, self.mask_l2_cap
            )
        self._mark_ready()

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Add the fitted mask to each aligned crop and paste the change back.

        Args:
            data.eval_imgs: Original photos of the person :meth:`fit` was run for.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        if self.person_mask is None:
            raise RuntimeError(
                "AdvCloak.protect() needs a person mask; call fit(FitData("
                "protectee_album=...)) first."
            )
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        sources = self._validate_images(torch.stack(crops), "eval_imgs").to(self.device)
        with torch.no_grad():
            cloaked = torch.clamp(sources + self.person_mask, 0, 255)
        return paste_deltas(photos, cloaked - sources, plans)

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        # Kaiming-normal weights for conv and LayerNorm layers, as in the reference code.
        classname = module.__class__.__name__
        if classname.find("Conv") != -1 or classname.find("LayerNorm") != -1:
            nn.init.kaiming_normal_(module.weight.data)

    @staticmethod
    def _project_l2(perturbation: torch.Tensor, cap: float) -> torch.Tensor:
        """Scale rows whose L2 norm exceeds ``cap`` down to ``cap``."""
        norm = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
        factor = torch.clamp(norm / cap, min=1.0).view(-1, 1, 1, 1)
        return perturbation / factor

    @staticmethod
    def _validate_images(images: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B, 3, H, W]")
        if not images.is_floating_point():
            images = images.to(torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains non-finite values")
        if images.min() < 0 or images.max() > 255:
            raise ValueError(f"{name} must be RGB in [0, 255]")
        return images.to(torch.float32)
