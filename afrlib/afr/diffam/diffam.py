"""DiffAM: fine-tune a DDPM for one impersonation target and makeup reference.

Reference Paper: DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection (CVPR 2024)
Reference Code: https://github.com/HansSunY/DiffAM
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
    TrainData,
)
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.afr.datasets import dataset_config
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, tqdm

from .data import (
    DEFAULT_REFERENCE,
    IMAGE_SIZE,
    REFERENCE_NAMES,
    ReferenceTriplet,
)
from .diffusion import (
    DEFAULT_INVERSION_POINTS,
    DEFAULT_REVERSE_STEPS,
    DEFAULT_T0,
    inversion_transitions,
    linear_beta_schedule,
    reverse_transitions,
    run_transitions,
)
from .losses import (
    CLIPMakeupDirectionLoss,
    reference_regions,
    regional_makeup_loss,
    source_regions,
    targeted_ensemble_loss,
)
from .models import DDPM

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus


logger = get_logger(__name__)


def _load_state(path: str | os.PathLike[str]) -> Mapping[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"DiffAM checkpoint is empty or invalid: {path}")
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    if any(str(key).startswith("module.") for key in state):
        state = {
            str(key).removeprefix("module."): value for key, value in state.items()
        }
    return state


class DiffAM(AFRMethod):
    """A CelebA-HQ DDPM fine-tuned so that inversion and regeneration add a reference's makeup and
    move the face toward one impersonation target; both are baked into the weights.

    Args:
        learning_rate: Adam learning rate.
        scheduler_gamma: Learning-rate growth factor.
        scheduler_step_iters: Grow the learning rate every this many steps (the paper uses 50);
            ``None`` grows it once per epoch, as the reference code does.
        outer_passes: Training epochs.
        makeup_only_passes: Leading epochs without the adversarial loss.
        max_train_images: CelebAMask-HQ images used for training.
        t0: Inversion depth, in diffusion steps.
        inversion_points: DDIM points of the inversion.
        reverse_steps: DDIM steps of the regeneration during training.
        seed: Training seed.
    """

    name = "DiffAM"
    training_corpus = "celebamask_hq"
    train_inputs = StageInputs(
        required=("corpus", "target_img"), optional=("makeup_reference_name",)
    )
    eval_inputs = StageInputs(required=("eval_imgs",))

    def __init__(
        self,
        device: torch.device | str,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: str | os.PathLike[str] | None = None,
        learning_rate: float = 4e-6,
        scheduler_gamma: float = 1.2,
        scheduler_step_iters: int | None = None,
        outer_passes: int = 6,
        makeup_only_passes: int = 0,
        max_train_images: int = 200,
        t0: int = DEFAULT_T0,
        inversion_points: int = DEFAULT_INVERSION_POINTS,
        reverse_steps: int = DEFAULT_REVERSE_STEPS,
        seed: int = 1234,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)
        # Set by train(): the makeup reference and the target are baked into
        # the tuned DDPM.
        self.reference_name: str | None = None
        self._reference: ReferenceTriplet | None = None

        self.learning_rate = float(learning_rate)
        self.scheduler_gamma = float(scheduler_gamma)
        self.scheduler_step_iters = (
            int(scheduler_step_iters) if scheduler_step_iters is not None else None
        )
        self.outer_passes = int(outer_passes)
        self.makeup_only_passes = int(makeup_only_passes)
        self.max_train_images = int(max_train_images)
        self.t0 = int(t0)
        self.inversion_points = int(inversion_points)
        self.reverse_steps = int(reverse_steps)
        self.seed = int(seed)
        if not (
            self.outer_passes >= 1
            and 0 <= self.makeup_only_passes <= self.outer_passes
            and self.max_train_images >= 1
            and (self.scheduler_step_iters is None or self.scheduler_step_iters >= 1)
        ):
            raise ValueError("Invalid DiffAM training schedule")

        self.betas = linear_beta_schedule(device=self.device)
        self.inversion_sequence = inversion_transitions(
            self.t0, self.inversion_points
        )
        self.reverse_sequence = reverse_transitions(self.t0, self.reverse_steps)
        base_checkpoint = Path(
            get_asset("diffam/celeba_hq.ckpt")
        ).expanduser().resolve()
        self.target_sha256: str | None = None
        self._baked_target: torch.Tensor | None = None
        self._train_dataset_config: dict[str, Any] | None = None

        self.base_model = DDPM().to(self.device).eval().requires_grad_(False)
        self.base_model.load_state_dict(
            _load_state(base_checkpoint), strict=True
        )

        # The tuned model starts as a copy of the base DDPM; _maybe_load_run
        # loads saved tuned weights over it.
        tuned_model = DDPM()
        tuned_model.load_state_dict(self.base_model.state_dict(), strict=True)
        self.model = tuned_model.to(self.device)
        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=IMAGE_SIZE, channel=ChannelType.RGB, crop_type=CropType.FFHQ,
            align_type=AlignType.NoAlign,
        )

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """CelebAMask-HQ loader with batch size 1, as in the upstream code."""
        from .data import CelebAMaskHQDataset

        return DataLoader(
            CelebAMaskHQDataset.from_corpus(
                corpus, max_records=self.max_train_images
            ),
            batch_size=1,
            shuffle=False,
        )

    @staticmethod
    def _resolve_tuned_checkpoint(path: str | os.PathLike[str]) -> Path:
        candidate = Path(path).expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "diffam.pth"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def load_run(self, load_path: str | os.PathLike[str]) -> Path:
        """Load the tuned DDPM written by :meth:`save_run`; the priors come from the asset cache."""
        root = super().load_run(load_path)
        checkpoint = self._resolve_tuned_checkpoint(load_path)
        self.model.load_state_dict(_load_state(checkpoint), strict=True)
        self._trained = True
        self._mark_ready()
        return root

    def _load_reference(self, name: str) -> ReferenceTriplet:
        if self._reference is None:
            self._reference = ReferenceTriplet.from_assets(name)
        return self._reference

    def _build_training_losses(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        import lpips

        clip_loss = CLIPMakeupDirectionLoss(self.device)
        perceptual = lpips.LPIPS(net="alex")
        return (
            clip_loss.to(self.device).eval().requires_grad_(False),
            perceptual.to(self.device).eval().requires_grad_(False),
        )

    def _embed(self, fr: FaceRecognizer, images_pm1: torch.Tensor) -> torch.Tensor:
        """Embed whole ``[-1, 1]`` frames, resized without antialiasing as upstream."""
        return fr.embed_crops(images_pm1, input_space="pm1", antialias=False)

    @staticmethod
    def _validate_public_images(
        images: torch.Tensor, *, name: str, require_single: bool = False
    ) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"{name} must have shape [B,3,256,256]")
        if require_single and images.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one image")
        images = images.float()
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains non-finite values")
        if float(images.min()) < 0 or float(images.max()) > 255:
            raise ValueError(f"{name} must be RGB in [0,255]")
        return images

    @staticmethod
    def _validate_target(target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if target.ndim != 4 or target.shape[0] != 1 or target.shape[1] != 3:
            raise ValueError("target_img must have shape [1,3,H,W]")
        target = target.float()
        if not torch.isfinite(target).all():
            raise ValueError("target_img contains non-finite values")
        if float(target.min()) < 0 or float(target.max()) > 255:
            raise ValueError("target_img must be RGB in [0,255]")
        return target

    @staticmethod
    def _unpack_batch(
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, list[torch.Tensor], str] | None:
        valid = batch.get("valid", True)
        if isinstance(valid, torch.Tensor) and not bool(valid.reshape(-1)[0]):
            return None
        if valid is False:
            return None
        image = batch["image"]
        masks = list(batch["masks"])
        filename_value = batch.get("filename", "")
        filename = (
            str(filename_value[0])
            if isinstance(filename_value, (list, tuple))
            else str(filename_value)
        )
        expected_masks = [(1, 1, IMAGE_SIZE, IMAGE_SIZE)] * 5
        if [tuple(mask.shape) for mask in masks] != expected_masks:
            raise ValueError("DiffAM source masks must be five [1,1,256,256] tensors")
        if tuple(image.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("DiffAM training images must be [1,3,256,256]")
        return image, masks, filename

    def _precompute_latents(
        self, train_loader: DataLoader
    ) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], str]]:
        if train_loader.batch_size != 1:
            raise ValueError("DiffAM requires DataLoader batch_size=1")
        cached = []
        # Inversion uses only the frozen base model; park the tuned copy on CPU.
        self.model.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.base_model.to(self.device)
        self.base_model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="DiffAM inversion"):
                unpacked = self._unpack_batch(batch)
                if unpacked is None:
                    continue
                source, masks, filename = unpacked
                source = source.to(self.device)
                latent = run_transitions(
                    source,
                    self.inversion_sequence,
                    self.base_model,
                    self.betas,
                )
                cached.append(
                    (
                        source.cpu(),
                        latent.cpu(),
                        [mask.cpu() for mask in masks],
                        filename,
                    )
                )
                if len(cached) >= self.max_train_images:
                    break
        if not cached:
            raise ValueError("DiffAM found no valid CelebAMask-HQ training samples")
        self.base_model.to("cpu")
        self.model.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return cached

    def train(self, data: TrainData) -> None:
        """Fine-tune the DDPM on ``L_MT + lambda_adv * L_adv + L_vis`` (Sec. 4.1); the first
        ``makeup_only_passes`` epochs omit ``L_adv``.

        Args:
            data.corpus: The CelebAMask-HQ corpus, read one image at a time.
            data.target_img: The impersonation target baked into the tuned
                DDPM, one ``[1, 3, H, W]`` RGB image in ``[0, 255]``.
            data.makeup_reference_name: Name of the released makeup reference
                baked into the tuned DDPM (see ``REFERENCE_NAMES``); defaults
                to ``"XMY-060"``.
        """
        self._check_inputs("train", data)
        makeup_reference = data.makeup_reference_name or DEFAULT_REFERENCE
        if makeup_reference not in REFERENCE_NAMES:
            raise ValueError(f"Unknown makeup_reference_name {makeup_reference!r}")
        self._require_fr()
        train_loader = self._build_loader(data.corpus)
        self._begin_training()
        self.reference_name = makeup_reference
        self._train_dataset_config = dataset_config(train_loader)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        target = self._validate_target(data.target_img)
        target_pm1 = target.to(self.device).div(127.5).sub(1.0)
        self._baked_target = target.detach().cpu()
        self.target_sha256 = hashlib.sha256(
            target.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()
        with torch.no_grad():
            target_embeddings = [
                self._embed(fr, target_pm1).detach() for fr in self.fr_models
            ]
        reference = self._load_reference(makeup_reference)
        makeup = reference.makeup.unsqueeze(0).to(self.device)
        non_makeup = reference.non_makeup.unsqueeze(0).to(self.device)
        reference_masks = {
            name: mask.to(self.device)
            for name, mask in reference_regions(reference.labels).items()
        }
        latent_cache = self._precompute_latents(train_loader)
        clip_loss, perceptual_loss = self._build_training_losses()

        self.model.train().requires_grad_(True)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0
        )
        # None steps the LR once per epoch, as the released code does; an int
        # steps it every N optimizer steps, as the paper describes.
        step_per_iter = self.scheduler_step_iters is not None
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_iters if step_per_iter else 1,
            gamma=self.scheduler_gamma,
        )
        for outer_pass in range(self.outer_passes):
            adversarial_stage = outer_pass >= self.makeup_only_passes
            for source_cpu, latent_cpu, masks_cpu, filename in latent_cache:
                source = source_cpu.to(self.device)
                latent = latent_cpu.to(self.device)
                masks = [mask.to(self.device) for mask in masks_cpu]
                optimizer.zero_grad(set_to_none=True)
                generated = run_transitions(
                    latent,
                    self.reverse_sequence,
                    self.model,
                    self.betas,
                    checkpoint_model=True,
                )
                regions = source_regions(masks)
                makeup_loss = regional_makeup_loss(
                    generated, makeup, regions, reference_masks
                )
                direction_loss = clip_loss(
                    source, non_makeup, generated, makeup
                )
                l1_loss = F.l1_loss(generated, source)
                lpips_loss = perceptual_loss(source, generated).mean()
                if adversarial_stage:
                    adversarial_embeddings = [
                        self._embed(fr, generated) for fr in self.fr_models
                    ]
                    adversarial_loss = targeted_ensemble_loss(
                        adversarial_embeddings, target_embeddings
                    )
                    total = (
                        1.6 * makeup_loss
                        + 0.5 * direction_loss
                        + 0.5 * adversarial_loss
                        + 10.0 * lpips_loss
                        + 3.0 * l1_loss
                    )
                else:
                    adversarial_loss = generated.new_zeros(())
                    total = (
                        makeup_loss
                        + 0.3 * direction_loss
                        + 10.0 * lpips_loss
                        + 3.0 * l1_loss
                    )
                total.backward()
                optimizer.step()
                if step_per_iter:
                    scheduler.step()
            if not step_per_iter:
                scheduler.step()
        self.model.eval().requires_grad_(False)
        del clip_loss, perceptual_loss
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._mark_ready()

    def protect(self, data: EvalData) -> list[torch.Tensor]:
        """Invert each face with the base DDPM, regenerate it with the tuned one, and paste it back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        sources = self._validate_public_images(torch.stack(crops), name="eval_imgs")
        outputs = []
        self.base_model.to(self.device).eval()
        self.model.eval()
        with torch.inference_mode():
            for source in sources:
                internal = source.unsqueeze(0).to(self.device).div(127.5).sub(1.0)
                latent = run_transitions(
                    internal,
                    self.inversion_sequence,
                    self.base_model,
                    self.betas,
                )
                generated = run_transitions(
                    latent, self.reverse_sequence, self.model, self.betas
                )
                outputs.append(generated.add(1.0).mul(127.5).clamp(0, 255)[0])
        return paste_faces(photos, outputs, plans)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "learning_rate": self.learning_rate,
                "scheduler_gamma": self.scheduler_gamma,
                "scheduler_step_iters": self.scheduler_step_iters,
                "outer_passes": self.outer_passes,
                "makeup_only_passes": self.makeup_only_passes,
                "max_train_images": self.max_train_images,
                "t0": self.t0,
                "inversion_points": self.inversion_points,
                "reverse_steps": self.reverse_steps,
                "seed": self.seed,
                # Provenance and fixed settings, not constructor arguments.
                "target_tensor_sha256": self.target_sha256,
                "reference_name": self.reference_name,
                "training_dataset": self._train_dataset_config,
                "recipe": (
                    "paper"
                    if self.makeup_only_passes == 0
                    and self.scheduler_step_iters is None
                    else "custom"
                ),
                "lr_step_per": (
                    "epoch" if self.scheduler_step_iters is None
                    else f"{self.scheduler_step_iters}_optimizer_steps"
                ),
                "image_size": IMAGE_SIZE,
                "external_range": [0.0, 255.0],
                "internal_range": [-1.0, 1.0],
                "batch_size": 1,
                "optimizer": "Adam",
                "weight_decay": 0.0,
                "scheduler": "StepLR",
                "diffusion_schedule": "linear",
                "diffusion_steps": 1_000,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "inversion_transitions": len(self.inversion_sequence),
                "eta": 0.0,
                "combined_weights": {
                    "histogram": 1.6,
                    "direction": 0.5,
                    "lpips": 10.0,
                    "l1": 3.0,
                    "adversarial": 0.5,
                },
                "makeup_only_weights": {
                    "histogram": 1.0,
                    "direction": 0.3,
                    "lpips": 10.0,
                    "l1": 3.0,
                    "adversarial": 0.0,
                },
            }
        )
        return config

    def save_run(self, save_path: str | os.PathLike[str]) -> Path:
        """Save the tuned DDPM (atomically), the baked target, and the config."""
        self._require_savable()
        destination = Path(save_path).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = destination / "diffam.pth"
        checkpoint_tmp = destination / ".diffam.pth.tmp"
        state = {
            key: value.detach().cpu() for key, value in self.model.state_dict().items()
        }
        torch.save(state, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint)
        inputs = (
            self._write_input_images(destination, "target", self._baked_target)
            if self._baked_target is not None
            else []
        )
        return super().save_run(
            destination, weights=["diffam.pth"], baked_inputs=inputs
        )


__all__ = ["DiffAM"]
