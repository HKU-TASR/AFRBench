"""AMT-GAN: adversarial makeup transfer toward one impersonation target.

Reference Paper: Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer (CVPR 2022)
Reference Code: https://github.com/CGCL-codes/AMT-GAN
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

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
from afrlib.afr.datasets import dataset_config
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.face_crop import natural_crop, paste_faces, prepare_inputs
from afrlib.utils.log import get_logger, tqdm

from .assets import DEFAULT_REFERENCE, REFERENCE_NAMES
from .losses import HistogramLoss, cosine_distance, lsgan_loss, make_lpips
from .models import Discriminator, Generator, Regularizer
from .preprocess import AMTGANPreprocessor, PreparedFace, laplacian_postprocess

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus


logger = get_logger(__name__)


class AMT_GAN(AFRMethod):
    """Makeup transferred from a donor photo by a generator trained to impersonate one target.

    Args:
        epochs: Training epochs.
        learning_rate: Adam learning rate.
        beta1: Adam beta1.
        beta2: Adam beta2.
        lambda_gan: Weight of the GAN losses.
        lambda_cycle: Weight of the regularized cycle losses.
        lambda_attack: Weight of the targeted FR losses.
        lambda_makeup: Weight of the histogram makeup loss.
        lambda_identity: Weight of the identity-mapping losses.
        diversity_samples: Noisy, input-diverse copies per output in the FR losses.
    """

    name = "AMT-GAN"
    training_corpus = "mt"
    train_inputs = StageInputs(required=("corpus", "target_img"))
    eval_inputs = StageInputs(
        required=("eval_imgs",), optional=("makeup_reference_name",)
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: Optional[str] = None,
        epochs: int = 5,
        learning_rate: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gan: float = 10.0,
        lambda_cycle: float = 10.0,
        lambda_attack: float = 5.0,
        lambda_makeup: float = 2.0,
        lambda_identity: float = 5.0,
        diversity_samples: int = 5,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.lambda_gan = float(lambda_gan)
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_attack = float(lambda_attack)
        self.lambda_makeup = float(lambda_makeup)
        self.lambda_identity = float(lambda_identity)
        self.diversity_samples = int(diversity_samples)
        self.region_weights = (1.0, 0.1, 1.0)  # lip, skin, eye

        self._preprocessor: Optional[AMTGANPreprocessor] = None
        # Baked training input: kept so save_run can write it into the run.
        self._baked_target: Optional[torch.Tensor] = None
        self.target_sha256: Optional[str] = None
        self._train_dataset_config: Optional[dict[str, Any]] = None

        self.G = Generator().to(self.device)
        self.H: Optional[Regularizer] = None
        self.D_A: Optional[Discriminator] = None
        self.D_B: Optional[Discriminator] = None
        self._maybe_load_run()


    def attack_spec(self) -> ImageSpec:
        # AMT-GAN works on an FFHQ-framed 256px face crop.
        return ImageSpec(
            size=256, channel=ChannelType.RGB, crop_type=CropType.FFHQ,
            align_type=AlignType.FFHQQuad,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the flat constructor config plus training provenance."""
        config = super().get_config()
        config.update(
            {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "lambda_gan": self.lambda_gan,
                "lambda_cycle": self.lambda_cycle,
                "lambda_attack": self.lambda_attack,
                "lambda_makeup": self.lambda_makeup,
                "lambda_identity": self.lambda_identity,
                "diversity_samples": self.diversity_samples,
                # Provenance and fixed settings, not constructor arguments.
                "target_tensor_sha256": self.target_sha256,
                "training_dataset": self._train_dataset_config,
                "histogram_region_weights": list(self.region_weights),
                "batch_size": 1,
                "image_size": 256,
                "distributed": False,
            }
        )
        return config

    def train(self, data: TrainData) -> None:
        """Train a fresh target-specific G/H/D system on the MT corpus.

        Args:
            data.corpus: The MT makeup corpus; turned into the released
                random-pair loader by ``AMTGANLMDBDataset.from_corpus``.
            data.target_img: The impersonation target baked into the generator,
                one ``[3,H,W]`` or ``[1,3,H,W]`` RGB image in ``[0, 255]``.
        """
        self._check_inputs("train", data)
        self._require_fr()
        train_loader = self._build_loader(data.corpus)
        self._begin_training()
        self._train_dataset_config = dataset_config(train_loader)
        target = self._validate_image_batch(
            data.target_img, name="target_img", require_single=True,
            require_1024=False,
        )
        target_fingerprint = hashlib.sha256(
            target.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()
        target_internal = target.to(self.device).div(127.5).sub(1.0)

        G = Generator().to(self.device)
        H = Regularizer().to(self.device)
        D_A = Discriminator(image_size=256).to(self.device)
        D_B = Discriminator(image_size=256).to(self.device)
        for network in (G, H, D_A, D_B):
            network.apply(self._init_xavier)
            network.train()

        g_optimizer = torch.optim.Adam(
            G.parameters(), self.learning_rate, betas=(self.beta1, self.beta2)
        )
        h_optimizer = torch.optim.Adam(
            H.parameters(), self.learning_rate, betas=(self.beta1, self.beta2)
        )
        d_a_optimizer = torch.optim.Adam(
            (parameter for parameter in D_A.parameters() if parameter.requires_grad),
            self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        d_b_optimizer = torch.optim.Adam(
            (parameter for parameter in D_B.parameters() if parameter.requires_grad),
            self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        histogram_loss = HistogramLoss().to(self.device)
        lpips_loss = make_lpips(self.device)

        with torch.no_grad():
            target_embeddings = [
                self._embed(fr, target_internal).detach() for fr in self.fr_models
            ]

        steps_per_epoch = len(train_loader)
        if steps_per_epoch < 1:
            raise ValueError("train_loader produced no batches")
        # Per-step progress: an epoch is about 2,700 expensive steps on MT.
        progress = tqdm(
            total=self.epochs * steps_per_epoch,
            desc="AMT-GAN training",
            unit="step",
        )
        for epoch in range(self.epochs):
            epoch_values: list[dict[str, float]] = []
            for source_input, reference_input in train_loader:
                image_s, mask_s, dist_s = self._move_training_input(source_input)
                image_r, mask_r, dist_r = self._move_training_input(reference_input)

                # D_A learns the makeup/reference domain.
                self._set_trainable((D_A,), True)
                self._set_trainable((G, H, D_B), False)
                with torch.no_grad():
                    fake_a_detached = G(
                        image_s, image_r, mask_s, mask_r, dist_s, dist_r
                    )
                d_a_loss = 0.5 * (
                    lsgan_loss(D_A(image_r), True)
                    + lsgan_loss(D_A(fake_a_detached), False)
                )
                d_a_optimizer.zero_grad(set_to_none=True)
                d_a_loss.backward()
                d_a_optimizer.step()

                # D_B learns the non-makeup/source domain.
                self._set_trainable((D_B,), True)
                self._set_trainable((G, H, D_A), False)
                with torch.no_grad():
                    fake_b_detached = G(
                        image_r, image_s, mask_r, mask_s, dist_r, dist_s
                    )
                d_b_loss = 0.5 * (
                    lsgan_loss(D_B(image_s), True)
                    + lsgan_loss(D_B(fake_b_detached), False)
                )
                d_b_optimizer.zero_grad(set_to_none=True)
                d_b_loss.backward()
                d_b_optimizer.step()

                # Generator objective: released LSGAN, identity, makeup,
                # regularized-cycle, and targeted FR losses.
                self._set_trainable((G,), True)
                self._set_trainable((H, D_A, D_B), False)
                fake_a = G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
                fake_b = G(image_r, image_s, mask_r, mask_s, dist_r, dist_s)
                loss_gan_g = 0.5 * self.lambda_gan * (
                    lsgan_loss(D_A(fake_a), True)
                    + lsgan_loss(D_B(fake_b), True)
                )

                idt_a = G(image_s, image_s, mask_s, mask_s, dist_s, dist_s)
                idt_b = G(image_r, image_r, mask_r, mask_r, dist_r, dist_r)
                loss_idt_g = 0.5 * self.lambda_identity * (
                    F.l1_loss(idt_a, image_s)
                    + lpips_loss(idt_a, image_s).mean()
                    + lpips_loss(fake_a, image_s).mean()
                    + F.l1_loss(idt_b, image_r)
                    + lpips_loss(idt_b, image_r).mean()
                    + lpips_loss(fake_b, image_r).mean()
                )
                loss_make_g = self._makeup_loss(
                    histogram_loss,
                    fake_a,
                    fake_b,
                    image_s,
                    image_r,
                    mask_s,
                    mask_r,
                )
                purified_a = H(fake_a)
                purified_b = H(fake_b)
                rec_a = H(
                    G(purified_a, image_s, mask_s, mask_s, dist_s, dist_s)
                )
                rec_b = H(
                    G(purified_b, image_r, mask_r, mask_r, dist_r, dist_r)
                )
                loss_cycle_g = 0.5 * self.lambda_cycle * (
                    F.l1_loss(rec_a, image_s) + F.l1_loss(rec_b, image_r)
                )
                diverse_a = [
                    self._input_diversity(self._input_noise(fake_a))
                    for _ in range(self.diversity_samples)
                ]
                diverse_b = [
                    self._input_diversity(self._input_noise(fake_b))
                    for _ in range(self.diversity_samples)
                ]
                attack_terms = []
                for model_index, fr in enumerate(self.fr_models):
                    target_embedding = target_embeddings[model_index]
                    for sample_index in range(self.diversity_samples):
                        attack_terms.extend(
                            (
                                0.5
                                * self.lambda_attack
                                * cosine_distance(
                                    self._embed(fr, diverse_a[sample_index]),
                                    target_embedding,
                                ),
                                0.5
                                * self.lambda_attack
                                * cosine_distance(
                                    self._embed(fr, diverse_b[sample_index]),
                                    target_embedding,
                                ),
                            )
                        )
                loss_attack_g = torch.stack(attack_terms).mean()
                g_loss = (
                    loss_gan_g
                    + loss_idt_g
                    + loss_make_g
                    + loss_cycle_g
                    + loss_attack_g
                ).mean()
                self._check_finite("generator loss", g_loss)
                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                self._check_gradients("generator", G)
                g_optimizer.step()

                # H removes adversarial identity while preserving makeup and
                # closes the released G -> H -> G -> H cycle.
                self._set_trainable((H,), True)
                self._set_trainable((G, D_A, D_B), False)
                with torch.no_grad():
                    fake_a_h = G(
                        image_s, image_r, mask_s, mask_r, dist_s, dist_r
                    )
                    fake_b_h = G(
                        image_r, image_s, mask_r, mask_s, dist_r, dist_s
                    )
                    idt_a_h = G(
                        image_s, image_s, mask_s, mask_s, dist_s, dist_s
                    )
                    idt_b_h = G(
                        image_r, image_r, mask_r, mask_r, dist_r, dist_r
                    )
                    clean_s_embeddings = [
                        self._embed(fr, image_s).detach() for fr in self.fr_models
                    ]
                    clean_r_embeddings = [
                        self._embed(fr, image_r).detach() for fr in self.fr_models
                    ]

                purified_a_h = H(fake_a_h)
                purified_b_h = H(fake_b_h)
                loss_gan_h = 0.5 * self.lambda_gan * (
                    lsgan_loss(D_A(purified_a_h), True)
                    + lsgan_loss(D_B(purified_b_h), True)
                )
                purified_idt_a = H(idt_a_h)
                purified_idt_b = H(idt_b_h)
                loss_idt_h = 0.5 * self.lambda_identity * (
                    F.l1_loss(purified_idt_a, image_s)
                    + lpips_loss(purified_idt_a, image_s).mean()
                    + lpips_loss(purified_a_h, image_s).mean()
                    + F.l1_loss(purified_idt_b, image_r)
                    + lpips_loss(purified_idt_b, image_r).mean()
                    + lpips_loss(purified_b_h, image_r).mean()
                )
                loss_make_h = self._makeup_loss(
                    histogram_loss,
                    purified_a_h,
                    purified_b_h,
                    image_s,
                    image_r,
                    mask_s,
                    mask_r,
                )
                rec_a_h = H(
                    G(purified_a_h, image_s, mask_s, mask_s, dist_s, dist_s)
                )
                rec_b_h = H(
                    G(purified_b_h, image_r, mask_r, mask_r, dist_r, dist_r)
                )
                loss_cycle_h = 0.5 * self.lambda_cycle * (
                    F.l1_loss(rec_a_h, image_s) + F.l1_loss(rec_b_h, image_r)
                )
                self_terms = []
                for model_index, fr in enumerate(self.fr_models):
                    self_terms.extend(
                        (
                            0.5
                            * self.lambda_attack
                            * cosine_distance(
                                self._embed(fr, purified_a_h),
                                clean_s_embeddings[model_index],
                            ),
                            0.5
                            * self.lambda_attack
                            * cosine_distance(
                                self._embed(fr, purified_b_h),
                                clean_r_embeddings[model_index],
                            ),
                        )
                    )
                loss_attack_h = torch.stack(self_terms).mean()
                h_loss = (
                    loss_gan_h
                    + loss_idt_h
                    + loss_make_h
                    + loss_cycle_h
                    + loss_attack_h
                ).mean()
                self._check_finite("regularizer loss", h_loss)
                h_optimizer.zero_grad(set_to_none=True)
                h_loss.backward()
                self._check_gradients("regularizer", H)
                h_optimizer.step()

                step_values = {
                    "D_A": float(d_a_loss.detach()),
                    "D_B": float(d_b_loss.detach()),
                    "G": float(g_loss.detach()),
                    "H": float(h_loss.detach()),
                }
                epoch_values.append(step_values)
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{self.epochs}",
                    G=f"{step_values['G']:.3f}",
                    H=f"{step_values['H']:.3f}",
                    refresh=False,
                )
                progress.update(1)

        progress.close()

        # Publish the complete trained state only after all epochs succeed.
        self.G = G.eval()
        self.H = H.eval()
        self.D_A = D_A.eval()
        self.D_B = D_B.eval()
        for network in (self.G, self.H, self.D_A, self.D_B):
            network.requires_grad_(False)
        self._mark_ready()
        self.target_sha256 = target_fingerprint
        self._baked_target = target.detach().cpu()
        logger.info(
            "Trained target-specific AMT-GAN for %d epochs (%d batches in final epoch).",
            self.epochs,
            len(epoch_values),
        )

    def save_run(self, save_path: str | os.PathLike[str]) -> Path:
        """Atomically save ``G``, any trained ``H``/``D_A``/``D_B`` and the baked target, but not
        the frozen priors.
        """
        self._require_savable()
        output = Path(save_path)
        output.mkdir(parents=True, exist_ok=True)
        networks: dict[str, nn.Module] = {"G": self.G}
        for name in ("H", "D_A", "D_B"):
            network = getattr(self, name)
            if network is not None:
                networks[name] = network
        for name, network in networks.items():
            destination = output / f"{name}.pth"
            temporary = output / f".{name}.pth.tmp"
            state = {
                key: value.detach().cpu() for key, value in network.state_dict().items()
            }
            torch.save(state, temporary)
            os.replace(temporary, destination)

        inputs = (
            self._write_input_images(output, "target", self._baked_target)
            if self._baked_target is not None
            else []
        )
        return super().save_run(
            output,
            weights=[f"{name}.pth" for name in networks],
            baked_inputs=inputs,
        )

    def load_run(self, load_path: str | os.PathLike[str]) -> Path:
        """Load ``G`` (required for :meth:`protect`) and any saved ``H``/``D_A``/``D_B``."""
        root = super().load_run(load_path)
        generator_path = root / "G.pth"
        if not generator_path.is_file():
            raise FileNotFoundError(
                f"No AMT-GAN generator found under {root}"
            )
        state = torch.load(
            generator_path, map_location=self.device, weights_only=True
        )
        self.G.load_state_dict(state, strict=True)
        self.G.eval().requires_grad_(False)
        for attr, builder in (
            ("H", Regularizer),
            ("D_A", Discriminator),
            ("D_B", Discriminator),
        ):
            path = root / f"{attr}.pth"
            if not path.is_file():
                continue
            network = getattr(self, attr)
            if network is None:
                network = (
                    builder(image_size=256) if attr.startswith("D_") else builder()
                ).to(self.device)
            network.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True),
                strict=True,
            )
            network.eval().requires_grad_(False)
            setattr(self, attr, network)
        # A generator-only run is a finished inference artifact even without
        # a config.json to restore the stage markers from.
        self._trained = True
        self._mark_ready()
        return root

    def protect(self, data: EvalData) -> list[torch.Tensor]:
        """Transfer adversarial makeup onto each face and paste it back; a crop with no usable
        face is returned unchanged.

        Args:
            data.makeup_reference_name: A released makeup-donor PNG (see ``REFERENCE_NAMES``);
                ``ref01.png`` by default.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        photos = [
            self._validate_image_batch(
                photo, name="eval_imgs", require_single=True, require_1024=False
            )[0]
            for photo in data.eval_imgs
        ]
        crops, plans = prepare_inputs(photos, data.eval_geometry, self.attack_spec())
        preprocessor = self._get_preprocessor()
        prepared_reference = preprocessor.prepare_inference(
            self._load_makeup_reference(data.makeup_reference_name)
        )
        if prepared_reference is None:
            raise RuntimeError("No face detected in the reference makeup image.")
        reference_image, reference_masks, reference_distances = (
            self._prepared_to_device(prepared_reference)
        )

        self.G.eval()
        protected: list[torch.Tensor] = []
        with torch.inference_mode():
            for index, (photo, crop, plan) in enumerate(zip(photos, crops, plans)):
                crop_pil = self._tensor_to_pil(crop)
                prepared_source = preprocessor.prepare_inference(crop_pil)
                if prepared_source is None:
                    logger.warning(
                        "AMT-GAN source %s: no usable face on the FFHQ crop; "
                        "returning the source crop",
                        index,
                    )
                    protected.append(crop.to(self.device))
                    continue
                source_image, source_masks, source_distances = (
                    self._prepared_to_device(prepared_source)
                )
                gamma, beta = self.G(
                    source_image,
                    reference_image,
                    source_masks,
                    reference_masks,
                    source_distances,
                    reference_distances,
                    ret=True,
                )
                generated = self.G(
                    source_image,
                    reference_image,
                    source_masks,
                    reference_masks,
                    source_distances,
                    reference_distances,
                    gamma=gamma,
                    beta=beta,
                ).squeeze(0)
                generated = (generated - generated.min()) / (
                    generated.max() - generated.min() + 1e-5
                )
                generated_pil = to_pil_image(generated.cpu())
                # The released post-process restores the crop's high frequencies at its natural
                # size, the regime it assumes (a face larger than 256 px), before the inverse warp.
                source = self._tensor_to_pil(natural_crop(photo, plan))
                restored = laplacian_postprocess(source, generated_pil)
                if not isinstance(restored, Image.Image):
                    restored = self._tensor_to_pil(torch.as_tensor(restored))
                protected.append(
                    self._pil_to_tensor(restored).to(self.device)
                )
        return paste_faces(photos, protected, plans)

    @staticmethod
    def _init_xavier(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight, gain=1.0)

    @staticmethod
    def _set_trainable(networks: Sequence[nn.Module], enabled: bool) -> None:
        for network in networks:
            for name, parameter in network.named_parameters():
                # AMT-GAN's custom spectral-normalization power-iteration
                # vectors are checkpointed Parameters but are never optimized.
                is_power_vector = name.endswith(("weight_u", "weight_v"))
                parameter.requires_grad_(enabled and not is_power_vector)

    @staticmethod
    def _check_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} became non-finite")

    @staticmethod
    def _check_gradients(name: str, network: nn.Module) -> None:
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in network.parameters()
        ):
            raise FloatingPointError(f"{name} gradients became non-finite")

    def _embed(self, fr: FaceRecognizer, internal_images: torch.Tensor) -> torch.Tensor:
        images = internal_images.contiguous()
        return fr.embed_crops(images, input_space="pm1", antialias=False)

    def _input_diversity(self, images: torch.Tensor) -> torch.Tensor:
        # The CPU RNG draws the resize and padding, as in the reference code.
        resized_side = int(torch.randint(230, 256, (1,), dtype=torch.int32).item())
        resized = F.interpolate(
            images,
            size=(resized_side, resized_side),
            mode="bilinear",
            align_corners=False,
        )
        remaining = 256 - resized_side
        pad_top = int(torch.randint(0, remaining, (1,), dtype=torch.int32).item())
        pad_left = int(torch.randint(0, remaining, (1,), dtype=torch.int32).item())
        padded = F.pad(
            resized,
            (
                pad_left,
                remaining - pad_left,
                pad_top,
                remaining - pad_top,
            ),
        )
        return padded if torch.rand(1) < 0.5 else images

    @staticmethod
    def _input_noise(images: torch.Tensor) -> torch.Tensor:
        amplitude = torch.rand(1, device=images.device) * (0.1**0.5)
        noised = images + amplitude * torch.randn_like(images)
        return noised if torch.rand(1) < 0.5 else images

    def _makeup_loss(
        self,
        criterion: HistogramLoss,
        forward: torch.Tensor,
        reverse: torch.Tensor,
        source: torch.Tensor,
        reference: torch.Tensor,
        source_masks: torch.Tensor,
        reference_masks: torch.Tensor,
    ) -> torch.Tensor:
        forward_loss = forward.new_zeros(())
        reverse_loss = reverse.new_zeros(())
        for region, region_weight in enumerate(self.region_weights):
            forward_loss = forward_loss + region_weight * criterion(
                forward,
                reference,
                source_masks[:, region],
                reference_masks[:, region],
                self.device,
            )
            reverse_loss = reverse_loss + region_weight * criterion(
                reverse,
                source,
                reference_masks[:, region],
                source_masks[:, region],
                self.device,
            )
        return 0.5 * self.lambda_makeup * (forward_loss + reverse_loss)

    def _move_training_input(
        self, values: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(values) != 3:
            raise ValueError("Each LMDB domain input must contain image, mask, distance")
        image, mask, distance = (
            value.to(self.device, dtype=torch.float32) for value in values
        )
        expected = (
            (1, 3, 256, 256),
            (1, 3, 1, 256, 256),
            (1, 3, 136, 64, 64),
        )
        actual = (tuple(image.shape), tuple(mask.shape), tuple(distance.shape))
        if actual != expected:
            raise ValueError(
                f"AMT-GAN training tensor contract mismatch: got {actual}, "
                f"expected {expected}"
            )
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Training images must be normalized to [-1, 1]")
        return image, mask, distance

    @staticmethod
    def _validate_train_loader(train_loader: DataLoader) -> None:
        from .data import _AMTGANPairReader

        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a torch DataLoader")
        if train_loader.batch_size != 1:
            raise ValueError("AMT-GAN structurally requires DataLoader batch_size=1")
        dataset = train_loader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if not isinstance(dataset, _AMTGANPairReader):
            raise TypeError(
                "train_loader must use an AMT-GAN pair reader "
                "(AMTGANLMDBDataset or AMTGANCorpusDataset)"
            )

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """The released MT pair loader over the filtered corpus, one pair per batch as the
        histogram and region losses require.
        """
        from .data import AMTGANLMDBDataset, AMTGANTrainingPreprocessor

        loader = DataLoader(
            AMTGANLMDBDataset.from_corpus(
                corpus, preprocessor=AMTGANTrainingPreprocessor()
            ),
            batch_size=1,
        )
        self._validate_train_loader(loader)
        return loader

    @staticmethod
    def _validate_image_batch(
        images: torch.Tensor,
        *,
        name: str,
        require_single: bool,
        require_1024: bool,
    ) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B,3,H,W]")
        if require_single and images.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one target image")
        if require_1024 and tuple(images.shape[-2:]) != (1024, 1024):
            raise ValueError(f"{name} must contain 1024x1024 FFHQ images")
        if not images.is_floating_point():
            images = images.to(torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains non-finite values")
        if images.min() < 0 or images.max() > 255:
            raise ValueError(f"{name} must be RGB values in [0,255]")
        return images.to(torch.float32)

    @staticmethod
    def _load_makeup_reference(name: str | None) -> Image.Image:
        """Load a released makeup-donor PNG from the asset cache."""
        reference = name or DEFAULT_REFERENCE
        if reference not in REFERENCE_NAMES:
            raise ValueError(
                f"Unknown makeup_reference_name {reference!r}; "
                f"expected one of {REFERENCE_NAMES}"
            )
        with Image.open(get_asset(f"amt_gan/{reference}")) as image:
            return image.convert("RGB").copy()

    def _get_preprocessor(self) -> AMTGANPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = AMTGANPreprocessor(
                get_asset("amt_gan/face_parser.pth"),
                get_asset("amt_gan/shape_predictor_68_face_landmarks.dat"),
                self.device,
            )
        return self._preprocessor

    def _prepared_to_device(
        self, prepared: PreparedFace
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            prepared.image.to(self.device),
            prepared.masks.to(self.device),
            prepared.distances.to(self.device),
        )

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        tensor = image.detach().to("cpu")
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Expected a CHW RGB tensor")
        array = (
            tensor.permute(1, 2, 0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        return Image.fromarray(array, mode="RGB")

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB")).copy()
        return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32)