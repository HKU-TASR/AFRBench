"""Adv-Makeup: a target-specific adversarial eye-makeup generator.

Reference Paper: Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition (IJCAI 2021)
Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (commit 359af7e; academic research only, see LICENSE)
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
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
from afrlib.afr.datasets import dataset_config
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.utils.log import get_logger, tqdm

from .losses import (
    MeanShift,
    VGG16Features,
    cosine_distance,
    discriminator_mse,
    gradient_loss,
    style_loss,
    total_variation_loss,
)
from .models import Decoder, Discriminator, Encoder, initialize_discriminator
from .preprocess import (
    EYE_INPUT_SIZE,
    Landmark106Detector,
    composite_generated_eye,
    eye_crop_to_internal,
    poisson_restore_batch,
    validate_landmarks,
    build_orbital_geometry,
)


if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

logger = get_logger(__name__)


class AdvMakeup(AFRMethod):
    """Adversarial eye makeup from a generator trained for one impersonation target, with
    leave-one-model-out meta-learning over the surrogates.

    Args:
        epochs: Training epochs.
        generator_learning_rate: Adam learning rate of the encoder and decoder.
        discriminator_learning_rate: Adam learning rate of the discriminator.
        meta_learning_rate: Step size of the meta-train update.
        beta1: Adam beta1.
        beta2: Adam beta2.
        weight_decay: Adam weight decay.
        gradient_clip: Gradient-norm cap of the generator.
        poisson_blend: Poisson-blend the generated eye region instead of pasting it with a
            hard mask.
    """

    name = "Adv-Makeup"
    training_corpus = "ladn"
    min_surrogates = 2
    train_inputs = StageInputs(required=("corpus", "target_img"))
    eval_inputs = StageInputs(required=("eval_imgs",))

    #: Training batch size, as in the reference code.
    _BATCH_SIZE = 4

    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: Optional[str] = None,
        epochs: int = 300,
        generator_learning_rate: float = 1e-3,
        discriminator_learning_rate: float = 4e-4,
        meta_learning_rate: float = 1e-3,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 1e-4,
        gradient_clip: float = 5.0,
        poisson_blend: bool = False,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        self.epochs = int(epochs)
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        self.generator_learning_rate = float(generator_learning_rate)
        self.discriminator_learning_rate = float(
            discriminator_learning_rate
        )
        self.meta_learning_rate = float(meta_learning_rate)
        self.betas = (float(beta1), float(beta2))
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.poisson_blend = bool(poisson_blend)

        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.discriminator = initialize_discriminator(
            Discriminator()
        ).to(self.device)

        self.surrogate_order = list(range(len(self.fr_models)))
        self.target_sha256: Optional[str] = None
        # Target image baked in by train(); written out by save_run.
        self._baked_target: Optional[torch.Tensor] = None
        self._train_dataset_config: Optional[dict[str, Any]] = None
        self._landmark_detector: Optional[Landmark106Detector] = None

        self._maybe_load_run()


    def attack_spec(self) -> ImageSpec:
        # Adv-Makeup works on a 600px LADN-framed face crop.
        return ImageSpec(
            size=600,
            channel=ChannelType.RGB,
            crop_type=CropType.LADN600,
            align_type=AlignType.LADN600,
        )

    def get_config(self) -> dict[str, Any]:
        """Return the flat constructor config plus training provenance."""

        config = super().get_config()
        config.update(
            {
                "epochs": self.epochs,
                "generator_learning_rate": self.generator_learning_rate,
                "discriminator_learning_rate": self.discriminator_learning_rate,
                "meta_learning_rate": self.meta_learning_rate,
                "beta1": self.betas[0],
                "beta2": self.betas[1],
                "weight_decay": self.weight_decay,
                "gradient_clip": self.gradient_clip,
                "poisson_blend": self.poisson_blend,
                # Fixed settings and provenance, not constructor arguments.
                "target_tensor_sha256": self.target_sha256,
                "training_dataset": self._train_dataset_config,
                "batch_size": self._BATCH_SIZE,
                "training_face_shape": [3, 600, 600],
                "eye_shape": [3, *EYE_INPUT_SIZE],
                "external_range": [0.0, 255.0],
                "internal_range": [-1.0, 1.0],
                "optimizer": "Adam",
                "lr_decay_factor": 0.1,
                "lr_decay_period_epochs": 100,
                "surrogate_order": [
                    self.fr_models[index].name for index in self.surrogate_order
                ],
            }
        )
        return config

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """Build the LADN eye-region pair loader over ``corpus``."""
        from .data import AdvMakeupDataset

        loader = DataLoader(
            AdvMakeupDataset.from_corpus(corpus),
            batch_size=self._BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                f"Corpus {corpus.name!r} has fewer than {self._BATCH_SIZE} "
                "training pairs after filtering"
            )
        return loader

    def load_run(self, load_path: str | Path) -> Path:
        """Load the trained encoder, decoder and, if present, discriminator; the VGG and
        landmark priors come from the asset cache.
        """
        root = super().load_run(load_path)
        for network, name in (
            (self.encoder, "encoder"),
            (self.decoder, "decoder"),
        ):
            path = root / f"{name}.pth"
            if not path.is_file():
                raise FileNotFoundError(
                    f"No Adv-Makeup {name} checkpoint under {root}"
                )
            network.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True),
                strict=True,
            )
        discriminator_path = root / "discriminator.pth"
        if discriminator_path.is_file():
            self.discriminator.load_state_dict(
                torch.load(
                    discriminator_path,
                    map_location=self.device,
                    weights_only=True,
                ),
                strict=True,
            )
        self.encoder.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)
        self.discriminator.eval().requires_grad_(False)
        self._trained = True
        self._mark_ready()
        return root

    @staticmethod
    def _cpu_state_dict(network: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().to("cpu").clone()
            for key, value in network.state_dict().items()
        }

    def save_run(self, save_path: str | Path) -> Path:
        """Save the encoder, decoder, discriminator and baked target image (not the priors)."""

        self._require_savable()
        root = Path(save_path)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(
            self._cpu_state_dict(self.encoder), root / "encoder.pth"
        )
        torch.save(
            self._cpu_state_dict(self.decoder), root / "decoder.pth"
        )
        torch.save(
            self._cpu_state_dict(self.discriminator),
            root / "discriminator.pth",
        )
        inputs = (
            self._write_input_images(root, "target", self._baked_target)
            if self._baked_target is not None
            else []
        )
        return super().save_run(
            root,
            weights=["encoder.pth", "decoder.pth", "discriminator.pth"],
            baked_inputs=inputs,
        )

    @staticmethod
    def _validate_external_images(
        images: torch.Tensor,
        *,
        name: str,
        require_single: bool = False,
    ) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"{name} must have shape [B,3,H,W]")
        if images.shape[0] == 0:
            raise ValueError(f"{name} must be non-empty")
        if require_single and images.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one image")
        if not torch.is_floating_point(images):
            images = images.to(torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains non-finite values")
        minimum, maximum = float(images.min()), float(images.max())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"{name} must be RGB in [0,255], got [{minimum},{maximum}]"
            )
        return images.to(torch.float32)

    @staticmethod
    def _validate_training_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 4:
            raise ValueError(
                "Adv-Makeup loader must emit "
                "(source_eye, makeup_eye, source_face, source_landmarks)"
            )
        before, after, full, landmarks = batch
        if before.ndim != 4 or tuple(before.shape[1:]) != (
            3,
            *EYE_INPUT_SIZE,
        ):
            raise ValueError("source_eye must have shape [B,3,160,420]")
        if after.shape != before.shape:
            raise ValueError("makeup_eye must match source_eye shape")
        if full.ndim != 4 or tuple(full.shape[1:]) != (3, 600, 600):
            raise ValueError("source_face must have shape [B,3,600,600]")
        if not all(
            torch.is_floating_point(value)
            for value in (before, after, full, landmarks)
        ):
            raise TypeError("Adv-Makeup training tensors must be floating point")
        for name, value in (
            ("source_eye", before),
            ("makeup_eye", after),
            ("source_face", full),
        ):
            minimum, maximum = float(value.min()), float(value.max())
            if minimum < -1.0001 or maximum > 1.0001:
                raise ValueError(
                    f"{name} must be in [-1,1], got [{minimum},{maximum}]"
                )
        validate_landmarks(landmarks, before.shape[0])
        return before, after, full, landmarks

    @staticmethod
    def _set_trainable(
        networks: Sequence[nn.Module], enabled: bool
    ) -> None:
        for network in networks:
            network.requires_grad_(enabled)
            # Batch-norm running statistics are Parameters only to keep the
            # released checkpoint keys; they stay out of autograd.
            for name in ("vars_bn_enc", "vars_bn_dec"):
                running_statistics = getattr(network, name, ())
                for statistic in running_statistics:
                    statistic.requires_grad_(False)

    def _embed_internal(self, fr: FaceRecognizer, images: torch.Tensor) -> torch.Tensor:
        # Inputs are pre-cropped [-1, 1] face canvases, passed without clamping;
        # embed_crops applies each backend's own resize and normalization.
        embeddings = fr.embed_crops(images, input_space="pm1", antialias=True)
        if embeddings.shape[0] != images.shape[0]:
            raise RuntimeError(
                "FR recognizer dropped a pre-cropped Adv-Makeup image"
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise RuntimeError(
                f"Expected FR embeddings [B,512], got {tuple(embeddings.shape)}"
            )
        return embeddings

    def _target_loss(
        self,
        composites: torch.Tensor,
        target_embedding: torch.Tensor,
        surrogate_index: int,
    ) -> torch.Tensor:
        embedding = self._embed_internal(
            self.fr_models[surrogate_index], composites
        )
        return cosine_distance(embedding, target_embedding)

    def _new_optimizers(
        self, generator_lr: float, discriminator_lr: float
    ) -> tuple[torch.optim.Adam, torch.optim.Adam]:
        generator = torch.optim.Adam(
            self.encoder.fast_parameters()
            + self.decoder.fast_parameters(),
            lr=generator_lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return generator, discriminator

    def _rotate_surrogates(self) -> None:
        """Rotate which surrogate is held out for the meta-test step."""

        shift = len(self.surrogate_order) - 1
        self.surrogate_order = (
            self.surrogate_order[shift:] + self.surrogate_order[:shift]
        )

    def train(self, data: TrainData) -> None:
        """Train a target-specific generator with leave-one-model-out meta loss.

        Args:
            data.corpus: The LADN corpus, read as eye-region before/makeup
                pairs by :class:`~afrlib.afr.adv_makeup.data.AdvMakeupDataset`.
            data.target_img: The impersonation target baked into the
                generator, one ``[1,3,H,W]`` RGB image in ``[0, 255]``.
        """
        self._check_inputs("train", data)
        self._require_fr()
        train_loader = self._build_loader(data.corpus)
        self._begin_training()
        self._train_dataset_config = dataset_config(train_loader)
        target = self._validate_external_images(
            data.target_img, name="target_img", require_single=True
        )
        self._baked_target = target.detach().cpu()
        self.target_sha256 = hashlib.sha256(
            target.detach().to("cpu").contiguous().numpy().tobytes()
        ).hexdigest()
        target_internal = target.to(self.device).div(127.5).sub(1.0)

        self.encoder = Encoder().to(self.device).train()
        self.decoder = Decoder().to(self.device).train()
        self.discriminator = initialize_discriminator(
            Discriminator()
        ).to(self.device).train()
        self.surrogate_order = list(range(len(self.fr_models)))

        vgg_path = get_asset("adv_makeup/vgg16-397923af.pth")
        vgg = VGG16Features(vgg_path).to(self.device).eval()
        mean_shift = MeanShift().to(self.device).eval()
        with torch.no_grad():
            target_embeddings = [
                self._embed_internal(fr, target_internal).detach()
                for fr in self.fr_models
            ]

        generator_lr = self.generator_learning_rate
        discriminator_lr = self.discriminator_learning_rate
        generator_optimizer, discriminator_optimizer = self._new_optimizers(
            generator_lr, discriminator_lr
        )

        progress = tqdm(range(self.epochs), desc="Adv-Makeup training")
        for epoch in progress:
            if (epoch + 1) % 100 == 0:
                generator_lr /= 10.0
                discriminator_lr /= 10.0
                (
                    generator_optimizer,
                    discriminator_optimizer,
                ) = self._new_optimizers(generator_lr, discriminator_lr)

            epoch_values: list[dict[str, float]] = []
            for raw_batch in train_loader:
                before_cpu, after_cpu, full_cpu, landmarks_cpu = (
                    self._validate_training_batch(raw_batch)
                )
                before = before_cpu.to(self.device).detach()
                after = after_cpu.to(self.device).detach()
                full = full_cpu.to(self.device).detach()
                landmarks = landmarks_cpu.to(self.device).detach()

                # Discriminator step first, as in the released code; the fake
                # strips are detached so no generator gradients accumulate.
                self._set_trainable((self.encoder, self.decoder), False)
                self._set_trainable((self.discriminator,), True)
                with torch.no_grad():
                    fake_after = self.decoder(*self.encoder(before))
                discriminator_optimizer.zero_grad(set_to_none=True)
                d_real = discriminator_mse(
                    self.discriminator(after), real=True
                )
                d_fake = discriminator_mse(
                    self.discriminator(fake_after.detach()), real=False
                )
                d_loss = d_real + d_fake
                d_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.gradient_clip
                )
                discriminator_optimizer.step()

                self._set_trainable((self.encoder, self.decoder), True)
                self._set_trainable((self.discriminator,), False)
                generator_optimizer.zero_grad(set_to_none=True)
                first_feature, second_feature = self.encoder(before)
                fake_after = self.decoder(first_feature, second_feature)
                generated = composite_generated_eye(
                    full, fake_after, landmarks
                )

                meta_train = self.surrogate_order[:-1]
                meta_test = self.surrogate_order[-1]
                target_terms = [
                    self._target_loss(
                        generated.full_images,
                        target_embeddings[index],
                        index,
                    )
                    for index in meta_train
                ]
                for train_index in meta_train:
                    meta_train_loss = self._target_loss(
                        generated.full_images,
                        target_embeddings[train_index],
                        train_index,
                    )
                    encoder_gradients = torch.autograd.grad(
                        meta_train_loss,
                        self.encoder.fast_parameters(),
                        retain_graph=True,
                    )
                    decoder_gradients = torch.autograd.grad(
                        meta_train_loss,
                        self.decoder.fast_parameters(),
                        retain_graph=True,
                    )
                    fast_encoder = [
                        parameter - self.meta_learning_rate * gradient
                        for parameter, gradient in zip(
                            self.encoder.fast_parameters(),
                            encoder_gradients,
                        )
                    ]
                    fast_decoder = [
                        parameter - self.meta_learning_rate * gradient
                        for parameter, gradient in zip(
                            self.decoder.fast_parameters(),
                            decoder_gradients,
                        )
                    ]
                    fast_features = self.encoder(
                        before, fast_encoder
                    )
                    fast_eye = self.decoder(
                        *fast_features, fast_decoder
                    )
                    fast_composite = composite_generated_eye(
                        full, fast_eye, landmarks
                    )
                    target_terms.append(
                        self._target_loss(
                            fast_composite.full_images,
                            target_embeddings[meta_test],
                            meta_test,
                        )
                    )
                target_loss = torch.stack(target_terms).mean()

                loss_style = style_loss(
                    vgg,
                    mean_shift,
                    generated.original_crops,
                    generated.composite_crops,
                )
                loss_gradient = gradient_loss(
                    generated.original_crops,
                    generated.composite_crops,
                )
                loss_tv = total_variation_loss(
                    generated.composite_crops
                )
                loss_gan = discriminator_mse(
                    self.discriminator(fake_after), real=True
                )
                generator_loss = (
                    loss_gan
                    + target_loss
                    + loss_style
                    + loss_gradient
                    + loss_tv
                )
                generator_loss.backward()
                # Upstream clips the discriminator's gradients at this point;
                # the generator's are clipped here instead.
                nn.utils.clip_grad_norm_(
                    self.encoder.fast_parameters()
                    + self.decoder.fast_parameters(),
                    self.gradient_clip,
                )
                generator_optimizer.step()

                epoch_values.append(
                    {
                        "discriminator_real": float(d_real.detach()),
                        "discriminator_fake": float(d_fake.detach()),
                        "generator": float(generator_loss.detach()),
                        "gan": float(loss_gan.detach()),
                        "target": float(target_loss.detach()),
                        "style": float(loss_style.detach()),
                        "gradient": float(loss_gradient.detach()),
                        "tv": float(loss_tv.detach()),
                    }
                )

            summary = {
                key: sum(value[key] for value in epoch_values)
                / len(epoch_values)
                for key in epoch_values[0]
            }
            self._rotate_surrogates()
            progress.set_postfix(
                generator=f"{summary['generator']:.4f}",
                target=f"{summary['target']:.4f}",
            )

        self.encoder.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)
        self.discriminator.eval().requires_grad_(False)
        self._mark_ready()

    def _get_landmark_detector(self) -> Landmark106Detector:
        if self._landmark_detector is None:
            checkpoint = get_asset("adv_makeup/2d106det.onnx")
            self._landmark_detector = Landmark106Detector(
                checkpoint, self.device
            )
        return self._landmark_detector

    def protect(self, data: EvalData) -> list[torch.Tensor]:
        """Generate makeup on each face's eye region and paste the face back; an image without
        usable 106-point landmarks keeps its crop.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        photos = [
            self._validate_external_images(
                photo.unsqueeze(0), name="eval_imgs", require_single=True
            )[0]
            for photo in data.eval_imgs
        ]
        crops, plans = prepare_inputs(photos, data.eval_geometry, self.attack_spec())
        detector = self._get_landmark_detector()
        protected: list[torch.Tensor] = []
        with torch.inference_mode():
            for index, crop in enumerate(crops):
                source = crop.unsqueeze(0).to(self.device)
                try:
                    landmarks = detector(source).to(self.device)
                    image_size = tuple(map(int, source.shape[-2:]))
                    geometries = [
                        build_orbital_geometry(landmark, image_size)
                        for landmark in landmarks
                    ]
                except (RuntimeError, ValueError):
                    logger.warning(
                        "Adv-Makeup source %s: no usable 106-point landmarks; "
                        "returning the source crop",
                        index,
                    )
                    protected.append(crop.to(self.device).to(torch.float32))
                    continue
                eyes = torch.stack(
                    [
                        eye_crop_to_internal(image, geometry)
                        for image, geometry in zip(source, geometries)
                    ]
                ).to(self.device)
                generated_eyes = self.decoder(
                    *self.encoder(eyes, bn_training=False),
                    bn_training=False,
                )
                composite = composite_generated_eye(
                    source.div(127.5).sub(1.0),
                    generated_eyes,
                    landmarks,
                )
                raw = (
                    composite.full_images.add(1.0)
                    .mul(127.5)
                    .clamp(0, 255)
                    .cpu()
                )
                if self.poisson_blend:
                    blended = poisson_restore_batch(
                        source,
                        eyes,
                        composite.generated_crops,
                        composite.geometries,
                    )
                    protected.append(blended[0].to(torch.float32))
                else:
                    protected.append(raw[0].to(torch.float32))
        return paste_faces(photos, protected, plans)
