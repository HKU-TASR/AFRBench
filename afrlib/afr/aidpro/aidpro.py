"""AIDPro: privacy-preserving faces that carry an authentication watermark.

Reference Paper: Beyond Privacy: Generating Privacy-Preserving Faces Supporting Robust Image Authentication (IEEE TIFS 2025)
Reference Code: https://github.com/daizigege/AIDPro
"""
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
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

from .models import (
    Generator_Adain_Upsample,
    WatermarkDecoder,
    WatermarkInsert,
)

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus


logger = get_logger(__name__)

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


class AIDPro(AFRMethod):
    """A frozen face manipulation model swaps each face's identity for a watermarked code
    ``WI([embedding, message])``; ``WD`` recovers the message from the protected face.

    Args:
        primary_index: Index of the surrogate whose 512-D embedding conditions WI and WD.
        message_length: Bits per watermark message.
        epochs: Training epochs.
        batch_size: Training batch size.
        max_steps: Cap on training steps; ``None`` for none.
        learning_rate: Adam learning rate.
        beta1: Adam beta1.
        beta2: Adam beta2.
        lambda_identity: Weight of the privacy loss.
        lambda_reconstruction: Weight of the reconstruction loss.
        lambda_watermark: Weight of the message loss.
    """

    name = "AIDPro"
    training_corpus = "vggface2"
    train_inputs = StageInputs(required=("corpus",))
    eval_inputs = StageInputs(required=("eval_imgs",), optional=("message",))
    # Non-finite training steps are skipped; this many in a row aborts
    # training, since WI has then diverged.
    max_consecutive_nonfinite_steps = 8

    def __init__(
        self,
        device: torch.device | str,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: Optional[str | Path] = None,
        primary_index: int = 0,
        message_length: int = 30,
        epochs: int = 2,
        batch_size: int = 32,
        max_steps: Optional[int] = None,
        learning_rate: float = 1e-4,
        beta1: float = 0.999,
        beta2: float = 0.999,
        lambda_identity: float = 5.0,
        lambda_reconstruction: float = 5.0,
        lambda_watermark: float = 10.0,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        if primary_index < 0:
            raise ValueError("primary_index must be non-negative")
        self.primary_index = int(primary_index)

        if message_length < 1:
            raise ValueError("message_length must be positive")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.message_length = int(message_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.learning_rate = float(learning_rate)
        self.betas = (float(beta1), float(beta2))
        self.lambda_identity = float(lambda_identity)
        self.lambda_reconstruction = float(lambda_reconstruction)
        self.lambda_watermark = float(lambda_watermark)

        self._mean = torch.tensor(
            _IMAGE_MEAN, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            _IMAGE_STD, device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        path = Path(get_asset("aidpro/90000_net_G.pth")).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        generator = Generator_Adain_Upsample()
        generator.load_state_dict(state, strict=True)
        self.fmm = generator
        self.fmm.to(self.device).eval().requires_grad_(False)

        self.wi = WatermarkInsert(512 + self.message_length).to(self.device)
        self.wd = WatermarkDecoder(self.message_length).to(self.device)
        self._lpips: Optional[nn.Module] = None
        self._optimizer_wi: Optional[torch.optim.Optimizer] = None
        self._optimizer_wd: Optional[torch.optim.Optimizer] = None
        self._pending_optimizer_states: Optional[dict[str, Any]] = None
        self.actual_steps = 0
        self.last_messages: Optional[torch.Tensor] = None
        self._train_dataset_config: Optional[dict[str, Any]] = None

        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=224,
            channel=ChannelType.RGB,
            crop_type=CropType.FaceBox,
            align_type=AlignType.ArcFace5,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "primary_index": self.primary_index,
                "message_length": self.message_length,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
                "beta1": self.betas[0],
                "beta2": self.betas[1],
                "lambda_identity": self.lambda_identity,
                "lambda_reconstruction": self.lambda_reconstruction,
                "lambda_watermark": self.lambda_watermark,
                # Fixed settings and provenance, not constructor arguments.
                "training_dataset": self._train_dataset_config,
                "actual_steps": self.actual_steps,
                "image_size": 224,
                "external_range": [0.0, 255.0],
                "internal_mean": list(_IMAGE_MEAN),
                "internal_std": list(_IMAGE_STD),
                "optimizer": "Adam",
                "optimizer_eps": 1e-8,
                "weight_decay": 0.0,
                "reconstruction_l1_weight": 0.2,
                "reconstruction_lpips_weight": 0.8,
                "privacy_reduction": (
                    "single_cosine_hinge"
                    if len(self.fr_models) == 1
                    else "mean_of_surrogate_cosine_hinges"
                ),
            }
        )
        return config

    @staticmethod
    def _validate_images(images: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if (
            images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1:] != (3, 224, 224)
        ):
            raise ValueError(
                f"{name} must have non-empty shape [B,3,224,224], "
                f"got {tuple(images.shape)}"
            )
        images = images.to(torch.float32)
        if not torch.isfinite(images).all():
            raise ValueError(f"{name} contains NaN or Inf")
        minimum = float(images.detach().amin())
        maximum = float(images.detach().amax())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"{name} must be RGB in [0,255], "
                f"got [{minimum:.4f},{maximum:.4f}]"
            )
        return images

    def _to_internal(self, images255: torch.Tensor) -> torch.Tensor:
        images255 = images255.to(self.device, dtype=torch.float32)
        return (images255.div(255.0) - self._mean) / self._std

    def _to_public(self, internal: torch.Tensor) -> torch.Tensor:
        return (
            (internal * self._std + self._mean)
            .mul(255.0)
            .clamp(0.0, 255.0)
        )

    @property
    def min_surrogates(self) -> int:
        """Enough surrogates for ``primary_index`` to address one."""
        return self.primary_index + 1

    def _primary_recognizer(self) -> FaceRecognizer:
        """Validate and return the surrogate whose embedding conditions WI/WD."""
        self._require_fr()
        primary = self.fr_models[self.primary_index]
        # Only the primary embedding conditions WI/WD, so only it must be 512-D.
        if int(primary.embedding_dim) != 512:
            raise ValueError(
                "AIDPro requires a 512-D primary recognizer embedding, got "
                f"{primary.embedding_dim} ({primary.name})"
            )
        return primary

    def _training_recognizers(self) -> list[FaceRecognizer]:
        self._primary_recognizer()
        return list(self.fr_models)

    def _embed_internal(
        self, recognizer: FaceRecognizer, internal: torch.Tensor
    ) -> torch.Tensor:
        """L2-normalized embeddings of ImageNet-normalized faces, through the recognizer's own
        bicubic resize (no antialiasing).
        """
        rgb01 = internal * self._std + self._mean
        embeddings = recognizer.embed_crops(
            rgb01,
            input_space="01",
            mode="bicubic",
            antialias=False,
            l2_normalize=True,
        )
        if embeddings.shape != (internal.shape[0], 512):
            raise RuntimeError(
                f"Expected FR embeddings [B,512], got {embeddings.shape}"
            )
        return embeddings

    def _validate_messages(
        self, messages: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if not isinstance(messages, torch.Tensor):
            raise TypeError("messages must be a torch.Tensor")
        if messages.shape != (batch_size, self.message_length):
            raise ValueError(
                "messages must have shape "
                f"[{batch_size},{self.message_length}], got "
                f"{tuple(messages.shape)}"
            )
        messages = messages.to(self.device, dtype=torch.float32)
        if not torch.all((messages == 0) | (messages == 1)):
            raise ValueError("messages must contain only binary 0/1 values")
        return messages

    def _sample_messages(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            0,
            2,
            (batch_size, self.message_length),
            device=self.device,
            dtype=torch.int64,
        ).to(torch.float32)

    def _generate_internal(
        self,
        source_internal: torch.Tensor,
        source_primary: torch.Tensor,
        messages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wi_inputs = torch.cat([source_primary, messages], dim=1)
        transformed_identity = F.normalize(
            self.wi(wi_inputs), p=2, dim=1
        )
        protected_internal = self.fmm(
            source_internal, transformed_identity
        )
        return protected_internal, transformed_identity

    def _ensure_training_components(self) -> list[FaceRecognizer]:
        recognizers = self._training_recognizers()
        if self._lpips is None:
            import lpips

            self._lpips = (
                lpips.LPIPS(net="alex")
                .to(self.device)
                .eval()
                .requires_grad_(False)
            )
        if self._optimizer_wi is None:
            self._optimizer_wi = torch.optim.Adam(
                self.wi.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
            )
            self._optimizer_wd = torch.optim.Adam(
                self.wd.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
            )
            if self._pending_optimizer_states is not None:
                self._optimizer_wi.load_state_dict(
                    self._pending_optimizer_states["wi"]
                )
                assert self._optimizer_wd is not None
                self._optimizer_wd.load_state_dict(
                    self._pending_optimizer_states["wd"]
                )
                self._pending_optimizer_states = None
        return recognizers

    @staticmethod
    def _privacy_loss(
        source_embeddings: Sequence[torch.Tensor],
        protected_embeddings: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if (
            len(source_embeddings) == 0
            or len(source_embeddings) != len(protected_embeddings)
        ):
            raise ValueError(
                "AIDPro privacy loss requires matching non-empty model pairs"
            )
        terms = [
            F.cosine_similarity(source, protected, dim=1)
            .clamp_min(0.0)
            .mean()
            for source, protected in zip(
                source_embeddings, protected_embeddings
            )
        ]
        return torch.stack(terms).mean()

    @staticmethod
    def _term_status(value: torch.Tensor) -> str:
        if torch.isfinite(value).all():
            return f"{float(value.detach()):.4g}"
        return "nonfinite"

    def _train_step(
        self, images255: torch.Tensor, recognizers: Sequence[FaceRecognizer]
    ) -> dict[str, float] | None:
        assert self._lpips is not None
        assert self._optimizer_wi is not None
        assert self._optimizer_wd is not None
        source_internal = self._to_internal(
            self._validate_images(images255, "training images")
        )
        messages = self._sample_messages(source_internal.shape[0])

        with torch.no_grad():
            source_embeddings = [
                self._embed_internal(recognizer, source_internal)
                for recognizer in recognizers
            ]
        protected_internal, _ = self._generate_internal(
            source_internal,
            source_embeddings[self.primary_index],
            messages,
        )
        protected_embeddings = [
            self._embed_internal(recognizer, protected_internal)
            for recognizer in recognizers
        ]

        identity_loss = self._privacy_loss(
            source_embeddings, protected_embeddings
        )
        l1_loss = F.l1_loss(source_internal, protected_internal)
        lpips_loss = self._lpips(
            source_internal, protected_internal
        ).mean()
        reconstruction_loss = 0.2 * l1_loss + 0.8 * lpips_loss
        logits = self.wd(protected_embeddings[self.primary_index])
        watermark_loss = F.binary_cross_entropy_with_logits(
            logits, messages
        )
        total_loss = (
            self.lambda_identity * identity_loss
            + self.lambda_reconstruction * reconstruction_loss
            + self.lambda_watermark * watermark_loss
        )
        if not torch.isfinite(total_loss):
            logger.warning(
                "AIDPro skip non-finite step (id=%s l1=%s lpips=%s wm=%s |fmm|=%s)",
                self._term_status(identity_loss),
                self._term_status(l1_loss),
                self._term_status(lpips_loss),
                self._term_status(watermark_loss),
                self._term_status(protected_internal.detach().abs().amax()),
            )
            return None

        self._optimizer_wi.zero_grad(set_to_none=True)
        self._optimizer_wd.zero_grad(set_to_none=True)
        total_loss.backward()
        self._optimizer_wi.step()
        self._optimizer_wd.step()
        self.actual_steps += 1

        with torch.no_grad():
            bit_accuracy = (
                (logits.sigmoid() > 0.5) == (messages > 0.5)
            ).to(torch.float32).mean()
        return {
            "total": float(total_loss.detach()),
            "identity": float(identity_loss.detach()),
            "reconstruction": float(reconstruction_loss.detach()),
            "l1": float(l1_loss.detach()),
            "lpips": float(lpips_loss.detach()),
            "watermark": float(watermark_loss.detach()),
            "bit_accuracy": float(bit_accuracy.detach()),
            "actual_step": float(self.actual_steps),
        }

    def train(self, data: TrainData) -> None:
        """Train WI and WD; the FMM, FR models, and LPIPS stay frozen.

        Args:
            data.corpus: The VGGFace2 corpus, read as aligned RGB faces in
                ``[0, 255]`` by :class:`~afrlib.afr.aidpro.data.VGGFace2Dataset`.
        """
        self._check_inputs("train", data)
        max_steps = self.max_steps
        train_loader = self._build_loader(data.corpus)
        recognizers = self._ensure_training_components()
        self._begin_training()
        self._train_dataset_config = dataset_config(train_loader)
        self.fmm.eval().requires_grad_(False)
        self.wi.train()
        self.wd.train()

        total = self.epochs * len(train_loader)
        if max_steps is not None:
            total = min(total, max_steps)
        progress = tqdm(total=total, desc="AIDPro stage-two training")
        consecutive_nonfinite = 0
        for _epoch in range(self.epochs):
            epoch_values: list[dict[str, float]] = []
            for batch in train_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                values = self._train_step(images, recognizers)
                progress.update(1)
                if values is None:
                    consecutive_nonfinite += 1
                    progress.set_postfix(
                        loss="skip",
                        step=self.actual_steps,
                    )
                    if consecutive_nonfinite >= self.max_consecutive_nonfinite_steps:
                        progress.close()
                        raise FloatingPointError(
                            "AIDPro training loss is non-finite for "
                            f"{consecutive_nonfinite} consecutive steps"
                        )
                    continue
                consecutive_nonfinite = 0
                epoch_values.append(values)
                progress.set_postfix(
                    loss=f"{values['total']:.4f}",
                    bit_acc=f"{values['bit_accuracy']:.3f}",
                    step=self.actual_steps,
                )
                if max_steps is not None and self.actual_steps >= max_steps:
                    break
            if epoch_values:
                logger.info(
                    "AIDPro epoch completed at actual step %d (loss %.4f)",
                    self.actual_steps,
                    sum(value["total"] for value in epoch_values)
                    / len(epoch_values),
                )
            if max_steps is not None and self.actual_steps >= max_steps:
                break
        progress.close()
        self.wi.eval()
        self.wd.eval()
        self._mark_ready()

    def protect(self, data: EvalData) -> list[torch.Tensor]:
        """Generate a watermarked face for each photo and paste it back; the messages used are
        kept in ``last_messages``.

        Args:
            data.message: Optional ``[B, message_length]`` binary messages; random when omitted.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        images = self._validate_images(torch.stack(crops), "eval_imgs").to(self.device)
        if data.message is None:
            messages = self._sample_messages(images.shape[0])
        else:
            messages = self._validate_messages(data.message, images.shape[0])
        primary = self._primary_recognizer()
        self.wi.eval()
        self.wd.eval()
        self.fmm.eval()
        with torch.inference_mode():
            source_internal = self._to_internal(images)
            source_embedding = self._embed_internal(
                primary, source_internal
            )
            protected_internal, _ = self._generate_internal(
                source_internal, source_embedding, messages
            )
            protected = self._to_public(protected_internal)
        self.last_messages = messages.detach().to("cpu", dtype=torch.uint8)
        return paste_faces(photos, protected, plans)

    def extract_messages(self, protected_imgs: torch.Tensor) -> torch.Tensor:
        """Message-bit probabilities ``[B, message_length]`` of aligned protected faces.

        Args:
            protected_imgs: Aligned 224x224 RGB faces in ``[0, 255]``,
                ``[B, 3, 224, 224]``.
        """
        images = self._validate_images(
            protected_imgs, "protected_imgs"
        ).to(self.device)
        primary = self._primary_recognizer()
        self.wd.eval()
        with torch.inference_mode():
            internal = self._to_internal(images)
            embeddings = self._embed_internal(primary, internal)
            return self.wd(embeddings).sigmoid()

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """Shuffled stage-two loader over the VGGFace2 corpus (``drop_last``)."""
        from .data import VGGFace2Dataset

        loader = DataLoader(
            VGGFace2Dataset.from_corpus(corpus),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                f"Corpus {corpus.name!r} has fewer than {self.batch_size} "
                "training samples after filtering"
            )
        return loader

    @staticmethod
    def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    def save_run(self, save_path: str | Path) -> Path:
        """Save WI/WD and their optimizer state (atomically), then the config."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = root / "aidpro.pt"
        checkpoint_tmp = root / ".aidpro.pt.tmp"
        state: dict[str, Any] = {
            "format": 1,
            "method": "AIDPro",
            "wi": self._cpu_state_dict(self.wi),
            "wd": self._cpu_state_dict(self.wd),
            "actual_steps": self.actual_steps,
        }
        if self._optimizer_wi is not None and self._optimizer_wd is not None:
            state["optimizers"] = {
                "wi": self._optimizer_wi.state_dict(),
                "wd": self._optimizer_wd.state_dict(),
            }
        torch.save(state, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        return super().save_run(root, weights=[checkpoint_path.name])

    def load_run(self, load_path: str | Path) -> Path:
        """Load WI/WD and any optimizer state from ``aidpro.pt``; ``fr_models`` must be the
        surrogates the weights were trained with (not checked).
        """
        root = super().load_run(load_path)
        path = root / "aidpro.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "wi" in state and "wd" in state:
            wi_state, wd_state = state["wi"], state["wd"]
            optimizers = state.get("optimizers")
            self.actual_steps = int(state.get("actual_steps", 0))
        else:
            raise ValueError(f"Unrecognized AIDPro checkpoint: {path}")
        self.wi.load_state_dict(wi_state, strict=True)
        self.wd.load_state_dict(wd_state, strict=True)
        if optimizers is not None:
            self._pending_optimizer_states = optimizers
        self.wi.eval()
        self.wd.eval()
        self._trained = True
        self._mark_ready()
        return root


__all__ = ["AIDPro"]
