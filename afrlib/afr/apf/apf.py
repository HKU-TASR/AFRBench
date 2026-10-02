"""APF: an adversarial privacy-preserving filter for end-cloud face recognition.

Reference Paper: Adversarial Privacy-preserving Filter (ACM MM 2020)
Reference Code: https://github.com/adversarial-for-goodness/APF (GPL-3.0; see LICENSE)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch
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
from afrlib.afr.base import AFRMethod, read_run_config
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.afr.datasets import dataset_config
from afrlib.fr import FaceRecognizer
from afrlib.utils.log import get_logger, tqdm

from .attack import EPSILON, N_ITER, STEP_SIZE, i2fgsm, squared_l2_distance
from .models import Enhancement, UNetTransfer

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

logger = get_logger(__name__)

_IMAGE_SIZE = 112


class APF(AFRMethod):
    """Probe-model noise, transferred to the server models by ``T`` and enhanced by a universal
    perturbation: ``clamp(x + clip_eps(T(noise) + E(u)), 0, 255)``.

    Args:
        probe_index: Index in ``fr_models`` of the probe that generates the I2-FGSM noise; the
            other models are the servers, weighted equally.
        epochs: Training epochs of ``T`` and ``E``.
        batch_size: Training batch size.
        learning_rate: Adam learning rate of ``T`` and ``E``.
        epsilon: L-inf budget, in 8-bit levels.
        step_size: I2-FGSM step size, in 8-bit levels.
        n_iter: I2-FGSM iterations.
        uap_steps: Gradient-ascent steps of the universal perturbation.
        uap_step_size: Step size of the universal perturbation, in 8-bit levels.
        uap_pairs: Training images in the universal perturbation's objective.
        inference_batch_size: Chunk size for embedding and protection.
        enrolled_noise: Has no effect; accepted for configuration compatibility.
    """

    name = "APF"
    training_corpus = "ms1m"
    train_inputs = StageInputs(required=("corpus",))
    fit_inputs = StageInputs(required=("protectee_album",))
    eval_inputs = StageInputs(required=("eval_imgs",))
    def __init__(
        self,
        device: torch.device | str,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: Optional[str | Path] = None,
        probe_index: int = 0,
        epochs: int = 50,
        batch_size: int = 100,
        learning_rate: float = 1e-4,
        epsilon: float = EPSILON,
        step_size: float = STEP_SIZE,
        n_iter: int = N_ITER,
        uap_steps: int = 20,
        uap_step_size: float = 2.0,
        uap_pairs: int = 200,
        inference_batch_size: int = 32,
        enrolled_noise: bool = False,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        if probe_index < 0:
            raise ValueError("probe_index must be non-negative")
        self.probe_index = int(probe_index)

        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.n_iter = int(n_iter)
        self.uap_steps = int(uap_steps)
        self.uap_step_size = float(uap_step_size)
        self.uap_pairs = int(uap_pairs)
        self.inference_batch_size = int(inference_batch_size)
        self.enrolled_noise = bool(enrolled_noise)

        self.unet = UNetTransfer().to(self.device)
        # The enhancement module is built around the UAP, which is only known
        # after training (or loading); start from a zero UAP placeholder.
        self.enhance = Enhancement(torch.zeros(1, 3, _IMAGE_SIZE, _IMAGE_SIZE)).to(self.device)
        self._train_dataset_config: Optional[Dict[str, Any]] = None
        # Registered by fit(): the protectee's enrolled gallery probes.
        self.enrolled_album: Optional[torch.Tensor] = None
        self._album_offset = 0
        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=_IMAGE_SIZE,
            channel=ChannelType.RGB,
            crop_type=CropType.FaceBox,
            align_type=AlignType.ArcFace5,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "probe_index": self.probe_index,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "step_size": self.step_size,
                "n_iter": self.n_iter,
                "uap_steps": self.uap_steps,
                "uap_step_size": self.uap_step_size,
                "uap_pairs": self.uap_pairs,
                "inference_batch_size": self.inference_batch_size,
                "enrolled_noise": self.enrolled_noise,
                # Provenance, not constructor arguments.
                "training_dataset": self._train_dataset_config,
                "probe_model": (
                    self.fr_models[self.probe_index].name
                    if self.probe_index < len(self.fr_models)
                    else None
                ),
                "server_models": [
                    m.name for i, m in enumerate(self.fr_models) if i != self.probe_index
                ],
            }
        )
        return config

    # FR access

    @property
    def min_surrogates(self) -> int:
        """A probe plus at least one server model, with ``probe_index`` valid."""
        return max(2, self.probe_index + 1)

    def _probe(self) -> FaceRecognizer:
        """Validate the injected ensemble at the point of use, then pick the probe."""
        self._require_fr()
        return self.fr_models[self.probe_index]

    def _servers(self) -> List[FaceRecognizer]:
        return [m for i, m in enumerate(self.fr_models) if i != self.probe_index]

    def _embed(self, fr: FaceRecognizer, imgs255: torch.Tensor) -> torch.Tensor:
        """Embed ``[0, 255]`` RGB crops with one FR model (no detection/align)."""
        return fr.embed_crops(imgs255.contiguous() / 255.0, input_space="01", antialias=True)

    def _embed_batched(self, fr: FaceRecognizer, imgs255: torch.Tensor) -> torch.Tensor:
        """``_embed`` in checkpointed chunks of ``inference_batch_size``, so the UAP backward fits."""
        from torch.utils.checkpoint import checkpoint

        chunk = self.inference_batch_size
        parts = []
        for start in range(0, imgs255.shape[0], chunk):
            piece = imgs255[start : start + chunk]
            parts.append(
                checkpoint(lambda x: self._embed(fr, x), piece, use_reentrant=False)
            )
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def _server_distance(self, imgs255: torch.Tensor, enrolled255: torch.Tensor) -> torch.Tensor:
        """Per-image squared-L2 embedding distance to ``enrolled255``, averaged over the servers."""
        total = None
        for fr in self._servers():
            d = squared_l2_distance(
                self._embed_batched(fr, imgs255), self._embed_batched(fr, enrolled255)
            )
            total = d if total is None else total + d
        return total / len(self._servers())

    # Universal perturbation (stage C)

    def _pairwise_server_distance(self, imgs255: torch.Tensor) -> torch.Tensor:
        """Mean pairwise squared-L2 distance between batch embeddings, averaged over the servers
        (the inner sum of paper Eqn. 7).
        """
        total = None
        for fr in self._servers():
            emb = torch.nn.functional.normalize(self._embed_batched(fr, imgs255), p=2, dim=1)
            # ||a - b||^2 = 2 - 2 a.b for unit vectors.
            sim = emb @ emb.t()
            dist = (2.0 - 2.0 * sim)
            # Exclude the zero diagonal.
            n = dist.shape[0]
            off_diag = dist.sum() - torch.diagonal(dist).sum()
            d = off_diag / max(n * (n - 1), 1)
            total = d if total is None else total + d
        return total / len(self._servers())

    def _compute_uap(self, train_loader: DataLoader) -> torch.Tensor:
        """The universal perturbation for the injected servers: projected gradient ascent of the
        mean pairwise server distance (paper Eqn. 7) within the L-inf ``epsilon`` ball.
        """
        # Gather a fixed pool of training images for the pairwise objective.
        pool: List[torch.Tensor] = []
        seen = 0
        for images, _ in train_loader:
            pool.append(images.to(self.device))
            seen += images.shape[0]
            if seen >= self.uap_pairs:
                break
        pool = torch.cat(pool, dim=0)[: self.uap_pairs]
        if pool.shape[0] < 2:
            raise RuntimeError("UAP computation needs at least two training images")

        u = torch.zeros(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, device=self.device)
        for _ in range(self.uap_steps):
            u = u.detach().requires_grad_(True)
            perturbed = (pool + u).clamp(0.0, 255.0)
            objective = self._pairwise_server_distance(perturbed)
            grad = torch.autograd.grad(objective, u)[0]
            u = u.detach() + self.uap_step_size * torch.sign(grad)
            u = u.clamp(-self.epsilon, self.epsilon)
        logger.info(
            "Computed UAP: mean |u|=%.3f, max |u|=%.3f (eps=%.1f)",
            u.abs().mean().item(), u.abs().max().item(), self.epsilon,
        )
        return u.detach()

    # Filter

    def _generate_noise(self, x: torch.Tensor, enrolled: torch.Tensor) -> torch.Tensor:
        """Stage A: I2-FGSM image-specific noise from the probe model."""
        probe = self._probe()
        x_adv = i2fgsm(
            x,
            enrolled,
            lambda imgs: self._embed(probe, imgs),
            epsilon=self.epsilon,
            step_size=self.step_size,
            n_iter=self.n_iter,
        )
        return x_adv - x

    def _apply_filter(self, x: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
        """Stages B+C: transfer + enhance, clip to eps, add to the image, clamp."""
        transferred = self.unet(x_noise)
        enhanced = self.enhance(x.shape[0])
        s = (transferred + enhanced).clamp(-self.epsilon, self.epsilon)
        return (x + s).clamp(0.0, 255.0)

    # Lifecycle

    def train(self, data: TrainData) -> None:
        """Compute the universal perturbation, then train ``T`` and ``E`` to maximize the servers'
        distance between each protected image and an enrolled image (paper Eqn. 9).

        Args:
            data.corpus: A labelled face corpus, read as same-identity
                ``(image, enrolled)`` pairs by
                :class:`~afrlib.afr.apf.data.APFPairDataset`.
        """
        self._check_inputs("train", data)
        self._probe()  # validates the probe/server split before any work
        train_loader = self._build_loader(data.corpus)
        self._begin_training()
        self._train_dataset_config = dataset_config(train_loader)
        # Stage C prior: recompute and freeze the UAP, then rebuild E around it.
        uap = self._compute_uap(train_loader)
        self.enhance = Enhancement(uap).to(self.device)

        self.unet.train()
        self.enhance.train()
        params = list(self.unet.parameters()) + list(self.enhance.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"APF epoch {epoch + 1}/{self.epochs}")
            running = 0.0
            for images, enrolled in pbar:
                x = images.to(self.device)
                xe = enrolled.to(self.device)

                x_noise = self._generate_noise(x, xe)
                x_adv = self._apply_filter(x, x_noise)

                distance = self._server_distance(x_adv, xe).mean()
                loss = -distance
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += distance.item()
                pbar.set_postfix(server_dist=f"{distance.item():.4f}")
            logger.info(
                "epoch %d/%d: mean server distance %.4f",
                epoch + 1, self.epochs, running / max(len(train_loader), 1),
            )

        self.unet.eval()
        self.enhance.eval()

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """The same-identity pair loader over the run's corpus."""
        from .data import APFPairDataset

        loader = DataLoader(
            APFPairDataset.from_corpus(corpus),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError(
                f"Corpus {corpus.name!r} yields fewer than {self.batch_size} "
                "same-identity pairs after filtering"
            )
        return loader

    def fit(self, data: FitData) -> None:
        """Register the protectee's enrolled album, which :meth:`protect` pushes images away from.

        Args:
            data.protectee_album: Aligned RGB crops ``[B, 3, 112, 112]`` in ``[0, 255]``; any
                number, cycled through by :meth:`protect`.
        """
        self._check_inputs("fit", data)
        self._begin_fitting()
        self.enrolled_album = self._validate_images(
            data.protectee_album, name="protectee_album"
        ).detach().cpu()
        self._mark_ready()

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Filter each aligned crop away from one album image and paste the change back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        sources = self._validate_images(torch.stack(crops), name="eval_imgs")
        # Cycle the album to the batch length, continuing from the previous call,
        # so that each image is pushed away from one enrolled image.
        album_n = int(self.enrolled_album.shape[0])
        index = (
            torch.arange(sources.shape[0]) + int(self._album_offset)
        ) % album_n
        enrolled = self.enrolled_album[index]
        self._album_offset = (int(self._album_offset) + int(sources.shape[0])) % album_n

        outputs: List[torch.Tensor] = []
        for start in range(0, sources.shape[0], self.inference_batch_size):
            x = sources[start : start + self.inference_batch_size].to(self.device)
            xe = enrolled[start : start + self.inference_batch_size].to(self.device)
            # Stage A needs gradients (I2-FGSM); stages B+C are inference-only.
            x_noise = self._generate_noise(x, xe)
            with torch.no_grad():
                outputs.append(self._apply_filter(x, x_noise).cpu())
        filtered = torch.cat(outputs, dim=0).to(torch.float32)
        return paste_deltas(photos, filtered - sources.cpu(), plans)

    def _validate_images(self, imgs: torch.Tensor, *, name: str) -> torch.Tensor:
        if not torch.is_floating_point(imgs):
            raise TypeError(f"{name} must be floating point, got {imgs.dtype}")
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"{name} must be [B, 3, H, W], got {tuple(imgs.shape)}")
        if imgs.shape[-1] != _IMAGE_SIZE or imgs.shape[-2] != _IMAGE_SIZE:
            raise ValueError(
                f"{name} must be {_IMAGE_SIZE}x{_IMAGE_SIZE}, got {tuple(imgs.shape[-2:])}"
            )
        if not torch.isfinite(imgs).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return imgs

    # Persistence

    def save_run(self, save_path: str | os.PathLike[str]) -> Path:
        """Write the trained ``T``/``E``, the frozen UAP, and any fitted album."""
        self._require_savable()
        root = Path(save_path)
        root.mkdir(parents=True, exist_ok=True)
        weights = ["unet.pth", "enhance.pth", "uap.pt"]
        torch.save(self._cpu_state_dict(self.unet), root / "unet.pth")
        torch.save(self._cpu_state_dict(self.enhance), root / "enhance.pth")
        torch.save(self.enhance.uap.detach().cpu(), root / "uap.pt")
        if self.enrolled_album is not None:
            torch.save(self.enrolled_album, root / "enrolled_album.pt")
            weights.append("enrolled_album.pt")
        torch.save({"album_offset": int(self._album_offset)}, root / "album_offset.pt")
        weights.append("album_offset.pt")
        root = super().save_run(root, weights=weights)
        logger.info("Saved APF run to %s", root)
        return root

    def load_run(self, load_path: str | os.PathLike[str]) -> Path:
        root = super().load_run(load_path)
        uap = torch.load(root / "uap.pt", map_location=self.device, weights_only=True)
        self.enhance = Enhancement(uap).to(self.device)
        self.unet.load_state_dict(
            torch.load(root / "unet.pth", map_location=self.device, weights_only=True)
        )
        self.enhance.load_state_dict(
            torch.load(root / "enhance.pth", map_location=self.device, weights_only=True)
        )
        self.unet.eval().requires_grad_(False)
        self.enhance.eval().requires_grad_(False)
        self._train_dataset_config = read_run_config(root).get("training_dataset")
        album_path = root / "enrolled_album.pt"
        if album_path.is_file():
            self.enrolled_album = torch.load(
                album_path, map_location="cpu", weights_only=True
            )
            self._fitted = True
        offset_path = root / "album_offset.pt"
        if offset_path.is_file():
            payload = torch.load(offset_path, map_location="cpu", weights_only=True)
            self._album_offset = int(payload.get("album_offset") or 0)
        self._trained = True
        # Only a fitted run is ready: protect() pushes images away from the
        # protectee's enrolled album, so a train-only run still needs fit().
        if self._fitted:
            self._mark_ready()
        logger.info("Loaded APF run from %s", root)
        return root

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}
