"""DPA: diverse surrogate snapshots (DPO) and hard-model impersonation (HMA).

Reference Paper: Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation (CVPR 2025)
Reference Code: https://github.com/fengfanzhou/DPA
"""
from __future__ import annotations

import copy
import math
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
    ImageSpec,
    StageInputs,
    TrainData,
)
from afrlib.afr.base import AFRMethod, read_run_config
from afrlib.afr.datasets import dataset_config
from afrlib.fr import FaceRecognizer
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.utils.log import get_logger, tqdm

from .attack import ADV_FEAT_STEP, EPSILON, N_ITER, STEP_SIZE, hma_attack
from .models import (
    ArcMarginProduct,
    SurrogateForward,
    reinit_backbone,
    unwrap_conv_layers,
    wrap_conv_layers,
)

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

logger = get_logger(__name__)

_IMAGE_SIZE = 112
# Init branches in DPO: pretrained (from the injected recognizer) and random.
_INIT_TAGS = ("pretrained", "randomly_initialized")


def _backbone_state_dict(backbone: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """State dict of ``backbone`` without its ``HardModelConv`` wrappers, so it loads into a
    plain copy of the recognizer's backbone.
    """
    plain = copy.deepcopy(backbone)
    unwrap_conv_layers(plain)
    return {k: v.detach().cpu() for k, v in plain.state_dict().items()}


class DPA(AFRMethod):
    """Trains diverse snapshots of each surrogate's backbone, then attacks the whole snapshot
    ensemble with a targeted I-FGSM whose convolutions add a beneficial perturbation.

    Args:
        surrogate_indices: Indices in ``fr_models`` to snapshot; all of them if ``None``.
        dpo_epochs: Snapshot-training epochs per branch.
        batch_size: Snapshot-training batch size.
        lr: SGD learning rate.
        momentum: SGD momentum.
        weight_decay: SGD weight decay.
        margin_s: Scale of the ArcFace training head.
        margin_m: Angular margin of the ArcFace training head.
        data_parallel: Wrap the trained copies in ``DataParallel``.
        n_iter: Attack iterations.
        epsilon: L-inf budget, in 8-bit levels.
        step_size: Attack step size, in 8-bit levels.
        adv_feat_step: Size of the perturbation each wrapped convolution adds.
    """

    name = "DPA"
    training_corpus = "ms1m"
    train_inputs = StageInputs(required=("corpus",))
    eval_inputs = StageInputs(required=("eval_imgs", "target_img"))

    def __init__(
        self,
        device: torch.device | str,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        assets_path: Optional[str | Path] = None,
        surrogate_indices: Optional[Sequence[int]] = None,
        # DPO.
        dpo_epochs: int = 35,
        batch_size: int = 128,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        margin_s: float = 32.0,
        margin_m: float = 0.5,
        data_parallel: bool = False,
        # HMA.
        n_iter: int = N_ITER,
        epsilon: float = EPSILON,
        step_size: float = STEP_SIZE,
        adv_feat_step: float = ADV_FEAT_STEP,
    ) -> None:
        super().__init__(device, fr_models, assets_path=assets_path)

        # Resolved against the injected ensemble at the point of use, so an
        # inference-only reload does not need the surrogates up front.
        self._surrogate_indices_arg = (
            None if surrogate_indices is None else [int(i) for i in surrogate_indices]
        )
        self.surrogate_indices: List[int] = list(self._surrogate_indices_arg or [])

        if dpo_epochs < 1 or batch_size < 1:
            raise ValueError("dpo_epochs and batch_size must be positive")
        self.dpo_epochs = int(dpo_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.margin_s = float(margin_s)
        self.margin_m = float(margin_m)
        self.data_parallel = bool(data_parallel)
        self.n_iter = int(n_iter)
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.adv_feat_step = float(adv_feat_step)

        # The HMA surrogate set; built by train() or load_run().
        self._surrogates: List[SurrogateForward] = []
        self._wrappers = []
        self._train_dataset_config: Optional[Dict[str, Any]] = None
        self._maybe_load_run()

    # Config

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(
            size=_IMAGE_SIZE,
            channel=ChannelType.RGB,
            crop_type=CropType.FaceBox,
            align_type=AlignType.ArcFace5,
        )

    @property
    def min_surrogates(self) -> int:
        """Enough surrogates for every ``surrogate_indices`` entry to resolve."""
        return max(self.surrogate_indices, default=0) + 1

    def _resolve_surrogate_indices(self) -> List[int]:
        """Validate the injected ensemble and pin which of it DPO trains on."""
        self._require_fr()
        if not self.surrogate_indices:
            self.surrogate_indices = list(range(len(self.fr_models)))
        for i in self.surrogate_indices:
            if not 0 <= i < len(self.fr_models):
                raise ValueError(
                    f"surrogate_index {i} out of range for {len(self.fr_models)} models"
                )
        return self.surrogate_indices

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "surrogate_indices": self.surrogate_indices,
                "dpo_epochs": self.dpo_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "margin_s": self.margin_s,
                "margin_m": self.margin_m,
                "data_parallel": self.data_parallel,
                "n_iter": self.n_iter,
                "epsilon": self.epsilon,
                "step_size": self.step_size,
                "adv_feat_step": self.adv_feat_step,
                # Provenance, not constructor arguments.
                "training_dataset": self._train_dataset_config,
            }
        )
        return config

    # DPO

    def _build_loader(self, corpus: "TrainingCorpus") -> DataLoader:
        """The DPO classification loader over the run's corpus."""
        from .data import FaceClassDataset

        loader = DataLoader(
            FaceClassDataset.from_corpus(corpus, size=_IMAGE_SIZE),
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

    def _snapshot_epochs(self) -> List[int]:
        """Epochs after which the backbone is snapshotted: 0, every k, and the last."""
        interval = int(math.sqrt(self.dpo_epochs))
        return sorted(
            {
                e
                for e in range(self.dpo_epochs)
                if e == 0 or e == self.dpo_epochs - 1 or (interval and e % interval == 0)
            }
        )

    def train(self, data: TrainData) -> None:
        """Train a pretrained and a randomly initialized copy of each selected surrogate, keeping
        periodic snapshots as the attack ensemble.

        Args:
            data.corpus: A labelled face corpus, read as ``(img, label)``
                batches of RGB float ``[B, 3, 112, 112]`` in ``[0, 255]`` by
                :class:`~afrlib.afr.dpa.data.FaceClassDataset`.
        """
        self._check_inputs("train", data)
        train_loader = self._build_loader(data.corpus)
        num_classes = getattr(train_loader.dataset, "num_classes", None)
        if not num_classes:
            raise ValueError("DPO requires the dataset to expose num_classes")
        self._resolve_surrogate_indices()
        self._begin_training()
        self._train_dataset_config = dataset_config(train_loader)

        snapshot_epochs = set(self._snapshot_epochs())
        self._surrogates = []
        self._wrappers = []

        for index in self.surrogate_indices:
            recognizer = self.fr_models[index]
            for tag in _INIT_TAGS:
                backbone = copy.deepcopy(recognizer.backbone)
                if tag == "randomly_initialized":
                    reinit_backbone(backbone)
                # The copy inherits the injected recognizer's frozen parameters;
                # DPO trains the whole backbone, as the reference dpo.py does.
                backbone.requires_grad_(True)
                model = SurrogateForward(
                    backbone,
                    recognizer.input_size,
                    recognizer.mean,
                    recognizer.std,
                    recognizer.channel,
                ).to(self.device)
                head = ArcMarginProduct(
                    in_feature=int(recognizer.embedding_dim),
                    out_feature=num_classes,
                    s=self.margin_s,
                    m=self.margin_m,
                ).to(self.device)

                if self.data_parallel:
                    model = torch.nn.DataParallel(model)
                    head = torch.nn.DataParallel(head)
                model.train()
                head.train()
                trainable = list(model.parameters()) + list(head.parameters())
                optimizer = torch.optim.SGD(
                    trainable,
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                    nesterov=True,
                )
                criterion = torch.nn.CrossEntropyLoss()

                # Snapshots are taken after the epoch's training, as in dpo.py.
                for epoch in range(self.dpo_epochs):
                    pbar = tqdm(
                        train_loader,
                        desc=f"DPO[{recognizer.name}/{tag}] epoch {epoch + 1}/{self.dpo_epochs}",
                    )
                    running, steps = 0.0, 0
                    for imgs, labels in pbar:
                        imgs = imgs.to(self.device)
                        labels = labels.to(self.device)
                        optimizer.zero_grad()
                        embeddings = model(imgs)
                        if not torch.isfinite(embeddings).all():
                            raise FloatingPointError(
                                f"DPO produced non-finite embeddings for "
                                f"{recognizer.name}/{tag} at epoch {epoch + 1}"
                            )
                        logits = head(embeddings, labels)
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError(
                                f"DPO produced non-finite logits for "
                                f"{recognizer.name}/{tag} at epoch {epoch + 1}"
                            )
                        loss = criterion(logits, labels)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"DPO produced non-finite loss for "
                                f"{recognizer.name}/{tag} at epoch {epoch + 1}"
                            )
                        loss.backward()
                        if not all(
                            parameter.grad is None
                            or torch.isfinite(parameter.grad).all()
                            for parameter in trainable
                        ):
                            raise FloatingPointError(
                                f"DPO produced non-finite gradients for "
                                f"{recognizer.name}/{tag} at epoch {epoch + 1}"
                            )
                        optimizer.step()
                        running += loss.item()
                        steps += 1
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                    logger.info(
                        "DPO %s/%s epoch %d/%d mean loss %.4f",
                        recognizer.name, tag, epoch + 1, self.dpo_epochs, running / max(steps, 1),
                    )
                    if epoch in snapshot_epochs:
                        self._append_surrogate(recognizer, model)

        # The HMA ensemble also includes each selected surrogate's pretrained
        # backbone, as get_surrogate_models in the reference hma.py does.
        self._add_pristine_surrogates()
        self._mark_ready()
        logger.info(
            "DPO complete: %d surrogates across %d base model(s)",
            len(self._surrogates), len(self.surrogate_indices),
        )

    def _append_surrogate(self, recognizer: FaceRecognizer, model: torch.nn.Module) -> None:
        """Snapshot the current DPO backbone as a frozen hard-model surrogate."""
        source = model.module if isinstance(model, torch.nn.DataParallel) else model
        self._add_frozen_surrogate(recognizer, source.backbone)

    def _add_pristine_surrogates(self) -> None:
        """Append a frozen copy of each selected surrogate's pretrained backbone."""
        for index in self.surrogate_indices:
            recognizer = self.fr_models[index]
            self._add_frozen_surrogate(recognizer, recognizer.backbone)

    def _add_frozen_surrogate(self, recognizer: FaceRecognizer, backbone: torch.nn.Module) -> None:
        """Deep-copy ``backbone``, freeze it, wrap its convs, and register it."""
        copied = copy.deepcopy(backbone).eval().requires_grad_(False)
        snapshot = SurrogateForward(
            copied, recognizer.input_size, recognizer.mean, recognizer.std, recognizer.channel
        ).cpu()
        snapshot.eval()
        self._wrappers.extend(wrap_conv_layers(copied, self.adv_feat_step))
        self._surrogates.append(snapshot)

    # HMA

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Perturb each aligned crop toward the target against the snapshot ensemble and paste the
        change back.

        Args:
            data.target_img: The impersonation target, shared by all photos.
        """
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        sources = self._validate_images(torch.stack(crops), name="eval_imgs").to(self.device)
        adv = hma_attack(
            self._surrogates,
            self._wrappers,
            sources,
            data.target_img.to(self.device),
            epsilon=self.epsilon,
            step_size=self.step_size,
            n_iter=self.n_iter,
        ).to(torch.float32)
        return paste_deltas(photos, adv - sources, plans)

    def _validate_images(self, imgs: torch.Tensor, *, name: str) -> torch.Tensor:
        if not torch.is_floating_point(imgs):
            raise TypeError(f"{name} must be floating point, got {imgs.dtype}")
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() != 4 or imgs.shape[1] != 3:
            raise ValueError(f"{name} must be [B, 3, H, W], got {tuple(imgs.shape)}")
        if not torch.isfinite(imgs).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return imgs

    # Persistence

    def save_run(self, save_path: str | os.PathLike[str]) -> Path:
        """Write the DPO snapshots and the config under ``save_path``."""
        self._require_savable()
        root = Path(save_path)
        root.mkdir(parents=True, exist_ok=True)

        # Only the snapshots are saved, grouped by model; the pretrained backbones come from the
        # injected recognizers.
        n_snaps = len(self._snapshot_epochs()) * len(_INIT_TAGS)
        saved_surrogates = []
        for pos, index in enumerate(self.surrogate_indices):
            recognizer = self.fr_models[index]
            entry = {
                "name": recognizer.name,
                "fr_index": index,
                "input_size": list(recognizer.input_size),
                "embedding_dim": int(recognizer.embedding_dim),
                "snapshots": [],
            }
            for snap_idx, model in enumerate(
                self._surrogates[pos * n_snaps : (pos + 1) * n_snaps]
            ):
                fname = f"{pos}_{snap_idx}.pt"
                torch.save(_backbone_state_dict(model.backbone), root / fname)
                entry["snapshots"].append(fname)
            saved_surrogates.append(entry)

        root = super().save_run(root, surrogates=saved_surrogates)
        logger.info("Saved DPA run to %s", root)
        return root

    def load_run(self, load_path: str | os.PathLike[str]) -> Path:
        """Rebuild the attack ensemble written by :meth:`save_run`; the same ``fr_models`` must be
        injected, since each snapshot loads into a copy of their backbones.
        """
        root = super().load_run(load_path)
        config = read_run_config(root)

        # Restore the training-time bookkeeping so snapshot layout (and thus any
        # later save) matches how these weights were produced.
        for key in ("dpo_epochs", "surrogate_indices"):
            if key in config:
                setattr(self, key, config[key])
        self._require_fr()

        self._surrogates = []
        self._wrappers = []
        # Snapshots first (grouped by model), then the pretrained backbones:
        # the order train() produces.
        for entry in config["surrogates"]:
            recognizer = self.fr_models[int(entry["fr_index"])]
            for fname in entry["snapshots"]:
                backbone = copy.deepcopy(recognizer.backbone)
                state = torch.load(root / fname, map_location="cpu", weights_only=True)
                backbone.load_state_dict(state)
                self._add_frozen_surrogate(recognizer, backbone)
        self._add_pristine_surrogates()

        self._trained = True
        self._mark_ready()
        logger.info(
            "Loaded DPA run from %s (%d surrogates)", root, len(self._surrogates)
        )
        return root
