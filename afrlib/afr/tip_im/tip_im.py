"""TIP-IM: targeted identity-protection iterative method.

Reference Paper: Towards Face Encryption by Generating Adversarial Identity Masks (ICCV 2021)
Reference Code: https://github.com/ShawnXYang/TIP-IM (Apache-2.0; see LICENSE)
"""
from typing import List, Dict, Any, Optional, Sequence
import random

import numpy as np
import torch

from afrlib.utils.log import get_logger, tqdm
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)
from afrlib.afr.base import AFRMethod
from afrlib.fr import FaceRecognizer
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from .mmd import mmd_loss
from .input_diversity import input_diversity

logger = get_logger(__name__)


class TIP_IM(AFRMethod):
    """Pushes each face toward the best of several candidate identities and away from itself, with
    an MMD term keeping the protected batch close to the clean one.

    Args:
        epochs: Momentum sign-step iterations.
        epsilon: L-inf budget, in 8-bit levels.
        learning_rate: Step size, in 8-bit levels.
        gamma: Weight of the MMD naturalness term.
        num_targets: Candidate identities taken from ``target_pool``; the best is chosen per
            image and iteration.
        batch_size: Photos optimized together.
        mmd_augmentations: Transformed views per photo for the MMD term.
        seed: Seed for the target draw and the augmentations; ``None`` leaves them unseeded.
    """

    name = "TIP-IM"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_pool"),
    )
    def __init__(self,
                 device: torch.device,
                 fr_models: Sequence[FaceRecognizer] = (),
                 *,
                 epochs: int = 50,
                 epsilon: float = 12.,
                 learning_rate: float = 1.5,
                 gamma: float = 1.,
                 num_targets: int = 10,
                 batch_size: int = 8,
                 mmd_augmentations: int = 8,
                 seed: Optional[int] = 0,
                 ):
        super().__init__(device, fr_models)

        self.epochs = epochs
        self.epsilon = epsilon          # L-inf budget on the [0, 255] pixel scale
        self.learning_rate = learning_rate
        self.gamma = gamma              # weight of the MMD naturalness loss
        self.num_targets = num_targets  # candidate targets used from the provided set
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        if mmd_augmentations < 2:
            raise ValueError(
                f"mmd_augmentations must be at least 2, got {mmd_augmentations}"
            )
        self.mmd_augmentations = mmd_augmentations
        self.seed = seed
        self._python_rng = random.Random(seed)
        self._numpy_rng = np.random.RandomState(seed)


    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.ArcFace5)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "epochs": self.epochs,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "num_targets": self.num_targets,
            "batch_size": self.batch_size,
            "mmd_augmentations": self.mmd_augmentations,
            "seed": self.seed,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Perturb each aligned crop in jointly optimized batches and paste the change back.

        Args:
            data.target_pool: Candidate identities other than the protectee,
                ``(K, 3, 112, 112)``; the first ``num_targets`` are used.

        Returns:
            The protected originals, one per input image, at the input frame.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        if not photos:
            return []
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device)

        targets = data.target_pool[:self.num_targets].to(self.device)
        if targets.shape[0] == 0:
            raise ValueError("TIP-IM requires at least one target image")
        with torch.no_grad():
            target_feas = self._embed(targets)  # per model, (K, D)

        protected = []
        batch_slices = self._balanced_batch_slices(clean.shape[0], self.batch_size)
        pbar = tqdm(batch_slices, desc="TIP-IM")
        for start, end in pbar:
            protected.append(self._protect_batch(clean[start:end], target_feas))
        return paste_deltas(photos, torch.cat(protected, dim=0) - clean, plans)

    def _protect_batch(self, imgs: torch.Tensor, target_feas: List[torch.Tensor]) -> torch.Tensor:
        """Jointly optimize one source batch against the candidate target set."""
        with torch.no_grad():
            init_feas = self._embed(imgs)

        inputs = imgs.clone()
        # The L-inf box around the original also enforces the valid pixel range.
        min_img = torch.clamp(inputs - self.epsilon, min=0)
        max_img = torch.clamp(inputs + self.epsilon, max=255)
        adv_images = inputs.detach().clone().requires_grad_(True)
        sum_grad = torch.zeros_like(inputs)  # MIM momentum accumulator (decay factor 1)
        num_targets = target_feas[0].shape[0]

        for _ in range(self.epochs):
            std_proj = self._python_rng.uniform(0.01, 0.1)
            std_rotate = self._python_rng.uniform(0.01, 0.1)
            # Input diversity for generalization; one shared view per iteration.
            images = input_diversity(
                adv_images,
                std_proj,
                std_rotate,
                rng=self._numpy_rng,
            )
            adv_feas = self._embed(images)

            loss_mmd = self._mmd_naturalness(
                adv_images, inputs, std_proj, std_rotate
            ) if self.gamma != 0 else adv_images.new_zeros(())

            # Build one update candidate per target.
            tmp_advs, tmp_grads = [], []
            for idx in range(num_targets):
                loss_iden = self._identity_loss(
                    adv_feas,
                    [features[idx] for features in target_feas],
                    init_feas,
                )
                loss = loss_iden + self.gamma * loss_mmd

                grad = torch.autograd.grad(
                    loss,
                    adv_images,
                    retain_graph=idx + 1 < num_targets,
                )[0]
                if not torch.isfinite(grad).all():
                    raise RuntimeError("TIP-IM produced a non-finite input gradient")
                eps = torch.finfo(grad.dtype).eps
                grad_scale = grad.abs().mean(dim=[1, 2, 3], keepdim=True).clamp_min(eps)
                grad = grad / grad_scale
                tmp_sum_grad = sum_grad + grad

                tmp_adv_images = self._linf_step(
                    adv_images.detach(),
                    tmp_sum_grad,
                    self.learning_rate,
                    min_img,
                    max_img,
                )
                tmp_advs.append(tmp_adv_images)
                tmp_grads.append(tmp_sum_grad)

            # Greedy insertion: keep the candidate with the largest gain.
            best_idx = self._submodular_select(tmp_advs, init_feas, target_feas)
            batch_idx = torch.arange(adv_images.shape[0], device=self.device)
            sum_grad = torch.stack(tmp_grads)[best_idx, batch_idx]
            adv_images = torch.stack(tmp_advs)[best_idx, batch_idx].detach().requires_grad_(True)

        return adv_images.detach()

    def _submodular_select(self, tmp_advs: List[torch.Tensor], init_feas: List[torch.Tensor],
                           target_feas: List[torch.Tensor]) -> torch.Tensor:
        """Best target-candidate index for each image in the batch."""
        with torch.no_grad():
            num_targets = len(tmp_advs)
            batch_size = tmp_advs[0].shape[0]
            candidates = torch.stack(tmp_advs)
            cand_feas = self._embed(candidates.flatten(0, 1))
            gains = torch.zeros(num_targets, batch_size, device=self.device)
            for m in range(len(self.fr_models)):
                features = cand_feas[m].reshape(num_targets, batch_size, -1)
                d1 = torch.sum(
                    (features - init_feas[m].unsqueeze(0)) ** 2,
                    dim=2,
                )
                target_dist = torch.sum(
                    (features.unsqueeze(2) - target_feas[m][None, None]) ** 2,
                    dim=3,
                )
                d2 = torch.max(torch.exp(d1.unsqueeze(2) - target_dist), dim=2).values
                gains += torch.log(1.0 + d2) / len(self.fr_models)
        return torch.argmax(gains, dim=0)

    @staticmethod
    def _identity_loss(
        adv_feas: List[torch.Tensor],
        target_feas: List[torch.Tensor],
        init_feas: List[torch.Tensor],
    ) -> torch.Tensor:
        """Paper Eq. 2, averaged over images and surrogate models."""
        losses = []
        for adv, target, init in zip(adv_feas, target_feas, init_feas):
            target_dist = (adv - target).square().sum(dim=1)
            source_dist = (adv - init).square().sum(dim=1)
            losses.append((target_dist - source_dist).mean())
        return torch.stack(losses).mean()

    @staticmethod
    def _linf_step(
        images: torch.Tensor,
        momentum: torch.Tensor,
        learning_rate: float,
        min_img: torch.Tensor,
        max_img: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Eq. 5 for an L-inf constraint."""
        updated = images - learning_rate * momentum.sign()
        return torch.minimum(torch.maximum(updated, min_img), max_img)

    @staticmethod
    def _balanced_batch_slices(
        num_items: int,
        max_batch_size: int,
    ) -> List[tuple[int, int]]:
        """Partition inputs without a singleton tail when at least two exist."""
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {max_batch_size}"
            )
        if num_items <= 0:
            return []
        num_batches = (num_items + max_batch_size - 1) // max_batch_size
        base_size, larger_batches = divmod(num_items, num_batches)
        slices = []
        start = 0
        for batch_idx in range(num_batches):
            size = base_size + (batch_idx < larger_batches)
            slices.append((start, start + size))
            start += size
        return slices

    def _mmd_naturalness(
        self,
        adv_images: torch.Tensor,
        inputs: torch.Tensor,
        std_proj: float,
        std_rotate: float,
    ) -> torch.Tensor:
        """Paper Eq. 3, with paired transformed replicas for a singleton."""
        if adv_images.shape[0] > 1:
            return mmd_loss(adv_images.flatten(1), inputs.flatten(1))

        adv_repeated = adv_images.repeat(self.mmd_augmentations, 1, 1, 1)
        inputs_repeated = inputs.repeat(self.mmd_augmentations, 1, 1, 1)
        rng_state = self._numpy_rng.get_state()
        adv_views = input_diversity(
            adv_repeated,
            std_proj,
            std_rotate,
            rng=self._numpy_rng,
        )
        self._numpy_rng.set_state(rng_state)
        clean_views = input_diversity(
            inputs_repeated,
            std_proj,
            std_rotate,
            rng=self._numpy_rng,
        )
        return mmd_loss(adv_views.flatten(1), clean_views.flatten(1))

    def _embed(self, imgs255: torch.Tensor) -> List[torch.Tensor]:
        """Embed ``[0, 255]`` RGB faces with every FR model, L2-normalized as upstream."""
        return [fr.forward(imgs255, need_crop=False, tta=(), l2_normalize=True).embeddings
                for fr in self.fr_models]
