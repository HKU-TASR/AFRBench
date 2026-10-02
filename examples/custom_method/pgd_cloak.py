"""A minimal AFR method: an L-inf PGD cloak on the ArcFace-aligned face crop.

Use it as a template for your own method, e.g.
``python -m afrbench.benchmark --method pgd_cloak:PGDCloak ...`` with this folder on
``PYTHONPATH``.
"""
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from afrlib.afr.base import AFRMethod
from afrlib.contracts import AlignType, ChannelType, CropType, EvalData, ImageSpec, StageInputs
from afrlib.utils.face_crop import paste_deltas, prepare_inputs


class PGDCloak(AFRMethod):
    """Move each face's embedding away from its clean embedding within an L-inf budget."""

    name = "pgd_cloak"
    eval_inputs = StageInputs(required=("eval_imgs",))

    def __init__(self, device, fr_models=(), epsilon: float = 8.0, steps: int = 20,
                 step_size: float = 1.0, *, assets_path=None):
        super().__init__(device, fr_models, assets_path=assets_path)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112, channel=ChannelType.RGB, crop_type=CropType.FaceBox,
                         align_type=AlignType.ArcFace5)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon, "steps": self.steps, "step_size": self.step_size})
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        self._check_inputs("protect", data)
        frs = self._require_fr()
        crops, plans = prepare_inputs(list(data.eval_imgs), data.eval_geometry, self.attack_spec())
        clean = torch.stack(crops).to(self.device)  # [N, 3, 112, 112] in [0, 255]
        with torch.no_grad():
            clean_embeddings = [fr.embed_crops(clean / 255, l2_normalize=True) for fr in frs]
        # Random start: at delta = 0 the similarity is at its maximum and has no gradient.
        delta = torch.empty_like(clean).uniform_(-self.epsilon, self.epsilon).requires_grad_(True)
        for _ in range(self.steps):
            adv = (clean + delta).clamp(0, 255) / 255
            similarity = sum(
                F.cosine_similarity(fr.embed_crops(adv, l2_normalize=True), emb).mean()
                for fr, emb in zip(frs, clean_embeddings)
            )
            similarity.backward()
            with torch.no_grad():
                delta -= self.step_size * delta.grad.sign()
                delta.clamp_(-self.epsilon, self.epsilon)
            delta.grad = None
        adv = (clean + delta.detach()).clamp(0, 255)
        return paste_deltas(list(data.eval_imgs), list(adv - clean), plans)
