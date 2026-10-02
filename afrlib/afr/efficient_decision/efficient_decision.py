"""EfficientDecision: hard-label verification dodging with a (1+1)-CMA-ES search.

Reference Paper: Efficient Decision-based Black-box Adversarial Attacks on Face Recognition (CVPR 2019)
Reference Code: https://github.com/sgmath12/efficient-decision-based-black-box-adversarial-attacks-on-face-recognition (unofficial)
"""

from typing import List, Dict, Any, Sequence
from collections import deque
import math
from enum import Enum

import torch
import torch.nn.functional as F

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
from afrlib.utils.log import get_logger, tqdm

logger = get_logger(__name__)

class OracleRes(Enum):
    """Result of a :class:`VeriOracle` query."""

    NOT_SAME_ID = 0
    SAME_ID = 1
    EXCEEDED_MAX_QUERIES = 2

class VeriOracle:
    """Hard-label verification oracle: ``SAME_ID`` when the ensemble-mean cosine similarity to
    the reference reaches ``sim_threshold``, until ``max_cnt`` queries are used.
    """

    def __init__(self, fr_models: Sequence[FaceRecognizer], sim_threshold: float, max_cnt: int):
        self.fr_models = fr_models
        self.sim_threshold = sim_threshold
        self.max_cnt = max_cnt
        self.query_cnt = 0
        self.reference_embed_lst = None

    def query(self, query_img: torch.Tensor) -> OracleRes:
        if self.query_cnt >= self.max_cnt:
            return OracleRes.EXCEEDED_MAX_QUERIES
        self.query_cnt += 1
        with torch.inference_mode():
            sims = [F.cosine_similarity(q, r, dim=1) for q, r in zip(self._embed(query_img), self.reference_embed_lst)]
            mean_sim = torch.stack(sims, dim=0).mean().item()
            if mean_sim >= self.sim_threshold:
                return OracleRes.SAME_ID
            else:
                return OracleRes.NOT_SAME_ID

    def reset(self, reference_img: torch.Tensor) -> None:
        self.query_cnt = 0
        self.reference_embed_lst = self._embed(reference_img)

    def get_remaining_queries(self) -> int:
        return self.max_cnt - self.query_cnt

    def get_used_queries(self) -> int:
        return self.query_cnt

    def _embed(self, img01: torch.Tensor) -> List[torch.Tensor]:
        """Embed ``[0, 1]`` RGB images with every FR model's full pipeline, L2-normalized."""
        return [fr.forward(img01.clamp(0., 1.) * 255., need_crop=True, need_align=True, l2_normalize=True).embeddings for fr in self.fr_models]
        

class EfficientDecision(AFRMethod):
    """Searches, with hard-label queries only, for the photo closest to the original that the
    oracle no longer matches to a held-out photo of the same person.

    Args:
        max_queries: Oracle queries per photo.
        search_size: Side of the low-resolution search space, upscaled to the crop.
        sim_threshold: Cosine similarity at which the oracle answers "same identity"; the
            default is the reference implementation's SphereFace value, not calibrated for
            the injected models.
        mu_init: Initial step size of the search.
        sigma_factor: Scale of the search distribution relative to the current distance.
        success_window: Recent steps the 1/5 success rule looks at.
        mu_update_interval: Steps between step-size updates.
    """

    name = "EfficientDecision"
    eval_inputs = StageInputs(
        required=("eval_imgs", "reference_imgs"),
    )
    def __init__(self,
                 device: torch.device,
                 fr_models: Sequence[FaceRecognizer] = (),
                 *,
                 max_queries: int = 10000,
                 search_size: int = 45,
                 sim_threshold: float = 0.3,
                 cc: float = 0.01,
                 ccov: float = 0.001,
                 mu_init: float = 0.1,
                 sigma_factor: float = 0.001,
                 success_window: int = 15,
                 mu_update_interval: int = 5):
        super().__init__(device, fr_models)
        # The paper's input dimension n = 112 x 112 x 3 matches the attack crop.

        self.max_queries = max_queries
        self.search_size = search_size
        self.sim_threshold = sim_threshold
        self.oracle = VeriOracle(self.fr_models, self.sim_threshold, self.max_queries)
        self.cc = cc
        self.ccov = ccov
        self.mu_init = mu_init
        self.sigma_factor = sigma_factor          # sigma = sigma_factor * D(x*, x)
        self.success_window = success_window      # trials averaged for the 1/5 rule
        self.mu_update_interval = mu_update_interval

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.NoAlign)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "max_queries": self.max_queries,
            "search_size": self.search_size,
            "sim_threshold": self.sim_threshold,
            "cc": self.cc,
            "ccov": self.ccov,
            "mu_init": self.mu_init,
            "sigma_factor": self.sigma_factor,
            "success_window": self.success_window,
            "mu_update_interval": self.mu_update_interval,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Search each aligned crop for a perturbation the oracle rejects and paste it back; a photo
        the oracle already rejects is returned unchanged.

        Args:
            data.reference_imgs: One held-out photo of the same person per protected photo.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device)
        originals = clean / 255.  # work internally in [0, 1]
        references = data.reference_imgs.to(self.device) / 255.
        self.query_count = 0

        protected = []
        for i in tqdm(range(originals.shape[0]), desc="EfficientDecision/verif-dodge"):
            protected.append(self._protect_single(originals[i:i + 1], references[i:i + 1]))
        adv = torch.clamp(torch.cat(protected, dim=0), 0., 1.) * 255.
        return paste_deltas(photos, adv - clean, plans)

    def _protect_single(self, original: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Protect one query against its held-out same-identity reference."""
        self.oracle.reset(reference)
        if self.oracle.query(original) == OracleRes.NOT_SAME_ID:
            logger.info("Original image already rejected; returning unchanged.")
            return original
        x_adv = torch.rand_like(original, device=self.device)
        if self.oracle.query(x_adv) == OracleRes.SAME_ID:
            logger.warning("Random initialization failed to dodge; check the reference image, sim_threshold, and FR ensemble; returning unchanged.")
            return original
        dist = self._get_dist(x_adv, original)
        if not math.isfinite(dist):
            logger.warning("Random initialization loss is not finite; returning unchanged.")
            return original
        _, _, H, W = original.shape
        m = self.search_size * self.search_size * 3
        k = max(1, m // 20)
        c_diag = torch.ones(m, device=self.device)
        pc = torch.zeros(m, device=self.device)
        mu = self.mu_init
        recent_successes = deque(maxlen=self.success_window)

        it = 0
        pbar = tqdm(range(self.oracle.get_remaining_queries()), desc="EfficientDecision/Querying", leave=False)
        while self.oracle.get_remaining_queries() > 0:
            dist = self._get_dist(x_adv, original)
            sigma = self.sigma_factor * max(dist, 1e-12)

            # Sample coloured noise z ~ N(0, sigma^2 C) in the reduced space.
            z = sigma * torch.sqrt(c_diag) * torch.randn(m, device=self.device)

            # Stochastic coordinate selection: keep k coordinates with probability
            # proportional to their variance c_ii, zero out the rest.
            selected = torch.multinomial(c_diag, k, replacement=False)
            mask = torch.zeros(m, device=self.device)
            mask[selected] = 1.
            z = z * mask

            # Upscale to image space and bias the step toward the original image.
            z_up = F.interpolate(z.reshape(1, 3, self.search_size, self.search_size),
                                    size=(H, W), mode="bilinear", align_corners=False)
            z_up = z_up + mu * (original - x_adv)

            candidate = torch.clamp(x_adv + z_up, 0., 1.)
            # One oracle query; L(x*) = ||x*-x|| if adversarial, else +inf.
            new_dist = self._get_dist(candidate, original) if self.oracle.query(candidate) == OracleRes.NOT_SAME_ID else float("inf")
            improved = new_dist < dist
            if improved:
                x_adv, dist = candidate, new_dist
                # Adapt the search distribution from this successful move (Eqs. 3-4).
                pc = (1 - self.cc) * pc + math.sqrt(self.cc * (2 - self.cc)) * z / sigma
                c_diag = (1 - self.ccov) * c_diag + self.ccov * pc ** 2
            recent_successes.append(1. if improved else 0.)
            # 1/5 success rule: grow mu above a 1/5 success rate, shrink it below.
            if it >= self.success_window and it % self.mu_update_interval == 0:
                mu = mu * math.exp(sum(recent_successes) / self.success_window - 0.2)
            pbar.set_postfix(dist=f"{dist:.4f}", mu=f"{mu:.3f}", q=self.oracle.get_used_queries())
            pbar.update(1)
            it += 1
        pbar.close()
        return x_adv.clone().detach()
            
    @staticmethod
    def _get_dist(adv: torch.Tensor, orig: torch.Tensor) -> float:
        """Compute the loss for a candidate: L2 distance to the original."""
        return torch.norm((adv - orig).flatten(), p=2).item()
    
        
