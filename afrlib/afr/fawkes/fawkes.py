"""Fawkes: image cloaking against unauthorized face recognition.

Reference Paper: Fawkes: Protecting Privacy against Unauthorized Deep Learning Models (USENIX Security 2020)
Reference Code: https://github.com/Shawn-Shan/fawkes
"""
from typing import List, Dict, Any, Sequence

import numpy as np
import torch
from pytorch_msssim import ssim as calc_ssim

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
)
from afrlib.afr.base import AFRMethod
from afrlib.fr import FaceRecognizer
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.utils.log import tqdm, get_logger

logger = get_logger(__name__)


class Fawkes(AFRMethod):
    """Untargeted feature-space cloaking under a DSSIM budget.

    Args:
        dssim_thresh: DSSIM budget per image.
        max_step: Optimization steps per batch.
        lr: Adadelta learning rate.
        sd: Weight of the hinge-squared DSSIM penalty.
        batch_size: Images per optimization batch.
    """

    name = "Fawkes"

    def __init__(self,
                 device: torch.device,
                 fr_models: Sequence[FaceRecognizer] = (),
                 *,
                 dssim_thresh: float = 0.017, 
                 max_step: int = 150, 
                 lr: float = 15.0,
                 sd: float = 1e6,
                 batch_size: int = 8):
        super().__init__(device, fr_models)
        self.th = dssim_thresh
        self.max_step = max_step
        self.lr = lr
        self.sd = sd
        self.batch_size = batch_size
        self.tanh_constant = 2 - 1e-6       # keeps arctanh away from +-1
        self._last_trace: List[dict] = []   # per-iteration stats of the last batch

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.NoAlign)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({"dssim_thresh": self.th, "max_step": self.max_step, "lr": self.lr, "sd": self.sd, "batch_size": self.batch_size})
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Cloak each original image on its 112-px face box and paste the perturbation back."""
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device).float()
        cloaked = []
        pbar = tqdm(range(0, clean.shape[0], self.batch_size), desc="Fawkes")
        for start in pbar:
            cloaked.append(self._compute_batch(clean[start:start + self.batch_size]))
        return paste_deltas(photos, torch.cat(cloaked, dim=0) - clean, plans)

    def _compute_batch(self, source_imgs: torch.Tensor) -> torch.Tensor:
        """Cloak one batch; a port of the release's ``FawkesMaskGeneration.compute_batch``."""
        nb_imgs = source_imgs.shape[0]

        # maximize=True: track the largest feature deviation within the DSSIM budget.
        best_bottlesim = np.zeros(nb_imgs)
        best_adv = torch.zeros_like(source_imgs)

        simg_tanh = self._preprocess_arctanh(source_imgs)
        modifier = self._init_modifier(source_imgs)
        optimizer = self._make_optimizer(modifier)

        const = torch.full((nb_imgs,), self.sd, device=self.device, dtype=torch.float32)
        const_diff = np.ones(nb_imgs)
        outside_list = np.ones(nb_imgs)

        # Iteration 1 clamps around the raw source, later ones around its tanh round trip.
        simg_raw = source_imgs

        trace = []  # per-iteration stats, logged at debug level

        for it in range(1, self.max_step + 1):
            # Back from tanh space; the perturbation is clamped to +-15 on the [0, 255] scale.
            aimg_raw = self._reverse_arctanh(simg_tanh + modifier)
            actual_modifier = torch.clamp(aimg_raw - simg_raw, -15.0, 15.0)
            aimg_raw = simg_raw + actual_modifier
            simg_raw = self._reverse_arctanh(simg_tanh)

            # Input-space loss: hinge on DSSIM = (1 - SSIM) / 2 above the threshold.
            dist_raw = (1.0 - calc_ssim(aimg_raw, simg_raw, data_range=255.0, size_average=False)) / 2.0
            dist = torch.clamp(dist_raw - self.th, min=0.0)

            # Feature-space gain, summed over the ensemble.
            bottlesim = self._calc_bottlesim(aimg_raw, simg_raw)

            const_diff_t = torch.from_numpy(const_diff).to(device=self.device, dtype=torch.float32)
            loss = torch.sum(const * torch.square(dist) - bottlesim * const_diff_t)

            optimizer.zero_grad()
            loss.backward()
            grad = modifier.grad.detach().clone()
            optimizer.step()

            if it == 1:
                # One-off signed step; the release recreates its variable here, resetting Adadelta.
                with torch.no_grad():
                    modifier -= torch.sign(grad) * 0.01
                optimizer = self._make_optimizer(modifier)

            # Per-image weight schedule and best-iterate tracking (the release's hysteresis).
            dist_raw_np = dist_raw.detach().cpu().numpy()
            bottlesim_np = bottlesim.detach().cpu().numpy()
            for e in range(nb_imgs):
                input_dist = dist_raw_np[e]
                feature_d = bottlesim_np[e]

                if input_dist <= self.th * 0.9 and const_diff[e] <= 129:
                    const_diff[e] *= 2
                    if outside_list[e] == -1:
                        const_diff[e] = 1
                    outside_list[e] = 1
                elif input_dist >= self.th * 1.1 and const_diff[e] >= 1 / 129:
                    const_diff[e] /= 2
                    if outside_list[e] == 1:
                        const_diff[e] = 1
                    outside_list[e] = -1
                else:
                    const_diff[e] = 1.0
                    outside_list[e] = 0

                if input_dist <= self.th * 1.1 and feature_d > best_bottlesim[e]:
                    best_bottlesim[e] = feature_d
                    best_adv[e] = aimg_raw[e].detach()

            trace.append({"it": it, "loss": float(loss.detach()),
                          "dist_raw": dist_raw_np.tolist(), "bottlesim": bottlesim_np.tolist(),
                          "const_diff": const_diff.tolist()})
            logger.debug("ITER %d  Total Loss: %.2f %.4f raw; diff: %.4f",
                         it, trace[-1]["loss"], float(dist_raw_np.mean()), float(bottlesim_np.mean()))

        self._last_trace = trace
        logger.info("Final feature deviation: %.4f", float(np.mean(best_bottlesim)))

        # save_last_on_failed: keep the final iterate if no better one fit the budget.
        for e in range(nb_imgs):
            if best_bottlesim[e] < 0.3 and dist_raw_np[e] < 0.015 and bottlesim_np[e] > best_bottlesim[e]:
                best_adv[e] = aimg_raw[e].detach()

        return torch.clamp(best_adv, 0.0, 255.0).detach()

    def _make_optimizer(self, modifier: torch.Tensor) -> torch.optim.Adadelta:
        """Adadelta with the Keras defaults used by the release (rho 0.95, eps 1e-7)."""
        return torch.optim.Adadelta([modifier], lr=self.lr, rho=0.95, eps=1e-7)

    def _init_modifier(self, source_imgs: torch.Tensor) -> torch.Tensor:
        """Uniform(-1e-4, 1e-4) init, as in the release."""
        return ((torch.rand_like(source_imgs) * 2 - 1) * 1e-4).requires_grad_(True)

    def _preprocess_arctanh(self, imgs: torch.Tensor) -> torch.Tensor:
        """[0, 255] pixels -> tanh space."""
        return torch.arctanh((imgs / 255.0 - 0.5) * self.tanh_constant)

    def _reverse_arctanh(self, imgs: torch.Tensor) -> torch.Tensor:
        """tanh space -> [0, 255] pixels (up to the tanh_constant margin)."""
        return (torch.tanh(imgs) / self.tanh_constant + 0.5) * 255

    def _calc_bottlesim(self, aimg_raw: torch.Tensor, simg_raw: torch.Tensor) -> torch.Tensor:
        """Squared L2 deviation from the original embedding over its norm, summed over
        the ensemble (the release's ``calc_bottlesim``).
        """
        bottlesim = torch.zeros(aimg_raw.shape[0], device=self.device)
        for fr in self.fr_models:
            bottleneck_a = self._embed(fr, aimg_raw)
            with torch.no_grad():
                bottleneck_s = self._embed(fr, simg_raw)
            scale_factor = torch.sqrt(torch.sum(torch.square(bottleneck_s), dim=1))
            bottlesim = bottlesim + torch.sum(torch.square(bottleneck_a - bottleneck_s), dim=1) / scale_factor
        return bottlesim

    def _embed(self, fr: FaceRecognizer, imgs255: torch.Tensor) -> torch.Tensor:
        """Embed pre-cropped ``[0, 255]`` faces with one FR model (no TTA, L2-normalized)."""
        return fr.forward(imgs255, need_crop=False, tta=(), l2_normalize=True).embeddings
