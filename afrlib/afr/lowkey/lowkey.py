"""LowKey: an untargeted whole-photo attack on face-recognition galleries.

Reference Paper: LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition (ICLR 2021)
Reference Code: the ICLR 2021 supplementary material (https://openreview.net/forum?id=hJmtwocEqzc)
"""
import math
from typing import List, Dict, Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lpips import LPIPS
from frbench import FaceDetector, FRDetectResult

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
)
from afrlib.afr.base import AFRMethod
from afrlib.fr import FaceRecognizer
from afrlib.utils.log import tqdm, get_logger

logger = get_logger(__name__)


class GaussianSmoothing(nn.Module):
    """Fixed depthwise Gaussian blur with reflect padding; the loss's blurred view makes the
    perturbation survive smoothing.
    """

    def __init__(self, channels: int = 3, kernel_size: int = 7, sigma: float = 3.0):
        super().__init__()
        grid = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1) / 2
        kernel_1d = torch.exp(-((grid - mean) / sigma) ** 2 / 2) / (sigma * math.sqrt(2 * math.pi))
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.groups = channels
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x, weight=self.weight, groups=self.groups)


class LowKey(AFRMethod):
    """Perturbs the whole photo to move its embedding away from the clean one, blurred and
    unblurred (paper Eq. 1), under an LPIPS penalty instead of an L-inf budget.

    Args:
        steps: Signed-gradient steps per attempt.
        lr: Step size, on the ``[0, 1]`` scale.
        lpips_weight: Weight of the LPIPS penalty.
        init_noise: Half-width of the uniform initial noise, on the ``[0, 1]`` scale.
        attempts: Restarts; the one with the largest feature distance is kept.
        batch_size: Photos optimized together.
        blur_kernel: Side of the Gaussian blur kernel.
        blur_sigma: Standard deviation of the Gaussian blur.
    """

    name = "LowKey"

    #: Side of the ArcFace-aligned attack crop LowKey detects and optimizes.
    ALIGN_SIZE = 224

    def __init__(self,
                 device: torch.device,
                 fr_models: Sequence[FaceRecognizer] = (),
                 *,
                 epochs: int = 50,
                 learning_rate: float = 0.0025,
                 lpips_weight: float = 0.05,
                 init_noise: float = 0.005,
                 attempts: int = 1,
                 batch_size: int = 4,
                 gaussian_kernel_size: int = 7,
                 gaussian_sigma: float = 3.0):
        super().__init__(device, fr_models)
        self.fd = FaceDetector().to(self.device).eval()
        self.lpips_net = LPIPS(net='alex').to(self.device).eval().requires_grad_(False)
        self.gaussian = GaussianSmoothing(3, gaussian_kernel_size, gaussian_sigma).to(self.device)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lpips_weight = lpips_weight
        self.init_noise = init_noise
        self.attempts = attempts
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        # [0, 1] -> [-1, 1], the LPIPS input range.
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def attack_spec(self) -> ImageSpec:
        # LowKey consumes the whole photo and re-detects internally.
        return ImageSpec(channel=ChannelType.RGB, crop_type=CropType.Original, align_type=AlignType.NoAlign)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lpips_weight": self.lpips_weight,
            "init_noise": self.init_noise,
            "attempts": self.attempts,
            "batch_size": self.batch_size,
            "gaussian_kernel_size": self.gaussian_kernel_size,
            "gaussian_sigma": self.gaussian_sigma,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Perturb each whole photo, aligning it to the attack crop on every forward pass; faces
        come from LowKey's own RetinaFace, falling back to ``eval_geometry``.

        Args:
            data: ``eval_imgs`` as original photos, of any sizes.

        Returns:
            The protected originals, one per input image, at the input frame.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        geometry = data.eval_geometry
        num_imgs = len(photos)
        if geometry is not None and len(geometry) != num_imgs:
            raise ValueError(
                f"eval_geometry length {len(geometry)} != eval_imgs length {num_imgs}"
            )
        protected: List[torch.Tensor] = []
        num_batches = math.ceil(num_imgs / self.batch_size)
        for start in tqdm(range(0, num_imgs, self.batch_size), total=num_batches, desc="LowKey"):
            end = start + self.batch_size
            geo = None if geometry is None else geometry[start:end]
            protected.extend(self._protect_batch(photos[start:end], geo))
        return protected

    def _protect_batch(
        self,
        raw_imgs: List[torch.Tensor],
        geometry: Optional[Sequence[Any]] = None,
    ) -> List[torch.Tensor]:
        raw_imgs = [img.to(self.device) for img in raw_imgs]

        # Detect once; the clean embeddings of the aligned crops stay fixed
        # during the optimization.
        with torch.no_grad():
            dets = _fill_missing_detections(
                self.fd.detect(raw_imgs), geometry, self.device
            )
            clean = self._align_largest(raw_imgs, dets).div(255.)  # [0, 1] crops
            clean_feats = [self._embed(fr, clean) for fr in self.fr_models]
            raw01 = [img.div(255.) for img in raw_imgs]

        # The attack variable is the full photo, not the crop; it is aligned
        # differentiably on every forward pass.
        best_protected = [r.clone() for r in raw01]
        best_dist = torch.full((len(raw01),), float("-inf"), device=self.device)

        for attempt in range(self.attempts):
            logger.info("LowKey attempt %d/%d", attempt + 1, self.attempts)
            adv = [
                (r + 2 * self.init_noise * torch.rand_like(r) - self.init_noise)
                .detach()
                .requires_grad_(True)
                for r in raw01
            ]

            for _ in tqdm(range(self.epochs), desc=f"LowKey (attempt {attempt + 1}/{self.attempts})"):
                feat_loss, lpips_loss = self._losses(adv, dets, clean_feats, raw01)
                # Maximize feature distance while staying perceptually close.
                loss = feat_loss - self.lpips_weight * lpips_loss
                grads = torch.autograd.grad(loss, adv)
                # Signed gradient ascent; LPIPS (not an L-inf ball) bounds the change.
                adv = [
                    (a + self.learning_rate * g.sign()).detach().requires_grad_(True)
                    for a, g in zip(adv, grads)
                ]

            adv = [a.detach().clamp(0., 1.) for a in adv]
            with torch.no_grad():
                dist = self._feature_distance(
                    [self._align_one(a, d) for a, d in zip(adv, dets.detections)],
                    clean_feats,
                )
            for i in range(len(adv)):
                if dist[i] > best_dist[i]:
                    best_dist[i] = dist[i]
                    best_protected[i] = adv[i]

        # The optimized variable is the full photo; there is no crop to paste back.
        return [p.mul(255.) for p in best_protected]

    def _align_largest(self, imgs: List[torch.Tensor], detections: FRDetectResult) -> torch.Tensor:
        """ArcFace-align the largest detected face of each image; raise if one has none."""
        groups = self.fd.align(imgs, detections, size=(self.ALIGN_SIZE, self.ALIGN_SIZE), keep_largest=True)
        missing = [idx for idx, group in enumerate(groups) if group.shape[0] != 1]
        if missing:
            raise ValueError(f"LowKey requires one detectable face per image; failed indices: {missing}")
        return torch.cat(groups, dim=0)

    def _align_one(self, img01: torch.Tensor, detection: torch.Tensor) -> torch.Tensor:
        """Differentiably align one full image to the attack size with a raw ``grid_sample``, so the
        high-frequency perturbation the attack relies on is not smoothed away.
        """
        crop = self.fd.align(
            [img01 * 255.],
            [detection],
            size=(self.ALIGN_SIZE, self.ALIGN_SIZE),
            keep_largest=True,
            max_supersample=1,
        )[0]
        if crop.shape[0] != 1:
            raise ValueError("LowKey requires one detectable face per image")
        return crop.div(255.)

    def _losses(
        self,
        adv: List[torch.Tensor],
        detections: FRDetectResult,
        clean_feats: List[torch.Tensor],
        raw01: List[torch.Tensor],
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """The feature and LPIPS terms over the ragged batch of full images."""
        aligned = [
            self._align_one(a, d) for a, d in zip(adv, detections.detections)
        ]
        feat_loss = self._feature_distance(aligned, clean_feats).mean()
        lpips_losses = [
            self.lpips_net(self.normalize(a.unsqueeze(0)), self.normalize(r.unsqueeze(0))).mean()
            for a, r in zip(adv, raw01)
        ]
        return feat_loss, torch.stack(lpips_losses).mean()

    def _feature_distance(
        self, aligned: "torch.Tensor | List[torch.Tensor]", clean_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """Per-image feature term of paper Eq. 1, averaged over the models and over the blurred and
        unblurred views of the attack-size crops.
        """
        imgs = (
            torch.cat(aligned, dim=0) if isinstance(aligned, list) else aligned
        )
        blurred = self.gaussian(imgs)
        dist = torch.zeros(imgs.shape[0], device=self.device)
        for fr, clean in zip(self.fr_models, clean_feats):
            for view in (imgs, blurred):
                adv = self._embed(fr, view)
                dist = dist + self._normalized_squared_feature_distance(adv, clean)
        return dist / (2 * len(self.fr_models))

    @staticmethod
    def _normalized_squared_feature_distance(
        adv: torch.Tensor,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        """Per-model feature term of paper Eq. 1: ``||adv - clean||^2 / ||clean||``."""
        denominator = clean.norm(dim=1).clamp_min(1e-12)
        return (adv - clean).square().sum(dim=1) / denominator

    def _embed(self, fr: FaceRecognizer, aligned01: torch.Tensor) -> torch.Tensor:
        return fr.forward(aligned01.clamp(0., 1.).mul(255.), need_crop=False, need_align=False,
                          tta=("flip_horizontal",), l2_normalize=True).embeddings


def _geometry_detection(entry: Any, device: torch.device) -> Optional[torch.Tensor]:
    """Build one RetinaFace-layout detection from stored eval_geometry."""
    if not isinstance(entry, dict):
        return None
    landmarks = entry.get("landmarks")
    if landmarks is None:
        return None
    points = torch.as_tensor(landmarks, dtype=torch.float32, device=device)
    box = entry.get("face_box")
    boxes = None if box is None else torch.as_tensor(box, dtype=torch.float32, device=device)
    return FRDetectResult.from_landmarks(points, boxes=boxes).detections[0]


def _fill_missing_detections(
    detections: FRDetectResult,
    geometry: Optional[Sequence[Any]],
    device: torch.device,
) -> FRDetectResult:
    """Replace empty RetinaFace slots with stored 5-point geometry when present."""
    filled = list(detections.detections)
    for index, det in enumerate(filled):
        if det is not None and det.shape[0] > 0:
            continue
        entry = None
        if geometry is not None and index < len(geometry):
            entry = geometry[index]
        fallback = _geometry_detection(entry, device)
        if fallback is None:
            continue
        logger.warning(
            "LowKey RetinaFace found no face on eval image %s; "
            "using stored eval_geometry landmarks",
            index,
        )
        filled[index] = fallback
    return FRDetectResult(detections=filled)
