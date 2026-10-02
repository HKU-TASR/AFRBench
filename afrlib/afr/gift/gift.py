"""GIFT: impersonation by StyleGAN2 latent search with landmark regularization.

Reference Paper: Transferable Adversarial Facial Images for Privacy Protection (ACM MM 2024)
Reference Code: none published; the inversion follows https://github.com/kkang831/BDInvert_Release
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import lpips

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, suppress_output, tqdm
from .models.genforce.stylegan2_generator import StyleGAN2Generator
from .models.basecode_encoder import encoder_simple
from .models.bisenet.model import BiSeNet

logger = get_logger(__name__)

# ImageNet statistics; the BiSeNet face parser was trained with this normalization.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GIFT(AFRMethod):
    """Inverts each face into StyleGAN2's base and detail codes, then searches them for a render
    the surrogates read as the target while a face-parsing loss holds the landmarks in place.

    Args:
        t1_steps: Adam steps of the inversion (paper Eq. 2).
        t1_lr: Learning rate of the inversion.
        alpha_lpips: Weight of the LPIPS term in the reconstruction loss.
        t2_steps: Adam steps of the adversarial search.
        t2_lr: Learning rate of the adversarial search.
        lambda_adv: Weight of the identity loss.
        lambda_sem: Weight of the face-parsing loss.
        trans_prob: Probability of each input-diversity transform.
        resize_rate: Smallest relative size of the resize-and-pad transform.
        noise_scale: Variance of the input-diversity Gaussian noise.
        basecode_size: Spatial side of the base code.
        stylegan_ckpt: StyleGAN2 FFHQ-1024 checkpoint; the registered asset if ``None``.
        encoder_ckpt: Base-code encoder checkpoint; the registered asset if ``None``.
        parsing_ckpt: BiSeNet face-parser checkpoint; the registered asset if ``None``.
    """

    name = "GIFT"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            t1_steps: int = 1200,
            t1_lr: float = 0.01,
            alpha_lpips: float = 10.0,
            t2_steps: int = 50,
            t2_lr: float = 0.002,
            lambda_adv: float = 1.0,
            lambda_sem: float = 0.01,
            trans_prob: float = 0.5,
            resize_rate: float = 0.9,
            noise_scale: float = 0.1,
            basecode_size: int = 16,
            stylegan_ckpt: Optional[str] = None,
            encoder_ckpt: Optional[str] = None,
            parsing_ckpt: Optional[str] = None,
            ):
        super().__init__(device, fr_models)

        self.t1_steps = t1_steps
        self.t1_lr = t1_lr
        self.alpha_lpips = alpha_lpips
        self.t2_steps = t2_steps
        self.t2_lr = t2_lr
        self.lambda_adv = lambda_adv
        self.lambda_sem = lambda_sem
        self.trans_prob = trans_prob
        self.resize_rate = resize_rate
        self.noise_scale = noise_scale
        self.basecode_size = basecode_size
        self.stylegan_ckpt = stylegan_ckpt
        self.encoder_ckpt = encoder_ckpt
        self.parsing_ckpt = parsing_ckpt

        # Weights are resolved from the AFRLib asset cache unless explicit paths are given.
        if stylegan_ckpt is None:
            stylegan_ckpt = get_asset("gift/stylegan2_ffhq1024.pth")
        if encoder_ckpt is None:
            encoder_ckpt = get_asset("gift/basecode_encoder_ffhq1024_16.pth")
        if parsing_ckpt is None:
            parsing_ckpt = get_asset("gift/faceparsing_bisenet.pth")

        # Frozen GenForce FFHQ-1024 StyleGAN2; only the latent codes are optimized.
        gen_state = torch.load(stylegan_ckpt, map_location="cpu")
        self.generator = StyleGAN2Generator(resolution=1024)
        self.generator.load_state_dict(gen_state.get("generator_smooth", gen_state.get("generator")))
        self.generator = self.generator.to(device).eval().requires_grad_(False)
        # Layer at which the F base code is injected: 'x03' for a 16x16 base code
        # (same formula as BDInvert's invert.py).
        layer_idx = int(np.log2(basecode_size) - 2) * 2
        self.basecode_layer = f"x{layer_idx - 1:02d}"
        self.w_avg = self.generator.truncation.w_avg.detach().clone().to(device)  # (512,)
        self.num_layers = self.generator.num_layers                              # 18

        # BDInvert base-code encoder E_b (F latent space, 16x16).
        enc_out = self.generator.synthesis.get_nf(basecode_size)                 # 512 channels
        self.encoder = encoder_simple([3, basecode_size * 8, basecode_size * 8], [enc_out, basecode_size, basecode_size])
        self.encoder.load_state_dict(torch.load(encoder_ckpt, map_location="cpu")["encoder"])
        self.encoder = self.encoder.to(device).eval().requires_grad_(False)

        # BiSeNet face parser / semantic encoder E_s (KLR).
        self.parser = BiSeNet(n_classes=19)
        self.parser.load_state_dict(torch.load(parsing_ckpt, map_location="cpu"))
        self.parser = self.parser.to(device).eval().requires_grad_(False)
        self._seg_mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._seg_std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

        with suppress_output(logger):
            self.lpips_loss = lpips.LPIPS(net="vgg").to(device)
        self.lpips_loss.requires_grad_(False)


    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=1024, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "t1_steps": self.t1_steps,
            "t1_lr": self.t1_lr,
            "alpha_lpips": self.alpha_lpips,
            "t2_steps": self.t2_steps,
            "t2_lr": self.t2_lr,
            "lambda_adv": self.lambda_adv,
            "lambda_sem": self.lambda_sem,
            "trans_prob": self.trans_prob,
            "resize_rate": self.resize_rate,
            "noise_scale": self.noise_scale,
            "basecode_size": self.basecode_size,
            "stylegan_ckpt": self.stylegan_ckpt,
            "encoder_ckpt": self.encoder_ckpt,
            "parsing_ckpt": self.parsing_ckpt,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each face toward the target and paste it back.

        Args:
            data.target_img: The impersonation target as a whole square frame, shared by all
                photos.

        Returns:
            The protected originals, one per input image, at the input frame.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        unprotected_imgs = torch.stack(crops).to(self.device)
        target_img = data.target_img.to(self.device)

        protected = [
            self._protect_single(unprotected_imgs[i:i + 1], target_img,
                                 tag=f"{i + 1}/{unprotected_imgs.shape[0]}")
            for i in range(unprotected_imgs.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _protect_single(self, src_img: torch.Tensor, target_img: torch.Tensor, tag: str = "") -> torch.Tensor:
        """Run latent code initialization, then global adversarial latent search."""
        src01 = src_img / 255.                                  # (1, 3, 1024, 1024) in [0, 1]
        src_pm1 = src01 * 2 - 1                                 # [-1, 1] reconstruction target

        base, detail = self._initialize_latent(src_pm1, tag)
        base, detail = self._adversarial_search(base, detail, src01, target_img, tag)

        with torch.no_grad():
            render = self.generator.synthesis(
                detail, randomize_noise=False, basecode_layer=self.basecode_layer, basecode=base)["image"]
        return ((render + 1) / 2).clamp(0, 1) * 255.

    def _initialize_latent(self, src_pm1: torch.Tensor, tag: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Invert the source into its base and detail codes (paper Eq. 2); both come back as leaf
        tensors ready for the adversarial search.
        """
        with torch.no_grad():
            enc_in = F.interpolate(src_pm1, size=(self.basecode_size * 8, self.basecode_size * 8),
                                   mode="bilinear", align_corners=False, antialias=True)
            base = self.encoder(enc_in).clone()                 # (1, 512, 16, 16)
        detail = self.w_avg.view(1, 1, -1).repeat(1, self.num_layers, 1).clone()  # (1, 18, 512)
        base.requires_grad_(True)
        detail.requires_grad_(True)

        optimizer = optim.Adam([base, detail], lr=self.t1_lr)
        target_256 = F.interpolate(src_pm1, size=256, mode="bilinear", align_corners=False, antialias=True)

        pbar = tqdm(range(self.t1_steps), desc=f"GIFT inverting {tag}")
        for _ in pbar:
            optimizer.zero_grad()
            render = self.generator.synthesis(
                detail, randomize_noise=False, basecode_layer=self.basecode_layer, basecode=base)["image"]
            mse = ((render - src_pm1) ** 2).mean()
            render_256 = F.interpolate(render, size=256, mode="bilinear", align_corners=False, antialias=True)
            perceptual = self.lpips_loss(render_256, target_256).mean()
            loss = mse + self.alpha_lpips * perceptual
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        return base, detail

    def _adversarial_search(self, base: torch.Tensor, detail: torch.Tensor, src01: torch.Tensor,
                           target_img: torch.Tensor, tag: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Global Adversarial Latent Search + Key Landmark Regularization (Eq. 3-5)."""
        with torch.no_grad():
            # Paper Eq. 3 embeds x_t directly; AMT-GAN likewise downsamples the
            # whole square target without detection/alignment.
            target_embeddings = self._embed(target_img / 255.)
            # Source face-parsing labels (KLR reference), computed once.
            src_seg_labels = self.parser(self._seg_preprocess(src01))[0].argmax(dim=1)  # (1, 512, 512)

        optimizer = optim.Adam([base, detail], lr=self.t2_lr)
        pbar = tqdm(range(self.t2_steps), desc=f"GIFT optimizing {tag}")
        for _ in pbar:
            optimizer.zero_grad()
            render = self.generator.synthesis(
                detail, randomize_noise=False, basecode_layer=self.basecode_layer, basecode=base)["image"]
            render01 = ((render + 1) / 2).clamp(0, 1)

            # L_adv: impersonate the target, mean of (1 - cos) over the ensemble,
            # measured through the input-diversity transform T.
            render_embeddings = self._embed(self._transform(render01))
            adv_loss = sum(1 - F.cosine_similarity(g, t).mean()
                           for g, t in zip(render_embeddings, target_embeddings)) / len(self.fr_models)

            # L_sem: keep the source's face-parsing layout (KLR).
            seg_logits = self.parser(self._seg_preprocess(render01))[0]
            sem_loss = F.cross_entropy(seg_logits, src_seg_labels)

            loss = self.lambda_adv * adv_loss + self.lambda_sem * sem_loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(adv=f"{adv_loss.item():.3f}", sem=f"{sem_loss.item():.3f}")

        return base, detail

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Input-diversity transform ``T``: random Gaussian noise and a random square
        resize-and-pad, each with probability ``trans_prob``.
        """
        if random.random() < self.trans_prob:
            x = x + torch.rand(1, device=x.device) * (self.noise_scale ** 0.5) * torch.randn_like(x)
        if random.random() < self.trans_prob:
            h, w = x.shape[-2:]
            if h != w:
                raise ValueError(f"GIFT input diversity expects a square frame, got {(h, w)}")
            low = int(h * self.resize_rate)
            # One sample from [low, h), as in AMT-GAN; resize_rate=1 leaves the frame unchanged.
            rnd = random.randrange(low, h) if low < h else h
            x = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
            pad_top = random.randrange(h - rnd) if rnd < h else 0
            pad_left = random.randrange(w - rnd) if rnd < w else 0
            x = F.pad(x, [pad_left, w - rnd - pad_left, pad_top, h - rnd - pad_top])
        # The Gaussian noise can leave [0, 1]; _embed later rescales to [0, 255].
        return x.clamp(0, 1)

    def _seg_preprocess(self, imgs01: torch.Tensor) -> torch.Tensor:
        """Resize ``[0, 1]`` RGB to 512x512 and ImageNet-normalize it for the face parser."""
        x = F.interpolate(imgs01.contiguous(), size=(512, 512),
                          mode="bilinear", align_corners=False, antialias=True)
        return (x - self._seg_mean) / self._seg_std

    def _embed(self, imgs01: torch.Tensor) -> List[torch.Tensor]:
        """Embed whole ``[0, 1]`` RGB frames with every FR model, one ``(1, D)`` tensor per model."""
        imgs255 = imgs01.mul(255.).contiguous()
        return [fr.forward(imgs255, need_crop=False, tta=(), l2_normalize=False).embeddings
                for fr in self.fr_models]
