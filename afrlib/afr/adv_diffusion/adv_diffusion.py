"""Adv-Diffusion: identity-guided Stable Diffusion inpainting for face impersonation.

Reference Paper: Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model (AAAI 2024)
Reference Code: https://github.com/kopperx/Adv-Diffusion
"""
from typing import Dict, List, Optional, Any, Sequence
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from afrlib.afr.adv_diffusion.ldm.util import instantiate_from_config
from afrlib.afr.adv_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from .face_parser import FaceParseNet50
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, suppress_output, tqdm
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)

logger = get_logger(__name__)

# SD2 inpainting inference config, shipped with the package.
_DEFAULT_SAMPLER_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "v2-inpainting-inference.yaml")

class Adv_Diffusion(AFRMethod):
    """Regenerates the identity-agnostic face regions with identity guidance toward the target.

    Args:
        sampler_weight: Stable Diffusion 2 inpainting checkpoint; the registered asset if ``None``.
        sampler_config: The LDM inference config.
        face_parser_weight: Face-parser weights; the registered asset if ``None``.
        t: Forward-diffusion start step.
        ddim_steps: Requested DDIM steps (``t=999`` gives 46).
        classifier_scale: Identity-guidance scale ``s``.
    """

    name = "Adv-Diffusion"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        sampler_weight: Optional[str] = None,
        sampler_config: str = _DEFAULT_SAMPLER_CONFIG,
        face_parser_weight: Optional[str] = None,
        t: int = 999,
        ddim_steps: int = 45,
        classifier_scale: float = 300.0,
    ):
        super().__init__(device, fr_models)
        self.sampler_weight = sampler_weight
        self.sampler_config = sampler_config
        self.face_parser_weight = face_parser_weight

        # Explicit weight paths override the registered assets.
        if sampler_weight is None:
            sampler_weight = get_asset("adv_diffusion/512-inpainting-ema.ckpt")
        if face_parser_weight is None:
            face_parser_weight = get_asset("adv_diffusion/38_G.pth")

        # Frozen SD2 inpainting LDM from the vendored ``ldm/`` package; its build-time prints
        # go to the DEBUG log.
        config = OmegaConf.load(sampler_config)
        with suppress_output(logger):
            diff_model = instantiate_from_config(config.model)
            state_dict = torch.load(sampler_weight, map_location="cpu", weights_only=False)["state_dict"]
            diff_model.load_state_dict(state_dict, strict=False)
        diff_model.requires_grad_(False)
        self.diff_model = diff_model.to(self.device).eval()
        self.ddim_sampler = DDIMSampler(self.diff_model, device=self.device)

        # Face parser producing a 19-class semantic segmentation of the face.
        self.face_parser = FaceParseNet50(num_classes=19)
        self.face_parser.load_state_dict(
            torch.load(face_parser_weight, map_location="cpu", weights_only=False))
        self.face_parser.requires_grad_(False)
        self.face_parser = self.face_parser.to(self.device).eval()

        self.t = t                              # forward-diffusion start step (Algorithm 1)
        self.ddim_steps = ddim_steps            # DDIM steps requested (t=999 -> 46 actual)
        self.classifier_scale = classifier_scale  # s: identity attack strength
        self.eta = 1.0                          # DDIM eta (1.0 = ancestral/stochastic sampling)
        # Parser classes kept as the condition (skin, nose, glasses, eyes, brows, mouth, lips);
        # the rest (background, ears, hair, hat, jewelry, neck, clothes) is regenerated.
        self.identity_sensitive_parts = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=512, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "sampler_weight": self.sampler_weight,
            "sampler_config": self.sampler_config,
            "face_parser_weight": self.face_parser_weight,
            "t": self.t,
            "ddim_steps": self.ddim_steps,
            "classifier_scale": self.classifier_scale,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each face toward the target identity and paste it back.

        Args:
            data.eval_imgs: Original photos; each is attacked separately.
            data.target_img: The impersonation target, shared by all photos.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        unprotected_imgs = torch.stack(crops).to(self.device)
        target_img = data.target_img.to(self.device)

        # Target identity embeddings are constants w.r.t. the optimization.
        with torch.no_grad():
            target_embeddings = self._embed(target_img)  # per model, (1, D)

        # The face parser and the SD2 VAE take [-1, 1]; the FR models take [0, 255].
        src_pm1 = unprotected_imgs / 127.5 - 1.0
        protected = [
            self._run_diffusion(src_pm1[i:i + 1], target_embeddings)
            for i in range(src_pm1.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _embed(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """Embed aligned ``[0, 255]`` crops with every FR model, one tensor per model."""
        return [fr.forward(imgs, need_crop=False, tta=(), l2_normalize=False).embeddings for fr in self.fr_models]

    def _run_diffusion(
        self,
        src_pm1: torch.Tensor,
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Guided reverse diffusion for one source, in ``[0, 255]``; a non-finite decode is
        resampled once, then falls back to the source crop.
        """
        uc_full, z0 = self._prepare_inpainting(src_pm1)
        for attempt in range(2):
            decoded = self._guided_sample(src_pm1, target_embeddings, uc_full, z0)
            if decoded is not None:
                return decoded
            if attempt == 0:
                logger.warning(
                    "Adv-Diffusion decode was non-finite; resampling this crop once"
                )
        logger.warning(
            "Adv-Diffusion decode was non-finite after resample; returning the source crop"
        )
        return (src_pm1 + 1.0) / 2.0 * 255.0

    def _prepare_inpainting(
        self,
        src_pm1: torch.Tensor,
    ) -> tuple:
        """Mask, inpainting condition, and source latent ``z0`` for one crop."""
        B, _, H, W = src_pm1.shape

        # Identity-agnostic mask (1 = inpaint) and the preserved identity-only image.
        mask = self._get_masks(src_pm1)                     # (B, 1, H, W)
        masked_img = self._get_masked_imgs(src_pm1, mask)   # (B, 3, H, W)

        # Inpainting conditioning: empty text prompt + concat(mask, masked latent).
        c_concat = []
        for ck in self.diff_model.concat_keys:  # ("mask", "masked_image")
            if ck == self.diff_model.masked_image_key:
                cc = self.diff_model.get_first_stage_encoding(
                    self.diff_model.encode_first_stage(masked_img))  # (B, 4, H/8, W/8)
            else:
                cc = F.interpolate(mask, size=(H // 8, W // 8))       # (B, 1, H/8, W/8)
            c_concat.append(cc)
        c_concat = torch.cat(c_concat, dim=1)  # (B, 5, H/8, W/8)

        uc_cross = self.diff_model.get_unconditional_conditioning(B, "")
        uc_full = {"c_concat": [c_concat], "c_crossattn": [uc_cross]}
        z0 = self.diff_model.get_first_stage_encoding(
            self.diff_model.encode_first_stage(src_pm1)
        )
        return uc_full, z0

    def _guided_sample(
        self,
        src_pm1: torch.Tensor,
        target_embeddings: List[torch.Tensor],
        uc_full: Dict[str, Any],
        z0: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """One ancestral DDIM pass. Returns a ``[0, 255]`` crop, or ``None`` if decode is non-finite."""
        B = src_pm1.shape[0]
        # z_T = sqrt(a_bar_T) z_0 + sqrt(1 - a_bar_T) eps  (Algorithm 1, line 3).
        t = torch.full((B,), self.t, device=self.device, dtype=torch.long)
        x = self.diff_model.q_sample(x_start=z0, t=t)

        # DDIM schedule over [0, t] (t=999, ddim_steps=45 -> 46 actual).
        self.ddim_sampler.make_schedule(
            ddim_num_steps=self.ddim_steps, ddim_eta=self.eta, verbose=False, _t=self.t + 1)
        time_range = np.flip(self.ddim_sampler.ddim_timesteps)
        total_steps = self.ddim_sampler.ddim_timesteps.shape[0]

        for i, step in enumerate(tqdm(time_range, desc=self.name, total=total_steps)):
            index = total_steps - i - 1
            ts = torch.full((B,), step, device=self.device, dtype=torch.long)

            # Unguided step and x_0 estimate; classifier-free guidance is off (empty prompt),
            # so only the unconditional branch is evaluated.
            x_prev, pred_x0 = self.ddim_sampler.p_sample_ddim(x, uc_full, ts, index=index)
            # Identity guidance w_t = sigma_t * s (Algorithm 1, lines 8-9); a non-finite guided
            # update falls back to the unguided step.
            sigma_t = self.ddim_sampler.ddim_sigmas[index]
            if not torch.isfinite(x_prev).all() or not torch.isfinite(pred_x0).all():
                logger.warning(
                    "Adv-Diffusion unguided step was non-finite at t=%s; keeping the previous latent",
                    int(step),
                )
                continue
            guidance = self._identity_guidance(pred_x0, target_embeddings)
            if not torch.isfinite(guidance).all():
                logger.warning(
                    "Adv-Diffusion identity guidance was non-finite at t=%s; using the unguided update",
                    int(step),
                )
                x = x_prev
                continue
            candidate = x_prev + sigma_t * guidance
            x = candidate if torch.isfinite(candidate).all() else x_prev

        decoded = self.diff_model.decode_first_stage(x)
        if not torch.isfinite(decoded).all():
            return None
        img = torch.clamp(decoded, -1.0, 1.0)
        return (img + 1.0) / 2.0 * 255.0

    def _identity_guidance(
        self,
        pred_x0: torch.Tensor,
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """``classifier_scale`` times the gradient, w.r.t. ``pred_x0``, of the decoded image's
        ensemble-mean cosine similarity to the target.
        """
        with torch.enable_grad():
            x_in = pred_x0.detach().requires_grad_(True)
            img = self.diff_model.decode_first_stage(x_in)          # [-1, 1], (B, 3, H, W)
            img255 = ((img + 1.0) / 2.0 * 255.0).clamp(0, 255)      # FR expects [0, 255]

            score = 0.0
            for k, fr in enumerate(self.fr_models):
                feat = fr.forward(img255, need_crop=False, tta=(), l2_normalize=False).embeddings
                score = score + F.cosine_similarity(feat, target_embeddings[k]).sum()
            grad = torch.autograd.grad(score / len(self.fr_models), x_in)[0]
        return grad * self.classifier_scale

    def _get_masks(self, imgs: torch.Tensor) -> torch.Tensor:
        """Binary ``(B, 1, H, W)`` mask of a ``[-1, 1]`` batch: 1 to inpaint, 0 to keep."""
        with torch.no_grad():
            logits = self.face_parser(imgs)[0][-1]  # seg2 branch, (B, 19, h', w')
            logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=True)
            labels = torch.argmax(logits, dim=1)    # (B, H, W)

            sensitive = torch.zeros_like(labels, dtype=torch.bool)
            for part in self.identity_sensitive_parts:
                sensitive |= labels == part
            # Invert: identity-agnostic regions become 1 (to be inpainted).
            mask = (~sensitive).float().reshape(imgs.shape[0], 1, *imgs.shape[2:])
        return mask

    def _get_masked_imgs(self, imgs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Keep the identity-sensitive regions (mask == 0), zero out the rest."""
        return imgs * (masks < 0.5)
