"""CLIP2Protect: adversarial makeup found by text-guided StyleGAN2 latent search.

Reference Paper: CLIP2Protect: Protecting Facial Privacy Using Text-Guided Makeup via Adversarial Latent Search (CVPR 2023)
Reference Code: https://github.com/fahadshamshad/Clip2Protect
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import torch
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import lpips

from frbench import FaceDetector
from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, suppress_output, tqdm
from .models.stylegan2.model import Generator
from .models.encoders.psp_encoders import Encoder4Editing
from .nce_loss import NCELoss

logger = get_logger(__name__)


class Clip2Protect(AFRMethod):
    """Makeup from a text-guided StyleGAN2 latent search that also pulls the face toward the
    target identity: ``lambda_adv * L_adv + lambda_clip * L_clip + lambda_latent * ||w - w_inv||^2``.

    Args:
        steps: Adam steps of the latent search.
        lr: Learning rate of the latent search.
        lambda_latent: Weight of the latent-distance penalty.
        lambda_clip: Weight of the CLIP makeup loss.
        lambda_adv: Weight of the adversarial FR loss.
        tune_steps: Generator fine-tuning steps per image.
        tune_l2_weight: Weight of the MSE term in the fine-tuning loss.
        tune_lr: Learning rate of the fine-tuning.
        noise_optimize: Also optimize the noise maps below 512 px.
        clip_loss_type: ``"directional"`` (the paper's Eq. 2) or ``"infonce"`` (the reference
            code's).
        source_text: Source text of the CLIP edit direction.
        makeup_prompt: Makeup text; ``data.prompt`` overrides it per call.
        stylegan_ckpt: FFHQ StyleGAN2 checkpoint; the registered asset if ``None``.
        e4e_ckpt: e4e encoder checkpoint; the registered asset if ``None``.
        clip_ckpt: CLIP ViT-B/32 checkpoint; the registered asset if ``None``.
    """

    name = "CLIP2Protect"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
        optional=("prompt",),
    )
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            steps: int = 50,
            lr: float = 0.01,
            lambda_latent: float = 0.01,
            lambda_clip: float = 0.5,
            lambda_adv: float = 1.0,
            tune_steps: int = 450,
            tune_l2_weight: float = 0.5,
            tune_lr: float = 5e-4,
            noise_optimize: bool = False,
            clip_loss_type: str = "directional",
            source_text: str = "face",
            makeup_prompt: str = "red lipstick",
            stylegan_ckpt: Optional[str] = None,
            e4e_ckpt: Optional[str] = None,
            clip_ckpt: Optional[str] = None,
            ):
        super().__init__(device, fr_models)

        self.steps = steps
        self.lr = lr
        self.lambda_latent = lambda_latent
        self.lambda_clip = lambda_clip
        self.lambda_adv = lambda_adv
        self.tune_steps = tune_steps
        self.tune_l2_weight = tune_l2_weight
        self.tune_lr = tune_lr
        self.noise_optimize = noise_optimize
        if clip_loss_type not in ("directional", "infonce"):
            raise ValueError(f"clip_loss_type must be 'directional' or 'infonce', got {clip_loss_type!r}")
        self.clip_loss_type = clip_loss_type
        self.source_text = source_text
        self.makeup_prompt = makeup_prompt
        self.stylegan_ckpt = stylegan_ckpt
        self.e4e_ckpt = e4e_ckpt
        self.clip_ckpt = clip_ckpt

        # Weights come from the asset cache unless paths are given.
        if stylegan_ckpt is None:
            stylegan_ckpt = get_asset("clip2protect/stylegan2-ffhq-config-f.pt")
        if e4e_ckpt is None:
            e4e_ckpt = get_asset("clip2protect/e4e_ffhq_encode.pt")
        if clip_ckpt is None:
            clip_ckpt = get_asset("clip2protect/ViT-B-32.pt")

        # FFHQ StyleGAN2 generator (g_ema). The pretrained weights are kept on the
        # CPU so every image is fine-tuned from the same start.
        self._pristine_g_state = torch.load(stylegan_ckpt, map_location="cpu")["g_ema"]
        self.generator = Generator(1024, 512, 8).to(device)

        # e4e FFHQ encoder: only the encoder weights and latent_avg of the pSp checkpoint, added
        # as in the reference inference (scripts/inference.py:get_latents).
        e4e = torch.load(e4e_ckpt, map_location="cpu")
        self.encoder = Encoder4Editing(50, "ir_se", SimpleNamespace(stylegan_size=1024))
        self.encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in e4e["state_dict"].items() if k.startswith("encoder.")})
        self.encoder = self.encoder.eval().to(device).requires_grad_(False)
        self.latent_avg = e4e["latent_avg"].to(device)  # (18, 512)

        with suppress_output(logger):
            self.lpips_loss = lpips.LPIPS(net="vgg").to(device)
        self.lpips_loss.requires_grad_(False)
        self.nce_loss = NCELoss(device, clip_ckpt)
        self.nce_loss.model.requires_grad_(False)

        # One RetinaFace detector, in place of MTCNN, serves every box detection.
        self.detector = FaceDetector()

        # Random-perspective augmentation, used only by the "infonce" CLIP branch.
        self.augment = T.RandomPerspective(fill=0, p=1, distortion_scale=0.5)

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=1024, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "steps": self.steps,
            "lr": self.lr,
            "lambda_latent": self.lambda_latent,
            "lambda_clip": self.lambda_clip,
            "lambda_adv": self.lambda_adv,
            "tune_steps": self.tune_steps,
            "tune_l2_weight": self.tune_l2_weight,
            "tune_lr": self.tune_lr,
            "noise_optimize": self.noise_optimize,
            "clip_loss_type": self.clip_loss_type,
            "source_text": self.source_text,
            "makeup_prompt": self.makeup_prompt,
            "stylegan_ckpt": self.stylegan_ckpt,
            "e4e_ckpt": self.e4e_ckpt,
            "clip_ckpt": self.clip_ckpt,
        })
        return config

    # Public API

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each face with adversarial makeup and paste it back.

        Args:
            data.target_img: The impersonation target, shared by all photos.
            data.prompt: Makeup text for this call, overriding ``makeup_prompt``.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        unprotected_imgs = torch.stack(crops).to(self.device)
        target_img = data.target_img.to(self.device)
        prompt = data.prompt or self.makeup_prompt

        protected = [
            self._protect_single(unprotected_imgs[i:i + 1], target_img,
                                 prompt, tag=f"{i + 1}/{unprotected_imgs.shape[0]}")
            for i in range(unprotected_imgs.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    # Core stages

    def _protect_single(self, src_img: torch.Tensor, target_img: torch.Tensor,
                        prompt: str, tag: str = "") -> torch.Tensor:
        """Run e4e inversion, generator fine-tuning, and adversarial optimization."""
        src01 = src_img / 255.

        w_inv = self._imgs2latent(src_img)                      # (1, 18, 512)
        generator, noises = self._tune_generator(src01 * 2 - 1, w_inv, tag)

        # Detect the face box once on the clean source and reuse it for every
        # crop, as the reference code does (with RetinaFace instead of MTCNN).
        box = self._detect_box(src01)
        # The reference embeds unaligned box crops of clean target photos; crop
        # the target by its own detected box, the same domain as the renders.
        target01 = target_img / 255.
        target_box = self._detect_box(target01)
        with torch.no_grad():
            target_embeddings = self._embed(self._crop(target01, target_box))

        # Trainable: the latent and, with noise_optimize, the noise maps below
        # 512 px (as in the released code).
        latent = w_inv.detach().clone().requires_grad_(True)
        latent_ref = w_inv.detach().clone()
        for noise in noises:
            noise.requires_grad_(self.noise_optimize and noise.shape[2] < 512)
        opt_params = [latent] + ([n for n in noises if n.requires_grad] if self.noise_optimize else [])
        optimizer = optim.Adam(opt_params, lr=self.lr)

        # The "infonce" CLIP branch references the inverted render; the paper's
        # "directional" branch references the original image directly.
        init_render = None
        if self.clip_loss_type == "infonce":
            with torch.no_grad():
                init_render, _ = generator([latent], input_is_latent=True, noise=noises)
                init_render = ((init_render + 1) / 2).clamp(0, 1)

        pbar = tqdm(range(self.steps), desc=f"CLIP2Protect optimizing {tag}")
        for _ in pbar:
            optimizer.zero_grad()

            render, _ = generator([latent], input_is_latent=True, noise=noises)
            render01 = ((render + 1) / 2).clamp(0, 1)           # (1, 3, 1024, 1024)

            if self.clip_loss_type == "directional":
                clip_loss = self.nce_loss.directional(src01, self.source_text, render01, prompt)
            else:
                clip_loss = self.nce_loss(init_render, self.source_text,
                                          self.augment(render01), prompt).sum()
            latent_loss = ((latent_ref - latent) ** 2).sum()

            render_embeddings = self._embed(self._crop(render01, box))
            # L_adv summed over the ensemble, D = 1 - cos, toward the target only, as in the
            # reference code (Eq. 3 also subtracts D(gen, source)).
            adv_loss = sum(1 - F.cosine_similarity(g, t).mean()
                           for g, t in zip(render_embeddings, target_embeddings))

            loss = (self.lambda_adv * adv_loss + self.lambda_clip * clip_loss
                    + self.lambda_latent * latent_loss)
            loss.backward()

            # Identity-preservation mask: only fine style layers (8-17) are edited.
            latent.grad[0, :8] = 0
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # Render the final latent after all updates, as in the paper.
        with torch.no_grad():
            render, _ = generator([latent], input_is_latent=True, noise=noises)
        return ((render + 1) / 2).clamp(0, 1) * 255.

    def _tune_generator(self, real_img: torch.Tensor, w: torch.Tensor, tag: str = ""):
        """Latent code initialization: fit the generator to reconstruct ``real_img`` at ``w``.

        Args:
            real_img: ``(1, 3, 1024, 1024)`` in ``[-1, 1]``.
            w: ``(1, 18, 512)`` latent, held fixed.

        Returns:
            The tuned generator (eval mode, gradients disabled) and its fixed
            per-layer noise tensors (17 tensors, 4x4 ... 1024x1024).
        """
        # Restart from the pristine FFHQ weights for every image.
        self.generator.load_state_dict(self._pristine_g_state, strict=False)
        self.generator.requires_grad_(True)
        optimizer = optim.Adam(self.generator.parameters(), lr=self.tune_lr)

        # Fresh random noises per image, fixed during tuning (and reused in stage 3).
        noises = [n.repeat(1, 1, 1, 1).normal_() for n in self.generator.make_noise()]
        for noise in noises:
            noise.requires_grad = False

        w = w.detach()
        real_256 = F.interpolate(real_img, size=256)
        for _ in tqdm(range(self.tune_steps), desc=f"CLIP2Protect tuning {tag}"):
            render, _ = self.generator([w], input_is_latent=True, noise=noises)
            render_256 = F.interpolate(render, size=256)
            loss = (self.lpips_loss(render_256, real_256).sum()
                    + self.tune_l2_weight * F.mse_loss(render_256, real_256))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.generator.eval().requires_grad_(False)
        return self.generator, noises

    def _imgs2latent(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode ``[0, 255]`` RGB faces to W+ latents ``(B, 18, 512)`` (e4e + latent_avg)."""
        x = F.interpolate(imgs / 255., size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        with torch.no_grad():
            codes = self.encoder(x * 2 - 1)
        return codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

    # Helpers

    def _embed(self, imgs01: torch.Tensor) -> List[torch.Tensor]:
        """Embed ``[0, 1]`` RGB face crops with every FR model, one ``(1, D)`` tensor per model."""
        return [fr.forward(imgs01 * 255., need_crop=False, tta=(), l2_normalize=False).embeddings
                for fr in self.fr_models]

    def _detect_box(self, img01: torch.Tensor) -> Tuple[int, int, int, int]:
        """The main face box ``(x1, y1, x2, y2)`` of a ``[0, 1]`` image, expanded 1.2x like MTCNN's
        looser crops; the whole image if no face is found.
        """
        h, w = img01.shape[-2:]
        det = self.detector.detect(img01 * 255.).detections[0]
        if det is None or det.shape[0] == 0:
            logger.warning("No face detected; using the full image as the crop box.")
            return 0, 0, w, h
        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) * det[:, 5]
        x1, y1, x2, y2 = det[areas.argmax()][:4].float()
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half_w, half_h = (x2 - x1) * 0.6, (y2 - y1) * 0.6  # 1.2× / 2
        x1 = int(torch.floor(cx - half_w).item())
        y1 = int(torch.floor(cy - half_h).item())
        x2 = int(torch.ceil(cx + half_w).item())
        y2 = int(torch.ceil(cy + half_h).item())
        return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)

    def _crop(self, img01: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
        """Slice the (differentiable) face-box crop out of a ``(1, 3, H, W)`` image."""
        x1, y1, x2, y2 = box
        return img01[:, :, y1:y2, x1:x2]
