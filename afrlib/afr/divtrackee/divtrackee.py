"""DivTrackee: diverse StyleGAN2 makeup edits against dynamic FR trackers.

Reference Paper: DivTrackee versus DynTracker: Promoting Diversity in Anti-Facial Recognition against Dynamic FR Strategy (ACM CCS 2025)
Reference Code: https://github.com/fiora6/divtrackee
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import deque
from types import SimpleNamespace

import numpy as np
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


class DivTrackee(AFRMethod):
    """Makeup edits in StyleGAN2 W+ space that move each face away from the user's identity,
    toward a random auxiliary identity, and away from that user's earlier protected images.

    Args:
        steps: Adam steps of the latent search.
        lr: Learning rate of the latent search.
        lambda_latent: Weight of the latent-distance penalty.
        lambda_clip: Weight of the CLIP makeup loss.
        lambda_adv: Weight of the identity losses.
        lambda_queue: Weight of the diversity loss over the queue.
        guide_weight: Weight of the auxiliary-identity guidance within the identity losses.
        guide_margin: Similarity to the auxiliary identity beyond which guidance stops.
        num_aug: Random views per CLIP loss evaluation.
        queue_size: Protected images remembered for the diversity loss. The queue persists
            across :meth:`protect` calls, so one instance serves one user until
            :meth:`reset_protectee_state`.
        tune_steps: Pivot-tuning steps per image.
        tune_lr: Learning rate of the pivot tuning.
        noise_optimize: Also optimize the noise maps below 512 px.
        clip_loss_type: ``"infonce"`` (the reference code's) or ``"directional"`` (the paper's
            Eq. 13).
        optimizer_name: ``"adam"`` (the paper's) or ``"sgd"`` for ablations.
        source_text: Source text of the CLIP edit direction.
        makeup_prompt: Makeup text of the CLIP edit direction.
        stylegan_ckpt: FFHQ StyleGAN2 checkpoint; the registered asset if ``None``.
        e4e_ckpt: e4e encoder checkpoint; the registered asset if ``None``.
        clip_ckpt: CLIP ViT-B/32 checkpoint; the registered asset if ``None``.
        detector: Face detector for the FR crops; FRBench RetinaFace if ``None``.
        strict_detection: Raise instead of falling back to the whole image when no face is
            detected.
    """

    name = "DivTrackee"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_pool"),
        optional=("protected_imgs",),
    )
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            steps: int = 60,
            lr: float = 0.01,
            lambda_latent: float = 0.02,
            lambda_clip: float = 0.3,
            lambda_adv: float = 1.0,
            lambda_queue: float = 1.2,
            guide_weight: float = 0.6,
            guide_margin: float = 0.2,
            num_aug: int = 1,
            queue_size: int = 10,
            tune_steps: int = 450,
            tune_lr: float = 5e-4,
            noise_optimize: bool = True,
            clip_loss_type: str = "infonce",
            optimizer_name: str = "adam",
            source_text: str = "face",
            makeup_prompt: str = "natural makeup",
            stylegan_ckpt: Optional[str] = None,
            e4e_ckpt: Optional[str] = None,
            clip_ckpt: Optional[str] = None,
            detector: Optional[Any] = None,
            strict_detection: bool = False,
            ):
        super().__init__(device, fr_models)
        if not self.fr_models:
            raise ValueError(
                "DivTrackee requires a non-empty surrogate FR ensemble "
                "(paper: IR-SE50 + IR-152 + FaceNet); with no models every "
                "embedding-space loss term is silently empty."
            )

        self.steps = steps
        self.lr = lr
        self.lambda_latent = lambda_latent
        self.lambda_clip = lambda_clip
        self.lambda_adv = lambda_adv
        self.lambda_queue = lambda_queue
        self.guide_weight = guide_weight
        self.guide_margin = guide_margin
        self.num_aug = num_aug
        self.queue_size = queue_size
        self.tune_steps = tune_steps
        if optimizer_name not in ("sgd", "adam"):
            raise ValueError(f"optimizer_name must be 'sgd' or 'adam', got {optimizer_name!r}")
        self.optimizer_name = optimizer_name
        self.tune_lr = tune_lr
        self.noise_optimize = noise_optimize
        if clip_loss_type not in ("infonce", "directional"):
            raise ValueError(f"clip_loss_type must be 'infonce' or 'directional', got {clip_loss_type!r}")
        self.clip_loss_type = clip_loss_type
        self.source_text = source_text
        self.makeup_prompt = makeup_prompt
        self.strict_detection = strict_detection

        # Explicit weight paths override the registered assets; get_config records what was used.
        if stylegan_ckpt is None:
            stylegan_ckpt = get_asset("divtrackee/stylegan2-ffhq-config-f.pt")
        if e4e_ckpt is None:
            e4e_ckpt = get_asset("divtrackee/e4e_ffhq_encode.pt")
        if clip_ckpt is None:
            clip_ckpt = get_asset("divtrackee/ViT-B-32.pt")
        self.stylegan_ckpt = str(stylegan_ckpt)
        self.e4e_ckpt = str(e4e_ckpt)
        self.clip_ckpt = str(clip_ckpt)

        # The pretrained generator weights stay on the CPU, so every image is pivot-tuned from
        # the same start regardless of processing order.
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

        # The detector replaces the reference's MTCNN; any object with the
        # frbench ``detect`` API (e.g. an MTCNN adapter) can be injected.
        if detector is None:
            detector = FaceDetector().to(self.device).eval()
        self.detector = detector
        self.detector_name = getattr(detector, "name", type(detector).__name__)

        # Random-perspective views, used only by the InfoNCE CLIP loss.
        self.augment = T.RandomPerspective(fill=0, p=1, distortion_scale=0.5)

        # FIFO diversity queue: one list of per-FR-model embeddings per protected image.
        self._queue = deque(maxlen=queue_size)

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=1024, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def spec_for(self, role: str) -> ImageSpec:
        if role == "eval_imgs":
            return super().spec_for(role)
        # Auxiliary identities are 112x112 aligned crops; requesting 1024 would
        # upsample the whole pool before a single image is drawn.
        if role == "target_pool":
            return ImageSpec(
                size=112,
                channel=ChannelType.RGB,
                crop_type=CropType.FaceBox,
                align_type=AlignType.ArcFace5,
            )
        return self.attack_spec()

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "steps": self.steps,
            "lr": self.lr,
            "lambda_latent": self.lambda_latent,
            "lambda_clip": self.lambda_clip,
            "lambda_adv": self.lambda_adv,
            "lambda_queue": self.lambda_queue,
            "guide_weight": self.guide_weight,
            "guide_margin": self.guide_margin,
            "num_aug": self.num_aug,
            "queue_size": self.queue_size,
            "tune_steps": self.tune_steps,
            "tune_lr": self.tune_lr,
            "noise_optimize": self.noise_optimize,
            "clip_loss_type": self.clip_loss_type,
            "optimizer_name": self.optimizer_name,
            "source_text": self.source_text,
            "makeup_prompt": self.makeup_prompt,
            "stylegan_ckpt": self.stylegan_ckpt,
            "e4e_ckpt": self.e4e_ckpt,
            "clip_ckpt": self.clip_ckpt,
            "detector": self.detector_name,
            "strict_detection": self.strict_detection,
        })
        return config

    def reset_protectee_state(self) -> None:
        """Clear the diversity queue before protecting a different user."""
        self._queue.clear()

    # Public API

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each of one user's faces and paste it back, enqueueing the results for the
        diversity loss.

        Args:
            data.eval_imgs: The user's photos, all of the same person.
            data.target_pool: Clean auxiliary identities, ``(B, 3, 112, 112)`` in ``[0, 255]``;
                one is drawn per photo for the guidance loss.
            data.protected_imgs: Already-protected photos of the same user, enqueued first.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        unprotected_imgs = torch.stack(crops).to(self.device)
        target_imgs = data.target_pool.to(self.device)
        protected_imgs = data.protected_imgs

        if protected_imgs is not None:
            with torch.no_grad():
                for img in protected_imgs.to(self.device):
                    img01 = img.unsqueeze(0) / 255.
                    box = self._detect_box(img01)
                    embs = [e.detach() for e in self._embed(self._crop(img01, box))]
                    if not all(torch.isfinite(e).all() for e in embs):
                        raise RuntimeError("DivTrackee: non-finite embedding while seeding the diversity queue.")
                    self._queue.append(embs)
            logger.info("Seeded the diversity queue with %d protected images.", protected_imgs.shape[0])

        protected = [
            self._protect_single(unprotected_imgs[i:i + 1], target_imgs,
                                 tag=f"{i + 1}/{unprotected_imgs.shape[0]}")
            for i in range(unprotected_imgs.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    # Core stages

    def _protect_single(self, src_img: torch.Tensor, target_imgs: torch.Tensor, tag: str = "") -> torch.Tensor:
        """Run e4e inversion, pivot tuning, and latent optimization on one image."""
        src01 = src_img / 255.

        w_init = self._imgs2latent(src_img)                     # (1, 18, 512)
        generator, noises = self._tune_generator(src01 * 2 - 1, w_init, tag)

        # Detect the face box once on the clean source and reuse it for every
        # crop, as the reference code does (with RetinaFace instead of MTCNN).
        box = self._detect_box(src01)
        with torch.no_grad():
            source_embeddings = self._embed(self._crop(src01, box))
            aux01 = target_imgs[np.random.randint(target_imgs.shape[0])].unsqueeze(0) / 255.
            # As in the reference, the auxiliary image is cropped to its own
            # detected box, so auxiliary and render embeddings share a domain.
            aux_box = self._detect_box(aux01)
            aux_embeddings = self._embed(self._crop(aux01, aux_box))

        # Trainable: the latent and, with noise_optimize, the noise maps below 512 px.
        latent = w_init.detach().clone().requires_grad_(True)
        latent_ref = w_init.detach().clone()
        for noise in noises:
            noise.requires_grad_(self.noise_optimize and noise.shape[2] < 512)
        opt_params = [latent] + ([n for n in noises if n.requires_grad] if self.noise_optimize else [])
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(opt_params, lr=self.lr)
        else:
            optimizer = optim.SGD(opt_params, lr=self.lr)

        # Initial (unprotected) render: reference branch of the CLIP loss.
        with torch.no_grad():
            init_render, _ = generator([latent], input_is_latent=True, noise=noises)
            init_render = ((init_render + 1) / 2).clamp(0, 1).repeat(self.num_aug, 1, 1, 1)

        pbar = tqdm(range(self.steps), desc=f"DivTrackee optimizing {tag}")
        for _ in pbar:
            optimizer.zero_grad()

            render, _ = generator([latent], input_is_latent=True, noise=noises)
            render01 = ((render + 1) / 2).clamp(0, 1)           # (1, 3, 1024, 1024)

            # The released InfoNCE loss sees ``num_aug`` random-perspective views;
            # the directional loss (Eq. 13) sees the raw render.
            if self.clip_loss_type == "infonce":
                clip_views = torch.cat([self.augment(render01) for _ in range(self.num_aug)], dim=0)
                clip_loss = self.nce_loss(init_render, self.source_text,
                                          clip_views, self.makeup_prompt).sum()
            else:
                clip_loss = self.nce_loss.directional(
                    init_render[:1], self.source_text, render01, self.makeup_prompt)
            latent_loss = ((latent_ref - latent) ** 2).sum()

            render_embeddings = self._embed(self._crop(render01, box))
            # L_adv: move away from the user's own identity.
            dodge_loss = sum(1 - F.cosine_similarity(g, s).mean()
                             for g, s in zip(render_embeddings, source_embeddings))
            # L_guide: move toward the random auxiliary identity, up to the margin.
            guide_loss = sum((1 - F.cosine_similarity(g, a).mean() - self.guide_margin).clamp(min=0)
                             for g, a in zip(render_embeddings, aux_embeddings))
            adv_loss = self.guide_weight * guide_loss - dodge_loss
            queue_loss = self._queue_loss(render_embeddings)

            loss = (self.lambda_latent * latent_loss + self.lambda_adv * adv_loss
                    + self.lambda_clip * clip_loss + self.lambda_queue * queue_loss)
            loss.backward()

            # Freeze the coarse style layers: only fine layers (8-17) are edited.
            latent.grad[0, :8] = 0
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # Paper Eq. 7 / Sec. 5.2: return G(w_S) after S updates, and enqueue that
        # final embedding (not the pre-step render from the last iteration).
        with torch.no_grad():
            final_render, _ = generator([latent], input_is_latent=True, noise=noises)
            final01 = ((final_render + 1) / 2).clamp(0, 1)
            if not torch.isfinite(final01).all():
                raise RuntimeError("DivTrackee: non-finite protected render; refusing to enqueue/return.")
            final_embeddings = self._embed(self._crop(final01, box))
            if not all(torch.isfinite(e).all() for e in final_embeddings):
                raise RuntimeError("DivTrackee: non-finite final embedding; refusing to enqueue.")
        self._queue.append([e.detach() for e in final_embeddings])
        out = final01.detach() * 255.
        if not torch.isfinite(out).all():
            raise RuntimeError("DivTrackee: non-finite protected output.")
        return out

    def _tune_generator(self, real_img: torch.Tensor, w: torch.Tensor, tag: str = ""):
        """Pivot tuning: fit the generator to reconstruct ``real_img`` at ``w``.

        Args:
            real_img: ``(1, 3, 1024, 1024)`` in ``[-1, 1]``.
            w: ``(1, 18, 512)`` latent, held fixed.

        Returns:
            The tuned generator (eval mode, gradients disabled) and its fixed
            per-layer noise tensors (17 tensors, 4x4 ... 1024x1024).
        """
        # Restart from the pristine FFHQ weights for every image (per-image PTI).
        self.generator.load_state_dict(self._pristine_g_state, strict=False)
        self.generator.requires_grad_(True)
        optimizer = optim.Adam(self.generator.parameters(), lr=self.tune_lr)

        # Fresh random noises per image, fixed during tuning (and reused in stage 3).
        noises = [n.repeat(1, 1, 1, 1).normal_() for n in self.generator.make_noise()]
        for noise in noises:
            noise.requires_grad = False

        w = w.detach()
        real_256 = F.interpolate(real_img, size=256)
        for _ in tqdm(range(self.tune_steps), desc=f"DivTrackee tuning {tag}"):
            render, _ = self.generator([w], input_is_latent=True, noise=noises)
            render_256 = F.interpolate(render, size=256)
            loss = self.lpips_loss(render_256, real_256).sum() + 0.5 * F.mse_loss(render_256, real_256)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.generator.eval().requires_grad_(False)
        return self.generator, noises

    def _imgs2latent(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode ``[0, 255]`` RGB faces to W+ latents ``(B, 18, 512)``; the 256 px resize is
        bilinear without antialiasing, as in the reference code.
        """
        x = F.interpolate(imgs / 255., size=(256, 256), mode="bilinear", align_corners=False, antialias=False)
        with torch.no_grad():
            codes = self.encoder(x * 2 - 1)
        return codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

    # Helpers

    def _embed(self, imgs01: torch.Tensor) -> List[torch.Tensor]:
        """Embed ``[0, 1]`` RGB face crops with every FR model, resized without antialiasing as in
        the reference ``utils.get_embeddings``.
        """
        return [fr.embed_crops(imgs01, input_space="01", antialias=False,
                               tta=(), l2_normalize=False)
                for fr in self.fr_models]

    def _detect_box(self, img01: torch.Tensor) -> Tuple[int, int, int, int]:
        """The main face box ``(x1, y1, x2, y2)`` of a ``[0, 1]`` image; the whole image if no face
        is found, unless ``strict_detection``.

        Raises:
            RuntimeError: With ``strict_detection`` and no detected face.
        """
        h, w = img01.shape[-2:]
        det = self.detector.detect(img01 * 255.).detections[0]
        if det is None or det.shape[0] == 0:
            if self.strict_detection:
                raise RuntimeError(
                    "DivTrackee: no face detected and strict_detection=True; "
                    "prevalidate the image or disable strict_detection."
                )
            logger.warning("No face detected; using the full image as the crop box.")
            return 0, 0, w, h
        areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) * det[:, 5]
        x1, y1, x2, y2 = det[areas.argmax()][:4].round().long().tolist()
        return max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)

    def _crop(self, img01: torch.Tensor, box: Tuple[int, int, int, int]) -> torch.Tensor:
        """Slice the (differentiable) face-box crop out of a ``(1, 3, H, W)`` image."""
        x1, y1, x2, y2 = box
        return img01[:, :, y1:y2, x1:x2]

    def _queue_loss(self, render_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Diversity loss (paper Eq. 11): minus the mean distance to the queued images over the FR
        models; zero on an empty queue.
        """
        if len(self._queue) == 0:
            return torch.zeros((), device=self.device)
        total = sum(
            sum(1 - F.cosine_similarity(g, q).mean() for g, q in zip(render_embeddings, entry))
            for entry in self._queue)
        return -total / len(self._queue)
