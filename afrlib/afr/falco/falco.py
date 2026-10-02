"""FALCO: attribute-preserving face anonymization by StyleGAN2 latent optimization.

Reference Paper: Attribute-preserving Face Dataset Anonymization via Latent Code Optimization (CVPR 2023)
Reference Code: https://github.com/chi0tzp/FALCO
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
from types import SimpleNamespace

import torch
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import clip
from sklearn.neighbors import NearestNeighbors

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
from afrlib.utils.log import get_logger, tqdm
from .models.load_generator import load_generator
from .models.encoders.psp_encoders import Encoder4Editing
from .models.arcface import ArcFace

logger = get_logger(__name__)

# CLIP's image normalization (used by both the CLIP and FaRL pairing features).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_farl_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load FaRL weights into a CLIP ViT trunk, dropping the checkpoint's unused LM-head tensors
    but requiring a strict match on the trunk itself.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"FaRL checkpoint {ckpt_path!r} has no 'state_dict'")
    state = payload["state_dict"]
    model_keys = set(model.state_dict())
    clip_state = {k: v for k, v in state.items() if k in model_keys}
    if not clip_state:
        raise RuntimeError(
            f"FaRL checkpoint {ckpt_path!r} has no keys matching the CLIP ViT-B/16 trunk"
        )
    model.load_state_dict(clip_state, strict=True)


class FALCO(AFRMethod):
    """Re-renders each face from a StyleGAN2 latent paired with a generated fake, optimizing
    ``lambda_id * L_id + lambda_attr * L_attr`` so the identity changes but the attributes stay.

    Args:
        id_margin: Cosine similarity to the original that ``L_id`` drives toward.
        epochs: Adam steps per photo.
        lr: Adam learning rate.
        lr_milestones: Fractions of ``epochs`` at which the learning rate drops.
        lr_gamma: Learning-rate decay at each milestone.
        lambda_id: Weight of the identity loss.
        lambda_attr: Weight of the FaRL attribute loss.
        truncation: Truncation of the sampled fake latents.
        num_fake_samples_ratio: Fakes generated per protected photo.
        min_num_fakes: Fewest fakes to generate, so pairing stays meaningful for small batches.
        nn_feature: Pairing features: ``"farl"``, ``"clip"``, ``"dino"`` or ``"arcface"``.
        nn_metric: Pairing distance metric.
        nn_algorithm: ``NearestNeighbors`` algorithm used for pairing.
        stylegan_ckpt: StyleGAN2 FFHQ checkpoint; the registered asset if ``None``.
        e4e_ckpt: e4e encoder checkpoint; the registered asset if ``None``.
        farl_ckpt: FaRL checkpoint; the registered asset if ``None``.
        arcface_ckpt: IR-SE50 checkpoint for ``nn_feature="arcface"``; the registered asset if
            ``None``.
    """

    name = "FALCO"
    eval_inputs = StageInputs(
        required=("eval_imgs",),
    )
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            id_margin: float = 0.,
            epochs: int = 50,
            lr: float = 0.01,
            lr_milestones: Tuple[float, ...] = (0.75, 0.9),
            lr_gamma: float = 0.8,
            lambda_id: float = 10.0,
            lambda_attr: float = 0.1,
            truncation: float = 0.7,
            num_fake_samples_ratio: float = 2.0,
            min_num_fakes: int = 50,
            nn_feature: str = "farl",
            nn_metric: str = "euclidean",
            nn_algorithm: str = "auto",
            stylegan_ckpt: Optional[str] = None,
            e4e_ckpt: Optional[str] = None,
            farl_ckpt: Optional[str] = None,
            arcface_ckpt: Optional[str] = None,
            ):
        super().__init__(device, fr_models)

        self.id_margin = id_margin
        self.epochs = epochs
        self.lr = lr
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.lambda_id = lambda_id
        self.lambda_attr = lambda_attr
        self.truncation = truncation
        self.num_fake_samples_ratio = num_fake_samples_ratio
        self.min_num_fakes = min_num_fakes
        if nn_feature not in ("farl", "clip", "dino", "arcface"):
            raise ValueError(f"nn_feature must be 'farl', 'clip', 'dino' or 'arcface', got {nn_feature!r}")
        self.nn_feature = nn_feature
        self.nn_metric = nn_metric
        self.nn_algorithm = nn_algorithm
        self.stylegan_ckpt = stylegan_ckpt
        self.e4e_ckpt = e4e_ckpt
        self.farl_ckpt = farl_ckpt
        self.arcface_ckpt = arcface_ckpt

        # Weights are resolved from the AFRLib asset cache unless explicit paths are given.
        if stylegan_ckpt is None:
            stylegan_ckpt = get_asset("falco/stylegan2_ffhq1024.pth")
        if e4e_ckpt is None:
            e4e_ckpt = get_asset("falco/e4e_ffhq_encode.pt")
        if farl_ckpt is None:
            farl_ckpt = get_asset("falco/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")

        # FaRL ViT-B/16 (CLIP trunk + FaRL weights). Always needed: it drives
        # the attribute loss, and by default also the NN-pairing features.
        farl, _ = clip.load("ViT-B/16", device=device, jit=False)
        _load_farl_checkpoint(farl, farl_ckpt)
        self.farl = farl.float().eval().requires_grad_(False)
        # Pairing uses CLIP normalization; the attribute loss uses none, as in
        # the release.
        self._farl_pair_transform = T.Compose([
            T.Resize(224, antialias=True), T.CenterCrop(224), T.Normalize(_CLIP_MEAN, _CLIP_STD)])
        self._attr_transform = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224)])

        # NN-pairing feature extractor (steps 1-3).
        if nn_feature == "farl":
            self._nn_model, self._nn_transform = self.farl, self._farl_pair_transform
        elif nn_feature == "clip":
            clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self._nn_model = clip_model.float().eval().requires_grad_(False)
            self._nn_transform = T.Compose([
                T.Resize(224, antialias=True), T.CenterCrop(224), T.Normalize(_CLIP_MEAN, _CLIP_STD)])
        elif nn_feature == "dino":
            dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
            self._nn_model = dino.to(device).float().eval().requires_grad_(False)
            self._nn_transform = T.Compose([
                T.Resize(224, antialias=True), T.CenterCrop(224),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif nn_feature == "arcface":
            if arcface_ckpt is None:
                arcface_ckpt = get_asset("falco/model_ir_se50.pth")
            self._nn_model = ArcFace(pretrained_path=arcface_ckpt).to(device).float().eval()
            self._nn_transform = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(256)])

        # GenForce FFHQ-1024 StyleGAN2 generator, W+ input.
        self.generator = load_generator(stylegan_ckpt).to(device).eval().requires_grad_(False)

        # e4e FFHQ encoder. Only the encoder weights and latent_avg are taken
        # from the pSp checkpoint, as in the release's invert.py::get_latents.
        e4e = torch.load(e4e_ckpt, map_location="cpu")
        self.encoder = Encoder4Editing(50, "ir_se", SimpleNamespace(stylegan_size=1024))
        self.encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in e4e["state_dict"].items() if k.startswith("encoder.")})
        self.encoder = self.encoder.eval().to(device).requires_grad_(False)
        self.latent_avg = e4e["latent_avg"].to(device)  # (18, 512)

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=1024, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "id_margin": self.id_margin,
            "epochs": self.epochs,
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "lr_gamma": self.lr_gamma,
            "lambda_id": self.lambda_id,
            "lambda_attr": self.lambda_attr,
            "truncation": self.truncation,
            "num_fake_samples_ratio": self.num_fake_samples_ratio,
            "min_num_fakes": self.min_num_fakes,
            "nn_feature": self.nn_feature,
            "nn_metric": self.nn_metric,
            "nn_algorithm": self.nn_algorithm,
            "stylegan_ckpt": self.stylegan_ckpt,
            "e4e_ckpt": self.e4e_ckpt,
            "farl_ckpt": self.farl_ckpt,
            "arcface_ckpt": self.arcface_ckpt,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Anonymize each face against a generated fake and paste it back.

        Args:
            data.eval_imgs: The original photos; the fakes they are paired with are generated
                here, so no target is passed.

        Returns:
            The protected originals, one per input image, at the input frame.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        imgs01 = torch.stack(crops).to(self.device) / 255.
        num_imgs = imgs01.shape[0]

        # Steps 1-3: pair each unprotected image with its nearest fake.
        real_features = self._nn_features(imgs01)
        num_fakes = max(math.ceil(self.num_fake_samples_ratio * num_imgs), self.min_num_fakes)
        fake_wps, fake_features = self._generate_fakes(num_fakes)
        nn_indices = self._nn_pair(fake_features, real_features)

        # Step 4: e4e inversion of the unprotected images.
        w_real = self._imgs2latent(imgs01)                      # (B, 18, 512)

        # Step 5: per-image latent code optimization.
        protected = [
            self._protect_single(imgs01[i:i + 1], w_real[i:i + 1], fake_wps[nn_indices[i]:nn_indices[i] + 1],
                                 tag=f"{i + 1}/{num_imgs}")
            for i in range(num_imgs)
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _nn_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """Embed [0, 1] RGB images with the pairing feature extractor."""
        with torch.no_grad():
            x = self._nn_transform(imgs)
            if self.nn_feature in ("farl", "clip"):
                return self._nn_model.encode_image(x)
            return self._nn_model(x)                            # dino / arcface

    def _generate_fakes(self, num_fakes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample fakes from the GAN and return their W+ codes and pairing features, which are
        taken on ``[0, 1]`` renders so real and fake features share one domain (paper Sec. 3.2).
        """
        zs = torch.randn(num_fakes, self.generator.dim_z, device=self.device)
        wps, features = [], []
        with torch.no_grad():
            for i in tqdm(range(num_fakes), desc="FALCO generating fakes"):
                wp = self.generator.get_w(zs[i:i + 1], truncation=self.truncation)  # (1, 18, 512)
                render = self.generator(wp)                     # (1, 3, 1024, 1024), raw
                wps.append(wp)
                features.append(self._nn_features(((render + 1) / 2).clamp(0, 1)))
        return torch.cat(wps, dim=0), torch.cat(features, dim=0)

    def _nn_pair(self, fake_features: torch.Tensor, real_features: torch.Tensor) -> List[int]:
        """For each real image, the index of its nearest fake (``pair_nn.py``)."""
        nn_model = NearestNeighbors(n_neighbors=1, metric=self.nn_metric, algorithm=self.nn_algorithm)
        nn_model.fit(fake_features.detach().cpu().numpy())
        _, indices = nn_model.kneighbors(real_features.detach().cpu().numpy())
        return [int(indices[i, 0]) for i in range(real_features.shape[0])]

    def _imgs2latent(self, imgs01: torch.Tensor) -> torch.Tensor:
        """Encode [0, 1] RGB faces to W+ latents ``(B, 18, 512)`` (e4e + latent_avg)."""
        x = F.interpolate(imgs01, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        with torch.no_grad():
            codes = self.encoder(x * 2 - 1)
        return codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

    def _protect_single(self, img01: torch.Tensor, w_real: torch.Tensor, w_fake: torch.Tensor,
                        tag: str = "") -> torch.Tensor:
        """Optimize one latent ``[real 0-2 | trainable 3-7, from the paired fake | real 8-17]``, so
        only the five middle layers change.
        """
        head = w_real[:, :3].detach()                           # (1, 3, 512), frozen
        tail = w_real[:, 8:].detach()                           # (1, 10, 512), frozen
        trainable = w_fake[:, 3:8].detach().clone().requires_grad_(True)  # (1, 5, 512)

        optimizer = optim.Adam([trainable], lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(m * self.epochs) for m in self.lr_milestones], gamma=self.lr_gamma)

        # FALCO's IDLoss is detector-free: both sides use the same fixed square
        # crop on the FFHQ-aligned 256 canvas.
        with torch.no_grad():
            source_embeddings = self._embed(self._id_crop(img01))
            source_attr_tokens = self._farl_tokens(self._attr_transform(img01))

        pbar = tqdm(range(self.epochs), desc=f"FALCO optimizing {tag}")
        for _ in pbar:
            optimizer.zero_grad()

            latent = torch.cat([head, trainable, tail], dim=1)  # (1, 18, 512)
            render = self.generator(latent)                     # (1, 3, 1024, 1024), raw

            # ID loss: |cos - m| averaged over the FR ensemble (paper Eq. 2 is
            # a single ArcFace term); render mapped to [0, 1] for embedding.
            render01 = ((render + 1) / 2).clamp(0, 1)
            render_embeddings = self._embed(self._id_crop(render01))
            id_loss = sum(
                (F.cosine_similarity(g, s).mean() - self.id_margin).abs()
                for g, s in zip(render_embeddings, source_embeddings)) / len(self.fr_models)

            # Attribute loss on FaRL tokens; the render enters in [-1, 1], as in
            # the release.
            render_attr_tokens = self._farl_tokens(self._attr_transform(render))
            attr_loss = (source_attr_tokens.float() - render_attr_tokens.float()).abs().mean(dim=(0, 2)).sum()

            loss = self.lambda_id * id_loss + self.lambda_attr * attr_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # Final render, min-max normalized to [0, 255] as the release saves it
        # (lib/aux.py::tensor2image(adaptive=True)).
        with torch.no_grad():
            latent = torch.cat([head, trainable, tail], dim=1)
            render = self.generator(latent)
        render = (render - render.min()) / (render.max() - render.min())
        return render * 255.

    def _farl_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """All 197 FaRL ViT-B/16 token features ``(B, 197, 768)``, taken after the transformer and
        before the projection, as the reference ``extract_visual`` does.
        """
        v = self.farl.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)               # (B, width, grid**2)
        x = x.permute(0, 2, 1)                                  # (B, grid**2, width)
        x = torch.cat(
            [v.class_embedding.to(x.dtype)
             + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                                  # NLD -> LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                                  # LND -> NLD
        return x

    def _embed(self, imgs01: torch.Tensor) -> List[torch.Tensor]:
        """Embed [0, 1] RGB face crops with every FR model, one ``(1, D)`` tensor per model."""
        return [fr.forward(imgs01 * 255., need_crop=False, tta=(), l2_normalize=False).embeddings
                for fr in self.fr_models]

    @staticmethod
    def _id_crop(imgs01: torch.Tensor) -> torch.Tensor:
        """The reference IDLoss geometry: a 256 px resize and a fixed 188x188 slice, with no face
        detector on either side.
        """
        imgs256 = F.interpolate(
            imgs01, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        return imgs256[:, :, 35:223, 32:220].contiguous()
