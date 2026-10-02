"""Adv-Inversion: adversarial refinement of GAN-inversion features for impersonation.

Reference Paper: Adv-Inversion: Stealthy Adversarial Attacks via GAN-Inversion for Facial Privacy Protection (IEEE TIFS 2025)
Reference Code: https://github.com/BiiiGerrr/Adv-Inversion
"""

from typing import Dict, List, Optional, Any, Sequence
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from .models.Net import Net
from .models.Encoders import PostProcessModel_First
from .utils.bicubic import BicubicDownSample
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, suppress_output, tqdm
from afrlib.utils.seed import seeded
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)

logger = get_logger(__name__)

# The FFHQ latent average (2.7 KB) ships with the package.
_DEFAULT_LATENT_AVG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_avg.pt")
# The reference code's default seed.
_DEFAULT_SEED = 3407


class _StyleGANOpts:
    """Minimal option bag for the vendored ``Net`` (StyleGAN2 generator wrapper)."""

    def __init__(self, ckpt: str, device: str, size: int = 1024, latent: int = 512,
                 n_mlp: int = 8, channel_multiplier: int = 2):
        self.ckpt = ckpt
        self.device = device
        self.size = size
        self.latent = latent
        self.n_mlp = n_mlp
        self.channel_multiplier = channel_multiplier


class Adv_Inversion(AFRMethod):
    """Inverts each face into StyleGAN2 codes and refines its feature code toward the target
    (the identity-free variant).

    Args:
        stylegan_ckpt: StyleGAN2 FFHQ checkpoint; the registered asset if ``None``.
        fse_ckpt: Enhanced Feature-Style Encoder checkpoint; the registered asset if ``None``.
        arcface_ckpt: ArcFace IR-50 trunk the encoder is built on; the registered asset if
            ``None``.
        latent_avg_ckpt: FFHQ latent average.
        classifier_scale: Step size of the feature-space ascent (alpha).
        step: Refinement iterations (T).
        beta: Weight of the source-similarity term.
        di: Apply random input diversity before embedding.
        seed: Seed for the attack.
    """

    name = "Adv-Inversion"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=1024, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        stylegan_ckpt: Optional[str] = None,
        fse_ckpt: Optional[str] = None,
        arcface_ckpt: Optional[str] = None,
        latent_avg_ckpt: str = _DEFAULT_LATENT_AVG,
        classifier_scale: float = 60.0,
        step: int = 10,
        beta: float = 0.4,
        di: bool = True,
        seed: int = _DEFAULT_SEED,
    ):
        # The FSE uses AdaptiveAvgPool2d with non-divisible sizes, which MPS
        # rejects (the CPU-fallback env var does not apply), so MPS runs on CPU.
        if device.type == "mps":
            logger.warning("MPS lacks non-divisible adaptive pooling; Adv-Inversion falls back to CPU.")
            device = torch.device("cpu")
        super().__init__(device, fr_models)
        self.stylegan_ckpt = stylegan_ckpt
        self.fse_ckpt = fse_ckpt
        self.arcface_ckpt = arcface_ckpt
        self.latent_avg_ckpt = latent_avg_ckpt
        self.seed = seed

        # Weights come from the asset cache unless paths are given.
        if stylegan_ckpt is None:
            stylegan_ckpt = get_asset("adv_inversion/ffhq.pt")
            # Net looks for "<ckpt>_PCA.npz" next to the checkpoint and rebuilds it slowly if
            # absent; fetching the asset puts it there.
            get_asset("adv_inversion/ffhq_PCA.npz")
        if fse_ckpt is None:
            fse_ckpt = get_asset("adv_inversion/enhanced_fse.pth")
        if arcface_ckpt is None:
            arcface_ckpt = get_asset("adv_inversion/backbone_ir50.pth")

        # Frozen StyleGAN2 generator (FFHQ-1024) wrapped by the vendored ``Net``,
        # whose load-time prints are rerouted to the DEBUG log.
        with suppress_output(logger):
            self.net = Net(_StyleGANOpts(ckpt=stylegan_ckpt, device=str(device)))

        # Pretrained enhanced FSE; the FSE checkpoint overwrites its ArcFace trunk's weights.
        with suppress_output(logger):
            self.fse = PostProcessModel_First(
                arcface_model_path=arcface_ckpt, latent_avg_path=latent_avg_ckpt).to(device).eval()
        state_dict = torch.load(fse_ckpt, map_location="cpu")["model_state_dict"]
        if "module." in next(iter(state_dict)):  # trained with DataParallel
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.fse.load_state_dict(state_dict)
        # ``latent_avg`` is a plain tensor attribute, which ``.to(device)`` does not move.
        self.fse.latent_avg = self.fse.latent_avg.to(device)

        # Preprocessing to feed the FSE: [0, 1] source -> bicubic /4 to 256 -> [-1, 1].
        self.downsample_256 = BicubicDownSample(factor=4)
        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # IFSAR hyper-parameters.
        self.classifier_scale = classifier_scale  # alpha: F-space ascent step size
        self.step = step                          # T: number of refinement iterations
        self.beta = beta                           # beta: weight on the source-similarity term
        self.di = di                               # DIM: random input transform before FR embedding

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "stylegan_ckpt": self.stylegan_ckpt,
            "fse_ckpt": self.fse_ckpt,
            "arcface_ckpt": self.arcface_ckpt,
            "latent_avg_ckpt": self.latent_avg_ckpt,
            "classifier_scale": self.classifier_scale,
            "step": self.step,
            "beta": self.beta,
            "di": self.di,
            "seed": self.seed,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each face toward the target and paste it back; the attack runs seeded with
        ``seed`` and leaves the caller's RNG state unchanged.

        Args:
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
        with seeded(self.seed, self.device):
            # Reference identity embeddings are constants w.r.t. the optimization.
            with torch.no_grad():
                source_embeddings = self._embed(unprotected_imgs)  # per model, (B, D)
                target_embeddings = self._embed(target_img)        # per model, (1, D)

            protected = [
                self._protect_single(
                    unprotected_imgs[i:i + 1],
                    [emb[i:i + 1] for emb in source_embeddings],
                    target_embeddings,
                )
                for i in range(unprotected_imgs.shape[0])
            ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _embed(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """Embed aligned ``[0, 255]`` crops with every FR model, one tensor per model."""
        return [fr.forward(imgs, need_crop=False, tta=(), l2_normalize=False).embeddings for fr in self.fr_models]

    def _protect_single(
        self,
        src_img: torch.Tensor,
        source_embeddings: List[torch.Tensor],
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        # FSE input is the bicubic-downsampled 256 crop normalized to [-1, 1].
        src_256 = self.downsample_256(src_img / 255.0)        # [0, 1], (1, 3, 256, 256)
        src_256 = self.normalize(src_256)                      # [-1, 1]

        # FSE inversion -> W code and feature tensor. PostProcessModel_First ignores
        # its second (target) argument in the IDF path, so the source is passed twice.
        with torch.no_grad():
            w_code, feature = self.fse(src_256, src_256)

        adv_feature = self._ifsar(w_code, feature, source_embeddings, target_embeddings)

        with torch.no_grad():
            adv_img = self._generate(w_code, adv_feature)       # [-1, 1], (1, 3, 1024, 1024)
        return (adv_img + 1) / 2 * 255                          # [0, 255]

    def _generate(self, w_code: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """Decode a ``[-1, 1]`` image from ``w_code``, with ``feature`` injected at layers 5-8."""
        img, _ = self.net.generator(
            [w_code], input_is_latent=True, return_latents=False,
            start_layer=5, end_layer=8, layer_in=feature)
        return img.clip(-1, 1)

    def _ifsar(
        self,
        w_code: torch.Tensor,
        feature: torch.Tensor,
        source_embeddings: List[torch.Tensor],
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Iterative feature-space adversarial refinement: gradient ascent on the feature code of
        the surrogates' mean ``cos(F(x), target) - beta * cos(F(x), source)``.
        """
        adv_feature = feature.clone().detach().requires_grad_(True)
        for _ in tqdm(range(self.step), desc=self.name):
            recon = self._generate(w_code, adv_feature)
            if self.di:
                recon = self._random_transform(recon)

            recon_255 = ((recon + 1) / 2 * 255).clamp(0, 255)  # FR expects [0, 255]
            recon_embeddings = self._embed(recon_255)

            loss = sum(
                (F.cosine_similarity(recon_embeddings[k], target_embeddings[k], dim=-1)
                 - self.beta * F.cosine_similarity(recon_embeddings[k], source_embeddings[k], dim=-1)).sum()
                for k in range(len(self.fr_models))
            ) / len(self.fr_models)

            grad = torch.autograd.grad(loss, adv_feature)[0]
            adv_feature = (adv_feature + self.classifier_scale * grad).detach().requires_grad_(True)
        return adv_feature

    def _random_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Input diversity: each with probability 0.5, a horizontal flip and a 0.9-1.1x rescale
        (zero-padded or center-cropped back to size).
        """
        if torch.rand(1) > 0.5:
            x = torch.flip(x, [3])

        if torch.rand(1) > 0.5:
            scale_factor = torch.rand(1).item() * 0.2 + 0.9
            height, width = x.shape[2], x.shape[3]
            new_size = (int(height * scale_factor), int(width * scale_factor))
            x = F.interpolate(x, size=new_size, mode="bilinear", align_corners=True)

            if scale_factor < 1.0:
                pad_top = (height - new_size[0]) // 2
                pad_bottom = height - new_size[0] - pad_top
                pad_left = (width - new_size[1]) // 2
                pad_right = width - new_size[1] - pad_left
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
            elif scale_factor > 1.0:
                crop_top = (new_size[0] - height) // 2
                crop_bottom = crop_top + height
                crop_left = (new_size[1] - width) // 2
                crop_right = crop_left + width
                x = x[:, :, crop_top:crop_bottom, crop_left:crop_right]
        return x
