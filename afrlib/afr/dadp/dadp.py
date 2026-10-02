"""DADP: dual-latent adaptive diffusion protection on a frozen diffusion autoencoder.

Reference Paper: Transferable Stealthy Adversarial Example Generation via Dual-Latent Adaptive Diffusion for Facial Privacy Protection (IEEE TIFS 2025)
Reference Code: https://github.com/LiYuanBoJNU/DADP
"""
from typing import Dict, List, Optional, Any, Sequence

import torch
import torch.nn.functional as F
from torchvision import transforms

from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, tqdm
from afrlib.utils.seed import seeded
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)
from .diffae import DiffAutoencoder

logger = get_logger(__name__)

# DiffAE's LitModel seeds with conf.seed=0; protect() runs seeded with the same default.
_DEFAULT_SEED = 0


def random_transform(img: torch.Tensor) -> torch.Tensor:
    """Random input transformation T (paper Eq. 6): one of pad, random or center crop, rotation
    or flip, resized back to the input size.
    """
    size = img.shape[2]
    choice = transforms.RandomChoice([
        transforms.Pad(padding=10),
        transforms.RandomCrop(size * 0.8),
        transforms.CenterCrop(size * 0.8),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
    ])
    out = choice(img)
    out = F.interpolate(out, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=False)
    return out.contiguous()


class DADP(AFRMethod):
    """Optimizes each face's semantic and diffusion codes of a frozen DiffAE so that a short DDIM
    decode is recognized as the target, then decodes them with a long DDIM pass.

    Args:
        diffae_ckpt: DiffAE FFHQ-256 checkpoint; the registered asset if ``None``.
        t_enc: DDIM steps of the stochastic encoding (T).
        ts: Decode steps inside the optimization (Ts).
        tl: Decode steps of the final face (Tl).
        iters: AdamW iterations (N).
        adv_steps: Trailing decode steps that carry the adversarial loss.
        lam_z: Weight of the semantic-code L1 penalty.
        lam_x: Weight of the diffusion-code L1 penalty.
        lam_adv: Weight of the adversarial loss.
        lam_lr: Weight of the combined L1 regularization.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        init_noise: Standard deviation of the noise added to the initial codes.
        seed: Seed for the attack.
    """

    name = "DADP"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        diffae_ckpt: Optional[str] = None,
        t_enc: int = 100,
        ts: int = 5,
        tl: int = 100,
        iters: int = 50,
        adv_steps: int = 1,
        lam_z: float = 30.0,
        lam_x: float = 20.0,
        lam_adv: float = 1.0,
        lam_lr: float = 1.0,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        init_noise: float = 0.01,
        seed: int = _DEFAULT_SEED,
    ):
        super().__init__(device, fr_models)
        self.diffae_ckpt = diffae_ckpt  # as passed; None resolves from the asset cache below
        self.seed = int(seed)

        # Frozen DiffAE (semantic encoder + conditional DDIM encoder/decoder).
        if diffae_ckpt is None:
            diffae_ckpt = get_asset("dadp/ffhq256_autoenc.ckpt")
        self.diffae = DiffAutoencoder(diffae_ckpt, device)

        # Schedule and optimization hyper-parameters.
        self.t_enc = t_enc            # T: DDIM steps for stochastic encoding
        self.ts = ts                  # Ts: small-step decode length inside the optimization loop
        self.tl = tl                  # Tl: large-step decode for the final protected face
        self.iters = iters            # N: number of AdamW iterations
        self.adv_steps = adv_steps    # trailing decode steps (of ts) that carry the adversarial loss
        self.lam_z = lam_z            # semantic-code regularization weight
        self.lam_x = lam_x            # diffusion-code regularization weight
        self.lam_adv = lam_adv        # adversarial-loss weight
        self.lam_lr = lam_lr          # latent-regularization weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.init_noise = init_noise  # std of the noise that initializes the adversarial codes

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=256, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "diffae_ckpt": self.diffae_ckpt,
            "t_enc": self.t_enc,
            "ts": self.ts,
            "tl": self.tl,
            "iters": self.iters,
            "adv_steps": self.adv_steps,
            "lam_z": self.lam_z,
            "lam_x": self.lam_x,
            "lam_adv": self.lam_adv,
            "lam_lr": self.lam_lr,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "init_noise": self.init_noise,
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
            # Target identity embeddings are constants w.r.t. the optimization.
            with torch.no_grad():
                target_embeddings = self._embed(target_img)  # per model, (1, D)

            protected = [
                self._protect_single(unprotected_imgs[i:i + 1], target_embeddings)
                for i in range(unprotected_imgs.shape[0])
            ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _embed(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """Embed aligned ``[0, 255]`` crops with every FR model; targets and reconstructions take the
        same path, as in the paper.
        """
        return [fr.forward(imgs, need_crop=False, tta=(), l2_normalize=False).embeddings for fr in self.fr_models]

    def _protect_single(
        self,
        src_img: torch.Tensor,
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        # Stage 1: encode the source into its semantic and diffusion codes.
        src_pm1 = src_img / 127.5 - 1.0  # FR range [0, 255] -> DiffAE range [-1, 1]
        with torch.no_grad():
            z_sem = self.diffae.encode(src_pm1)
            x_T = self.diffae.encode_stochastic_all(src_pm1, z_sem, self.t_enc)[-1]

        # Stage 2: dual-latent adaptive optimization of the adversarial codes.
        z_adv = (z_sem + self.init_noise * torch.randn_like(z_sem)).requires_grad_(True)
        x_adv = (x_T + self.init_noise * torch.randn_like(x_T)).requires_grad_(True)
        optimizer = torch.optim.AdamW([x_adv, z_adv], lr=self.lr, weight_decay=self.weight_decay)

        warmup_steps = self.ts - self.adv_steps  # decode steps before the adversarial loss applies
        progress = tqdm(range(self.iters), desc="DADP")
        for _ in progress:
            # Small-step conditional DDIM decode over the last ts reverse timesteps
            # (t = ts-1 down to 0); gradients flow through the whole chain to both codes.
            x_t = x_adv.clone()
            for step in range(self.ts):
                t = torch.tensor([self.ts - 1 - step], device=self.device)
                if step < warmup_steps:
                    # Advance the reverse process without applying the adversarial loss yet.
                    _, x_t = self.diffae.render_step(x_t, z_adv, self.ts, t)
                    continue

                # Paper Eq. 6 / Fig. 2: decode first, then transform the reconstructed
                # face before FR. (Released code transformed x_t before the DDIM step.)
                pred_x0, x_t = self.diffae.render_step(x_t, z_adv, self.ts, t)
                adv_loss = self._adversarial_loss(random_transform(pred_x0), target_embeddings)
                reg_loss = self.lam_x * F.l1_loss(x_T, x_adv) + self.lam_z * F.l1_loss(z_sem, z_adv)
                loss = self.lam_adv * adv_loss + self.lam_lr * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                x_t = x_t.detach()
            progress.set_postfix(adv=f"{adv_loss.item():.4f}")

        # Stage 3: large-step decode of the optimized codes into the protected face.
        with torch.no_grad():
            protected = self.diffae.render_full(x_adv, z_adv, self.tl)  # [0, 1]
        return protected.clamp(0, 1) * 255

    def _adversarial_loss(
        self,
        pred_x0: torch.Tensor,
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Impersonation loss ``1 - cos(F(x), F(target))``, averaged over the FR ensemble."""
        out_embeddings = self._embed((pred_x0.clamp(0, 1) * 255))
        return torch.stack([
            1 - F.cosine_similarity(out_embeddings[k], target_embeddings[k], dim=1).mean()
            for k in range(len(self.fr_models))
        ]).mean()
