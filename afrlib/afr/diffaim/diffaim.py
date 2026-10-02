"""DiffAIM: identity-guided editing in the latent space of Stable Diffusion.

Reference Paper: Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection (ACM MM 2025)
Reference Code: https://github.com/Liqin-Wang/DiffAIM
"""
from typing import Dict, List, Tuple, Any, Optional, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import lpips
from diffusers import StableDiffusionPipeline, DDIMScheduler

from afrlib.afr.base import AFRMethod
from afrlib.utils.face_crop import paste_faces, prepare_inputs
from afrlib.fr import FaceRecognizer
from afrlib.utils.log import get_logger, tqdm
from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)

logger = get_logger(__name__)

VAE_SCALE = 0.18215  # SD latent scaling factor: latent = VAE_SCALE * encoder_output


# Input diversity for transferability: random noise and resize-and-pad, each
# applied with probability 0.5 before the FR ensemble, as in the released code.

def input_noise(x: torch.Tensor, diversity_prob: float = 0.5, noise_scale: float = 0.1) -> torch.Tensor:
    """Randomly add scaled Gaussian noise to ``x``."""
    if torch.rand(1) < diversity_prob:
        strength = torch.rand(1, device=x.device)
        return x + strength * (noise_scale ** 0.5) * torch.randn_like(x)
    return x


def input_diversity(x: torch.Tensor, resize_rate: float = 0.9, diversity_prob: float = 0.5) -> torch.Tensor:
    """Randomly resize ``x`` and zero-pad it back to its original spatial size."""
    if torch.rand(1) >= diversity_prob:
        return x

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    if resize_rate < 1:
        img_size, img_resize = img_resize, img_size

    new_size = torch.randint(low=img_size, high=img_resize + 1, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[new_size, new_size], mode='bilinear', align_corners=False)

    pad_total = img_resize - new_size
    pad_top = torch.randint(low=0, high=pad_total.item() + 1, size=(1,), dtype=torch.int32)
    pad_left = torch.randint(low=0, high=pad_total.item() + 1, size=(1,), dtype=torch.int32)
    return F.pad(rescaled, [pad_left.item(), (pad_total - pad_left).item(),
                            pad_top.item(), (pad_total - pad_top).item()], value=0)


# Structure-preserving attention control

class StructureAttentionController:
    """State shared with the patched attention layers: the structural loss (a squared-L2 sum over
    post-softmax self-attention maps, paper Eq. 11), the mid features and the paired flag.
    """

    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction="sum")
        self.loss = 0.0             # sum of benign-vs-adversarial self-attention ||·||₂² this step
        self.middle_feature = None  # deepest down-block features, shape (2, C, h, w)
        self.paired_active = False  # True only during the paired guidance UNet call

    def reset(self) -> None:
        self.loss = 0.0
        self.middle_feature = None


def _attention_similarity(module, hidden: torch.Tensor, context, is_cross: bool) -> torch.Tensor:
    """Scaled query-key similarity (pre-softmax attention logits) for one branch."""
    q = module.head_to_batch_dim(module.to_q(hidden))
    kv_source = context if is_cross else hidden
    k = module.head_to_batch_dim(module.to_k(kv_source))
    return torch.einsum("b i d, b j d -> b i j", q, k) * module.scale


def register_structure_attention(unet, controller: StructureAttentionController) -> None:
    """Patch every UNet attention module so that, while ``controller.paired_active``, it adds the
    two branches' structural distance to ``controller.loss``, and hook the deepest down-block.
    """

    def make_forward(module, original_forward):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Inversion and singleton reverse steps stay on the stock Diffusers path.
            if not controller.paired_active or hidden_states.shape[0] != 2:
                return original_forward(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )

            context = encoder_hidden_states
            is_cross = context is not None
            half = hidden_states.shape[0] // 2

            # Structural preservation: paper Eq. 11 compares post-softmax
            # self-attention maps of the benign and adversarial branches.
            if not is_cross:
                benign_sim = _attention_similarity(module, hidden_states[:half], context, is_cross)
                adv_sim = _attention_similarity(module, hidden_states[half:], context, is_cross)
                if adv_sim.shape[1] <= 32 ** 2:
                    benign_sim = benign_sim.softmax(dim=-1)
                    adv_sim = adv_sim.softmax(dim=-1)
                    controller.loss += controller.criterion(benign_sim, adv_sim)

            # Diffusers AttnProcessor2_0 sizes cross-attention from the context
            # batch, so expand the singleton null prompt to match the pair.
            if is_cross and context is not None and context.shape[0] == 1:
                context = context.expand(hidden_states.shape[0], *context.shape[1:])

            return original_forward(
                hidden_states,
                encoder_hidden_states=context,
                attention_mask=attention_mask,
                **kwargs,
            )

        return forward

    def patch_attentions(module):
        if module.__class__.__name__ == "Attention":
            # Idempotent: do not wrap an already-wrapped forward.
            if getattr(module, "_diffaim_original_forward", None) is None:
                module._diffaim_original_forward = module.forward
            module.forward = make_forward(module, module._diffaim_original_forward)
        elif hasattr(module, "children"):
            for child in module.children():
                patch_attentions(child)

    def capture_middle_feature(module, inputs, output):
        if controller.paired_active and output[0].shape[0] == 2:
            controller.middle_feature = output[0]

    for name, block in unet.named_children():
        if name in ("down_blocks", "up_blocks", "mid_block"):
            patch_attentions(block)
        if name == "down_blocks":
            block[-1].register_forward_hook(capture_middle_feature)


class DiffAIM(AFRMethod):
    """Inverts each face into Stable Diffusion v1.5 and injects identity guidance toward the
    target during the last ``truncate_step`` reverse steps.

    Args:
        sd_model: Stable Diffusion v1.5 model id or path.
        diffusion_steps: DDIM steps of the inversion and sampling.
        truncate_step: Final reverse steps that receive guidance.
        attack_iters: Guidance iterations per step.
        step_size: Gradient-ascent step size of the guidance.
        eps: L-inf bound of the guidance on the latent.
        eta: DDPM eta; 1.0 is a fully stochastic inversion.
        diversity: Augmented views per identity score.
        id_weight: Weight of the identity term.
        semantic_weight: Weight of the semantic-divergence term.
        structural_weight: Weight of the structural term.
        lpips_weight: Weight of the LPIPS penalty.
        lpips_eps: LPIPS below which there is no penalty.
        source_repulsion: Weight of an optional source-repulsion term; 0 keeps the paper's
            target-only objective.
    """

    name = "DiffAIM"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        diffusion_steps: int = 100,
        truncate_step: int = 20,
        attack_iters: int = 10,
        step_size: float = 3.0,
        eps: float = 0.05,
        eta: float = 1.0,
        diversity: int = 5,
        id_weight: float = 2.0,
        semantic_weight: float = 1.0,
        structural_weight: float = 0.001,
        lpips_weight: float = 8.0,
        lpips_eps: float = 0.01,
        source_repulsion: float = 0.0,
    ):
        super().__init__(device, fr_models)
        self.sd_model = sd_model
        # Optional FR source-repulsion term in the style of the released code;
        # 0 keeps the paper's target-only identity objective.
        self.source_repulsion = float(source_repulsion)

        # Frozen Stable Diffusion v1.5 (VAE + text encoder + UNet) with DDIM sampling.
        self.diff_model = StableDiffusionPipeline.from_pretrained(sd_model).to(self.device)
        self.diff_model.scheduler = DDIMScheduler.from_config(sd_model, subfolder="scheduler")
        self.diff_model.scheduler.set_timesteps(diffusion_steps)
        self.diff_model.unet.sample_size = 32  # 256x256 images -> 32x32 latents

        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval().requires_grad_(False)

        # Schedule and optimization hyper-parameters.
        self.diffusion_steps = diffusion_steps    # DDIM steps T for inversion and sampling
        self.truncate_step = truncate_step        # t_s: guidance is injected only in the last t_s steps
        self.attack_iters = attack_iters          # N_a inner optimization steps per timestep
        self.step_size = step_size                # eta: adversarial gradient-ascent step size
        self.eps = eps                            # kappa: L-inf bound of the injected guidance
        self.eta = eta                            # DDPM/DDIM eta (1.0 = fully stochastic inversion)
        self.diversity = diversity                # number of augmented views for the identity score

        # Maximized: id + semantic - λ_str * struct - λ_lpips * clamp(lpips - eps, 0).
        self.id_weight = id_weight
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.lpips_weight = lpips_weight
        self.lpips_eps = lpips_eps

        # One controller, patched into the UNet once; the paired path is gated so
        # inversion still uses the original Diffusers attention forward.
        self.controller = StructureAttentionController()
        register_structure_attention(self.diff_model.unet, self.controller)

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=256, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "sd_model": self.sd_model,
            "diffusion_steps": self.diffusion_steps,
            "truncate_step": self.truncate_step,
            "attack_iters": self.attack_iters,
            "step_size": self.step_size,
            "eps": self.eps,
            "eta": self.eta,
            "diversity": self.diversity,
            "id_weight": self.id_weight,
            "semantic_weight": self.semantic_weight,
            "structural_weight": self.structural_weight,
            "lpips_weight": self.lpips_weight,
            "lpips_eps": self.lpips_eps,
            "source_repulsion": self.source_repulsion,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Regenerate each face toward the target and paste it back.

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

        # Source FR embeddings are needed only for ``source_repulsion > 0``; paper Eq. 8 uses
        # U-Net semantic divergence instead.
        with torch.no_grad():
            target_embeddings = self._embed(target_img)  # per model, (1, D)
            source_embeddings = (
                self._embed(unprotected_imgs) if self.source_repulsion > 0 else None
            )

        protected = [
            self._protect_single(
                unprotected_imgs[i:i + 1],
                target_embeddings,
                None if source_embeddings is None else [emb[i:i + 1] for emb in source_embeddings],
            )
            for i in range(unprotected_imgs.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _embed(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        """Embed aligned ``[0, 255]`` crops with every FR model, one tensor per model."""
        return [fr.forward(imgs, need_crop=False, tta=(), l2_normalize=False).embeddings for fr in self.fr_models]

    @contextmanager
    def _paired_attention(self):
        """Enable the paired attention path for the enclosed UNet call."""
        self.controller.paired_active = True
        try:
            yield
        finally:
            self.controller.paired_active = False

    def _protect_single(
        self,
        src_img: torch.Tensor,
        target_embeddings: List[torch.Tensor],
        source_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        src_pm1 = src_img / 127.5 - 1.0  # FR range [0, 255] -> VAE range [-1, 1]

        # Stage 1: edit-friendly latent mapping (DDPM inversion) of the source face.
        x0 = self._encode_image(src_pm1)
        _, noise_maps, benign_latents = self._ddpm_inversion(x0)

        # Truncate to the last t_s steps: start from x_{t_s}, keep only that tail of
        # the benign trajectory (x_0..x_{t_s}) and its consistent noise maps (z_1..z_{t_s}).
        adv_latent = benign_latents[self.truncate_step]
        benign_latents = benign_latents[:self.truncate_step + 1]
        noise_maps = noise_maps[:self.truncate_step]

        timesteps = self.diff_model.scheduler.timesteps.to(self.device)
        active_timesteps = timesteps[-self.truncate_step:]           # the last t_s (smallest) timesteps
        t_to_idx = {int(t): k for k, t in enumerate(active_timesteps)}

        adv_latent = adv_latent.unsqueeze(0) if adv_latent.dim() == 3 else adv_latent
        uncond_embedding = self._get_null_text_embedding()

        # Adaptive ensemble weights, initialized uniformly and updated each inner step.
        weights = torch.full((len(self.fr_models),), 1.0 / len(self.fr_models), device=self.device)

        # Stage 2: iterative adversarial guidance injection over the truncated reverse process.
        for t in tqdm(active_timesteps, desc="DiffAIM"):
            step_idx = self.truncate_step - 1 - t_to_idx[int(t)]  # index into the truncated arrays
            benign_latent = benign_latents[step_idx + 1]
            benign_latent = benign_latent.unsqueeze(0) if benign_latent.dim() == 3 else benign_latent

            # Benign reverse step x_{t-1} = mu_t(x_hat_t) + sigma_t z_t (guidance is added afterwards).
            latent_before = adv_latent.detach().clone()
            benign_next = self._reverse_one_step(latent_before, t, noise_maps[step_idx], uncond_embedding)["xtm1"].detach()

            for _ in range(self.attack_iters):
                self.controller.reset()

                adv_candidate = adv_latent.detach().clone().requires_grad_(True)
                # Pair [benign, adversarial] so the controller can compare their attention maps.
                paired = torch.cat([benign_latent, adv_candidate])
                with self._paired_attention():
                    pred_x0 = self._reverse_one_step(paired, t, noise_maps[step_idx], uncond_embedding,
                                                     requires_grad=True)["pred_x0"]
                img = self.diff_model.vae.decode(pred_x0 / VAE_SCALE).sample  # [-1, 1], (1, 3, 256, 256)

                loss, weights = self._total_loss(
                    img, src_pm1, target_embeddings, weights, source_embeddings
                )

                # Gradient-ascent guidance, projected onto the eps-ball around the current latent.
                adv_candidate.grad = torch.autograd.grad(loss, adv_candidate)[0]
                adv_candidate = adv_candidate + adv_candidate.grad * self.step_size
                guidance = (adv_candidate - latent_before).clamp(-self.eps, self.eps)
                adv_latent = (latent_before + guidance).detach()

            # x_hat_{t-1} = x_{t-1} + accumulated guidance G_t.
            adv_latent = benign_next + (adv_latent - latent_before)

        # Stage 3: decode the final adversarial latent to a protected face in [0, 255].
        with torch.no_grad():
            x_adv = self.diff_model.vae.decode(adv_latent.detach() / VAE_SCALE).sample
        return ((x_adv + 1) / 2).clamp(0, 1) * 255

    def _total_loss(
        self,
        img: torch.Tensor,
        src_pm1: torch.Tensor,
        target_embeddings: List[torch.Tensor],
        weights: torch.Tensor,
        source_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The scalar to maximize (paper Eq. 12 plus an LPIPS penalty), and the updated ensemble
        weights.
        """
        # Identity convergence (+ adaptive ensemble): target similarity only (Eqs. 8--10),
        # optionally minus a continuous source-repulsion term.
        id_loss, weights = self._identity_score(
            img, target_embeddings, weights, source_embeddings=source_embeddings
        )

        # Semantic divergence: push the deepest UNet feature of the adversarial branch
        # away from the benign one (higher = more divergent).
        feats = self.controller.middle_feature
        feats = feats.reshape(feats.shape[0], feats.shape[1], -1)  # (2, C, h*w)
        semantic_loss = 1 - F.cosine_similarity(feats[0], feats[1]).mean()

        # Structural preservation: closeness of adversarial vs benign self-attention maps.
        structural_loss = self.controller.loss

        # Release naturalness guard: penalize perceptual drift beyond lpips_eps.
        lpips_loss = torch.clamp(self.lpips(img, src_pm1) - self.lpips_eps, min=0).mean()

        # Maximized: id + semantic - λ_str * structural - λ_lpips * lpips.
        loss = (self.id_weight * id_loss
                + self.semantic_weight * semantic_loss
                - self.structural_weight * structural_loss
                - self.lpips_weight * lpips_loss)
        return loss, weights

    def _identity_score(
        self,
        img: torch.Tensor,
        target_embeddings: List[torch.Tensor],
        weights: torch.Tensor,
        source_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble-weighted similarity to the target over ``diversity`` augmented views, and the
        updated per-model weights (paper Eq. 10).
        """
        views = [input_diversity(input_noise(img)) for _ in range(self.diversity)]
        # Map each view from the VAE range [-1, 1] to the FR input range [0, 255];
        # augmentation noise can overshoot, so clamp.
        views_255 = torch.cat([((v + 1) / 2 * 255).clamp(0, 255) for v in views], dim=0)

        scores = []
        for k, fr in enumerate(self.fr_models):
            feats = fr.forward(views_255, need_crop=False, tta=(), l2_normalize=False).embeddings  # (diversity, D)
            target_sim = F.cosine_similarity(feats, target_embeddings[k])
            score = target_sim.mean()
            if (
                self.source_repulsion > 0
                and source_embeddings is not None
            ):
                source_sim = F.cosine_similarity(feats, source_embeddings[k]).mean()
                score = score - self.source_repulsion * source_sim
            scores.append(score)

        scores = torch.stack(scores)
        weighted_score = torch.sum(scores * weights)
        # Harder-to-attack models (lower score) get more weight next step (Eq. 10).
        new_weights = torch.exp(1 - scores) / torch.sum(torch.exp(1 - scores))
        return weighted_score, new_weights

    def _ddpm_inversion(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Edit-friendly DDPM inversion: the final latent, the consistent noise maps, and the
        forward trajectory, so the reverse process reconstructs the source exactly.
        """
        uncond_embedding = self._get_null_text_embedding()
        timesteps = self.diff_model.scheduler.timesteps.to(self.device)
        num_steps = self.diffusion_steps

        noise_maps = torch.zeros((num_steps, *x0.shape[1:]), device=self.device)
        benign_latents = self._sample_benign_trajectory(x0)

        t_to_idx = {int(t): k for k, t in enumerate(timesteps)}
        x_t = x0
        for t in tqdm(timesteps, desc="DDPM inversion"):
            idx = num_steps - t_to_idx[int(t)] - 1
            x_t = benign_latents[idx + 1][None]  # benign_latents[0] = x0
            noise_pred = self._get_noise_pred(x_t, t, uncond_embedding)
            mu_xt, _, variance = self._posterior_mean(x_t, t, noise_pred)

            # Solve Eq. (4) for the noise map z_t that maps x_t -> x_{t-1}.
            x_tm1 = benign_latents[idx][None]
            z = (x_tm1 - mu_xt) / (self.eta * variance ** 0.5)
            noise_maps[idx] = z
            # Re-noise with the just-computed z so reconstruction is exact (no error accumulation).
            benign_latents[idx] = mu_xt + (self.eta * variance ** 0.5) * z

        noise_maps[0] = torch.zeros_like(noise_maps[0])
        return x_t, noise_maps, benign_latents

    def _sample_benign_trajectory(self, x0: torch.Tensor) -> torch.Tensor:
        """Forward diffusion with independent noise: ``[x_0, x_1, ..., x_T]`` (Eq. 3)."""
        alpha_bar = self.diff_model.scheduler.alphas_cumprod
        # Computed once outside the loop, as in the released code.
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        timesteps = self.diff_model.scheduler.timesteps.to(self.device)
        num_steps = self.diffusion_steps

        t_to_idx = {int(t): k for k, t in enumerate(timesteps)}
        latents = torch.zeros((num_steps + 1, *x0.shape[1:]), device=x0.device)
        latents[0] = x0
        for t in reversed(timesteps):
            idx = num_steps - t_to_idx[int(t)]
            latents[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        return latents

    def _reverse_one_step(self, x_t: torch.Tensor, t, z: torch.Tensor, uncond_embedding: torch.Tensor,
                          requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        """One consistent-noise reverse step ``x_{t-1} = mu_t(x_t) + sigma_t z_t``; of a paired
        batch, only the adversarial half is carried forward.
        """
        noise_pred = self._get_noise_pred(x_t, t, uncond_embedding, requires_grad)
        if noise_pred.shape[0] == 2:
            noise_pred = noise_pred[1][None]
            x_t = x_t[1][None]
        variance_noise = z.unsqueeze(0) if z.dim() == 3 else z
        mu_xt, pred_x0, variance = self._posterior_mean(x_t, t, noise_pred)
        x_tm1 = mu_xt + self.eta * variance ** 0.5 * variance_noise
        return {"pred_x0": pred_x0, "xtm1": x_tm1}

    def _posterior_mean(self, x_t: torch.Tensor, t, noise_pred: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DDPM posterior: returns ``(mu_t, predicted_x0, variance)``."""
        scheduler = self.diff_model.scheduler
        alpha_bar = scheduler.alphas_cumprod
        prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        alpha_bar_prev = alpha_bar[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod

        variance = ((1 - alpha_bar_prev) / (1 - alpha_bar[t])) * (1 - alpha_bar[t] / alpha_bar_prev)
        pred_x0 = (x_t - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5
        direction = (1 - alpha_bar_prev - self.eta * variance) ** 0.5 * noise_pred
        mu_xt = alpha_bar_prev ** 0.5 * pred_x0 + direction
        return mu_xt, pred_x0, variance

    def _get_noise_pred(self, x_t: torch.Tensor, t, uncond_embedding: torch.Tensor,
                        requires_grad: bool = False) -> torch.Tensor:
        """UNet noise prediction under the null (empty) prompt."""
        with torch.set_grad_enabled(requires_grad):
            return self.diff_model.unet(x_t, timestep=t, encoder_hidden_states=uncond_embedding).sample

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a ``[-1, 1]`` image into a scaled VAE latent (deterministic mode)."""
        with torch.no_grad():
            return (self.diff_model.vae.encode(img).latent_dist.mode() * VAE_SCALE).float()

    def _get_null_text_embedding(self) -> torch.Tensor:
        """Text-encoder embedding of the empty prompt; shape ``(1, 77, 768)`` for SD1.5."""
        tokens = self.diff_model.tokenizer(
            [""], padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.diff_model.text_encoder(tokens.input_ids.to(self.device))[0]
