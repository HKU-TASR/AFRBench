"""FPP-WDP: adversarial latent optimization in Stable Diffusion for face privacy.

Reference Paper: Enhancing Facial Privacy Protection via Weakening Diffusion Purification (CVPR 2025)
Reference Code: https://github.com/parham1998/Facial-Privacy-Protection
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Sequence

import torch
from torch import optim
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler

from frbench import FaceDetector
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

VAE_SCALE = 0.18215

class AttentionControlEdit(object):
    """Accumulates the structure loss: the MSE between the adversarial and reference branches'
    self-attention maps.

    Args:
        num_steps: Reverse steps the schedule covers.
        self_replace_steps: Fraction, or start and end fractions, of the steps that contribute.
    """

    def __init__(
        self,
        num_steps: int,
        self_replace_steps: Union[float, Tuple[float, float]],
    ):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = (
            int(num_steps * self_replace_steps[0]),
            int(num_steps * self_replace_steps[1]),
        )
        self.loss = 0.0
        self.criterion = torch.nn.MSELoss()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)

    def _accumulate_loss(self, attn, is_cross: bool):
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                self.loss += self.criterion(
                    attn[1:], self.replace_self_attention(attn_base, attn_repalce))

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self._accumulate_loss(attn[h // 2:], is_cross)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return attn

class FPP_WDP(AFRMethod):
    """Inverts each face into Stable Diffusion latents, then optimizes one latent toward the
    target under a self-attention structure loss that holds the face's layout.

    Args:
        sd_model: Stable Diffusion model id or path.
        diffusion_steps: DDIM steps of the inversion and sampling.
        start_step: Reverse step whose latent is optimized.
        null_opt_steps: AdamW steps per null-text embedding.
        prot_opt_steps: AdamW steps of the adversarial optimization.
        adv_loss_weight: Weight of the identity loss against the structure loss; the default
            suits ``adv_reduce="mean"``.
        is_obfuscation: Use the reference code's ``cos(out, source) - cos(out, target)`` loss
            instead of the paper's Eq. 5.
        use_face_crop: Embed a fixed face-box crop, approximating the reference code's cropping,
            instead of the whole frame.
        adv_reduce: ``"mean"`` or ``"sum"`` over the surrogates in the identity loss.
    """

    name = "FPP-WDP"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        sd_model: str = "sd2-community/stable-diffusion-2-base",
        diffusion_steps: int = 20,
        start_step: int = 17,
        null_opt_steps: int = 20,
        prot_opt_steps: int = 35,
        adv_loss_weight: float = 0.006,
        is_obfuscation: bool = False,
        use_face_crop: bool = True,
        adv_reduce: str = "mean",
    ):
        super().__init__(device, fr_models)
        self.sd_model = sd_model
        if adv_reduce not in ("mean", "sum"):
            raise ValueError(f"adv_reduce must be 'mean' or 'sum', got {adv_reduce!r}")
        self.use_face_crop = bool(use_face_crop)
        self.adv_reduce = str(adv_reduce)

        # Frozen Stable Diffusion (VAE + text encoder + UNet) with DDIM sampling.
        self.diff_model = StableDiffusionPipeline.from_pretrained(sd_model).to(self.device)
        self.diff_model.scheduler = DDIMScheduler.from_config(self.diff_model.scheduler.config)
        self.diff_model.vae.requires_grad_(False)
        self.diff_model.text_encoder.requires_grad_(False)
        self.diff_model.unet.requires_grad_(False)

        # Faces are detected once on the clean source and target; the source box is reused
        # throughout the optimization, as in the reference code.
        self.detector = FaceDetector().to(self.device).eval() if self.use_face_crop else None

        # Schedule and optimization hyper-parameters.
        self.diffusion_steps = diffusion_steps    # DDIM steps for inversion and sampling (T)
        self.start_step = start_step              # reverse process begins here; edits span T - start_step steps
        self.null_opt_steps = null_opt_steps      # inner steps per timestep when learning the null-text embedding
        self.prot_opt_steps = prot_opt_steps      # adversarial latent optimization steps
        self.adv_loss_weight = adv_loss_weight    # lambda weighting the FR loss against the structure loss
        self.is_obfuscation = is_obfuscation

        self.controller = AttentionControlEdit(num_steps=self.diffusion_steps, self_replace_steps=1.0)
        self.generator = torch.Generator(device=self.device).manual_seed(8888)  # deterministic sampling

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=256, channel=ChannelType.RGB, crop_type=CropType.FFHQ, align_type=AlignType.FFHQQuad)

    # Public API

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "sd_model": self.sd_model,
            "diffusion_steps": self.diffusion_steps,
            "start_step": self.start_step,
            "null_opt_steps": self.null_opt_steps,
            "prot_opt_steps": self.prot_opt_steps,
            "adv_loss_weight": self.adv_loss_weight,
            "is_obfuscation": self.is_obfuscation,
            "use_face_crop": self.use_face_crop,
            "adv_reduce": self.adv_reduce,
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

        source_boxes = self._detect_boxes(unprotected_imgs) if self.use_face_crop else None
        target_boxes = self._detect_boxes(target_img) if self.use_face_crop else None

        # Reference identity embeddings are constants w.r.t. the optimization.
        with torch.no_grad():
            source_embeddings = self._embed(unprotected_imgs, boxes=source_boxes)  # per model, (B, D)
            target_embeddings = self._embed(target_img, boxes=target_boxes)        # per model, (1, D)

        protected = [
            self._protect_single(
                unprotected_imgs[i:i + 1],
                [emb[i:i + 1] for emb in source_embeddings],
                target_embeddings,
                None if source_boxes is None else source_boxes[i],
            )
            for i in range(unprotected_imgs.shape[0])
        ]
        return paste_faces(photos, torch.cat(protected, dim=0), plans)

    def _embed(
        self,
        imgs: torch.Tensor,
        boxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> List[torch.Tensor]:
        """Embed ``[0, 255]`` frames with every FR model; given boxes, each fixed unaligned crop is
        stretched to the model's input, keeping the reference's differentiable crop.
        """
        if boxes is None:
            return [
                fr.forward(imgs, need_crop=False, tta=(), l2_normalize=False).embeddings
                for fr in self.fr_models
            ]

        crops = [self._crop(imgs[i:i + 1], box) for i, box in enumerate(boxes)]
        return [
            torch.cat([
                fr.forward(crop, need_crop=False, tta=(), l2_normalize=False).embeddings
                for crop in crops
            ], dim=0)
            for fr in self.fr_models
        ]

    def _detect_boxes(self, imgs: torch.Tensor) -> List[Optional[Tuple[int, int, int, int]]]:
        if self.detector is None:
            return [None] * imgs.size(0)
        detections = self.detector.detect(imgs).detections
        boxes: List[Optional[Tuple[int, int, int, int]]] = []
        h, w = imgs.shape[-2:]
        for det in detections:
            if det is None or det.shape[0] == 0:
                logger.warning("No face detected; using the full frame for FPP-WDP.")
                boxes.append(None)
                continue
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]) * det[:, 5]
            x1, y1, x2, y2 = det[areas.argmax()][:4].round().long().tolist()
            boxes.append((max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)))
        return boxes

    @staticmethod
    def _crop(
        img: torch.Tensor,
        box: Optional[Tuple[int, int, int, int]],
    ) -> torch.Tensor:
        if box is None:
            return img
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return img
        return img[:, :, y1:y2, x1:x2]

    # Core stages

    def _protect_single(
        self,
        unprotected_img: torch.Tensor,
        source_embeddings: List[torch.Tensor],
        target_embeddings: List[torch.Tensor],
        source_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        # Stage 1: invert the image into a sequence of latents (index 0 = noisiest).
        unprotected_img = unprotected_img / 127.5 - 1.0  # FR range [0, 255] -> VAE range [-1, 1]
        inversion_latents = self._ddim_invert(unprotected_img)[::-1]

        # Stage 2: learn the per-timestep null-text guidance from those latents.
        uncond_embeddings = self._optimize_null_text_embedding(inversion_latents)
        # The UNet runs on a 4-way batch (see _diffusion_step), so replicate the guidance.
        null_guidance = [torch.cat([emb] * 4) for emb in uncond_embeddings]

        # Stage 3: adversarially optimize the latent while preserving structure.
        self._register_attention(self.controller)
        adv_latent = inversion_latents[self.start_step - 1]
        init_latent = adv_latent.detach().clone()  # frozen reference branch for the structure loss
        adv_latent.requires_grad_(True)
        optimizer = optim.AdamW([adv_latent], lr=0.01)
        for _ in tqdm(range(self.prot_opt_steps), desc="Protecting"):
            self.controller.loss = 0.0  # self-attention structure loss, accumulated during sampling
            self.controller.reset()

            latents = self._sample_from(init_latent, adv_latent, null_guidance)
            protected_img = self._decode_to_255(latents)  # adversarial branch, [0, 255]
            out_embeddings = self._embed(protected_img, boxes=[source_box] if self.use_face_crop else None)

            structure_loss = self.controller.loss
            loss = structure_loss + self.adv_loss_weight * self._adversarial_loss(
                out_embeddings, source_embeddings, target_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Render the final protected face with the optimized latent.
        with torch.no_grad():
            self.controller.loss = 0.0
            self.controller.reset()
            latents = self._sample_from(init_latent, adv_latent, null_guidance)
            protected_img = self._decode_to_255(latents)
        self._register_attention(controller=None)  # restore the UNet's plain attention
        return protected_img

    def _adversarial_loss(
        self,
        out_embeddings: List[torch.Tensor],
        source_embeddings: List[torch.Tensor],
        target_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """``1 - cos(out, target)`` over the surrogates (paper Eq. 5), or, with ``is_obfuscation``,
        ``cos(out, source) - cos(out, target)``.
        """
        if self.is_obfuscation:
            return sum(
                (F.cosine_similarity(out_embeddings[k], source_embeddings[k], dim=1)
                 - F.cosine_similarity(out_embeddings[k], target_embeddings[k], dim=1)).sum()
                for k in range(len(self.fr_models))
            )
        k = len(self.fr_models)
        terms = [
            (1.0 - F.cosine_similarity(out_embeddings[i], target_embeddings[i], dim=1)).mean()
            for i in range(k)
        ]
        total = sum(terms)
        return total if self.adv_reduce == "sum" else total / k

    def _sample_from(
        self,
        init_latent: torch.Tensor,
        adv_latent: torch.Tensor,
        null_guidance: List[torch.Tensor],
    ) -> torch.Tensor:
        """Run the reverse DDIM steps on the paired ``[reference, adversarial]`` latents."""
        latents = torch.cat([init_latent, adv_latent])
        for i in range(self.start_step, self.diffusion_steps):
            t = self.diff_model.scheduler.timesteps[i]
            latents = self._diffusion_step(latents, null_guidance[i - self.start_step], t)
        return latents

    def _ddim_invert(self, img: torch.Tensor) -> List[torch.Tensor]:
        """DDIM inversion: map a ``[-1, 1]`` image to its latent trajectory ``[z_0, ..., z_T]``."""
        uncond_embeddings = self._get_null_text_embedding()
        self.diff_model.scheduler.set_timesteps(self.diffusion_steps)
        latent = self._encode_to_latent(img)
        latents = [latent]
        for step in tqdm(range(self.diffusion_steps - 1), desc="DDIM Inversion"):
            t = self.diff_model.scheduler.timesteps[self.diffusion_steps - step - 1]
            noise_pred = self.diff_model.unet(latent, t, encoder_hidden_states=uncond_embeddings)["sample"]
            next_timestep = t + self.diff_model.scheduler.config.num_train_timesteps // self.diff_model.scheduler.num_inference_steps
            if next_timestep <= self.diff_model.scheduler.config.num_train_timesteps:
                alpha_bar_next = self.diff_model.scheduler.alphas_cumprod[next_timestep]
            else:
                alpha_bar_next = torch.tensor(0.0, device=self.device)
            # Predicted clean latent, then re-noise toward the next (higher) timestep.
            reverse_x0 = (1 / torch.sqrt(self.diff_model.scheduler.alphas_cumprod[t])
                          * (latent - noise_pred * torch.sqrt(1 - self.diff_model.scheduler.alphas_cumprod[t])))
            latent = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred
            latents.append(latent)
        return latents

    def _optimize_null_text_embedding(self, inversion_latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """Learn one unconditional embedding per reverse timestep, so that each sampling step lands
        on the corresponding inversion latent.
        """
        optimized_uncond_embeddings = []
        latent = inversion_latents[self.start_step - 1]
        uncond_embeddings = self._get_null_text_embedding()
        uncond_embeddings.requires_grad_(True)
        optimizer = torch.optim.AdamW([uncond_embeddings], lr=0.1)
        criterion = torch.nn.MSELoss()
        for i in tqdm(range(self.start_step, self.diffusion_steps), desc="Optimizing Null Text Embedding"):
            t = self.diff_model.scheduler.timesteps[i]
            for _ in range(self.null_opt_steps):
                out_latent = self._diffusion_step(latent, uncond_embeddings, t, is_null_opt=True)
                optimizer.zero_grad()
                loss = criterion(out_latent, inversion_latents[i])
                loss.backward()
                optimizer.step()
            # Advance the latent with the just-learned embedding and freeze a copy.
            with torch.no_grad():
                latent = self._diffusion_step(latent, uncond_embeddings, t, is_null_opt=True).detach()
                optimized_uncond_embeddings.append(uncond_embeddings.detach().clone())
        uncond_embeddings.requires_grad_(False)
        return optimized_uncond_embeddings

    # Diffusion helpers

    def _diffusion_step(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        t: int,
        is_null_opt: bool = False,
    ) -> torch.Tensor:
        """One DDIM denoising step: a single latent during null-text learning, otherwise the
        ``[reference, adversarial]`` pair, which the attention controller splits.
        """
        if is_null_opt:
            noise_pred = self.diff_model.unet(latent, t, encoder_hidden_states=context)["sample"]
        else:
            noise_pred, _ = self.diff_model.unet(
                torch.cat([latent] * 2, dim=0), t, encoder_hidden_states=context)["sample"].chunk(2)
        return self.diff_model.scheduler.step(noise_pred, t, latent)["prev_sample"]

    def _get_null_text_embedding(self) -> torch.Tensor:
        """Text-encoder embedding of the empty prompt; shape ``(1, 77, 1024)`` for SD2."""
        uncond_input = self.diff_model.tokenizer(
            [""], padding="max_length",
            max_length=self.diff_model.tokenizer.model_max_length, return_tensors="pt")
        return self.diff_model.text_encoder(uncond_input.input_ids.to(self.device))[0]

    def _encode_to_latent(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a ``[-1, 1]`` image into a scaled VAE latent (seeded sample)."""
        with torch.no_grad():
            latents = self.diff_model.vae.encode(img).latent_dist.sample(generator=self.generator)
        return latents.mul(VAE_SCALE)

    def _decode_to_255(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the ``[reference, adversarial]`` latent pair and return the adversarial face as
        RGB in ``[0, 255]``.
        """
        img = self.diff_model.vae.decode(1 / VAE_SCALE * latents)["sample"][1:]
        return (img / 2 + 0.5).clamp(0, 1) * 255

    # Attention hooks

    def _register_attention(self, controller: Optional[AttentionControlEdit]) -> None:
        """Patch every ``Attention.forward`` in the UNet to feed the controller its self-attention
        maps, leaving the outputs unchanged; ``controller=None`` restores them.

        Raises:
            AssertionError: If a controller is given but no ``Attention`` module is found.
        """
        def make_forward(module, place_in_unet, original_forward):
            def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                if controller is not None:
                    is_cross = encoder_hidden_states is not None
                    # Paired protect path: UNet sees batch 4 = cat([ref,adv]*2).
                    if (not is_cross
                            and hidden_states.ndim == 3
                            and hidden_states.shape[0] == 4):
                        q = module.head_to_batch_dim(module.to_q(hidden_states))
                        k = module.head_to_batch_dim(module.to_k(hidden_states))
                        attn = (torch.einsum("b i d, b j d -> b i j", q, k) * module.scale).softmax(dim=-1)
                        if attn.shape[1] <= 32 ** 2:
                            controller(attn, is_cross=False, place_in_unet=place_in_unet)
                        else:
                            # Still advance the layer counter for large maps.
                            controller.cur_att_layer += 1
                            if controller.cur_att_layer == controller.num_att_layers:
                                controller.cur_att_layer = 0
                                controller.cur_step += 1
                    else:
                        # Cross-attn / non-protect calls: advance the controller clock.
                        controller.cur_att_layer += 1
                        if controller.cur_att_layer == controller.num_att_layers:
                            controller.cur_att_layer = 0
                            controller.cur_step += 1
                return original_forward(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            return forward

        def register(module, place_in_unet):
            if module.__class__.__name__ == "Attention":
                if getattr(module, "_fpp_original_forward", None) is None:
                    module._fpp_original_forward = module.forward
                if controller is None:
                    module.forward = module._fpp_original_forward
                else:
                    module.forward = make_forward(module, place_in_unet, module._fpp_original_forward)
                return 1
            count = 0
            if hasattr(module, "children"):
                for child in module.children():
                    count += register(child, place_in_unet)
            return count

        count = 0
        for name, module in self.diff_model.unet.named_children():
            if "down" in name:
                count += register(module, "down")
            elif "up" in name:
                count += register(module, "up")
            elif "mid" in name:
                count += register(module, "mid")
        if controller is not None:
            assert count > 0, (
                f"FPP-WDP expected Diffusers Attention modules to hook; found {count}. "
                "Check the Diffusers version (0.39 uses class name Attention)."
            )
            controller.num_att_layers = count
