"""Minimal builder for the pretrained FFHQ-256 DiffAE used by DADP.

This replaces DiffAE's sprawling ``TrainConfig`` / ``templates.py`` / ``LitModel``
(``experiment.py``) machinery with only what inference needs: the one autoencoder
config, a DDIM sampler factory, and a thin wrapper exposing the three operations
DADP calls -- semantic encoding, stochastic (diffusion) encoding, and decoding.
The heavy UNet + Gaussian-diffusion internals under ``model/`` and ``diffusion/``
are vendored verbatim.
"""
from typing import List

import torch

from .choices import (GenerativeType, LossType, ModelMeanType, ModelType,
                      ModelVarType)
from .model.unet_autoenc import BeatGANsAutoencConfig
from .diffusion.base import get_named_beta_schedule
from .diffusion.diffusion import SpacedDiffusionBeatGansConfig, space_timesteps

# The DiffAE pretrained diffusion always uses 1000 base steps with a linear schedule.
NUM_TRAIN_TIMESTEPS = 1000


def ffhq256_autoenc_conf() -> BeatGANsAutoencConfig:
    """The exact ``ffhq256_autoenc`` architecture from DiffAE's templates."""
    return BeatGANsAutoencConfig(
        image_size=256,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        num_input_res_blocks=None,
        embed_channels=512,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
        resnet_two_cond=True,
        resnet_cond_channels=None,
        resnet_use_zero_module=True,
        # semantic encoder (half-UNet) settings
        enc_out_channels=512,
        enc_attn_resolutions=None,
        enc_pool="adaptivenonzero",
        enc_num_res_block=2,
        enc_channel_mult=(1, 1, 2, 2, 4, 4, 4),
        enc_grad_checkpoint=False,
        latent_net_conf=None,
    )


def make_ddim_sampler(num_steps: int, fp16: bool = True):
    """Build a DDIM sampler over ``num_steps`` evenly strided from the 1000 base steps."""
    conf = SpacedDiffusionBeatGansConfig(
        gen_type=GenerativeType.ddim,
        betas=get_named_beta_schedule("linear", NUM_TRAIN_TIMESTEPS),
        model_type=ModelType.autoencoder,
        model_mean_type=ModelMeanType.eps,
        model_var_type=ModelVarType.fixed_large,
        loss_type=LossType.mse,
        rescale_timesteps=False,
        fp16=fp16,
        use_timesteps=space_timesteps(NUM_TRAIN_TIMESTEPS, f"ddim{num_steps}"),
    )
    return conf.make_sampler()


class DiffAutoencoder:
    """Inference-only DiffAE: semantic encode, stochastic encode, and decode.

    Wraps the pretrained EMA autoencoder and reproduces the three ``LitModel``
    methods DADP relies on. All images are in ``[-1, 1]``; diffusion codes and
    decoded predictions share the image shape ``(B, 3, 256, 256)`` because DiffAE
    diffuses in pixel space conditioned on the semantic code.
    """

    def __init__(self, ckpt_path: str, device: torch.device, fp16: bool = True):
        self.device = device
        self.conf = ffhq256_autoenc_conf()
        self.ema_model = self.conf.make_model().to(device).eval()
        self._load_ema(ckpt_path)
        self.ema_model.requires_grad_(False)
        self._fp16 = fp16
        self._samplers = {}  # cache: num_steps -> sampler

    def _load_ema(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        prefix = "ema_model."
        ema_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not ema_state:
            raise RuntimeError(
                f"DiffAutoencoder checkpoint {ckpt_path!r} has no '{prefix}*' keys"
            )
        # Exact EMA match is required: a partial load would leave random weights.
        self.ema_model.load_state_dict(ema_state, strict=True)

    def _sampler(self, num_steps: int):
        if num_steps not in self._samplers:
            self._samplers[num_steps] = make_ddim_sampler(num_steps, fp16=self._fp16)
        return self._samplers[num_steps]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Semantic code z_sem = Enc(x); shape ``(B, 512)``."""
        return self.ema_model.encoder.forward(x)

    def encode_stochastic_all(self, x: torch.Tensor, cond: torch.Tensor, num_steps: int) -> List[torch.Tensor]:
        """Conditional DDIM stochastic encoding; returns the trajectory ``[x_1, ..., x_T]``."""
        out = self._sampler(num_steps).ddim_reverse_sample_loop(
            self.ema_model, x, model_kwargs={"cond": cond})
        return out["sample_t"]

    def render_step(self, x_t: torch.Tensor, cond: torch.Tensor, num_steps: int, t: torch.Tensor):
        """One DDIM reverse step; returns ``(pred_x0 in [0, 1], x_{t-1})``."""
        out = self._sampler(num_steps).ddim_sample(
            self.ema_model, x=x_t, t=t, model_kwargs={"cond": cond})
        pred_x0 = (out["pred_xstart"] + 1) / 2
        return pred_x0, out["sample"]

    def render_full(self, x_T: torch.Tensor, cond: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Full conditional DDIM decode from ``x_T``; returns an image in ``[0, 1]``."""
        img = self._sampler(num_steps).sample(self.ema_model, noise=x_T, model_kwargs={"cond": cond})
        return (img + 1) / 2
