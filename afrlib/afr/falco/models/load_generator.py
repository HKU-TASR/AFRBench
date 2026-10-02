"""Builds the GenForce StyleGAN2 generator with FALCO's pre-trained weights."""
import torch

from .genforce.models import StyleGAN2Generator


def load_generator(checkpoint_path: str, resolution: int = 1024) -> StyleGAN2Generator:
    """Build a W+-input StyleGAN2 generator and load a GenForce checkpoint.

    Mirrors the reference ``models/load_generator.py`` for the default
    ``stylegan2_ffhq1024`` model (``latent_is_w=True``, as used by
    ``anonymize.py``), without the wget-on-miss branch: the checkpoint is
    resolved by the caller (AFRLib asset cache).
    """
    generator = StyleGAN2Generator(resolution=resolution, latent_is_w=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "generator_smooth" in checkpoint:
        generator.load_state_dict(checkpoint["generator_smooth"])
    else:
        generator.load_state_dict(checkpoint["generator"])
    generator.dim_z = generator.z_space_dim
    return generator
