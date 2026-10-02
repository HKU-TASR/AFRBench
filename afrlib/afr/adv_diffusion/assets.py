"""Adv-Diffusion weights: Stable Diffusion 2 inpainting and the face parser."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, HFHub, register_assets

register_assets(
    Asset(
        key="adv_diffusion/512-inpainting-ema.ckpt",
        kind=AssetKind.PRIOR,
        sha256="2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3",
        size=5_214_921_607,
        sources=(
            HFHub("sd2-community/stable-diffusion-2-inpainting", "512-inpainting-ema.ckpt"),
        ),
        description="Stable Diffusion 2 inpainting EMA checkpoint (512x512)",
    ),
    Asset(
        key="adv_diffusion/38_G.pth",
        kind=AssetKind.PRIOR,
        sha256="814389542aa1906787d6566c51f638ae7eb89cb701b95caf8353318772c0a3ff",
        size=100_219_165,
        sources=(
            # From TracelessLe/FaceParsing.PyTorch (link in the Adv-Diffusion README).
            GDrive("1neFVTZCWZcCeIoYA7V3i1Kk3DqaK4iei"),
        ),
        description="FaceParseNet50 face-parsing weights (19 classes, CelebAMask-HQ)",
    ),
)
