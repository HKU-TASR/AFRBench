"""Assets AFRBench itself depends on, registered under the ``bench/`` prefix."""
from afrlib import Asset, AssetKind, Url, register_assets


register_assets(
    Asset(
        key="bench/gfpgan_v1.3.pth",
        kind=AssetKind.PRIOR,
        sha256="c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70",
        size=348_632_874,
        sources=(
            Url(
                "https://github.com/TencentARC/GFPGAN/releases/download/"
                "v1.3.0/GFPGANv1.3.pth"
            ),
        ),
        description="GFPGAN v1.3 clean (params_ema); FFHQ-512, no RealESRGAN",
    ),
    Asset(
        key="bench/codeformer.pth",
        kind=AssetKind.PRIOR,
        sha256="1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        size=376_637_898,
        sources=(
            Url(
                "https://github.com/sczhou/CodeFormer/releases/download/"
                "v0.1.0/codeformer.pth"
            ),
        ),
        description="CodeFormer restoration net (params_ema); FFHQ-512",
    ),
    Asset(
        key="bench/diffpure_celeba_hq.ckpt",
        kind=AssetKind.PRIOR,
        sha256="6cbb630eb856ff3ebe4854fe87af3c21a052765678955999db97ea15b5eeb8a5",
        size=454_777_119,
        sources=(
            Url(
                "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/"
                "checkpoints/celeba_hq.ckpt"
            ),
        ),
        description="DiffPure CelebA-HQ DDPM 256 (SDEdit t=500)",
    ),
)
