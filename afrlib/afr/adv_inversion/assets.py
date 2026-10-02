"""Adv-Inversion weights: StyleGAN2 FFHQ and its PCA, the enhanced FSE, and its ArcFace trunk."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets

register_assets(
    Asset(
        key="adv_inversion/ffhq.pt",
        kind=AssetKind.PRIOR,
        sha256="48a146c502117a283437d1c14f755a9416eb9255694e0964403f95b14fc094e9",
        size=132_699_565,
        sources=(
            # Direct file id used by the reference repo's utils/drive.py; the
            # README's StyleGAN+ArcFace bundle carries the same file.
            GDrive("1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS"),
            GDrive("1Ugf4yB9NeLtbKaxrJGPrrLZLH1YxpC_s", member="StyleGAN/ffhq.pt"),
        ),
        description="StyleGAN2 generator (FFHQ, 1024x1024)",
    ),
    Asset(
        key="adv_inversion/ffhq_PCA.npz",
        kind=AssetKind.PRIOR,
        sha256="e2458021339ea5b251f5a4dd561c552eb2b83096723e8585ca79328033958008",
        size=2_108_402,
        sources=(
            GDrive("1Ugf4yB9NeLtbKaxrJGPrrLZLH1YxpC_s", member="StyleGAN/ffhq_PCA.npz"),
        ),
        description="PCA of the StyleGAN2 FFHQ style space (the vendored Net "
                    "expects it next to ffhq.pt and would otherwise rebuild it)",
    ),
    Asset(
        key="adv_inversion/enhanced_fse.pth",
        kind=AssetKind.PRIOR,
        sha256="bbe2228eba7a6fdcc76a2a4b13862439083c2be13e1cfc86e1974c7a1cfa3d83",
        size=866_552_599,
        sources=(
            GDrive("1OfeeV6AVGec6bkkKmw-nzETr9bJeLDPk"),
        ),
        description="Enhanced Feature-Style Encoder weights (author-provided)",
    ),
    Asset(
        key="adv_inversion/backbone_ir50.pth",
        kind=AssetKind.PRIOR,
        sha256="7f187506810d697785b8f0fa3fc22190c003aad3c0782513708efbe1ccd28b35",
        size=174_586_629,
        sources=(
            # Only published inside the README's StyleGAN+ArcFace bundle.
            GDrive("1Ugf4yB9NeLtbKaxrJGPrrLZLH1YxpC_s", member="ArcFace/backbone_ir50.pth"),
        ),
        description="ArcFace IR-50 trunk used to build the enhanced FSE",
    ),
)
