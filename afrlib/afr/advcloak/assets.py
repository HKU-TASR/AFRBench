"""Released AdvCloak checkpoints, trained by the authors against their CASIA IR-50 surrogate."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets

register_assets(
    # Plain state dicts, not run folders; pass one as ``assets_path``.
    Asset(
        key="advcloak/netG.pth",
        kind=AssetKind.RELEASED,
        sha256="be2dedd3cdb307e2a22a1e64965ba01414cf20a5684603c60cc5399a526cb6d2",
        size=29_152_744,
        sources=(GDrive("1AReEmgLOYOfONVFHj0SLNHVrLhU1E0Hb", member="AdvCloak.pth"),),
        description=(
            "Official final AdvCloak generator (plain-cosine variant); trained "
            "against the authors' CASIA IR-50 surrogate; usable via assets_path "
            "without any training"
        ),
    ),
    Asset(
        key="advcloak/netG_convex_hull.pth",
        kind=AssetKind.RELEASED,
        sha256="bdcd0e9e8d97ae4a81b9f398d553ce450f149cd9d6a6e902d66717a2571e4814",
        size=29_152_857,
        sources=(GDrive("1AReEmgLOYOfONVFHj0SLNHVrLhU1E0Hb", member="AdvCloak_ConvexHull.pth"),),
        description=(
            "Official final AdvCloak generator, ConvexHull subspace variant; "
            "trained against the authors' CASIA IR-50 surrogate"
        ),
    ),
    Asset(
        key="advcloak/netG_first_stage.pth",
        kind=AssetKind.RELEASED,
        sha256="6f01203c5a671a7d17016c3b1c12645ad60adf5941b15adb7d0084b2f47bc900",
        size=29_152_857,
        sources=(GDrive("13ri-EIgmoL9AQTTwM5n9yjdusKofbVxG", member="netG_first_stage.pth"),),
        description=(
            "Official stage-1 generator (warm start for stage-2 training); "
            "trained against the authors' CASIA IR-50 surrogate"
        ),
    ),
    Asset(
        key="advcloak/netD_first_stage.pth",
        kind=AssetKind.RELEASED,
        sha256="23c07d413921c81953069a855d31008833b5e8ae43318c07219b8fe6187a36db",
        size=7_511_585,
        sources=(GDrive("13ri-EIgmoL9AQTTwM5n9yjdusKofbVxG", member="netD_first_stage.pth"),),
        description=(
            "Official stage-1 discriminator (warm start for stage-2 training); "
            "trained against the authors' CASIA IR-50 surrogate"
        ),
    ),
)
