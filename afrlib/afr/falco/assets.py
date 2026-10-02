"""FALCO weights: the FFHQ StyleGAN2 generator, the e4e encoder, FaRL, and an IR-SE50 backbone."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, Url, register_assets

register_assets(
    Asset(
        key="falco/stylegan2_ffhq1024.pth",
        kind=AssetKind.PRIOR,
        # Byte-identical to gift/stylegan2_ffhq1024.pth (GenForce port).
        sha256="d68c291b390869bee4f83f144d0480bd4c4cb7aab0fee0a9ee551073ea2d2163",
        size=381_464_183,
        sources=(
            Url("https://www.dropbox.com/scl/fi/yec5wgg8j388fc0saigbj/genforce.tar"
                "?rlkey=kkhkkxvnc985746ichtdyct3f&dl=1",
                member="genforce/stylegan2_ffhq1024.pth"),
            GDrive("1iUJn2jOqs1nl1f9ZS8GkKAaXtiD63hjn"),
        ),
        description="StyleGAN2 generator (FFHQ, 1024x1024, GenForce port)",
        allow_symlink=True,
    ),
    Asset(
        key="falco/e4e_ffhq_encode.pt",
        kind=AssetKind.PRIOR,
        # Byte-identical to clip2protect/ and divtrackee/e4e_ffhq_encode.pt.
        sha256="2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999",
        size=1_201_649_680,
        sources=(
            Url("https://www.dropbox.com/s/1jujsdr6ytzilym/e4e.tar?dl=1",
                member="e4e/e4e_ffhq_encode.pt"),
            # Same file as published by the e4e authors (used by other methods).
            GDrive("1cUv_reLE6k3604or78EranS7XzuVMWeO"),
        ),
        description="e4e FFHQ encoder (full pSp checkpoint; only the encoder "
                    "weights and latent_avg are used)",
        allow_symlink=True,
    ),
    Asset(
        key="falco/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
        kind=AssetKind.PRIOR,
        sha256="cd9ebb262bf9fd37b96013fd90e46ab317173467c5701a2f28c07bcd49df1195",
        size=652_110_304,
        sources=(
            Url("https://www.dropbox.com/s/xxhmvo3q7avlcac/farl.tar?dl=1",
                member="farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth"),
        ),
        description="FaRL ViT-B/16 face representation weights (epoch 64; "
                    "NN pairing features and the attribute-preservation loss)",
    ),
    Asset(
        key="falco/model_ir_se50.pth",
        kind=AssetKind.PRIOR,
        # Byte-identical to the IR-SE50 model among AMT-GAN's released FR models.
        sha256="a035c768259b98ab1ce0e646312f48b9e1e218197a0f80ac6765e88f8b6ddf28",
        size=175_367_323,
        sources=(
            # The release's own arcface.tar link is gone; this is the same IR-SE50 checkpoint
            # from another source, used only for nn_feature="arcface".
            GDrive("1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn"),
        ),
        description="ArcFace IR-SE50 backbone (optional 'arcface' NN-pairing features)",
        allow_symlink=True,
    ),
)
