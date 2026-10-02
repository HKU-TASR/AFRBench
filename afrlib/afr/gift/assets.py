"""GIFT weights: the FFHQ-1024 StyleGAN2, the base-code encoder, and the BiSeNet face parser."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, Url, register_assets

register_assets(
    Asset(
        key="gift/stylegan2_ffhq1024.pth",
        kind=AssetKind.PRIOR,
        # Byte-identical to falco/stylegan2_ffhq1024.pth (GenForce port).
        sha256="d68c291b390869bee4f83f144d0480bd4c4cb7aab0fee0a9ee551073ea2d2163",
        size=381_464_183,
        sources=(
            Url("https://www.dropbox.com/scl/fi/yec5wgg8j388fc0saigbj/genforce.tar"
                "?rlkey=kkhkkxvnc985746ichtdyct3f&dl=1",
                member="genforce/stylegan2_ffhq1024.pth"),
        ),
        description="StyleGAN2 generator (FFHQ, 1024x1024, GenForce port)",
        allow_symlink=True,
    ),
    Asset(
        key="gift/basecode_encoder_ffhq1024_16.pth",
        kind=AssetKind.PRIOR,
        sha256="c557f3421839c4fcce4263c8cf11eeb4e2228489ec1eb796791e8cd84b145206",
        size=148_875_947,
        sources=(
            # BDInvert's pretrained base-code encoder for the FFHQ-1024 16x16 latent space,
            # linked from its README; the weights are under the "encoder" key.
            GDrive("1klLuhOyJthcnG6X_l6_lhQXG3Y-deiii"),
        ),
        description="BDInvert base-code encoder E_b (StyleGAN2 FFHQ-1024, 16x16 F-space)",
        allow_symlink=True,
    ),
    Asset(
        key="gift/faceparsing_bisenet.pth",
        kind=AssetKind.PRIOR,
        # Byte-identical to amt_gan/face_parser.pth.
        sha256="468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",
        size=53_289_463,
        sources=(
            # zllrunning/face-parsing.PyTorch BiSeNet (CelebAMask-HQ, 19 classes),
            # the "79999_iter.pth" checkpoint; a copy also ships with AMT-GAN.
            GDrive("154JgKpzCPW82qINcVieuPH3fZ2e0P812"),
        ),
        description="BiSeNet face parser / semantic encoder E_s (Key Landmark Regularization)",
        allow_symlink=True,
    ),
)
