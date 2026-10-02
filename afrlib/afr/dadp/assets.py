"""DADP weights: the DiffAE FFHQ-256 autoencoder."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets

register_assets(
    Asset(
        key="dadp/ffhq256_autoenc.ckpt",
        kind=AssetKind.PRIOR,
        sha256="9bd2ba9e4c22afde8f18958026a4e8e625ef67d2c3ee76bb3aa72f3e67d3b9ca",
        size=2_722_085_968,
        sources=(
            # The authors publish one bundle of checkpoints and datasets; only
            # the DiffAE FFHQ-256 autoencoder checkpoint is extracted.
            GDrive("12XYcHKrtwODfL0Ii93BdGUHFQmQUoTsa",
                   member="checkpoints/ffhq256_autoenc/last.ckpt"),
        ),
        description="Pretrained DiffAE (FFHQ-256 autoencoder) EMA checkpoint",
    ),
)
