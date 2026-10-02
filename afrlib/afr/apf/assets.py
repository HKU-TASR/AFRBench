"""The authors' released APF TF1 checkpoint (not converted; APF trains its own networks)."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets

PRETRAINED_FILTER_ASSET_KEY = "apf/model_apf"

register_assets(
    Asset(
        key=PRETRAINED_FILTER_ASSET_KEY,
        kind=AssetKind.RELEASED,
        # Directory hash over the four files in APF_pretrained_model/ (the ckpt
        # data, index and meta files and the checkpoint descriptor).
        sha256="f34b154d14a1479a1bee8d80ea03a0221bed2bb706a80515eaad788e2fe2d5e8",
        size=38_541_667,
        sources=(
            GDrive(
                "187SQXnhhULR-8X0wPAyjleSORRNJISTX",
                member="APF_pretrained_model/",
            ),
        ),
        description=(
            "Authors' released APF filter (TF1 ckpt: U-Net T + E-module), "
            "trained against MobileFaceNet_TF probe + InsightFace ArcFace "
            "server; for original-setting sanity checks only"
        ),
        is_directory=True,
    ),
)
