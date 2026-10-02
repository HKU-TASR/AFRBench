"""Assets for Protego: the SMIRK checkpoint and the FLAME 2020 head model used for UV mapping."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, Manual, register_assets

register_assets(
    Asset(
        key="protego/SMIRK_em1.pt",
        kind=AssetKind.PRIOR,
        sha256="26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0",
        size=140_495_895,
        sources=(
            # Official SMIRK release checkpoint (encoder + generator); Protego
            # only loads the encoder half.
            GDrive("1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE"),
        ),
        description="Pretrained SMIRK encoder checkpoint used for UV mapping",
    ),
    Asset(
        key="protego/generic_model.pkl",
        kind=AssetKind.PRIOR,
        sha256="efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b",
        size=53_023_716,
        sources=(
            # The FLAME license requires registration, so the model is not redistributed.
            Manual(
                "https://flame.is.tue.mpg.de",
                "Register, download FLAME2020.zip and extract generic_model.pkl.",
            ),
        ),
        description="FLAME 2020 generic head model (manual download, FLAME license)",
    ),
)
