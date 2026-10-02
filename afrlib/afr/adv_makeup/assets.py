"""Adv-Makeup priors, the released LADN training-subset manifest, and its sample pairs."""

from afrlib.utils.assets import Asset, AssetKind, Url, register_assets


_ADV_MAKEUP_REVISION = "359af7e19ad4efc69c2775ea00ce2bdff69cc491"
_ADV_MAKEUP_RAW = (
    "https://raw.githubusercontent.com/TencentYoutuResearch/Adv-Makeup/"
    f"{_ADV_MAKEUP_REVISION}"
)


register_assets(
    Asset(
        key="adv_makeup/vgg16-397923af.pth",
        kind=AssetKind.PRIOR,
        sha256="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
        size=553_433_881,
        sources=(
            Url("https://download.pytorch.org/models/vgg16-397923af.pth"),
        ),
        description="VGG16 ImageNet weights used by Adv-Makeup style loss",
    ),
    Asset(
        key="adv_makeup/2d106det.onnx",
        kind=AssetKind.PRIOR,
        sha256="f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
        size=5_030_888,
        sources=(
            Url(
                "https://github.com/deepinsight/insightface/releases/download/"
                "v0.7/buffalo_l.zip",
                member="2d106det.onnx",
            ),
        ),
        description="InsightFace 106-point landmark regressor",
    ),
    Asset(
        key="adv_makeup/landmark_aligned_600.pk",
        kind=AssetKind.PRIOR,
        sha256="c090f9715957cda06fad25ad72cdc98e0b8fd39d7fb940e4c59197d7a595f457",
        size=950_124,
        sources=(
            Url(
                f"{_ADV_MAKEUP_RAW}/Datasets_Makeup/"
                "landmark_aligned_600.pk"
            ),
        ),
        description=(
            "Released Adv-Makeup 600px alignment manifest selecting 332 "
            "training-source and 198 makeup LADN faces"
        ),
    ),
    Asset(
        key="adv_makeup/Datasets_Makeup",
        kind=AssetKind.PRIOR,
        sha256="d179155e745ececc2c9961c39b5e7df709766dfcd3d2fde02953cc31577e31f6",
        size=1_254_569,
        sources=(
            Url(
                "https://github.com/TencentYoutuResearch/Adv-Makeup/"
                f"archive/{_ADV_MAKEUP_REVISION}.zip",
                member=(
                    f"Adv-Makeup-{_ADV_MAKEUP_REVISION}/"
                    "Datasets_Makeup/"
                ),
            ),
        ),
        description="Official four-pair Adv-Makeup smoke-test dataset",
        is_directory=True,
    ),
)
