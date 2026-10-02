"""DivTrackee weights: the FFHQ StyleGAN2 generator, the e4e encoder, and CLIP (shared with CLIP2Protect)."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, Url, register_assets

register_assets(
    Asset(
        key="divtrackee/stylegan2-ffhq-config-f.pt",
        kind=AssetKind.PRIOR,
        sha256="bae494ef77e32a9cd1792a81a3c167692a0e64f6bcd8b06592ff42917e2ed46e",
        size=381_462_551,
        sources=(
            GDrive("1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT"),
        ),
        description="StyleGAN2 generator (FFHQ, 1024x1024, rosinality port)",
        allow_symlink=True,
    ),
    Asset(
        key="divtrackee/e4e_ffhq_encode.pt",
        kind=AssetKind.PRIOR,
        sha256="2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999",
        size=1_201_649_680,
        sources=(
            GDrive("1cUv_reLE6k3604or78EranS7XzuVMWeO"),
        ),
        description="e4e FFHQ encoder (full pSp checkpoint; only the encoder "
                    "weights and latent_avg are used)",
        allow_symlink=True,
    ),
    Asset(
        key="divtrackee/ViT-B-32.pt",
        kind=AssetKind.PRIOR,
        sha256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        size=353_976_522,
        sources=(
            Url("https://openaipublic.azureedge.net/clip/models/"
                "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
        ),
        description="OpenAI CLIP ViT-B/32 weights (text-guided makeup loss)",
        allow_symlink=True,
    ),
)
