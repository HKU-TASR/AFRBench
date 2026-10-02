"""AMT-GAN weights, face parser and makeup-donor images, from the authors' archives."""

from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets

# The authors' assets.zip (FR models, MT-dataset, CelebA-HQ subset, references).
_AMT_GAN_ASSETS_ZIP = "1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8"

_REFERENCE_IMAGES = (
    ("ref01.png", "5698fe636770add67dc402227cb43112ee3ef5bef44a43b248995703a850fe21", 130_897),
    ("ref02.png", "7fe8856753a363ac0f5db6b0223e37c64ccfd24dd1ff2d08f05f40d34eb5e000", 156_088),
    ("ref03.png", "206b6108339405cf6ed2f94a1d87051e9156801705b5df731ee1715874cde455", 201_006),
    ("ref04.png", "ab7d80b22067be84a77ff0fb362f3fa59c404292e45b143dabb1c7fd14a6bad4", 139_540),
    ("ref06.png", "38a391cac322eda8e9187f09c2604876592ecb69ad49f5b8fcb86e08042b3803", 123_056),
    ("ref07.png", "d618cfb15db03b79f9f4609c17f943410335723258b85ed2dbae849ce0afabf4", 137_006),
    ("ref08.png", "edeb4e6df6e5a3870c9add5a6ca03b10e0ee9e19b54c12c5d5f7449b9af82356", 172_194),
    ("ref09.png", "ee8a66281cb559447e85df5d767bc439f379eb9619d5d228c0ac9315fb3abf7e", 191_009),
    ("ref10.png", "eeabf764e5472ee673ebb04bf78f3919188a93c97a664605bf0eb5e6b7ce942b", 179_710),
)

#: Filenames of the released makeup-donor PNGs under ``amt_gan/``.
REFERENCE_NAMES = tuple(name for name, _, _ in _REFERENCE_IMAGES)
DEFAULT_REFERENCE = "ref01.png"


register_assets(
    Asset(
        key="amt_gan/G.pth",
        kind=AssetKind.RELEASED,
        sha256="e732fd8e92a729f91cd4d6ab7f4895a4539de76222a95c66782b7ffb0bdf0b48",
        size=50_482_508,
        sources=(GDrive("1QSsH_s8fKAwhFgSBv5014yGtewlmNJkI", member="G.pth"),),
        description=(
            "Official AMT-GAN generator, baked to impersonation target identity "
            "085807 and trained against the authors' IR-152/IR-SE-50/FaceNet/"
            "MobileFace surrogate ensemble"
        ),
    ),
    Asset(
        key="amt_gan/face_parser.pth",
        kind=AssetKind.PRIOR,
        # Byte-identical to gift/faceparsing_bisenet.pth.
        sha256="468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",
        size=53_289_463,
        sources=(GDrive("154JgKpzCPW82qINcVieuPH3fZ2e0P812"),),
        description="BiSeNet face parser used by the released AMT-GAN",
        allow_symlink=True,
    ),
    Asset(
        key="amt_gan/shape_predictor_68_face_landmarks.dat",
        kind=AssetKind.PRIOR,
        sha256="fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f",
        size=99_693_937,
        sources=(GDrive("1XA2z0j1SlvgdfXkBvw53WrHCsj-mmX-Y"),),
        description="dlib 68-point predictor used by AMT-GAN preprocessing",
        allow_symlink=True,
    ),
    *(
        Asset(
            key=f"amt_gan/{name}",
            kind=AssetKind.PRIOR,
            sha256=sha256,
            size=size,
            sources=(
                GDrive(
                    _AMT_GAN_ASSETS_ZIP,
                    member=f"assets/datasets/reference/{name}",
                ),
            ),
            description=f"Released AMT-GAN reference makeup image {name}",
        )
        for name, sha256, size in _REFERENCE_IMAGES
    ),
)
