"""DiffAM weights: the CelebA-HQ DDPM checkpoint and the released makeup references."""

from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets


_ASSETS_ARCHIVE = "1IKiWLv99eUbv3llpj-dOegF3O7FWW29J"
_PAIR_FILES = (
    ("vFG137_mask.png", "be809dd2004c0627ef3d0cb85e2cc75d1cbd2cbf7831e0bc19968b8e0706dea8", 3_482),
    ("vFG137_m.png", "d4a5229deeb771b64c1a1c11145f70aed58d06d3f77a0d8b02f890c719ac3e6c", 92_337),
    ("vFG137_nm.png", "c2a2893d587188ed64117683930bd74817aee772fee0d7907240fe44ec20fdd7", 114_119),
    ("vRX912_mask.png", "22e3d4bb878dfd2f566359f7be2c7e3731283a5db161253ed9d4ed08f8a66579", 3_041),
    ("vRX912_m.png", "804ea2a96d7237b2f2c289d30c57a21424aeb2ec442cf32365295cb9b222f097", 78_431),
    ("vRX912_nm.png", "b37a25a09638e5561ce671b787d34e4a2c7821ddcd1038eea9520307260038ba", 98_716),
    ("XMY-060_mask.png", "c5b769cf12ef139bea3d8b779626285236acdec96bcc9c71781ad61af611cdb7", 2_855),
    ("XMY-060_m.png", "3cfe17d863122527d573de44d7970cbb0b8e042e03c11b527c70800ff0971a54", 89_093),
    ("XMY-060_nm.png", "a60530627a98a509fa62130c9f5d1be2b597190a4d757bf7f786bd1c62704e39", 111_575),
    ("XMY-254_mask.png", "4650b17114de8ac001848cd474735d2fa9453c6cbae90dde2072fa5af43c558e", 2_811),
    ("XMY-254_m.png", "7dced923a95c9ed4841e1cfeebc96c9a16fc29deb7d093671d7dc5c2b88ce0de", 93_484),
    ("XMY-254_nm.png", "9fe02af423ba0f949bc66057892c6a74339a0e5d5b6ec7d23036de7943f8aeb3", 108_426),
    ("XYH-045_mask.png", "074190c2b2dd53ef8443fa3b2334b295b638c396181e48ff3d34fc4352845e21", 2_738),
    ("XYH-045_m.png", "98b6627eb6f177f516c24d6ce349ac29183c71abb74c82fb6a489fc7ed9d2db9", 109_292),
    ("XYH-045_nm.png", "2f8815b59c40082c43b8cfb65d0b5482e75d3ba640eab2eb4b45662473d425b2", 121_821),
)


register_assets(
    Asset(
        key="diffam/celeba_hq.ckpt",
        kind=AssetKind.PRIOR,
        sha256="6cbb630eb856ff3ebe4854fe87af3c21a052765678955999db97ea15b5eeb8a5",
        size=454_777_119,
        sources=(
            GDrive(
                "1Y1bSNCfZoe5T5HhZs3APnEUxwMkQZ_s5",
            ),
        ),
        description="CelebA-HQ DDPM base checkpoint used by DiffAM MT",
    ),
    *(
        Asset(
            key=f"diffam/{name}",
            kind=AssetKind.PRIOR,
            sha256=sha256,
            size=size,
            sources=(
                GDrive(
                    _ASSETS_ARCHIVE,
                    member=f"assets/datasets/pairs/{name}",
                ),
            ),
            description=f"Released DiffAM makeup reference asset {name}",
        )
        for name, sha256, size in _PAIR_FILES
    ),
)
