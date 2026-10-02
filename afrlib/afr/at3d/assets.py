"""AT3D-P weights: the BFM front-patch model and the Deep3DFaceRecon regressor."""

from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets


register_assets(
    Asset(
        key="at3d/BFM_model_front_patch.mat",
        kind=AssetKind.PRIOR,
        sha256="645e7b2f2be89ff63401e5318a618592d825e89952a1bcd6df431748f6892e85",
        size=129_736_392,
        # The AT3D authors' link (README, step 3). Contains Basel Face Model 2009 data,
        # which is under the BFM license.
        sources=(GDrive("10scXpVnxMNNHqsdVThSYVvJP4oa20n7i"),),
        description="BFM09 bases, face topology, and AT3D regional triangle buffers",
    ),
    Asset(
        key="at3d/epoch_20.pth",
        kind=AssetKind.PRIOR,
        sha256="6d17a6b23457b521801baae583cb6a58f7238fe6721fc3d65d76407460e9149b",
        size=288_860_037,
        sources=(GDrive("1BlDBB4dLLrlN3cJhVL4nmrd_g6Jx6uP0"),),
        description="Deep3DFaceRecon ResNet-50 coefficient regressor",
    ),
)
