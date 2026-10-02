"""AIDPro weights: the frozen FMM generator, the released WI/WD modules, and SCRFD."""
from afrlib.utils.assets import Asset, AssetKind, GDrive, register_assets


_PRETRAINED_ARCHIVE = "1SnGWrNU4KQ2KVedLCTWLJjQG33uZkG1k"
_INSIGHTFACE_ARCHIVE = "1oh4yrp45vZ1YwZUNFWDEKbPwU_PsJipl"

# The released WI/WD checkpoint of the reference test_generation.py, trained against the
# authors' SimSwap IR-SE-100 ArcFace.
OFFICIAL_WI_WD_ASSET = "aidpro/wi_wd_step_60000.pt"


register_assets(
    Asset(
        key="aidpro/90000_net_G.pth",
        kind=AssetKind.PRIOR,
        sha256=(
            "2a94343f0ba1ae0fefcc3d0875268703774e1585b1adc0ea6ff9d71a21a4714b"
        ),
        size=220_261_131,
        sources=(
            GDrive(
                _PRETRAINED_ARCHIVE,
                member="pretrained_models/90000_net_G.pth",
            ),
        ),
        description="Official frozen SimSwap-style AIDPro FMM generator",
    ),
    Asset(
        key=OFFICIAL_WI_WD_ASSET,
        kind=AssetKind.RELEASED,
        sha256=(
            "7b460e550f70c6a9dd234cd171288f71dc8dcf2bba9a05333b78bfbb10c9332e"
        ),
        size=9_831_389,
        sources=(
            GDrive(
                _PRETRAINED_ARCHIVE,
                member=(
                    "pretrained_models/"
                    "LPIPS_L1(0.2)_all_loss_id_5_rec_5_wa_10_step_60000.pt"
                ),
            ),
        ),
        description=(
            "Official AIDPro WI/WD checkpoint "
            "(LPIPS_L1(0.2)_all_loss_id_5_rec_5_wa_10_step_60000.pt); trained "
            "against the authors' SimSwap IR-SE-100 ArcFace surrogate"
        ),
    ),
    Asset(
        key="aidpro/scrfd_10g_bnkps.onnx",
        kind=AssetKind.PRIOR,
        sha256=(
            "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
        ),
        size=16_923_827,
        sources=(
            GDrive(
                _INSIGHTFACE_ARCHIVE,
                member=(
                    "insightface_func/models/antelope/"
                    "scrfd_10g_bnkps.onnx"
                ),
            ),
        ),
        description="Official SCRFD detector used for AIDPro alignment",
    ),
)
