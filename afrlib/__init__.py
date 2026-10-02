"""AFRLib: a library of anti-facial-recognition (AFR) methods, imported lazily."""
from __future__ import annotations

from typing import TYPE_CHECKING
from .utils.log import (
    AFRLibWarning,
    add_file_handler,
    are_progress_bars_enabled,
    disable_progress_bars,
    enable_progress_bars,
    get_logger,
    get_verbosity,
    remove_handler,
    set_verbosity,
    set_verbosity_debug,
    set_verbosity_error,
    set_verbosity_info,
    set_verbosity_warning,
    tqdm,
    warn,
)
from .utils.assets import (
    Asset,
    AssetKind,
    GDrive,
    GitHubRelease,
    HFHub,
    Kaggle,
    Manual,
    Url,
    add_local,
    download,
    get_asset,
    get_cache_dir,
    hash_path,
    list_assets,
    purge,
    register_assets,
    set_cache_dir,
)
from .contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    FitData,
    ImageSpec,
    StageInputs,
    TrainData,
)
from .afr import AFR_REGISTRY

if TYPE_CHECKING:
    from .afr.base import AFRMethod

_FR_NAMES = frozenset({
    "FaceRecognizer",
    "FR_REGISTRY",
    "build_fr",
    "build_fr_ensemble",
    "FREmbedResult",
    "FRDetectResult",
})


def __getattr__(name: str):
    """Lazy access to FR backends and ``from afrlib import Fawkes``-style imports."""
    if name in _FR_NAMES:
        from . import fr
        value = getattr(fr, name)
    else:
        from . import afr
        try:
            value = getattr(afr, name)
        except AttributeError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    globals()[name] = value
    return value


__all__ = [
    # logging
    "AFRLibWarning",
    "get_logger",
    "get_verbosity",
    "set_verbosity",
    "set_verbosity_debug",
    "set_verbosity_info",
    "set_verbosity_warning",
    "set_verbosity_error",
    "add_file_handler",
    "remove_handler",
    "enable_progress_bars",
    "disable_progress_bars",
    "are_progress_bars_enabled",
    "tqdm",
    "warn",
    # assets
    "Asset",
    "AssetKind",
    "Url",
    "GitHubRelease",
    "GDrive",
    "HFHub",
    "Kaggle",
    "Manual",
    "register_assets",
    "list_assets",
    "get_asset",
    "download",
    "add_local",
    "purge",
    "get_cache_dir",
    "set_cache_dir",
    "hash_path",
    # contracts
    "ImageSpec",
    "ChannelType",
    "CropType",
    "AlignType",
    "TrainData",
    "FitData",
    "EvalData",
    "StageInputs",
    # AFR methods
    "AFR_REGISTRY",
    "AFRMethod",
    # FR backends (lazy)
    "FaceRecognizer",
    "FR_REGISTRY",
    "build_fr",
    "build_fr_ensemble",
    "FREmbedResult",
    "FRDetectResult",
]
