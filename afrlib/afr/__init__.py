"""AFR method packages and the lazy registry, which also resolves ``package.module:Class``."""
from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Dict, Tuple

_AFR_TARGETS: Dict[str, Tuple[str, str]] = {
    "adv_diffusion": ("afrlib.afr.adv_diffusion", "Adv_Diffusion"),
    "dadp": ("afrlib.afr.dadp", "DADP"),
    "diffaim": ("afrlib.afr.diffaim", "DiffAIM"),
    "fpp_wdp": ("afrlib.afr.fpp_wdp", "FPP_WDP"),
    "adv_inversion": ("afrlib.afr.adv_inversion", "Adv_Inversion"),
    "clip2protect": ("afrlib.afr.clip2protect", "Clip2Protect"),
    "divtrackee": ("afrlib.afr.divtrackee", "DivTrackee"),
    "falco": ("afrlib.afr.falco", "FALCO"),
    "fawkes": ("afrlib.afr.fawkes", "Fawkes"),
    "gift": ("afrlib.afr.gift", "GIFT"),
    "protego": ("afrlib.afr.protego", "Protego"),
    "opom": ("afrlib.afr.opom", "OPOM"),
    "chameleon": ("afrlib.afr.chameleon", "Chameleon"),
    "tip_im": ("afrlib.afr.tip_im", "TIP_IM"),
    "lowkey": ("afrlib.afr.lowkey", "LowKey"),
    "at3d": ("afrlib.afr.at3d", "AT3D"),
    "efficient_decision": ("afrlib.afr.efficient_decision", "EfficientDecision"),
    "accessorize": ("afrlib.afr.accessorize", "Accessorize"),
    "adv_makeup": ("afrlib.afr.adv_makeup", "AdvMakeup"),
    "advhat": ("afrlib.afr.advhat", "AdvHat"),
    "aidpro": ("afrlib.afr.aidpro", "AIDPro"),
    "amt_gan": ("afrlib.afr.amt_gan", "AMT_GAN"),
    "diffam": ("afrlib.afr.diffam", "DiffAM"),
    "apf": ("afrlib.afr.apf", "APF"),
    "dpa": ("afrlib.afr.dpa", "DPA"),
    "advcloak": ("afrlib.afr.advcloak", "AdvCloak"),
}


class _LazyAFRRegistry(Mapping[str, type]):
    """Read-only method registry that imports implementations on first use."""

    def __init__(self, targets: Dict[str, Tuple[str, str]]):
        self._targets = targets
        self._loaded: Dict[str, type] = {}

    def __getitem__(self, key: str) -> type:
        if key not in self._loaded:
            if key in self._targets:
                module_name, class_name = self._targets[key]
                self._loaded[key] = getattr(import_module(module_name), class_name)
            elif ":" in key:
                self._loaded[key] = _import_plugin(key)
            else:
                raise KeyError(key)
        return self._loaded[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def __len__(self) -> int:
        return len(self._targets)


def _import_plugin(path: str) -> type:
    """Import an out-of-tree method given as ``"package.module:Class"``."""
    from .base import AFRMethod

    module_name, class_name = path.split(":", 1)
    cls = getattr(import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, AFRMethod)):
        raise TypeError(f"{path} is not an AFRMethod subclass")
    return cls


AFR_REGISTRY: Mapping[str, type] = _LazyAFRRegistry(_AFR_TARGETS)
_CLASS_TO_KEY = {class_name: key for key, (_, class_name) in _AFR_TARGETS.items()}


def __getattr__(name: str):
    """Lazy ``from afrlib.afr import Fawkes``/``AFRMethod``-style imports."""
    if name == "AFRMethod":
        from .base import AFRMethod

        globals()[name] = AFRMethod
        return AFRMethod
    try:
        key = _CLASS_TO_KEY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = AFR_REGISTRY[key]
    globals()[name] = value
    return value


__all__ = ["AFR_REGISTRY", "AFRMethod"]
