"""Method-registry coverage, the shared additive paste helper, and default epsilon bounds."""
from __future__ import annotations

import pytest

import inspect

from afrlib.afr import AFR_REGISTRY
from afrlib.utils import face_crop


FAMILIES = {
    "additive": ("protego", "opom", "chameleon", "lowkey", "fawkes"),
    "makeup": ("adv_makeup", "amt_gan", "diffam"),
    "diffusion": ("adv_diffusion", "dadp", "diffaim"),
    "accessory": ("accessorize", "advhat"),
    "train_fit": ("advcloak", "apf", "aidpro"),
    "target_pool": ("tip_im", "divtrackee"),
}


def test_registry_has_every_family_and_dpa_is_present():
    for names in FAMILIES.values():
        for name in names:
            assert name in AFR_REGISTRY
    assert "dpa" in AFR_REGISTRY
    assert len(AFR_REGISTRY) == 26


def test_paste_deltas_is_the_common_additive_helper():
    source = inspect.getsource(face_crop.paste_deltas)
    assert "paste_delta" in source
    assert "_resize_delta" in inspect.getsource(face_crop.paste_delta)
    assert "BILINEAR" in inspect.getsource(face_crop._resize_delta)
    assert "BICUBIC" in inspect.getsource(face_crop._resize_image)


def test_protego_opom_chameleon_default_epsilon():
    pytest.importorskip("pytorch3d")
    from afrlib.afr.chameleon.chameleon import Chameleon
    from afrlib.afr.opom.opom import OPOM
    from afrlib.afr.protego.protego import Protego

    for cls in (Protego, OPOM, Chameleon):
        params = inspect.signature(cls.__init__).parameters
        assert params["epsilon"].default == 16 / 255.0
