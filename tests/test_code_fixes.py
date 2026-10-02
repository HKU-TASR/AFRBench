"""Fixes from the 2026-10-01 review: paste_delta coverage, natural_crop, seeding, AdvCloak loading, Pillow resampling."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from afrlib.utils.face_crop import (
    CropPlan,
    _resize_delta,
    _theta,
    _warp,
    apply_plan,
    natural_crop,
    paste_delta,
)
from afrlib.utils.seed import seeded


def _plan(natural: int, native: int, photo: int, offset: float) -> CropPlan:
    return CropPlan(
        matrix=torch.tensor([[1.0, 0.0, -offset], [0.0, 1.0, -offset]], dtype=torch.float32),
        natural_size=natural,
        native_size=native,
        photo_size=(photo, photo),
    )


def test_paste_delta_adds_the_coverage_weighted_warp_once():
    photo = torch.full((3, 80, 80), 100.0)
    plan = _plan(natural=40, native=40, photo=80, offset=20.3)
    delta = torch.full((3, 40, 40), 10.0)
    out = paste_delta(photo, delta, plan)
    theta = _theta(plan.matrix, plan.photo_size, (40, 40), invert=False)
    warped = _warp(_resize_delta(delta, 40), theta, plan.photo_size, "zeros")
    assert torch.allclose(out, photo + warped)
    coverage = _warp(torch.ones(3, 40, 40), theta, plan.photo_size, "zeros")[0]
    fringe = (coverage > 0.05) & (coverage < 0.95)
    assert int(fringe.sum()) > 0
    # On the fringe the delta is weighted by coverage once, not twice.
    assert torch.allclose((out - photo)[0][fringe], 10.0 * coverage[fringe], atol=1e-4)


def test_natural_crop_then_resize_is_apply_plan():
    photo = torch.rand(3, 90, 90, generator=torch.Generator().manual_seed(1)) * 255
    plan = _plan(natural=48, native=32, photo=90, offset=13.6)
    from afrlib.utils.face_crop import _resize

    crop = natural_crop(photo, plan)
    assert crop.shape == (3, 48, 48)
    expected = _resize(crop, 32).clamp(plan.range_min, plan.range_max)
    assert torch.equal(apply_plan(photo, plan), expected)


def _draws():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_seeded_matches_a_global_reset_and_restores_state():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    outer_before = _draws()
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    with seeded(3407):
        inside = _draws()
    outer_after = _draws()
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407)
    assert inside == _draws()
    assert outer_after == outer_before
    assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == flags


def test_advcloak_loads_the_generator_file_it_is_given(tmp_path):
    from afrlib.afr.advcloak import AdvCloak

    reference = AdvCloak("cpu")
    variant = {k: torch.randn_like(v) if v.is_floating_point() else v for k, v in reference.netG.state_dict().items()}
    torch.save(reference.netG.state_dict(), tmp_path / "netG.pth")
    torch.save(variant, tmp_path / "netG_variant.pth")
    loaded = AdvCloak("cpu", assets_path=tmp_path / "netG_variant.pth")
    for key, value in loaded.netG.state_dict().items():
        assert torch.equal(value, variant[key])
    from_folder = AdvCloak("cpu", assets_path=tmp_path)
    for key, value in from_folder.netG.state_dict().items():
        assert torch.equal(value, reference.netG.state_dict()[key])


def test_explicit_pillow_resampling_is_byte_identical():
    import warnings

    from PIL import Image

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (37, 91, 3), dtype=np.uint8)
    gray = rng.random((29, 41)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old_rgb = np.asarray(Image.fromarray(rgb, mode="RGB").resize((420, 160)))
        old_gray = np.asarray(Image.fromarray(gray, mode="F").resize((64, 48), resample=Image.Resampling.BICUBIC))
    new_rgb = np.asarray(Image.fromarray(rgb).resize((420, 160), resample=Image.Resampling.BICUBIC))
    new_gray = np.asarray(Image.fromarray(gray).resize((64, 48), resample=Image.Resampling.BICUBIC))
    assert np.array_equal(old_rgb, new_rgb)
    assert np.array_equal(old_gray, new_gray)


def test_manual_asset_explains_how_to_register_it(tmp_path, monkeypatch):
    from afrlib.utils import assets

    monkeypatch.delenv("AFRLIB_OFFLINE", raising=False)
    monkeypatch.setattr(assets, "_REGISTRY", dict(assets._REGISTRY))
    assets.set_cache_dir(str(tmp_path))
    try:
        assets.register_assets(assets.Asset(
            key="test/manual.bin", sha256="", size=1,
            sources=(assets.Manual("https://example.org/form", "Accept the license, then download it."),),
        ))
        with pytest.raises(RuntimeError, match=r"python -m afrlib add test/manual.bin <file>"):
            assets.get_asset("test/manual.bin")
        local = tmp_path / "copy.bin"
        local.write_bytes(b"x")
        assets.add_local("test/manual.bin", str(local))
        assert open(assets.get_asset("test/manual.bin"), "rb").read() == b"x"
    finally:
        assets.set_cache_dir(None)
