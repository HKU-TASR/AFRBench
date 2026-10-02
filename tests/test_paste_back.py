"""Premultiplied paste-back: identity paste has no dark fringe."""
from __future__ import annotations

import torch

from afrlib.utils.face_crop import (
    CropPlan,
    _theta,
    _warp,
    apply_plan,
    paste_back,
)


def _plan(natural: int, native: int, photo: int, offset: float) -> CropPlan:
    """Axis-aligned crop of ``natural`` at ``(offset, offset)`` in the photo."""
    return CropPlan(
        matrix=torch.tensor(
            [[1.0, 0.0, -offset], [0.0, 1.0, -offset]], dtype=torch.float32
        ),
        natural_size=natural,
        native_size=native,
        photo_size=(photo, photo),
    )


def _old_mix(photo: torch.Tensor, protected: torch.Tensor, plan: CropPlan) -> torch.Tensor:
    """Straight-alpha paste used as the baseline; returns ``(image, coverage_mask)``."""
    from afrlib.utils.face_crop import _resize

    crop = _resize(protected, plan.natural_size)
    natural = (plan.natural_size, plan.natural_size)
    theta = _theta(plan.matrix, plan.photo_size, natural, invert=False)
    warped = _warp(crop, theta, plan.photo_size, padding="zeros")
    ones = torch.ones(3, *natural, device=crop.device, dtype=crop.dtype)
    mask = _warp(ones, theta, plan.photo_size, padding="zeros").clamp(0, 1)
    return (photo * (1.0 - mask) + warped * mask).clamp(plan.range_min, plan.range_max), mask


def test_identity_paste_premult_matches_photo_old_mix_darker_on_fringe():
    photo = torch.full((3, 80, 80), 200.0)
    # Non-integer offset so bilinear zeros-warp produces a mid-alpha fringe.
    plan = _plan(natural=40, native=40, photo=80, offset=20.3)
    protected = apply_plan(photo, plan)
    old, mask = _old_mix(photo, protected, plan)
    new = paste_back(photo, protected, plan)

    fringe = (mask[0] > 0.05) & (mask[0] < 0.95)
    interior = mask[0] > 0.99
    assert int(fringe.sum()) > 0
    assert int(interior.sum()) > 0

    # The straight-alpha paste applies coverage twice and darkens the soft edge.
    assert float((photo - old).mean(0)[fringe].mean()) > 1.0
    # Premultiplied src-over matches the photo on the same fringe.
    assert float((new - photo).abs().mean(0)[fringe].max()) < 0.5
    # Interior (full coverage) is unchanged under either mix.
    assert float((new - photo).abs().mean(0)[interior].max()) < 0.5
    assert float((old - photo).abs().mean(0)[interior].max()) < 0.5
