"""Additive paste-back is L∞-nonexpansive within numerical tolerance."""
from __future__ import annotations

import torch

from afrlib.utils.face_crop import CropPlan, _resize_delta, _resize_image, paste_delta


def _plan(natural: int, native: int, photo: int) -> CropPlan:
    return CropPlan(
        matrix=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        natural_size=natural,
        native_size=native,
        photo_size=(photo, photo),
    )


def test_resize_delta_is_linf_nonexpansive():
    cases = []
    impulse = torch.zeros(3, 17, 17)
    impulse[:, 8, 8] = 16.0
    cases.append(impulse)
    checker = torch.ones(3, 15, 15)
    checker[:, ::2, ::2] = -8.0
    cases.append(checker)
    cases.append(torch.randn(3, 9, 13).clamp(-12, 12))
    for delta in cases:
        budget = float(delta.abs().max())
        for size in (8, 16, 32, 33):
            resized = _resize_delta(delta, size)
            assert float(resized.abs().max()) <= budget + 1e-3


def test_resize_image_stays_bicubic_and_delta_is_bilinear():
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).repeat(3, 1, 1)
    assert not torch.equal(_resize_image(image, 8), _resize_delta(image, 8))


def test_paste_delta_without_plan_and_with_affine():
    photo = torch.zeros(3, 32, 32)
    delta = torch.full((3, 32, 32), 7.5)
    out = paste_delta(photo, delta, None)
    assert float((out - photo).abs().max()) <= 7.5 + 1e-3
    crop = torch.zeros(3, 16, 16)
    crop[:, 8, 8] = -11.0
    protected = paste_delta(photo, crop, _plan(32, 16, 32))
    assert float((protected - photo).abs().max()) <= 11.0 + 0.25
