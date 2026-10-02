"""Adv-Diffusion per-image resample after a non-finite decode."""
from __future__ import annotations

import torch

from afrlib.afr.adv_diffusion.adv_diffusion import Adv_Diffusion


def _bare_method() -> Adv_Diffusion:
    method = Adv_Diffusion.__new__(Adv_Diffusion)
    method.device = torch.device("cpu")
    return method


def test_finite_first_pass_does_not_resample(monkeypatch):
    method = _bare_method()
    src = torch.zeros(1, 3, 4, 4)
    hits = {"n": 0}

    monkeypatch.setattr(method, "_prepare_inpainting", lambda _src: ({}, src))

    def _sample(*_args):
        hits["n"] += 1
        return torch.full_like(src, 10.0)

    monkeypatch.setattr(method, "_guided_sample", _sample)
    out = method._run_diffusion(src, [])
    assert hits["n"] == 1
    assert torch.equal(out, torch.full_like(src, 10.0))


def test_resample_recovers_after_nonfinite_decode(monkeypatch):
    method = _bare_method()
    src = torch.zeros(1, 3, 4, 4)
    hits = {"n": 0}

    monkeypatch.setattr(method, "_prepare_inpainting", lambda _src: ({}, src))

    def _sample(*_args):
        hits["n"] += 1
        if hits["n"] == 1:
            return None
        return torch.full_like(src, 32.0)

    monkeypatch.setattr(method, "_guided_sample", _sample)
    out = method._run_diffusion(src, [])
    assert hits["n"] == 2
    assert torch.equal(out, torch.full_like(src, 32.0))


def test_double_nonfinite_decode_returns_source_crop(monkeypatch):
    method = _bare_method()
    src = torch.linspace(-1.0, 1.0, 48).reshape(1, 3, 4, 4)
    hits = {"n": 0}

    monkeypatch.setattr(method, "_prepare_inpainting", lambda _src: ({}, src))

    def _sample(*_args):
        hits["n"] += 1
        return None

    monkeypatch.setattr(method, "_guided_sample", _sample)
    out = method._run_diffusion(src, [])
    assert hits["n"] == 2
    expected = (src + 1.0) / 2.0 * 255.0
    assert torch.equal(out, expected)
    assert torch.isfinite(out).all()
