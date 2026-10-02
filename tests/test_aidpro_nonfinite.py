"""AIDPro skips a non-finite train step and raises after too many consecutive skips."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from afrlib.afr.aidpro.aidpro import AIDPro
from afrlib.afr.aidpro.models import WatermarkDecoder, WatermarkInsert
from afrlib.contracts import TrainData


def _bare_method() -> AIDPro:
    method = AIDPro.__new__(AIDPro)
    method.device = torch.device("cpu")
    method.primary_index = 0
    method.message_length = 30
    method.epochs = 1
    method.batch_size = 2
    method.max_steps = None
    method.lambda_identity = 5.0
    method.lambda_reconstruction = 5.0
    method.lambda_watermark = 10.0
    method.max_consecutive_nonfinite_steps = 8
    method._mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=torch.float32
    ).view(1, 3, 1, 1)
    method._std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=torch.float32
    ).view(1, 3, 1, 1)
    method.wi = WatermarkInsert(542)
    method.wd = WatermarkDecoder(30)
    method.fmm = nn.Identity()
    method._lpips = _ZeroLpips()
    method._optimizer_wi = torch.optim.Adam(method.wi.parameters(), lr=1e-4)
    method._optimizer_wd = torch.optim.Adam(method.wd.parameters(), lr=1e-4)
    method.actual_steps = 0
    return method


class _ZeroLpips(nn.Module):
    def forward(self, *_args, **_kwargs):
        return torch.zeros(1, 1, 1, 1)


def _unit_embed(_recognizer, internal: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.ones(internal.shape[0], 512), dim=1)


def _faces(batch: int = 2) -> torch.Tensor:
    return torch.linspace(0, 255, batch * 3 * 224 * 224).reshape(batch, 3, 224, 224)


def test_train_step_skips_optimizer_when_loss_nonfinite(monkeypatch):
    method = _bare_method()
    monkeypatch.setattr(method, "_embed_internal", _unit_embed)
    monkeypatch.setattr(
        method,
        "_generate_internal",
        lambda source, _identity, _messages: (
            torch.full_like(source, float("inf")),
            _identity,
        ),
    )
    before_wi = method.wi.fc1.weight.detach().clone()
    before_wd = method.wd.fc1.weight.detach().clone()
    out = method._train_step(_faces(), [object()])
    assert out is None
    assert method.actual_steps == 0
    assert torch.equal(method.wi.fc1.weight, before_wi)
    assert torch.equal(method.wd.fc1.weight, before_wd)


def test_train_step_finite_loss_still_steps(monkeypatch):
    method = _bare_method()
    monkeypatch.setattr(method, "_embed_internal", _unit_embed)
    monkeypatch.setattr(
        method,
        "_generate_internal",
        lambda source, identity, _messages: (source, identity),
    )
    before_wd = method.wd.fc1.weight.detach().clone()
    out = method._train_step(_faces(), [object()])
    assert out is not None
    assert method.actual_steps == 1
    assert torch.isfinite(torch.tensor(out["total"]))
    assert not torch.equal(method.wd.fc1.weight, before_wd)


def test_train_continues_after_one_nonfinite_step(monkeypatch):
    method = _bare_method()
    hits = {"n": 0}

    def _step(_images, _recognizers):
        hits["n"] += 1
        if hits["n"] == 1:
            return None
        method.actual_steps += 1
        return {
            "total": 0.8,
            "bit_accuracy": 0.99,
            "actual_step": float(method.actual_steps),
        }

    monkeypatch.setattr(method, "_check_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(method, "_begin_training", lambda: None)
    monkeypatch.setattr(method, "_ensure_training_components", lambda: [object()])
    monkeypatch.setattr(
        method,
        "_build_loader",
        lambda _corpus: [_faces(), _faces()],
    )
    monkeypatch.setattr(method, "_mark_ready", lambda: None)
    monkeypatch.setattr(method, "_train_step", _step)
    method.train(TrainData(corpus=object()))
    assert hits["n"] == 2
    assert method.actual_steps == 1


def test_train_raises_after_consecutive_nonfinite_skips(monkeypatch):
    method = _bare_method()
    method.max_consecutive_nonfinite_steps = 2
    monkeypatch.setattr(method, "_check_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(method, "_begin_training", lambda: None)
    monkeypatch.setattr(method, "_ensure_training_components", lambda: [object()])
    monkeypatch.setattr(
        method,
        "_build_loader",
        lambda _corpus: [_faces(), _faces(), _faces()],
    )
    monkeypatch.setattr(method, "_mark_ready", lambda: None)
    monkeypatch.setattr(method, "_train_step", lambda *_args, **_kw: None)
    with pytest.raises(FloatingPointError, match="2 consecutive steps"):
        method.train(TrainData(corpus=object()))
    assert method.actual_steps == 0
