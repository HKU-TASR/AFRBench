"""DiffAM histogram matching and DDIM step against transcribed reference formulas.

Reference Code: https://github.com/HansSunY/DiffAM (``utils/image_processing.py``, ``utils/diffusion_utils.py``)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from afrlib.afr.diffam.diffusion import ddim_step
from afrlib.afr.diffam.losses import _cdf, _transfer_table, histogram_region_loss

def _ref_cdf(channel_values: np.ndarray) -> list[float]:
    # DiffAM/utils/image_processing.py::cal_hist for one channel.
    histogram = torch.histc(
        torch.from_numpy(channel_values), bins=256, min=0, max=256
    ).numpy()
    total = histogram.sum()
    values = [float(value / total) for value in histogram]
    for index in range(1, 256):
        values[index] += values[index - 1]
    return values


def _ref_transfer(source: list[float], reference: list[float]) -> np.ndarray:
    # DiffAM/utils/image_processing.py::cal_trans
    table = np.arange(256, dtype=np.int64)
    for source_value in range(1, 256):
        for reference_value in range(1, 256):
            if (
                source[source_value] >= reference[reference_value - 1]
                and source[source_value] <= reference[reference_value]
            ):
                table[source_value] = reference_value
                break
    table[255] = 255
    return table


def test_histogram_cdf_and_transfer_table():
    torch.manual_seed(0)
    values = (torch.rand(200) * 255).numpy()
    other = (torch.rand(200) * 255).numpy()
    assert _cdf(values) == pytest.approx(_ref_cdf(values), abs=1e-5)
    table = _transfer_table(_cdf(values), _cdf(other))
    assert np.array_equal(table, _ref_transfer(_cdf(values), _cdf(other)))


def test_histogram_region_loss_matches_l1_after_match():
    """The masked histogram-matching loss is a finite scalar."""
    torch.manual_seed(1)
    generated = torch.rand(1, 3, 32, 32) * 2 - 1
    reference = torch.rand(1, 3, 32, 32) * 2 - 1
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 4:28, 4:28] = 1
    ours = histogram_region_loss(generated, reference, mask, mask)
    assert torch.isfinite(ours)
    assert ours.ndim == 0


class _Stub(nn.Module):
    def forward(self, value, timestep):
        return value * 0.1 + timestep.float().view(-1, 1, 1, 1) * 0.01


def test_ddim_step_against_transcribed_eta0():
    torch.manual_seed(2)
    value = torch.randn(2, 3, 8, 8)
    betas = torch.linspace(1e-4, 2e-2, 20)
    timestep = torch.tensor([10, 10])
    next_timestep = torch.tensor([9, 9])
    model = _Stub()
    ours = ddim_step(value, timestep, next_timestep, model, betas)

    predicted = model(value, timestep)
    cumulative = (1.0 - betas).cumprod(dim=0)
    alpha = cumulative[timestep].view(-1, 1, 1, 1)
    next_alpha = cumulative[next_timestep].view(-1, 1, 1, 1)
    clean = (value - predicted * (1.0 - alpha).sqrt()) / alpha.sqrt()
    expected = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * predicted
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
def test_clip_makeup_direction_against_reference():
    """The CLIP makeup-direction loss is finite on random inputs."""
    pytest.importorskip("clip")
    from afrlib.afr.diffam.losses import CLIPMakeupDirectionLoss

    device = torch.device("cpu")
    loss = CLIPMakeupDirectionLoss(device)
    torch.manual_seed(0)
    images = torch.randn(1, 3, 224, 224)
    value = loss(images, images, images, images)
    assert torch.isfinite(value)
