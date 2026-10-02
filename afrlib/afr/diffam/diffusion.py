"""Deterministic DDIM operations used by DiffAM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


DIFFUSION_STEPS = 1_000
DEFAULT_T0 = 60
DEFAULT_INVERSION_POINTS = 20
DEFAULT_REVERSE_STEPS = 6


def linear_beta_schedule(
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    steps: int = DIFFUSION_STEPS,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Linear beta schedule in float32, computed as in the upstream code."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    # Upstream builds the schedule in float64 (NumPy) and casts to float32.
    # Build it on the CPU (MPS has no float64), then move it to the device.
    schedule = torch.linspace(
        beta_start, beta_end, steps, dtype=torch.float64
    ).float()
    return schedule if device is None else schedule.to(device)


def timestep_grid(t0: int, points: int) -> tuple[int, ...]:
    """Timesteps ``[int(s) for s in np.linspace(0, 1, points) * t0]``, as upstream."""

    if t0 < 0 or points < 2:
        raise ValueError("t0 must be non-negative and points must be at least two")
    values = torch.linspace(0.0, 1.0, points, dtype=torch.float64) * t0
    return tuple(int(value) for value in values.tolist())


def inversion_transitions(
    t0: int = DEFAULT_T0, points: int = DEFAULT_INVERSION_POINTS
) -> tuple[tuple[int, int], ...]:
    """Ascending ``(t, t_next)`` pairs that invert a clean image up to ``t0``."""
    sequence = timestep_grid(t0, points)
    previous = (-1,) + sequence[:-1]
    return tuple(zip(previous[1:], sequence[1:]))


def reverse_transitions(
    t0: int = DEFAULT_T0, steps: int = DEFAULT_REVERSE_STEPS
) -> tuple[tuple[int, int], ...]:
    """Descending ``(t, t_next)`` pairs from ``t0`` to the clean image (``-1``)."""
    sequence = timestep_grid(t0, steps)
    previous = (-1,) + sequence[:-1]
    return tuple(zip(reversed(sequence), reversed(previous)))


def _extract(
    values: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]
) -> torch.Tensor:
    if timesteps.ndim != 1 or timesteps.shape[0] != shape[0]:
        raise ValueError("timestep batch must match the image batch")
    gathered = values.to(timesteps.device).gather(0, timesteps.long())
    return gathered.reshape((shape[0],) + (1,) * (len(shape) - 1))


def ddim_step(
    value: torch.Tensor,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    model: nn.Module,
    betas: torch.Tensor,
    *,
    checkpoint_model: bool = False,
) -> torch.Tensor:
    """One eta-zero DDIM transition, valid in either temporal direction."""

    predicted_noise = (
        checkpoint(model, value, timestep, use_reentrant=False)
        if checkpoint_model
        else model(value, timestep)
    )
    cumulative_alpha = (1.0 - betas).cumprod(dim=0)
    alpha = _extract(cumulative_alpha, timestep, value.shape)
    if bool((next_timestep == -1).all()):
        next_alpha = torch.ones_like(alpha)
    elif bool((next_timestep < 0).any()):
        raise ValueError("next_timestep must be entirely -1 or non-negative")
    else:
        next_alpha = _extract(cumulative_alpha, next_timestep, value.shape)
    predicted_clean = (
        value - predicted_noise * (1.0 - alpha).sqrt()
    ) / alpha.sqrt()
    return (
        next_alpha.sqrt() * predicted_clean
        + (1.0 - next_alpha).sqrt() * predicted_noise
    )


def run_transitions(
    value: torch.Tensor,
    transitions: Sequence[tuple[int, int]],
    model: nn.Module,
    betas: torch.Tensor,
    *,
    checkpoint_model: bool = False,
) -> torch.Tensor:
    """Apply a fixed transition sequence to a batch."""

    for current, following in transitions:
        timestep = torch.full(
            (value.shape[0],), current, device=value.device, dtype=torch.long
        )
        next_timestep = torch.full_like(timestep, following)
        value = ddim_step(
            value,
            timestep,
            next_timestep,
            model,
            betas,
            checkpoint_model=checkpoint_model,
        )
    return value


__all__ = [
    "DEFAULT_INVERSION_POINTS",
    "DEFAULT_REVERSE_STEPS",
    "DEFAULT_T0",
    "DIFFUSION_STEPS",
    "ddim_step",
    "inversion_transitions",
    "linear_beta_schedule",
    "reverse_transitions",
    "run_transitions",
    "timestep_grid",
]
