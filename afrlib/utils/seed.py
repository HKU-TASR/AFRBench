"""Scoped RNG seeding: seed every generator for one block, then restore the caller's state."""
from __future__ import annotations

import contextlib
import random
from typing import Iterator, Optional

import numpy as np
import torch


@contextlib.contextmanager
def seeded(seed: int, device: "torch.device | str | None" = None) -> Iterator[None]:
    """Seed Python, NumPy and torch for the block, as a global reset to ``seed`` would, and
    restore the caller's generators and cuDNN flags on exit.

    Args:
        seed: The seed applied inside the block.
        device: A CUDA device whose generator is also seeded and restored.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    devices = []
    if device is not None and torch.device(device).type == "cuda":
        index = torch.device(device).index
        devices = [torch.cuda.current_device() if index is None else index]
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags
