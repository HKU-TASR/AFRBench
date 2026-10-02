"""Shared pytest configuration; ``slow`` tests need weights and are opt-in."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run tests that need model weights and minutes, not seconds.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs model weights; run with --runslow"
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    import os

    if config.getoption("--runslow") or os.environ.get("AFRBENCH_SLOW_TESTS") == "1":
        return
    skip_slow = pytest.mark.skip(reason="need --runslow or AFRBENCH_SLOW_TESTS=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
