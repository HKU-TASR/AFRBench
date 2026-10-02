"""docs/methods.md lists every registered method with the roles its class declares."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from afrlib.afr import _AFR_TARGETS

DOC = Path(__file__).resolve().parents[1] / "docs" / "methods.md"


def _rows():
    rows = {}
    for line in DOC.read_text().splitlines():
        match = re.match(r"\| `([a-z0-9_]+)`", line)
        if match:
            rows[match.group(1)] = [cell.strip() for cell in line.strip("|").split("|")]
    return rows


def test_table_lists_every_method():
    assert sorted(_rows()) == sorted(_AFR_TARGETS)


@pytest.mark.parametrize("name", sorted(_AFR_TARGETS))
def test_table_matches_declared_roles(name):
    from afrlib import AFR_REGISTRY

    try:
        cls = AFR_REGISTRY[name]
    except ImportError as exc:
        pytest.skip(f"{name} dependencies are not installed: {exc}")
    row = _rows()[name]
    documented = {role for cell in row[3:6] for role in re.findall(r"`([a-z_]+)`", cell)}
    declared = set()
    for stage in ("train_inputs", "fit_inputs", "eval_inputs"):
        declared |= set(getattr(cls, stage).known)
    declared.discard("eval_geometry")
    assert documented == declared
