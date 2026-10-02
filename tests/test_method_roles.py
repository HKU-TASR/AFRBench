"""Derived method-role predicates match the expected registry buckets."""
from __future__ import annotations

import pytest

from afrlib import AFR_REGISTRY
from afrbench.protocol import bakes_target, method_roles, protects_per_target, uses_target_pool

EXPECTED_BAKED = frozenset({"adv_makeup", "amt_gan", "diffam"})
EXPECTED_RUNTIME = frozenset({
    "dpa", "adv_diffusion", "fpp_wdp", "adv_inversion",
    "clip2protect", "diffaim", "dadp", "gift", "at3d",
})
EXPECTED_PER_TARGET = EXPECTED_BAKED | EXPECTED_RUNTIME
EXPECTED_POOL = frozenset({"tip_im", "divtrackee"})


def test_derived_sets_equal_literals():
    # Walking the lazy registry imports every method; pytorch3d and dlib are
    # source-build-only dependencies of protego and amt_gan.
    pytest.importorskip("pytorch3d")
    pytest.importorskip("dlib")
    baked = frozenset(name for name in AFR_REGISTRY if bakes_target(name))
    per_target = frozenset(name for name in AFR_REGISTRY if protects_per_target(name))
    pool = frozenset(name for name in AFR_REGISTRY if uses_target_pool(name))
    assert baked == EXPECTED_BAKED
    assert per_target == EXPECTED_PER_TARGET
    assert pool == EXPECTED_POOL


def test_every_method_in_exactly_one_bucket():
    pytest.importorskip("pytorch3d")
    pytest.importorskip("dlib")
    seen = []
    for name in AFR_REGISTRY:
        roles = method_roles(AFR_REGISTRY[name])
        baked = roles["target_baked"]
        runtime = roles["runtime_target"] and not baked
        pool = roles["target_pool"]
        untargeted = not (baked or runtime or pool)
        flags = [baked, runtime, pool, untargeted]
        assert sum(flags) == 1, name
        seen.append((baked, runtime, pool, untargeted))
    assert any(row[0] for row in seen)
    assert any(row[1] for row in seen)
    assert any(row[2] for row in seen)
    assert any(row[3] for row in seen)
