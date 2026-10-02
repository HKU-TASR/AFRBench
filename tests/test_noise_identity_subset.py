"""Identity-subset draw: targets stay in, seed is stable."""
from __future__ import annotations

from afrbench.dataset import BenchDataset, select_identity_subset
from afrbench.evaluator.job import resolve_gallery_distractors
from tests.test_frozen_roles import _write_v4


def test_select_identity_subset_keeps_2005_targets_and_is_seed_stable():
    targets = [f"t{i:04d}" for i in range(2005)]
    extras = [f"n{i:05d}" for i in range(10_000)]
    available = targets + extras
    first = select_identity_subset(available, 4000, seed=0, ensure_identities=targets)
    second = select_identity_subset(available, 4000, seed=0, ensure_identities=targets)
    other = select_identity_subset(available, 4000, seed=1, ensure_identities=targets)
    assert len(first) == 4000
    assert set(targets) <= set(first)
    assert first == second
    assert first != other
    assert set(targets) <= set(other)


def test_select_identity_subset_keeps_all_when_count_exceeds_pool():
    available = [f"id{i}" for i in range(10)]
    assert select_identity_subset(available, 50, seed=0, ensure_identities=["id1"]) == available


def test_noise_identity_subset_on_dataset_keeps_ensure_set(tmp_path, monkeypatch):
    ds = _write_v4(tmp_path)
    hq = ["T0001", "T0002", "T0003", "T0004", "T0005"]
    pool = [f"P{i:04d}" for i in range(2000)]
    monkeypatch.setattr(BenchDataset, "hq_target_identities", lambda self: list(hq))
    monkeypatch.setattr(BenchDataset, "target_identities", lambda self: list(pool))
    ensure = ds.hq_target_identities() + ds.target_identities()
    assert len(set(ensure)) == 2005
    universe = sorted(set(ensure) | {f"x{i:04d}" for i in range(3000)})
    monkeypatch.setattr(BenchDataset, "noise_identity_ids", lambda self: list(universe))
    subset = ds.noise_identity_subset(2500, seed=0, ensure_identities=ensure)
    assert set(ensure) <= set(subset)
    assert len(subset) == 2500
    assert ds.noise_identity_subset(2500, seed=0, ensure_identities=ensure) == subset


def test_resolve_identities_mode_excludes_the_complement(tmp_path, monkeypatch):
    ds = _write_v4(tmp_path)
    monkeypatch.setattr(BenchDataset, "hq_target_identities", lambda self: ["T1"])
    monkeypatch.setattr(BenchDataset, "target_identities", lambda self: ["P1"])
    universe = ["T1", "P1", "n1", "n2", "n3", "n4"]
    monkeypatch.setattr(BenchDataset, "noise_identity_ids", lambda self: list(universe))
    distractors, full_noise, exclude = resolve_gallery_distractors(
        ds,
        {"mode": "identities", "identity_count": 4, "seed": 0},
        seed=0,
    )
    assert distractors == -1
    assert full_noise is True
    selected = set(universe) - set(exclude or [])
    assert {"T1", "P1"} <= selected
    assert len(selected) == 4
