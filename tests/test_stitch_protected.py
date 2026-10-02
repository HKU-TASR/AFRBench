"""Stitching disjoint per-variant protected runs: grouping rules and assigned-store reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from afrbench.dataset import BenchDataset
from afrbench.evaluator.gallery import build_scenario, protected_set_name
from afrbench.evaluator.job import group_runs_for_scoring, stitch_protected_embeddings
from afrbench.evaluator.store import EmbeddingSet, EmbeddingStore
from afrbench.runs import ProtectedImage, ProtectedRun


def _write_identities(root: Path, names: tuple[str, ...]) -> BenchDataset:
    root.mkdir(parents=True, exist_ok=True)
    images = []
    identities = {}
    for identity in names:
        identities[identity] = {}
        for index, role in enumerate(("album", "album", "query", "gallery")):
            letter = "abcdefghij"[index]
            images.append({
                "path": f"protectees/{identity}/{letter}.jpg",
                "identity": identity,
                "split": "protectee",
                "role": role,
                "image_id": f"{identity}-{letter}",
                "sha256": f"{identity}{letter}",
                "size": [8, 8],
                "session": f"{identity}-{role}",
                "original_box": [0, 0, 8],
                "face_box": [1, 1, 7, 7],
                "landmarks": [[2, 2], [6, 2], [4, 4], [2, 6], [6, 6]],
            })
    (root / "annotations.json").write_text(json.dumps({
        "schema": "afrbench-dataset-v4",
        "images": images,
        "meta": {},
    }))
    (root / "identities.json").write_text(json.dumps({
        "schema": "afrbench-identities-v4",
        "identities": identities,
    }))
    (root / "noise").mkdir(exist_ok=True)
    (root / "noise" / "filter.json").write_text(json.dumps({
        "schema": "afrbench-noise-filter-v1",
        "identities": [],
    }))
    return BenchDataset(root)


def _eval_paths(identity: str) -> list[str]:
    return [
        f"protectees/{identity}/c.jpg",
        f"protectees/{identity}/d.jpg",
    ]


def _run(root: Path, digest: str, identities: tuple[str, ...]) -> ProtectedRun:
    images = [
        ProtectedImage(source_path=path, output=root / "unused.npy", dtype="float32")
        for identity in identities
        for path in _eval_paths(identity)
    ]
    return ProtectedRun(root=root, images=images, surrogates=[], digest=digest)


def _put(store: EmbeddingStore, source: str, ids: list[str], vectors, failed=()):
    store.put(
        source,
        "sys",
        EmbeddingSet(
            ids=list(ids),
            embeddings=np.stack(vectors).astype(np.float32),
            failed_ids=list(failed),
        ),
    )


def test_group_and_stitch_four_identities(tmp_path):
    ds = _write_identities(tmp_path / "ds", ("A", "B", "C", "D"))
    left = _run(tmp_path / "left", "d-left", ("A", "B"))
    right = _run(tmp_path / "right", "d-right", ("C", "D"))
    eval_lists = {name: _eval_paths(name) for name in ("A", "B", "C", "D")}
    loaded = {"m@t1": left, "m@t2": right}
    specs = group_runs_for_scoring(ds, loaded, eval_lists)
    assert len(specs) == 1
    assert specs[0]["stitch"] is True
    assert specs[0]["score_key"] == "m"

    store = EmbeddingStore(tmp_path / "cache")
    left_ids = _eval_paths("A") + _eval_paths("B")
    right_ids = _eval_paths("C") + _eval_paths("D")
    left_src = protected_set_name("m@t1", left.digest)
    right_src = protected_set_name("m@t2", right.digest)
    _put(store, left_src, left_ids, [np.array([1, 0], np.float32)] * 4)
    _put(store, right_src, right_ids, [np.array([0, 1], np.float32)] * 4)
    source, digest = stitch_protected_embeddings(
        dataset=ds,
        store=store,
        system_digest="sys",
        method="m",
        variants=loaded,
        run_sources={"m@t1": left_src, "m@t2": right_src},
        eval_lists=eval_lists,
    )
    stitched = store.get(source, "sys")
    assert set(stitched.ids) == set(left_ids + right_ids)
    index = stitched.row_index()
    for path in left_ids:
        assert np.allclose(stitched.embeddings[index[path]], [1, 0])
    for path in right_ids:
        assert np.allclose(stitched.embeddings[index[path]], [0, 1])
    assert stitched.failed_ids == []
    assert digest

    originals = EmbeddingSet(
        ids=left_ids + right_ids,
        embeddings=np.stack([np.array([0.5, 0.5], np.float32)] * 8),
        failed_ids=[],
    )
    store.put("originals", "sys", originals)
    for kind in ("1A", "2A", "2B"):
        built = build_scenario(
            kind,
            dataset=ds,
            store=store,
            system_digest="sys",
            protected_source=source,
            eval_lists=eval_lists,
            distractors=0,
        )
        assert built.n_query_failed == 0
        assert built.n_gallery_failed == 0
        assert set(built.query_ids) | set(built.gallery_ids) == set(left_ids + right_ids)


def test_group_singleton_and_full_overlap_are_separate(tmp_path):
    ds = _write_identities(tmp_path / "ds", ("A", "B"))
    eval_lists = {name: _eval_paths(name) for name in ("A", "B")}
    one = _run(tmp_path / "one", "d1", ("A", "B"))
    specs = group_runs_for_scoring(ds, {"fawkes": one}, eval_lists)
    assert specs == [{
        "score_key": "fawkes",
        "stitch": False,
        "variants": {"fawkes": one},
    }]
    left = _run(tmp_path / "l", "dl", ("A", "B"))
    right = _run(tmp_path / "r", "dr", ("A", "B"))
    specs = group_runs_for_scoring(ds, {"m@t1": left, "m@t2": right}, eval_lists)
    assert [spec["score_key"] for spec in specs] == ["m@t1", "m@t2"]
    assert all(not spec["stitch"] for spec in specs)


def test_group_overlap_or_uncovered_raises(tmp_path):
    ds = _write_identities(tmp_path / "ds", ("A", "B", "C"))
    eval_lists = {name: _eval_paths(name) for name in ("A", "B", "C")}
    left = _run(tmp_path / "l", "dl", ("A", "B"))
    right = _run(tmp_path / "r", "dr", ("B", "C"))
    with pytest.raises(ValueError, match="cannot score"):
        group_runs_for_scoring(ds, {"m@t1": left, "m@t2": right}, eval_lists)
    incomplete = _run(tmp_path / "i", "di", ("A",))
    other = _run(tmp_path / "o", "do", ("B",))
    with pytest.raises(ValueError, match="cannot score"):
        group_runs_for_scoring(ds, {"m@t1": incomplete, "m@t2": other}, eval_lists)


def test_stitch_propagates_assigned_failures_and_rejects_gaps(tmp_path):
    ds = _write_identities(tmp_path / "ds", ("A", "B"))
    eval_lists = {name: _eval_paths(name) for name in ("A", "B")}
    left = _run(tmp_path / "l", "dl", ("A",))
    right = _run(tmp_path / "r", "dr", ("B",))
    store = EmbeddingStore(tmp_path / "cache")
    left_src = protected_set_name("m@t1", left.digest)
    right_src = protected_set_name("m@t2", right.digest)
    failed_path = _eval_paths("A")[0]
    _put(
        store, left_src, [_eval_paths("A")[1]],
        [np.array([1, 0], np.float32)],
        failed=[failed_path],
    )
    _put(store, right_src, _eval_paths("B"), [np.array([0, 1], np.float32)] * 2)
    source, _ = stitch_protected_embeddings(
        dataset=ds,
        store=store,
        system_digest="sys",
        method="m",
        variants={"m@t1": left, "m@t2": right},
        run_sources={"m@t1": left_src, "m@t2": right_src},
        eval_lists=eval_lists,
    )
    stitched = store.get(source, "sys")
    assert failed_path in stitched.failed_ids
    assert failed_path not in stitched.ids

    gap_src = protected_set_name("m@t2", "other")
    store.put(gap_src, "sys", EmbeddingSet(
        ids=[_eval_paths("B")[0]],
        embeddings=np.array([[0, 1]], np.float32),
        failed_ids=[],
    ))
    with pytest.raises(ValueError, match="execution errors"):
        stitch_protected_embeddings(
            dataset=ds,
            store=store,
            system_digest="sys",
            method="m",
            variants={"m@t1": left, "m@t2": _run(tmp_path / "r2", "other", ("B",))},
            run_sources={"m@t1": left_src, "m@t2": gap_src},
            eval_lists=eval_lists,
        )
