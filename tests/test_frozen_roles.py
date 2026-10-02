"""Hand-computed frozen-role scenario tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from afrbench.dataset import BenchDataset
from afrbench.evaluator.gallery import build_scenario, per_user_split
from afrbench.evaluator.metrics.ranking import compute_ranking
from afrbench.evaluator.metrics.retrieval import per_query
from afrbench.evaluator.store import EmbeddingSet, EmbeddingStore


def _write_v4(root: Path) -> BenchDataset:
    images = []
    for identity, letters in (("A", "abcd"), ("B", "efgh")):
        for index, letter in enumerate(letters):
            role = ("album", "album", "query", "gallery")[index]
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
        "identities": {"A": {}, "B": {}},
    }))
    (root / "noise").mkdir(exist_ok=True)
    (root / "noise" / "filter.json").write_text(json.dumps({
        "schema": "afrbench-noise-filter-v1",
        "identities": [],
    }))
    return BenchDataset(root)


class _FakeStore(EmbeddingStore):
    def __init__(self, sets):
        self._sets = sets

    def get(self, source, digest, **kwargs):
        return self._sets[source]


def _set(ids, vectors, failed=()):
    matrix = np.stack(vectors).astype(np.float32)
    return EmbeddingSet(
        ids=list(ids),
        embeddings=matrix,
        failed_ids=list(failed),
    )


def test_frozen_roles_and_protection_set(tmp_path):
    ds = _write_v4(tmp_path)
    assert ds.query_images("A")[0].path.endswith("c.jpg")
    assert ds.gallery_images("A")[0].path.endswith("d.jpg")
    assert [r.path for r in ds.eval_images("A")] == [
        r.path for r in ds.protection_images("A")
    ]
    assert {r.role for r in ds.eval_images("A")} == {"query", "gallery"}


def test_per_user_split_is_removed():
    with pytest.raises(RuntimeError, match="frozen"):
        per_user_split()


def test_four_scenarios_and_failed_gallery_denominator(tmp_path):
    ds = _write_v4(tmp_path)
    q_a = "protectees/A/c.jpg"
    g_a = "protectees/A/d.jpg"
    q_b = "protectees/B/g.jpg"
    g_b = "protectees/B/h.jpg"
    # Hand-picked embeddings with a known ranking.
    originals = _set(
        [q_a, g_a, q_b, g_b],
        [
            np.array([1, 0, 0, 0], np.float32),
            np.array([0.9, 0.1, 0, 0], np.float32),
            np.array([0, 1, 0, 0], np.float32),
            np.array([0, 0.9, 0.1, 0], np.float32),
        ],
    )
    protected = _set(
        [q_a, g_a, q_b, g_b],
        [
            np.array([0, 0, 1, 0], np.float32),
            np.array([0, 0, 0.9, 0.1], np.float32),
            np.array([0, 0, 0, 1], np.float32),
            np.array([0.1, 0, 0, 0.9], np.float32),
        ],
        failed=[g_a],
    )
    store = _FakeStore({"originals": originals, "protected__m": protected})
    eval_lists = {
        "A": [q_a, g_a],
        "B": [q_b, g_b],
    }
    built = build_scenario(
        "2A",
        dataset=ds,
        store=store,
        system_digest="x",
        protected_source="protected__m",
        eval_lists=eval_lists,
        distractors=0,
    )
    assert built.intended_relevant == {"A": 1, "B": 1}
    assert g_a not in built.gallery_ids
    ranking = compute_ranking(
        built.query_embeddings, built.gallery_embeddings,
        built.query_labels, built.gallery_labels, top_k=2, device="cpu",
    )
    recalls = per_query(ranking, built, ks=("user",))["recall@K(user)"]
    # A's gallery image failed; the intended denominator stays 1, so recall is 0.
    assert recalls[built.query_ids.index(q_a)] == 0.0


def test_mix_assignment_uses_image_id_not_detection(tmp_path):
    ds = _write_v4(tmp_path)
    ids = [
        "protectees/A/c.jpg",
        "protectees/A/d.jpg",
        "protectees/B/g.jpg",
        "protectees/B/h.jpg",
    ]
    vecs = [np.eye(4, dtype=np.float32)[i] for i in range(4)]
    originals = _set(ids, vecs)
    protected = _set(ids, vecs, failed=["protectees/A/d.jpg"])
    store = _FakeStore({"originals": originals, "protected__m": protected})
    first = build_scenario(
        "mix-gallery",
        dataset=ds,
        store=store,
        system_digest="x",
        protected_source="protected__m",
        eval_lists={"A": ids[:2], "B": ids[2:]},
        distractors=0,
        mix_r=0.5,
    )
    second = build_scenario(
        "mix-gallery",
        dataset=ds,
        store=store,
        system_digest="x",
        protected_source="protected__m",
        eval_lists={"A": ids[:2], "B": ids[2:]},
        distractors=0,
        mix_r=0.5,
    )
    assert first.gallery_ids == second.gallery_ids
