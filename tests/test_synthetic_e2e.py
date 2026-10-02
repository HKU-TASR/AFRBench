"""Small synthetic end-to-end checks: dataset roles, noise cache, and block ranking."""
from __future__ import annotations

import json

import numpy as np
import torch

from afrbench.dataset import BenchDataset
from afrbench.evaluator.metrics.ranking import compute_ranking
from afrbench.evaluator.metrics.retrieval import per_query
from afrbench.evaluator.noise_cache import IncrementalWriter
from tests.test_frozen_roles import _write_v4


def test_synthetic_roles_and_intended_denominator(tmp_path):
    dataset = _write_v4(tmp_path)
    assert [row.role for row in dataset.images_of("A")] == ["album", "album", "query", "gallery"]
    assert len(dataset.query_images("A")) == 1
    assert len(dataset.gallery_images("A")) == 1


def test_synthetic_full_noise_ranking_keeps_denominator(tmp_path):
    writer = IncrementalWriter(tmp_path / "mc", dim=2, expected=4)
    emb = np.array([[1, 0], [0.2, 0.98], [0.99, 0.14], [0, 1]], dtype=np.float32)
    writer.append(["n__0", "n__1", "n__2", "n__3"], emb, ["n0", "n1", "n2", "n3"])
    writer.finalize(pack_id="tiny", fr_digest="fr", weight_identity="w")
    from afrbench.evaluator.noise_cache import FullNoiseCache

    reader = FullNoiseCache(tmp_path / "mc").reader()
    blocks = [
        (torch.from_numpy(np.asarray(e)), labs)
        for e, labs, _k in reader.iter_blocks(exclude_identities=["n1"], chunk=2)
    ]
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    gallery = torch.nn.functional.normalize(torch.tensor([[0.8, 0.6]]), dim=1)
    ranking = compute_ranking(
        query, gallery, ["A"], ["A"], extra_blocks=blocks, top_k=3, chunk=2
    )
    scenario = type("S", (), {
        "query_labels": ["A"],
        "intended_relevant": {"A": 25},
        "n_query_failed": 0,
        "n_gallery_failed": 1,
    })()
    metrics = per_query(ranking, scenario, ks=("user",), fixed_ks=(25,))
    assert metrics["recall@K(user)"][0] <= 1.0
    assert ranking.gallery_size == 1 + sum(block[0].shape[0] for block in blocks)
