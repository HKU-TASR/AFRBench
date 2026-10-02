"""Hand-computed identification metrics."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from afrbench.evaluator.metrics.identification import evaluate, per_query
from afrbench.evaluator.metrics.ranking import Ranking
from afrbench.evaluator.metrics.retrieval import mean_with_failures


def _ranking(top_labels, ranks=None):
    n = len(top_labels)
    k = len(top_labels[0]) if top_labels else 0
    if ranks is None:
        ranks = [torch.tensor([1], dtype=torch.int64) for _ in range(n)]
    return Ranking(
        ranks=ranks,
        top_labels=top_labels,
        top_sims=torch.zeros(n, k),
        gallery_size=k,
    )


def _scenario(labels, n_failed=0, **extra):
    payload = {
        "query_labels": labels,
        "n_query_failed": n_failed,
    }
    payload.update(extra)
    return SimpleNamespace(**payload)


def test_plurality_tie_and_strict_majority():
    ranking = _ranking([["A", "B", "A", "C", "B"]])
    scenario = _scenario(["A"])
    out = per_query(ranking, scenario, ks=(5,))
    assert out["rank5_appearance"] == [1.0]
    assert out["rank5_plurality"] == [1.0]
    assert out["rank5_majority"] == [0.0]

    ranking = _ranking([["A", "A", "A", "B", "C"]])
    out = per_query(ranking, scenario, ks=(5,))
    assert out["rank5_majority"] == [1.0]


def test_rank1_all_three_agree():
    ranking = _ranking([["A", "B", "C"]])
    out = per_query(ranking, _scenario(["A"]), ks=(1,))
    assert out["rank1_appearance"] == [1.0]
    assert out["rank1_plurality"] == [1.0]
    assert out["rank1_majority"] == [1.0]
    ranking = _ranking([["B", "A", "C"]], ranks=[torch.tensor([2], dtype=torch.int64)])
    out = per_query(ranking, _scenario(["A"]), ks=(1,))
    assert out["rank1_appearance"] == [0.0]
    assert out["rank1_plurality"] == [0.0]
    assert out["rank1_majority"] == [0.0]


def test_failed_queries_lower_rate():
    ranking = _ranking([["A", "A", "A", "B", "C"]])
    scenario = _scenario(["A"], n_failed=1)
    rates = evaluate(ranking, scenario, ks=(5,))
    assert rates["rank5_majority"] == mean_with_failures([1.0], 1)
    assert rates["rank5_majority"] == 0.5


def test_identification_only_emits_own_identity_metrics():
    ranking = Ranking(
        ranks=[torch.tensor([3], dtype=torch.int64)],
        top_labels=[["noise/T", "A", "B"]],
        top_sims=torch.zeros(1, 3),
        gallery_size=3,
        top_keys=[["T__0", "A__q", "B__q"]],
    )
    out = per_query(ranking, _scenario(["A"]), ks=(1, 5))
    assert all(not name.startswith("target_") for name in out)
    assert out["rank1_appearance"] == [0.0]
    assert out["rank5_appearance"] == [1.0]
