"""Retrieval metrics over one scenario's ranking."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch

from afrbench.evaluator.metrics.ranking import Ranking


def _user_gallery_sizes(scenario) -> List[int]:
    """Each query's genuine-gallery size before detection failures (Recall@K denominator)."""
    intended = getattr(scenario, "intended_relevant", None) or {}
    if intended:
        return [int(intended.get(label, 0)) for label in scenario.query_labels]
    counts = Counter(scenario.gallery_labels)
    return [counts[label] for label in scenario.query_labels]


def recall_per_query(
    ranking: Ranking, k, own_sizes: Sequence[int]
) -> List[float]:
    """Per-query Recall@K; ``k="user"`` uses each query's own gallery size."""
    out = []
    for ranks, own in zip(ranking.ranks, own_sizes):
        cutoff = own if k == "user" else int(k)
        denominator = own if k == "user" else max(own, 1)
        if not denominator or not cutoff:
            out.append(0.0)
            continue
        out.append(float((ranks <= cutoff).sum().item()) / denominator)
    return out


def average_precision_per_query(
    ranking: Ranking,
    k: Optional[int] = None,
    intended: Optional[Sequence[int]] = None,
) -> List[float]:
    """Per-query average precision, the mean of ``j / r_j`` over the relevant ranks; truncated at
    ``k``, only hits within ``k`` count and the denominator is ``min(intended, k)``.
    """
    out = []
    for index, ranks in enumerate(ranking.ranks):
        total = int(intended[index]) if intended is not None else int(ranks.numel())
        denominator = total if k is None else min(total, k)
        visible = ranks if k is None else ranks[ranks <= k]
        if not denominator or visible.numel() == 0:
            out.append(0.0)
            continue
        hits = torch.arange(1, visible.numel() + 1, dtype=torch.float64)
        out.append(float((hits / visible.to(torch.float64)).sum().item()) / denominator)
    return out


def ndcg_per_query(ranking: Ranking, k: int, own_sizes: Sequence[int]) -> List[float]:
    """Per-query NDCG@K with binary relevance."""
    discounts = [1.0 / math.log2(rank + 1) for rank in range(1, k + 1)]
    out = []
    for ranks, own in zip(ranking.ranks, own_sizes):
        within = ranks[ranks <= k]
        dcg = sum(discounts[int(rank) - 1] for rank in within.tolist())
        idcg = sum(discounts[: min(own, k)])
        out.append(dcg / idcg if idcg else 0.0)
    return out


def per_query(
    ranking: Ranking,
    scenario,
    *,
    ks: Sequence = ("user",),
    fixed_ks: Sequence[int] = (),
    ndcg_ks: Sequence[int] = (),
    ap: bool = True,
    ap_ks: Sequence[int] = (),
) -> Dict[str, List[float]]:
    """Every retrieval metric as a per-query vector, parallel to ``query_ids``."""
    own_sizes = _user_gallery_sizes(scenario)
    out: Dict[str, List[float]] = {}
    for k in list(ks) + list(fixed_ks):
        name = "recall@K(user)" if k == "user" else f"recall@{int(k)}"
        out[name] = recall_per_query(ranking, k, own_sizes)
    if ap:
        out["AP"] = average_precision_per_query(ranking, intended=own_sizes)
    for k in ap_ks:
        out[f"AP@{int(k)}"] = average_precision_per_query(
            ranking, k=int(k), intended=own_sizes
        )
    for k in ndcg_ks:
        out[f"NDCG@{int(k)}"] = ndcg_per_query(ranking, int(k), own_sizes)
    return out


def evaluate(
    ranking: Ranking,
    scenario,
    *,
    ks: Sequence = ("user",),
    fixed_ks: Sequence[int] = (),
    ndcg_ks: Sequence[int] = (),
    ap: bool = True,
    ap_ks: Sequence[int] = (),
) -> Dict[str, float]:
    """All retrieval metrics for one scenario, as a flat ``{name: mean}``."""
    values = per_query(
        ranking, scenario, ks=ks, fixed_ks=fixed_ks, ndcg_ks=ndcg_ks, ap=ap, ap_ks=ap_ks
    )
    out = {name: mean_with_failures(v, scenario.n_query_failed) for name, v in values.items()}
    out["det_fail_query"] = float(scenario.n_query_failed)
    out["det_fail_gallery"] = float(scenario.n_gallery_failed)
    return out


def mean_with_failures(values: Sequence[float], n_failed: int) -> float:
    """Mean over ranked queries plus ``n_failed`` zero-scoring failed queries."""
    n = len(values) + n_failed
    return float(sum(values) / n) if n else 0.0
