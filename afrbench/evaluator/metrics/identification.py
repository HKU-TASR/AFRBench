"""Closed-set identification metrics over one scenario's ranking."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from afrbench.evaluator.metrics.ranking import Ranking
from afrbench.evaluator.metrics.retrieval import mean_with_failures


def per_query(
    ranking: Ranking,
    scenario,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, List[float]]:
    """Per-query identification outcomes (1.0 identified, 0.0 not)."""
    out: Dict[str, List[float]] = {}
    for k in ks:
        appearance, plurality, majority = [], [], []
        for ranks, top, query_label in zip(
            ranking.ranks, ranking.top_labels, scenario.query_labels
        ):
            appearance.append(float(bool((ranks <= k).any())))
            head = top[:k]
            is_plurality = False
            is_majority = False
            if head:
                counts = Counter(head)
                top_count = max(counts.values())
                winners = [label for label, count in counts.items() if count == top_count]
                # Ties keep the first top-K label that reached the winning count.
                winner = next(label for label in head if label in winners)
                is_plurality = winner == query_label
                is_majority = counts[query_label] > k / 2
            plurality.append(float(is_plurality))
            majority.append(float(is_majority))
        out[f"rank{k}_appearance"] = appearance
        out[f"rank{k}_plurality"] = plurality
        out[f"rank{k}_majority"] = majority
    return out


def evaluate(
    ranking: Ranking,
    scenario,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Identification rates for one scenario, as a flat ``{name: rate}``."""
    values = per_query(ranking, scenario, ks)
    out = {
        name: mean_with_failures(v, scenario.n_query_failed) for name, v in values.items()
    }
    out["det_fail_query"] = float(scenario.n_query_failed)
    return out
