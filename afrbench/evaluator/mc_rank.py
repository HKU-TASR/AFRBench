"""Rank scenarios that share one query matrix against a single MillionCelebs pass."""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

import torch

from afrbench.evaluator.metrics.ranking import (
    DEFAULT_EXTRA_CHUNK,
    ExtraRankState,
    extra_rank_states,
)


def query_digest(query_embeddings: torch.Tensor) -> str:
    """Stable identity of a query matrix for grouping extra streams."""
    tensor = query_embeddings.detach().to(dtype=torch.float32).cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def extra_states_for_own_sims(
    query_embeddings: torch.Tensor,
    extra_blocks: Iterable,
    own_sims_list: Sequence[torch.Tensor],
    *,
    top_k: int = 10,
    device: torch.device | str = "cpu",
    chunk: int = DEFAULT_EXTRA_CHUNK,
) -> List[ExtraRankState]:
    """One :class:`ExtraRankState` per ``own_sims`` tensor, from one ``extra_blocks`` pass."""
    return extra_rank_states(
        query_embeddings,
        extra_blocks,
        own_sims_list,
        top_k=top_k,
        device=device,
        chunk=chunk,
    )
