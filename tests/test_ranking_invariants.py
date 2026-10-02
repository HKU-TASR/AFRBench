"""Correctness and performance invariants of the dodging ranking path."""
from __future__ import annotations

import math
import os
import time

import pytest
import torch

from afrbench.evaluator.metrics.identification import per_query as id_per_query
from afrbench.evaluator.metrics.ranking import (
    _add_greater_from_block,
    _count_greater,
    _merge_running_topk,
    compute_ranking,
    extra_rank_states,
)
from afrbench.evaluator.metrics.retrieval import (
    average_precision_per_query,
    ndcg_per_query,
    recall_per_query,
)

def _unit(cosine: float) -> list[float]:
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine**2))]


def test_equal_similarity_does_not_outrank():
    """A gallery item tied with a relevant item does not rank ahead of it."""
    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([_unit(0.8), _unit(0.8), _unit(0.1)])
    ranking = compute_ranking(
        queries, gallery, ["user"], ["user", "noise", "noise"], top_k=3
    )
    assert ranking.ranks[0].tolist() == [1]


def test_inf_padded_own_sims_add_zero_greater():
    sims = torch.randn(4, 64)
    own = torch.full((4, 3), float("inf"))
    own[:, 0] = 0.0
    greater = torch.zeros(4, 3, dtype=torch.long)
    _add_greater_from_block(sims, [own], [greater])
    assert torch.equal(greater[:, 1:], torch.zeros(4, 2, dtype=torch.long))
    assert torch.equal(greater[:, 0], (sims > 0).sum(dim=1))


def test_shared_extra_stream_matches_independent_passes():
    generator = torch.Generator().manual_seed(11)
    queries = torch.nn.functional.normalize(torch.randn(6, 16, generator=generator), dim=1)
    extra = torch.nn.functional.normalize(torch.randn(80, 16, generator=generator), dim=1)
    labels = [f"noise/{i % 7}" for i in range(80)]
    keys = [f"k{i}" for i in range(80)]
    blocks = [(extra, labels, keys)]
    galleries = [
        torch.nn.functional.normalize(torch.randn(5, 16, generator=generator), dim=1)
        for _ in range(4)
    ]
    q_labels = ["u"] * 6
    g_labels = ["u", "u", "v", "v", "w"]
    independent = [
        compute_ranking(queries, gallery, q_labels, g_labels, top_k=5, extra_blocks=list(blocks))
        for gallery in galleries
    ]
    from afrbench.evaluator.metrics.ranking import _prepare_primary

    owns = [
        _prepare_primary(
            queries, gallery, q_labels, g_labels, top_k=5, device=torch.device("cpu")
        ).own_sims
        for gallery in galleries
    ]
    states = extra_rank_states(queries, blocks, owns, top_k=5)
    shared = [
        compute_ranking(
            queries, gallery, q_labels, g_labels, top_k=5, extra_state=state
        )
        for gallery, state in zip(galleries, states)
    ]
    for left, right in zip(independent, shared):
        for i in range(len(left.ranks)):
            assert torch.equal(left.ranks[i], right.ranks[i])
        assert left.gallery_size == right.gallery_size
        assert left.top_labels == right.top_labels
        assert torch.allclose(left.top_sims, right.top_sims)


def test_keep_mask_string_and_compact_agree_on_metrics():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([_unit(0.05), _unit(0.04)])
    extras = torch.tensor([
        _unit(0.95),
        _unit(0.20),
        _unit(0.85),
        _unit(0.10),
        _unit(0.70),
    ])
    keep = torch.tensor([True, False, True, True, False])
    compact_idx = keep.nonzero(as_tuple=False).flatten()
    string_blocks = [(
        extras[compact_idx],
        [f"noise/{i}" for i in compact_idx.tolist()],
        [f"k{i}" for i in compact_idx.tolist()],
    )]
    class _Codes:
        def __init__(self, blocks):
            self._blocks = blocks

        def __iter__(self):
            return iter(self._blocks)

        def finish_topk(self, codes, index):
            labels = []
            keys = []
            for row_c, row_i in zip(codes.tolist(), index.tolist()):
                labels.append([f"noise/{c}" for c, i in zip(row_c, row_i) if i >= 0])
                keys.append([None] * len(labels[-1]))
            return labels, keys

    compact_blocks = _Codes([(
        extras[compact_idx],
        compact_idx.to(torch.int64),
        compact_idx.to(torch.int64),
    )])
    masked_blocks = _Codes([(
        extras,
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64),
        keep,
    )])
    string = compute_ranking(
        queries, gallery, ["a", "b"], ["a", "b"], top_k=3, extra_blocks=string_blocks
    )
    compact = compute_ranking(
        queries, gallery, ["a", "b"], ["a", "b"], top_k=3, extra_blocks=compact_blocks
    )
    masked = compute_ranking(
        queries, gallery, ["a", "b"], ["a", "b"], top_k=3, extra_blocks=masked_blocks
    )
    for index in range(2):
        assert torch.equal(string.ranks[index], compact.ranks[index])
        assert torch.equal(string.ranks[index], masked.ranks[index])
    assert string.gallery_size == compact.gallery_size == masked.gallery_size == 2 + int(keep.sum())
    assert string.top_labels == compact.top_labels == masked.top_labels


def test_searchsorted_counts_match_loop_with_mixed_widths():
    generator = torch.Generator().manual_seed(21)
    sims = torch.randn(16, 8192, generator=generator)
    owns = [
        torch.randn(16, 7, generator=generator),
        torch.randn(16, 1, generator=generator),
        torch.full((16, 4), float("inf")),
    ]
    owns[2][:, 0] = torch.randn(16, generator=generator)
    got = [torch.zeros(own.shape, dtype=torch.long) for own in owns]
    ref = [torch.zeros(own.shape, dtype=torch.long) for own in owns]
    _add_greater_from_block(sims, owns, got)
    for own, greater in zip(owns, ref):
        _count_greater(sims, own, greater)
    for left, right in zip(got, ref):
        assert torch.equal(left, right)


def test_merge_running_topk_matches_expanded_cat():
    generator = torch.Generator().manual_seed(5)
    q, b, k = 9, 400, 10
    sims = torch.randn(q, b, generator=generator)
    codes = torch.arange(b)
    index = torch.arange(1000, 1000 + b)
    top_sims = torch.full((q, k), -float("inf"))
    top_codes = torch.full((q, k), -1, dtype=torch.long)
    top_index = torch.full((q, k), -1, dtype=torch.long)
    merged = _merge_running_topk(top_sims, top_codes, top_index, sims, codes, index, k)
    combined = torch.cat([top_sims, sims], dim=1)
    expanded_codes = torch.cat([top_codes, codes.unsqueeze(0).expand(q, -1)], dim=1)
    expanded_index = torch.cat([top_index, index.unsqueeze(0).expand(q, -1)], dim=1)
    ref_sims, order = combined.topk(k, dim=1)
    assert torch.equal(merged[0], ref_sims)
    assert torch.equal(merged[1], expanded_codes.gather(1, order))
    assert torch.equal(merged[2], expanded_index.gather(1, order))


def test_retrieval_and_identification_are_closed_forms_of_ranks():
    ranks = [torch.tensor([1, 4, 25])]
    own = [3]
    top_labels = [["user", "noise", "user", "other", "user"]]
    from afrbench.evaluator.metrics.ranking import Ranking

    ranking = Ranking(
        ranks=ranks,
        top_labels=top_labels,
        top_sims=torch.zeros(1, 5),
        gallery_size=100,
    )
    scenario = type("S", (), {"query_labels": ["user"], "n_query_failed": 0})()
    assert recall_per_query(ranking, 25, own) == [1.0]
    assert recall_per_query(ranking, "user", own) == pytest.approx([1.0 / 3.0])
    assert average_precision_per_query(ranking, k=100, intended=own)[0] == pytest.approx(
        (1 / 1 + 2 / 4 + 3 / 25) / 3
    )
    assert ndcg_per_query(ranking, 10, own)[0] > 0
    ident = id_per_query(ranking, scenario, ks=(1, 5))
    assert ident["rank1_appearance"] == [1.0]
    assert ident["rank1_plurality"] == [1.0]
    assert ident["rank5_majority"] == [1.0]
    assert all(not name.startswith("target_") for name in ident)


@pytest.mark.slow
def test_sort_count_equals_per_own_loop_on_wide_blocks():
    """The sort count matches the per-own loop and is faster unless OpenMP is single-threaded."""
    generator = torch.Generator().manual_seed(1)
    sims = torch.randn(32, 16384, generator=generator)
    owns = [torch.randn(32, 8, generator=generator) for _ in range(12)]

    # Minimum over repeats: host contention can only inflate a timing.
    def _best_of(run, repeats=5):
        best, out = float("inf"), None
        for _ in range(repeats):
            greater = [torch.zeros(own.shape, dtype=torch.long) for own in owns]
            t0 = time.perf_counter()
            run(greater)
            best = min(best, time.perf_counter() - t0)
            out = greater
        return best, out

    sort_s, got = _best_of(
        lambda greater: _add_greater_from_block(sims, owns, greater)
    )
    loop_s, ref = _best_of(
        lambda greater: [
            _count_greater(sims, own, slot) for own, slot in zip(owns, greater)
        ]
    )
    for left, right in zip(got, ref):
        assert torch.equal(left, right)
    if os.environ.get("OMP_NUM_THREADS") == "1":
        # torch.set_num_threads cannot widen an OpenMP pool capped before it was
        # built (e.g. by pin_cpu_threads), and single-threaded the sort is slower.
        pytest.skip("OpenMP pinned to one thread; the sort advantage needs threads")
    assert sort_s < loop_s


@pytest.mark.slow
def test_masked_fill_is_faster_than_fancy_index_gather():
    generator = torch.Generator().manual_seed(2)
    block = torch.randn(65536, 64, generator=generator)
    keep = torch.ones(65536, dtype=torch.bool)
    keep[::50] = False
    t0 = time.perf_counter()
    compact = block[keep]
    gather_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    masked = block.clone()
    masked[~keep] = 0
    mask_s = time.perf_counter() - t0
    assert compact.shape[0] == int(keep.sum())
    assert masked.shape == block.shape
    assert mask_s < gather_s * 4 or gather_s < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ranking check")
def test_cuda_extra_rank_states_twice_is_bitwise_equal():
    """Two CUDA ``extra_rank_states`` passes over the same blocks are bitwise equal."""
    generator = torch.Generator().manual_seed(19)
    queries = torch.nn.functional.normalize(
        torch.randn(12, 64, generator=generator), dim=1
    )
    owns = [
        torch.rand(12, 4, generator=generator) * 0.4 + 0.2,
        torch.rand(12, 4, generator=generator) * 0.4 + 0.2,
    ]
    raw = []
    for _ in range(4):
        extra = torch.nn.functional.normalize(
            torch.randn(4096, 64, generator=generator), dim=1
        )
        n = extra.shape[0]
        raw.append((
            extra,
            torch.arange(n, dtype=torch.int64),
            torch.arange(n, dtype=torch.int64),
            torch.ones(n, dtype=torch.bool),
        ))

    class _Codes:
        def __init__(self, blocks):
            self._blocks = blocks

        def __iter__(self):
            return iter(self._blocks)

        def finish_topk(self, codes, index):
            labels, keys = [], []
            for row_c, row_i in zip(codes.tolist(), index.tolist()):
                labels.append([f"noise/{c}" for c, i in zip(row_c, row_i) if i >= 0])
                keys.append([None] * len(labels[-1]))
            return labels, keys

    first = extra_rank_states(queries, _Codes(raw), owns, top_k=5, device="cuda")
    second = extra_rank_states(queries, _Codes(raw), owns, top_k=5, device="cuda")
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert torch.equal(left.greater, right.greater)
        assert torch.equal(left.top_sims, right.top_sims)
        assert left.n_extra == right.n_extra


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ranking check")
def test_cpu_and_cuda_ranks_match_for_extra_blocks():
    generator = torch.Generator().manual_seed(3)
    queries = torch.nn.functional.normalize(torch.randn(8, 32, generator=generator), dim=1)
    gallery = torch.nn.functional.normalize(torch.randn(12, 32, generator=generator), dim=1)
    extra = torch.nn.functional.normalize(torch.randn(2000, 32, generator=generator), dim=1)
    keep = torch.ones(2000, dtype=torch.bool)
    keep[::17] = False

    class _Codes:
        def __init__(self, blocks):
            self._blocks = blocks

        def __iter__(self):
            return iter(self._blocks)

        def finish_topk(self, codes, index):
            labels, keys = [], []
            for row_c, row_i in zip(codes.tolist(), index.tolist()):
                labels.append([f"noise/{c}" for c, i in zip(row_c, row_i) if i >= 0])
                keys.append([None] * len(labels[-1]))
            return labels, keys

    blocks = _Codes([(
        extra,
        torch.arange(2000, dtype=torch.int64),
        torch.arange(2000, dtype=torch.int64),
        keep,
    )])
    labels_q = ["u"] * 8
    labels_g = ["u"] * 6 + ["v"] * 6
    cpu = compute_ranking(
        queries, gallery, labels_q, labels_g, top_k=5, extra_blocks=blocks, device="cpu"
    )
    gpu = compute_ranking(
        queries, gallery, labels_q, labels_g, top_k=5, extra_blocks=blocks, device="cuda:0"
    )
    for i in range(8):
        assert torch.equal(cpu.ranks[i], gpu.ranks[i])
    assert cpu.gallery_size == gpu.gallery_size
    assert cpu.top_labels == gpu.top_labels
