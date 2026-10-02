"""One shared ranking pass over a scenario, on a torch device.

Reference Code: https://github.com/HKU-TASR/Protego (its ``retrieve`` scoring)
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

#: Gallery rows per chunk; bounds the (Q, chunk) similarity block.
DEFAULT_CHUNK = 65_536

#: Extra-gallery rows per GEMM; at Q ~ 2500 the ``(Q, B)`` score block is ~2.6 GB.
DEFAULT_EXTRA_CHUNK = 262_144

#: Maximum boolean elements materialized while comparing a gallery block with
#: one relevant-item similarity. Query sub-batching keeps this bounded.
_MAX_RANK_COMPARE_ELEMENTS = 64 * 1024 * 1024

ExtraBlock = Union[
    Tuple[torch.Tensor, Sequence[str]],
    Tuple[torch.Tensor, Sequence[str], Sequence[str]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


@dataclasses.dataclass
class Ranking:
    """The per-query ranking facts every metric derives from."""

    ranks: List[torch.Tensor]  # per query, sorted 1-based ranks of its relevant items
    top_labels: List[List[str]]  # per query, the gallery's top-K labels
    top_sims: torch.Tensor  # (Q, K) similarities of the top-K
    gallery_size: int
    top_keys: List[List[Optional[str]]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ExtraRankState:
    """The extra-gallery (MillionCelebs) part of a ranking, for one ``own_sims`` tensor."""

    greater: torch.Tensor  # (Q, R)
    top_sims: torch.Tensor  # (Q, K)
    top_labels: List[List[str]]
    top_keys: List[List[Optional[str]]]
    n_extra: int


@dataclasses.dataclass
class _PrimaryState:
    queries: torch.Tensor
    n_queries: int
    n_primary: int
    k: int
    own_sims: torch.Tensor
    relevant_counts: List[int]
    max_relevant: int
    greater: torch.Tensor
    top_sims: torch.Tensor
    top_labels: List[List[str]]
    top_keys: List[List[Optional[str]]]
    id_to_label: Dict[int, str]


def _is_int_sidecar(value: Any) -> bool:
    """True when the extra-block sidecar is identity codes, not label strings."""
    if torch.is_tensor(value):
        return not value.is_floating_point()
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    return kind in ("i", "u")


def _unpack_extra(item: ExtraBlock) -> Tuple[torch.Tensor, Sequence[str], Optional[Sequence[str]]]:
    if len(item) == 3 and not _is_int_sidecar(item[1]):
        return item[0], item[1], item[2]
    if len(item) == 2:
        return item[0], item[1], None
    raise TypeError("string extra-blocks are (embeddings, labels) or (embeddings, labels, keys)")


def _h2d(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    if device.type == "cuda" and tensor.device.type == "cpu":
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=True)
    return tensor.to(device)


def _as_index(value: Any, device: torch.device) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return _h2d(tensor.to(dtype=torch.long), device)


def _prepare_extra_item(
    item: ExtraBlock,
    device: torch.device,
    intern_label,
    intern_key,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Embeddings plus integer ids. Labels/keys stay off the GEMM path."""
    keep: Optional[torch.Tensor] = None
    if len(item) in (3, 4) and _is_int_sidecar(item[1]):
        embeddings, codes, index = item[0], item[1], item[2]
        if len(item) == 4:
            keep = _h2d(torch.as_tensor(item[3]).to(dtype=torch.bool), device)
        tensor = embeddings if torch.is_tensor(embeddings) else torch.as_tensor(embeddings)
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        tensor = _h2d(tensor, device)
        if codes.shape[0] != tensor.shape[0] or index.shape[0] != tensor.shape[0]:
            raise ValueError("extra-block codes and indices must align with embeddings")
        if keep is not None and keep.shape[0] != tensor.shape[0]:
            raise ValueError("extra-block keep mask must align with embeddings")
        return tensor, _as_index(codes, device), _as_index(index, device), keep
    tensor, labels, keys = _unpack_extra(item)
    tensor = tensor if torch.is_tensor(tensor) else torch.as_tensor(tensor)
    tensor = _h2d(tensor.to(dtype=torch.float32), device)
    if keys is not None and len(keys) != tensor.shape[0]:
        raise ValueError("extra-block keys must align with embeddings")
    if len(labels) != tensor.shape[0]:
        raise ValueError("extra-block labels must align with embeddings")
    codes = torch.tensor(
        [intern_label(str(label)) for label in labels], dtype=torch.long
    )
    if keys is None:
        index = torch.full((tensor.shape[0],), -1, dtype=torch.long)
    else:
        index = torch.tensor(
            [-1 if key is None else intern_key(str(key)) for key in keys],
            dtype=torch.long,
        )
    return tensor, _h2d(codes, device), _h2d(index, device), None


def _iter_prepared_blocks(
    extra_blocks: Iterable[ExtraBlock],
    device: torch.device,
    intern_label,
    intern_key,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Move each extra block to ``device`` on the default stream, so the copy is ordered before
    the matmul; prefetching into a reused pinned buffer would race it and corrupt the counts.
    """
    for item in extra_blocks:
        yield _prepare_extra_item(item, device, intern_label, intern_key)


def _count_greater(
    sims: torch.Tensor,
    own_sims: torch.Tensor,
    greater: torch.Tensor,
) -> None:
    """Add ``#{block items strictly above own_sim[q, r]}`` into ``greater``."""
    n_queries, n_block = sims.shape
    max_relevant = own_sims.shape[1]
    if not max_relevant or n_block == 0:
        return
    query_step = max(1, min(n_queries, _MAX_RANK_COMPARE_ELEMENTS // max(1, n_block)))
    for query_start in range(0, n_queries, query_step):
        query_end = min(query_start + query_step, n_queries)
        sl = slice(query_start, query_end)
        block = sims[sl]
        for rank in range(max_relevant):
            greater[sl, rank] += (block > own_sims[sl, rank : rank + 1]).sum(dim=1)


def _merge_running_topk(
    top_sims: torch.Tensor,
    top_codes: torch.Tensor,
    top_index: torch.Tensor,
    sims: torch.Tensor,
    codes: torch.Tensor,
    index: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold one block into the running top-K: take the block's best ``K``, then merge with the
    running buffers, so no ``(Q, B)`` label tensor is ever built.
    """
    if k == 0 or sims.shape[1] == 0:
        return top_sims, top_codes, top_index
    take = min(k, sims.shape[1])
    block_sims, order = sims.topk(take, dim=1)
    combined_sims = torch.cat([top_sims, block_sims], dim=1)
    combined_codes = torch.cat([top_codes, codes[order]], dim=1)
    combined_index = torch.cat([top_index, index[order]], dim=1)
    top_sims, pick = combined_sims.topk(k, dim=1)
    return top_sims, combined_codes.gather(1, pick), combined_index.gather(1, pick)


def _add_greater_from_block(
    sims: torch.Tensor,
    own_list: Sequence[torch.Tensor],
    greater_list: Sequence[torch.Tensor],
) -> None:
    """Add strictly-greater counts for every ``own_sims`` against one block, with one sort plus
    ``searchsorted`` shared by all of them; small blocks are counted directly.
    """
    if not own_list or sims.numel() == 0:
        return
    n_queries, n_block = sims.shape
    if n_block < 4096 or n_queries < 8:
        for own_sims, greater in zip(own_list, greater_list):
            _count_greater(sims, own_sims, greater)
        return
    sorted_sims, _ = torch.sort(sims, dim=1)
    max_relevant = max(own_sims.shape[1] for own_sims in own_list)
    stacked = sims.new_full((n_queries, len(own_list) * max_relevant), float("inf"))
    for unit, own_sims in enumerate(own_list):
        width = own_sims.shape[1]
        if width:
            stacked[:, unit * max_relevant : unit * max_relevant + width] = own_sims
    counts = n_block - torch.searchsorted(sorted_sims, stacked, right=True)
    for unit, (own_sims, greater) in enumerate(zip(own_list, greater_list)):
        width = own_sims.shape[1]
        if width:
            greater += counts[:, unit * max_relevant : unit * max_relevant + width]


def _prepare_primary(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    top_k: int,
    device: torch.device,
    chunk: int = DEFAULT_CHUNK,
) -> _PrimaryState:
    queries = query_embeddings.to(device, torch.float32)
    gallery = gallery_embeddings.to(device, torch.float32)
    n_queries = queries.shape[0]
    n_primary = gallery.shape[0]
    k = max(0, int(top_k))

    label_ids: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}

    def intern_label(label: str) -> int:
        existing = label_ids.get(label)
        if existing is not None:
            return existing
        index = len(label_ids)
        label_ids[label] = index
        id_to_label[index] = label
        return index

    for label in list(gallery_labels) + list(query_labels):
        intern_label(label)
    gallery_label_id = torch.tensor(
        [label_ids[label] for label in gallery_labels], dtype=torch.long, device=device
    )
    query_label_id = torch.tensor(
        [label_ids[label] for label in query_labels], dtype=torch.long, device=device
    )

    relevant_columns: List[torch.Tensor] = []
    relevant_counts: List[int] = []
    max_relevant = 0
    for i in range(n_queries):
        columns = torch.nonzero(
            gallery_label_id == query_label_id[i], as_tuple=False
        ).flatten()
        relevant_columns.append(columns)
        relevant_counts.append(int(columns.numel()))
        max_relevant = max(max_relevant, int(columns.numel()))

    own_sims = torch.full(
        (n_queries, max_relevant), float("inf"), dtype=torch.float32, device=device
    )
    for i, columns in enumerate(relevant_columns):
        if columns.numel():
            own_sims[i, : columns.numel()] = gallery[columns] @ queries[i]

    greater = torch.zeros((n_queries, max_relevant), dtype=torch.long, device=device)
    if k == 0:
        top_sims = torch.zeros((n_queries, 0), device=device)
    else:
        top_sims = torch.full((n_queries, k), -float("inf"), device=device)
    top_label_ids = torch.full((n_queries, max(k, 1)), -1, dtype=torch.long, device=device)
    top_keys: List[List[Optional[str]]] = [[None] * k for _ in range(n_queries)]

    def _consume_primary(block: torch.Tensor, block_labels: torch.Tensor, start: int) -> None:
        nonlocal top_sims, top_label_ids
        if block.shape[0] == 0:
            return
        sims = queries @ block.T
        if start < n_primary:
            for i, columns in enumerate(relevant_columns):
                in_block = (columns >= start) & (columns < start + block.shape[0])
                if in_block.any():
                    positions = torch.nonzero(in_block, as_tuple=False).flatten()
                    local_columns = columns[positions] - start
                    sims[i, local_columns] = own_sims[i, positions]
        _count_greater(sims, own_sims, greater)
        if k == 0:
            return
        combined_sims = torch.cat([top_sims, sims], dim=1)
        combined_labels = torch.cat(
            [top_label_ids[:, :k], block_labels.unsqueeze(0).expand(n_queries, -1)],
            dim=1,
        )
        top_sims, order = combined_sims.topk(k, dim=1)
        top_label_ids = combined_labels.gather(1, order)

    for start in range(0, n_primary, chunk):
        sl = slice(start, start + chunk)
        _consume_primary(gallery[sl], gallery_label_id[sl], start)

    if k == 0:
        top_labels: List[List[str]] = [[] for _ in range(n_queries)]
    else:
        top_labels = [
            [id_to_label[int(index)] for index in row.tolist() if index >= 0]
            for row in top_label_ids[:, :k]
        ]
    return _PrimaryState(
        queries=queries,
        n_queries=n_queries,
        n_primary=n_primary,
        k=k,
        own_sims=own_sims,
        relevant_counts=relevant_counts,
        max_relevant=max_relevant,
        greater=greater,
        top_sims=top_sims.cpu() if k else top_sims,
        top_labels=top_labels,
        top_keys=top_keys,
        id_to_label=id_to_label,
    )


def _merge_topk(
    left_sims: torch.Tensor,
    left_labels: List[List[str]],
    left_keys: List[List[Optional[str]]],
    right_sims: torch.Tensor,
    right_labels: List[List[str]],
    right_keys: List[List[Optional[str]]],
    k: int,
) -> Tuple[torch.Tensor, List[List[str]], List[List[Optional[str]]]]:
    if k == 0:
        n = left_sims.shape[0]
        return torch.zeros((n, 0)), [[] for _ in range(n)], [[] for _ in range(n)]
    combined = torch.cat([left_sims.to(right_sims.device), right_sims], dim=1)
    top_sims, order = combined.topk(k, dim=1)
    labels: List[List[str]] = []
    keys: List[List[Optional[str]]] = []
    left_k = left_sims.shape[1]
    for q, row in enumerate(order.tolist()):
        chosen_labels: List[str] = []
        chosen_keys: List[Optional[str]] = []
        for index in row:
            if index < left_k:
                if index < len(left_labels[q]):
                    chosen_labels.append(left_labels[q][index])
                    chosen_keys.append(left_keys[q][index] if index < len(left_keys[q]) else None)
            else:
                extra = index - left_k
                if extra < len(right_labels[q]):
                    chosen_labels.append(right_labels[q][extra])
                    chosen_keys.append(right_keys[q][extra] if extra < len(right_keys[q]) else None)
        labels.append(chosen_labels)
        keys.append(chosen_keys)
    return top_sims.cpu(), labels, keys


def _finish(primary: _PrimaryState, extra: Optional[ExtraRankState], n_extra: int) -> Ranking:
    greater = primary.greater
    if extra is not None and extra.n_extra:
        greater = greater + extra.greater.to(greater.device)
        top_sims, top_labels, top_keys = _merge_topk(
            primary.top_sims,
            primary.top_labels,
            primary.top_keys,
            extra.top_sims,
            extra.top_labels,
            extra.top_keys,
            primary.k,
        )
        n_extra = extra.n_extra
    else:
        top_sims = primary.top_sims if primary.k else torch.zeros((primary.n_queries, 0))
        top_labels = primary.top_labels
        top_keys = primary.top_keys
    ranks = [
        torch.sort(1 + greater[i, :count]).values.cpu()
        if count else torch.zeros(0, dtype=torch.long)
        for i, count in enumerate(primary.relevant_counts)
    ]
    return Ranking(
        ranks=ranks,
        top_labels=top_labels,
        top_sims=top_sims.cpu() if top_sims.device.type != "cpu" else top_sims,
        gallery_size=primary.n_primary + n_extra,
        top_keys=top_keys,
    )


def extra_rank_states(
    query_embeddings: torch.Tensor,
    extra_blocks: Iterable[ExtraBlock],
    own_sims_list: Sequence[torch.Tensor],
    *,
    top_k: int = 10,
    device: torch.device | str = "cpu",
    chunk: int = DEFAULT_EXTRA_CHUNK,
) -> List[ExtraRankState]:
    """One pass over ``extra_blocks`` giving one state per ``own_sims`` tensor, so the scenarios
    that share a query matrix also share the pass; labels are resolved only for the winners.
    """
    device = torch.device(device)
    queries = query_embeddings.to(device, torch.float32)
    n_queries = queries.shape[0]
    k = max(0, int(top_k))
    own = [tensor.to(device, torch.float32) for tensor in own_sims_list]
    greaters = [
        torch.zeros(tensor.shape, dtype=torch.long, device=device) for tensor in own
    ]
    if k == 0:
        top_sims = torch.zeros((n_queries, 0), device=device)
        top_codes = torch.zeros((n_queries, 0), dtype=torch.long, device=device)
        top_index = torch.zeros((n_queries, 0), dtype=torch.long, device=device)
    else:
        top_sims = torch.full((n_queries, k), -float("inf"), device=device)
        top_codes = torch.full((n_queries, k), -1, dtype=torch.long, device=device)
        top_index = torch.full((n_queries, k), -1, dtype=torch.long, device=device)
    n_extra = 0
    label_ids: Dict[str, int] = {}
    id_to_label: Dict[int, str] = {}
    key_ids: Dict[str, int] = {}
    id_to_key: Dict[int, str] = {}

    def intern_label(label: str) -> int:
        existing = label_ids.get(label)
        if existing is not None:
            return existing
        index = len(label_ids)
        label_ids[label] = index
        id_to_label[index] = label
        return index

    def intern_key(key: str) -> int:
        existing = key_ids.get(key)
        if existing is not None:
            return existing
        index = len(key_ids)
        key_ids[key] = index
        id_to_key[index] = key
        return index

    for tensor, codes, index, keep in _iter_prepared_blocks(
        extra_blocks, device, intern_label, intern_key
    ):
        if tensor.shape[0] == 0:
            continue
        for start in range(0, tensor.shape[0], chunk):
            sl = slice(start, start + chunk)
            block = tensor[sl]
            if block.shape[0] == 0:
                continue
            sims = queries @ block.T
            block_keep = None if keep is None else keep[sl]
            if block_keep is not None:
                sims.masked_fill_(~block_keep.unsqueeze(0), float("-inf"))
            _add_greater_from_block(sims, own, greaters)
            if k == 0:
                continue
            top_sims, top_codes, top_index = _merge_running_topk(
                top_sims, top_codes, top_index, sims, codes[sl], index[sl], k
            )
        n_extra += (
            int(keep.sum().item()) if keep is not None else int(tensor.shape[0])
        )

    if k == 0:
        top_labels: List[List[str]] = [[] for _ in range(n_queries)]
        top_keys: List[List[Optional[str]]] = [[] for _ in range(n_queries)]
    elif hasattr(extra_blocks, "finish_topk"):
        top_labels, top_keys = extra_blocks.finish_topk(top_codes, top_index)
    else:
        top_labels = []
        top_keys = []
        for codes_row, index_row in zip(
            top_codes.detach().cpu().tolist(), top_index.detach().cpu().tolist()
        ):
            labels_row: List[str] = []
            keys_row: List[Optional[str]] = []
            for code, key_id in zip(codes_row, index_row):
                if code < 0:
                    continue
                labels_row.append(id_to_label[int(code)])
                keys_row.append(None if key_id < 0 else id_to_key[int(key_id)])
            top_labels.append(labels_row)
            top_keys.append(keys_row)

    top_sims_cpu = top_sims.cpu()
    states = []
    for greater in greaters:
        states.append(
            ExtraRankState(
                greater=greater,
                top_sims=top_sims_cpu,
                top_labels=[list(row) for row in top_labels],
                top_keys=[list(row) for row in top_keys],
                n_extra=n_extra,
            )
        )
    return states


def compute_ranking(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: Sequence[str],
    gallery_labels: Sequence[str],
    *,
    top_k: int = 10,
    device: torch.device | str = "cpu",
    chunk: int = DEFAULT_CHUNK,
    extra_blocks: Optional[Iterable[ExtraBlock]] = None,
    extra_state: Optional[ExtraRankState] = None,
) -> Ranking:
    """Rank each query's relevant gallery items and keep the gallery top-K, over L2-normalized
    embeddings.

    Args:
        query_embeddings: The queries.
        gallery_embeddings: The protectee gallery, the only source of relevant items.
        query_labels: Identity of each query.
        gallery_labels: Identity of each gallery row.
        top_k: Gallery entries kept per query.
        device: Device to score on.
        chunk: Gallery rows per matmul block.
        extra_blocks: Further gallery rows, streamed rather than concatenated.
        extra_state: A stream already scored against these queries, to reuse.
    """
    device = torch.device(device)
    primary = _prepare_primary(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        top_k=top_k,
        device=device,
        chunk=chunk,
    )
    if extra_state is not None:
        return _finish(primary, extra_state, extra_state.n_extra)
    if extra_blocks is None:
        return _finish(primary, None, 0)
    states = extra_rank_states(
        primary.queries,
        extra_blocks,
        [primary.own_sims],
        top_k=primary.k,
        device=device,
    )
    return _finish(primary, states[0], states[0].n_extra)
