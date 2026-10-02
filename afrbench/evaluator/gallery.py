"""Assemble the AFRBench scenarios from cached embeddings."""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from afrbench.dataset import BenchDataset
from afrbench.evaluator.metrics.ranking import DEFAULT_EXTRA_CHUNK
from afrbench.evaluator.store import EmbeddingSet, EmbeddingStore

#: Embedding-source names in the cache.
ORIGINAL_SET = "originals"
NOISE_SET = "noise"
TARGET_SET = "targets"
MIX_RS = tuple(round(r * 0.1, 1) for r in range(11))


def protected_set_name(run_key: str, digest: str) -> str:
    """Content-addressed embedding-store key for one protect run."""
    short = digest[:12] if digest else "missing"
    return f"protected__{run_key}@{short}"


@dataclasses.dataclass
class Scenario:
    """A resolved gallery and query pair of embedding matrices, ready to score.

    Attributes:
        n_query_failed: Queries whose face was not detected, listed in ``failed_query_ids``.
        intended_relevant: Each protectee's gallery size before detection failures, which is
            the Recall@K denominator.
        extra_block_iter: Streams the full distractor gallery, when set.
    """

    name: str
    gallery_embeddings: torch.Tensor  # protectee gallery (G, D) float32
    gallery_labels: List[str]
    gallery_ids: List[str]
    query_embeddings: torch.Tensor  # (Q, D) float32
    query_labels: List[str]
    query_ids: List[str]
    n_query_failed: int = 0
    n_gallery_failed: int = 0
    failed_query_ids: List[str] = dataclasses.field(default_factory=list)
    intended_relevant: Dict[str, int] = dataclasses.field(default_factory=dict)
    intended_query_ids: List[str] = dataclasses.field(default_factory=list)
    extra_blocks: List[Tuple[torch.Tensor, List[str]]] = dataclasses.field(default_factory=list)
    extra_block_iter: Optional[Any] = None
    noise_image_count: int = 0
    noise_identity_count: int = 0

    def iter_extra_blocks(self):
        if self.extra_block_iter is not None:
            return self.extra_block_iter()
        return self.extra_blocks


def frozen_roles(
    dataset: BenchDataset,
    eval_lists: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Frozen ``(gallery, query)`` paths per protectee, limited to ``eval_lists`` if given."""
    split: Dict[str, Tuple[List[str], List[str]]] = {}
    names = (
        list(eval_lists)
        if eval_lists is not None
        else dataset.protectee_identities()
    )
    for protectee in names:
        roles = dataset.frozen_roles(protectee)
        gallery = list(roles["gallery"])
        query = list(roles["query"])
        if eval_lists is not None:
            allowed = set(eval_lists[protectee])
            gallery = [path for path in gallery if path in allowed]
            query = [path for path in query if path in allowed]
        split[protectee] = (gallery, query)
    return split


def per_user_split(*_args, **_kwargs):
    raise RuntimeError(
        "per_user_split was removed; scenarios consume frozen album/query/gallery roles"
    )


def _row_indices(
    embedding_set: EmbeddingSet, ids: Sequence[str]
) -> Tuple[List[int], List[str], List[str]]:
    """The store row of each requested id, dropping (and listing) failures."""
    index = embedding_set.row_index()
    failed = set(embedding_set.failed_ids)
    rows, kept, failed_ids = [], [], []
    for image_id in ids:
        if image_id in failed or image_id not in index:
            failed_ids.append(image_id)
            continue
        rows.append(index[image_id])
        kept.append(image_id)
    return rows, kept, failed_ids


def _gather(embedding_set: EmbeddingSet, rows: Sequence[int]) -> torch.Tensor:
    if not rows:
        return torch.zeros((0, 1), dtype=torch.float32)
    import numpy as np

    return torch.from_numpy(np.asarray(embedding_set.embeddings)[rows])


_READER_CACHE: Dict[str, Any] = {}


def _full_noise_reader(dataset: BenchDataset, system):
    """Reader over the shared full-pack cache for ``system``'s gallery digest (memoized)."""
    from afrbench.evaluator.noise_cache import FullNoiseCache, cache_identity, shared_cache_root
    from afrbench.noise import physical_pack_id

    pack_id = physical_pack_id(dataset.noise_lmdb, dataset.noise_provenance)
    gallery = system.for_gallery()
    digest = cache_identity(
        pack_id=pack_id,
        fr_digest=gallery.digest,
        weight_identity=gallery.weight_identity,
    )
    root = shared_cache_root(dataset.root) / digest
    key = str(root)
    reader = _READER_CACHE.get(key)
    if reader is None:
        reader = FullNoiseCache(root).reader()
        _READER_CACHE[key] = reader
    return reader


@dataclasses.dataclass
class NoiseRankStream:
    """The purged distractor gallery as a stream of ``(embeddings, identity_codes,
    global_index[, keep_mask])`` blocks, whose labels are resolved only for the final top-K.
    """

    reader: Any
    excluded: List[str]
    chunk: int = DEFAULT_EXTRA_CHUNK

    def __call__(self) -> "NoiseRankStream":
        return self

    def __iter__(self):
        for item in self.reader.iter_rank_blocks(
            exclude_identities=self.excluded, chunk=self.chunk
        ):
            embeddings, codes, index = item[0], item[1], item[2]
            if embeddings.size == 0:
                continue
            payload = (
                torch.from_numpy(np.asarray(embeddings)),
                torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(index)),
            )
            if len(item) == 4 and not np.asarray(item[3]).all():
                yield payload + (torch.from_numpy(np.ascontiguousarray(item[3])),)
            else:
                yield payload

    def finish_topk(
        self, codes: torch.Tensor, index: torch.Tensor
    ) -> Tuple[List[List[str]], List[List[Optional[str]]]]:
        """Resolve top-K labels from identity codes, returning ``None`` keys, since identification
        needs only the identity; ``index`` is part of the ranking loop's interface.
        """
        del index
        vocab = self.reader.vocab
        labels: List[List[str]] = []
        keys: List[List[Optional[str]]] = []
        width = int(codes.shape[1]) if codes.ndim == 2 else 0
        for row in codes.detach().cpu().tolist():
            label_row: List[str] = []
            for col_i in range(width):
                code = int(row[col_i])
                if code < 0:
                    continue
                identity = vocab[code] if 0 <= code < len(vocab) else ""
                label_row.append(f"noise/{identity}" if identity else f"noise/{code}")
            labels.append(label_row)
            keys.append([None] * len(label_row))
        return labels, keys


def _full_noise_view(
    dataset: BenchDataset,
    system,
    exclude_identities: Optional[Sequence[str]],
    chunk: int = DEFAULT_EXTRA_CHUNK,
):
    """Lazy view over the full distractor cache, purged by the dataset's noise filter plus
    ``exclude_identities``; returns ``(stream, n_failed, kept_images, kept_identities)``.
    """
    reader = _full_noise_reader(dataset, system)
    excluded = {str(item) for item in exclude_identities or ()}
    excluded.update(str(item) for item in dataset.noise_filter.identities)
    kept_images, kept_identities = reader.kept_counts(sorted(excluded))
    stream = NoiseRankStream(reader, sorted(excluded), chunk)
    return stream, len(reader.failed), kept_images, kept_identities


def _distractor_rows(
    dataset: BenchDataset,
    store: EmbeddingStore,
    system_digest: str,
    count: int,
    exclude_identities: Optional[Sequence[str]],
    seed: int,
) -> Tuple[List[Tuple[EmbeddingSet, int]], List[str], List[str], int]:
    """A seeded distractor draw from a sampled noise view, which AFRBench-basic does not use.

    Raises:
        ValueError: When more distractors are requested than were embedded.
    """
    if count < 0:
        raise ValueError("sampled distractors cannot be negative; use mode=full")
    view_id = dataset.noise_view_id_for(exclude_identities)
    noise_set = store.get(
        NOISE_SET,
        system_digest,
        noise_view_id=view_id,
        noise_seed=seed,
        noise_count=count,
    )
    requested = sorted(set(noise_set.ids) | set(noise_set.failed_ids))
    if count > len(requested):
        raise ValueError(
            f"Scenario requests {count} distractors but only {len(requested)} "
            f"are embedded; raise the eval config's distractor cap."
        )
    keys = requested
    if exclude_identities:
        excluded = set(exclude_identities)
        leaked = [k for k in keys if k.split("__")[0] in excluded]
        if leaked:
            raise ValueError(
                f"noise embedding view contains {len(leaked)} excluded identities"
            )
    if count < len(keys):
        keys = sorted(random.Random(f"noise:{seed}").sample(keys, count))
    rows, kept, failed_ids = _row_indices(noise_set, keys)
    labels = [f"noise/{key.split('__')[0]}" for key in kept]
    return [(noise_set, row) for row in rows], kept, labels, len(failed_ids)


def build_scenario(
    kind: str,
    *,
    dataset: BenchDataset,
    store: EmbeddingStore,
    system_digest: str,
    protected_source: Optional[str],
    eval_lists: Dict[str, Sequence[str]],
    distractors: int,
    exclude_identities: Optional[Sequence[str]] = None,
    mix_r: Optional[float] = None,
    seed: int = 0,
    distractor_seed: Optional[int] = None,
    full_noise: bool = False,
    system=None,
) -> Scenario:
    """Resolve one scenario to embedding matrices.

    Args:
        kind: ``1A``, ``1B``, ``2A``, ``2B``, or ``mix-gallery``.
        dataset: The benchmark dataset.
        store: Cache the embeddings come from.
        system_digest: FR-system digest the embeddings are keyed by.
        protected_source: The protected run's store key, or ``None`` to put the originals in
            the protected role, which is the 1B baseline.
        eval_lists: Per-protectee eval images.
        distractors: Sampled distractors to draw, when not streaming the full gallery.
        exclude_identities: Distractor identities to leave out.
        mix_r: Protected fraction of the gallery, for ``mix-gallery``.
        seed: Seed of the mix choice, and the default distractor seed.
        distractor_seed: Seed the sampled distractors were embedded under.
        full_noise: Stream the complete purged gallery, which needs ``system``.
        system: The FR system, required by ``full_noise``.
    """
    originals = store.get(ORIGINAL_SET, system_digest)
    protected = (
        originals if protected_source is None else store.get(protected_source, system_digest)
    )

    # Query/gallery source per scenario: (query_set, gallery_set).
    query_is_protected = kind in ("2A", "2B", "mix-gallery")
    gallery_is_protected = kind in ("1A", "2A", "mix-gallery")
    query_set = protected if query_is_protected else originals
    gallery_set = protected if gallery_is_protected else originals

    split = frozen_roles(dataset, eval_lists)
    gallery_rows: List[Tuple[EmbeddingSet, int]] = []
    gallery_ids: List[str] = []
    gallery_labels: List[str] = []
    query_ids: List[str] = []
    query_labels: List[str] = []
    intended_query_ids: List[str] = []
    intended_relevant: Dict[str, int] = {}
    failed_query_ids: List[str] = []
    n_gallery_failed = 0
    n_query_failed = 0

    for user, (user_gallery, user_query) in split.items():
        intended_relevant[user] = len(user_gallery)
        intended_query_ids.extend(user_query)
        q_rows, kept_query, q_failed = _row_indices(query_set, user_query)
        n_query_failed += len(q_failed)
        failed_query_ids.extend(q_failed)
        query_ids.extend(kept_query)
        query_labels.extend([user] * len(kept_query))

        if kind == "mix-gallery" and mix_r is not None:
            clean_index = originals.row_index()
            prot_index = protected.row_index()
            clean_failed = set(originals.failed_ids)
            prot_failed = set(protected.failed_ids)
            for image_id in user_gallery:
                use_protected = (
                    random.Random(f"mix-gallery:{image_id}").random() < mix_r
                )
                source = protected if use_protected else originals
                index = prot_index if use_protected else clean_index
                failed = prot_failed if use_protected else clean_failed
                if image_id in failed or image_id not in index:
                    n_gallery_failed += 1
                    continue
                gallery_rows.append((source, index[image_id]))
                gallery_ids.append(image_id)
                gallery_labels.append(user)
        else:
            rows, kept, g_failed = _row_indices(gallery_set, user_gallery)
            n_gallery_failed += len(g_failed)
            gallery_rows.extend((gallery_set, row) for row in rows)
            gallery_ids.extend(kept)
            gallery_labels.extend([user] * len(kept))

    extra_blocks: List[Tuple[torch.Tensor, List[str]]] = []
    extra_block_iter = None
    noise_image_count = 0
    noise_identity_count = 0
    if full_noise:
        if system is None:
            raise ValueError("full MillionCelebs ranking needs the FR system identity")
        extra_block_iter, d_dropped, noise_image_count, noise_identity_count = _full_noise_view(
            dataset, system, exclude_identities
        )
        n_gallery_failed += d_dropped
    elif distractors > 0:
        noise_digest = system.for_gallery().digest if system is not None else system_digest
        d_rows, d_ids, d_labels, d_dropped = _distractor_rows(
            dataset, store, noise_digest, distractors, exclude_identities,
            seed if distractor_seed is None else distractor_seed,
        )
        n_gallery_failed += d_dropped
        gallery_rows.extend(d_rows)
        gallery_ids.extend(d_ids)
        gallery_labels.extend(d_labels)

    gallery_blocks: List[torch.Tensor] = []
    start = 0
    while start < len(gallery_rows):
        embedding_set = gallery_rows[start][0]
        end = start
        while end < len(gallery_rows) and gallery_rows[end][0] is embedding_set:
            end += 1
        gallery_blocks.append(
            _gather(embedding_set, [row for _set, row in gallery_rows[start:end]])
        )
        start = end
    gallery_matrix = (
        torch.cat(gallery_blocks) if gallery_blocks else torch.zeros((0, 1), dtype=torch.float32)
    )
    query_index = query_set.row_index()
    query_matrix = _gather(query_set, [query_index[i] for i in query_ids])

    name = kind if mix_r is None else f"mix-gallery-r{mix_r}"
    return Scenario(
        name=name,
        gallery_embeddings=gallery_matrix,
        gallery_labels=gallery_labels,
        gallery_ids=gallery_ids,
        query_embeddings=query_matrix,
        query_labels=query_labels,
        query_ids=query_ids,
        n_query_failed=n_query_failed,
        n_gallery_failed=n_gallery_failed,
        failed_query_ids=failed_query_ids,
        intended_relevant=intended_relevant,
        intended_query_ids=intended_query_ids,
        extra_blocks=extra_blocks,
        extra_block_iter=extra_block_iter,
        noise_image_count=noise_image_count,
        noise_identity_count=noise_identity_count,
    )
