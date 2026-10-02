"""Fill the embedding cache: embed one image source with one FR system."""
from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from afrbench.dataset import BenchDataset
from afrbench.evaluator.store import EmbeddingSet, EmbeddingStore
from afrbench.fr_system import FRSystem
from afrbench.runs import ProtectedRun

Item = Tuple[str, torch.Tensor]


class LazyImages(AbcSequence):
    """``(id, image)`` pairs that decode only when the batch is taken."""

    def __init__(
        self,
        ids: Sequence[str],
        load: Callable[[str], torch.Tensor],
        *,
        host_prefetch: bool = True,
    ) -> None:
        self._ids = list(ids)
        self._load = load
        # File-backed loaders may prefetch the next slice on a host thread; LMDB
        # noise stays serial (the full-noise cache writer uses a DataLoader).
        self.host_prefetch = bool(host_prefetch)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return [(i, self._load(i)) for i in self._ids[index]]
        image_id = self._ids[index]
        return image_id, self._load(image_id)


def _originals(
    dataset: BenchDataset, eval_lists: Dict[str, Sequence[str]]
) -> LazyImages:
    """The original crop of every framed image in the eval lists."""
    paths = [
        path
        for paths in eval_lists.values()
        for path in paths
        if dataset.record(path).original_box is not None
    ]
    return LazyImages(paths, lambda p: dataset.original(dataset.record(p)))


def _targets(dataset: BenchDataset) -> LazyImages:
    """The HQ targets' high-resolution images; the pool targets arrive with the noise source, so
    embedding them here would enroll them twice.
    """
    paths = [
        record.path
        for identity in dataset.hq_target_identities()
        for record in dataset.images_of(identity, split="target")
        if record.has_face
    ]
    return LazyImages(paths, lambda p: dataset.image(dataset.record(p)))


def _noise(dataset: BenchDataset, keys: Sequence[str]) -> LazyImages:
    return LazyImages(keys, dataset.noise_image, host_prefetch=False)


def _protected(run: ProtectedRun) -> LazyImages:
    by_source = run.by_source()
    return LazyImages(run.source_paths, lambda p: by_source[p].load())


def embed_source(
    source: str,
    system: FRSystem,
    dataset: BenchDataset,
    store: EmbeddingStore,
    device: torch.device,
    *,
    eval_lists: Optional[Dict[str, Sequence[str]]] = None,
    noise_count: Optional[int] = None,
    noise_seed: int = 0,
    noise_exclude_identities: Optional[Sequence[str]] = None,
    run: Optional[ProtectedRun] = None,
    batch_size: int = 32,
) -> Path:
    """Embed one source with one FR system into the cache, skipping a current entry.

    Args:
        source: ``originals``, ``targets``, ``noise``, or a protected run's store name.
        system: The FR system to embed with.
        dataset: The benchmark dataset.
        store: Cache the entry is written to.
        device: Device to embed on.
        eval_lists: Per-protectee eval images, for the ``originals`` source.
        noise_count: Distractor keys to draw; ``None`` embeds all of them.
        noise_seed: Seed of that draw.
        noise_exclude_identities: Identities left out before drawing.
        run: The protect run to embed, for a protected source.
        batch_size: Images per batch.

    Returns:
        The cache entry's directory; each (view, seed, count) is its own entry.
    """
    if source == "noise":
        system = system.for_gallery()
    digest = system.digest
    noise_view_id = (
        dataset.noise_view_id_for(noise_exclude_identities)
        if source == "noise" else None
    )
    if store.is_current(
        source, digest, noise_count, noise_view_id, noise_seed
    ):
        return store._dir(
            source, digest, noise_view_id, noise_seed, noise_count
        )

    items: Sequence[Item]
    aligned = False
    if source == "originals":
        items = _originals(dataset, eval_lists or {})
    elif source == "targets":
        items = _targets(dataset)
    elif source == "noise":
        keys = (
            dataset.noise_keys(noise_exclude_identities)
            if noise_count is None
            else dataset.noise_subset(
                noise_count,
                seed=noise_seed,
                exclude_identities=noise_exclude_identities,
            )
        )
        items = _noise(dataset, keys)
        aligned = dataset.noise_aligned
    elif source.startswith("protected__"):
        if run is None:
            raise ValueError(f"source {source!r} needs a ProtectedRun")
        items = _protected(run)
    else:
        raise ValueError(f"Unknown embedding source {source!r}")

    embed_kw: Dict[str, Any] = {"batch_size": batch_size}
    if not aligned:
        ids, embeddings, failed = system.embed(items, device, **embed_kw)
    else:
        ids, embeddings, failed = system.embed_aligned(items, device, **embed_kw)
    meta = {"fr_system": system.to_dict(), "dataset": str(dataset.root), "aligned": aligned}
    if source == "noise" and noise_count is not None:
        meta["noise_count"] = int(noise_count)
    if noise_view_id is not None:
        meta["noise_view_id"] = noise_view_id
        meta["noise_seed"] = int(noise_seed)
        meta["exclude_identities"] = sorted(
            {str(item) for item in noise_exclude_identities or ()}
        )
    embedding_set = EmbeddingSet(
        ids=ids, embeddings=embeddings.numpy().astype("float32"), failed_ids=failed
    )
    return store.put(
        source,
        digest,
        embedding_set,
        meta=meta,
        noise_view_id=noise_view_id,
        noise_seed=noise_seed,
        noise_count=noise_count,
    )
