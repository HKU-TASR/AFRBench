"""Full-MillionCelebs embedding cache, shared across datasets and their purges.

    <dataset root>/../_shared/cache/millioncelebs/<cache-id>/
        embeddings.f32       (N, D) float32 memmap
        identities.i32       (N,) int32 identity codes
        keys.jsonl           one LMDB key per row (failed rows hold zero vectors)
        failed.json          keys whose embedding failed
        identity_vocab.json  identity name of each code
        state.json           resumable write cursor, removed on completion
        meta.json            written last; its presence marks the cache complete
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from afrlib import get_logger

logger = get_logger(__name__)

EMBEDDINGS_NAME = "embeddings.f32"
IDENTITIES_NAME = "identities.i32"
KEYS_NAME = "keys.jsonl"
FAILED_NAME = "failed.json"
VOCAB_NAME = "identity_vocab.json"
VOCAB_LINES_NAME = "identity_vocab.jsonl"
STATE_NAME = "state.json"
META_NAME = "meta.json"
LOCK_NAME = ".write.lock"
#: Minimum batch size for the MillionCelebs embed loop; a smaller eval
#: ``batch_size`` is raised to it.
CACHE_EMBED_BATCH = 256
CACHE_DATALOADER_WORKERS = 4
CACHE_PREFETCH_FACTOR = 4


def shared_cache_root(dataset_root: "str | Path") -> Path:
    """``<dataset root>/../_shared/cache/millioncelebs``, shared by sibling datasets."""
    export = Path(dataset_root).resolve()
    return export.parent / "_shared" / "cache" / "millioncelebs"


def cache_identity(
    *,
    pack_id: str,
    fr_digest: str,
    weight_identity: str,
) -> str:
    """Cache directory name: a hash of the pack id, FR-system digest, and weight identity."""
    payload = json.dumps(
        {
            "pack_id": pack_id,
            "fr_digest": fr_digest,
            "weight_identity": weight_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _file_lock(path: Path):
    """Cross-process exclusive lock; no-op if ``fcntl`` is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except ImportError:
        pass
    return handle


class FullNoiseCache:
    """One full-pack cache directory: completeness check, validation, and reader."""

    def __init__(self, directory: "str | Path") -> None:
        self.root = Path(directory)

    @property
    def meta_path(self) -> Path:
        return self.root / META_NAME

    def is_complete(self) -> bool:
        """True when ``meta.json`` marks the cache complete and the arrays have its sizes."""
        if not self.meta_path.is_file():
            return False
        try:
            meta = json.loads(self.meta_path.read_text())
        except json.JSONDecodeError:
            return False
        if not meta.get("complete"):
            return False
        n = int(meta.get("count") or 0)
        dim = int(meta.get("dim") or 0)
        embeddings = self.root / EMBEDDINGS_NAME
        identities = self.root / IDENTITIES_NAME
        keys = self.root / KEYS_NAME
        if not embeddings.is_file() or not identities.is_file() or not keys.is_file():
            return False
        if embeddings.stat().st_size != n * dim * 4:
            return False
        if identities.stat().st_size != n * 4:
            return False
        return True

    def meta(self) -> Dict[str, Any]:
        return json.loads(self.meta_path.read_text())

    def reader(self) -> "FullNoiseReader":
        if not self.is_complete():
            raise FileNotFoundError(f"incomplete MillionCelebs cache at {self.root}")
        return FullNoiseReader(self.root, self.meta())

    def expected_checksums(self) -> Dict[str, str]:
        meta = self.meta()
        return {
            "embeddings_sha256": _memmap_sha256(self.root / EMBEDDINGS_NAME),
            "identities_sha256": _memmap_sha256(self.root / IDENTITIES_NAME),
            **{k: meta.get(k) for k in ("pack_id", "fr_digest", "weight_identity")},
        }

    def validate(self) -> Dict[str, Any]:
        """Raise if a complete cache's arrays or provenance do not match meta."""
        if not self.is_complete():
            raise ValueError(f"MillionCelebs cache at {self.root} is not complete")
        meta = self.meta()
        embeddings_hash = _memmap_sha256(self.root / EMBEDDINGS_NAME)
        identities_hash = _memmap_sha256(self.root / IDENTITIES_NAME)
        if meta.get("embeddings_sha256") and meta["embeddings_sha256"] != embeddings_hash:
            raise ValueError("embeddings checksum mismatch")
        if meta.get("identities_sha256") and meta["identities_sha256"] != identities_hash:
            raise ValueError("identities checksum mismatch")
        return meta


class FullNoiseReader:
    """Memory-mapped view over a completed cache."""

    def __init__(self, root: Path, meta: Dict[str, Any]) -> None:
        self.root = root
        self.meta = meta
        self.count = int(meta["count"])
        self.dim = int(meta["dim"])
        self.embeddings = np.memmap(
            root / EMBEDDINGS_NAME, mode="r", dtype=np.float32, shape=(self.count, self.dim)
        )
        self.identity_codes = np.memmap(
            root / IDENTITIES_NAME, mode="r", dtype=np.int32, shape=(self.count,)
        )
        self.vocab: List[str] = json.loads((root / VOCAB_NAME).read_text())
        self.failed = set(json.loads((root / FAILED_NAME).read_text()))
        self._keys: Optional[List[str]] = None
        self._kept_cache: Dict[frozenset, Tuple[int, int]] = {}

    def iter_keys(self) -> Iterator[str]:
        """Stream LMDB keys in cache order. Does not materialize the sidecar."""
        n = 0
        with (self.root / KEYS_NAME).open(encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")
                n += 1
        if n != self.count:
            raise ValueError(f"key sidecar has {n} rows, cache count is {self.count}")

    def keys(self) -> List[str]:
        """Materialize every cache key. Prefer :meth:`iter_keys` / :meth:`iter_blocks`."""
        if self._keys is None:
            self._keys = list(self.iter_keys())
        return self._keys

    def identity_at(self, index: int) -> str:
        code = int(self.identity_codes[index])
        return self.vocab[code] if 0 <= code < len(self.vocab) else ""

    def kept_counts(
        self, exclude_identities: Optional[Sequence[str]] = None
    ) -> Tuple[int, int]:
        """``(n_images, n_identities)`` surviving ``exclude_identities``, counted from the
        identity-code memmap and cached.
        """
        excluded = frozenset(str(item) for item in exclude_identities or ())
        cached = self._kept_cache.get(excluded)
        if cached is not None:
            return cached
        exclude_codes = self._exclude_codes(set(excluded))
        codes = np.asarray(self.identity_codes)
        keep = np.ones(self.count, dtype=bool)
        if exclude_codes.size:
            keep &= ~np.isin(codes, exclude_codes)
        if self.failed:
            failed = self.failed
            for index, key in enumerate(self.iter_keys()):
                if key in failed:
                    keep[index] = False
        kept = codes[keep]
        n_images = int(kept.size)
        if n_images == 0:
            result = (0, 0)
        else:
            empty = np.array(
                [index for index, name in enumerate(self.vocab) if not name],
                dtype=np.int32,
            )
            named = kept if empty.size == 0 else kept[~np.isin(kept, empty)]
            result = (n_images, int(np.unique(named).size) if named.size else 0)
        self._kept_cache[excluded] = result
        return result

    def _exclude_codes(self, excluded: set[str]) -> np.ndarray:
        if not excluded:
            return np.zeros((0,), dtype=np.int32)
        codes = [index for index, name in enumerate(self.vocab) if name in excluded]
        return np.asarray(codes, dtype=np.int32)

    def _keep_mask(
        self,
        start: int,
        end: int,
        exclude_codes: np.ndarray,
        chunk_keys: Optional[Sequence[str]],
    ) -> np.ndarray:
        n = end - start
        keep = np.ones(n, dtype=bool)
        codes = np.asarray(self.identity_codes[start:end])
        if exclude_codes.size:
            keep &= ~np.isin(codes, exclude_codes)
        if chunk_keys is not None and self.failed:
            keep &= np.fromiter(
                (key not in self.failed for key in chunk_keys),
                dtype=bool,
                count=n,
            )
        return keep

    def iter_rank_blocks(
        self,
        *,
        exclude_identities: Optional[Sequence[str]] = None,
        chunk: int = 65_536,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """Yield ``(embeddings, codes, global_index[, keep])`` blocks of the purged pack, as
        uncopied memmap slices with a keep mask unless a chunk is sparse enough to compact.
        """
        excluded = {str(item) for item in exclude_identities or ()}
        exclude_codes = self._exclude_codes(excluded)
        key_iter = self.iter_keys() if self.failed else None
        for start in range(0, self.count, chunk):
            end = min(start + chunk, self.count)
            n = end - start
            chunk_keys = [next(key_iter) for _ in range(n)] if key_iter is not None else None
            keep = self._keep_mask(start, end, exclude_codes, chunk_keys)
            if not keep.any():
                continue
            codes = np.asarray(self.identity_codes[start:end])
            embeddings = np.asarray(self.embeddings[start:end])
            gidx = np.arange(start, end, dtype=np.int64)
            if keep.all():
                yield embeddings, np.ascontiguousarray(codes, dtype=np.int32), gidx
            elif keep.mean() >= 0.25:
                yield embeddings, np.ascontiguousarray(codes, dtype=np.int32), gidx, keep
            else:
                index = np.nonzero(keep)[0]
                yield (
                    np.ascontiguousarray(embeddings[index]),
                    np.ascontiguousarray(codes[index], dtype=np.int32),
                    np.ascontiguousarray(gidx[index]),
                )

    def lookup_keys(self, indices: Sequence[int]) -> List[str]:
        """Read ``keys.jsonl`` for the requested global rows, in one pass."""
        order = [int(index) for index in indices]
        if not order:
            return []
        unique = sorted(set(order))
        found: Dict[int, str] = {}
        want = 0
        with (self.root / KEYS_NAME).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if want < len(unique) and index == unique[want]:
                    found[index] = line.rstrip("\n")
                    want += 1
                    if want == len(unique):
                        break
        missing = [index for index in unique if index not in found]
        if missing:
            raise KeyError(f"keys.jsonl missing rows {missing[:8]}")
        return [found[index] for index in order]

    def iter_blocks(
        self,
        *,
        exclude_identities: Optional[Sequence[str]] = None,
        chunk: int = 65_536,
    ) -> Iterator[Tuple[np.ndarray, List[str], List[str]]]:
        """Yield ``(embeddings, labels, keys)`` for rows that survive the purge."""
        excluded = {str(item) for item in exclude_identities or ()}
        exclude_codes = self._exclude_codes(excluded)
        key_iter = self.iter_keys()
        vocab = self.vocab
        for start in range(0, self.count, chunk):
            end = min(start + chunk, self.count)
            n = end - start
            chunk_keys = [next(key_iter) for _ in range(n)]
            keep = self._keep_mask(start, end, exclude_codes, chunk_keys)
            index = np.nonzero(keep)[0]
            if index.size == 0:
                continue
            codes = np.asarray(self.identity_codes[start:end])
            labels: List[str] = []
            kept_keys: List[str] = []
            for local in index.tolist():
                code = int(codes[local])
                key = chunk_keys[local]
                identity = vocab[code] if 0 <= code < len(vocab) else ""
                labels.append(f"noise/{identity}" if identity else f"noise/{key}")
                kept_keys.append(key)
            yield np.asarray(self.embeddings[start:end][index]), labels, kept_keys


class IncrementalWriter:
    """Resumable memmap writer for a full-pack cache; :meth:`finalize` writes ``meta.json``."""

    def __init__(self, directory: "str | Path", *, dim: int, expected: int) -> None:
        self.root = Path(directory)
        self.dim = int(dim)
        self.expected = int(expected)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = _file_lock(self.root / LOCK_NAME)
        self._thread = threading.Lock()
        self._failed_dirty = False
        state_path = self.root / STATE_NAME
        if state_path.is_file() and not (self.root / META_NAME).is_file():
            state = json.loads(state_path.read_text())
            self.next_index = int(state.get("next_index") or 0)
            self.vocab = _load_vocab(self.root)
            self._vocab_index = {name: i for i, name in enumerate(self.vocab)}
            self.failed: List[str] = json.loads((self.root / FAILED_NAME).read_text())
            _rewrite_vocab_lines(self.root, self.vocab)
        else:
            if (self.root / META_NAME).is_file():
                raise RuntimeError(f"refusing to reopen a complete cache at {self.root}")
            self.next_index = 0
            self.vocab = []
            self._vocab_index = {}
            self.failed = []
            (self.root / FAILED_NAME).write_text("[]")
            (self.root / VOCAB_NAME).write_text("[]")
            (self.root / VOCAB_LINES_NAME).write_text("")
            (self.root / KEYS_NAME).write_text("")
            self._save_state()
        self._vocab_lines = (self.root / VOCAB_LINES_NAME).open("a", encoding="utf-8")
        self.embeddings = _open_memmap(
            self.root / EMBEDDINGS_NAME, dtype=np.float32, shape=(self.expected, self.dim)
        )
        self.identities = _open_memmap(
            self.root / IDENTITIES_NAME, dtype=np.int32, shape=(self.expected,)
        )

    def append(
        self,
        keys: Sequence[str],
        embeddings: np.ndarray,
        identities: Sequence[str],
        failed: Sequence[str] = (),
    ) -> None:
        """Write the next rows and persist the write cursor."""
        with self._thread:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
                raise ValueError(f"expected (*, {self.dim}) embeddings, got {embeddings.shape}")
            if embeddings.shape[0] != len(keys) or len(keys) != len(identities):
                raise ValueError("keys, embeddings, and identities must align")
            end = self.next_index + len(keys)
            if end > self.expected:
                raise ValueError(
                    f"writer would exceed expected {self.expected} rows (next={self.next_index})"
                )
            codes = np.fromiter(
                (self._code(identity) for identity in identities),
                dtype=np.int32,
                count=len(identities),
            )
            self.embeddings[self.next_index : end] = embeddings
            self.identities[self.next_index : end] = codes
            with (self.root / KEYS_NAME).open("a", encoding="utf-8") as handle:
                for key in keys:
                    handle.write(f"{key}\n")
            if failed:
                self.failed.extend(str(item) for item in failed)
                self._failed_dirty = True
            self.next_index = end
            self._vocab_lines.flush()
            self._save_state()

    def finalize(
        self,
        *,
        pack_id: str,
        fr_digest: str,
        weight_identity: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Check the row count, write checksums and ``meta.json``, and drop the resume state."""
        with self._thread:
            if self.next_index != self.expected:
                raise ValueError(
                    f"cache expected {self.expected} rows but wrote {self.next_index}"
                )
            self.embeddings.flush()
            self.identities.flush()
            raw_emb = self.root / EMBEDDINGS_NAME
            raw_ids = self.root / IDENTITIES_NAME
            (self.root / FAILED_NAME).write_text(json.dumps(sorted(set(self.failed))))
            (self.root / VOCAB_NAME).write_text(json.dumps(self.vocab))
            meta = {
                "complete": True,
                "count": self.expected,
                "dim": self.dim,
                "pack_id": pack_id,
                "fr_digest": fr_digest,
                "weight_identity": weight_identity,
                "embeddings_sha256": _memmap_sha256(raw_emb),
                "identities_sha256": _memmap_sha256(raw_ids),
            }
            if extra:
                meta.update(extra)
            temporary = self.root / f".{META_NAME}.tmp"
            temporary.write_text(json.dumps(meta, indent=2, sort_keys=True))
            os.replace(temporary, self.root / META_NAME)
            if (self.root / STATE_NAME).is_file():
                (self.root / STATE_NAME).unlink()
            try:
                self._vocab_lines.flush()
                self._vocab_lines.close()
            except Exception:
                pass
            lines = self.root / VOCAB_LINES_NAME
            if lines.is_file():
                lines.unlink()
        return self.root / META_NAME

    def _code(self, identity: str) -> int:
        name = str(identity or "")
        existing = self._vocab_index.get(name)
        if existing is not None:
            return existing
        index = len(self.vocab)
        self.vocab.append(name)
        self._vocab_index[name] = index
        self._vocab_lines.write(json.dumps(name) + "\n")
        return index

    def _save_state(self) -> None:
        # The vocabulary is appended to identity_vocab.jsonl as codes are
        # assigned, so only the failed list and the cursor are saved here.
        if self._failed_dirty:
            (self.root / FAILED_NAME).write_text(json.dumps(self.failed))
            self._failed_dirty = False
        temporary = self.root / f".{STATE_NAME}.tmp"
        temporary.write_text(json.dumps({"next_index": self.next_index, "expected": self.expected}))
        os.replace(temporary, self.root / STATE_NAME)

    def close(self) -> None:
        try:
            self._vocab_lines.flush()
            self._vocab_lines.close()
        except Exception:
            pass
        try:
            self._lock_handle.close()
        except Exception:
            pass


def _cache_embed_batch(system, batch_size: int) -> int:
    """``batch_size`` raised to at least ``CACHE_EMBED_BATCH``."""
    del system
    return max(int(batch_size), CACHE_EMBED_BATCH)


def _load_vocab(root: Path) -> List[str]:
    lines_path = root / VOCAB_LINES_NAME
    if lines_path.is_file() and lines_path.stat().st_size:
        return [
            json.loads(line)
            for line in lines_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    vocab_path = root / VOCAB_NAME
    if vocab_path.is_file():
        return list(json.loads(vocab_path.read_text()))
    return []


def _rewrite_vocab_lines(root: Path, vocab: Sequence[str]) -> None:
    temporary = root / f".{VOCAB_LINES_NAME}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for name in vocab:
            handle.write(json.dumps(name) + "\n")
    os.replace(temporary, root / VOCAB_LINES_NAME)


def build_full_noise_cache(
    *,
    dataset,
    system,
    device,
    cache_root: Optional["str | Path"] = None,
    batch_size: int = 32,
    num_workers: int = CACHE_DATALOADER_WORKERS,
) -> Path:
    """Embed every sample of the noise pack into the shared cache, with no sampling or purge,
    resuming a partial cache and reusing a complete one.
    """
    import time

    from torch.utils.data import DataLoader

    from afrbench.dataset import decode_image
    from afrbench.evaluator.mc_crops import MillionCelebsCrops, collate_crops
    from afrbench.noise import physical_pack_id
    from afrlib.data.lmdb_io import iter_sample_records, read_count, read_field

    pack_id = physical_pack_id(dataset.noise_lmdb, dataset.noise_provenance)
    system = system.for_gallery()
    weight_id = system.weight_identity
    digest = cache_identity(
        pack_id=pack_id, fr_digest=system.digest, weight_identity=weight_id
    )
    root = Path(cache_root or shared_cache_root(dataset.root)) / digest
    cache = FullNoiseCache(root)
    if cache.is_complete():
        cache.validate()
        recorded = cache.meta()
        if recorded.get("pack_id") != pack_id:
            raise ValueError("existing cache pack_id does not match the current LMDB")
        return root

    expected = read_count(dataset.noise_lmdb)
    if expected <= 0:
        raise ValueError(f"noise pack at {dataset.noise_lmdb} has no keys")
    first = next(iter_sample_records(dataset.noise_lmdb), None)
    if first is None:
        raise ValueError(f"noise pack at {dataset.noise_lmdb} has no keys")
    first_key = str(first["key"])
    probe = decode_image(read_field(dataset.noise_lmdb, first_key, "image"))
    aligned = bool(dataset.noise_aligned)
    if aligned:
        _, probe_emb, _ = system.embed_aligned(
            [(first_key, probe)], device, batch_size=1
        )
    else:
        _, probe_emb, _ = system.embed([(first_key, probe)], device, batch_size=1)
    dim = int(probe_emb.shape[-1])
    writer = IncrementalWriter(root, dim=dim, expected=expected)
    embed_batch = _cache_embed_batch(system, batch_size)
    logger.info(
        "MillionCelebs cache %s: resume %d/%d, embed batch %d workers=%d",
        digest,
        writer.next_index,
        expected,
        embed_batch,
        num_workers,
    )
    pin = str(device).startswith("cuda") or (
        getattr(device, "type", None) == "cuda"
    )
    started = time.monotonic()
    last_log = started
    last_count = writer.next_index
    workers = max(int(num_workers), 0)
    try:
        if writer.next_index >= expected:
            writer.finalize(
                pack_id=pack_id,
                fr_digest=system.digest,
                weight_identity=weight_id,
                extra={"aligned": aligned, "source": str(dataset.noise_lmdb)},
            )
            return root
        loader = DataLoader(
            MillionCelebsCrops(
                dataset.noise_lmdb, start=writer.next_index, scale_01=False
            ),
            batch_size=embed_batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
            prefetch_factor=CACHE_PREFETCH_FACTOR if workers > 0 else None,
            collate_fn=collate_crops,
        )
        for images, keys, identities in loader:
            images = images.to(device, non_blocking=True)
            failed: List[str] = []
            if aligned:
                embeddings = system.embed_aligned_batch(images, device)
                ids = list(keys)
            else:
                items = [(key, images[index]) for index, key in enumerate(keys)]
                ids, embeddings, failed = system.embed(
                    items, device, batch_size=len(keys), prefetch=False
                )
            by_id = {image_id: embeddings[i] for i, image_id in enumerate(ids)}
            rows = []
            for key in keys:
                if key in by_id:
                    value = by_id[key]
                    rows.append(
                        value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                    )
                else:
                    rows.append(np.zeros((dim,), dtype=np.float32))
            writer.append(
                keys,
                np.stack(rows, axis=0),
                identities,
                failed=failed,
            )
            now = time.monotonic()
            if now - last_log >= 30.0 or writer.next_index == expected:
                delta = max(now - last_log, 1e-6)
                rate = (writer.next_index - last_count) / delta
                elapsed = now - started
                logger.info(
                    "MillionCelebs cache %s: %d/%d (%.1f img/s, %.1f min elapsed)",
                    digest,
                    writer.next_index,
                    expected,
                    rate,
                    elapsed / 60.0,
                )
                last_log = now
                last_count = writer.next_index
        writer.finalize(
            pack_id=pack_id,
            fr_digest=system.digest,
            weight_identity=weight_id,
            extra={"aligned": aligned, "source": str(dataset.noise_lmdb)},
        )
    finally:
        writer.close()
    return root


def _open_memmap(path: Path, *, dtype, shape) -> np.memmap:
    mode = "r+" if path.is_file() else "w+"
    return np.memmap(path, mode=mode, dtype=dtype, shape=shape)


def _memmap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
