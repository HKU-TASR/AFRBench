"""Disk-cached FR embeddings, keyed by source and FR-system digest.

    <results_root>/cache/embeddings/<source>/<system_digest>/
        embeddings.npy   (N, D) float32, L2-normalized, successful rows only
        ids.json         the N image ids (annotation paths or noise keys)
        failed.json      ids whose face was not detected
        meta.json        provenance: fr_system spec, dataset, counts
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

#: Default results root; the cache lives under ``<results>/cache``.
DEFAULT_RESULTS = Path("results")


@dataclasses.dataclass
class EmbeddingSet:
    """One embedded image set: successful rows plus the detection failures."""

    ids: List[str]
    embeddings: np.ndarray  # (N, D) float32, L2-normalized
    failed_ids: List[str]

    def __len__(self) -> int:
        return len(self.ids)

    def row_index(self) -> Dict[str, int]:
        return {image_id: i for i, image_id in enumerate(self.ids)}


def cache_root(results_root: "str | Path" = DEFAULT_RESULTS) -> Path:
    return Path(results_root) / "cache"


class EmbeddingStore:
    """The embedding cache under ``<cache>/embeddings``."""

    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)

    def _dir(
        self,
        source: str,
        system_digest: str,
        noise_view_id: Optional[str] = None,
        noise_seed: int = 0,
        noise_count: Optional[int] = None,
    ) -> Path:
        directory = self.root / source / system_digest
        if source == "noise" and noise_view_id is not None:
            variant = hashlib.sha256(
                f"{noise_view_id}\0{int(noise_seed)}\0{noise_count}".encode("utf-8")
            ).hexdigest()
            directory = directory / variant
        return directory

    def has(
        self,
        source: str,
        system_digest: str,
        noise_view_id: Optional[str] = None,
        noise_seed: int = 0,
        noise_count: Optional[int] = None,
    ) -> bool:
        return (
            self._dir(
                source, system_digest, noise_view_id, noise_seed, noise_count
            ) / "meta.json"
        ).is_file()

    def is_current(
        self,
        source: str,
        system_digest: str,
        noise_count: Optional[int] = None,
        noise_view_id: Optional[str] = None,
        noise_seed: int = 0,
    ) -> bool:
        """True when the entry exists and, for ``noise``, its count, view id, and seed match."""
        meta_path = self._dir(
            source, system_digest, noise_view_id, noise_seed, noise_count
        ) / "meta.json"
        if not meta_path.is_file():
            return False
        if source == "noise":
            meta = json.loads(meta_path.read_text())
            recorded = meta.get("noise_count")
            if noise_count is not None and int(recorded or -1) != int(noise_count):
                return False
            if noise_view_id is not None and meta.get("noise_view_id") != noise_view_id:
                return False
            if int(meta.get("noise_seed", 0)) != int(noise_seed):
                return False
        return True

    def get(
        self,
        source: str,
        system_digest: str,
        noise_view_id: Optional[str] = None,
        noise_seed: int = 0,
        noise_count: Optional[int] = None,
    ) -> EmbeddingSet:
        directory = self._dir(
            source, system_digest, noise_view_id, noise_seed, noise_count
        )
        if not self.has(
            source, system_digest, noise_view_id, noise_seed, noise_count
        ):
            raise KeyError(f"No embeddings at {directory}")
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        ids = json.loads((directory / "ids.json").read_text())
        failed = json.loads((directory / "failed.json").read_text())
        return EmbeddingSet(ids=ids, embeddings=embeddings, failed_ids=failed)

    def put(
        self,
        source: str,
        system_digest: str,
        embedding_set: EmbeddingSet,
        meta: Optional[Dict[str, Any]] = None,
        *,
        noise_view_id: Optional[str] = None,
        noise_seed: int = 0,
        noise_count: Optional[int] = None,
    ) -> Path:
        """Write an embedding set atomically under a file lock, keeping an existing complete entry;
        returns the entry's directory.
        """
        directory = self._dir(
            source, system_digest, noise_view_id, noise_seed, noise_count
        )
        lock_path = directory.parent / f".{directory.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            if self.has(source, system_digest, noise_view_id, noise_seed, noise_count):
                return directory
            temporary = directory.with_name(
                f"{directory.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            temporary.mkdir(parents=True, exist_ok=True)
            embeddings = np.asarray(embedding_set.embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(embedding_set.ids):
                raise ValueError(
                    f"embeddings {embeddings.shape} do not match {len(embedding_set.ids)} ids"
                )
            np.save(temporary / "embeddings.npy", embeddings)
            (temporary / "ids.json").write_text(json.dumps(embedding_set.ids))
            (temporary / "failed.json").write_text(json.dumps(embedding_set.failed_ids))
            provenance = {"count": len(embedding_set.ids), "failed": len(embedding_set.failed_ids)}
            provenance.update(meta or {})
            (temporary / "meta.json").write_text(json.dumps(provenance, indent=2, default=str))
            directory.parent.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                import shutil

                if source == "noise":
                    shutil.rmtree(temporary)
                    return directory
                shutil.rmtree(directory)
            try:
                os.replace(temporary, directory)
            except OSError:
                import shutil

                if source != "noise" or not directory.exists():
                    raise
                shutil.rmtree(temporary)
            return directory
        finally:
            lock_handle.close()
