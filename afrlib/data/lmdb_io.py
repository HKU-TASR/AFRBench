"""On-disk LMDB schema of a packed corpus, with its writer and readers."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

SCHEMA = "afrlib-corpus-v1"

_SCHEMA_KEY = b"meta/schema"
_COUNT_KEY = b"meta/count"
_JSON_KEY = b"meta/json"
_SAMPLE_PREFIX = "meta/sample/"

_MEBIBYTE = 1024 * 1024

# LMDB cannot open one environment twice per process, so read-only environments are shared
# here, keyed by pid so DataLoader workers reopen after a fork.
_ENVS: Dict[tuple, Any] = {}


def lmdb_env(path: "str | os.PathLike[str]"):
    """Shared read-only LMDB environment for ``path`` in this process; a handle inherited across
    a fork is closed first, as LMDB will not map one environment twice.
    """
    import lmdb

    real = os.path.realpath(str(path))
    pid = os.getpid()
    key = (real, pid)
    env = _ENVS.get(key)
    if env is not None:
        return env
    for stale in [item for item in _ENVS if item[0] == real and item[1] != pid]:
        old = _ENVS.pop(stale, None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
    env = lmdb.open(real, readonly=True, lock=False, readahead=False, meminit=False)
    _ENVS[key] = env
    return env


def _env_key(path: "str | os.PathLike[str]") -> tuple:
    return (os.path.realpath(str(path)), os.getpid())


def invalidate(path: "str | os.PathLike[str]") -> None:
    """Close and forget the cached environment for ``path``, which a repack must do or this
    process keeps reading the replaced corpus.
    """
    env = _ENVS.pop(_env_key(path), None)
    if env is not None:
        env.close()


def estimate_map_size(
    payload_bytes: int,
    samples: int,
    *,
    safety_factor: float = 2.0,
    minimum: int = 64 * _MEBIBYTE,
) -> int:
    """MiB-aligned LMDB map size for this payload, allowing 16 KiB per sample for keys, metadata
    and page overhead before the safety factor.
    """
    estimated = int((payload_bytes + samples * 16 * 1024) * safety_factor)
    return int(math.ceil(max(estimated, minimum) / _MEBIBYTE) * _MEBIBYTE)


class CorpusWriter:
    """Writes one packed corpus atomically: records go to a temporary sibling that replaces
    ``path`` on :meth:`close`, so an interrupted pack leaves nothing partial behind.

    Example:
        ::

            with CorpusWriter(path, map_size=..., provenance={...}) as writer:
                writer.add("00000001", {"image": jpeg_bytes}, identity="n000123")
    """

    def __init__(
        self,
        path: "str | os.PathLike[str]",
        *,
        map_size: int,
        provenance: Optional[Mapping[str, Any]] = None,
        batch_size: int = 512,
    ) -> None:
        import lmdb

        self.path = Path(path).expanduser().resolve()
        self.provenance = dict(provenance or {})
        self._batch_size = batch_size
        self._count = 0
        self._pending = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = Path(
            tempfile.mkdtemp(prefix=f".{self.path.name}.tmp-", dir=self.path.parent)
        )
        try:
            self._env = lmdb.open(
                str(self._tmp), subdir=True, map_size=map_size, meminit=False
            )
            self._txn = self._env.begin(write=True)
        except Exception:
            shutil.rmtree(self._tmp, ignore_errors=True)
            raise

    def add(self, key: str, fields: Mapping[str, bytes], **meta: Any) -> None:
        """Store one sample's payload fields and metadata.

        Args:
            key: Sample key, which must not contain ``/``.
            fields: Payload bytes per field, ``image`` always among them.
            **meta: What the dataset records about the sample, such as ``identity``,
                ``domain`` or ``name``; ``None`` values are dropped.
        """
        if "/" in key:
            # The reader splits "<key>/<field>", so a key with a slash in it
            # would be ambiguous.
            raise ValueError(f"Sample key {key!r} must not contain '/'")
        for field, payload in fields.items():
            self._txn.put(f"{key}/{field}".encode("utf-8"), payload)
        record = {"key": key, **{k: v for k, v in meta.items() if v is not None}}
        self._txn.put(
            f"{_SAMPLE_PREFIX}{self._count:08d}".encode("ascii"),
            json.dumps(record, separators=(",", ":")).encode("utf-8"),
        )
        self._count += 1
        self._pending += 1
        if self._pending >= self._batch_size:
            self._commit()

    def _commit(self) -> None:
        self._txn.commit()
        self._txn = self._env.begin(write=True)
        self._pending = 0

    def close(self) -> None:
        """Finalize the metadata and move the corpus into place, writing the schema marker last so
        readers treat an unfinished corpus as unpacked.
        """
        if self._count == 0:
            raise ValueError(f"Refusing to write an empty corpus at {self.path}")
        self._txn.put(_COUNT_KEY, str(self._count).encode("ascii"))
        self._txn.put(
            _JSON_KEY,
            json.dumps(
                {**self.provenance, "count": self._count, "schema": SCHEMA},
                sort_keys=True,
                default=str,
            ).encode("utf-8"),
        )
        self._txn.put(_SCHEMA_KEY, SCHEMA.encode("ascii"))
        self._txn.commit()
        self._env.sync()
        self._env.close()
        if self.path.exists():
            shutil.rmtree(self.path) if self.path.is_dir() else self.path.unlink()
        os.replace(self._tmp, self.path)
        invalidate(self.path)

    def abort(self) -> None:
        """Discard the partially written corpus."""
        try:
            self._txn.abort()
            self._env.close()
        finally:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> "CorpusWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


# Reading


def is_packed(path: "str | os.PathLike[str]") -> bool:
    """Whether ``path`` holds a complete corpus in this schema."""
    path = Path(path)
    if not path.is_dir():
        return False
    try:
        with lmdb_env(path).begin() as txn:
            return txn.get(_SCHEMA_KEY) == SCHEMA.encode("ascii")
    except Exception:
        return False


def read_provenance(path: "str | os.PathLike[str]") -> Dict[str, Any]:
    """The packer provenance recorded when the corpus was written."""
    with lmdb_env(path).begin() as txn:
        raw = txn.get(_JSON_KEY)
    return json.loads(raw) if raw else {}


def read_count(path: "str | os.PathLike[str]") -> int:
    """How many samples the corpus holds."""
    with lmdb_env(path).begin() as txn:
        raw = txn.get(_COUNT_KEY)
    if raw is None:
        raise ValueError(f"{path} is not an AFRLib corpus (no {_COUNT_KEY!r})")
    return int(raw)


def iter_sample_records(path: "str | os.PathLike[str]") -> Iterator[Dict[str, Any]]:
    """Stream every sample's metadata record, in packing order."""
    env = lmdb_env(path)
    with env.begin() as txn:
        for index in range(read_count(path)):
            raw = txn.get(f"{_SAMPLE_PREFIX}{index:08d}".encode("ascii"))
            if raw is None:
                raise ValueError(f"{path} is missing metadata for sample {index}")
            yield json.loads(raw)


def read_field(
    path: "str | os.PathLike[str]", key: str, field: str
) -> Optional[bytes]:
    """One payload field of one sample, or ``None`` when absent."""
    with lmdb_env(path).begin() as txn:
        return txn.get(f"{key}/{field}".encode("utf-8"))
