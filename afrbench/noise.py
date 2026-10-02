"""Identity purge filters and fingerprints for the packed MillionCelebs noise pool."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

PURGE_SCHEMA = "afrbench-purge-v1"
NOISE_CORPUS = "millioncelebs"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity_from_noise_key(key: str) -> str:
    """Return the MillionCelebs identity encoded in a packed sample key."""
    identity, separator, _filename = str(key).partition("__")
    if not separator or not identity:
        raise ValueError(f"noise key {key!r} does not encode '<identity>__<filename>'")
    return identity


def physical_pack_id(path: "str | Path", provenance: Mapping[str, Any]) -> str:
    """The pack's recorded ``pack_id``, else a hash of its data file's path, size, and mtime."""
    recorded = provenance.get("pack_id")
    if recorded:
        return str(recorded)
    root = Path(path)
    data = root / "data.mdb" if root.is_dir() else root
    stat = data.stat()
    return _canonical_hash({
        "path": str(data.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "count": provenance.get("count") or provenance.get("n_samples"),
        "source": provenance.get("source"),
    })


def selection_view_id(base_view_id: str, exclude_identities=()) -> str:
    """Fingerprint of a base view plus extra excluded identities (the base id if none)."""
    extra = sorted({str(item) for item in exclude_identities or []})
    if not extra:
        return str(base_view_id)
    return _canonical_hash({
        "base_view_id": str(base_view_id),
        "exclude_identities": extra,
    })


def lock_noise_identities(lock: Mapping[str, Any]) -> frozenset[str]:
    """Union of the MillionCelebs purge lists over the ``selected`` rows of a final lock."""
    return frozenset(
        str(item)
        for row in lock.get("selected") or []
        for item in (
            ((row.get("noise_overlap") or {}).get(NOISE_CORPUS) or {})
            .get("purge") or []
        )
    )


def validate_physical_noise_provenance(provenance: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless provenance shows a complete, aligned, reusable pack."""
    problems = []
    if not provenance.get("lock_independent"):
        problems.append("lock_independent is not true")
    if provenance.get("lock_id") or provenance.get("purge_id"):
        problems.append("pack is bound to a lock/purge")
    if list(provenance.get("identity_exclusions") or []):
        problems.append("physical pack records identity exclusions")
    if not provenance.get("aligned") or not provenance.get("alignment_id"):
        problems.append("complete ArcFace alignment is not certified")
    if not provenance.get("full_source"):
        problems.append("full_source is not true")
    samples = provenance.get("n_samples") or provenance.get("count")
    expected_samples = provenance.get("expected_source_samples")
    if expected_samples is None or int(samples or -1) != int(expected_samples):
        problems.append(
            f"packed samples {samples!r} != expected source {expected_samples!r}"
        )
    identities = provenance.get("n_identities")
    expected_identities = provenance.get("expected_source_identities")
    if expected_identities is None or int(identities or -1) != int(expected_identities):
        problems.append(
            f"packed identities {identities!r} != expected source "
            f"{expected_identities!r}"
        )
    if problems:
        raise ValueError("invalid reusable noise provenance: " + "; ".join(problems))


@dataclass(frozen=True)
class NoiseIdentityFilter:
    """A validated MillionCelebs identity purge for one dataset lock."""

    identities: frozenset[str]
    lock_id: str
    purge_id: str
    filter_id: str
    path: Optional[Path] = None

    def keep_identity(self, identity: str) -> bool:
        return str(identity) not in self.identities

    def keep_record(self, record: Mapping[str, Any]) -> bool:
        identity = str(record.get("identity") or "")
        if not identity:
            raise ValueError(f"noise record {record.get('key')!r} has no identity")
        return self.keep_identity(identity)

    def keep_key(self, key: str) -> bool:
        return self.keep_identity(identity_from_noise_key(key))

    def view_id(self, pack_id: str) -> str:
        """Fingerprint of the pack ``pack_id`` as filtered by this purge."""
        return _canonical_hash({"pack_id": str(pack_id), "filter_id": self.filter_id})


def filter_from_entries(
    entries: "set[str] | frozenset[str]",
    *,
    lock_id: str = "",
    purge_id: str = "",
    path: Optional[Path] = None,
) -> NoiseIdentityFilter:
    """Build a filter; ``filter_id`` hashes the entries and the lock and purge ids."""
    identities = frozenset(str(item) for item in entries)
    scientific = {
        "schema": PURGE_SCHEMA,
        "corpus": NOISE_CORPUS,
        "key": "identity",
        "entries": sorted(identities),
        "lock_id": str(lock_id),
        "purge_id": str(purge_id),
    }
    return NoiseIdentityFilter(
        identities=identities,
        lock_id=str(lock_id),
        purge_id=str(purge_id),
        filter_id=_canonical_hash(scientific),
        path=path,
    )


def load_noise_filter(
    path: "str | Path",
    *,
    expected_lock_id: Optional[str] = None,
    require_entries: bool = True,
) -> NoiseIdentityFilter:
    """Load and validate ``purge/millioncelebs.json``.

    Raises:
        ValueError: On a wrong schema, corpus, key or lock id, a non-production file, or, with
            ``require_entries``, an empty purge list.
    """
    source = Path(path)
    payload = json.loads(source.read_text())
    if payload.get("schema") != PURGE_SCHEMA:
        raise ValueError(
            f"{source} schema {payload.get('schema')!r} != {PURGE_SCHEMA!r}"
        )
    if payload.get("corpus") != NOISE_CORPUS:
        raise ValueError(
            f"{source} corpus {payload.get('corpus')!r} != {NOISE_CORPUS!r}"
        )
    if payload.get("key") != "identity":
        raise ValueError(f"{source} must be keyed on identity")
    if payload.get("production") is False:
        raise ValueError(f"{source} is marked smoke/non-production")
    entries = {str(item) for item in payload.get("entries") or []}
    if require_entries and not entries:
        raise ValueError(f"{source} has no entries")
    lock_id = str(payload.get("lock_id") or "")
    if expected_lock_id is not None and lock_id != str(expected_lock_id):
        raise ValueError(
            f"{source} lock_id {lock_id!r} does not match {expected_lock_id!r}"
        )
    return filter_from_entries(
        entries,
        lock_id=lock_id,
        purge_id=str(payload.get("artifact_id") or ""),
        path=source,
    )
