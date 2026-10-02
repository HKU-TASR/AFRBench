"""The handle a method receives for its training corpus."""
from __future__ import annotations

import json
import os
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

from afrlib.utils.log import get_logger

from . import lmdb_io

if TYPE_CHECKING:
    from .filters import Filter

logger = get_logger(__name__)


@dataclass(frozen=True)
class SampleMeta:
    """Metadata of one sample: ``identity`` and ``domain`` as fields of their own, anything else
    the packer recorded in ``extra``.
    """

    key: str
    identity: Optional[str] = None
    domain: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        """Read a field by name, first-class or not (``None`` when absent)."""
        if name in ("key", "identity", "domain"):
            return getattr(self, name)
        return self.extra.get(name)


class TrainingCorpus:
    """A packed corpus seen through a set of filters, built by :func:`afrlib.data.get_corpus`;
    filtering is lazy and cached.
    """

    def __init__(
        self,
        name: str,
        lmdb_path: "str | os.PathLike[str]",
        filters: Sequence["Filter"] = (),
        *,
        source: str = "",
    ) -> None:
        self.name = name
        self.lmdb_path = Path(lmdb_path).expanduser().resolve()
        self.filters = tuple(filters)
        self.source = source
        self._metas: Optional[Tuple[SampleMeta, ...]] = None
        self._promoted: frozenset = frozenset()

    def metas(self, need: Sequence[str] = ()) -> Tuple[SampleMeta, ...]:
        """Metadata of every sample that passes the filters, in packing order.

        Args:
            need: Fields the caller will read; any the packer did not record are loaded from the
                payloads.

        Raises:
            ValueError: For a field the corpus cannot supply.
        """
        if self._metas is None:
            self._metas = tuple(self._filtered())
        missing = frozenset(need) - self._promoted
        if missing and self._metas:
            self._metas = tuple(
                self._promote(dict(meta.extra, key=meta.key,
                                   identity=meta.identity, domain=meta.domain),
                              missing)
                for meta in self._metas
            )
            self._promoted |= missing
        return self._metas

    def allowed_keys(self) -> Tuple[str, ...]:
        """Keys of the samples a training run may use."""
        return tuple(meta.key for meta in self.metas())

    def by_identity(self) -> Dict[str, Tuple[str, ...]]:
        """Allowed sample keys grouped by identity, skipping samples without one, so an unlabelled
        corpus gives an empty mapping.
        """
        groups: Dict[str, list] = {}
        for meta in self.metas():
            if meta.identity is None:
                continue
            groups.setdefault(meta.identity, []).append(meta.key)
        return {identity: tuple(keys) for identity, keys in groups.items()}

    def __len__(self) -> int:
        return len(self.metas())

    def __repr__(self) -> str:
        return (
            f"TrainingCorpus(name={self.name!r}, samples={len(self)}, "
            f"filters={len(self.filters)})"
        )

    def read(self, key: str, field_name: str = "image") -> bytes:
        """One payload field of one sample, as packed.

        Raises:
            KeyError: If the sample has no such field.
        """
        payload = lmdb_io.read_field(self.lmdb_path, key, field_name)
        if payload is None:
            raise KeyError(
                f"Corpus {self.name!r} has no {field_name!r} for sample {key!r}"
            )
        return payload

    def has_field(self, field_name: str) -> bool:
        """Whether samples carry this payload field; corpora are packed uniformly, so only the
        first allowed sample is checked.
        """
        keys = self.allowed_keys()
        if not keys:
            return False
        return lmdb_io.read_field(self.lmdb_path, keys[0], field_name) is not None

    def get_config(self) -> Dict[str, Any]:
        """Provenance (source, filters, sample counts) for the run config."""
        return {
            "corpus": self.name,
            "lmdb_path": str(self.lmdb_path),
            "source": self.source,
            "num_samples": len(self),
            "num_packed": lmdb_io.read_count(self.lmdb_path),
            "filters": [f.describe() for f in self.filters],
        }

    def _filtered(self):
        """Yield the metadata of the samples that pass every filter, then let each filter validate
        the values it saw, so entries that matched nothing are reported.
        """
        from .filters import needed_fields

        wanted = needed_fields(self.filters)
        self._promoted = wanted
        observed: Dict[str, set] = {name: set() for name in wanted}
        packed = kept = 0
        for record in lmdb_io.iter_sample_records(self.lmdb_path):
            meta = self._promote(record, wanted)
            packed += 1
            for name in wanted:
                value = meta.get(name)
                if isinstance(value, Hashable):
                    observed[name].add(value)
            if all(f.keep(meta) for f in self.filters):
                kept += 1
                yield meta
        for filter_ in self.filters:
            validate = getattr(filter_, "validate", None)
            if validate is not None:
                validate(self.name, observed)
        if self.filters:
            logger.info(
                "Corpus %r: %d of %d samples pass %s",
                self.name,
                kept,
                packed,
                ", ".join(type(f).__name__ for f in self.filters),
            )

    def _promote(self, record: Mapping[str, Any], wanted: "frozenset") -> SampleMeta:
        """Build a :class:`SampleMeta`, reading any ``wanted`` field the packed record lacks from
        the sample's payload.

        Raises:
            ValueError: Naming a field the corpus cannot supply.
        """
        key = record["key"]
        extra = {k: v for k, v in record.items() if k not in ("key", "identity", "domain")}
        for name in wanted:
            if name in ("key", "identity", "domain"):
                if record.get(name) is None and name != "key":
                    raise ValueError(
                        f"Corpus {self.name!r} does not record {name!r}; it is "
                        "not a labelled/multi-domain corpus."
                    )
                continue
            if name in extra:
                continue
            payload = lmdb_io.read_field(self.lmdb_path, key, name)
            if payload is None:
                raise ValueError(
                    f"Corpus {self.name!r} cannot supply the field {name!r}; it "
                    "is neither packed metadata nor a payload field."
                )
            extra[name] = _decode(payload)
        return SampleMeta(
            key=key,
            identity=record.get("identity"),
            domain=record.get("domain"),
            extra=extra,
        )


def _decode(payload: bytes) -> Any:
    """Best-effort decode of a promoted payload field (JSON, else text)."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
