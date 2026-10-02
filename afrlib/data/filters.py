"""Corpus filters: which samples a training run may use."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Iterable, Mapping, Protocol, runtime_checkable

from afrlib.utils.log import get_logger

from .corpus import SampleMeta

logger = get_logger(__name__)


@runtime_checkable
class Filter(Protocol):
    """Votes on whether one sample may be used for training."""

    #: Sample fields :meth:`keep` reads, so the corpus can supply them.
    needs: tuple[str, ...]

    def keep(self, meta: SampleMeta) -> bool:
        """Whether to keep this sample."""

    def describe(self) -> Dict[str, Any]:
        """JSON-able provenance recorded in the run config."""

    def validate(self, corpus: str, observed: Mapping[str, Collection[Any]]) -> None:
        """Optional hook called once after filtering with every value seen for each field in
        ``needs``, which the exclusion filters use to report entries that matched nothing.
        """


def _report_unmatched(
    kind: str,
    corpus: str,
    wanted: Collection[str],
    seen: Collection[Any],
    *,
    on_unmatched: str = "warn",
) -> None:
    """Warn or raise about exclusion entries no sample in the corpus matched."""
    unmatched = sorted(name for name in wanted if name not in seen)
    if not unmatched:
        return
    if on_unmatched == "error":
        raise ValueError(
            f"{kind}: {len(unmatched)} of {len(wanted)} entries matched nothing "
            f"in corpus {corpus!r} ({', '.join(unmatched[:5])}"
            f"{', ...' if len(unmatched) > 5 else ''})"
        )
    logger.warning(
        "%s: %d of %d entries matched nothing in corpus %r (%s%s). Nothing was "
        "excluded for them -- check they are named the way this corpus names "
        "its samples.",
        kind,
        len(unmatched),
        len(wanted),
        corpus,
        ", ".join(unmatched[:5]),
        ", ..." if len(unmatched) > 5 else "",
    )


@dataclass(frozen=True)
class ExcludeIdentities:
    """Drop every image of these identities, in a labelled corpus.

    Args:
        identities: Identities to exclude.
        on_unmatched: ``"warn"`` or ``"error"`` for identities that match no sample.
    """

    identities: frozenset[str]
    on_unmatched: str = "warn"
    needs = ("identity",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identities", frozenset(self.identities))
        if self.on_unmatched not in {"warn", "error"}:
            raise ValueError(f"on_unmatched must be warn|error, got {self.on_unmatched!r}")

    def keep(self, meta: SampleMeta) -> bool:
        return meta.identity not in self.identities

    def validate(self, corpus: str, observed: Mapping[str, Collection[Any]]) -> None:
        _report_unmatched(
            "ExcludeIdentities",
            corpus,
            self.identities,
            observed.get("identity", ()),
            on_unmatched=self.on_unmatched,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "filter": "ExcludeIdentities",
            "count": len(self.identities),
            "identities": sorted(self.identities),
        }


@dataclass(frozen=True)
class ExcludeImages:
    """Drop individual images by name, the option for an unlabelled corpus.

    Args:
        names: Dataset-relative ``name`` fields as the packer recorded them, such as
            ``"n000002/0001_01.jpg"``, not the keys assigned at pack time.
        on_unmatched: ``"warn"`` or ``"error"`` for names that match no sample.
    """

    names: frozenset[str]
    on_unmatched: str = "warn"
    needs = ("name",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", frozenset(self.names))
        if self.on_unmatched not in {"warn", "error"}:
            raise ValueError(f"on_unmatched must be warn|error, got {self.on_unmatched!r}")

    def keep(self, meta: SampleMeta) -> bool:
        return meta.get("name") not in self.names

    def validate(self, corpus: str, observed: Mapping[str, Collection[Any]]) -> None:
        _report_unmatched(
            "ExcludeImages",
            corpus,
            self.names,
            observed.get("name", ()),
            on_unmatched=self.on_unmatched,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "filter": "ExcludeImages",
            "count": len(self.names),
            "names": sorted(self.names),
        }


def needed_fields(filters: Iterable[Filter]) -> frozenset[str]:
    """Every sample field the given filters read."""
    return frozenset(field for f in filters for field in f.needs)
