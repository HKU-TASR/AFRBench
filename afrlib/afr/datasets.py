"""Base class for the training datasets of AFRLib's trainable methods."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypeVar

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

T_co = TypeVar("T_co", covariant=True)

# LMDB cannot open one environment twice per process, so read-only environments are shared
# here, keyed by pid so DataLoader workers reopen after a fork.
_LMDB_ENVS: Dict[Tuple[str, int], Any] = {}


def lmdb_env(path: "str | os.PathLike[str]"):
    """Shared read-only LMDB environment for ``path`` in this process."""
    import lmdb

    key = (str(path), os.getpid())
    env = _LMDB_ENVS.get(key)
    if env is None:
        env = lmdb.open(
            str(path), readonly=True, lock=False, readahead=False, meminit=False
        )
        _LMDB_ENVS[key] = env
    return env


class AFRTrainingDataset(Dataset[T_co], ABC):
    """Base class of a training dataset read by ``AFRMethod.train``: method-specific items from
    a registry corpus, the released default corpus, a directory or an LMDB.

    Example:
        ::

            SomeDataset.from_corpus(corpus)          # registry corpus, filtered
            SomeDataset.from_assets()                # released default corpus
            SomeDataset.from_dir("/path/to/corpus")  # custom corpus
    """

    #: How this instance was built, e.g. ``"assets"``, ``"dir:<path>"``,
    #: ``"lmdb:<path>"`` or ``"corpus:<name>"``.
    source: str = "custom"

    #: Asset key when the corpus came from the cache, else ``None``.
    asset_key: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_assets(cls, **kwargs: Any) -> "AFRTrainingDataset":
        """Build from the released corpus registered in the asset cache."""

    @classmethod
    @abstractmethod
    def from_dir(
        cls, root: "str | os.PathLike[str]", **kwargs: Any
    ) -> "AFRTrainingDataset":
        """Build from a custom corpus on disk."""

    @classmethod
    def from_corpus(
        cls, corpus: "TrainingCorpus", **kwargs: Any
    ) -> "AFRTrainingDataset":
        """Build from a packed, filtered corpus, which supplies the samples its filters allow while
        this class pairs, crops and aligns them; unsupported unless overridden.
        """
        raise NotImplementedError(
            f"{cls.__name__} cannot read a registry corpus; use from_assets() "
            "or from_dir()."
        )

    @classmethod
    def from_lmdb(
        cls, lmdb_path: "str | os.PathLike[str]", **kwargs: Any
    ) -> "AFRTrainingDataset":
        """Build from a method-specific LMDB; unsupported unless overridden."""
        raise NotImplementedError(
            f"{cls.__name__} has no LMDB reader; use from_assets() or from_dir()."
        )

    def get_config(self) -> Dict[str, Any]:
        """Provenance describing how this corpus was built."""
        return {
            "dataset": type(self).__name__,
            "source": self.source,
            "asset_key": self.asset_key,
            "num_samples": len(self),
        }


def dataset_config(dataset: Any) -> Optional[Dict[str, Any]]:
    """Best-effort provenance for ``config.json`` from a dataset, a ``DataLoader`` wrapping one,
    or ``None``.
    """
    if dataset is None:
        return None
    inner = getattr(dataset, "dataset", dataset)  # unwrap a DataLoader
    get_config = getattr(inner, "get_config", None)
    if callable(get_config):
        return get_config()
    try:
        return {"dataset": type(inner).__name__, "num_samples": len(inner)}
    except TypeError:
        return {"dataset": type(inner).__name__}
