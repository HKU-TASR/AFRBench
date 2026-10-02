"""Map-style dataset over the packed MillionCelebs LMDB, for the full-noise cache writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from afrbench.dataset import decode_image
from afrlib.data.lmdb_io import lmdb_env, read_count


class MillionCelebsCrops(Dataset):
    """Packed-LMDB crops in sample order from index ``start``, as ``(image, key, identity)``
    items in ``[0, 255]``, or ``[0, 1]`` with ``scale_01``.
    """

    def __init__(
        self, lmdb_path: "str | Path", *, start: int = 0, scale_01: bool = False
    ) -> None:
        self.lmdb_path = Path(lmdb_path)
        self.start = int(start)
        self.scale_01 = bool(scale_01)
        self.total = read_count(self.lmdb_path)
        if self.start < 0 or self.start > self.total:
            raise ValueError(f"start {self.start} out of range for {self.total} samples")
        self._env = None

    def __len__(self) -> int:
        return self.total - self.start

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def _txn_env(self):
        if self._env is None:
            self._env = lmdb_env(self.lmdb_path)
        return self._env

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, str]:
        absolute = self.start + int(index)
        env = self._txn_env()
        with env.begin() as txn:
            raw = txn.get(f"meta/sample/{absolute:08d}".encode("ascii"))
            if raw is None:
                raise KeyError(absolute)
            record = json.loads(raw)
            key = str(record["key"])
            payload = txn.get(f"{key}/image".encode("utf-8"))
            if payload is None:
                raise KeyError(key)
        tensor = decode_image(payload)
        if self.scale_01:
            tensor = tensor / 255.0
        return tensor, key, str(record.get("identity") or "")


def collate_crops(
    batch: Sequence[Tuple[torch.Tensor, str, str]],
) -> Tuple[torch.Tensor, List[str], List[str]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    keys = [item[1] for item in batch]
    identities = [item[2] for item in batch]
    return images, keys, identities
