"""CelebAMask-HQ training data and the released makeup references for DiffAM."""

from __future__ import annotations

from dataclasses import dataclass
import io
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PIL import Image
import torch

from afrlib.afr.datasets import AFRTrainingDataset
from afrlib.utils.assets import get_asset

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus


IMAGE_SIZE = 256
REFERENCE_NAMES = ("XMY-060", "XYH-045", "XMY-254", "vRX912", "vFG137")
DEFAULT_REFERENCE = "XMY-060"
REQUIRED_REGIONS = ("l_eye", "r_eye", "l_lip", "u_lip", "skin")


def _rgb_pm1(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0)


def _to_mask01(image: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    resized = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0).div(255.0)


def _mask01(path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    with Image.open(path) as image:
        return _to_mask01(image, size)


class _FolderSource:
    """CelebAMask-HQ read from its released directory layout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.images_dir = root / "CelebA-HQ-img"
        self.masks_dir = root / "CelebAMask-HQ-mask-anno"
        self.attributes = root / "CelebAMask-HQ-attribute-anno.txt"
        for path in (self.images_dir, self.masks_dir):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if not self.attributes.is_file():
            raise FileNotFoundError(self.attributes)

    def names(self) -> list[str]:
        lines = self.attributes.read_text(encoding="utf-8").splitlines()[2:]
        return [line.split()[0] for line in lines if line.split()]

    def image(self, name: str) -> torch.Tensor:
        path = self.images_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return _rgb_pm1(image)

    def mask(self, name: str, region: str) -> Optional[torch.Tensor]:
        stem = Path(name).stem
        path = self.masks_dir / str(int(stem) // 2_000) / f"{stem.zfill(5)}_{region}.png"
        return _mask01(path) if path.is_file() else None

    def describe(self) -> dict[str, Any]:
        return {"root": str(self.root)}


class _CorpusSource:
    """CelebAMask-HQ read from a packed, filtered corpus."""

    def __init__(self, corpus: "TrainingCorpus") -> None:
        self.corpus = corpus
        self.keys = {meta.get("name"): meta.key for meta in corpus.metas()}

    def names(self) -> list[str]:
        return list(self.keys)

    def image(self, name: str) -> torch.Tensor:
        with Image.open(io.BytesIO(self.corpus.read(self.keys[name]))) as image:
            return _rgb_pm1(image)

    def mask(self, name: str, region: str) -> Optional[torch.Tensor]:
        try:
            payload = self.corpus.read(self.keys[name], f"mask_{region}")
        except KeyError:
            return None
        with Image.open(io.BytesIO(payload)) as image:
            return _to_mask01(image)

    def describe(self) -> dict[str, Any]:
        return self.corpus.get_config()


@dataclass(frozen=True)
class ReferenceTriplet:
    """One released DiffAM makeup reference.

    Attributes:
        name: Reference name.
        makeup: The made-up face, ``[3, 256, 256]`` RGB in ``[-1, 1]``.
        non_makeup: The same face without makeup, likewise.
        labels: ``[1, 256, 256]`` face-parsing map with classes 0-13.
    """

    name: str
    makeup: torch.Tensor
    non_makeup: torch.Tensor
    labels: torch.Tensor
    paths: tuple[Path, Path, Path]

    @classmethod
    def from_paths(
        cls,
        name: str,
        makeup_path: str | Path,
        non_makeup_path: str | Path,
        mask_path: str | Path,
    ) -> "ReferenceTriplet":
        """Load and validate a reference from its three image files."""
        if name not in REFERENCE_NAMES:
            raise ValueError(
                f"Unknown released DiffAM reference {name!r}; "
                f"expected one of {REFERENCE_NAMES}"
            )
        paths = tuple(
            Path(path).expanduser().resolve()
            for path in (makeup_path, non_makeup_path, mask_path)
        )
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        with Image.open(paths[0]) as image:
            makeup = _rgb_pm1(image)
        with Image.open(paths[1]) as image:
            non_makeup = _rgb_pm1(image)
        with Image.open(paths[2]) as image:
            labels_image = image.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
            )
            labels = torch.from_numpy(
                np.asarray(labels_image, dtype=np.float32).copy()
            ).unsqueeze(0)
        unique = set(int(value) for value in labels.unique().tolist())
        if not unique.issubset(set(range(14))):
            raise ValueError(f"Unexpected reference labels in {paths[2]}: {unique}")
        return cls(name, makeup, non_makeup, labels, paths)

    @classmethod
    def from_assets(cls, name: str = DEFAULT_REFERENCE) -> "ReferenceTriplet":
        """Load the released reference ``name`` from the asset cache."""
        return cls.from_paths(
            name,
            get_asset(f"diffam/{name}_m.png"),
            get_asset(f"diffam/{name}_nm.png"),
            get_asset(f"diffam/{name}_mask.png"),
        )


class CelebAMaskHQDataset(AFRTrainingDataset[dict[str, Any]]):
    """CelebAMask-HQ split used by DiffAM, with five regional masks per image."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        max_records: int | None = None,
    ) -> None:
        root = Path(root).expanduser().resolve()
        self._init_from_source(
            _FolderSource(root), split=split, max_records=max_records
        )
        self.source = f"dir:{root}"

    def _init_from_source(
        self,
        images: Any,
        *,
        split: str,
        max_records: int | None,
    ) -> None:
        """Select the upstream split: a seeded shuffle, then a 1,000-image window."""
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        names = images.names()
        random.Random(1234).shuffle(names)
        selected = names[4_000:5_000] if split == "train" else names[15_000:16_000]
        if max_records is not None:
            if max_records < 1:
                raise ValueError("max_records must be positive")
            selected = selected[:max_records]
        if not selected:
            raise ValueError(f"CelebAMask-HQ {split} split is empty")
        self.filenames = tuple(selected)
        self.images = images
        self.split = split

    @classmethod
    def from_assets(
        cls, *, split: str = "train", max_records: int | None = None
    ) -> "CelebAMaskHQDataset":
        """The ``celebamask_hq`` corpus from the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(
            get_corpus("celebamask_hq"), split=split, max_records=max_records
        )

    @classmethod
    def from_corpus(
        cls, corpus: "TrainingCorpus", **kwargs: Any
    ) -> "CelebAMaskHQDataset":
        """CelebAMask-HQ from a packed corpus in the attribute file's row order, so an unfiltered
        corpus yields the reference window and filters shift it.
        """
        dataset = cls.__new__(cls)
        dataset._init_from_source(
            _CorpusSource(corpus),
            split=kwargs.pop("split", "train"),
            max_records=kwargs.pop("max_records", None),
        )
        dataset.source = f"corpus:{corpus.name}"
        return dataset

    @classmethod
    def from_dir(cls, root: str | Path, **kwargs: Any) -> "CelebAMaskHQDataset":
        """A CelebAMask-HQ tree at ``root`` with the released layout."""
        return cls(root, **kwargs)

    def __len__(self) -> int:
        return len(self.filenames)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"split": self.split, **self.images.describe()})
        return config

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self.filenames[index]
        source = self.images.image(filename)
        regions = {
            region: self.images.mask(filename, region)
            for region in REQUIRED_REGIONS
        }
        valid = all(mask is not None for mask in regions.values())
        if not valid:
            zeros = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            masks = tuple(zeros.clone() for _ in range(5))
        else:
            left_eye = regions["l_eye"]
            right_eye = regions["r_eye"]
            lip = (regions["l_lip"] + regions["u_lip"]).clamp(0, 1)
            skin = regions["skin"] - left_eye - right_eye - lip
            face = skin.clone()

            for brow in ("l_brow", "r_brow"):
                brow_mask = self.images.mask(filename, brow)
                if brow_mask is not None:
                    skin = skin - brow_mask
                    face = face - brow_mask
            neck = self.images.mask(filename, "neck")
            if neck is not None:
                skin = skin + neck
            mouth = self.images.mask(filename, "mouth")
            if mouth is not None:
                skin = skin - mouth
                face = face - mouth
            masks = (
                left_eye,
                right_eye,
                lip,
                skin.clamp(0, 1),
                face.clamp(0, 1),
            )
        return {
            "image": source,
            "masks": masks,
            "valid": valid,
            "filename": filename,
        }


__all__ = [
    "CelebAMaskHQDataset",
    "DEFAULT_REFERENCE",
    "IMAGE_SIZE",
    "REFERENCE_NAMES",
    "ReferenceTriplet",
]
