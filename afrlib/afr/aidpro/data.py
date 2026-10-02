"""VGGFace2 faces for AIDPro's stage-two training."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from afrlib.afr.datasets import AFRTrainingDataset

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

IMAGE_SIZE = 224


def _identity_images(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.jpg"))


class VGGFace2Dataset(AFRTrainingDataset[torch.Tensor]):
    """Aligned faces as ``[3, 224, 224]`` RGB tensors in ``[0, 255]``; other sizes are resized
    bicubically, which cannot restore missing detail.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        images = _identity_images(self.root)
        if not images:
            raise ValueError(
                "VGGFace2 must contain one-level identity folders with JPEG "
                f"images; none found under {self.root}"
            )
        self._paths = tuple(self._limit(images, max_samples))
        self._corpus: Optional["TrainingCorpus"] = None
        self._keys: tuple[str, ...] = ()
        self.source = f"dir:{self.root}"

    @staticmethod
    def _limit(items: list[Any], max_samples: Optional[int]) -> list[Any]:
        if max_samples is None:
            return items
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        return items[:max_samples]

    @classmethod
    def from_assets(cls, **kwargs: Any) -> "VGGFace2Dataset":
        """The ``vggface2`` corpus from the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("vggface2"), **kwargs)

    @classmethod
    def from_corpus(
        cls,
        corpus: "TrainingCorpus",
        *,
        max_samples: Optional[int] = None,
    ) -> "VGGFace2Dataset":
        """Aligned faces from a packed corpus, honouring its filters."""
        dataset = cls.__new__(cls)
        dataset.root = None
        dataset._paths = ()
        dataset._corpus = corpus
        dataset._keys = tuple(cls._limit(list(corpus.allowed_keys()), max_samples))
        if not dataset._keys:
            raise ValueError(f"Corpus {corpus.name!r} is empty after filtering")
        dataset.source = f"corpus:{corpus.name}"
        return dataset

    @classmethod
    def from_dir(
        cls, root: "str | os.PathLike[str]", **kwargs: Any
    ) -> "VGGFace2Dataset":
        """One-level identity folders of JPEGs at ``root``."""
        return cls(root, **kwargs)

    def __len__(self) -> int:
        return len(self._keys) if self._corpus is not None else len(self._paths)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        if self._corpus is not None:
            config["corpus"] = self._corpus.get_config()
        else:
            config["root"] = str(self.root)
        config["num_samples"] = len(self)
        return config

    def _open(self, index: int) -> Image.Image:
        if self._corpus is not None:
            return Image.open(io.BytesIO(self._corpus.read(self._keys[index])))
        return Image.open(self._paths[index])

    def __getitem__(self, index: int) -> torch.Tensor:
        with self._open(index) as image:
            image = image.convert("RGB")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                image = TF.resize(
                    image,
                    [IMAGE_SIZE, IMAGE_SIZE],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            return TF.pil_to_tensor(image).to(torch.float32)


__all__ = [
    "IMAGE_SIZE",
    "VGGFace2Dataset",
]
