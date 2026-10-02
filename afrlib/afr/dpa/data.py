"""Face-classification dataset for DPA's snapshot training; items are ``(img, label)``."""
from __future__ import annotations

import io
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from PIL import Image

from afrlib.afr.datasets import AFRTrainingDataset, lmdb_env as _lmdb_env

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_Ref = TypeVar("_Ref")


def _to_tensor_255(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB array -> contiguous ``[3, H, W]`` float32 in ``[0, 255]``."""
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).contiguous()


def _read_image_255(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return _to_tensor_255(np.asarray(im))


def _identity_images(root: Path) -> Dict[str, List[Path]]:
    """Map identity -> sorted image paths, from a ``root/<identity>/<img>`` or
    ``root/<group>/<identity>/<img>`` tree; identities are keyed by folder name.
    """
    root = Path(root)
    mapping: Dict[str, List[Path]] = {}
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    for sub in subdirs:
        if any(p.suffix.lower() in _IMAGE_EXTS for p in sub.iterdir() if p.is_file()):
            imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            if imgs:
                mapping.setdefault(sub.name, []).extend(imgs)
        else:
            # One extra level (e.g. race/identity/img).
            for id_dir in sorted(p for p in sub.iterdir() if p.is_dir()):
                imgs = sorted(p for p in id_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
                if imgs:
                    mapping.setdefault(id_dir.name, []).extend(imgs)
    return mapping


def _assign_labels(
    mapping: Dict[str, List[_Ref]],
    num_subjects: Optional[int],
    images_per_subject: Optional[int],
    seed: int,
) -> Tuple[List[_Ref], List[int]]:
    """Sample a seeded subset of ``mapping`` (identity -> sample references) and assign
    contiguous class labels; with ``images_per_subject``, smaller identities are skipped.
    """
    rng = np.random.RandomState(seed)
    identities = sorted(mapping)
    if images_per_subject is not None:
        identities = [k for k in identities if len(mapping[k]) >= images_per_subject]
        if not identities:
            raise RuntimeError(
                f"No identity has >= {images_per_subject} images "
                f"(largest: {max((len(v) for v in mapping.values()), default=0)})"
            )
    if num_subjects is not None and num_subjects < len(identities):
        order = rng.permutation(len(identities))
        identities = sorted(identities[i] for i in order[:num_subjects])

    samples: List[_Ref] = []
    labels: List[int] = []
    for label, identity in enumerate(identities):
        refs = mapping[identity]
        if images_per_subject is not None and len(refs) > images_per_subject:
            choose = rng.permutation(len(refs))[:images_per_subject]
            refs = [refs[i] for i in sorted(choose)]
        for ref in refs:
            samples.append(ref)
            labels.append(label)
    return samples, labels


def _resize_255(array: np.ndarray, size: int) -> torch.Tensor:
    if array.shape[0] != size or array.shape[1] != size:
        array = np.asarray(
            Image.fromarray(array).resize((size, size), Image.BILINEAR)
        )
    return _to_tensor_255(array)


class FaceClassDataset(AFRTrainingDataset):
    """``(img, label)`` pairs for snapshot training; the paper uses BUPT-Balancedface, this the
    ``ms1m`` corpus.

    Args:
        samples: Sample references, parallel to ``labels``.
        labels: Contiguous class labels.
        loader: Turns a reference into a ``[3, size, size]`` RGB tensor in ``[0, 255]``.
    """

    def __init__(
        self,
        samples: Sequence,
        labels: Sequence[int],
        loader: Callable[[object], torch.Tensor],
        size: int,
        num_classes: int,
        *,
        source: str,
        asset_key: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("FaceClassDataset requires at least one sample")
        self.samples = list(samples)
        self.labels = [int(v) for v in labels]
        self._loader = loader
        self.size = int(size)
        self.num_classes = int(num_classes)
        self.source = source
        self.asset_key = asset_key
        self.seed = int(seed)

    def __getstate__(self):
        # Drop the LMDB handle so the dataset stays picklable across workers.
        state = dict(self.__dict__)
        state.pop("_lmdb_env", None)
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._loader(self.samples[index]), self.labels[index]

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "size": self.size,
                "seed": self.seed,
            }
        )
        return config

    # Constructors

    @classmethod
    def from_dir(
        cls,
        root,
        *,
        num_subjects: Optional[int] = None,
        images_per_subject: Optional[int] = None,
        seed: int = 0,
        size: int = 112,
        source: Optional[str] = None,
        asset_key: Optional[str] = None,
    ) -> "FaceClassDataset":
        """Folder-per-identity tree at ``root`` (MS1M / BUPT-Balancedface / CBFace)."""
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Face dataset not found: {root}")
        mapping = _identity_images(root)
        if not mapping:
            raise RuntimeError(f"No identity images found under {root}")
        samples, labels = _assign_labels(mapping, num_subjects, images_per_subject, seed)
        return cls(
            samples,
            labels,
            lambda p: _read_image_255(Path(p), size),
            size,
            num_classes=len(set(labels)),
            source=source or f"dir:{root}",
            asset_key=asset_key,
            seed=seed,
        )

    @classmethod
    def from_assets(cls, **kwargs) -> "FaceClassDataset":
        """The default DPO corpus (MS-Celeb-1M) via the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("ms1m"), **kwargs)

    @classmethod
    def from_corpus(
        cls,
        corpus: "TrainingCorpus",
        *,
        num_subjects: Optional[int] = None,
        images_per_subject: Optional[int] = None,
        seed: int = 0,
        size: int = 112,
    ) -> "FaceClassDataset":
        """A packed labelled corpus, with labels reassigned over the identities that pass its
        filters, so no class is left empty.
        """
        mapping: Dict[str, List[str]] = {}
        for meta in corpus.metas(need=("identity",)):
            mapping.setdefault(str(meta.identity), []).append(meta.key)
        if not mapping:
            raise RuntimeError(f"Corpus {corpus.name!r} is empty after filtering")
        samples, labels = _assign_labels(
            mapping, num_subjects, images_per_subject, seed
        )

        def load(key: str) -> torch.Tensor:
            with Image.open(io.BytesIO(corpus.read(key))) as image:
                return _resize_255(np.asarray(image.convert("RGB")), size)

        return cls(
            samples,
            labels,
            load,
            size,
            num_classes=len(set(labels)),
            source=f"corpus:{corpus.name}",
            seed=seed,
        )

    @classmethod
    def from_lmdb(
        cls,
        lmdb_path,
        *,
        num_subjects: Optional[int] = None,
        images_per_subject: Optional[int] = None,
        seed: int = 0,
        size: int = 112,
    ) -> "FaceClassDataset":
        """Local facescrub/lfw LMDB (112x112 aligned)."""
        lmdb_path = str(lmdb_path)
        with _lmdb_env(lmdb_path).begin() as txn:
            n = int(txn.get(b"length").decode())
            labels_arr = np.array(
                [int(txn.get(f"label-{i}".encode()).decode()) for i in range(n)],
                dtype=np.int64,
            )
        mapping: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels_arr.tolist()):
            mapping.setdefault(str(label), []).append(idx)
        samples, labels = _assign_labels(
            mapping, num_subjects, images_per_subject, seed
        )

        def _load(idx: int) -> torch.Tensor:
            env = _lmdb_env(lmdb_path)
            with env.begin() as txn:
                with Image.open(
                    io.BytesIO(txn.get(f"image-{idx}".encode()))
                ) as image:
                    return _resize_255(np.asarray(image.convert("RGB")), size)

        return cls(
            samples,
            labels,
            _load,
            size,
            num_classes=len(set(labels)),
            source=f"lmdb:{lmdb_path}",
            asset_key=None,
            seed=seed,
        )
