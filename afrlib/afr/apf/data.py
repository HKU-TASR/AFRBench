"""Same-identity ``(image, enrolled)`` pair dataset for APF training."""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from afrlib.afr.datasets import AFRTrainingDataset, lmdb_env as _lmdb_env

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _to_tensor_255(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB array -> ``[3, H, W]`` float32 in ``[0, 255]``."""
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)


def _read_image_255(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        return _to_tensor_255(np.asarray(im))


def _identity_images(root: Path) -> Dict[str, List[Path]]:
    """Map identity -> sorted image paths, from ``root/<identity>/<img>`` or a flat
    ``root/<identity>_<img>`` layout.
    """
    root = Path(root)
    mapping: Dict[str, List[Path]] = {}
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if subdirs:
        for sub in subdirs:
            imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            if imgs:
                mapping[sub.name] = imgs
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                identity = p.stem.split("_")[0]
                mapping.setdefault(identity, []).append(p)
    return mapping


def _sample_subsets(
    mapping: Dict[str, List],
    num_subjects: int,
    images_per_subject: int,
    seed: int,
) -> Dict[str, List]:
    """Seeded ``num_subjects`` identities (all, if fewer qualify) with ``images_per_subject``
    images each, as in the paper's subset.
    """
    rng = np.random.RandomState(seed)
    eligible = sorted(k for k, v in mapping.items() if len(v) >= images_per_subject)
    if not eligible:
        raise RuntimeError(
            f"No identity has >= {images_per_subject} images "
            f"(largest: {max((len(v) for v in mapping.values()), default=0)})"
        )
    order = rng.permutation(len(eligible))
    picked = [eligible[i] for i in order[: min(num_subjects, len(eligible))]]
    subset: Dict[str, List] = {}
    for identity in sorted(picked):
        imgs = mapping[identity]
        choose = rng.permutation(len(imgs))[:images_per_subject]
        subset[identity] = [imgs[i] for i in sorted(choose)]
    return subset


def _make_pairs(
    subset: Dict[str, Sequence], seed: int
) -> List[Tuple[object, object]]:
    """Pair every image with another image of its identity, chosen by a random cyclic shift."""
    rng = np.random.RandomState(seed)
    pairs: List[Tuple[object, object]] = []
    for identity in sorted(subset):
        imgs = list(subset[identity])
        if len(imgs) < 2:
            continue
        shift = int(rng.randint(1, len(imgs)))
        for i, img in enumerate(imgs):
            enrolled = imgs[(i + shift) % len(imgs)]
            pairs.append((img, enrolled))
    rng.shuffle(pairs)
    return pairs


class APFPairDataset(AFRTrainingDataset):
    """Same-identity ``(image, enrolled)`` pairs of ``[3, 112, 112]`` RGB tensors in ``[0, 255]``;
    the paper uses 1,000 MS-Celeb-1M identities with 50 images each.

    Args:
        pairs: ``(image_ref, enrolled_ref)`` tuples.
        loader: Turns a ref into an image tensor.
    """

    def __init__(
        self,
        pairs,
        loader,
        size: int = 112,
        *,
        source: str,
        asset_key: Optional[str] = None,
        num_subjects: Optional[int] = None,
        images_per_subject: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if not pairs:
            raise ValueError("APFPairDataset requires at least one pair")
        self.pairs = list(pairs)
        self._loader = loader
        self.size = int(size)
        self.source = source
        self.asset_key = asset_key
        self.num_subjects = num_subjects
        self.images_per_subject = images_per_subject
        self.seed = int(seed)

    def __getstate__(self):
        # LMDB handles are not picklable; each DataLoader worker reopens its own.
        state = dict(self.__dict__)
        state.pop("_lmdb_env", None)
        return state

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_ref, enrolled_ref = self.pairs[index]
        return self._loader(img_ref), self._loader(enrolled_ref)

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "num_subjects": self.num_subjects,
                "images_per_subject": self.images_per_subject,
                "seed": self.seed,
                "size": self.size,
                "pairing": "cyclic_shift_within_identity",
            }
        )
        return config

    @classmethod
    def from_dir(
        cls,
        root,
        *,
        num_subjects: int = 1000,
        images_per_subject: int = 50,
        seed: int = 0,
        size: int = 112,
        source: Optional[str] = None,
        asset_key: Optional[str] = None,
    ) -> "APFPairDataset":
        """The paper's subset (default 1000 subjects x 50 images) from a folder-per-identity tree."""
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"MS1M dataset not found: {root}")
        mapping = _identity_images(root)
        if not mapping:
            raise RuntimeError(f"No identity images found under {root}")
        subset = _sample_subsets(mapping, num_subjects, images_per_subject, seed)
        pairs = _make_pairs(subset, seed)
        return cls(
            pairs,
            lambda p: _read_image_255(Path(p), size),
            size=size,
            source=source or f"dir:{root}",
            asset_key=asset_key,
            num_subjects=len(subset),
            images_per_subject=images_per_subject,
            seed=seed,
        )

    @classmethod
    def from_assets(cls, **kwargs) -> "APFPairDataset":
        """The registered MS-Celeb-1M corpus, via the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("ms1m"), **kwargs)

    @classmethod
    def from_corpus(
        cls,
        corpus: "TrainingCorpus",
        *,
        num_subjects: int = 1000,
        images_per_subject: int = 50,
        seed: int = 0,
        size: int = 112,
    ) -> "APFPairDataset":
        """The paper's subset, sampled from a packed corpus after its filters are applied."""
        mapping: Dict[str, List[str]] = {}
        for meta in corpus.metas(need=("identity",)):
            mapping.setdefault(str(meta.identity), []).append(meta.key)
        if not mapping:
            raise RuntimeError(f"Corpus {corpus.name!r} is empty after filtering")
        subset = _sample_subsets(mapping, num_subjects, images_per_subject, seed)

        def load(key: str) -> torch.Tensor:
            with Image.open(io.BytesIO(corpus.read(key))) as image:
                array = np.asarray(image.convert("RGB"))
            if array.shape[0] != size or array.shape[1] != size:
                array = np.asarray(
                    Image.fromarray(array).resize((size, size), Image.BILINEAR)
                )
            return _to_tensor_255(array)

        return cls(
            _make_pairs(subset, seed),
            load,
            size=size,
            source=f"corpus:{corpus.name}",
            num_subjects=len(subset),
            images_per_subject=images_per_subject,
            seed=seed,
        )

    @classmethod
    def from_lmdb(
        cls,
        lmdb_path,
        *,
        num_subjects: Optional[int] = None,
        images_per_subject: int = 50,
        seed: int = 0,
        size: int = 112,
    ) -> "APFPairDataset":
        """An LMDB of aligned 112x112 faces with ``length``, ``label-{i}`` and ``image-{i}`` keys,
        opened once per path and process.
        """
        lmdb_path = str(lmdb_path)
        with _lmdb_env(lmdb_path).begin() as txn:
            n = int(txn.get(b"length").decode())
            labels = np.array(
                [int(txn.get(f"label-{i}".encode()).decode()) for i in range(n)],
                dtype=np.int64,
            )
        mapping: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels.tolist()):
            mapping.setdefault(str(label), []).append(idx)
        subset = _sample_subsets(
            mapping,
            num_subjects if num_subjects is not None else len(mapping),
            images_per_subject,
            seed,
        )
        pairs = _make_pairs(subset, seed)

        def _load(idx: int) -> torch.Tensor:
            env = _lmdb_env(lmdb_path)
            with env.begin() as txn:
                arr = np.array(
                    Image.open(io.BytesIO(txn.get(f"image-{idx}".encode()))).convert("RGB")
                )
            if arr.shape[0] != size or arr.shape[1] != size:
                arr = np.asarray(
                    Image.fromarray(arr).resize((size, size), Image.BILINEAR)
                )
            return _to_tensor_255(arr)

        return cls(
            pairs,
            _load,
            size=size,
            source=f"lmdb:{lmdb_path}",
            asset_key=None,
            num_subjects=len(subset),
            images_per_subject=images_per_subject,
            seed=seed,
        )
