"""Training datasets of AdvCloak's two GAN stages; items are ``[3, 112, 112]`` in ``[0, 255]``."""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from afrlib.afr.datasets import AFRTrainingDataset, lmdb_env as _lmdb_env

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

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
    """Map identity -> sorted image paths from a folder-per-identity tree."""
    root = Path(root)
    mapping: Dict[str, List[Path]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if imgs:
            mapping[sub.name] = imgs
    return mapping


def _corpus_identities(corpus: "TrainingCorpus") -> Dict[str, List[str]]:
    """Map identity -> the corpus keys that survived its filters."""
    mapping: Dict[str, List[str]] = {}
    for meta in corpus.metas(need=("identity",)):
        mapping.setdefault(str(meta.identity), []).append(meta.key)
    if not mapping:
        raise RuntimeError(f"Corpus {corpus.name!r} is empty after filtering")
    return mapping


def _corpus_loader(
    corpus: "TrainingCorpus", size: int
) -> Callable[[str], torch.Tensor]:
    def load(key: str) -> torch.Tensor:
        with Image.open(io.BytesIO(corpus.read(key))) as image:
            array = np.asarray(image.convert("RGB"))
        if array.shape[0] != size or array.shape[1] != size:
            array = np.asarray(
                Image.fromarray(array).resize((size, size), Image.BILINEAR)
            )
        return _to_tensor_255(array)

    return load


class _FaceDataset(AFRTrainingDataset):
    """Flat, lazily loaded image dataset (base of the stage-1 dataset)."""

    def __init__(
        self,
        samples: Sequence,
        loader: Callable[[object], torch.Tensor],
        size: int,
        *,
        source: str,
        asset_key: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("AdvCloak datasets require at least one sample")
        self.samples = list(samples)
        self._loader = loader
        self.size = int(size)
        self.source = source
        self.asset_key = asset_key
        self.seed = int(seed)

    def __getstate__(self):
        # Keep LMDB handles out of pickled state.
        state = dict(self.__dict__)
        state.pop("_lmdb_env", None)
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._loader(self.samples[index])

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update({"size": self.size, "seed": self.seed})
        return config


def _sample_flat(
    samples: List, num_images: Optional[int], seed: int
) -> List:
    """Deterministically sample up to ``num_images`` images from ``samples``."""
    if num_images is None or num_images >= len(samples):
        return list(samples)
    rng = np.random.RandomState(seed)
    choose = rng.permutation(len(samples))[:num_images]
    return [samples[i] for i in sorted(choose)]


class AdvCloakStage1Dataset(_FaceDataset):
    """Flat face set for stage 1 (one mask per image); the paper samples 50,000 images."""

    @classmethod
    def from_dir(
        cls,
        root,
        *,
        num_images: Optional[int] = 50_000,
        seed: int = 0,
        size: int = 112,
        source: Optional[str] = None,
        asset_key: Optional[str] = None,
    ) -> "AdvCloakStage1Dataset":
        """A folder-per-identity tree at ``root``, sampled flat."""
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CASIA dataset not found: {root}")
        mapping = _identity_images(root)
        if not mapping:
            raise RuntimeError(f"No identity images found under {root}")
        flat = [p for imgs in mapping.values() for p in imgs]
        samples = _sample_flat(flat, num_images, seed)
        return cls(
            samples,
            lambda p: _read_image_255(Path(p), size),
            size,
            source=source or f"dir:{root}",
            asset_key=asset_key,
            seed=seed,
        )

    @classmethod
    def from_assets(cls, **kwargs) -> "AdvCloakStage1Dataset":
        """The registered CASIA-WebFace corpus, via the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("casia_webface"), **kwargs)

    @classmethod
    def from_corpus(
        cls,
        corpus: "TrainingCorpus",
        *,
        num_images: Optional[int] = 50_000,
        seed: int = 0,
        size: int = 112,
    ) -> "AdvCloakStage1Dataset":
        """Flat sampling over the corpus images that pass its filters."""
        samples = _sample_flat(list(corpus.allowed_keys()), num_images, seed)
        if not samples:
            raise RuntimeError(f"Corpus {corpus.name!r} is empty after filtering")
        return cls(
            samples,
            _corpus_loader(corpus, size),
            size,
            source=f"corpus:{corpus.name}",
            seed=seed,
        )

    @classmethod
    def from_lmdb(
        cls,
        lmdb_path,
        *,
        num_images: Optional[int] = 50_000,
        seed: int = 0,
        size: int = 112,
    ) -> "AdvCloakStage1Dataset":
        """A 112x112 aligned LMDB with ``length`` and ``image-<i>`` keys."""
        lmdb_path = str(lmdb_path)
        with _lmdb_env(lmdb_path).begin() as txn:
            n = int(txn.get(b"length").decode())
        idxs = list(range(n))
        samples = _sample_flat(idxs, num_images, seed)

        def _load(idx: int) -> torch.Tensor:
            env = _lmdb_env(lmdb_path)
            with env.begin() as txn:
                arr = np.array(
                    Image.open(io.BytesIO(txn.get(f"image-{idx}".encode()))).convert("RGB")
                )
            if arr.shape[0] != size or arr.shape[1] != size:
                arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
            return _to_tensor_255(arr)

        return cls(
            samples, _load, size, source=f"lmdb:{lmdb_path}", asset_key=None, seed=seed
        )


class AdvCloakStage2Dataset(AFRTrainingDataset):
    """Identity-major dataset for stage 2 (one mask per person): item
    ``i * images_per_subject + j`` is image ``j`` of identity ``i``.
    """

    def __init__(
        self,
        groups: Sequence[Sequence],
        loader: Callable[[object], torch.Tensor],
        size: int,
        *,
        source: str,
        asset_key: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        if not groups:
            raise ValueError("AdvCloakStage2Dataset requires at least one identity")
        self.groups = [list(g) for g in groups]
        self._loader = loader
        self.size = int(size)
        self.source = source
        self.asset_key = asset_key
        self.seed = int(seed)
        self.images_per_subject = len(self.groups[0])

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_lmdb_env", None)
        return state

    def __len__(self) -> int:
        return len(self.groups) * self.images_per_subject

    def __getitem__(self, index: int) -> torch.Tensor:
        identity = index // self.images_per_subject
        within = index % self.images_per_subject
        return self._loader(self.groups[identity][within])

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "num_subjects": len(self.groups),
                "images_per_subject": self.images_per_subject,
                "size": self.size,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def _from_mapping(
        cls,
        mapping: Dict[str, List],
        loader: Callable[[object], torch.Tensor],
        num_subjects: Optional[int],
        images_per_subject: int,
        seed: int,
        size: int,
        source: str,
        asset_key: Optional[str],
    ) -> "AdvCloakStage2Dataset":
        rng = np.random.RandomState(seed)
        eligible = sorted(k for k, v in mapping.items() if len(v) >= images_per_subject)
        if not eligible:
            raise RuntimeError(
                f"No identity has >= {images_per_subject} images "
                f"(largest: {max((len(v) for v in mapping.values()), default=0)})"
            )
        if num_subjects is not None and num_subjects < len(eligible):
            order = rng.permutation(len(eligible))
            eligible = sorted(eligible[i] for i in order[:num_subjects])
        groups: List[List] = []
        for identity in eligible:
            imgs = mapping[identity]
            if len(imgs) > images_per_subject:
                choose = rng.permutation(len(imgs))[:images_per_subject]
                imgs = [imgs[i] for i in sorted(choose)]
            groups.append(list(imgs))
        return cls(groups, loader, size, source=source, asset_key=asset_key, seed=seed)

    @classmethod
    def from_dir(
        cls,
        root,
        *,
        num_subjects: Optional[int] = 3_000,
        images_per_subject: int = 16,
        seed: int = 0,
        size: int = 112,
        source: Optional[str] = None,
        asset_key: Optional[str] = None,
    ) -> "AdvCloakStage2Dataset":
        """A folder-per-identity tree at ``root``."""
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CASIA dataset not found: {root}")
        mapping = _identity_images(root)
        if not mapping:
            raise RuntimeError(f"No identity images found under {root}")
        return cls._from_mapping(
            mapping,
            lambda p: _read_image_255(Path(p), size),
            num_subjects,
            images_per_subject,
            seed,
            size,
            source or f"dir:{root}",
            asset_key,
        )

    @classmethod
    def from_assets(cls, **kwargs) -> "AdvCloakStage2Dataset":
        """The registered CASIA-WebFace corpus, via the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("casia_webface"), **kwargs)

    @classmethod
    def from_corpus(
        cls,
        corpus: "TrainingCorpus",
        *,
        num_subjects: Optional[int] = 3_000,
        images_per_subject: int = 16,
        seed: int = 0,
        size: int = 112,
    ) -> "AdvCloakStage2Dataset":
        """Identity-grouped sampling over the corpus identities that pass its filters."""
        return cls._from_mapping(
            _corpus_identities(corpus),
            _corpus_loader(corpus, size),
            num_subjects,
            images_per_subject,
            seed,
            size,
            f"corpus:{corpus.name}",
            None,
        )

    @classmethod
    def from_lmdb(
        cls,
        lmdb_path,
        *,
        num_subjects: Optional[int] = 3_000,
        images_per_subject: int = 16,
        seed: int = 0,
        size: int = 112,
    ) -> "AdvCloakStage2Dataset":
        """A 112x112 aligned LMDB with ``length``, ``image-<i>`` and ``label-<i>`` keys."""
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

        def _load(idx: int) -> torch.Tensor:
            env = _lmdb_env(lmdb_path)
            with env.begin() as txn:
                arr = np.array(
                    Image.open(io.BytesIO(txn.get(f"image-{idx}".encode()))).convert("RGB")
                )
            if arr.shape[0] != size or arr.shape[1] != size:
                arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
            return _to_tensor_255(arr)

        return cls._from_mapping(
            mapping,
            _load,
            num_subjects,
            images_per_subject,
            seed,
            size,
            f"lmdb:{lmdb_path}",
            None,
        )
