"""MT-dataset readers for AMT-GAN training, backed by an LMDB or an :mod:`afrlib.data` corpus."""

from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import get_worker_info

from afrlib.afr.datasets import AFRTrainingDataset


SCHEMA_VERSION = "amt-gan-mt-lmdb-v1"
_SPLIT_MAKEUP = "makeup"
_SPLIT_NON_MAKEUP = "non-makeup"
_RECORD_FIELDS = ("image", "mask", "landmarks", "name")
_ENV_CACHE: Dict[Tuple[int, str, bool, int], Tuple[Any, int]] = {}
_ENV_CACHE_LOCK = threading.Lock()


def _environment_key(
    path: Path, readahead: bool, max_readers: int
) -> Tuple[int, str, bool, int]:
    return os.getpid(), str(path), readahead, max_readers


def _drop_inherited_environments() -> None:
    """Close cache entries copied by fork before a child opens its own."""

    pid = os.getpid()
    inherited = [key for key in _ENV_CACHE if key[0] != pid]
    if not inherited:
        return
    environments = {id(_ENV_CACHE[key][0]): _ENV_CACHE[key][0] for key in inherited}
    for environment in environments.values():
        environment.close()
    for key in inherited:
        _ENV_CACHE.pop(key, None)


def _acquire_environment(
    path: Path, readahead: bool, max_readers: int
) -> Tuple[Any, Tuple[int, str, bool, int]]:
    key = _environment_key(path, readahead, max_readers)
    with _ENV_CACHE_LOCK:
        _drop_inherited_environments()
        cached = _ENV_CACHE.get(key)
        if cached is not None:
            environment, references = cached
            _ENV_CACHE[key] = (environment, references + 1)
            return environment, key

        environment = lmdb.open(
            str(path),
            subdir=path.is_dir(),
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=max_readers,
        )
        _ENV_CACHE[key] = (environment, 1)
        return environment, key


def _release_environment(
    environment: Any, key: Optional[Tuple[int, str, bool, int]]
) -> None:
    if key is None:
        return
    with _ENV_CACHE_LOCK:
        cached = _ENV_CACHE.get(key)
        if cached is None or cached[0] is not environment:
            return
        references = cached[1] - 1
        if references == 0:
            environment.close()
            _ENV_CACHE.pop(key, None)
        else:
            _ENV_CACHE[key] = (environment, references)


@dataclass(frozen=True)
class AMTGANRawRecord:
    """One decoded LMDB record before AMT-GAN training preprocessing."""

    name: str
    image: Image.Image
    mask: np.ndarray
    landmarks: np.ndarray


class AMTGANTrainingPreprocessor:
    """Training tensors as the released loader makes them: ``[-1, 1]`` images, three region
    masks and landmark-distance maps, from the stored masks and 68 landmarks.
    """

    def __init__(
        self,
        image_size: int = 256,
        diff_size: int = 64,
        lip_classes: Tuple[int, int] = (7, 9),
        face_classes: Tuple[int, int] = (1, 6),
        eye_margin: int = 16,
    ) -> None:
        if image_size <= 0 or diff_size <= 0:
            raise ValueError("image_size and diff_size must be positive")
        if eye_margin < 0:
            raise ValueError("eye_margin must be non-negative")

        self.image_size = int(image_size)
        self.diff_size = int(diff_size)
        self.lip_classes = tuple(int(value) for value in lip_classes)
        self.face_classes = tuple(int(value) for value in face_classes)
        self.eye_margin = int(eye_margin)

        coordinates = torch.arange(self.image_size, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self._coordinate_grid = torch.cat(
            (
                grid_y.unsqueeze(0).repeat(68, 1, 1),
                grid_x.unsqueeze(0).repeat(68, 1, 1),
            ),
            dim=0,
        )

    @staticmethod
    def _resampling(name: str) -> int:
        namespace = getattr(Image, "Resampling", Image)
        return int(getattr(namespace, name))

    def _copy_eye_region(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        landmarks: np.ndarray,
    ) -> None:
        # Preserve the released slicing semantics, including negative or
        # out-of-frame indices for the few malformed landmark records.
        x0 = int(np.min(landmarks[:, 1])) - self.eye_margin
        y0 = int(np.min(landmarks[:, 0])) - self.eye_margin
        x1 = int(np.max(landmarks[:, 1])) + self.eye_margin + 1
        y1 = int(np.max(landmarks[:, 0])) + self.eye_margin + 1
        target[:, :, y0:y1, x0:x1] = source[:, :, y0:y1, x0:x1]
        source[:, :, y0:y1, x0:x1] = 0

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if landmarks.shape != (68, 2):
            raise ValueError(
                f"Expected landmarks with shape (68, 2), got {landmarks.shape}"
            )
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2-D semantic mask, got {mask.shape}")

        source_width, source_height = image.size
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Invalid source image size: {image.size}")
        scaled_landmarks = np.asarray(landmarks, dtype=np.float32).copy()
        scaled_landmarks *= np.asarray(
            (
                self.image_size / source_width,
                self.image_size / source_height,
            ),
            dtype=np.float32,
        )

        resized_image = image.resize(
            (self.image_size, self.image_size),
            self._resampling("BILINEAR"),
        )
        image_array = np.asarray(resized_image, dtype=np.float32).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = image_tensor.div(255.0).sub(0.5).div(0.5)

        # MT images are 361x361 and masks 321x321; as in the released loader,
        # each is resized to 256 independently.
        mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
        mask_image = mask_image.resize(
            (self.image_size, self.image_size),
            self._resampling("NEAREST"),
        )
        mask_tensor = torch.from_numpy(np.asarray(mask_image).copy())
        mask_tensor = mask_tensor.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mask_lip = (mask_tensor == self.lip_classes[0]).float()
        mask_lip += (mask_tensor == self.lip_classes[1]).float()
        mask_face = (mask_tensor == self.face_classes[0]).float()
        mask_face += (mask_tensor == self.face_classes[1]).float()

        mask_eyes = torch.zeros_like(mask_tensor)
        self._copy_eye_region(mask_eyes, mask_face, scaled_landmarks[42:48])
        self._copy_eye_region(mask_eyes, mask_face, scaled_landmarks[36:42])
        masks = torch.cat((mask_lip, mask_face, mask_eyes), dim=0)

        landmark_vector = torch.from_numpy(
            scaled_landmarks.transpose(1, 0).reshape(136, 1, 1)
        )
        distances = (self._coordinate_grid - landmark_vector).unsqueeze(0)

        mask_weights = F.interpolate(
            masks,
            size=(self.diff_size, self.diff_size),
            mode="nearest",
        ).repeat(1, 136, 1, 1)
        distances = F.interpolate(
            distances,
            size=(self.diff_size, self.diff_size),
            mode="nearest",
        ).repeat(3, 1, 1, 1)
        distances *= mask_weights

        norms = torch.linalg.vector_norm(distances, dim=1, keepdim=True)
        norms = torch.where(norms == 0, torch.full_like(norms, 1e10), norms)
        distances = distances / norms

        return image_tensor, masks, distances


class _AMTGANPairReader(AFRTrainingDataset):
    """Independent random (non-makeup, makeup) pairs, drawn as the released
    ``DataReader.pick()`` does; subclasses supply the group sizes and :meth:`_record`.
    """

    preprocessor: Optional[
        Callable[
            [Image.Image, np.ndarray, np.ndarray],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ] = None
    seed: Optional[int] = None
    makeup_count: int = 0
    non_makeup_count: int = 0

    def _record(self, split: str, index: int) -> AMTGANRawRecord:
        """Read one record of ``split`` ("makeup"/"non-makeup") by position."""
        raise NotImplementedError

    def read_record(self, split: str, index: int) -> AMTGANRawRecord:
        """Read a specific record; primarily useful for validation/debugging."""
        if split not in (_SPLIT_MAKEUP, _SPLIT_NON_MAKEUP):
            raise ValueError(f"Unknown split: {split!r}")
        count = (
            self.makeup_count if split == _SPLIT_MAKEUP else self.non_makeup_count
        )
        if index < 0 or index >= count:
            raise IndexError(f"{split} index {index} is outside [0, {count})")
        return self._record(split, index)

    def _ensure_rng(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            worker = get_worker_info()
            if self.seed is None:
                rng_seed = worker.seed if worker is not None else torch.initial_seed()
            elif worker is None:
                rng_seed = self.seed
            else:
                rng_seed = self.seed + worker.id
            self._rng = np.random.default_rng(rng_seed)
            self._rng_pid = pid
        return self._rng

    def _prepare(
        self, record: AMTGANRawRecord
    ) -> AMTGANRawRecord | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preprocessor is None:
            return record
        return self.preprocessor(record.image, record.mask, record.landmarks)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        AMTGANRawRecord | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        AMTGANRawRecord | Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        del index  # Random pair per call, as in the released DataReader.pick().
        rng = self._ensure_rng()
        source = self._record(
            _SPLIT_NON_MAKEUP, int(rng.integers(0, self.non_makeup_count))
        )
        reference = self._record(
            _SPLIT_MAKEUP, int(rng.integers(0, self.makeup_count))
        )
        return self._prepare(source), self._prepare(reference)

    def __len__(self) -> int:
        return max(self.makeup_count, self.non_makeup_count)


class AMTGANLMDBDataset(_AMTGANPairReader):
    """Random-pair MT reader over the released LMDB, opened read-only and lazily per process.

    Example:
        Each batch holds ``[1, 3, 256, 256]`` images, ``[1, 3, 1, 256, 256]`` masks and
        ``[1, 3, 136, 64, 64]`` distances for the source and the reference::

            dataset = AMTGANLMDBDataset(
                "MT-dataset.lmdb", preprocessor=AMTGANTrainingPreprocessor()
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    """

    def __init__(
        self,
        lmdb_path: os.PathLike[str] | str,
        preprocessor: Optional[
            Callable[
                [Image.Image, np.ndarray, np.ndarray],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ] = None,
        *,
        seed: Optional[int] = None,
        readahead: bool = False,
        max_readers: int = 256,
    ) -> None:
        self.lmdb_path = Path(lmdb_path).expanduser().resolve()
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.lmdb_path}")
        if max_readers <= 0:
            raise ValueError("max_readers must be positive")
        self.source = f"lmdb:{self.lmdb_path}"

        self.preprocessor = preprocessor
        self.seed = seed
        self.readahead = bool(readahead)
        self.max_readers = int(max_readers)

        self._env: Optional[Any] = None
        self._env_pid: Optional[int] = None
        self._env_key: Optional[Tuple[int, str, bool, int]] = None
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: Optional[int] = None

        metadata = self._read_metadata()
        self.metadata: Dict[str, Any] = metadata
        self.makeup_count = self._metadata_count(metadata, "makeup_count")
        self.non_makeup_count = self._metadata_count(
            metadata, "non_makeup_count"
        )
        total_count = self._metadata_count(metadata, "count")
        if total_count != self.makeup_count + self.non_makeup_count:
            raise ValueError(
                "LMDB metadata count mismatch: "
                f"count={total_count}, makeup={self.makeup_count}, "
                f"non-makeup={self.non_makeup_count}"
            )
        if self.makeup_count == 0 or self.non_makeup_count == 0:
            raise ValueError("LMDB must contain both makeup and non-makeup items")

    @classmethod
    def from_assets(cls, **kwargs: Any) -> "AMTGANLMDBDataset":
        """Open the released MT-dataset LMDB from the dataset registry."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("mt"), **kwargs)

    @classmethod
    def from_corpus(cls, corpus: Any, **kwargs: Any) -> "AMTGANCorpusDataset":
        """Read the MT corpus, honouring the run's filters."""
        return AMTGANCorpusDataset(corpus, **kwargs)

    @classmethod
    def from_lmdb(
        cls, lmdb_path: "str | os.PathLike[str]", **kwargs: Any
    ) -> "AMTGANLMDBDataset":
        """Open an MT-dataset LMDB written by :mod:`afrlib.afr.amt_gan.mt_to_lmdb`."""
        return cls(lmdb_path, **kwargs)

    @classmethod
    def from_dir(
        cls,
        root: "str | os.PathLike[str]",
        *,
        lmdb_path: "str | os.PathLike[str] | None" = None,
        **kwargs: Any,
    ) -> "AMTGANLMDBDataset":
        """Read a raw MT-dataset tree, converted once to an LMDB at ``lmdb_path`` (by default
        ``<root>.lmdb``).
        """
        from .mt_to_lmdb import convert_mt_dataset_to_lmdb

        root = Path(root).expanduser().resolve()
        destination = (
            Path(lmdb_path).expanduser().resolve()
            if lmdb_path is not None
            else root.with_suffix(root.suffix + ".lmdb")
        )
        if not destination.exists():
            convert_mt_dataset_to_lmdb(root, destination)
        return cls(destination, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "lmdb_path": str(self.lmdb_path),
                "makeup_count": self.makeup_count,
                "non_makeup_count": self.non_makeup_count,
                "seed": self.seed,
            }
        )
        return config

    @staticmethod
    def _metadata_count(metadata: Dict[str, Any], key: str) -> int:
        try:
            value = int(metadata[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing LMDB metadata field: {key}") from exc
        if value < 0:
            raise ValueError(f"LMDB metadata field {key} cannot be negative")
        return value

    def _open_environment(
        self,
    ) -> Tuple[Any, Tuple[int, str, bool, int]]:
        return _acquire_environment(
            self.lmdb_path, self.readahead, self.max_readers
        )

    def _read_metadata(self) -> Dict[str, Any]:
        env, env_key = self._open_environment()
        try:
            with env.begin(write=False) as transaction:
                version_bytes = transaction.get(b"meta/version")
                metadata_bytes = transaction.get(b"meta/json")
                complete = transaction.get(b"meta/complete")
            if complete != b"1":
                raise ValueError("LMDB is incomplete or missing meta/complete")
            if version_bytes is None:
                raise ValueError("LMDB is missing meta/version")
            version = version_bytes.decode("utf-8")
            if version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported AMT-GAN LMDB version {version!r}; "
                    f"expected {SCHEMA_VERSION!r}"
                )
            if metadata_bytes is None:
                raise ValueError("LMDB is missing meta/json")
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("LMDB meta/json must contain a JSON object")
            if metadata.get("version") != version:
                raise ValueError("LMDB version metadata is inconsistent")
            return metadata
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed LMDB metadata: {exc}") from exc
        finally:
            _release_environment(env, env_key)

    def _ensure_environment(self) -> Any:
        pid = os.getpid()
        if self._env is None or self._env_pid != pid:
            if self._env is not None:
                _release_environment(self._env, self._env_key)
            self._env, self._env_key = self._open_environment()
            self._env_pid = pid
        return self._env

    @staticmethod
    def _key(split: str, index: int, field: str) -> bytes:
        return f"{split}/{index:08d}/{field}".encode("ascii")

    def _record(self, split: str, index: int) -> AMTGANRawRecord:
        env = self._ensure_environment()
        with env.begin(write=False) as transaction:
            return self._read_record(transaction, split, index)

    def _read_record(
        self, transaction: Any, split: str, index: int
    ) -> AMTGANRawRecord:
        values = {
            field: transaction.get(self._key(split, index, field))
            for field in _RECORD_FIELDS
        }
        missing = [field for field, value in values.items() if value is None]
        if missing:
            raise KeyError(
                f"LMDB record {split}[{index}] is missing: {', '.join(missing)}"
            )

        try:
            image = Image.open(io.BytesIO(values["image"])).convert("RGB")
            image.load()
            mask_image = Image.open(io.BytesIO(values["mask"]))
            mask_image.load()
            mask = np.asarray(mask_image).copy()
            if mask.ndim != 2:
                raise ValueError(f"semantic mask has shape {mask.shape}")

            landmarks = np.frombuffer(values["landmarks"], dtype="<f4")
            if landmarks.size != 136:
                raise ValueError(
                    f"landmark payload has {landmarks.size} values, expected 136"
                )
            landmarks = landmarks.reshape(68, 2).copy()
            name = values["name"].decode("utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Malformed LMDB record {split}[{index}]: {exc}") from exc

        if not np.isfinite(landmarks).all():
            raise ValueError(
                f"Malformed LMDB record {split}[{index}]: non-finite landmarks"
            )
        return AMTGANRawRecord(name, image, mask, landmarks)

    def close(self) -> None:
        """Close this process's lazy LMDB environment, if open."""

        if self._env is not None:
            _release_environment(self._env, self._env_key)
        self._env = None
        self._env_pid = None
        self._env_key = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_env"] = None
        state["_env_pid"] = None
        state["_env_key"] = None
        state["_rng"] = None
        state["_rng_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class AMTGANCorpusDataset(_AMTGANPairReader):
    """The same random-pair reader over a filtered :mod:`afrlib.data` corpus, grouped by each
    record's ``domain`` tag.
    """

    def __init__(
        self,
        corpus: Any,
        preprocessor: Optional[
            Callable[
                [Image.Image, np.ndarray, np.ndarray],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.corpus = corpus
        self.preprocessor = preprocessor
        self.seed = seed
        self.source = f"corpus:{corpus.name}"
        self.asset_key = None
        self._rng: Optional[np.random.Generator] = None
        self._rng_pid: Optional[int] = None

        self._keys: Dict[str, list] = {_SPLIT_MAKEUP: [], _SPLIT_NON_MAKEUP: []}
        self._names: Dict[str, list] = {_SPLIT_MAKEUP: [], _SPLIT_NON_MAKEUP: []}
        for meta in corpus.metas():
            if meta.domain in self._keys:
                self._keys[meta.domain].append(meta.key)
                self._names[meta.domain].append(meta.get("name") or meta.key)
        self.makeup_count = len(self._keys[_SPLIT_MAKEUP])
        self.non_makeup_count = len(self._keys[_SPLIT_NON_MAKEUP])
        if self.makeup_count == 0 or self.non_makeup_count == 0:
            raise ValueError(
                f"Corpus {corpus.name!r} must hold both makeup and non-makeup "
                f"faces after filtering, got {self.makeup_count} and "
                f"{self.non_makeup_count}"
            )

    @classmethod
    def from_assets(cls, **kwargs: Any) -> "AMTGANCorpusDataset":
        """The registered MT corpus, via the dataset registry."""
        from afrlib.data import get_corpus

        return cls(get_corpus("mt"), **kwargs)

    @classmethod
    def from_dir(
        cls,
        root: "str | os.PathLike[str]",
        **kwargs: Any,
    ) -> "AMTGANLMDBDataset":
        """Directory trees convert to LMDB; the corpus reader is not used."""
        return AMTGANLMDBDataset.from_dir(root, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        return {
            "dataset": type(self).__name__,
            "source": self.source,
            "makeup_count": self.makeup_count,
            "non_makeup_count": self.non_makeup_count,
            "seed": self.seed,
            **self.corpus.get_config(),
        }

    def _record(self, split: str, index: int) -> AMTGANRawRecord:
        key = self._keys[split][index]
        image = Image.open(io.BytesIO(self.corpus.read(key, "image"))).convert("RGB")
        image.load()
        mask_image = Image.open(io.BytesIO(self.corpus.read(key, "mask")))
        mask_image.load()
        mask = np.asarray(mask_image).copy()
        if mask.ndim != 2:
            raise ValueError(f"{split}[{index}] semantic mask has shape {mask.shape}")
        landmarks = np.frombuffer(
            self.corpus.read(key, "landmarks"), dtype="<f4"
        )
        if landmarks.size != 136:
            raise ValueError(
                f"{split}[{index}] has {landmarks.size} landmark values, expected 136"
            )
        landmarks = landmarks.reshape(68, 2).copy()
        if not np.isfinite(landmarks).all():
            raise ValueError(f"{split}[{index}] has non-finite landmarks")
        return AMTGANRawRecord(self._names[split][index], image, mask, landmarks)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_rng"] = None
        state["_rng_pid"] = None
        return state
