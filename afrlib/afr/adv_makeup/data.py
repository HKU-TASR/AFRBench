"""Aligned LADN eye-region training pairs for Adv-Makeup.

Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (commit 359af7e; academic research only, see LICENSE)
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import pickle
from typing import TYPE_CHECKING, Any, Callable
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

from afrlib.afr.datasets import AFRTrainingDataset
from afrlib.utils.assets import get_asset

from .preprocess import build_orbital_geometry, eye_crop_to_internal

if TYPE_CHECKING:
    from afrlib.data import TrainingCorpus


#: Released 600px alignment manifest that selects Adv-Makeup's LADN subset.
ALIGNED_LANDMARK_ASSET_KEY = "adv_makeup/landmark_aligned_600.pk"
FULL_DATASET_COUNTS = {"before": 332, "makeup": 198}
_ALIGNED_SIZE = 600


def _load_pickle(path: Path) -> object:
    # The released pickles use an old NumPy dtype descriptor that triggers a
    # NumPy 2.x compatibility warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with path.open("rb") as handle:
            return pickle.load(handle)


def _facepp_points(record: object, relative: str) -> np.ndarray:
    if not isinstance(record, dict):
        raise ValueError(f"{relative} Face++ record must be a mapping")
    faces = record.get("faces")
    if not isinstance(faces, list) or len(faces) != 1:
        raise ValueError(f"{relative} must contain exactly one detected face")
    landmark = faces[0].get("landmark")
    if not isinstance(landmark, dict) or len(landmark) != 106:
        raise ValueError(f"{relative} must contain 106 Face++ landmarks")
    try:
        points = np.asarray(
            [[point["x"], point["y"]] for point in landmark.values()],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{relative} contains malformed Face++ landmarks") from error
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError(f"{relative} contains invalid Face++ coordinates")
    return points


def _validate_points(points: np.ndarray, relative: str) -> np.ndarray:
    if points.shape != (106, 2) or not np.isfinite(points).all():
        raise ValueError(f"{relative} landmarks must be finite 106x2")
    return points


def _fit_alignment(
    source: np.ndarray,
    destination: np.ndarray,
    relative: str,
) -> tuple[np.ndarray, float]:
    """Fit the raw-to-600px affine map from paired landmarks; return it and its max residual.

    Raises:
        ValueError: If the residual exceeds 1e-3 px.
    """

    design = np.column_stack((source, np.ones(source.shape[0])))
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        destination,
        rcond=None,
    )
    if rank != 3:
        raise ValueError(f"{relative} landmarks cannot determine an affine map")
    projected = design @ coefficients
    maximum_error = float(np.abs(projected - destination).max())
    if not np.isfinite(maximum_error) or maximum_error > 1e-3:
        raise ValueError(
            f"{relative} raw/aligned landmarks disagree "
            f"(maximum affine residual {maximum_error:.6g}px)"
        )
    return coefficients.T, maximum_error


class AdvMakeupDataset(
    AFRTrainingDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """LADN items ``(source_eye, makeup_eye, source_face, landmarks)``: ``[3, 160, 420]`` eye and
    ``[3, 600, 600]`` face crops in ``[-1, 1]``, and ``[106, 2]`` landmarks in face pixels.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        before_dir: str = "before_aligned_600",
        makeup_dir: str = "after_aligned_600",
        landmark_file: str = "landmark_aligned_600.pk",
    ) -> None:
        """Load an already-aligned 600px tree (see :meth:`from_dir`)."""

        self._raw_mode = False
        self.root = Path(root).expanduser().resolve()
        self.before_dir = before_dir
        self.makeup_dir = makeup_dir
        self.landmark_file = landmark_file
        self.domain_roots: dict[str, Path] = {}
        self.affine_matrices: dict[str, np.ndarray] = {}
        self.maximum_alignment_error = 0.0
        self.corpus_config: dict[str, Any] | None = None
        if not self.root.is_dir():
            raise FileNotFoundError(f"Adv-Makeup dataset not found: {self.root}")

        before_root = self.root / before_dir
        makeup_root = self.root / makeup_dir
        landmark_path = self.root / landmark_file
        for path in (before_root, makeup_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if not landmark_path.is_file():
            raise FileNotFoundError(landmark_path)

        self.before_names = sorted(
            path.name for path in before_root.iterdir() if path.is_file()
        )
        self.makeup_names = sorted(
            path.name for path in makeup_root.iterdir() if path.is_file()
        )
        if not self.before_names or not self.makeup_names:
            raise ValueError("Adv-Makeup needs non-empty before and makeup domains")

        landmarks = _load_pickle(landmark_path)
        if not isinstance(landmarks, dict):
            raise ValueError("Landmark asset must contain a path-to-array mapping")
        self.landmarks = landmarks

        missing = [
            relative
            for relative in (
                *(
                    f"{before_dir}/{name}" for name in self.before_names
                ),
                *(
                    f"{makeup_dir}/{name}" for name in self.makeup_names
                ),
            )
            if relative not in self.landmarks
        ]
        if missing:
            raise ValueError(
                f"Landmark asset is missing {len(missing)} image entries; "
                f"first missing: {missing[0]}"
            )
        self.source = f"dir:{self.root}"

    @classmethod
    def from_dir(
        cls, root: str | os.PathLike[str], **kwargs: Any
    ) -> "AdvMakeupDataset":
        """Load an aligned 600 px tree with ``before_aligned_600/``, ``after_aligned_600/`` and
        ``landmark_aligned_600.pk`` (names can be overridden by keyword).
        """
        return cls(root, **kwargs)

    @classmethod
    def from_ladn(
        cls,
        before_root: str | os.PathLike[str],
        makeup_root: str | os.PathLike[str],
        raw_landmark_file: str | os.PathLike[str],
        aligned_landmark_file: str | os.PathLike[str],
    ) -> "AdvMakeupDataset":
        """Rebuild the released 600 px subset from a raw LADN tree, recovering each image's
        alignment from its raw and aligned landmarks.
        """
        raw_path = Path(raw_landmark_file).expanduser().resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        raw_records = _load_pickle(raw_path)
        if not isinstance(raw_records, dict):
            raise ValueError("Raw LADN landmark asset must contain a mapping")

        domain_roots = {
            "before": Path(before_root).expanduser().resolve(),
            "after": Path(makeup_root).expanduser().resolve(),
        }
        for path in domain_roots.values():
            if not path.is_dir():
                raise FileNotFoundError(path)

        def points(relative: str) -> np.ndarray:
            if relative not in raw_records:
                raise ValueError(f"Raw landmark asset is missing {relative}")
            return _facepp_points(raw_records[relative], relative)

        def image(domain: str, name: str) -> np.ndarray:
            path = domain_roots[domain] / name
            if not path.is_file():
                raise FileNotFoundError(path)
            return cls._load_rgb(path, require_aligned=False)

        dataset = cls._from_raw(aligned_landmark_file, points, image)
        dataset.domain_roots = domain_roots
        dataset.source = f"dir:{domain_roots['before'].parent}"
        return dataset

    @classmethod
    def from_corpus(cls, corpus: "TrainingCorpus") -> "AdvMakeupDataset":
        """Rebuild the released 600 px subset from the packed LADN corpus.

        Raises:
            ValueError: If the corpus, for example after filtering, lacks a face the manifest
                needs.
        """
        keys = {
            (meta.get("domain"), meta.get("name")): meta.key
            for meta in corpus.metas(need=("domain", "name"))
        }

        def points(relative: str) -> np.ndarray:
            domain, name = relative.split("/", 1)
            key = keys.get((domain, name))
            if key is None:
                raise ValueError(
                    f"Corpus {corpus.name!r} does not provide LADN {relative}"
                )
            return _validate_points(
                np.asarray(json.loads(corpus.read(key, "landmarks")), np.float64),
                relative,
            )

        def image(domain: str, name: str) -> np.ndarray:
            return cls._load_rgb_bytes(
                corpus.read(keys[(domain, name)]), f"{domain}/{name}"
            )

        dataset = cls._from_raw(
            get_asset(ALIGNED_LANDMARK_ASSET_KEY), points, image
        )
        dataset.source = f"corpus:{corpus.name}"
        dataset.corpus_config = corpus.get_config()
        return dataset

    @classmethod
    def _from_raw(
        cls,
        aligned_landmark_file: str | os.PathLike[str],
        points: Callable[[str], np.ndarray],
        image: Callable[[str, str], np.ndarray],
    ) -> "AdvMakeupDataset":
        """Recover the released 600px alignment for every manifest entry."""
        aligned_path = Path(aligned_landmark_file).expanduser().resolve()
        if not aligned_path.is_file():
            raise FileNotFoundError(aligned_path)
        aligned_records = _load_pickle(aligned_path)
        if not isinstance(aligned_records, dict):
            raise ValueError("Aligned landmark asset must contain a mapping")

        dataset = cls.__new__(cls)
        dataset._raw_mode = True
        dataset.root = None
        dataset.before_dir = "before"
        dataset.makeup_dir = "after"
        dataset.landmark_file = str(aligned_path)
        dataset.domain_roots = {}
        dataset.corpus_config = None
        dataset._read_raw = image

        prefixes = {"before": "before_aligned_600/", "after": "after_aligned_600/"}
        names_by_domain = {
            domain: sorted(
                relative[len(prefix) :]
                for relative in aligned_records
                if isinstance(relative, str)
                and relative.startswith(prefix)
                and "/" not in relative[len(prefix) :]
            )
            for domain, prefix in prefixes.items()
        }
        dataset.before_names = names_by_domain["before"]
        dataset.makeup_names = names_by_domain["after"]
        if not dataset.before_names or not dataset.makeup_names:
            raise ValueError("Aligned manifest has empty before or makeup domain")

        dataset.landmarks = {}
        dataset.affine_matrices = {}
        maximum_error = 0.0
        for domain, names in names_by_domain.items():
            for name in names:
                raw_relative = f"{domain}/{name}"
                destination = _validate_points(
                    np.asarray(
                        aligned_records[f"{prefixes[domain]}{name}"], np.float64
                    ),
                    f"{prefixes[domain]}{name}",
                )
                matrix, residual = _fit_alignment(
                    points(raw_relative), destination, raw_relative
                )
                dataset.affine_matrices[raw_relative] = matrix
                dataset.landmarks[raw_relative] = destination.astype(np.float32)
                maximum_error = max(maximum_error, residual)
        dataset.maximum_alignment_error = maximum_error
        observed = {
            "before": len(dataset.before_names),
            "makeup": len(dataset.makeup_names),
        }
        if observed != FULL_DATASET_COUNTS:
            raise ValueError(
                "Adv-Makeup manifest has unexpected domain counts: "
                f"{observed}, expected {FULL_DATASET_COUNTS}"
            )
        return dataset

    @classmethod
    def from_assets(cls) -> "AdvMakeupDataset":
        """The complete training dataset, via the LADN corpus."""
        from afrlib.data import get_corpus

        return cls.from_corpus(get_corpus("ladn"))

    def __len__(self) -> int:
        return len(self.before_names)

    @staticmethod
    def _load_rgb(path: Path, *, require_aligned: bool) -> np.ndarray:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
        if require_aligned and image.shape != (
            _ALIGNED_SIZE,
            _ALIGNED_SIZE,
            3,
        ):
            raise ValueError(
                f"Adv-Makeup training images must be 600x600 RGB, got {image.shape}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Adv-Makeup training image is not RGB: {path}")
        return image

    @staticmethod
    def _load_rgb_bytes(payload: bytes, relative: str) -> np.ndarray:
        with Image.open(io.BytesIO(payload)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Adv-Makeup training image is not RGB: {relative}")
        return array

    def _read_domain(
        self, directory: str, name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relative = f"{directory}/{name}"
        if self._raw_mode:
            raw = self._read_raw(directory, name)
            aligned = cv2.warpAffine(
                raw,
                self.affine_matrices[relative],
                (_ALIGNED_SIZE, _ALIGNED_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            image255 = torch.from_numpy(
                np.ascontiguousarray(aligned)
            ).permute(2, 0, 1).to(torch.float32)
        else:
            assert self.root is not None
            aligned = self._load_rgb(
                self.root / relative,
                require_aligned=True,
            )
            image255 = torch.from_numpy(aligned).permute(2, 0, 1).to(
                torch.float32
            )
        landmarks = torch.as_tensor(
            np.asarray(self.landmarks[relative]), dtype=torch.float32
        )
        if tuple(landmarks.shape) != (106, 2):
            raise ValueError(
                f"{relative} landmarks must be 106x2, got {tuple(landmarks.shape)}"
            )
        geometry = build_orbital_geometry(landmarks, (600, 600))
        eye = eye_crop_to_internal(image255, geometry)
        full = image255.div(127.5).sub(1.0)
        return eye, full, landmarks

    def get_config(self) -> dict[str, object]:
        return {
            "source": self.source,
            "reconstructed_from_raw": self._raw_mode,
            "corpus": self.corpus_config,
            "aligned_manifest": ALIGNED_LANDMARK_ASSET_KEY,
            "before_count": len(self.before_names),
            "makeup_count": len(self.makeup_names),
            "paper_reported_counts": {"before": 333, "makeup": 198},
            "selection": "released_600px_landmark_manifest",
            "aligned_size": _ALIGNED_SIZE,
            "maximum_landmark_alignment_residual_px": (
                self.maximum_alignment_error
            ),
            "pairing": "source_index_modulo_makeup_count",
        }

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        before_name = self.before_names[index]
        makeup_name = self.makeup_names[index % len(self.makeup_names)]
        before_eye, before_full, before_landmarks = self._read_domain(
            self.before_dir, before_name
        )
        makeup_eye, _, _ = self._read_domain(self.makeup_dir, makeup_name)
        return before_eye, makeup_eye, before_full, before_landmarks
