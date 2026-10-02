"""Load protect runs: every protectee ``manifest.json`` under one ``runs:`` root."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

MANIFEST_FILENAME = "manifest.json"


@dataclasses.dataclass(frozen=True)
class ProtectedImage:
    """One protected original: its dataset source path and its ``.npy`` output."""

    source_path: str  # dataset-relative path of the photo it protects
    output: Path  # the .npy, resolved against its run directory
    dtype: str
    sha256: Optional[str] = None

    def load(self) -> torch.Tensor:
        """The protected original as ``[3, H, W]`` float32 in ``[0, 255]``."""
        import numpy as np

        return torch.from_numpy(np.load(self.output)).to(torch.float32)


@dataclasses.dataclass(frozen=True)
class ProtectedRun:
    """Every protected image under one run root, in dataset-path order, with ``digest`` hashing
    the sorted ``(source path, image sha256)`` pairs.
    """

    root: Path
    images: List[ProtectedImage]
    surrogates: List[Dict[str, Any]]
    target: Optional[str] = None
    digest: str = ""

    @classmethod
    def load(cls, root: "str | Path") -> "ProtectedRun":
        """Collect every manifest under ``root``; a source photo protected twice raises."""
        root = Path(root)
        images: List[ProtectedImage] = []
        surrogates: List[Dict[str, Any]] = []
        target = None
        pairs: List[Tuple[str, str]] = []
        for manifest_path in sorted(root.rglob(MANIFEST_FILENAME)):
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            fallback = hashlib.sha256(manifest_bytes).hexdigest()
            run_dir = manifest_path.parent
            if target is None:
                target = manifest.get("target")
            for source_path, entry in manifest.get("images", {}).items():
                image_hash = str(entry["sha256"]) if entry.get("sha256") else fallback
                pairs.append((source_path, image_hash))
                images.append(ProtectedImage(
                    source_path=source_path,
                    output=run_dir / entry["output"],
                    dtype=str(entry.get("dtype", "float32")),
                    sha256=image_hash,
                ))
            for spec in manifest.get("surrogates", []):
                if spec not in surrogates:
                    surrogates.append(spec)
        images.sort(key=lambda i: i.source_path)
        _reject_duplicates(root, images)
        digest = hashlib.sha256()
        for source_path, image_hash in sorted(pairs):
            digest.update(source_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(image_hash.encode("utf-8"))
            digest.update(b"\0")
        return cls(
            root=root,
            images=images,
            surrogates=surrogates,
            target=target,
            digest=digest.hexdigest(),
        )

    @property
    def short_digest(self) -> str:
        return self.digest[:12]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def source_paths(self) -> List[str]:
        """The dataset-relative paths this run protected."""
        return [image.source_path for image in self.images]

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """``(source path, protected original)`` pairs, loaded one at a time."""
        for image in self.images:
            yield image.source_path, image.load()

    def by_source(self) -> Dict[str, ProtectedImage]:
        return {image.source_path: image for image in self.images}


def _reject_duplicates(root: Path, images: List[ProtectedImage]) -> None:
    """Raise if a source photo has two protected versions, i.e. the root spans two runs, whose
    embeddings would overwrite each other.
    """
    seen: Dict[str, Path] = {}
    for image in images:
        first = seen.get(image.source_path)
        if first is not None:
            raise ValueError(
                f"{root} holds two protected versions of {image.source_path} "
                f"({first} and {image.output}); a runs: root must name one run, "
                "so point it at the per-target subdirectory."
            )
        seen[image.source_path] = image.output
