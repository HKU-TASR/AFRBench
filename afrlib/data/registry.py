"""Registry of the training corpora AFRLib can download, pack, and filter."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from afrlib.utils.assets import GDrive, Kaggle, Source, fetch, get_home_dir
from afrlib.utils.log import get_logger

from . import lmdb_io
from .corpus import TrainingCorpus
from .filters import Filter
from .packers import PACKERS

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawPart:
    """One downloadable piece of a corpus's raw release; ``within_root`` is where it lives in a
    custom ``root=`` tree laid out like the original.
    """

    name: str
    sources: Tuple[Source, ...]
    sha256: str = ""
    size: int = 0
    is_directory: bool = False
    allow_symlink: bool = False
    within_root: str = "."


@dataclass(frozen=True)
class Corpus:
    """How to obtain and pack one named corpus."""

    name: str
    description: str
    parts: Tuple[RawPart, ...]
    packer: str
    #: Human description of the layout a custom ``root=`` must have.
    layout: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(part.size for part in self.parts)


_MS1M = GDrive("1X202mvYe5tiXFhOx82z4rPiPogXD435i", member="imgs/")
_LADN_ARCHIVE = "1Y0AlgKSVZNsjUG4u0YUHzoxtsG0qgexZ"

DATASET_REGISTRY: Dict[str, Corpus] = {
    corpus.name: corpus
    for corpus in (
        Corpus(
            name="ms1m",
            description=(
                "MS-Celeb-1M, ArcFace-aligned 112x112, folder-per-identity. "
                "APF's probe/server filter and DPA's DPO snapshots train on it."
            ),
            parts=(
                RawPart(
                    name="root",
                    # sha256 of the downloaded ms1m_align_112.zip, not of the
                    # extracted imgs/ tree.
                    sha256="c3d6bbb07dd53fd504241d624dabe55513b394eb672417005fb8f3af680b5bea",
                    size=26_907_316_148,
                    sources=(_MS1M,),
                    is_directory=True,
                    allow_symlink=True,
                ),
            ),
            packer="classification",
            layout="<identity>/<image>.jpg",
        ),
        Corpus(
            name="casia_webface",
            description=(
                "CASIA-WebFace (maxpy-clean), folder-per-identity: 10,575 "
                "identities, 455,594 images. AdvCloak's training corpus."
            ),
            parts=(
                RawPart(
                    name="root",
                    # sha256 of the reassembled combined.zip; the original
                    # upload was a spanned zip that zipfile cannot read.
                    sha256="b13713de909715a3abe95322db1593ef2090f14f9052346d7b18915fa9b97375",
                    size=3_935_597_161,
                    sources=(
                        GDrive(
                            "1I45bTzEGJMhMpG-jchdAdga74WP58dVA",
                            member="CASIA-maxpy-clean/",
                        ),
                    ),
                    is_directory=True,
                    allow_symlink=True,
                ),
            ),
            packer="classification",
            layout="<identity>/<image>.jpg",
        ),
        Corpus(
            name="vggface2",
            description=(
                "VGGFace2 pre-aligned to 112x112 (8.6k identities, 3.1M "
                "images), mirrored on Kaggle from OpenSphere; AIDPro's "
                "stage-two corpus, a smoke surrogate for the released 224px "
                "original."
            ),
            parts=(
                RawPart(
                    name="root",
                    sha256="",  # not hashed (~19 GB)
                    size=20_401_094_656,
                    sources=(Kaggle("yakhyokhuja/vggface2-112x112"),),
                    is_directory=True,
                    allow_symlink=True,
                ),
            ),
            packer="classification",
            layout="<identity>/<image>.jpg",
        ),
        Corpus(
            name="mt",
            description=(
                "Makeup Transfer (MT) dataset: makeup / non-makeup faces with "
                "semantic masks and 68-point landmarks. AMT-GAN's corpus."
            ),
            parts=(
                RawPart(
                    name="lmdb",
                    sha256="68ccf2a3733bef08e8654d94f664b7fe1c229f35d869e4808577393b5c9ab1fa",
                    size=198_406_144,
                    sources=(
                        GDrive(
                            "1oGYHAy77d2FPyONlWZHpKXMWoGDGlsCq",
                            member="MT-dataset.lmdb/",
                        ),
                    ),
                    is_directory=True,
                    within_root="MT-dataset.lmdb",
                ),
            ),
            packer="mt",
            layout="MT-dataset.lmdb/ (schema amt-gan-mt-lmdb-v1)",
        ),
        Corpus(
            name="ladn",
            description=(
                "LADN makeup dataset: 334 non-makeup and 355 makeup faces with "
                "the authors' Face++ 106-point landmarks. Adv-Makeup's corpus."
            ),
            parts=(
                RawPart(
                    name="before",
                    # Members of one Drive archive with no recorded digest; the
                    # shared-archive cache downloads it once for all three.
                    size=12_710_900,
                    sources=(GDrive(_LADN_ARCHIVE, member="makeup/before/"),),
                    is_directory=True,
                    within_root="before",
                ),
                RawPart(
                    name="after",
                    size=13_482_245,
                    sources=(GDrive(_LADN_ARCHIVE, member="makeup/after/"),),
                    is_directory=True,
                    within_root="after",
                ),
                RawPart(
                    name="landmarks",
                    sha256="3d813b428a90b90b60207279017f84cc7565fb62ee5f3ef41c5627777294224e",
                    size=3_703_056,
                    sources=(GDrive(_LADN_ARCHIVE, member="makeup/landmark.pk"),),
                    within_root="landmark.pk",
                ),
            ),
            packer="ladn",
            layout="before/, after/, landmark.pk",
        ),
        Corpus(
            name="celebamask_hq",
            description=(
                "CelebAMask-HQ: 30k 1024px faces with 19-region semantic masks "
                "and attribute annotations. DiffAM's makeup-transfer corpus."
            ),
            parts=(
                RawPart(
                    name="root",
                    sha256="cadce6e3a9b895342195980d517ec65c99f7a42071db5f37756fc3858dce722c",
                    size=3_343_653_186,
                    sources=(GDrive("1badu11NqxGf6qM3PTTooQDJvQbejgbTv", member="CelebAMask-HQ/"),),
                    is_directory=True,
                ),
            ),
            packer="celebamask_hq",
            layout="CelebA-HQ-img/, CelebAMask-HQ-mask-anno/, CelebAMask-HQ-attribute-anno.txt",
        ),
    )
}


def datasets_dir() -> Path:
    """Where packed corpora and their raw downloads live (``<home>/datasets``)."""
    return Path(get_home_dir()) / "datasets"


def get_corpus(
    name: str,
    *,
    filters: Sequence[Filter] = (),
    root: "str | os.PathLike[str] | None" = None,
    repack: bool = False,
) -> TrainingCorpus:
    """Resolve a registered corpus, packing it on first use.

    Args:
        name: A key of :data:`DATASET_REGISTRY`.
        filters: Applied to every sample the returned handle exposes; the
            benchmark uses these to keep evaluation identities out of training.
        root: A local corpus to use instead of the registered release. Either a
            raw tree laid out like the original (see ``Corpus.layout``), packed
            and cached separately, or an already-packed corpus, used as is.
        repack: Pack again even if a packed corpus is already cached. A custom
            ``root`` is cached by its path, not its contents, so a tree edited
            in place needs this.

    Returns:
        A :class:`~afrlib.data.corpus.TrainingCorpus` over the packed corpus.

    Raises:
        KeyError: If ``name`` is not registered.
        FileNotFoundError: If a raw part needed for packing is missing.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown corpus {name!r}. Registered: {sorted(DATASET_REGISTRY)}"
        )
    descriptor = DATASET_REGISTRY[name]

    if root is not None:
        root = Path(root).expanduser().resolve()
        if lmdb_io.is_packed(root):
            return TrainingCorpus(name, root, filters, source=f"packed:{root}")
        lmdb_path = datasets_dir() / "custom" / f"{name}-{_digest(root)}.lmdb"
        source = f"root:{root}"
        parts = {
            part.name: root / part.within_root if part.within_root != "." else root
            for part in descriptor.parts
        }
    else:
        lmdb_path = datasets_dir() / f"{name}.lmdb"
        source = "registry"
        parts = None  # downloaded only if packing is needed

    if repack or not lmdb_io.is_packed(lmdb_path):
        if parts is None:
            parts = _fetch_parts(descriptor)
        _missing = [str(p) for p in parts.values() if not Path(p).exists()]
        if _missing:
            raise FileNotFoundError(
                f"Corpus {name!r} cannot be packed; missing {_missing}. "
                f"Expected layout: {descriptor.layout or '(see the registry)'}"
            )
        logger.info("Packing corpus %r into %s ...", name, lmdb_path)
        PACKERS[descriptor.packer](parts, lmdb_path, source=source, **descriptor.options)

    return TrainingCorpus(name, lmdb_path, filters, source=source)


def _fetch_parts(descriptor: Corpus) -> Dict[str, Path]:
    """Download every raw piece of ``descriptor`` into the dataset cache."""
    raw_root = datasets_dir() / "raw" / descriptor.name
    resolved: Dict[str, Path] = {}
    for part in descriptor.parts:
        destination = raw_root / part.name
        if not destination.exists():
            fetch(
                str(destination),
                part.sources,
                sha256=part.sha256,
                size=part.size,
                label=f"{descriptor.name}/{part.name}",
                allow_symlink=part.allow_symlink,
            )
        resolved[part.name] = destination
    return resolved


def _digest(path: Path) -> str:
    """Short stable id of a custom corpus root, for its cache filename."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
