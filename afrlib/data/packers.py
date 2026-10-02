"""Packers: convert a dataset's raw release into the corpus LMDB schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from afrlib.utils.log import get_logger, tqdm

from .lmdb_io import CorpusWriter, estimate_map_size

logger = get_logger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _image_files(directory: Path) -> List[Path]:
    """Every image directly inside ``directory``, in a stable order."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _identity_folders(root: Path) -> Dict[str, List[Path]]:
    """Map identity -> its images from an ``<identity>/<image>`` or ``<group>/<identity>/<image>``
    tree, falling back to a flat ``<identity>_<n>.<ext>`` directory.
    """
    groups: Dict[str, List[Path]] = {}
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        images = _image_files(entry)
        if images:
            groups.setdefault(entry.name, []).extend(images)
            continue
        for nested in sorted(p for p in entry.iterdir() if p.is_dir()):
            nested_images = _image_files(nested)
            if nested_images:
                groups.setdefault(nested.name, []).extend(nested_images)
    if not groups:
        for path in _image_files(root):
            groups.setdefault(path.stem.split("_")[0], []).append(path)
    if not groups:
        raise ValueError(f"No images found under {root}")
    return groups


def _payload_size(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _write(
    dest: Path,
    records: Sequence[Tuple[str, Dict[str, bytes], Dict[str, Any]]],
    *,
    provenance: Mapping[str, Any],
    payload_bytes: int,
) -> None:
    """Write already-materialized records (small corpora only)."""
    with CorpusWriter(
        dest,
        map_size=estimate_map_size(payload_bytes, len(records)),
        provenance=provenance,
    ) as writer:
        for key, fields, meta in records:
            writer.add(key, fields, **meta)


def pack_classification(parts: Mapping[str, Path], dest: Path, **options: Any) -> None:
    """Pack a folder-per-identity face corpus, taking each folder name as the sample's
    ``identity`` and ``<identity>/<file>`` as its ``name``.
    """
    root = Path(parts["root"])
    groups = _identity_folders(root)
    total = sum(len(paths) for paths in groups.values())
    payload = _payload_size(path for paths in groups.values() for path in paths)
    logger.info(
        "Packing %d images of %d identities from %s", total, len(groups), root
    )
    provenance = {
        "packer": "pack_classification",
        "root": str(root),
        "identities": len(groups),
        **options,
    }
    with CorpusWriter(
        dest, map_size=estimate_map_size(payload, total), provenance=provenance
    ) as writer:
        index = 0
        with tqdm(total=total, desc=f"packing {dest.name}") as bar:
            for identity in sorted(groups):
                for path in groups[identity]:
                    writer.add(
                        f"{index:08d}",
                        {"image": path.read_bytes()},
                        identity=identity,
                        name=f"{identity}/{path.name}",
                    )
                    index += 1
                    bar.update(1)


def pack_ladn(parts: Mapping[str, Path], dest: Path, **options: Any) -> None:
    """Pack the LADN makeup corpus: faces tagged ``domain`` ``"before"`` or ``"after"``, with the
    106 Face++ points of single-face records as a ``landmarks`` field.
    """
    import pickle

    landmark_records = pickle.loads(Path(parts["landmarks"]).read_bytes())
    records: List[Tuple[str, Dict[str, bytes], Dict[str, Any]]] = []
    payload = 0
    index = 0
    for domain in ("before", "after"):
        directory = Path(parts[domain])
        for path in _image_files(directory):
            entry = landmark_records.get(f"{domain}/{path.name}")
            fields = {"image": path.read_bytes()}
            points = _faceplusplus_points(entry) if entry is not None else None
            if points is not None:
                fields["landmarks"] = json.dumps(points).encode("utf-8")
            payload += sum(len(value) for value in fields.values())
            records.append(
                (
                    f"{index:08d}",
                    fields,
                    {"domain": domain, "name": path.name},
                )
            )
            index += 1
    _write(
        dest,
        records,
        provenance={"packer": "pack_ladn", **options},
        payload_bytes=payload,
    )


def _faceplusplus_points(entry: Any) -> Optional[List[List[float]]]:
    """The 106 ``(x, y)`` points of a single-face Face++ record, in the record's own order, or
    ``None``; methods pair them index by index with their own manifests.
    """
    faces = entry.get("faces") if isinstance(entry, Mapping) else None
    if not isinstance(faces, list) or len(faces) != 1:
        return None
    landmark = faces[0].get("landmark")
    if not isinstance(landmark, Mapping) or len(landmark) != 106:
        return None
    return [[float(point["x"]), float(point["y"])] for point in landmark.values()]


def pack_mt(parts: Mapping[str, Path], dest: Path, **options: Any) -> None:
    """Repack the MT-dataset LMDB written by :mod:`afrlib.afr.amt_gan.mt_to_lmdb`, copying its
    bytes unchanged and turning the makeup split into ``domain``.
    """
    from afrlib.afr.amt_gan.data import SCHEMA_VERSION

    from .lmdb_io import lmdb_env

    source = Path(parts["lmdb"])
    env = lmdb_env(source)
    records: List[Tuple[str, Dict[str, bytes], Dict[str, Any]]] = []
    payload = 0
    with env.begin() as txn:
        raw_meta = txn.get(b"meta/json")
        version = txn.get(b"meta/version")
        if raw_meta is None or version != SCHEMA_VERSION.encode("ascii"):
            raise ValueError(
                f"{source} is not an MT-dataset LMDB (expected schema "
                f"{SCHEMA_VERSION!r}, found {version!r})"
            )
        meta = json.loads(raw_meta)
        counts = {
            "non-makeup": int(meta["non_makeup_count"]),
            "makeup": int(meta["makeup_count"]),
        }
        index = 0
        for domain, count in counts.items():
            for position in range(count):
                prefix = f"{domain}/{position:08d}"
                fields = {}
                for field in ("image", "mask", "landmarks"):
                    value = txn.get(f"{prefix}/{field}".encode("ascii"))
                    if value is not None:
                        fields[field] = value
                if "image" not in fields:
                    raise ValueError(f"{source} is missing the image at {prefix}")
                name = txn.get(f"{prefix}/name".encode("ascii"))
                payload += sum(len(v) for v in fields.values())
                records.append(
                    (
                        f"{index:08d}",
                        fields,
                        {
                            "domain": domain,
                            "name": name.decode("utf-8") if name else prefix,
                        },
                    )
                )
                index += 1
    if not records:
        raise ValueError(f"{source} holds no MT records")
    _write(
        dest,
        records,
        provenance={"packer": "pack_mt", "source_lmdb": str(source), **options},
        payload_bytes=payload,
    )


def pack_celebamask_hq(parts: Mapping[str, Path], dest: Path, **options: Any) -> None:
    """Pack CelebAMask-HQ in the attribute file's row order: each region mask as its own
    ``mask_<region>`` field and the attribute row as ``attributes``.
    """
    root = Path(parts["root"])
    images_dir = root / "CelebA-HQ-img"
    masks_dir = root / "CelebAMask-HQ-mask-anno"
    attributes = root / "CelebAMask-HQ-attribute-anno.txt"
    for required in (images_dir, attributes):
        if not required.exists():
            raise FileNotFoundError(f"CelebAMask-HQ is missing {required}")

    lines = attributes.read_text(encoding="utf-8").splitlines()
    header = lines[1].split() if len(lines) > 1 else []
    rows = [line.split() for line in lines[2:] if line.split()]

    records: List[Tuple[str, Dict[str, bytes], Dict[str, Any]]] = []
    payload = 0
    for index, row in enumerate(tqdm(rows, desc=f"packing {dest.name}")):
        filename = row[0]
        image_path = images_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"CelebAMask-HQ is missing image {image_path}")
        fields = {
            "image": image_path.read_bytes(),
            "attributes": json.dumps(dict(zip(header, row[1:]))).encode("utf-8"),
        }
        stem = Path(filename).stem
        region_dir = masks_dir / str(int(stem) // 2000)
        for mask_path in sorted(region_dir.glob(f"{stem.zfill(5)}_*.png")):
            region = mask_path.stem[len(stem.zfill(5)) + 1 :]
            fields[f"mask_{region}"] = mask_path.read_bytes()
        payload += sum(len(value) for value in fields.values())
        records.append((f"{index:08d}", fields, {"name": filename}))
    _write(
        dest,
        records,
        provenance={"packer": "pack_celebamask_hq", "root": str(root), **options},
        payload_bytes=payload,
    )


#: Packers by name, so a corpus descriptor can name one declaratively.
PACKERS: Dict[str, Callable[..., None]] = {
    "classification": pack_classification,
    "ladn": pack_ladn,
    "mt": pack_mt,
    "celebamask_hq": pack_celebamask_hq,
}
