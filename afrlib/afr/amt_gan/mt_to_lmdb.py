"""Convert the released AMT-GAN MT-dataset to a validated LMDB (``python -m afrlib.afr.amt_gan.mt_to_lmdb``)."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb
import numpy as np
from PIL import Image


SCHEMA_VERSION = "amt-gan-mt-lmdb-v1"
JPEG_QUALITY = 95
_MEBIBYTE = 1024 * 1024
_SPLITS = (
    ("non-makeup", "non-makeup.txt"),
    ("makeup", "makeup.txt"),
)


@dataclass(frozen=True)
class SourceRecord:
    split: str
    index: int
    name: str
    image_path: Path
    mask_path: Path
    landmark_path: Path


@dataclass(frozen=True)
class EncodedRecord:
    image: bytes
    mask: bytes
    landmarks: bytes
    landmark_array: np.ndarray


def _normalize_relative_name(raw_name: str, list_path: Path, line_number: int) -> str:
    name = raw_name.strip().replace("\\", "/")
    if not name:
        raise ValueError(f"{list_path}:{line_number}: empty dataset name")

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{list_path}:{line_number}: name must stay inside dataset: {name!r}"
        )
    normalized = path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"{list_path}:{line_number}: invalid dataset name {name!r}")
    if "\x00" in normalized:
        raise ValueError(f"{list_path}:{line_number}: name contains a NUL byte")
    return normalized


def _read_name_list(root: Path, filename: str) -> List[str]:
    list_path = root / filename
    if not list_path.is_file():
        raise FileNotFoundError(f"Required split list not found: {list_path}")

    names: List[str] = []
    seen = set()
    try:
        with list_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                name = _normalize_relative_name(line, list_path, line_number)
                if name in seen:
                    raise ValueError(
                        f"{list_path}:{line_number}: duplicate name {name!r}"
                    )
                seen.add(name)
                names.append(name)
    except UnicodeDecodeError as exc:
        raise ValueError(f"Split list is not valid UTF-8: {list_path}") from exc

    if not names:
        raise ValueError(f"Split list contains no names: {list_path}")
    return names


def _path_for_name(base: Path, name: str) -> Path:
    return base.joinpath(*PurePosixPath(name).parts)


def collect_records(root: os.PathLike[str] | str) -> List[SourceRecord]:
    """Read both split lists and validate that every triple exists."""

    dataset_root = Path(root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"MT-dataset root is not a directory: {dataset_root}")

    directories = {
        "image": dataset_root / "images",
        "mask": dataset_root / "segs",
        "landmark": dataset_root / "landmarks",
    }
    for kind, directory in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Required {kind} directory not found: {directory}")

    records: List[SourceRecord] = []
    names_by_split: Dict[str, set[str]] = {}
    for split, filename in _SPLITS:
        names = _read_name_list(dataset_root, filename)
        names_by_split[split] = set(names)
        for index, name in enumerate(names):
            paths = {
                "image": _path_for_name(directories["image"], name),
                "mask": _path_for_name(directories["mask"], name),
                "landmark": _path_for_name(directories["landmark"], name),
            }
            missing = [
                f"{kind}={path}"
                for kind, path in paths.items()
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{split}[{index}] {name!r} has missing triple files: "
                    + ", ".join(missing)
                )
            records.append(
                SourceRecord(
                    split=split,
                    index=index,
                    name=name,
                    image_path=paths["image"],
                    mask_path=paths["mask"],
                    landmark_path=paths["landmark"],
                )
            )

    overlap = names_by_split["makeup"] & names_by_split["non-makeup"]
    if overlap:
        examples = ", ".join(repr(name) for name in sorted(overlap)[:5])
        raise ValueError(
            "Names must not occur in both makeup and non-makeup lists; "
            f"examples: {examples}"
        )
    return records


def estimate_map_size(
    records: Sequence[SourceRecord],
    *,
    safety_factor: float = 2.0,
    minimum_size: int = 64 * _MEBIBYTE,
) -> int:
    """Estimate and MiB-align the LMDB map from source payload sizes."""

    if safety_factor < 1.0:
        raise ValueError("map safety_factor must be at least 1.0")
    if minimum_size <= 0:
        raise ValueError("minimum map size must be positive")

    source_bytes = 0
    for record in records:
        for path in (record.image_path, record.mask_path, record.landmark_path):
            try:
                source_bytes += path.stat().st_size
            except OSError as exc:
                raise OSError(f"Cannot stat source file {path}: {exc}") from exc

    # Per-record allowance covers keys, LMDB pages, and JPEG growth when an
    # unusually compressed source is re-encoded at quality 95.
    estimated = int((source_bytes + len(records) * 16 * 1024) * safety_factor)
    estimated = max(estimated, minimum_size)
    return int(math.ceil(estimated / _MEBIBYTE) * _MEBIBYTE)


def _encode_record(record: SourceRecord) -> EncodedRecord:
    try:
        with Image.open(record.image_path) as source_image:
            source_image.load()
            image = source_image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode image {record.image_path}: {exc}") from exc
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"image has invalid size {image.size}: {record.image_path}")

    image_buffer = io.BytesIO()
    try:
        image.save(image_buffer, format="JPEG", quality=JPEG_QUALITY)
    except OSError as exc:
        raise ValueError(f"cannot encode image {record.image_path}: {exc}") from exc

    try:
        with Image.open(record.mask_path) as source_mask:
            source_mask.load()
            mask_array = np.asarray(source_mask)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode mask {record.mask_path}: {exc}") from exc
    if mask_array.ndim != 2:
        raise ValueError(
            f"semantic mask must be 2-D, got {mask_array.shape}: {record.mask_path}"
        )
    # Mask and image sizes are not compared: MT pairs 361x361 images with
    # 321x321 masks, and the released loader resizes each independently.
    if not np.issubdtype(mask_array.dtype, np.integer):
        raise ValueError(
            f"semantic mask must contain integer labels, got {mask_array.dtype}: "
            f"{record.mask_path}"
        )
    if mask_array.size:
        minimum = int(mask_array.min())
        maximum = int(mask_array.max())
        if minimum < 0 or maximum > 255:
            raise ValueError(
                "semantic labels must fit losslessly in uint8, got range "
                f"[{minimum}, {maximum}]: {record.mask_path}"
            )
    mask_array = np.asarray(mask_array, dtype=np.uint8)
    mask_buffer = io.BytesIO()
    Image.fromarray(mask_array, mode="L").save(mask_buffer, format="PNG")

    try:
        with record.landmark_path.open("rb") as handle:
            legacy_dtype_warning = getattr(
                getattr(np, "exceptions", np),
                "VisibleDeprecationWarning",
                Warning,
            )
            with warnings.catch_warnings():
                # MT landmark pickles encode NumPy dtype alignment as integer 0;
                # NumPy 2.4 warns but still decodes them correctly.
                warnings.simplefilter("ignore", legacy_dtype_warning)
                landmark_object = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise ValueError(
            f"cannot decode landmark pickle {record.landmark_path}: {exc}"
        ) from exc
    try:
        landmarks = np.asarray(landmark_object, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"landmarks are not a numeric array: {record.landmark_path}"
        ) from exc
    if landmarks.shape != (68, 2):
        raise ValueError(
            f"landmarks must have shape (68, 2), got {landmarks.shape}: "
            f"{record.landmark_path}"
        )
    if not np.isfinite(landmarks).all():
        raise ValueError(f"landmarks contain non-finite values: {record.landmark_path}")
    landmarks = np.ascontiguousarray(landmarks, dtype="<f4")

    return EncodedRecord(
        image=image_buffer.getvalue(),
        mask=mask_buffer.getvalue(),
        landmarks=landmarks.tobytes(order="C"),
        landmark_array=landmarks,
    )


def _record_key(record: SourceRecord, field: str) -> bytes:
    return f"{record.split}/{record.index:08d}/{field}".encode("ascii")


def _write_sanity_record(
    sanity_dir: Path, record: SourceRecord, encoded: EncodedRecord
) -> None:
    record_dir = sanity_dir / record.split / f"{record.index:08d}"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "image.jpg").write_bytes(encoded.image)
    (record_dir / "mask.png").write_bytes(encoded.mask)
    np.save(record_dir / "landmarks.npy", encoded.landmark_array)
    (record_dir / "name.txt").write_text(record.name + "\n", encoding="utf-8")


def _metadata(records: Iterable[SourceRecord]) -> Dict[str, object]:
    counts = {split: 0 for split, _ in _SPLITS}
    count = 0
    for record in records:
        counts[record.split] += 1
        count += 1
    return {
        "version": SCHEMA_VERSION,
        "count": count,
        "makeup_count": counts["makeup"],
        "non_makeup_count": counts["non-makeup"],
        "image_format": "jpeg",
        "image_mode": "RGB",
        "image_quality": JPEG_QUALITY,
        "mask_format": "png",
        "mask_mode": "L",
        "landmarks_dtype": "float32-le",
        "landmarks_shape": [68, 2],
        "name_format": "relative-posix-utf8",
    }


def _write_database(
    temporary_path: Path,
    records: Sequence[SourceRecord],
    metadata: Dict[str, object],
    map_size: int,
    batch_size: int,
    sanity_dir: Optional[Path],
    sanity_count: int,
) -> int:
    env = lmdb.open(
        str(temporary_path),
        subdir=True,
        map_size=map_size,
        meminit=False,
        map_async=False,
    )
    total_payload_bytes = 0
    transaction = env.begin(write=True)
    pending = 0
    try:
        for record in records:
            try:
                encoded = _encode_record(record)
            except Exception as exc:
                raise ValueError(
                    f"Malformed {record.split}[{record.index}] "
                    f"{record.name!r}: {exc}"
                ) from exc

            fields = {
                "image": encoded.image,
                "mask": encoded.mask,
                "landmarks": encoded.landmarks,
                "name": record.name.encode("utf-8"),
            }
            for field, value in fields.items():
                transaction.put(_record_key(record, field), value)
                total_payload_bytes += len(value)

            if sanity_dir is not None and record.index < sanity_count:
                _write_sanity_record(sanity_dir, record, encoded)

            pending += 1
            if pending >= batch_size:
                transaction.commit()
                transaction = env.begin(write=True)
                pending = 0

        transaction.commit()
        transaction = None

        metadata_json = json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with env.begin(write=True) as metadata_transaction:
            metadata_transaction.put(b"meta/version", SCHEMA_VERSION.encode("ascii"))
            metadata_transaction.put(
                b"meta/count", str(metadata["count"]).encode("ascii")
            )
            metadata_transaction.put(
                b"meta/makeup_count",
                str(metadata["makeup_count"]).encode("ascii"),
            )
            metadata_transaction.put(
                b"meta/non_makeup_count",
                str(metadata["non_makeup_count"]).encode("ascii"),
            )
            metadata_transaction.put(b"meta/json", metadata_json)
            metadata_transaction.put(b"meta/complete", b"1")
        env.sync()
    except lmdb.MapFullError as exc:
        if transaction is not None:
            transaction.abort()
        raise RuntimeError(
            f"LMDB map_size {map_size} bytes is too small; rerun with "
            "--map-size set to a larger value"
        ) from exc
    except Exception:
        if transaction is not None:
            transaction.abort()
        raise
    finally:
        env.close()
    return total_payload_bytes


def convert_mt_dataset_to_lmdb(
    input_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    batch_size: int = 512,
    map_size: Optional[int] = None,
    map_safety_factor: float = 2.0,
    sanity_dir: Optional[os.PathLike[str] | str] = None,
    sanity_count: int = 0,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Validate and convert an MT-dataset directory.

    Args:
        input_path: The released tree: ``makeup.txt``, ``non-makeup.txt``, and ``images/``,
            ``segs/`` and ``landmarks/`` (68x2 pickles) entries for each listed name.
        output_path: Destination LMDB (schema ``amt-gan-mt-lmdb-v1``: ``meta/*`` metadata and
            ``<split>/<index>/<field>`` records), written only after a successful conversion.
        batch_size: Records per write transaction.
        map_size: LMDB map size; estimated from the records when ``None``.
        map_safety_factor: Margin on the estimated map size.
        sanity_dir: Folder for decoded sample records, if given.
        sanity_count: Sample records per split written to ``sanity_dir``.
        overwrite: Replace an existing destination.

    Returns:
        The metadata, plus ``map_size`` and ``payload_bytes``.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sanity_count < 0:
        raise ValueError("sanity_count cannot be negative")

    records = collect_records(input_path)
    metadata = _metadata(records)

    if map_size is None:
        map_size = estimate_map_size(
            records, safety_factor=map_safety_factor
        )
    elif map_size <= 0:
        raise ValueError("map_size must be positive")

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination} (use --overwrite to replace it)"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    sanity_path = (
        Path(sanity_dir).expanduser().resolve() if sanity_dir is not None else None
    )
    if sanity_path is not None:
        sanity_path.mkdir(parents=True, exist_ok=True)

    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=str(destination.parent),
        )
    )
    try:
        payload_bytes = _write_database(
            temporary_path=temporary_path,
            records=records,
            metadata=metadata,
            map_size=int(map_size),
            batch_size=batch_size,
            sanity_dir=sanity_path,
            sanity_count=sanity_count,
        )

        if destination.exists():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(temporary_path, destination)
    except Exception:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    result = dict(metadata)
    result["map_size"] = int(map_size)
    result["payload_bytes"] = payload_bytes
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _byte_size(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    for suffix in sorted(multipliers, key=len, reverse=True):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            try:
                size = float(number) * multipliers[suffix]
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid byte size: {value!r}"
                ) from exc
            if size <= 0:
                raise argparse.ArgumentTypeError("map size must be positive")
            return int(size)
    try:
        size = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use bytes or a suffix such as 512MiB or 8GiB"
        ) from exc
    if size <= 0:
        raise argparse.ArgumentTypeError("map size must be positive")
    return size


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert AMT-GAN MT-dataset triples to LMDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="MT-dataset root containing the two lists and triple directories",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="destination LMDB directory",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=512,
        help="records per write transaction",
    )
    parser.add_argument(
        "--map-size",
        type=_byte_size,
        default=None,
        help="LMDB map size; auto-estimated when omitted (e.g. 8GiB)",
    )
    parser.add_argument(
        "--map-safety-factor",
        type=float,
        default=2.0,
        help="source-size multiplier used by automatic map sizing",
    )
    parser.add_argument(
        "--sanity-dir",
        type=Path,
        default=None,
        help="optional directory for decoded image/mask/landmark samples",
    )
    parser.add_argument(
        "--sanity-count",
        type=int,
        default=0,
        help="number of sanity records exported from each split",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing destination after conversion succeeds",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    result = convert_mt_dataset_to_lmdb(
        args.input,
        args.output,
        batch_size=args.batch_size,
        map_size=args.map_size,
        map_safety_factor=args.map_safety_factor,
        sanity_dir=args.sanity_dir,
        sanity_count=args.sanity_count,
        overwrite=args.overwrite,
    )
    print(f"Created {args.output}")
    print(
        "Records: "
        f"{result['count']} total "
        f"({result['non_makeup_count']} non-makeup, "
        f"{result['makeup_count']} makeup)"
    )
    print(
        f"Map size: {result['map_size'] / (1024**3):.2f} GiB; "
        f"encoded payload: {result['payload_bytes'] / (1024**2):.2f} MiB"
    )
    if args.sanity_dir is not None and args.sanity_count:
        print(f"Sanity samples: {args.sanity_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
