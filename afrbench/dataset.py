"""An AFRBench dataset (schema v4): raw images plus geometry derived at load time."""
from __future__ import annotations

import dataclasses
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from afrlib.utils.face_crop import extract_frame, plan_from_stored

SCHEMA = "afrbench-dataset-v4"
_SUPERSEDED = {
    "afrbench-dataset-v3",
    "afrbench-dataset-v2",
    "afrbench-dataset-v1",
}

ROLES = ("album", "query", "gallery")
PROTECTION_ROLES = ("query", "gallery")
POOLS = ROLES


@dataclasses.dataclass(frozen=True)
class ImageRecord:
    """One annotated image: identity, role, and face geometry.

    ``labels`` holds optional attributes of the photo (for AFRBench-DF its camera, emotion and
    light) that ``python -m afrbench.groups`` reports results by; ``source`` is provenance only.
    """

    path: str
    identity: str
    split: str  # "protectee" | "target"
    sha256: str
    size: List[int] = dataclasses.field(default_factory=list)
    role: str = "album"
    image_id: Optional[str] = None
    pool: str = ""  # same as role
    num_faces: int = 0
    face_box: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None  # 5x2, raw-photo pixels
    face_square: Optional[List[float]] = None
    original_box: Optional[List[int]] = None  # (x, y, side) original-crop window
    face_fraction: Optional[float] = None
    padding: Optional[List[int]] = None  # left, top, right, bottom
    valid_rect: Optional[List[int]] = None  # x, y, w, h in crop coordinates
    padding_fraction: Optional[float] = None
    natural_size: Optional[int] = None
    resize_factor: Optional[float] = None
    original_ratio: Optional[float] = None
    eyes_closed: Optional[bool] = None
    identity_cosine: Optional[float] = None
    identity_outlier: Optional[bool] = None
    det_score: Optional[float] = None
    detector_iou: Optional[float] = None
    quality_score: Optional[float] = None
    session: Optional[str] = None
    album_reason: Optional[str] = None
    labels: Dict[str, str] = dataclasses.field(default_factory=dict)  # attributes to group results by
    source: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def has_face(self) -> bool:
        return self.face_box is not None and self.landmarks is not None

    def __post_init__(self) -> None:
        if self.split == "target":
            if self.role in PROTECTION_ROLES:
                raise ValueError(
                    f"HQ target {self.path!r} cannot use protectee role {self.role!r}"
                )
            object.__setattr__(self, "role", "target")
            object.__setattr__(self, "pool", "target")
            return
        role = self.role or self.pool
        if role == "eval":
            raise ValueError(
                f"legacy eval role is not accepted for {self.path!r}"
            )
        if role not in ROLES:
            raise ValueError(
                f"protectee {self.path!r} must declare album|query|gallery, got {role!r}"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "pool", role)

    @property
    def is_eval(self) -> bool:
        return self.role in PROTECTION_ROLES

    @property
    def is_protection(self) -> bool:
        return self.role in PROTECTION_ROLES


class BenchDataset:
    """The dataset rooted at ``root``, backed by its ``annotations.json``."""

    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)
        annotations = self._read_annotations()
        self.annotations = annotations
        self.meta: Dict[str, Any] = annotations.get("meta", {})
        self.records: List[ImageRecord] = [
            _record_from_entry(entry) for entry in annotations["images"]
        ]
        self._by_path: Dict[str, ImageRecord] = {r.path: r for r in self.records}
        self._by_identity: Dict[str, List[ImageRecord]] = {}
        for record in self.records:
            self._by_identity.setdefault(record.identity, []).append(record)
        for group in self._by_identity.values():
            group.sort(key=lambda r: r.path)

        self._dataset_json: Optional[Dict[str, Any]] = None
        self._identities: Optional[Dict[str, Any]] = None
        self._target_pool: Optional[Dict[str, Any]] = None
        self._noise_provenance: Optional[Dict[str, Any]] = None
        self._noise_filter = None
        self._noise_identity_ids: Optional[List[str]] = None
        self._physical_noise_identity_ids: Optional[List[str]] = None

    def _read_annotations(self) -> Dict[str, Any]:
        path = self.root / "annotations.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No annotations.json under {self.root}; assemble the dataset first "
                "(python -m afrbench.prepare)."
            )
        annotations = json.loads(path.read_text())
        schema = annotations.get("schema")
        if schema in _SUPERSEDED:
            raise ValueError(
                f"{path} is schema {schema!r}; this loader reads {SCHEMA!r} only "
                f"(fixed album/query/gallery roles). Re-assemble it with "
                f"python -m afrbench.prepare."
            )
        if schema != SCHEMA:
            raise ValueError(f"{path} has schema {schema!r}, expected {SCHEMA!r}")
        return annotations

    # Side files

    @property
    def info(self) -> Dict[str, Any]:
        """``dataset.json``: counts, seeds, provenance. Empty when absent."""
        if self._dataset_json is None:
            path = self.root / "dataset.json"
            self._dataset_json = json.loads(path.read_text()) if path.is_file() else {}
        return self._dataset_json

    @property
    def identities(self) -> Dict[str, Dict[str, Any]]:
        """``identities.json`` keyed by identity directory name."""
        if self._identities is None:
            path = self.root / "identities.json"
            payload = json.loads(path.read_text()) if path.is_file() else {}
            self._identities = payload.get("identities", {})
        return self._identities

    def identity_meta(self, identity: str) -> Dict[str, Any]:
        return self.identities.get(identity, {})

    @property
    def target_pool(self) -> Dict[str, Any]:
        """``targets/pool.json``: the 2000 target identities and their noise keys.

        Raises:
            ValueError: If the file was built for a different noise pack, filter or lock, or
                lists an identity this dataset's purge excludes.
        """
        if self._target_pool is None:
            path = self.root / "targets" / "pool.json"
            payload = json.loads(path.read_text()) if path.is_file() else {}
            if payload:
                from afrbench.noise import physical_pack_id

                identity_filter = self.noise_filter
                expected = {
                    "noise_pack_id": physical_pack_id(
                        self.noise_lmdb, self.noise_provenance
                    ),
                    "noise_filter_id": identity_filter.filter_id,
                    "lock_id": identity_filter.lock_id,
                    "purge_id": identity_filter.purge_id,
                }
                claimed = [key for key in expected if key in payload]
                stale = {
                    key: (payload.get(key), expected[key])
                    for key in claimed
                    if payload.get(key) != expected[key]
                }
                if stale:
                    raise ValueError(
                        "targets/pool.json was built for a different "
                        f"MillionCelebs pack/filter/lock: {stale}"
                    )
                forbidden = sorted(
                    {
                        str(entry.get("identity") or "")
                        for entry in payload.get("identities") or []
                    }
                    & identity_filter.identities
                )
                if forbidden:
                    raise ValueError(
                        f"targets/pool.json contains {len(forbidden)} identities "
                        f"forbidden by this lock: {forbidden[:5]}"
                    )
            self._target_pool = payload
        return self._target_pool

    # Access

    def protectee_identities(self, cohort: Optional[str] = None) -> List[str]:
        """Protectee identities, sorted, optionally only one cohort: ``core`` identities appear in
        all three FR training corpora, ``probe`` identities in none.
        """
        names = sorted(
            {r.identity for r in self.records if r.split == "protectee"}
        )
        if cohort is None:
            return names
        return [n for n in names if self.identity_meta(n).get("cohort") == cohort]

    def core_identities(self) -> List[str]:
        return self.protectee_identities("core")

    def probe_identities(self) -> List[str]:
        return self.protectee_identities("probe")

    def hq_target_identities(self) -> List[str]:
        """Target identities with high-resolution image files (the HQ targets)."""
        return sorted({r.identity for r in self.records if r.split == "target"})

    def target_identities(self) -> List[str]:
        """All 2000 MillionCelebs target identities, or the HQ targets when the pool file is
        absent; :meth:`hq_target_identities` is the subset with high-resolution files.
        """
        entries = self.target_pool.get("identities")
        if entries:
            return sorted(entry["identity"] for entry in entries)
        return self.hq_target_identities()

    def _hq_target_entries(self) -> List[Dict[str, Any]]:
        hq = list(self.target_pool.get("hq") or [])
        if hq:
            return hq
        return [
            {
                "identity": identity,
                "keys": list(self.identity_meta(identity).get("enrollment_keys") or []),
            }
            for identity in self.hq_target_identities()
        ]

    def target_keys(self, identity: Optional[str] = None) -> List[str]:
        """Enrollment keys of one target or of every target: the HQ targets' frozen 20-key entries,
        otherwise the pool file's keys, else those in ``identities.json``.
        """
        hq = self._hq_target_entries()
        pool = list(self.target_pool.get("identities") or [])
        if identity is None:
            keys = [key for entry in hq + pool for key in entry.get("keys", [])]
            return sorted(set(str(key) for key in keys))
        for entry in hq:
            if str(entry.get("identity")) == str(identity):
                return sorted(str(key) for key in entry.get("keys") or [])
        for entry in pool:
            if str(entry.get("identity")) == str(identity):
                return sorted(str(key) for key in entry.get("keys") or [])
        meta = self.identity_meta(str(identity))
        if meta.get("enrollment_keys"):
            return sorted(str(key) for key in meta["enrollment_keys"])
        raise KeyError(f"{identity} is not in targets/pool.json")

    def images_of(
        self,
        identity: str,
        split: str = "protectee",
        *,
        pool: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[ImageRecord]:
        """One identity's records in path order, optionally one role only; ``"eval"`` selects the
        query and gallery images together.
        """
        group = [r for r in self._by_identity.get(identity, []) if r.split == split]
        wanted = role or pool
        if wanted == "eval":
            group = [r for r in group if r.role in PROTECTION_ROLES]
        elif wanted is not None:
            group = [r for r in group if r.role == wanted]
        return group

    def eval_images(self, identity: str) -> List[ImageRecord]:
        """Query + gallery records: the frozen protection set."""
        return self.images_of(identity, role="eval")

    def protection_images(self, identity: str) -> List[ImageRecord]:
        return self.eval_images(identity)

    def album_images(self, identity: str) -> List[ImageRecord]:
        return self.images_of(identity, role="album")

    def query_images(self, identity: str) -> List[ImageRecord]:
        return self.images_of(identity, role="query")

    def gallery_images(self, identity: str) -> List[ImageRecord]:
        return self.images_of(identity, role="gallery")

    def reference_face_side(self) -> int:
        """Median face side (``face_square[2]``) over every query and gallery record."""
        sides: List[float] = []
        for identity in self.protectee_identities():
            for record in (*self.query_images(identity), *self.gallery_images(identity)):
                if record.face_square and len(record.face_square) >= 3:
                    sides.append(float(record.face_square[2]))
        if not sides:
            raise ValueError("no query/gallery face_square values to take a median of")
        sides.sort()
        mid = len(sides) // 2
        if len(sides) % 2:
            return int(round(sides[mid]))
        return int(round((sides[mid - 1] + sides[mid]) / 2.0))

    def frozen_roles(self, identity: str) -> Dict[str, List[str]]:
        """Stable image ids (paths) for the three frozen roles."""
        return {
            "album": [r.path for r in self.album_images(identity)],
            "query": [r.path for r in self.query_images(identity)],
            "gallery": [r.path for r in self.gallery_images(identity)],
        }

    def record(self, path: str) -> ImageRecord:
        """The record for one dataset-relative path."""
        try:
            return self._by_path[path]
        except KeyError:
            raise KeyError(path) from None

    def sessions_of(self, identity: str, pool: Optional[str] = None) -> List[str]:
        """Distinct session tags; album and eval images are disjoint over sessions."""
        return sorted(
            {
                r.session
                for r in self.images_of(identity, pool=pool)
                if r.session
            }
        )

    # Decoding

    def image(self, record: ImageRecord) -> torch.Tensor:
        """Decode a record's photo as ``[3, H, W]`` RGB float in ``[0, 255]``."""
        return decode_image((self.root / record.path).read_bytes())

    def original(self, record: ImageRecord) -> torch.Tensor:
        """The original crop of a photo: its ``original_box`` window, padded where needed."""
        if record.original_box is None:
            raise ValueError(f"{record.path} has no original-crop window (no face)")
        return extract_frame(
            self.image(record),
            tuple(record.original_box),
            tuple(record.size[:2]) if record.size else None,
        )

    def frame_plan(self, record: ImageRecord):
        """The stored frame plan of the original crop: window, padding, and valid rectangle."""
        if record.original_box is None:
            raise ValueError(f"{record.path} has no original-crop window (no face)")
        face_side = float(record.face_square[2]) if record.face_square else 0.0
        return plan_from_stored(
            record.original_box,
            (int(record.size[0]), int(record.size[1])),
            ratio=float(record.original_ratio or 2.0),
            face_side=face_side,
            pad=record.padding,
            valid_rect=record.valid_rect,
            padding_fraction=record.padding_fraction,
        )

    def geometry(self, record: ImageRecord) -> Dict[str, Any]:
        """The ``eval_geometry`` entry for one image: its landmarks and face box, shifted into the
        coordinates of the frame :meth:`original` returns.
        """
        if record.face_box is None or record.landmarks is None:
            raise ValueError(f"{record.path} has no face geometry")
        x, y, _side = record.original_box or (0, 0, 0)
        return {
            "landmarks": [[px - x, py - y] for px, py in record.landmarks],
            "face_box": [
                record.face_box[0] - x, record.face_box[1] - y,
                record.face_box[2] - x, record.face_box[3] - y,
            ],
        }

    # Noise pool

    @property
    def noise_lmdb(self) -> Path:
        return self.root / "noise.lmdb"

    @property
    def noise_provenance(self) -> Dict[str, Any]:
        """Provenance of the packed noise LMDB, validated on first access."""
        if self._noise_provenance is None:
            from afrlib.data.lmdb_io import is_packed, read_provenance
            from afrbench.noise import validate_physical_noise_provenance

            self._noise_provenance = (
                read_provenance(self.noise_lmdb) if is_packed(self.noise_lmdb) else {}
            )
            validate_physical_noise_provenance(self._noise_provenance)
        return self._noise_provenance

    @property
    def noise_filter(self):
        """This dataset's MillionCelebs identity purge, from ``purge/millioncelebs.json`` where it
        exists and ``noise/filter.json`` otherwise.
        """
        if self._noise_filter is None:
            from afrbench.noise import (
                filter_from_entries,
                load_noise_filter,
                lock_noise_identities,
            )

            purge_path = self.root / "purge" / "millioncelebs.json"
            filter_path = self.root / "noise" / "filter.json"
            lock_path = self.root / "final_lock.json"
            if purge_path.is_file():
                lock_id = ""
                if lock_path.is_file():
                    lock = json.loads(lock_path.read_text())
                    lock_id = str(lock.get("artifact_id") or "")
                    loaded = load_noise_filter(
                        purge_path,
                        expected_lock_id=lock_id or None,
                        require_entries=False,
                    )
                    if lock_id:
                        lock_union = lock_noise_identities(lock)
                        if loaded.identities != lock_union:
                            raise ValueError(
                                "purge/millioncelebs.json entries do not equal "
                                "the final-lock purge union"
                            )
                    self._noise_filter = loaded
                else:
                    self._noise_filter = load_noise_filter(
                        purge_path, require_entries=False
                    )
            elif filter_path.is_file():
                payload = json.loads(filter_path.read_text())
                self._noise_filter = filter_from_entries(
                    {str(item) for item in payload.get("identities") or []},
                    lock_id=str(payload.get("lock_id") or ""),
                    purge_id=str(payload.get("purge_id") or ""),
                    path=filter_path,
                )
            else:
                raise FileNotFoundError(
                    f"neither {purge_path} nor {filter_path} exists"
                )
        return self._noise_filter

    @property
    def noise_view_id(self) -> str:
        """Fingerprint of the noise pack under this dataset's purge; keys noise embeddings."""
        from afrbench.noise import physical_pack_id

        pack_id = physical_pack_id(self.noise_lmdb, self.noise_provenance)
        return self.noise_filter.view_id(pack_id)

    def noise_view_id_for(
        self, exclude_identities: Optional[Sequence[str]] = None
    ) -> str:
        """:attr:`noise_view_id` extended with extra excluded identities."""
        from afrbench.noise import selection_view_id

        return selection_view_id(self.noise_view_id, exclude_identities or ())

    @property
    def noise_aligned(self) -> bool:
        """Whether noise samples are already aligned crops, which are embedded directly rather than
        through the eval detector.
        """
        return bool(self.noise_provenance.get("aligned", False))

    def noise_keys(
        self, exclude_identities: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Every noise key this dataset's purge allows, minus ``exclude_identities``."""
        from afrlib.data.lmdb_io import iter_sample_records

        _ = self.noise_provenance
        identity_filter = self.noise_filter
        extra = {str(item) for item in exclude_identities or ()}
        return [
            record["key"]
            for record in iter_sample_records(self.noise_lmdb)
            if identity_filter.keep_record(record)
            and str(record.get("identity") or "") not in extra
        ]

    def noise_identities(self) -> Dict[str, List[str]]:
        """Allowed noise keys grouped by identity, from the packed ``identity`` field."""
        from afrlib.data.lmdb_io import iter_sample_records

        _ = self.noise_provenance
        grouped: Dict[str, List[str]] = {}
        identity_filter = self.noise_filter
        for record in iter_sample_records(self.noise_lmdb):
            if not identity_filter.keep_record(record):
                continue
            grouped.setdefault(record.get("identity", ""), []).append(record["key"])
        return grouped

    def physical_noise_identity_ids(self) -> List[str]:
        """Sorted unique identities in the physical pack, before any purge; a large pack's list is
        cached next to the dataset.
        """
        if self._physical_noise_identity_ids is not None:
            return list(self._physical_noise_identity_ids)
        from afrbench.noise import physical_pack_id
        from afrlib.data.lmdb_io import iter_sample_records

        provenance = self.noise_provenance
        pack_id = physical_pack_id(self.noise_lmdb, provenance)
        cache_path = (
            self.root.resolve().parent
            / "_shared"
            / "cache"
            / "millioncelebs"
            / f"identities-{pack_id}.json"
        )
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text())
            if payload.get("pack_id") == pack_id and payload.get("identities"):
                self._physical_noise_identity_ids = [
                    str(item) for item in payload["identities"]
                ]
                return list(self._physical_noise_identity_ids)
        from afrlib import get_logger

        found: set[str] = set()
        logger = get_logger(__name__)
        for index, record in enumerate(iter_sample_records(self.noise_lmdb), start=1):
            identity = str(record.get("identity") or "")
            if identity:
                found.add(identity)
            if index % 2_000_000 == 0:
                logger.info(
                    "indexing MillionCelebs identities: %d records, %d ids",
                    index, len(found),
                )
        identities = sorted(found)
        self._physical_noise_identity_ids = identities
        expected = provenance.get("n_identities") or provenance.get(
            "expected_source_identities"
        )
        if expected is not None and int(expected) != len(identities):
            raise ValueError(
                f"pack identity count {len(identities)} != provenance {expected}"
            )
        if len(identities) >= 10_000:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(cache_path.name + ".tmp")
            temporary.write_text(
                json.dumps({"pack_id": pack_id, "identities": identities}) + "\n"
            )
            temporary.replace(cache_path)
        return list(identities)

    def noise_identity_ids(self) -> List[str]:
        """Sorted unique identities that survive this dataset's protectee purge."""
        if self._noise_identity_ids is not None:
            return list(self._noise_identity_ids)
        identity_filter = self.noise_filter
        kept = [
            identity
            for identity in self.physical_noise_identity_ids()
            if identity_filter.keep_identity(identity)
        ]
        self._noise_identity_ids = kept
        return list(kept)

    def noise_identity_subset(
        self,
        count: int,
        *,
        seed: int = 0,
        ensure_identities: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """A seeded ``count``-identity subset of the purged pack, always keeping
        ``ensure_identities`` and returning everything when ``count`` is larger than the pack.
        """
        return select_identity_subset(
            self.noise_identity_ids(),
            count,
            seed=seed,
            ensure_identities=ensure_identities,
        )

    def noise_subset(
        self,
        count: int,
        *,
        seed: int = 0,
        exclude_identities: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """A seeded ``count``-image reservoir sample of the allowed noise keys, sorted; the eval
        config's seed lets a gallery-size sweep be repeated under another draw.
        """
        from afrlib.data.lmdb_io import iter_sample_records

        if count < 0:
            raise ValueError(f"noise subset count must be non-negative, got {count}")
        if count == 0:
            return []
        _ = self.noise_provenance
        identity_filter = self.noise_filter
        extra = {str(item) for item in exclude_identities or ()}
        rng = random.Random(f"noise:{seed}")
        reservoir: List[str] = []
        seen = 0
        for record in iter_sample_records(self.noise_lmdb):
            identity = str(record.get("identity") or "")
            if not identity_filter.keep_record(record) or identity in extra:
                continue
            key = str(record["key"])
            if seen < count:
                reservoir.append(key)
            else:
                replacement = rng.randrange(seen + 1)
                if replacement < count:
                    reservoir[replacement] = key
            seen += 1
        return sorted(reservoir)

    def noise_image(self, key: str) -> torch.Tensor:
        """Decode one noise sample by key; a key purged for this dataset raises ``KeyError``."""
        from afrlib.data.lmdb_io import read_field

        _ = self.noise_provenance
        if not self.noise_filter.keep_key(key):
            raise KeyError(f"Noise key {key!r} is forbidden by this dataset's lock")
        payload = read_field(self.noise_lmdb, key, "image")
        if payload is None:
            raise KeyError(f"No noise image for key {key!r}")
        return decode_image(payload)

    def iter_noise(self, keys: Optional[List[str]] = None) -> Iterator[torch.Tensor]:
        for key in keys if keys is not None else self.noise_keys():
            yield self.noise_image(key)


def select_identity_subset(
    available: Sequence[str],
    count: int,
    *,
    seed: int = 0,
    ensure_identities: Optional[Sequence[str]] = None,
) -> List[str]:
    """Pick ``count`` identities from ``available`` with a seeded draw, always keeping
    ``ensure_identities``, which can make the result larger than ``count``.
    """
    if count < 0:
        raise ValueError(f"identity subset count must be non-negative, got {count}")
    universe = sorted({str(item) for item in available if item})
    ensure = {str(item) for item in ensure_identities or () if item}
    forced = [item for item in universe if item in ensure]
    if count == 0 and not forced:
        return []
    if count >= len(universe):
        return list(universe)
    need = max(count, len(forced))
    if need >= len(universe):
        return list(universe)
    extra_n = need - len(forced)
    forced_set = set(forced)
    pool = [item for item in universe if item not in forced_set]
    rng = random.Random(f"noise-identities:{seed}")
    picked = rng.sample(pool, extra_n) if extra_n else []
    return sorted(forced_set | set(picked))


_RECORD_FIELDS = {field.name for field in dataclasses.fields(ImageRecord)}


def _record_from_entry(entry: Dict[str, Any]) -> ImageRecord:
    payload = {key: value for key, value in entry.items() if key in _RECORD_FIELDS}
    split = str(entry.get("split") or payload.get("split") or "protectee")
    payload["split"] = split
    if split == "target":
        if entry.get("role") in ROLES or entry.get("pool") in ROLES:
            raise ValueError(
                f"HQ target {entry.get('path')!r} cannot use a protectee role"
            )
        payload["role"] = "target"
        return ImageRecord(**payload)
    if "role" not in entry and "pool" not in entry:
        raise ValueError(
            f"protectee row {entry.get('path')!r} must declare album|query|gallery"
        )
    if entry.get("role") == "eval" or entry.get("pool") == "eval":
        raise ValueError(f"legacy eval role is not accepted for {entry.get('path')!r}")
    return ImageRecord(**payload)


def decode_image(payload: bytes) -> torch.Tensor:
    """Decode image bytes to ``[3, H, W]`` RGB float in ``[0, 255]``."""
    with Image.open(io.BytesIO(payload)) as image:
        return pil_to_tensor(image.convert("RGB")).to(torch.float32)
