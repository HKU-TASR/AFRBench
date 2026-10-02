"""Train, fit, and protect job bodies that run :mod:`afrlib` methods on the dataset."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from afrlib import AFR_REGISTRY, get_logger
from afrlib.afr.base import AFRMethod
from afrlib.contracts import EvalData, FitData, TrainData, assemble_stage
from afrlib.fr import build_fr_ensemble
from afrlib.utils.face_crop import prepare_input
from afrbench.configs import JobConfig, is_current, write_lock
from afrbench.dataset import BenchDataset

logger = get_logger(__name__)


def seed_job_rng(seed: int) -> None:
    """Seed Python, NumPy, and Torch so protect-time sampling is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Role preparation


def _collate(crops: List[torch.Tensor], spec) -> "torch.Tensor | List[torch.Tensor]":
    """Stack a fixed-size spec to ``[N,3,S,S]``; keep a ragged list when size is None."""
    if spec.size is None:
        return crops
    return torch.stack(crops)


def _prepare_records(
    records, dataset: BenchDataset, spec, device, cap: Optional[int] = None
) -> "torch.Tensor | List[torch.Tensor] | None":
    """Dataset records prepared to ``spec``, skipping faceless ones; ``None`` if none remain."""
    images = []
    for record in (records[:cap] if cap else records):
        if not record.has_face:
            continue
        photo = dataset.image(record).to(device)
        crop, _plan = prepare_input(photo, spec, record.landmarks, record.face_box)
        images.append(crop)
    if not images:
        return None
    return _collate(images, spec)


def _prepare_thumbnails(
    keys: Sequence[str], dataset: BenchDataset, spec, device
) -> "torch.Tensor | List[torch.Tensor] | None":
    """Pre-aligned noise crops prepared to ``spec``; their landmarks are the ArcFace template at
    the crop's scale, so no detector pass is needed.
    """
    from frbench import ARCFACE_112_TEMPLATE

    images = []
    for key in keys:
        crop = dataset.noise_image(key).to(device)
        side = int(crop.shape[-1])
        landmarks = (ARCFACE_112_TEMPLATE.to(torch.float64) * (side / 112.0)).tolist()
        face_box = [side * 0.1, side * 0.1, side * 0.9, side * 0.9]
        prepared, _plan = prepare_input(crop, spec, landmarks, face_box)
        images.append(prepared)
    if not images:
        return None
    return _collate(images, spec)


def _target_pool_keys(dataset: BenchDataset, config: JobConfig, size: int) -> List[str]:
    """One noise key for each of ``size`` target identities, seeded on the job seed; the draw
    spans the target identities and holds out the job's own target.
    """
    identities = [
        name for name in dataset.target_identities() if name != config.body.get("target")
    ]
    if not identities:
        return []
    rng = random.Random(f"target_pool:{config.seed}")
    drawn = rng.sample(identities, min(size, len(identities)))
    keys = []
    for identity in sorted(drawn):
        candidates = dataset.target_keys(identity)
        if candidates:
            keys.append(rng.choice(candidates))
    return keys


def _efficient_decision_references(
    dataset: BenchDataset,
    method: AFRMethod,
    eval_records: Sequence[Any],
    config: JobConfig,
) -> Optional[List[Any]]:
    """One seeded album reference per eval image, from another session where possible, or ``None``
    unless the method's protect stage reads ``reference_imgs``.
    """
    if "reference_imgs" not in method.input_specs("protect"):
        return None
    album_paths = _as_path_list((config.body.get("protectee") or {}).get("album") or [])
    album = [dataset.record(path) for path in album_paths]
    if not album:
        raise ValueError("EfficientDecision requires album images for reference_imgs")
    paired = []
    for record in eval_records:
        disjoint = [
            item for item in album
            if item.session and record.session and item.session != record.session
        ]
        pool = disjoint or album
        rng = random.Random(f"ed-ref:{config.seed}:{record.path}")
        paired.append(pool[rng.randrange(len(pool))])
    return paired


def _role_pool(
    dataset: BenchDataset,
    method: AFRMethod,
    device,
    config: JobConfig,
    *,
    stage: str,
) -> Dict[str, Any]:
    """The target and scalar roles for one config, prepared to ``stage``'s input specs."""
    pool: Dict[str, Any] = {}
    roles = dict(config.body.get("roles", {}))
    target_identity = config.body.get("target")
    specs = method.input_specs(stage)

    if "target_img" in roles:
        records = [dataset.record(p) for p in _as_path_list(roles["target_img"])]
        batch = _prepare_records(records, dataset, specs.get("target_img"), device, 1)
        if batch is not None:
            pool["target_img"] = batch
    elif target_identity and specs.get("target_img") is not None:
        records = dataset.images_of(target_identity, split="target")
        batch = (
            _prepare_records(records, dataset, specs.get("target_img"), device, 1)
            if records
            else _prepare_thumbnails(
                dataset.target_keys(target_identity)[:1],
                dataset, specs.get("target_img"), device,
            )
        )
        if batch is not None:
            pool["target_img"] = batch

    if "target_pool" in roles:
        records = [dataset.record(p) for p in _as_path_list(roles["target_pool"])]
        batch = _prepare_records(records, dataset, specs.get("target_pool"), device)
        if batch is not None:
            pool["target_pool"] = batch
    elif dataset.target_pool and specs.get("target_pool") is not None:
        size = int(config.body.get("target_pool_size", 32))
        keys = _target_pool_keys(dataset, config, size)
        batch = _prepare_thumbnails(keys, dataset, specs.get("target_pool"), device)
        if batch is not None:
            pool["target_pool"] = batch

    # Scalar roles pass straight through.
    for role in ("prompt", "message", "makeup_reference_name"):
        if role in roles:
            pool[role] = roles[role]
    return pool


def _as_path_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _output_stem(record) -> str:
    ident = getattr(record, "image_id", None) or record.path
    return str(ident).replace("/", "__").replace("\\", "__")


def _check_protected(
    afr: str, originals: Sequence[torch.Tensor], protected: Any
) -> List[int]:
    """Check the count, type, shape, finiteness and range of ``protect()`` output, and return the
    indices it left unmodified, which a method's documented fallback may do.
    """
    if isinstance(protected, torch.Tensor) or not isinstance(protected, (list, tuple)):
        raise TypeError(
            f"{afr}.protect() must return one protected original per input image "
            f"as a list; got {type(protected).__name__}."
        )
    if len(protected) != len(originals):
        raise TypeError(
            f"{afr}.protect() returned {len(protected)} frames for "
            f"{len(originals)} images."
        )
    for index, (original, result) in enumerate(zip(originals, protected)):
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"{afr}.protect() item {index} is {type(result).__name__}, not a tensor"
            )
        if result.ndim != 3 or int(result.shape[0]) != 3:
            raise ValueError(
                f"{afr}.protect() item {index} has shape {tuple(result.shape)}, "
                "expected RGB [3,H,W]"
            )
        if tuple(result.shape) != tuple(original.shape):
            raise ValueError(
                f"{afr}.protect() returned {tuple(result.shape)} for image "
                f"{index}, but the original is {tuple(original.shape)}"
            )
        if not torch.isfinite(result).all():
            raise ValueError(f"{afr}.protect() item {index} contains non-finite values")
        minimum = float(result.min().item())
        maximum = float(result.max().item())
        if minimum < -1e-3 or maximum > 255.0 + 1e-3:
            raise ValueError(
                f"{afr}.protect() item {index} is outside [0,255] "
                f"(min={minimum}, max={maximum})"
            )
    unchanged = [
        index
        for index, (original, result) in enumerate(zip(originals, protected))
        if torch.equal(result.detach().to(original.dtype), original)
    ]
    if unchanged:
        logger.warning(
            "%s.protect() returned the unmodified original for %d/%d frames "
            "(indices %s); those frames are unprotected",
            afr, len(unchanged), len(originals), unchanged[:8],
        )
    return unchanged


def _build_method(config: JobConfig, device, fr_models) -> AFRMethod:
    """The method instance for a job, from ``from_run`` or fresh."""
    cls = AFR_REGISTRY[config.afr]
    overrides = config.afr_overrides
    if config.from_run:
        return cls.from_run(Path(config.from_run), device, fr_models, **overrides)
    return cls(device, fr_models, **overrides)


# Job bodies


def _corpus_filters(corpus_cfg: Dict[str, Any]) -> List[Any]:
    """The purge filters a corpus block names: identities for a labelled corpus, image names for
    one without an identity field.
    """
    from afrlib.data import ExcludeIdentities, ExcludeImages

    filters: List[Any] = []
    identities = list(corpus_cfg.get("exclude_identities", []))
    if identities:
        filters.append(ExcludeIdentities(identities, on_unmatched="error"))
    names = list(corpus_cfg.get("exclude_images", []))
    if names:
        filters.append(ExcludeImages(names, on_unmatched="error"))
    return filters


def train_job(config: JobConfig, device: torch.device, *, force: bool = False) -> Path:
    """Corpus-train one method into ``save_path``; skip when the lock matches."""
    if force:
        _clear_lock(config)
    if not force and is_current(config):
        logger.info("train %s already done; skipping", config.save_path)
        return config.out_dir
    from afrlib.data import get_corpus

    cls = AFR_REGISTRY[config.afr]
    frs = build_fr_ensemble(config.surrogates, device)
    method = cls(device, frs, **config.afr_overrides)
    dataset = BenchDataset(config.body["dataset"])
    corpus_cfg = dict(config.body.get("corpus", {}))
    corpus = get_corpus(
        corpus_cfg.get("name", cls.training_corpus),
        filters=_corpus_filters(corpus_cfg),
        root=corpus_cfg.get("root"),
    )
    pool = {**_role_pool(dataset, method, device, config, stage="train"), "corpus": corpus}
    method.train(assemble_stage(cls.train_inputs, TrainData, pool))
    method.save_run(config.out_dir)
    write_lock(config, extra={
        "method_config": method.get_config(),
        "surrogate_describe": [model.describe() for model in frs],
    })
    return config.out_dir


def fit_job(config: JobConfig, device: torch.device, *, force: bool = False) -> Path:
    """Fit one method to one protectee into ``save_path``; skip when current."""
    if force:
        _clear_lock(config)
    if not force and is_current(config):
        logger.info("fit %s already done; skipping", config.save_path)
        return config.out_dir
    frs = build_fr_ensemble(config.surrogates, device)
    method = _build_method(config, device, frs)
    dataset = BenchDataset(config.body["dataset"])
    album_paths = _as_path_list(config.body["protectee"]["album"])
    album_records = [dataset.record(p) for p in album_paths]
    album = _prepare_records(
        album_records,
        dataset,
        method.input_specs("fit").get("protectee_album"),
        device,
    )
    method.fit(assemble_stage(method.fit_inputs, FitData, {"protectee_album": album}))
    method.save_run(config.out_dir)
    write_lock(config, extra={
        "method_config": method.get_config(),
        "surrogate_describe": [model.describe() for model in frs],
    })
    return config.out_dir


def _clear_lock(config: JobConfig) -> None:
    lock = config.out_dir / "config.lock.yaml"
    if lock.is_file():
        lock.unlink()


def protect_job(config: JobConfig, device: torch.device, *, force: bool = False) -> Path:
    """Protect one protectee's images into ``save_path``, writing each frame as ``.npy`` with a
    JPEG preview and a ``manifest.json``; skipped when the lock is current.
    """
    if force:
        _clear_lock(config)
    if not force and is_current(config):
        logger.info("protect %s already done; skipping", config.save_path)
        return config.out_dir
    seed_job_rng(int(config.seed))
    started = time.monotonic()
    frs = build_fr_ensemble(config.surrogates, device)
    method = _build_method(config, device, frs)
    dataset = BenchDataset(config.body["dataset"])

    # Fit in place when the method has a fit stage and no fitted run was given.
    if method.implements_fit() and not method.fitted:
        album_paths = _as_path_list(config.body["protectee"]["album"])
        album_records = [dataset.record(p) for p in album_paths]
        album = _prepare_records(
            album_records,
            dataset,
            method.input_specs("fit").get("protectee_album"),
            device,
        )
        method.fit(assemble_stage(method.fit_inputs, FitData, {"protectee_album": album}))

    image_paths = _as_path_list(config.body["protectee"]["images"])
    records = [dataset.record(p) for p in image_paths]
    pool_roles = _role_pool(dataset, method, device, config, stage="protect")
    batch_size = int(config.body.get("batch_size", 8))
    reference_records = _efficient_decision_references(
        dataset, method, records, config,
    )

    out_dir = config.out_dir
    outputs_dir = out_dir / "outputs"
    preview_dir = out_dir / "preview"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "afr": config.afr,
        "protectee": config.body["protectee"]["identity"],
        "seed": config.seed,
        "target": config.body.get("target"),
        "surrogates": [m.get_spec() for m in method.fr_models],
        "runtime_surrogates": [m.get_spec() for m in method.fr_models],
        "training_surrogates": list(config.surrogates),
        "images": {},
    }

    def _batches(records):
        for start in range(0, len(records), batch_size):
            yield records[start: start + batch_size]

    for chunk in _batches(records):
        originals, geometry, kept = [], [], []
        for record in chunk:
            if record.original_box is None or not record.has_face:
                raise ValueError(f"{record.path} has no geometry; refusing to skip")
            original = dataset.original(record).to(device)
            originals.append(original)
            geometry.append(dataset.geometry(record))
            kept.append(record)
        if not originals:
            continue
        chunk_roles = dict(pool_roles)
        if reference_records is not None:
            by_path = {rec.path: ref for rec, ref in zip(records, reference_records)}
            ref_chunk = [by_path[rec.path] for rec in kept]
            refs = _prepare_records(
                ref_chunk,
                dataset,
                method.input_specs("protect").get("reference_imgs"),
                device,
            )
            if refs is None:
                raise ValueError("EfficientDecision reference_imgs could not be prepared")
            chunk_roles["reference_imgs"] = refs
        protected = method.protect(assemble_stage(
            method.eval_inputs, EvalData,
            {**chunk_roles, "eval_imgs": originals, "eval_geometry": geometry},
        ))
        unchanged = set(_check_protected(config.afr, originals, protected))
        for index, (record, original, protected_original) in enumerate(
            zip(kept, originals, protected)
        ):
            array = protected_original.detach().cpu().numpy().astype(np.float32, copy=False)
            stem = _output_stem(record)
            npy_path = outputs_dir / f"{stem}.npy"
            # Write then rename: a crash cannot leave a truncated array, and
            # hardlinked copies of an earlier run keep their own inode.
            npy_tmp = outputs_dir / f".{stem}.tmp.npy"
            np.save(npy_tmp, array)
            os.replace(npy_tmp, npy_path)
            preview = protected_original.detach().cpu().to(torch.float32).div(255.0).clamp(0, 1)
            preview_path = preview_dir / f"{stem}.jpg"
            preview_tmp = preview_dir / f".{stem}.tmp.jpg"
            to_pil_image(preview).save(preview_tmp, quality=90)
            os.replace(preview_tmp, preview_path)
            manifest["images"][record.path] = {
                "output": str(npy_path.relative_to(out_dir)),
                "image_id": getattr(record, "image_id", None) or record.path,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "original_box": list(record.original_box) if record.original_box else None,
                "target": config.body.get("target"),
                "seed": config.seed,
                "unmodified": index in unchanged,
            }

    manifest["unmodified_count"] = sum(
        1 for row in manifest["images"].values() if row.get("unmodified")
    )
    manifest["seconds"] = time.monotonic() - started
    temporary = out_dir / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(temporary, out_dir / "manifest.json")
    write_lock(config, extra={
        "method_config": method.get_config(),
        "surrogate_describe": [model.describe() for model in frs],
        "output_count": len(manifest["images"]),
    })
    return out_dir
