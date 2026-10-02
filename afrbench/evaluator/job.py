"""The ``kind: eval`` job: score protect runs against one FR system."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from afrbench.configs import JobConfig, eval_code_digest, write_lock
from afrbench.dataset import BenchDataset
from afrbench.evaluator.embed import embed_source
from afrbench.evaluator.gallery import MIX_RS, build_scenario, protected_set_name
from afrbench.evaluator.store import EmbeddingSet, EmbeddingStore, cache_root
from afrbench.fr_system import FRSystem
from afrbench.runs import ProtectedRun
from afrlib import get_logger

logger = get_logger(__name__)

#: The base scenarios, in a fixed order.
BASE_SCENARIOS = ("1A", "1B", "2A", "2B")

#: The run key the method-independent No-Protection baseline (1B) is filed under.
BASELINE_RUN = "no_protection"

#: The scenario name the per-image visual-quality rows are filed under.
QUALITY_SCENARIO = "visual_quality"

#: Metric names written for a failed query when no ranked queries remain.
_FAILED_QUERY_METRICS = (
    "recall@K(user)",
    "recall@25",
    "recall@100",
    "AP",
    "AP@100",
    "NDCG@10",
    "rank1_appearance",
    "rank1_plurality",
    "rank1_majority",
    "rank5_appearance",
    "rank5_plurality",
    "rank5_majority",
    "rank10_appearance",
    "rank10_plurality",
    "rank10_majority",
)

#: Fewest images FID is computed on; below it, small-sample bias dominates.
FID_MIN_IMAGES = 50

#: The square size FID frames are written at (InceptionV3's native input).
FID_IMAGE_SIZE = 299


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def resolve_gallery_distractors(
    dataset: BenchDataset,
    distractor_cfg: Dict[str, Any],
    *,
    seed: int,
) -> Tuple[int, bool, Optional[List[str]]]:
    """Resolve an eval's ``gallery.distractors`` block into the distractor set to rank against.

    Args:
        dataset: The benchmark dataset.
        distractor_cfg: ``mode`` is ``count`` (default; ``count`` sampled images),
            ``identities`` (``identity_count`` sampled identities plus all targets), or
            ``full`` (the whole purged pack).
        seed: Sampling seed when the block sets none.

    Returns:
        ``(distractors, full_noise, exclude_identities)``; ``distractors`` is -1 for a
        full stream.

    Raises:
        ValueError: On an unknown mode or a 500,000-image sample.
    """
    mode = str(distractor_cfg.get("mode") or "count")
    explicit = [str(item) for item in distractor_cfg.get("exclude_identities") or []]
    if mode == "full":
        return -1, True, explicit or None
    if mode == "identities":
        identity_count = int(distractor_cfg["identity_count"])
        subset_seed = int(distractor_cfg.get("seed", seed))
        ensure = list(dataset.hq_target_identities()) + list(dataset.target_identities())
        selected = set(
            dataset.noise_identity_subset(
                identity_count, seed=subset_seed, ensure_identities=ensure
            )
        )
        exclude = [
            identity
            for identity in dataset.noise_identity_ids()
            if identity not in selected
        ]
        if explicit:
            exclude.extend(item for item in explicit if item not in selected)
        return -1, True, exclude or None
    if mode != "count":
        raise ValueError(f"unknown gallery.distractors.mode {mode!r}")
    distractors = int(distractor_cfg.get("count", 0))
    if distractors == 500_000:
        raise ValueError(
            "refusing a 500000-image MillionCelebs sample; "
            "set gallery.distractors.mode=full"
        )
    return distractors, False, explicit or None


def _seen(run: ProtectedRun, system: FRSystem) -> bool:
    """True when the run's surrogates include this system's FR (white-box)."""
    system_fr = json.dumps(system.fr_spec, sort_keys=True, default=str)
    return any(
        system_fr == json.dumps(spec, sort_keys=True, default=str)
        for spec in run.surrogates
    )


def _score_scenario(
    built, device, extra_state=None
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """One scenario's means and per-query values from one ranking; failed queries score 0."""
    from afrbench.evaluator.metrics import identification, retrieval
    from afrbench.evaluator.metrics.ranking import compute_ranking

    ranking = compute_ranking(
        built.query_embeddings, built.gallery_embeddings,
        built.query_labels, built.gallery_labels, top_k=10, device=device,
        extra_blocks=None if extra_state is not None else built.iter_extra_blocks(),
        extra_state=extra_state,
    )
    per_query: Dict[str, List[float]] = {}
    per_query.update(retrieval.per_query(
        ranking, built, ks=("user",), fixed_ks=(25, 100), ndcg_ks=(10,), ap_ks=(100,),
    ))
    per_query.update(identification.per_query(ranking, built))

    out: Dict[str, Any] = {
        name: retrieval.mean_with_failures(values, built.n_query_failed)
        for name, values in per_query.items()
    }
    out["det_fail_query"] = float(built.n_query_failed)
    out["det_fail_gallery"] = float(built.n_gallery_failed)
    out["gallery_size"] = int(ranking.gallery_size)
    out["query_size"] = int(built.query_embeddings.shape[0])
    if getattr(built, "noise_image_count", 0):
        out["noise_image_count"] = int(built.noise_image_count)
        out["noise_identity_count"] = int(built.noise_identity_count)
    return out, per_query


def _cached_visual_quality(
    out_dir: Path,
    run_key: str,
    run: ProtectedRun,
    dataset: BenchDataset,
    device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One run's visual quality, cached under ``vq_partial/`` so a restarted eval skips it."""
    safe = run_key.replace("/", "_")
    cache = out_dir / "vq_partial" / f"{safe}.json"
    rows_path = out_dir / "vq_partial" / f"{safe}.jsonl"
    if cache.is_file() and rows_path.is_file():
        logger.info("reuse visual quality cache %s", run_key)
        aggregate = json.loads(cache.read_text())
        quality_rows = [
            json.loads(line) for line in rows_path.read_text().splitlines() if line
        ]
        return aggregate, quality_rows
    aggregate, quality_rows = _visual_quality(run, dataset, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cache, _jsonable(aggregate))
    _write_jsonl(rows_path, quality_rows)
    return aggregate, quality_rows


def _visual_quality(
    run: ProtectedRun, dataset: BenchDataset, device
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """A run's visual-quality aggregate and per-image rows, on uint8-rounded unpadded frames."""
    from afrbench.evaluator.metrics import quality

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="afrbench-fid-") as tmp:
        protected_dir, clean_dir = Path(tmp) / "protected", Path(tmp) / "clean"
        protected_dir.mkdir()
        clean_dir.mkdir()
        n_total = len(run)
        for index, (path, protected) in enumerate(run.items()):
            record = dataset.record(path)
            if record.original_box is None:
                continue
            protected = protected.round().clamp(0, 255)
            clean = dataset.original(record).round().clamp(0, 255)
            valid = record.valid_rect
            with torch.inference_mode():
                rows.append({
                    "source_path": path,
                    **quality.per_image(protected, clean, device, valid_rect=valid),
                })
            _write_fid_frame(
                quality.valid_region(protected, valid),
                protected_dir / f"{index:07d}.png",
            )
            _write_fid_frame(
                quality.valid_region(clean, valid),
                clean_dir / f"{index:07d}.png",
            )
            if (len(rows) % 50) == 0 or len(rows) == n_total:
                logger.info(
                    "visual quality %s %d/%d",
                    run.root, len(rows), n_total,
                )
        if not rows:
            return {}, []
        aggregate: Dict[str, Any] = quality.mean_of(rows)
        aggregate["n_images"] = len(rows)
        if len(rows) >= FID_MIN_IMAGES:
            aggregate["fid"] = quality.fid(protected_dir, clean_dir, device)
        else:
            logger.info(
                "Skipping FID for %s: %d images is below the %d-image floor",
                run.root, len(rows), FID_MIN_IMAGES,
            )
    return aggregate, rows


def _write_fid_frame(image: torch.Tensor, path: Path) -> None:
    """Write one FID frame as a PNG, bicubic-resized to ``FID_IMAGE_SIZE`` square."""
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode, resize

    frame = resize(
        image, [FID_IMAGE_SIZE, FID_IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC, antialias=True,
    )
    array = (
        frame.detach().clamp(0, 255).to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    Image.fromarray(array).save(path, compress_level=1)


def embed_eval_sources(
    system: FRSystem,
    dataset: BenchDataset,
    store: EmbeddingStore,
    device: torch.device,
    *,
    eval_lists: Dict[str, Sequence[str]],
    loaded: Dict[str, ProtectedRun],
    run_sources: Dict[str, str],
    full_noise: bool,
    distractors: int,
    noise_seed: int,
    exclude_identities: Optional[List[str]],
    batch_size: int,
) -> None:
    """Embed originals, HQ targets, and protected runs, then the MillionCelebs gallery."""
    embed_source(
        "originals",
        system,
        dataset,
        store,
        device,
        eval_lists=eval_lists,
        batch_size=batch_size,
    )
    embed_source("targets", system, dataset, store, device, batch_size=batch_size)
    for run_key, run in loaded.items():
        embed_source(
            run_sources[run_key],
            system,
            dataset,
            store,
            device,
            run=run,
            batch_size=batch_size,
        )
    if full_noise:
        from afrbench.evaluator.noise_cache import build_full_noise_cache

        build_full_noise_cache(
            dataset=dataset, system=system, device=device, batch_size=batch_size
        )
    elif distractors > 0:
        embed_source(
            "noise",
            system,
            dataset,
            store,
            device,
            noise_count=distractors,
            noise_seed=noise_seed,
            noise_exclude_identities=exclude_identities,
            batch_size=batch_size,
        )


def eval_job(
    config: JobConfig,
    device: torch.device,
    *,
    reuse_visual_quality: bool = False,
    force: bool = False,
) -> Path:
    """Run one eval config and write ``per_image.jsonl`` and ``metrics.json``.

    Args:
        config: The eval job config.
        device: Device for embedding and ranking.
        reuse_visual_quality: Keep the existing visual-quality results and rescore only
            the protection metrics.
        force: Delete the existing lock and re-score cached scenarios.

    Returns:
        The output directory.
    """
    body = config.body
    dataset = BenchDataset(body["dataset"])
    system = FRSystem.from_dict(body["fr_system"])
    runs: Dict[str, str] = dict(body.get("runs", {}))
    protection = bool(body.get("protection", True))
    visual_quality = bool(body.get("visual_quality", True))
    gallery_cfg = dict(body.get("gallery", {}))
    distractor_cfg = dict(gallery_cfg.get("distractors", {}))
    seed = int(config.seed)
    noise_seed = int(distractor_cfg.get("seed", seed))
    distractors, full_noise, exclude_identities = resolve_gallery_distractors(
        dataset, distractor_cfg, seed=seed
    )
    batch_size = int(body.get("batch_size", 32))

    if force:
        lock = config.out_dir / "config.lock.yaml"
        if lock.is_file():
            lock.unlink()
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store = EmbeddingStore(cache_root(body.get("results_root", "results")) / "embeddings")

    loaded = {key: ProtectedRun.load(path) for key, path in runs.items()}
    run_sources = {key: protected_set_name(key, run.digest) for key, run in loaded.items()}
    code_digest = eval_code_digest(config) or ""
    for key, run in loaded.items():
        if not run.images:
            raise ValueError(f"Run {key!r} has no protected images under {run.root}")
        _assert_run_sources(dataset, run)
    reused_quality: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
    if reuse_visual_quality:
        if not visual_quality:
            raise ValueError(
                "cannot reuse visual quality when the eval config disables it"
            )
        reused_quality = _load_visual_quality(out_dir, loaded, system.name)
    eval_lists = _eval_lists(dataset, loaded)

    embed_eval_sources(
        system,
        dataset,
        store,
        device,
        eval_lists=eval_lists,
        loaded=loaded,
        run_sources=run_sources,
        full_noise=full_noise,
        distractors=distractors,
        noise_seed=noise_seed,
        exclude_identities=exclude_identities,
        batch_size=batch_size,
    )

    score_groups = group_runs_for_scoring(dataset, loaded, eval_lists)
    stitched_sources: Dict[str, Tuple[str, str]] = {}
    runs_meta: Dict[str, Any] = {}
    for spec in score_groups:
        variants = spec["variants"]
        if spec["stitch"]:
            source, digest = stitch_protected_embeddings(
                dataset=dataset,
                store=store,
                system_digest=system.digest,
                method=spec["score_key"],
                variants=variants,
                run_sources=run_sources,
                eval_lists=eval_lists,
            )
            stitched_sources[spec["score_key"]] = (source, digest)
            runs_meta[spec["score_key"]] = {
                "variants": {
                    key: {"root": str(run.root), "digest": run.digest}
                    for key, run in variants.items()
                },
                "digest": digest,
            }
        else:
            key = spec["score_key"]
            run = variants[key]
            runs_meta[key] = {"root": str(run.root), "digest": run.digest}

    per_image_rows: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {
        "fr_system": system.name,
        "scenarios": {},
        "runs": runs_meta,
    }

    def _prepare_unit(
        kind: str,
        *,
        run_key: str,
        protected_source: Optional[str],
        digest: str,
        seen,
        mix_r=None,
    ) -> Dict[str, Any]:
        exclude = tuple(sorted(exclude_identities or []))
        built = build_scenario(
            kind, dataset=dataset, store=store, system_digest=system.digest,
            protected_source=protected_source,
            eval_lists=eval_lists,
            distractors=max(distractors, 0),
            exclude_identities=list(exclude) or None, mix_r=mix_r, seed=seed,
            distractor_seed=noise_seed,
            full_noise=full_noise, system=system,
        )
        unit_dir = out_dir / "units" / hashlib.sha256(
            f"{run_key}\0{digest}"
            f"\0{system.digest}\0{built.name}\0{code_digest}".encode()
        ).hexdigest()[:16]
        return {
            "built": built,
            "unit_dir": unit_dir,
            "run_key": run_key,
            "seen": seen,
            "exclude": exclude,
        }

    def _record_unit(unit: Dict[str, Any], scored, per_query) -> None:
        built = unit["built"]
        scored["seen"] = unit["seen"]
        metrics["scenarios"].setdefault(unit["run_key"], {})[built.name] = scored
        per_image_rows.extend(_per_image_rows(
            dataset, unit["run_key"], system.name, built.name, built, per_query, unit["seen"]
        ))

    def _write_unit(unit: Dict[str, Any], scored, per_query) -> None:
        unit["unit_dir"].mkdir(parents=True, exist_ok=True)
        (unit["unit_dir"] / "metrics.json").write_text(
            json.dumps(_jsonable(scored), indent=2) + "\n"
        )
        (unit["unit_dir"] / "per_query.json").write_text(
            json.dumps(_jsonable(per_query)) + "\n"
        )
        (unit["unit_dir"] / "SUCCESS").write_text(unit["built"].name + "\n")

    pending: List[Dict[str, Any]] = []

    def _score_named(
        kind: str,
        *,
        run_key: str,
        protected_source: Optional[str],
        digest: str,
        seen,
        mix_r=None,
    ) -> None:
        pending.append(
            _prepare_unit(
                kind,
                run_key=run_key,
                protected_source=protected_source,
                digest=digest,
                seen=seen,
                mix_r=mix_r,
            )
        )

    def _commit_units(units: List[Dict[str, Any]]) -> None:
        from collections import defaultdict

        from afrbench.evaluator.mc_rank import extra_states_for_own_sims, query_digest
        from afrbench.evaluator.metrics.ranking import _prepare_primary

        need: List[Dict[str, Any]] = []
        for unit in units:
            if (unit["unit_dir"] / "SUCCESS").is_file() and not force:
                scored = json.loads((unit["unit_dir"] / "metrics.json").read_text())
                per_query = json.loads((unit["unit_dir"] / "per_query.json").read_text())
                _record_unit(unit, scored, per_query)
            else:
                need.append(unit)
        groups: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
        for unit in need:
            groups[
                query_digest(unit["built"].query_embeddings),
                unit.get("exclude") or (),
            ].append(unit)
        for group in groups.values():
            extra_states = None
            first = group[0]["built"]
            if full_noise and first.extra_block_iter is not None and len(group) > 1:
                owns = []
                for unit in group:
                    primary = _prepare_primary(
                        unit["built"].query_embeddings,
                        unit["built"].gallery_embeddings,
                        unit["built"].query_labels,
                        unit["built"].gallery_labels,
                        top_k=10,
                        device=torch.device(device),
                    )
                    owns.append(primary.own_sims)
                extra_states = extra_states_for_own_sims(
                    first.query_embeddings,
                    first.iter_extra_blocks(),
                    owns,
                    top_k=10,
                    device=device,
                )
            for index, unit in enumerate(group):
                extra = None if extra_states is None else extra_states[index]
                scored, per_query = _score_scenario(unit["built"], device, extra)
                _write_unit(unit, scored, per_query)
                _record_unit(unit, scored, per_query)

    # The method-independent No-Protection baseline (1B), computed once.
    _score_named(
        "1B", run_key=BASELINE_RUN, protected_source=None, digest="baseline", seen=None
    )

    for run_key, run in loaded.items():
        embed_source(
            run_sources[run_key], system, dataset, store, device,
            run=run, batch_size=batch_size,
        )
        seen = _seen(run, system) if protection else None
        if visual_quality:
            if reuse_visual_quality:
                aggregate, quality_rows = reused_quality[run_key]
            else:
                aggregate, quality_rows = _cached_visual_quality(
                    out_dir, run_key, run, dataset, device
                )
            metrics.setdefault("visual_quality", {})[run_key] = aggregate
            if reuse_visual_quality:
                per_image_rows.extend(quality_rows)
            else:
                per_image_rows.extend(
                    _quality_row(dataset, run_key, system.name, seen, row)
                    for row in quality_rows
                )

    if protection:
        for spec in score_groups:
            variants = spec["variants"]
            if spec["stitch"]:
                source, digest = stitched_sources[spec["score_key"]]
            else:
                key = spec["score_key"]
                source, digest = run_sources[key], variants[key].digest
            seen = _seen_group(list(variants.values()), system)
            scenarios = [(kind, None) for kind in BASE_SCENARIOS if kind != "1B"]
            # The interior gallery-mix points; the endpoints are 2B (r=0) and 2A (r=1).
            scenarios += [("mix-gallery", r) for r in MIX_RS if r not in (0.0, 1.0)]
            pending.extend(
                _prepare_unit(
                    kind,
                    run_key=spec["score_key"],
                    protected_source=source,
                    digest=digest,
                    seen=seen,
                    mix_r=mix_r,
                )
                for kind, mix_r in scenarios
            )

    _commit_units(pending)
    _write_jsonl(out_dir / "per_image.jsonl", per_image_rows)
    _write_json(out_dir / "metrics.json", _jsonable(metrics))
    write_lock(config)
    logger.info("eval %s -> %s", system.name, out_dir)
    return out_dir


def _load_visual_quality(
    out_dir: Path,
    runs: Dict[str, ProtectedRun],
    system_name: str,
) -> Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Load a complete quality section from an existing matching eval output."""
    metrics_path = out_dir / "metrics.json"
    rows_path = out_dir / "per_image.jsonl"
    if not metrics_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(
            "visual-quality reuse needs existing metrics.json and per_image.jsonl"
        )

    metrics = json.loads(metrics_path.read_text())
    if metrics.get("fr_system") != system_name:
        raise ValueError(
            "existing visual-quality output uses a different FR system"
        )
    aggregates = metrics.get("visual_quality") or {}
    expected = set(runs)
    if set(aggregates) != expected:
        raise ValueError(
            "existing visual-quality runs do not match the eval config: "
            f"expected {sorted(expected)}, found {sorted(aggregates)}"
        )

    rows_by_run: Dict[str, List[Dict[str, Any]]] = {key: [] for key in runs}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("scenario") != QUALITY_SCENARIO:
            continue
        run_key = row.get("run")
        if run_key in rows_by_run and row.get("fr_system") == system_name:
            rows_by_run[run_key].append(row)
    missing = [key for key, rows in rows_by_run.items() if not rows]
    if missing:
        raise ValueError(
            f"existing output has no visual-quality rows for {sorted(missing)}"
        )
    return {
        key: (dict(aggregates[key]), rows_by_run[key])
        for key in runs
    }


def _assert_run_sources(dataset: BenchDataset, run: ProtectedRun) -> None:
    """Raise unless every output exists and each source is a distinct query or gallery image."""
    missing_files = [str(image.output) for image in run.images if not image.output.is_file()]
    if missing_files:
        raise ValueError(f"{run.root} is missing protected outputs: {missing_files[:5]}")
    declared = list(run.source_paths)
    if len(declared) != len(set(declared)):
        raise ValueError(f"{run.root} declares duplicate source paths")
    extra = []
    for path in declared:
        try:
            record = dataset.record(path)
        except Exception as exc:
            raise ValueError(f"{run.root} source {path} is not in the dataset") from exc
        if record.role not in ("query", "gallery"):
            extra.append(path)
    if extra:
        raise ValueError(
            f"{run.root} includes non-eval source paths: {extra[:5]}"
        )


def _method_key(run_key: str) -> str:
    """``adv_diffusion@103354`` -> ``adv_diffusion``; untargeted keys stay themselves."""
    return run_key.split("@", 1)[0]


def _run_coverage(dataset: BenchDataset, run: ProtectedRun) -> Set[str]:
    return {dataset.record(path).identity for path in run.source_paths}


def _combined_digest(variants: Dict[str, ProtectedRun]) -> str:
    payload = "\n".join(f"{key}:{run.digest}" for key, run in sorted(variants.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seen_group(runs: Sequence[ProtectedRun], system: FRSystem) -> bool:
    """Shared seen/unseen tag; variants of one method must agree."""
    values = {_seen(run, system) for run in runs}
    if len(values) != 1:
        raise ValueError(
            "surrogates disagree across variants of one method; "
            f"seen tags {sorted(values)}"
        )
    return next(iter(values))


def group_runs_for_scoring(
    dataset: BenchDataset,
    loaded: Dict[str, ProtectedRun],
    eval_lists: Dict[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    """Group runs by method (key before ``@``); stitch variants that split the protectees.

    Returns:
        One spec per scored key, with ``score_key``, ``stitch`` and ``variants``.

    Raises:
        ValueError: When a method's variants neither split the protectees nor each cover
            all of them.
    """
    from collections import defaultdict

    universe = set(eval_lists)
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in loaded:
        groups[_method_key(key)].append(key)

    specs: List[Dict[str, Any]] = []
    for method, keys in groups.items():
        coverages = {key: _run_coverage(dataset, loaded[key]) for key in keys}
        if len(keys) == 1:
            key = keys[0]
            specs.append({
                "score_key": key,
                "stitch": False,
                "variants": {key: loaded[key]},
            })
            continue
        if all(coverages[key] == universe for key in keys):
            for key in keys:
                specs.append({
                    "score_key": key,
                    "stitch": False,
                    "variants": {key: loaded[key]},
                })
            continue
        pairwise = list(coverages.values())
        disjoint = True
        for index, left in enumerate(pairwise):
            for right in pairwise[index + 1 :]:
                if left & right:
                    disjoint = False
                    break
            if not disjoint:
                break
        union = set().union(*pairwise) if pairwise else set()
        if disjoint and union == universe:
            specs.append({
                "score_key": method,
                "stitch": True,
                "variants": {key: loaded[key] for key in keys},
            })
            continue
        raise ValueError(
            f"cannot score {method}: variant coverages { {k: sorted(v) for k, v in coverages.items()} } "
            f"are neither disjoint-and-complete nor each the full set {sorted(universe)}"
        )
    return specs


def stitch_protected_embeddings(
    *,
    dataset: BenchDataset,
    store: EmbeddingStore,
    system_digest: str,
    method: str,
    variants: Dict[str, ProtectedRun],
    run_sources: Dict[str, str],
    eval_lists: Dict[str, Sequence[str]],
) -> Tuple[str, str]:
    """Stitch one protected embedding set from the variant that covered each protectee.

    Returns:
        The stitched set's store name and combined digest.

    Raises:
        ValueError: When a protectee, or one of its embeddings, is missing from its
            variant; detection failures stay failures.
    """
    import numpy as np

    combined = _combined_digest(variants)
    source = protected_set_name(method, combined)
    if store.has(source, system_digest):
        return source, combined

    assigned: Dict[str, str] = {}
    for key, run in variants.items():
        for identity in _run_coverage(dataset, run):
            if identity in assigned:
                raise ValueError(
                    f"{method}: identity {identity!r} is in both "
                    f"{assigned[identity]!r} and {key!r}"
                )
            assigned[identity] = key
    missing = sorted(set(eval_lists) - set(assigned))
    if missing:
        raise ValueError(
            f"{method}: no variant covered {missing}; missing protected "
            "outputs are execution errors, not detection failures"
        )

    sets = {key: store.get(run_sources[key], system_digest) for key in variants}
    ids: List[str] = []
    rows: List[Any] = []
    failed: List[str] = []
    for identity in sorted(eval_lists):
        key = assigned[identity]
        embedding_set = sets[key]
        index = embedding_set.row_index()
        failed_set = set(embedding_set.failed_ids)
        for path in eval_lists[identity]:
            if path in failed_set:
                failed.append(path)
                continue
            if path not in index:
                raise ValueError(
                    f"{method}: {path} of {identity} is not in assigned "
                    f"store {key!r}; missing protected embeddings are "
                    "execution errors"
                )
            ids.append(path)
            rows.append(np.asarray(embedding_set.embeddings[index[path]]))
    if rows:
        matrix = np.stack(rows).astype(np.float32)
    else:
        matrix = np.zeros((0, 1), dtype=np.float32)
    store.put(
        source,
        system_digest,
        EmbeddingSet(ids=ids, embeddings=matrix, failed_ids=failed),
        meta={"method": method, "variants": sorted(variants), "stitched": True},
    )
    return source, combined


def _eval_lists(
    dataset: BenchDataset, runs: Dict[str, ProtectedRun]
) -> Dict[str, List[str]]:
    """Protectee -> eval images protected by any run, so every method uses the same split."""
    lists: Dict[str, List[str]] = {}
    for run in runs.values():
        for path in run.source_paths:
            lists.setdefault(dataset.record(path).identity, []).append(path)
    return {k: sorted(set(v)) for k, v in lists.items()}


def _per_image_rows(
    dataset: BenchDataset,
    run_key: str,
    system_name: str,
    scenario: str,
    built,
    per_query: Dict[str, List[float]],
    seen,
) -> List[Dict[str, Any]]:
    """One row per query image with its identity and metrics; undetected queries score 0."""
    rows = []
    metric_names = list(per_query.keys()) or list(_FAILED_QUERY_METRICS)
    for index, query_id in enumerate(built.query_ids):
        record = dataset.record(query_id)
        row: Dict[str, Any] = {
            "run": run_key, "fr_system": system_name, "scenario": scenario,
            "query_id": query_id, "seen": seen,
        }
        row["identity"] = record.identity
        row.update({name: values[index] for name, values in per_query.items()})
        rows.append(row)
    for query_id in getattr(built, "failed_query_ids", []) or ():
        record = dataset.record(query_id)
        row = {
            "run": run_key, "fr_system": system_name, "scenario": scenario,
            "query_id": query_id, "seen": seen,
        }
        row["identity"] = record.identity
        row.update({name: 0.0 for name in metric_names})
        rows.append(row)
    return rows


def _quality_row(
    dataset: BenchDataset, run_key: str, system_name: str, seen, values: Dict[str, Any]
) -> Dict[str, Any]:
    """One protected image's quality metrics as a per-image row under ``visual_quality``."""
    record = dataset.record(values["source_path"])
    row: Dict[str, Any] = {
        "run": run_key, "fr_system": system_name, "scenario": QUALITY_SCENARIO,
        "query_id": values["source_path"], "seen": seen,
    }
    row["identity"] = record.identity
    row.update({k: v for k, v in values.items() if k != "source_path"})
    return row


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in rows))
    os.replace(temporary, path)
