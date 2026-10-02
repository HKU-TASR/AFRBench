"""Run the AFRBench-basic protocol on one method.

``python -m afrbench.benchmark --method <name | package.module:Class> --dataset D --out O``
writes the method's train/fit/protect/eval job configs under ``O/configs`` and runs
them in order, one ``python -m afrbench <config>`` process per job. Finished jobs
are skipped, so re-running the same command resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from afrbench import protocol as P

KINDS = ("train", "fit", "protect", "eval")
SUMMARY_METRICS = ("recall@25", "recall@K(user)", "AP", "rank1_appearance", "det_fail_query")
QUALITY_METRICS = ("psnr", "ssim", "lpips", "fid")


def method_dir(method: str) -> str:
    """Directory and run name of a method: its registry key, or a plugin's class name."""
    if ":" not in method:
        return method
    return re.sub(r"[^0-9A-Za-z_.-]", "_", method.rsplit(":", 1)[1]).lower()


def parse_fr(value: str) -> Dict[str, Any]:
    """``backbone/loss/dataset`` (an FRBench model) or ``package.module:Class``."""
    if ":" in value:
        return {"backend": value}
    parts = value.split("/")
    if len(parts) != 3:
        raise ValueError(f"expected backbone/loss/dataset or package.module:Class, got {value!r}")
    return {"backend": "frbench", "backbone": parts[0], "loss": parts[1], "dataset": parts[2]}


def _key_values(items: Sequence[str], flag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def _dataset_key(dataset_root: Path) -> str:
    return hashlib.sha256((dataset_root / "annotations.json").read_bytes()).hexdigest()[:12]


class _Writer:
    """Writes configs under ``<out>/configs`` without ever replacing a different file."""

    def __init__(self, root: Path, overwrite: bool):
        self.root = root
        self.overwrite = overwrite
        self.paths: Dict[str, List[Path]] = {kind: [] for kind in KINDS}

    def emit(self, kind: str, rel: str, save_path: Path, body: Dict[str, Any]) -> None:
        config = {
            "kind": kind,
            "experiment": P.EXPERIMENT,
            "seed": P.SEED,
            "save_path": str(save_path),
            **body,
        }
        text = yaml.safe_dump(config, sort_keys=False)
        path = self.root / kind / rel
        if path.is_file() and path.read_text() != text and not self.overwrite:
            raise SystemExit(f"{path} exists with different content; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.paths[kind].append(path)


def generate(
    method: str,
    dataset_root: "str | Path",
    out: "str | Path",
    *,
    overrides: Optional[Dict[str, Any]] = None,
    evals: Sequence[str] = ("minimal",),
    eval_fr: Optional[Dict[str, Any]] = None,
    target_pool_size: Optional[int] = None,
    corpus_roots: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
) -> Dict[str, List[Path]]:
    """Write one method's AFRBench-basic job configs; return their paths by kind."""
    from afrlib import AFR_REGISTRY
    from afrbench.dataset import BenchDataset

    if method in P.HELD:
        raise SystemExit(f"{method} is implemented in afrlib but is not part of AFRBench-basic")
    if ":" not in method and method not in AFR_REGISTRY:
        raise SystemExit(f"unknown method {method!r}; built-in methods: {P.roster()}")
    cls = AFR_REGISTRY[method]
    roles = P.method_roles(cls)
    dataset_root = Path(dataset_root).absolute()
    out = Path(out).absolute()
    dataset = BenchDataset(dataset_root)
    protectees = dataset.protectee_identities()
    targets = dataset.hq_target_identities()
    assigned = P.protectee_targets(dataset_root)
    missing = sorted(set(protectees) - set(assigned))
    if missing:
        raise SystemExit(f"protectee_groups.json has no target for {missing}")

    name = method_dir(method)
    afr = {"name": method, "set": {**P.resolved_method_defaults(method), **(overrides or {})}}
    common = {
        "afr": afr,
        "dataset": str(dataset_root),
    }
    writer = _Writer(out / "configs", overwrite)

    train_saves: Dict[Optional[str], Path] = {}
    if roles["train"]:
        corpus = P.corpus_block(cls.training_corpus, P.purge_lists(dataset_root))
        if corpus_roots and corpus["name"] in corpus_roots:
            corpus["root"] = corpus_roots[corpus["name"]]
        for target in (targets if roles["target_baked"] else [None]):
            save = out / "train" / name if target is None else out / "train" / name / target
            body = {
                **common,
                "eval_fr": P.DEFAULT_FR,
                "surrogates": list(P.SURROGATES),
                "corpus": corpus,
                "lifecycle": "target_baked" if target else "target_independent",
            }
            if target:
                body["target"] = target
            writer.emit("train", f"{name}.yaml" if target is None else f"{name}/{target}.yaml", save, body)
            train_saves[target] = save

    per_target = roles["target_baked"] or roles["runtime_target"]
    pool_size = target_pool_size or (2000 if method == "divtrackee" else 32)
    runs: Dict[str, str] = {}
    protect_saves: List[str] = []
    for protectee in protectees:
        images = [record.path for record in dataset.eval_images(protectee)]
        album_paths = [record.path for record in dataset.album_images(protectee)]
        album = P.select_fit_album(album_paths, seed=P.SEED, identity=protectee)
        album_block = {"album": album, "album_seed": P.SEED, "album_n": len(album)}
        fit_save = None
        if roles["fit"]:
            fit_save = out / "fit" / name / protectee
            from_run = train_saves.get(None)
            writer.emit("fit", f"{name}/{protectee}.yaml", fit_save, {
                **common,
                "from_run": str(from_run) if from_run else None,
                "eval_fr": P.DEFAULT_FR,
                "surrogates": list(P.SURROGATES),
                "protectee": {"identity": protectee, **album_block},
            })
        target = assigned[protectee] if per_target else None
        parent = fit_save or train_saves.get(target) or train_saves.get(None)
        body: Dict[str, Any] = {
            **common,
            "from_run": str(parent) if parent else None,
            "eval_fr": P.DEFAULT_FR,
            "surrogates": list(P.SURROGATES),
            "protectee": {"identity": protectee, "images": images, **album_block},
            "batch_size": 8,
        }
        if target:
            body["target"] = target
            body["lifecycle"] = "target_baked" if roles["target_baked"] else "runtime_fixed_target"
            save, rel = out / "protect" / name / target / protectee, f"{name}/{target}/{protectee}.yaml"
            runs[f"{name}@{target}"] = str(save.parent)
        else:
            if roles["target_pool"]:
                body["lifecycle"] = "target_pool"
                body["target_pool_size"] = pool_size
            else:
                body["lifecycle"] = "untargeted"
            save, rel = out / "protect" / name / protectee, f"{name}/{protectee}.yaml"
            runs[name] = str(save.parent)
        writer.emit("protect", rel, save, body)
        protect_saves.append(str(save))

    jobs = []
    for job in evals:
        jobs.append((job, *P.eval_job(job, dataset)))
        if eval_fr is not None and job in {*P.PIPELINE_EVALS, "jpeg"}:
            system = P.jpeg_only_system(eval_fr) if job == "jpeg" else P.fr_system(P.PIPELINE_EVALS[job], eval_fr)
            jobs.append((f"{job}@{P.fr_slug(eval_fr)}", system, False, {"mode": "full"}))
    results_root = out / "embeddings" / _dataset_key(dataset_root)
    for job, system, visual_quality, distractors in jobs:
        writer.emit("eval", f"{job}.yaml", out / "eval" / job, {
            "dataset": str(dataset_root),
            "results_root": str(results_root),
            "fr_system": system,
            "runs": dict(sorted(runs.items())),
            "protection": True,
            "visual_quality": visual_quality,
            "gallery": {"distractors": distractors},
            "batch_size": 32,
            "depends_on": protect_saves,
        })
    return writer.paths


def _eval_fingerprint(config) -> str:
    """What the evaluator's unit and quality caches do not key on: distractors and run contents."""
    from afrbench.configs import _runs_digest

    payload = {
        "gallery": config.body.get("gallery"),
        "fr_system": config.body.get("fr_system"),
        "runs": _runs_digest(config),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _reset_stale_eval(config) -> None:
    """Drop cached units and quality rows when the eval's inputs changed since they were written."""
    marker = config.out_dir / ".benchmark_inputs"
    current = _eval_fingerprint(config)
    if marker.is_file() and marker.read_text() != current:
        shutil.rmtree(config.out_dir / "units", ignore_errors=True)
        shutil.rmtree(config.out_dir / "vq_partial", ignore_errors=True)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(current)


def run(paths: Dict[str, List[Path]], device: str = "auto", force: bool = False) -> None:
    """Run the configs kind by kind; stop at the first failed job."""
    from afrbench.configs import JobConfig, is_current

    for kind in KINDS:
        for path in paths[kind]:
            config = JobConfig.load(path)
            if not force and is_current(config):
                print(f"[done] {path}")
                continue
            if kind == "eval":
                _reset_stale_eval(config)
            print(f"[run ] {path}", flush=True)
            command = [sys.executable, "-m", "afrbench", str(path), "--device", device]
            if force:
                command.append("--force")
            if subprocess.run(command).returncode != 0:
                raise SystemExit(f"job failed: {path}\nre-run it alone with: {' '.join(command)}")


def summarize(paths: Dict[str, List[Path]], method: str) -> Optional[Path]:
    """Print the main table of every finished eval and write ``<out>/summary.csv``."""
    import pandas as pd

    from afrbench.configs import JobConfig
    from afrbench.evaluator.report import summarize_experiment

    name = method_dir(method)
    tables = []
    out = None
    for path in paths["eval"]:
        config = JobConfig.load(path)
        metrics_path = config.out_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        out = config.out_dir.parent.parent
        metrics = json.loads(metrics_path.read_text())
        scenarios = metrics.get("scenarios", {})
        print(f"\n{path.stem}  ({metrics.get('fr_system')})")
        print(f"{'scenario':<22}" + "".join(f"{m:>18}" for m in SUMMARY_METRICS))
        rows = [("1B (no protection)", scenarios.get("no_protection", {}).get("1B", {}))]
        rows += [(s, scenarios.get(name, {}).get(s, {})) for s in ("1A", "2A", "2B")]
        for label, values in rows:
            cells = "".join(
                f"{values[m]:>18.4f}" if isinstance(values.get(m), (int, float)) else f"{'-':>18}"
                for m in SUMMARY_METRICS
            )
            print(f"{label:<22}{cells}")
        # Per-target methods have one visual-quality entry per target run (``m@t``).
        for run, quality in sorted((metrics.get("visual_quality") or {}).items()):
            if run != name and not run.startswith(f"{name}@"):
                continue
            label = "visual quality" if run == name else f"visual quality [{run.split('@', 1)[1]}]"
            print(f"{label}: " + ", ".join(
                f"{m}={quality[m]:.4f}" for m in QUALITY_METRICS if isinstance(quality.get(m), (int, float))
            ))
        table_path = config.out_dir / "summary.csv"
        summarize_experiment([config.out_dir], table_path)
        table = pd.read_csv(table_path)
        table.insert(0, "eval", path.stem)
        tables.append(table)
    if not tables or out is None:
        return None
    summary = out / "summary.csv"
    pd.concat(tables, ignore_index=True).to_csv(summary, index=False)
    print(f"\nsummary: {summary}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrbench.benchmark", description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", required=True, help="built-in method name or package.module:Class")
    parser.add_argument("--dataset", required=True, help="prepared AFRBench-basic dataset (python -m afrbench.prepare)")
    parser.add_argument("--out", required=True, help="output folder for configs, runs, and evals")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="method constructor override (repeatable)")
    parser.add_argument("--evals", default="minimal",
                        help=f"comma-separated suite jobs or 'all' (choices: {', '.join(P.EVAL_SUITE)})")
    parser.add_argument("--eval-fr", default=None,
                        help="also score the pipeline evals with this FR model: backbone/loss/dataset or package.module:Class")
    parser.add_argument("--target-pool-size", type=int, default=None,
                        help="target_pool size for a plugin that reads one (default 32)")
    parser.add_argument("--corpus-root", action="append", default=[], metavar="NAME=PATH",
                        help="use a local copy of a training corpus (repeatable)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--configs-only", action="store_true", help="write the configs and stop")
    parser.add_argument("--force", action="store_true", help="re-run jobs even when they are done")
    parser.add_argument("--overwrite", action="store_true", help="replace existing configs that differ")
    args = parser.parse_args(argv)

    evals = list(P.EVAL_SUITE) if args.evals == "all" else [e for e in args.evals.split(",") if e]
    unknown = sorted(set(evals) - set(P.EVAL_SUITE))
    if unknown:
        parser.error(f"unknown eval jobs {unknown}")
    paths = generate(
        args.method, args.dataset, args.out,
        overrides=_key_values(args.set, "--set"),
        evals=evals,
        eval_fr=parse_fr(args.eval_fr) if args.eval_fr else None,
        target_pool_size=args.target_pool_size,
        corpus_roots={k: str(v) for k, v in _key_values(args.corpus_root, "--corpus-root").items()},
        overwrite=args.overwrite,
    )
    print("configs: " + ", ".join(f"{len(paths[k])} {k}" for k in KINDS))
    if args.configs_only:
        return 0
    run(paths, device=args.device, force=args.force)
    summarize(paths, args.method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
