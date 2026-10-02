"""Run one AFRBench job: ``python -m afrbench <config.yaml> [--device DEVICE] [--force]``.

The config's ``kind:`` (train, fit, protect, or eval) selects the job. A job whose
``save_path/config.lock.yaml`` matches the config is skipped unless ``--force``
is given, so re-running a config is safe.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import torch

from afrlib import get_logger
from afrbench.configs import JobConfig, is_current

logger = get_logger(__name__)


def _device(name: str) -> torch.device:
    """Resolve ``--device``: an explicit device, or ``auto`` = cuda -> mps -> cpu."""
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_job(
    config: JobConfig,
    device: torch.device,
    *,
    reuse_visual_quality: bool = False,
    force: bool = False,
) -> None:
    """Dispatch one config to its job body."""
    if config.kind == "train":
        from afrbench.runner import train_job
        from afrbench.threads import pin_cpu_threads

        pin_cpu_threads()
        train_job(config, device, force=force)
    elif config.kind == "fit":
        from afrbench.runner import fit_job

        fit_job(config, device, force=force)
    elif config.kind == "protect":
        from afrbench.runner import protect_job

        protect_job(config, device, force=force)
    elif config.kind == "eval":
        from afrbench.evaluator import eval_job

        eval_job(
            config,
            device,
            reuse_visual_quality=reuse_visual_quality,
            force=force,
        )
    else:  # pragma: no cover - JobConfig.load already validates kind
        raise ValueError(f"unknown kind {config.kind!r}")


def _assert_reusable_visual_quality(config: JobConfig) -> None:
    """Check that existing visual-quality rows cover these runs under this FR system."""
    from afrbench.evaluator.job import _load_visual_quality
    from afrbench.fr_system import FRSystem

    runs = config.body.get("runs") or {}
    if not isinstance(runs, dict) or not runs:
        raise ValueError("eval config has no runs to reuse visual quality from")
    system_name = FRSystem.from_dict(config.body["fr_system"]).name
    _load_visual_quality(config.out_dir, runs, system_name)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m afrbench", description=__doc__)
    parser.add_argument("config", help="path to one job config YAML")
    parser.add_argument(
        "--device", default="auto",
        help="torch device, or 'auto' = cuda -> mps -> cpu (default)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="run even when the config's lock says it is already done",
    )
    parser.add_argument(
        "--reuse-visual-quality", action="store_true",
        help=(
            "for a forced eval rerun, retain current visual-quality results "
            "while recomputing rank-derived metrics"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from afrbench.threads import pin_cpu_threads

    pin_cpu_threads()
    args = parse_args(argv)
    config = JobConfig.load(args.config)
    if args.reuse_visual_quality:
        if config.kind != "eval":
            raise SystemExit("--reuse-visual-quality is valid only for eval jobs")
        if not args.force:
            raise SystemExit("--reuse-visual-quality requires --force")
        if not bool(config.body.get("visual_quality", True)):
            raise SystemExit(
                "--reuse-visual-quality requires visual_quality: true on the eval"
            )
        try:
            _assert_reusable_visual_quality(config)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"cannot reuse visual quality: {exc}") from exc
    if not args.force and is_current(config):
        logger.info("%s already done (lock matches); skipping", config.save_path)
        return 0
    run_job(
        config,
        _device(args.device),
        reuse_visual_quality=args.reuse_visual_quality,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
