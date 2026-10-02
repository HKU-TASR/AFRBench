"""Re-stamp the lock fingerprints of finished jobs after code or dataset-metadata changes."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

from afrbench.configs import (
    LOCK_FILENAME,
    JobConfig,
    _dataset_digest,
    _method_code_digest,
    _outputs_present,
    _upstream_digest,
)


def _kind_set(kinds: str) -> set[str]:
    return {part.strip() for part in kinds.split(",") if part.strip()}


def iter_locks(results_root: Path) -> Iterable[Path]:
    """Every ``config.lock.yaml`` under ``results_root``, in sorted order."""
    yield from sorted(Path(results_root).rglob(LOCK_FILENAME))


def relock_one(lock_path: Path, kinds: Sequence[str], *, dry_run: bool) -> Optional[str]:
    """Re-stamp one lock and return a summary line, or ``None`` if it is skipped or current."""
    recorded = yaml.safe_load(lock_path.read_text())
    if not isinstance(recorded, dict):
        return None
    fingerprint = recorded.get("lock_fingerprint")
    envelope = {key: value for key, value in recorded.items() if key != "lock_fingerprint"}
    try:
        config = JobConfig.from_dict(dict(envelope))
    except (TypeError, ValueError):
        return None
    if config.to_dict() != envelope:
        return None
    if config.kind not in kinds:
        return None
    if not _outputs_present(config):
        return None
    stamp = dict(fingerprint) if isinstance(fingerprint, dict) else {}
    updates: Dict[str, Any] = {}
    new_method = _method_code_digest(config)
    if stamp.get("method_code_sha256") != new_method:
        updates["method_code_sha256"] = new_method
    new_upstream = _upstream_digest(config)
    if stamp.get("from_run_lock") != new_upstream:
        updates["from_run_lock"] = new_upstream
    if config.kind in {"train", "fit", "protect"}:
        new_dataset = _dataset_digest(config)
        if stamp.get("dataset_sha256") != new_dataset:
            updates["dataset_sha256"] = new_dataset
    if not updates:
        return None
    parts = [
        f"{key} {stamp.get(key)!s} -> {value!s}"
        for key, value in updates.items()
    ]
    line = f"{lock_path}: {', '.join(parts)}"
    if dry_run:
        return line
    payload = dict(recorded)
    # A restamp certifies outputs without re-running the job; keep what it replaced.
    history = list(stamp.get("relocked") or [])
    history.append({
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "replaced": {key: stamp.get(key) for key in updates},
    })
    stamp.update(updates)
    stamp["relocked"] = history
    payload["lock_fingerprint"] = stamp
    temporary = lock_path.with_name(f".{LOCK_FILENAME}.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=True))
    os.replace(temporary, lock_path)
    return line


def relock(results_root: "str | Path", kinds: str, *, dry_run: bool = False) -> List[str]:
    """Re-stamp every eligible lock under ``results_root``; ``kinds`` is comma-separated."""
    wanted = _kind_set(kinds)
    changed: List[str] = []
    for lock_path in iter_locks(Path(results_root)):
        line = relock_one(lock_path, wanted, dry_run=dry_run)
        if line:
            changed.append(line)
    return changed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--kinds", default="train,fit,protect")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    changed = relock(args.results_root, args.kinds, dry_run=args.dry_run)
    for line in changed:
        print(line)
    print(f"{'would restamp' if args.dry_run else 'restamped'} {len(changed)} lock(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
