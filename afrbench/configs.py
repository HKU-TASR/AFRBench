"""Job configs (train, fit, protect, eval) and the lock file that makes a job idempotent."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: The four job kinds, in the order a scheduler runs them.
KINDS = ("train", "fit", "protect", "eval")

LOCK_FILENAME = "config.lock.yaml"


@dataclasses.dataclass(frozen=True)
class JobConfig:
    """One job: the shared envelope plus a free-form, kind-specific ``body``."""

    kind: str
    experiment: str
    seed: int
    save_path: str
    body: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JobConfig":
        raw = dict(raw)
        kind = raw.pop("kind", None)
        if kind not in KINDS:
            raise ValueError(f"config kind must be one of {KINDS}; got {kind!r}")
        experiment = raw.pop("experiment", None)
        save_path = raw.pop("save_path", None)
        if not experiment or not save_path:
            raise ValueError("config needs both 'experiment' and 'save_path'")
        return cls(
            kind=kind,
            experiment=str(experiment),
            seed=int(raw.pop("seed", 0)),
            save_path=str(save_path),
            body=raw,
        )

    @classmethod
    def load(cls, path: "str | Path") -> "JobConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "experiment": self.experiment,
            "seed": self.seed,
            "save_path": self.save_path,
            **self.body,
        }

    @property
    def out_dir(self) -> Path:
        return Path(self.save_path)

    @property
    def afr(self) -> str:
        """The method key (train/fit/protect)."""
        afr = self.body.get("afr")
        if isinstance(afr, dict):
            return str(afr.get("name"))
        return str(afr)

    @property
    def afr_overrides(self) -> Dict[str, Any]:
        afr = self.body.get("afr")
        if isinstance(afr, dict):
            return dict(afr.get("set", {}))
        return {}

    @property
    def from_run(self) -> Optional[str]:
        value = self.body.get("from_run")
        return None if value in (None, "null", "") else str(value)

    @property
    def surrogates(self) -> List[Dict[str, Any]]:
        return [dict(s) for s in self.body.get("surrogates", [])]


# Idempotence


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _upstream_digest(config: JobConfig) -> Optional[str]:
    from_run = config.from_run
    if not from_run:
        return None
    lock = Path(from_run) / LOCK_FILENAME
    if not lock.is_file():
        return None
    return _sha256_text(lock.read_text())


def _paths_digest(paths: Optional[List[str]]) -> Optional[str]:
    if not paths:
        return None
    return _sha256_text("\n".join(sorted(str(p) for p in paths)))


def _hash_files(paths: List[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _eval_code_digest(config: JobConfig) -> Optional[str]:
    """Digest of the code that can change eval embeddings, scenarios or ranking metrics, which
    excludes ``report.py`` and ``metrics/quality.py``.
    """
    if config.kind != "eval":
        return None
    package = Path(__file__).resolve().parent
    skip = {"report.py", "quality.py"}
    paths = []
    for directory in ("evaluator", "fr_system"):
        paths.extend(
            path
            for path in (package / directory).rglob("*.py")
            if path.name not in skip
        )
    paths.extend(package / name for name in ("dataset.py", "runs.py"))
    return _hash_files(paths, package)


def eval_code_digest(config: JobConfig) -> Optional[str]:
    """Public name for the eval-code fingerprint (unit cache / lock)."""
    return _eval_code_digest(config)


#: Shared modules that affect the protected pixels of every method.
_SHARED_METHOD_CODE = (
    "afrlib/contracts.py",
    "afrlib/afr/base.py",
    "afrlib/utils/face_crop.py",
    "afrlib/utils/seed.py",
)


def _method_code_digest(config: JobConfig) -> Optional[str]:
    if config.kind not in {"train", "fit", "protect"}:
        return None
    package = Path(__file__).resolve().parent
    repo = package.parent
    paths = [package / "runner.py"]
    paths.extend(
        repo / rel for rel in _SHARED_METHOD_CODE if (repo / rel).is_file()
    )
    fr_backends = repo / "afrlib" / "fr"
    if fr_backends.is_dir():
        paths.extend(fr_backends.rglob("*.py"))
    method = repo / "afrlib" / "afr" / config.afr
    if method.is_dir():
        paths.extend(path for path in method.rglob("*.py"))
    digest = _hash_files(sorted(set(paths)), repo)
    targets = [config.afr, *(str(spec.get("backend", "")) for spec in config.surrogates)]
    plugins = sorted({target for target in targets if ":" in target})
    if not plugins:
        return digest
    parts = [digest] + [f"{target}={_plugin_digest(target)}" for target in plugins]
    return _sha256_text("\n".join(parts))


def _plugin_digest(target: str) -> str:
    """Digest of the package (or module) that defines a ``"package.module:Class"`` target."""
    import importlib.util

    top = target.split(":", 1)[0].split(".", 1)[0]
    spec = importlib.util.find_spec(top)
    if spec is None or spec.origin is None:
        return "unresolved"
    if spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        return _hash_files(list(root.rglob("*.py")), root.parent)
    origin = Path(spec.origin)
    return _hash_files([origin], origin.parent)


def _runs_digest(config: JobConfig) -> Optional[Dict[str, Optional[str]]]:
    """Per-run content digests of an eval config's ``runs:`` tree."""
    if config.kind != "eval":
        return None
    from afrbench.runs import ProtectedRun

    runs = config.body.get("runs") or {}
    out: Dict[str, Optional[str]] = {}
    for run_key, root in sorted(runs.items()):
        path = Path(root)
        if not path.exists():
            out[str(run_key)] = None
        else:
            out[str(run_key)] = ProtectedRun.load(path).digest
    return out


def _dataset_digest(config: JobConfig) -> Optional[str]:
    root = config.body.get("dataset")
    if not root:
        return None
    dataset = Path(root)
    paths = [
        dataset / name
        for name in (
            "dataset.json",
            "identities.json",
            "annotations.json",
            "selection.json",
            "targets/pool.json",
            "purge/millioncelebs.json",
            "noise/filter.json",
        )
        if (dataset / name).is_file()
    ]
    assets = dataset / "protectees"
    if assets.is_symlink() or assets.is_dir():
        manifest = (assets.parent / "manifest.json")
        if not manifest.is_file():
            manifest = dataset.parent.parent / "assets" / "protectees" / "manifest.json"
        if manifest.is_file():
            paths.append(manifest)
    mc = dataset.parent.parent / "assets" / "millioncelebs.manifest.json"
    if mc.is_file():
        paths.append(mc)
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _output_manifest_digest(config: JobConfig) -> Optional[str]:
    if config.kind != "protect":
        return None
    manifest = config.out_dir / "manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _identity_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Config fields that define the result (everything except ``resources``)."""
    payload = dict(raw)
    payload.pop("resources", None)
    return payload


def lock_record(config: JobConfig, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Config plus input/upstream fingerprints used for provenance-safe resume."""
    payload = config.to_dict()
    protectee = config.body.get("protectee") or {}
    fingerprint: Dict[str, Any] = {
        "config_sha256": _sha256_text(yaml.safe_dump(config.to_dict(), sort_keys=True)),
        "from_run_lock": _upstream_digest(config),
        "method_defaults": (
            (config.body.get("afr") or {}).get("set")
            if isinstance(config.body.get("afr"), dict)
            else None
        ),
        "surrogates": config.surrogates,
        "eval_fr": config.body.get("eval_fr"),
        "album_sha256": _paths_digest(protectee.get("album")),
        "images_sha256": _paths_digest(protectee.get("images")),
        "eval_code_sha256": _eval_code_digest(config),
        "method_code_sha256": _method_code_digest(config),
        "dataset_sha256": _dataset_digest(config),
        "output_manifest_sha256": _output_manifest_digest(config),
        "runs_sha256": _runs_digest(config),
    }
    if extra:
        fingerprint.update(json.loads(json.dumps(extra, default=str)))
    payload["lock_fingerprint"] = fingerprint
    return payload


def is_current(config: JobConfig, *, include_runtime: bool = True) -> bool:
    """True when the lock matches this config and its upstream fingerprint; a lock without a
    fingerprint is never current.
    """
    lock = config.out_dir / LOCK_FILENAME
    if not lock.is_file():
        return False
    try:
        recorded = yaml.safe_load(lock.read_text())
    except yaml.YAMLError:
        return False
    if not isinstance(recorded, dict):
        return False
    fingerprint = recorded.pop("lock_fingerprint", None)
    if not isinstance(fingerprint, dict) or not fingerprint:
        return False
    if _identity_payload(recorded) != _identity_payload(config.to_dict()):
        return False
    expected = _upstream_digest(config)
    if fingerprint.get("from_run_lock") != expected:
        return False
    if include_runtime and config.kind == "eval":
        if fingerprint.get("eval_code_sha256") != _eval_code_digest(config):
            return False
        if not (config.out_dir / "metrics.json").is_file():
            return False
        recorded_runs = fingerprint.get("runs_sha256")
        if recorded_runs and recorded_runs != _runs_digest(config):
            return False
    if include_runtime and config.kind in {"train", "fit", "protect"}:
        if fingerprint.get("method_code_sha256") != _method_code_digest(config):
            return False
        if not _outputs_present(config):
            return False
        if config.kind == "protect":
            if fingerprint.get("output_manifest_sha256") != _output_manifest_digest(config):
                return False
    if include_runtime:
        if fingerprint.get("dataset_sha256") != _dataset_digest(config):
            return False
    return True


def _outputs_present(config: JobConfig) -> bool:
    out = config.out_dir
    if config.kind == "protect":
        manifest = out / "manifest.json"
        if not manifest.is_file():
            return False
        try:
            payload = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            return False
        images = payload.get("images") or {}
        expected = config.body.get("protectee", {}).get("images") or []
        if expected and len(images) != len(expected):
            return False
        return all((out / row["output"]).is_file() for row in images.values() if row.get("output"))
    if config.kind in {"train", "fit"}:
        return any(path.is_file() for path in out.rglob("*") if path.name != LOCK_FILENAME)
    return True


def write_lock(config: JobConfig, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Atomically write the config as its own lock; call as a job's last step."""
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / LOCK_FILENAME
    temporary = out / f".{LOCK_FILENAME}.tmp"
    temporary.write_text(yaml.safe_dump(lock_record(config, extra), sort_keys=True))
    os.replace(temporary, path)
    return path
