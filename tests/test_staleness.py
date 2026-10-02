"""Content digests invalidate eval locks and cache keys; ``relock`` refreshes stale lock digests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from afrbench.configs import JobConfig, eval_code_digest, is_current, write_lock
from afrbench.evaluator.gallery import protected_set_name
from afrbench.relock import relock
from afrbench.runs import ProtectedRun


def _write_protect(root: Path, array: np.ndarray) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    output = root / "out.npy"
    np.save(output, array)
    sha = hashlib.sha256(array.tobytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({
        "images": {
            "protectees/A/c.jpg": {
                "output": "out.npy",
                "dtype": "float32",
                "sha256": sha,
            }
        }
    }))
    return root


def _eval_config(tmp_path: Path, run_root: Path) -> JobConfig:
    save = tmp_path / "eval"
    save.mkdir(parents=True, exist_ok=True)
    (save / "metrics.json").write_text("{}\n")
    return JobConfig(
        kind="eval",
        experiment="staleness",
        seed=0,
        save_path=str(save),
        body={"runs": {"m": str(run_root)}},
    )


def _unit_key(run_key: str, digest: str, system_digest: str, name: str, code: str) -> str:
    return hashlib.sha256(
        f"{run_key}\0{digest}\0{system_digest}\0{name}\0{code}".encode()
    ).hexdigest()[:16]


def test_rewriting_protect_run_invalidates_eval_and_keys(tmp_path):
    run = _write_protect(tmp_path / "protect" / "Ada", np.zeros((3, 8, 8), np.float32))
    loaded = ProtectedRun.load(run)
    before = loaded.digest
    source = protected_set_name("m", before)
    config = _eval_config(tmp_path, run)
    write_lock(config)
    assert is_current(config)
    code = eval_code_digest(config) or ""
    unit_before = _unit_key("m", before, "sys", "2A", code)

    mutated = np.ones((3, 8, 8), np.float32)
    np.save(run / "out.npy", mutated)
    sha = hashlib.sha256(mutated.tobytes()).hexdigest()
    (run / "manifest.json").write_text(json.dumps({
        "images": {
            "protectees/A/c.jpg": {
                "output": "out.npy",
                "dtype": "float32",
                "sha256": sha,
            }
        }
    }))
    after = ProtectedRun.load(run)
    assert after.digest != before
    assert protected_set_name("m", after.digest) != source
    assert _unit_key("m", after.digest, "sys", "2A", code) != unit_before
    assert not is_current(config)


def test_unchanged_run_keeps_eval_current(tmp_path):
    run = _write_protect(tmp_path / "protect" / "Ada", np.zeros((3, 8, 8), np.float32))
    first = ProtectedRun.load(run).digest
    config = _eval_config(tmp_path, run)
    write_lock(config)
    assert is_current(config)
    assert ProtectedRun.load(run).digest == first
    assert is_current(config)


def test_relock_restamps_method_digest_only(tmp_path):
    out = tmp_path / "train" / "fawkes"
    out.mkdir(parents=True)
    (out / "weights.pth").write_bytes(b"weight")
    config = JobConfig(
        kind="train",
        experiment="staleness",
        seed=0,
        save_path=str(out),
        body={"afr": {"name": "fawkes", "set": {}}},
    )
    write_lock(config)
    lock = out / "config.lock.yaml"
    recorded = yaml.safe_load(lock.read_text())
    recorded["lock_fingerprint"]["method_code_sha256"] = "stale"
    lock.write_text(yaml.safe_dump(recorded, sort_keys=True))
    assert not is_current(config)
    changed = relock(tmp_path, "train", dry_run=False)
    assert changed
    assert is_current(config)
    restamped = yaml.safe_load(lock.read_text())
    assert restamped["experiment"] == "staleness"
    assert restamped["lock_fingerprint"]["method_code_sha256"] != "stale"


def test_relock_refreshes_digest_even_when_resources_differ(tmp_path):
    out = tmp_path / "protect" / "adv_diffusion"
    out.mkdir(parents=True)
    (out / "outputs").mkdir()
    (out / "outputs" / "a.npy").write_bytes(b"1")
    (out / "manifest.json").write_text(
        __import__("json").dumps({"images": {"a.jpg": {"output": "outputs/a.npy"}}})
    )
    config = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={
            "afr": {"name": "adv_diffusion", "set": {}},
            "protectee": {"images": ["a.jpg"], "album": []},
            "resources": {"gpu_memory_gb": 14.0},
        },
    )
    write_lock(config)
    lock = out / "config.lock.yaml"
    recorded = yaml.safe_load(lock.read_text())
    recorded["lock_fingerprint"]["method_code_sha256"] = "stale-digest"
    lock.write_text(yaml.safe_dump(recorded, sort_keys=True))
    patched = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={
            "afr": {"name": "adv_diffusion", "set": {}},
            "protectee": {"images": ["a.jpg"], "album": []},
            "resources": {"gpu_memory_gb": 18.0},
        },
    )
    assert not is_current(patched)
    changed = relock(tmp_path, "protect", dry_run=False)
    assert changed
    assert is_current(patched)


def test_relock_protect_refreshes_upstream_after_train_relock(tmp_path):
    train_out = tmp_path / "train" / "amt_gan"
    train_out.mkdir(parents=True)
    (train_out / "G.pth").write_bytes(b"weight")
    train = JobConfig(
        kind="train",
        experiment="staleness",
        seed=0,
        save_path=str(train_out),
        body={"afr": {"name": "amt_gan", "set": {}}},
    )
    write_lock(train)
    protect_out = tmp_path / "protect" / "amt_gan"
    protect_out.mkdir(parents=True)
    (protect_out / "outputs").mkdir()
    (protect_out / "outputs" / "a.npy").write_bytes(b"1")
    (protect_out / "manifest.json").write_text(
        __import__("json").dumps({"images": {"a.jpg": {"output": "outputs/a.npy"}}})
    )
    protect = JobConfig(
        kind="protect",
        experiment="staleness",
        seed=0,
        save_path=str(protect_out),
        body={
            "afr": {"name": "amt_gan", "set": {}},
            "from_run": str(train_out),
            "protectee": {"images": ["a.jpg"], "album": []},
        },
    )
    write_lock(protect)
    assert is_current(protect)
    train_lock = train_out / "config.lock.yaml"
    recorded = yaml.safe_load(train_lock.read_text())
    recorded["lock_fingerprint"]["note"] = "restamp-changes-lock-hash"
    train_lock.write_text(yaml.safe_dump(recorded, sort_keys=True))
    assert not is_current(protect)
    assert relock(tmp_path / "protect", "protect", dry_run=False)
    assert is_current(protect)


def test_relock_train_refreshes_dataset_digest(tmp_path):
    dataset = tmp_path / "basic"
    dataset.mkdir()
    (dataset / "dataset.json").write_text('{"name":"old"}\n')
    out = tmp_path / "train" / "fawkes"
    out.mkdir(parents=True)
    (out / "weights.pth").write_bytes(b"weight")
    config = JobConfig(
        kind="train",
        experiment="staleness",
        seed=0,
        save_path=str(out),
        body={"afr": {"name": "fawkes", "set": {}}, "dataset": str(dataset)},
    )
    write_lock(config)
    lock = out / "config.lock.yaml"
    recorded = yaml.safe_load(lock.read_text())
    recorded["lock_fingerprint"]["dataset_sha256"] = "stale-dataset"
    recorded["lock_fingerprint"]["album_sha256"] = "keep-album"
    lock.write_text(yaml.safe_dump(recorded, sort_keys=True))
    (dataset / "dataset.json").write_text('{"name":"getty-v6"}\n')
    assert not is_current(config)
    changed = relock(tmp_path, "train", dry_run=False)
    assert changed
    assert is_current(config)
    restamped = yaml.safe_load(lock.read_text())
    assert restamped["lock_fingerprint"]["dataset_sha256"] != "stale-dataset"
    assert restamped["lock_fingerprint"]["album_sha256"] == "keep-album"


def test_relock_fit_refreshes_dataset_digest(tmp_path):
    dataset = tmp_path / "basic"
    dataset.mkdir()
    (dataset / "dataset.json").write_text('{"name":"v6"}\n')
    out = tmp_path / "fit" / "fawkes"
    out.mkdir(parents=True)
    (out / "weights.pth").write_bytes(b"weight")
    config = JobConfig(
        kind="fit",
        experiment="staleness",
        seed=0,
        save_path=str(out),
        body={
            "afr": {"name": "fawkes", "set": {}},
            "dataset": str(dataset),
            "protectee": {"identity": "Ada", "album": []},
        },
    )
    write_lock(config)
    lock = out / "config.lock.yaml"
    recorded = yaml.safe_load(lock.read_text())
    recorded["lock_fingerprint"]["dataset_sha256"] = "stale-dataset"
    lock.write_text(yaml.safe_dump(recorded, sort_keys=True))
    (dataset / "dataset.json").write_text('{"name":"getty-v6-idmatch"}\n')
    changed = relock(tmp_path, "fit", dry_run=False)
    restamped = yaml.safe_load(lock.read_text())
    assert restamped["lock_fingerprint"]["dataset_sha256"] != "stale-dataset"
    assert changed
