"""Lock currentness: required outputs, and which config changes invalidate a lock."""
from __future__ import annotations

import json

from afrbench.configs import JobConfig, is_current, write_lock


def test_protect_currentness_requires_outputs(tmp_path):
    out = tmp_path / "protect"
    out.mkdir()
    config = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={
            "afr": {"name": "lowkey", "set": {}},
            "protectee": {"images": ["a.jpg", "b.jpg"], "album": []},
        },
    )
    write_lock(config)
    assert is_current(config) is False
    manifest = {
        "images": {
            "a.jpg": {"output": "outputs/a.npy"},
            "b.jpg": {"output": "outputs/b.npy"},
        }
    }
    (out / "outputs").mkdir()
    (out / "outputs" / "a.npy").write_bytes(b"1")
    (out / "outputs" / "b.npy").write_bytes(b"2")
    (out / "manifest.json").write_text(json.dumps(manifest))
    write_lock(config)
    assert is_current(config) is True
    (out / "outputs" / "b.npy").unlink()
    assert is_current(config) is False


def test_resource_only_yaml_change_keeps_protect_current(tmp_path):
    out = tmp_path / "protect"
    out.mkdir()
    body = {
        "afr": {"name": "lowkey", "set": {}},
        "protectee": {"images": ["a.jpg"], "album": []},
        "resources": {"gpu_memory_gb": 12.0},
    }
    config = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body=dict(body),
    )
    (out / "outputs").mkdir()
    (out / "outputs" / "a.npy").write_bytes(b"1")
    (out / "manifest.json").write_text(json.dumps({
        "images": {"a.jpg": {"output": "outputs/a.npy"}},
    }))
    write_lock(config)
    assert is_current(config) is True
    patched = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={**body, "resources": {"gpu_memory_gb": 7.0}},
    )
    assert is_current(patched) is True
    changed = JobConfig(
        kind="protect",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={**body, "afr": {"name": "lowkey", "set": {"eps": 16}}},
    )
    assert is_current(changed) is False


def test_eval_currentness_requires_metrics(tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    config = JobConfig(
        kind="eval",
        experiment="basic",
        seed=0,
        save_path=str(out),
        body={"dataset": str(tmp_path), "fr_system": {}},
    )
    write_lock(config)
    assert is_current(config) is False
    (out / "metrics.json").write_text("{}")
    write_lock(config)
    assert is_current(config) is True
