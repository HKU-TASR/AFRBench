"""Reporting results by photo label."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from afrbench import groups
from afrbench.dataset import BenchDataset
from tests.test_prepare import _write_json

CAMERAS = {"a1": "front", "a2": "front", "a3": "left", "a4": "left"}


def _dataset(root: Path) -> BenchDataset:
    images = []
    for name, camera in CAMERAS.items():
        images.append({
            "path": f"protectees/A/{name}.png", "identity": "A", "split": "protectee", "role": "query",
            "image_id": name, "size": [8, 8], "sha256": "", "labels": {"camera": camera, "sex": "M"},
        })
    images.append({
        "path": "protectees/B/b1.png", "identity": "B", "split": "protectee", "role": "gallery", "image_id": "b1",
        "size": [8, 8], "sha256": "",
    })
    _write_json(root / "annotations.json", {"schema": "afrbench-dataset-v4", "meta": {}, "images": images})
    return BenchDataset(root)


def _rows() -> pd.DataFrame:
    rows = []
    for scenario, recalls in (("1A", [1.0, 0.5, 0.0, 0.0]), ("2A", [1.0, 1.0, 0.5, 0.5])):
        for (name, _), recall in zip(CAMERAS.items(), recalls):
            rows.append({"run": "fawkes", "fr_system": "irse", "seen": False, "scenario": scenario,
                         "query_id": f"protectees/A/{name}.png", "identity": "A", "recall@25": recall, "AP": recall / 2})
    rows.append({"run": "fawkes", "fr_system": "irse", "seen": False, "scenario": "visual_quality",
                 "query_id": "protectees/A/a1.png", "identity": "A", "psnr": 30.0})
    return pd.DataFrame(rows)


def test_grouped_averages_each_label_value(tmp_path):
    table = groups.grouped(_dataset(tmp_path), _rows(), ["camera"], ["recall@25"])
    one_a = table[(table["scenario"] == "1A")].set_index("camera")
    assert one_a.loc["front", "recall@25"] == 0.75 and one_a.loc["left", "recall@25"] == 0.0
    assert one_a.loc["front", "n"] == 2
    two_a = table[(table["scenario"] == "2A")].set_index("camera")
    assert two_a.loc["front", "recall@25"] == 1.0 and two_a.loc["left", "recall@25"] == 0.5
    assert list(table.columns) == ["run", "fr_system", "seen", "scenario", "camera", "n", "recall@25"]


def test_grouped_drops_metrics_a_selection_does_not_have(tmp_path):
    rows = _rows()
    table = groups.grouped(_dataset(tmp_path), rows[rows["scenario"] == "1A"], ["camera"])
    assert "psnr" not in table.columns and "recall@25" in table.columns


def test_grouped_takes_default_metrics_present_and_crosses_labels(tmp_path):
    table = groups.grouped(_dataset(tmp_path), _rows(), ["camera", "sex"])
    assert {"recall@25", "AP", "psnr"} <= set(table.columns)
    assert set(table["sex"]) == {"M"}
    quality = table[table["scenario"] == "visual_quality"]
    assert quality["psnr"].tolist() == [30.0]


def test_grouped_accepts_record_fields_and_marks_unlabelled_images(tmp_path):
    dataset = _dataset(tmp_path)
    by_identity = groups.grouped(dataset, _rows(), ["identity"], ["recall@25"])
    assert set(by_identity["identity"]) == {"A"}
    assert groups.label_value(dataset.record("protectees/B/b1.png"), "camera") == ""
    assert groups.label_keys(dataset) == ["camera", "identity", "role", "session", "sex"]


def test_grouped_rejects_unknown_labels_and_metrics(tmp_path):
    dataset = _dataset(tmp_path)
    with pytest.raises(ValueError, match="unknown label"):
        groups.grouped(dataset, _rows(), ["pose"])
    with pytest.raises(ValueError, match="no metric"):
        groups.grouped(dataset, _rows(), ["camera"], ["ndcg"])


def test_cli_writes_the_table(tmp_path, capsys):
    root = tmp_path / "data"
    root.mkdir()
    _dataset(root)
    eval_dir = tmp_path / "eval" / "minimal"
    eval_dir.mkdir(parents=True)
    (eval_dir / "per_image.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows().to_dict("records")))
    assert groups.main(["--dataset", str(root), "--eval", str(eval_dir), "--by", "camera", "--scenario", "1A"]) == 0
    written = pd.read_csv(eval_dir / "groups.csv")
    assert set(written["scenario"]) == {"1A"} and set(written["camera"]) == {"front", "left"}
    assert "front" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        groups.read_rows([tmp_path / "nowhere"])
