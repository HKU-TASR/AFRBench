"""Assembling AFRBench-basic from shipped metadata, user images, and a noise pack."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from afrbench.prepare import prepare
from afrlib.data.lmdb_io import CorpusWriter

SIZE = [64, 48]  # [W, H]
GEOMETRY = {
    "face_box": [16.0, 8.0, 48.0, 40.0],
    "landmarks": [[26, 20], [38, 20], [32, 27], [27, 33], [37, 33]],
    "face_square": [16.0, 8.0, 32.0],
    "original_box": [0, -8, 64],
    "padding": [0, 8, 0, 8],
    "valid_rect": [0, 8, 64, 48],
    "padding_fraction": 0.25,
    "original_ratio": 2.0,
}


def _jpeg(path: Path, size=SIZE, orientation: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", tuple(size), (120, 90, 60))
    exif = Image.Exif()
    if orientation != 1:
        exif[0x0112] = orientation
    image.save(path, format="JPEG", exif=exif)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _metadata(root: Path) -> Path:
    meta = root / "meta"
    images = []
    for role, asset in (("album", "1001"), ("query", "1002"), ("gallery", "1003")):
        images.append({
            "path": f"protectees/A/A__gettyimages-{asset}_x.jpg", "identity": "A",
            "split": "protectee", "role": role, "image_id": f"A__{asset}", "size": SIZE,
            "sha256": "", "getty_asset_id": asset, **GEOMETRY,
        })
    images.append({
        "path": "targets/hq/T/t.png", "identity": "T", "split": "target", "role": "target",
        "image_id": "t.png", "size": SIZE, "sha256": "",
        "face_box": GEOMETRY["face_box"], "landmarks": GEOMETRY["landmarks"],
        "original_box": GEOMETRY["original_box"],
    })
    _write_json(meta / "annotations.json", {"schema": "afrbench-dataset-v4", "meta": {}, "images": images})
    _write_json(meta / "identities.json", {"identities": {}})
    _write_json(meta / "dataset.json", {"name": "tiny"})
    _write_json(meta / "protectee_groups.json", {"membership": {"A": "T"}})
    _write_json(meta / "purge" / "millioncelebs.json", {
        "schema": "afrbench-purge-v1", "corpus": "millioncelebs", "key": "identity", "entries": ["A"],
    })
    _write_json(meta / "targets" / "pool.json", {
        "hq": [{"identity": "T", "keys": ["T__0.jpg"]}],
        "identities": [{"identity": "T", "keys": ["T__0.jpg"]}],
    })
    target = meta / "targets" / "hq" / "T" / "t.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tuple(SIZE), (10, 20, 30)).save(target)
    return meta


def _noise(root: Path) -> Path:
    path = root / "noise.lmdb"
    buffer = io.BytesIO()
    Image.new("RGB", (112, 112), (1, 2, 3)).save(buffer, format="JPEG")
    provenance = {
        "lock_independent": True, "aligned": True, "alignment_id": "test", "full_source": True,
        "identity_exclusions": [], "expected_source_samples": 2, "expected_source_identities": 2,
        "n_samples": 2, "n_identities": 2,
    }
    with CorpusWriter(path, map_size=1 << 24, provenance=provenance) as writer:
        writer.add("T__0.jpg", {"image": buffer.getvalue()}, identity="T")
        writer.add("B__0.jpg", {"image": buffer.getvalue()}, identity="B")
    return path


def test_prepare_links_images_and_validates(tmp_path):
    meta, noise = _metadata(tmp_path), _noise(tmp_path)
    for asset in ("1001", "1002", "1003"):
        _jpeg(tmp_path / "getty" / "any" / f"gettyimages-{asset}-2048x2048.jpg")
    out = tmp_path / "basic"
    assert prepare(tmp_path / "getty", noise, out, metadata=meta) == 0
    annotations = json.loads((out / "annotations.json").read_text())
    assert all(len(row["sha256"]) == 64 for row in annotations["images"])
    assert (out / "protectees" / "A" / "A__gettyimages-1001_x.jpg").is_symlink()
    assert (out / "targets" / "hq" / "T" / "t.png").is_file()

    from afrbench.dataset import BenchDataset

    dataset = BenchDataset(out)
    record = dataset.record("protectees/A/A__gettyimages-1002_x.jpg")
    assert tuple(dataset.original(record).shape) == (3, 64, 64)


@pytest.mark.parametrize("problem", ["missing", "size", "rotated"])
def test_prepare_reports_bad_inputs(tmp_path, problem):
    meta, noise = _metadata(tmp_path), _noise(tmp_path)
    for asset in ("1001", "1002", "1003"):
        if problem == "missing" and asset == "1003":
            continue
        size = [32, 32] if problem == "size" and asset == "1002" else SIZE
        orientation = 6 if problem == "rotated" and asset == "1001" else 1
        _jpeg(tmp_path / "getty" / f"gettyimages-{asset}-2048x2048.jpg", size, orientation)
    out = tmp_path / "basic"
    assert prepare(tmp_path / "getty", noise, out, metadata=meta) == 1
    report = (out / "prepare_report.txt").read_text()
    assert {"missing": "missing", "size": "size", "rotated": "EXIF"}[problem] in report
