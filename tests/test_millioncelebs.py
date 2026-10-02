"""Packing a (synthetic) MillionCelebs release into the distractor LMDB."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afrbench import millioncelebs
from afrbench.noise import validate_physical_noise_provenance

LANDMARKS = "28 32 52 32 40 45 30 58 50 58"


def _source(root: Path, identities=("7", "9"), per_identity=3) -> Path:
    src = root / "src"
    src.mkdir()
    rows = ["IMAGE\tX1\tY1\tX2\tY2\tPTX1\tPTY1\tPTX2\tPTY2\tPTX3\tPTY3\tPTX4\tPTY4\tPTX5\tPTY5"]
    rng = np.random.default_rng(0)
    with tarfile.open(src / "MillionCelebs_part0.tar.gz", "w:gz") as tar:
        for identity in identities:
            for index in range(per_identity):
                buffer = io.BytesIO()
                Image.fromarray(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(buffer, format="JPEG")
                info = tarfile.TarInfo(f"images/{identity}/{index}.jpg")
                info.size = len(buffer.getvalue())
                tar.addfile(info, io.BytesIO(buffer.getvalue()))
                rows.append("\t".join([f"{identity}/{index}.jpg", "10", "10", "70", "70", *LANDMARKS.split()]))
    (src / "lmk.tsv").write_text("\n".join(rows) + "\n")
    return src


def test_pack_writes_aligned_samples_and_valid_provenance(tmp_path):
    import lmdb

    src = _source(tmp_path)
    out = tmp_path / "noise.lmdb"
    assert millioncelebs.pack(src, out) == 0
    env = lmdb.open(str(out), readonly=True, lock=False)
    with env.begin() as txn:
        provenance = json.loads(txn.get(b"meta/json"))
        assert int(txn.get(b"meta/count")) == 6
        first = json.loads(txn.get(b"meta/sample/00000000"))
        image = Image.open(io.BytesIO(txn.get(f"{first['key']}/image".encode())))
    assert first["key"] == "7__0.jpg" and first["identity"] == "7"
    assert image.size == (112, 112)
    validate_physical_noise_provenance(provenance)
    assert provenance["n_identities"] == 2
    assert not (out / millioncelebs.PROGRESS).exists()
    with pytest.raises(SystemExit):
        millioncelebs.pack(src, out)


def test_pack_rejects_incomplete_source(tmp_path):
    src = _source(tmp_path)
    with open(src / "lmk.tsv", "a") as handle:
        handle.write("\t".join(["9/99.jpg", "10", "10", "70", "70", *LANDMARKS.split()]) + "\n")
    with pytest.raises(SystemExit):
        millioncelebs.pack(src, tmp_path / "noise.lmdb")
