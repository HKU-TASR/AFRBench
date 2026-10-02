"""Assemble the AFRBench-basic dataset from the shipped metadata and your own image copies.

``python -m afrbench.prepare --getty DIR --noise PACK --out DIR`` links each Getty
image (matched by the asset id in its file name) to its recorded path, links the
MillionCelebs pack, copies the metadata and HQ targets, and checks the result.
Images are never re-encoded: each must decode to the recorded size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

METADATA = Path(__file__).resolve().parents[1] / "data" / "afrbench-basic"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXIF_ORIENTATION = 0x0112


def index_by_asset_id(root: Path, asset_ids: set) -> Dict[str, List[Path]]:
    """Map each asset id to the files whose name contains it as a whole digit run."""
    found: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        for run in set(re.findall(r"\d+", path.name)) & asset_ids:
            found.setdefault(run, []).append(path)
    return found


def check_image(path: Path, size: Sequence[int]) -> Optional[str]:
    """Why ``path`` cannot stand in for a record of ``size`` = [W, H], or None."""
    with Image.open(path) as image:
        if image.getexif().get(EXIF_ORIENTATION, 1) != 1:
            return "EXIF-rotated"
        if list(image.size) != list(size[:2]):
            return f"size {list(image.size)} != recorded {list(size[:2])}"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link(source: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if copy:
        shutil.copyfile(source, dest)
    else:
        os.symlink(source.resolve(), dest)


def check_pack(dataset) -> Optional[str]:
    """Validate the noise pack and confirm every target-pool key is present."""
    from afrlib.data.lmdb_io import lmdb_env

    pool = dataset.target_pool
    keys = [key for entry in pool.get("hq", []) + pool.get("identities", []) for key in entry["keys"]]
    with lmdb_env(dataset.noise_lmdb).begin() as txn:
        missing = [key for key in keys if txn.get(f"{key}/image".encode("utf-8")) is None]
    if missing:
        return f"{len(missing)} target-pool keys are missing from the noise pack, e.g. {missing[:3]}"
    return None


def prepare(getty: Path, noise: Path, out: Path, metadata: Path = METADATA, copy: bool = False) -> int:
    if not (metadata / "annotations.json").is_file():
        raise SystemExit(
            f"no AFRBench-basic metadata at {metadata}; run from a clone of the repository "
            "(pip install -e .) or pass --metadata <clone>/data/afrbench-basic"
        )
    annotations = json.loads((metadata / "annotations.json").read_text())
    records = annotations["images"]
    protectees = [r for r in records if r["split"] == "protectee"]
    found = index_by_asset_id(getty, {r["getty_asset_id"] for r in protectees})

    problems: List[str] = []
    for record in protectees:
        candidates = found.get(record["getty_asset_id"], [])
        if not candidates:
            problems.append(f"missing  {record['getty_asset_id']}  ({record['identity']})")
        elif len(candidates) > 1:
            problems.append(f"ambiguous {record['getty_asset_id']}: {[str(p) for p in candidates]}")
        else:
            reason = check_image(candidates[0], record["size"])
            if reason:
                problems.append(f"{reason}  {candidates[0]}")
    if not (noise / "data.mdb").is_file():
        problems.append(f"no LMDB at {noise} (run: python -m afrbench.millioncelebs pack)")
    out.mkdir(parents=True, exist_ok=True)
    report = out / "prepare_report.txt"
    if problems:
        report.write_text("\n".join(problems) + "\n")
        print(f"{len(problems)} problems; first ones:", *problems[:10], f"full list: {report}", sep="\n  ")
        return 1

    for name in ("identities.json", "dataset.json", "protectee_groups.json"):
        shutil.copyfile(metadata / name, out / name)
    for sub in ("purge", "targets"):
        shutil.copytree(metadata / sub, out / sub, dirs_exist_ok=True)
    for record in protectees:
        source = found[record["getty_asset_id"]][0]
        _link(source, out / record["path"], copy)
        record["sha256"] = _sha256(source)
    for record in records:
        if record["split"] == "target":
            record["sha256"] = _sha256(out / record["path"])
    (out / "annotations.json").write_text(json.dumps(annotations, indent=1) + "\n")
    _link(noise, out / "noise.lmdb", copy=False)

    from afrbench.dataset import BenchDataset

    dataset = BenchDataset(out)
    problem = check_pack(dataset)
    if problem:
        report.write_text(problem + "\n")
        print(problem)
        return 1
    if report.exists():
        report.unlink()
    print(f"AFRBench-basic ready at {out}: {len(protectees)} protectee images, "
          f"{len(dataset.hq_target_identities())} HQ targets, {len(dataset.target_identities())} pool targets")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrbench.prepare", description=__doc__.split("\n\n")[0])
    parser.add_argument("--getty", required=True, type=Path, help="folder with your Getty images (any layout)")
    parser.add_argument("--noise", required=True, type=Path, help="packed MillionCelebs LMDB folder")
    parser.add_argument("--out", required=True, type=Path, help="where to assemble the dataset")
    parser.add_argument("--metadata", type=Path, default=METADATA, help="shipped metadata (data/afrbench-basic)")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    args = parser.parse_args(argv)
    return prepare(args.getty, args.noise, args.out, args.metadata, args.copy)


if __name__ == "__main__":
    sys.exit(main())
