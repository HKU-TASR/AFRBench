"""Download MillionCelebs from its authors and pack it as the AFRBench distractor LMDB.

``python -m afrbench.millioncelebs download --out DIR`` fetches the release from the
official OneDrive share (https://buptzyb.github.io/MillionCelebs/). ``python -m
afrbench.millioncelebs pack --src DIR --out DIR/noise.lmdb`` aligns every face to
the ArcFace 112x112 template with the landmarks in ``lmk.tsv`` and stores it as
JPEG (quality 95) under the key ``<identity>__<filename>``. Both are resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np

SHARE_URL = (
    "https://bupteducn-my.sharepoint.com/:f:/g/personal/beijingyoudiandaxue_bupt_edu_cn/"
    "EjZWjPJjB1lKl6cRKcHBwYwBAqrARC5BbnwDZV1b10MtIg?e=ShSbjh"
)
BASE_URL = "https://bupteducn-my.sharepoint.com/personal/beijingyoudiandaxue_bupt_edu_cn"
FOLDER = "/personal/beijingyoudiandaxue_bupt_edu_cn/Documents/MillionCelebs"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
#: The release files and their sizes in bytes.
FILES = {
    **{f"MillionCelebs_part{i}.tar.gz": size for i, size in enumerate((
        30912752467, 31068697512, 30863546856, 31105716915, 30518883756,
        30573443335, 30879514138, 31040583156, 31000297197, 30443644878,
        30915030067, 30607618244, 30970899691, 30655981872, 12019114616,
    ))},
    "lmk.tsv": 1408967302,
}

PACK_SCHEMA = "afrbench-noise-pack-v2"
ALIGNED_SIZE = 112
JPEG_QUALITY = 95
PROGRESS = "pack_progress.json"


# Download


def _file_url(name: str) -> str:
    return f"{BASE_URL}/_layouts/15/download.aspx?SourceUrl={quote(f'{FOLDER}/{name}', safe='')}"


def _session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.get(SHARE_URL, timeout=60).raise_for_status()
    if "FedAuth" not in session.cookies.get_dict():
        raise RuntimeError("the MillionCelebs share did not grant a session; open it in a browser to check")
    return session


def _fetch(session, name: str, dest: Path, size: int) -> None:
    """Stream one file with HTTP range resume."""
    done = dest.stat().st_size if dest.exists() else 0
    if done == size:
        return
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(_file_url(name), headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        mode = "ab" if done and response.status_code == 206 else "wb"
        with open(dest, mode) as handle:
            for chunk in response.iter_content(chunk_size=1 << 22):
                handle.write(chunk)


def download(out: Path, names: Optional[Sequence[str]] = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    names = list(names or FILES)
    session = _session()
    if shutil.which("aria2c"):
        cookie = "; ".join(f"{k}={v}" for k, v in session.cookies.get_dict().items())
        lines = []
        for name in names:
            lines += [_file_url(name), f"  out={name}", "  continue=true", "  split=8",
                      "  max-connection-per-server=8", f"  header=User-Agent: {USER_AGENT}",
                      f"  header=Cookie: {cookie}"]
        spec = out / ".aria2.txt"
        spec.write_text("\n".join(lines) + "\n")
        try:
            subprocess.run(["aria2c", "-d", str(out), "-i", str(spec)], check=True)
        finally:
            spec.unlink()
    else:
        for name in names:
            print(f"[download] {name}", flush=True)
            _fetch(session, name, out / name, FILES[name])
    wrong = [n for n in names if (out / n).stat().st_size != FILES[n]]
    if wrong:
        print(f"incomplete or unexpected sizes: {wrong}; re-run to resume")
        return 1
    print(f"MillionCelebs downloaded to {out}")
    return 0


# Pack


class Landmarks:
    """Per-identity access to ``lmk.tsv`` without loading its 22.8M rows."""

    def __init__(self, path: Path):
        self.path = path
        self.index: Dict[str, Tuple[int, int, int]] = {}
        with open(path, "rb") as handle:
            handle.readline()
            offset = handle.tell()
            current, start, rows = None, offset, 0
            for line in handle:
                identity = line.split(b"/", 1)[0].decode()
                if identity != current:
                    if current is not None:
                        self._close(current, start, offset - start, rows)
                    current, start, rows = identity, offset, 0
                rows += 1
                offset += len(line)
            if current is not None:
                self._close(current, start, offset - start, rows)
        self._cached: Tuple[Optional[str], Dict[str, np.ndarray]] = (None, {})

    def _close(self, identity: str, start: int, length: int, rows: int) -> None:
        if identity in self.index:
            raise ValueError(f"lmk.tsv rows of identity {identity} are not contiguous")
        self.index[identity] = (start, length, rows)

    @property
    def n_samples(self) -> int:
        return sum(rows for _, _, rows in self.index.values())

    def get(self, identity: str, filename: str) -> np.ndarray:
        if self._cached[0] != identity:
            start, length, _ = self.index[identity]
            with open(self.path, "rb") as handle:
                handle.seek(start)
                block = handle.read(length).decode("utf-8")
            rows = {}
            for line in block.splitlines():
                parts = line.split("\t")
                if len(parts) < 15:
                    raise ValueError(f"malformed lmk.tsv row: {line[:80]!r}")
                rows[parts[0].partition("/")[2]] = np.array(
                    [float(parts[5 + j]) for j in range(10)], dtype=np.float32
                ).reshape(5, 2)
            self._cached = (identity, rows)
        return self._cached[1][filename]


def align_arcface(image: np.ndarray, landmarks: np.ndarray, size: int = ALIGNED_SIZE) -> np.ndarray:
    """Warp a loose face crop to the ArcFace template with a similarity transform."""
    import torch
    from PIL import Image
    from frbench import arcface_template, estimate_similarity_transform

    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError(f"invalid ArcFace landmarks with shape {landmarks.shape}")
    matrix = estimate_similarity_transform(
        torch.from_numpy(landmarks).unsqueeze(0), arcface_template((size, size))
    )[0].cpu().numpy()
    # PIL warps backwards, so it takes the inverse (destination -> source) map.
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    det = a * d - b * c
    if not np.isfinite(matrix).all() or not np.isfinite(det) or abs(float(det)) < 1e-8:
        raise ValueError("degenerate ArcFace similarity transform")
    inverse = np.array([
        [d / det, -b / det, (-d * tx + b * ty) / det],
        [-c / det, a / det, (c * tx - a * ty) / det],
    ])
    warped = Image.fromarray(image).transform(
        (size, size), Image.AFFINE, inverse.reshape(-1), resample=Image.BILINEAR
    )
    return np.asarray(warped, dtype=np.uint8)


def _member_identity(name: str) -> Optional[Tuple[str, str]]:
    """``images/<identity>/<file>`` (or ``<identity>/<file>``) -> (identity, file)."""
    parts = name.split("/")
    if len(parts) >= 3 and parts[-2] and parts[-1]:
        return parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pack(src: Path, out: Path) -> int:
    import lmdb
    from PIL import Image
    from afrlib.data.lmdb_io import SCHEMA, estimate_map_size, invalidate

    archives = sorted(p for p in src.iterdir() if p.name.endswith((".tar", ".tar.gz")))
    if not archives or not (src / "lmk.tsv").is_file():
        raise SystemExit(f"{src} must hold the MillionCelebs archives and lmk.tsv")
    print("[pack] indexing lmk.tsv", flush=True)
    landmarks = Landmarks(src / "lmk.tsv")
    expected_samples, expected_identities = landmarks.n_samples, len(landmarks.index)
    alignment_id = _canonical({"lmk_sha256": _sha256(src / "lmk.tsv"), "template": "arcface_112",
                               "backend": "frbench_similarity_transform"})
    pack_id = _canonical({"schema": PACK_SCHEMA, "archives": [[p.name, p.stat().st_size] for p in archives],
                          "aligned_size": ALIGNED_SIZE, "jpeg_quality": JPEG_QUALITY, "alignment_id": alignment_id})

    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / PROGRESS
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if (out / "data.mdb").is_file() and progress.get("pack_id") != pack_id:
        raise SystemExit(f"{out} holds a different or finished pack; remove it to rebuild")
    done = set(progress.get("archives", []))
    count = int(progress.get("count", 0))
    per_identity: Dict[str, int] = progress.get("per_identity", {})

    env = lmdb.open(str(out), subdir=True, meminit=False,
                    map_size=estimate_map_size(expected_samples * 6000, expected_samples))
    try:
        for archive in archives:
            if archive.name in done:
                continue
            print(f"[pack] {archive.name} from sample {count}", flush=True)
            txn = env.begin(write=True)
            pending = 0
            with tarfile.open(archive, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    parsed = _member_identity(member.name)
                    if parsed is None:
                        continue
                    identity, filename = parsed
                    with Image.open(io.BytesIO(tar.extractfile(member).read())) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    aligned = Image.fromarray(align_arcface(rgb, landmarks.get(identity, filename)))
                    buffer = io.BytesIO()
                    aligned.save(buffer, format="JPEG", quality=JPEG_QUALITY)
                    key = f"{identity}__{filename}"
                    txn.put(f"{key}/image".encode("utf-8"), buffer.getvalue())
                    record = {"key": key, "identity": identity, "name": member.name}
                    txn.put(f"meta/sample/{count:08d}".encode("ascii"),
                            json.dumps(record, separators=(",", ":")).encode("utf-8"))
                    count += 1
                    per_identity[identity] = per_identity.get(identity, 0) + 1
                    pending += 1
                    if pending >= 512:
                        txn.commit()
                        txn = env.begin(write=True)
                        pending = 0
            txn.commit()
            done.add(archive.name)
            progress_path.write_text(json.dumps({"pack_id": pack_id, "archives": sorted(done),
                                                 "count": count, "per_identity": per_identity}))
        short = [i for i, (_, _, rows) in landmarks.index.items() if per_identity.get(i, 0) != rows]
        if count != expected_samples or short:
            raise SystemExit(f"packed {count}/{expected_samples} samples; {len(short)} identities incomplete "
                             f"(e.g. {short[:3]}); the source is incomplete")
        provenance = {
            "name": "afrbench_noise", "source": "MillionCelebs", "pack_schema": PACK_SCHEMA,
            "pack_id": pack_id, "aligned": True, "align_template": "arcface_112",
            "alignment_id": alignment_id, "expected_source_samples": expected_samples,
            "expected_source_identities": expected_identities, "lock_independent": True,
            "identity_exclusions": [], "full_source": True, "smoke": False,
            "n_samples": count, "n_identities": len(per_identity), "count": count, "schema": SCHEMA,
        }
        with env.begin(write=True) as txn:
            txn.put(b"meta/count", str(count).encode("ascii"))
            txn.put(b"meta/json", json.dumps(provenance, sort_keys=True).encode("utf-8"))
            txn.put(b"meta/schema", SCHEMA.encode("ascii"))
        env.sync()
    finally:
        env.close()
    invalidate(out)
    progress_path.unlink()
    print(f"[pack] {count} samples, {len(per_identity)} identities -> {out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrbench.millioncelebs", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="fetch the release from the official share")
    get.add_argument("--out", type=Path, required=True)
    get.add_argument("--files", nargs="*", default=None, help=f"subset of {sorted(FILES)}")
    make = commands.add_parser("pack", help="align and pack into the distractor LMDB")
    make.add_argument("--src", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        return download(args.out, args.files)
    return pack(args.src, args.out)


if __name__ == "__main__":
    sys.exit(main())
