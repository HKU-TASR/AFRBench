"""Render the README showcase: one face, protected by every AFR method that can run on a single photo.

The subjects are the HQ target faces shipped in ``data/afrbench-basic`` (CelebA-HQ). For each subject the script builds
every method the way the benchmark does (the protocol's attack surrogates, the constructor defaults of
``afrbench.protocol.resolved_method_defaults``, seed 0), protects the subject's face frame with it, and crops every
output to the same face window. A method with a train stage uses a finished run folder (``--train-runs``, the ``train``
folder of a benchmark output). Methods that fit to a protectee's album need 50 photos of the same person, so they are
not shown; the script records why each skipped method was skipped.

    # on a GPU machine: protect and cache one PNG per (subject, method) under --cache
    python tools/readme_figures/make_showcase.py --train-runs results/basic/train --cache showcase_cache
    # anywhere with Pillow: lay the cached tiles out as docs/assets/showcase_<subject>.jpg
    python tools/readme_figures/make_showcase.py --assemble-only --cache showcase_cache
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

PARADIGM_ORDER = ["masks", "per_sample", "face", "makeup", "accessory"]
PARADIGM_TITLES = {
    "masks": "Personalized masks", "per_sample": "Per-sample perturbation optimization", "face": "Face synthesis",
    "makeup": "Makeup synthesis", "accessory": "Accessory",
}
PARADIGM_HUES = {"masks": 186, "per_sample": 248, "face": 12, "makeup": 328, "accessory": 78}


def load_methods(path: Path = HERE / "methods.json") -> Dict[str, Dict[str, Any]]:
    return {m["key"]: m for m in json.loads(path.read_text())}


# --------------------------------------------------------------------------------------------------------------------
# Protect (needs a GPU and the AFRLib weights)
# --------------------------------------------------------------------------------------------------------------------


def valid_rect_of(record) -> List[int]:
    """The ``(x, y, w, h)`` of the photo inside its padded frame, from the record or, for the shipped HQ targets,
    from the frame window and the image size.
    """
    if record.valid_rect:
        return [int(v) for v in record.valid_rect]
    ox, oy, side = record.original_box
    width, height = record.size[:2]
    x0, y0 = max(-ox, 0), max(-oy, 0)
    return [x0, y0, min(-ox + width, side) - x0, min(-oy + height, side) - y0]


def display_window(geometry: Dict[str, Any], valid_rect: List[int], scale: float) -> List[int]:
    """The square window ``(x, y, side)`` of the frame shown in a tile: ``scale`` face sides around the face centre,
    moved inside the photo's valid area where it can be.
    """
    x1, y1, x2, y2 = geometry["face_box"]
    side = max(x2 - x1, y2 - y1) * scale
    vx, vy, vw, vh = valid_rect
    side = min(side, vw, vh)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x0 = min(max(cx - side / 2.0, vx), vx + vw - side)
    y0 = min(max(cy - side / 2.0, vy), vy + vh - side)
    return [int(round(x0)), int(round(y0)), int(round(side))]


def psnr_on_valid(original, protected, valid_rect: List[int]) -> float:
    """PSNR between the original and the protected frame inside the photo's valid area, after rounding to uint8."""
    import torch

    vx, vy, vw, vh = valid_rect
    a = original[:, vy:vy + vh, vx:vx + vw].round().clamp(0, 255)
    b = protected[:, vy:vy + vh, vx:vx + vw].round().clamp(0, 255)
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def save_tile(frame, window: List[int], path: Path) -> None:
    """Write the ``window`` of a ``[3, H, W]`` float frame in ``[0, 255]`` as a lossless PNG."""
    import numpy as np
    from PIL import Image

    x, y, side = window
    crop = frame[:, y:y + side, x:x + side].detach().round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(crop)).save(path)


def build_method(key: str, device, surrogates, train_runs: Optional[Path], target: str):
    """The method as a benchmark job builds it, or ``(None, reason)`` when it cannot be shown."""
    from afrlib import AFR_REGISTRY
    from afrbench import protocol as P
    from afrbench.benchmark import method_dir

    cls = AFR_REGISTRY[key]
    roles = P.method_roles(cls)
    overrides = P.resolved_method_defaults(key)
    if roles["fit"]:
        return None, "fits to a protectee's album (50 photos of one person)"
    if roles["train"]:
        if train_runs is None:
            return None, "has a train stage and no --train-runs folder was given"
        run = train_runs / method_dir(key) / target if roles["target_baked"] else train_runs / method_dir(key)
        if not (run / "config.json").is_file():
            return None, f"no finished train run at {run}"
        return cls.from_run(run, device, surrogates, **overrides), None
    return cls(device, surrogates, **overrides), None


def protect_one(method, key: str, dataset, record, target_record, pool_records, device):
    """Protect the subject's face frame; returns ``(original_frame, protected_frame)``."""
    from afrlib.contracts import EvalData, assemble_stage
    from afrbench.runner import _prepare_records, seed_job_rng

    specs = method.input_specs("protect")
    roles: Dict[str, Any] = {}
    if "target_img" in specs:
        roles["target_img"] = _prepare_records([target_record], dataset, specs["target_img"], device, 1)
    if "target_pool" in specs:
        roles["target_pool"] = _prepare_records(pool_records, dataset, specs["target_pool"], device)
    original = dataset.original(record).to(device)
    geometry = dataset.geometry(record)
    seed_job_rng(0)
    protected = method.protect(assemble_stage(
        method.eval_inputs, EvalData, {**roles, "eval_imgs": [original], "eval_geometry": [geometry]},
    ))[0]
    return original, protected


def run(args: argparse.Namespace) -> None:
    import torch

    from afrlib import build_fr_ensemble
    from afrbench import protocol as P
    from afrbench.dataset import BenchDataset

    methods = load_methods()
    dataset = BenchDataset(args.dataset)
    hq = dataset.hq_target_identities()
    subjects = args.subjects or hq
    unknown = [s for s in subjects if s not in hq]
    if unknown:
        raise SystemExit(f"{unknown} are not HQ targets of {args.dataset}; they are {hq}")
    wanted = args.methods or sorted(methods)
    device = torch.device(args.device)
    surrogates = None  # built lazily: a run with every method cached needs no GPU
    for subject in subjects:
        target = hq[(hq.index(subject) + 1) % len(hq)]
        pool_ids = [i for i in hq if i not in (subject, target)]
        record = dataset.images_of(subject, split="target")[0]
        target_record = dataset.images_of(target, split="target")[0]
        pool_records = [dataset.images_of(i, split="target")[0] for i in pool_ids]
        geometry = dataset.geometry(record)
        valid = valid_rect_of(record)
        window = display_window(geometry, valid, args.window)
        folder = args.cache / subject
        status_path = folder / "status.json"
        status: Dict[str, Any] = json.loads(status_path.read_text()) if status_path.is_file() else {}
        status["_subject"] = {"identity": subject, "target": target, "pool": pool_ids, "window": window}
        original = dataset.original(record)
        if not (folder / "original.png").is_file():
            save_tile(original, window, folder / "original.png")
        for key in wanted:
            if key in P.HELD:
                status[key] = {"status": "skipped", "reason": "not part of the AFRBench-basic protocol"}
                continue
            done = status.get(key, {})
            if done.get("status") in ("ok", "skipped") and not args.force:
                continue
            started = time.monotonic()
            try:
                if surrogates is None:
                    surrogates = build_fr_ensemble(P.SURROGATES, device)
                method, reason = build_method(key, device, surrogates, args.train_runs, target)
                if method is None:
                    status[key] = {"status": "skipped", "reason": reason}
                    print(f"[{subject}] {key}: skipped, {reason}", flush=True)
                else:
                    original, protected = protect_one(method, key, dataset, record, target_record, pool_records,
                                                      device)
                    frame = protected.detach().cpu()
                    original = original.detach().cpu()
                    psnr = psnr_on_valid(original, frame, valid)
                    unmodified = bool(torch.equal(frame.to(original.dtype), original))
                    save_tile(frame, window, folder / f"{key}.png")
                    status[key] = {
                        "status": "ok", "psnr": psnr, "unmodified": unmodified,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                    print(f"[{subject}] {key}: ok, PSNR {psnr:.2f} dB, {status[key]['seconds']} s"
                          + (" (returned the original unchanged)" if unmodified else ""), flush=True)
                    del method
            except Exception as error:  # keep going: one failing method must not lose the others
                status[key] = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                               "trace": traceback.format_exc()[-1500:]}
                print(f"[{subject}] {key}: FAILED {type(error).__name__}: {error}", flush=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            status_path.write_text(json.dumps(status, indent=2))
        status_path.write_text(json.dumps(status, indent=2))


# --------------------------------------------------------------------------------------------------------------------
# Assemble (Pillow only)
# --------------------------------------------------------------------------------------------------------------------


def hsl_rgb(hue: float, sat: float, light: float):
    import colorsys

    r, g, b = colorsys.hls_to_rgb(hue / 360.0, light, sat)
    return int(r * 255), int(g * 255), int(b * 255)


def assemble(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    methods = load_methods()
    subjects = args.subjects or sorted(p.name for p in args.cache.iterdir() if (p / "status.json").is_file())
    name_font = ImageFont.load_default(size=21)
    small_font = ImageFont.load_default(size=15)
    for subject in subjects:
        folder = args.cache / subject
        status = json.loads((folder / "status.json").read_text())
        shown = [k for k, v in status.items() if not k.startswith("_") and v.get("status") == "ok"]
        shown.sort(key=lambda k: (PARADIGM_ORDER.index(methods[k]["paradigm"]), -methods[k]["year"],
                                  methods[k]["name"].lower()))
        cells = [("Original", "the unprotected photo", (60, 66, 72), folder / "original.png", None)]
        for key in shown:
            m = methods[key]
            cells.append((m["name"], f"{status[key]['psnr']:.1f} dB PSNR" if math.isfinite(status[key]["psnr"])
                          else "identical", hsl_rgb(PARADIGM_HUES[m["paradigm"]], 0.55, 0.45),
                          folder / f"{key}.png", m["paradigm"]))
        columns, tile, gap, caption = args.columns, args.tile, 12, 52
        rows = math.ceil(len(cells) / columns)
        width = columns * tile + (columns + 1) * gap
        legend_h = 40
        height = rows * (tile + caption) + (rows + 1) * gap + legend_h
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (name, note, color, path, _paradigm) in enumerate(cells):
            r, c = divmod(index, columns)
            x = gap + c * (tile + gap)
            y = gap + r * (tile + caption + gap)
            image = Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS)
            canvas.paste(image, (x, y))
            draw.rectangle([x, y + tile, x + tile - 1, y + tile + 4], fill=color)
            draw.text((x + 2, y + tile + 9), name, fill=(31, 35, 40), font=name_font)
            draw.text((x + 2, y + tile + 33), note, fill=(89, 99, 110), font=small_font)
        # Legend of paradigm colours.
        lx, ly = gap, height - legend_h + 10
        for paradigm in PARADIGM_ORDER:
            if not any(methods[k]["paradigm"] == paradigm for k in shown):
                continue
            color = hsl_rgb(PARADIGM_HUES[paradigm], 0.55, 0.45)
            draw.rectangle([lx, ly + 3, lx + 22, ly + 9], fill=color)
            draw.text((lx + 30, ly - 3), PARADIGM_TITLES[paradigm], fill=(89, 99, 110), font=small_font)
            lx += 30 + int(draw.textlength(PARADIGM_TITLES[paradigm], font=small_font)) + 26
        out = args.out / f"showcase_{subject}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out, quality=args.quality, optimize=True, progressive=True)
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(shown)} methods)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=REPO / "data" / "afrbench-basic")
    parser.add_argument("--train-runs", type=Path, default=None, help="the 'train' folder of a benchmark output")
    parser.add_argument("--cache", type=Path, default=Path("showcase_cache"))
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "assets")
    parser.add_argument("--subjects", nargs="*", default=None, help="HQ target identities (default: all five)")
    parser.add_argument("--methods", nargs="*", default=None, help="registry names (default: every method)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=float, default=1.2, help="tile side in face-box sides")
    parser.add_argument("--force", action="store_true", help="recompute cached methods")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--tile", type=int, default=288, help="tile side in pixels")
    parser.add_argument("--quality", type=int, default=88)
    args = parser.parse_args()
    if not args.assemble_only:
        run(args)
    assemble(args)


if __name__ == "__main__":
    main()
