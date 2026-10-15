"""Draw the architecture chart of the README overview: AFRLib (the library) under AFRBench (the benchmark).

Every box states what the code does:

* corpora: ``afrlib/data/registry.py``; image formats: ``afrlib/contracts.py`` (``ImageSpec``) and
  ``afrlib/utils/face_crop.py`` (crop, align, paste back); stages: ``afrlib/afr/base.py``
* data, jobs, scenarios, FR system, metrics: ``docs/protocol.md``

    python tools/readme_figures/architecture.py         # writes docs/assets/architecture.svg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

try:
    from tools.readme_figures.svgkit import INK, MUTED, Svg, hsl, text_width
except ImportError:  # run as a script
    from svgkit import INK, MUTED, Svg, hsl, text_width

HERE = Path(__file__).resolve().parent
BENCH_HUE, LIB_HUE = 232, 172


def box(svg: Svg, x: float, y: float, w: float, h: float, hue: float, title: str, lines: List[str],
        kicker: Optional[str] = None) -> None:
    svg.rect(x, y, w, h, "#ffffff", stroke=hsl(hue, 0.45, 0.62), radius=10, stroke_width=1.3)
    ty = y + 25
    if kicker:
        svg.text(x + 12, y + 20, kicker.upper(), size=9.5, fill=hsl(hue, 0.5, 0.42), bold=True)
        ty = y + 40
    svg.text(x + 12, ty, title, size=14, fill=hsl(hue, 0.55, 0.24), bold=True)
    for i, line in enumerate(lines):
        svg.text(x + 12, ty + 22 + 17 * i, line, size=11.5, fill=INK if i == 0 else "#3d444d")


def arrow(svg: Svg, x1: float, y1: float, x2: float, y2: float, color: str) -> None:
    svg.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2"/>')
    # Arrow head pointing from (x1, y1) to (x2, y2); the figure only draws horizontal and vertical arrows.
    if abs(x2 - x1) < 1e-6:
        d = 1 if y2 > y1 else -1
        pts = f"{x2 - 5:.1f},{y2 - 8 * d:.1f} {x2 + 5:.1f},{y2 - 8 * d:.1f} {x2:.1f},{y2:.1f}"
    else:
        d = 1 if x2 > x1 else -1
        pts = f"{x2 - 8 * d:.1f},{y2 - 5:.1f} {x2 - 8 * d:.1f},{y2 + 5:.1f} {x2:.1f},{y2:.1f}"
    svg.raw(f'<polygon points="{pts}" fill="{color}"/>')


def build() -> str:
    width, height, pad = 960, 548, 22
    svg = Svg(
        width, height, "AFRBench architecture",
        "AFRBench (data, jobs, FR system, protection and quality metrics) runs on AFRLib (training corpora, image "
        "distribution, the train, fit and protect stages, and FR models).",
    )
    col_w, gap = 176, 9
    cols = [pad + i * (col_w + gap) for i in range(5)]
    inner_x, inner_w = pad + 14, width - 2 * pad - 28

    # AFRBench band.
    band_y, band_h = 16, 236
    svg.rect(pad, band_y, width - 2 * pad, band_h, hsl(BENCH_HUE, 0.6, 0.97), stroke=hsl(BENCH_HUE, 0.4, 0.86), radius=14)
    svg.circle(pad + 22, band_y + 25, 6, hsl(BENCH_HUE, 0.55, 0.5))
    svg.text(pad + 36, band_y + 30, "AFRBench", size=16, fill=hsl(BENCH_HUE, 0.55, 0.28), bold=True)
    svg.text(pad + 124, band_y + 30, "the benchmark: one protocol for every method", size=12, fill=MUTED)
    top_y, top_h = band_y + 48, 172
    cw = (width - 2 * pad - 16 - 4 * 8) / 5
    cx = [pad + 8 + i * (cw + 8) for i in range(5)]
    box(svg, cx[0], top_y, cw, top_h, BENCH_HUE, "Data", [
        "50 protectees,", "100 hand-picked photos each", "5 HQ targets, 2,000-ID pool", "22.8M distractor faces",
        "AFRBench-DF: video frames",
    ], kicker="datasets")
    box(svg, cx[1], top_y, cw, top_h, BENCH_HUE, "Jobs", [
        "One YAML job per step,", "locked, resumable", "train, fit, protect, eval", "scenarios 1A, 1B, 2A, 2B",
    ], kicker="protocol")
    box(svg, cx[2], top_y, cw, top_h, BENCH_HUE, "FR system", [
        "quantize, JPEG, resize", "RetinaFace, ArcFace align", "purifiers: median, GFPGAN,", "CodeFormer, DiffPure",
        "flip test-time augmentation",
    ], kicker="intruder")
    mx, mw = cx[3], cx[4] + cw - cx[3]
    svg.rect(mx, top_y, mw, top_h, "#ffffff", stroke=hsl(BENCH_HUE, 0.45, 0.62), radius=10, stroke_width=1.3)
    svg.text(mx + 12, top_y + 20, "SCORING", size=9.5, fill=hsl(BENCH_HUE, 0.5, 0.42), bold=True)
    svg.text(mx + 12, top_y + 40, "Metrics", size=14, fill=hsl(BENCH_HUE, 0.55, 0.24), bold=True)
    half = (mw - 36) / 2
    for k, (head, lines) in enumerate([
        ("Protection", ["recall@K, AP, NDCG", "rank-k identification", "1,250 queries per run"]),
        ("Visual quality", ["PSNR, SSIM, MS-SSIM", "LPIPS, BRISQUE, NIQE", "FID per method"]),
    ]):
        sx = mx + 12 + k * (half + 12)
        svg.rect(sx, top_y + 52, half, top_h - 64, hsl(BENCH_HUE, 0.5, 0.96), radius=8)
        svg.text(sx + 10, top_y + 74, head, size=12.5, fill=hsl(BENCH_HUE, 0.55, 0.28), bold=True)
        for i, line in enumerate(lines):
            svg.text(sx + 10, top_y + 96 + 17 * i, line, size=11.5, fill="#3d444d")
    for i in range(3):
        arrow(svg, cx[i] + cw + 0.5, top_y + top_h / 2, cx[i + 1] - 0.5, top_y + top_h / 2, hsl(BENCH_HUE, 0.5, 0.55))

    # AFRLib band.
    lib_y, lib_h = 306, 224
    svg.rect(pad, lib_y, width - 2 * pad, lib_h, hsl(LIB_HUE, 0.5, 0.96), stroke=hsl(LIB_HUE, 0.35, 0.82), radius=14)
    svg.circle(pad + 22, lib_y + 25, 6, hsl(LIB_HUE, 0.55, 0.4))
    svg.text(pad + 36, lib_y + 30, "AFRLib", size=16, fill=hsl(LIB_HUE, 0.6, 0.2), bold=True)
    svg.text(width - pad - 18, lib_y + 30, "the library: 26 methods behind one interface", size=12, fill=MUTED,
             anchor="end")
    low_y, low_h = lib_y + 48, 160
    box(svg, cx[0], low_y, cw, low_h, LIB_HUE, "Training corpora", [
        "MS1M, CASIA-WebFace,", "VGGFace2, MT, LADN,", "CelebAMask-HQ", "protectees filtered out",
    ], kicker="data")
    box(svg, cx[1], low_y, cw, low_h, LIB_HUE, "Image distribution", [
        "An ImageSpec per input:", "crop, align, size, range", "paste-back into the", "original photo",
    ], kicker="formats")
    sx0, sw = cx[2], cx[3] + cw - cx[2]
    svg.rect(sx0, low_y, sw, low_h, "#ffffff", stroke=hsl(LIB_HUE, 0.45, 0.55), radius=10, stroke_width=1.3)
    svg.text(sx0 + 12, low_y + 20, "METHODS", size=9.5, fill=hsl(LIB_HUE, 0.5, 0.36), bold=True)
    svg.text(sx0 + 12, low_y + 40, "Three stages, one contract", size=14, fill=hsl(LIB_HUE, 0.6, 0.2), bold=True)
    stage_w = (sw - 24 - 2 * 20) / 3
    for k, (name, who, what) in enumerate([
        ("train", "the population", "corpus, target"),
        ("fit", "one protectee", "album photos"),
        ("protect", "any photo", "photo in, photo out"),
    ]):
        bx = sx0 + 12 + k * (stage_w + 20)
        svg.rect(bx, low_y + 54, stage_w, 62, hsl(LIB_HUE, 0.45, 0.94), stroke=hsl(LIB_HUE, 0.4, 0.7), radius=8)
        svg.text(bx + 10, low_y + 74, name, size=13.5, fill=hsl(LIB_HUE, 0.6, 0.2), bold=True)
        svg.text(bx + 10, low_y + 91, who, size=11, fill="#3d444d")
        svg.text(bx + 10, low_y + 106, what, size=10, fill=MUTED)
        if k < 2:
            arrow(svg, bx + stage_w + 3, low_y + 85, bx + stage_w + 17, low_y + 85, hsl(LIB_HUE, 0.5, 0.45))
    svg.text(sx0 + 12, low_y + 134, "Each method declares the inputs its stages read,", size=11, fill="#3d444d")
    svg.text(sx0 + 12, low_y + 150, "so callers never branch on the method.", size=11, fill="#3d444d")
    box(svg, cx[4], low_y, cw, low_h, LIB_HUE, "FR models", [
        "FRBench: 45+ models", "FaceNet, or your own,", "injected as attack", "surrogates",
    ], kicker="targets")

    # Arrows between the layers.
    down_x, up_x = cx[1] + cw / 2, cx[2] + cw / 2
    arrow(svg, down_x, top_y + top_h + 2, down_x, low_y - 2, hsl(BENCH_HUE, 0.5, 0.55))
    svg.text(down_x + 10, (top_y + top_h + low_y) / 2 + 4, "original photos, roles", size=11, fill=MUTED)
    arrow(svg, up_x, low_y - 2, up_x, top_y + top_h + 2, hsl(LIB_HUE, 0.5, 0.45))
    svg.text(up_x + 10, (top_y + top_h + low_y) / 2 + 4, "protected photos", size=11, fill=MUTED)
    return svg.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "assets" / "architecture.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
