"""Draw the "by the numbers" strip of the README.

Every figure on the strip comes from code or from the protocol document:

* FR models, backbones, losses: ``frbench.list_models()`` (see ``model_zoo.py``; counted from ``model_zoo.json``)
* methods: ``methods.json`` (24 evaluated of 26 implemented; the strip says 24+ and 45+ because both grow)
* photos, distractors, scenarios, evaluation jobs: ``docs/protocol.md``

    python tools/readme_figures/by_the_numbers.py      # writes docs/assets/by_the_numbers.svg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.readme_figures.svgkit import INK, MUTED, Svg
except ImportError:  # run as a script
    from svgkit import INK, MUTED, Svg

HERE = Path(__file__).resolve().parent
ACCENT = "#0969da"


def build(zoo: list, methods: list) -> str:
    backbones = len({m["backbone"] for m in zoo})
    losses = len({m["loss"] for m in zoo})
    datasets = len({m["dataset"] for m in zoo})
    evaluated = sum(m["evaluated"] for m in methods)
    tiles = [
        (f"{len(zoo)}+", "FR models", [f"{backbones} backbones, {losses} losses,", f"{datasets} training sets"]),
        (f"{evaluated}+", "anti-FR methods", [f"{len({m['paradigm'] for m in methods})} paradigms,",
                                              f"{min(m['year'] for m in methods)} to {max(m['year'] for m in methods)}"]),
        ("5,000", "hand-picked photos", ["50 people,", "100 high-resolution photos each"]),
        ("22.8M", "distractor faces", ["MillionCelebs,", "719,563 identities"]),
        ("4", "threat scenarios", ["1A, 1B, 2A, 2B"]),
        ("17", "evaluation jobs", ["defenses, white-box models,", "distractor variants"]),
    ]
    width, height, pad = 960, 142, 22
    svg = Svg(width, height, "AFRBench by the numbers", "; ".join(f"{n} {label}" for n, label, _ in tiles))
    tile_w = (width - 2 * pad) / len(tiles)
    for i, (number, label, detail) in enumerate(tiles):
        x = pad + i * tile_w
        if i:
            svg.line(x, 26, x, height - 26, stroke="#eaeef2")
        cx = x + tile_w / 2
        svg.text(cx, 62, number, size=34, fill=ACCENT, bold=True, anchor="middle")
        svg.text(cx, 86, label, size=13.5, fill=INK, bold=True, anchor="middle")
        for k, line in enumerate(detail):
            svg.text(cx, 106 + 15 * k, line, size=10.5, fill=MUTED, anchor="middle")
    return svg.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", type=Path, default=HERE / "methods.json")
    parser.add_argument("--zoo", type=Path, default=HERE / "model_zoo.json")
    parser.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "assets" / "by_the_numbers.svg")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(json.loads(args.zoo.read_text()), json.loads(args.methods.read_text())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
