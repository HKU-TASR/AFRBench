"""Draw the method-coverage chart: the anti-FR methods grouped by paradigm, newest first.

Within a paradigm the order is publication year (newest first), then name. Methods marked with a dagger are
described and registered in AFRLib but not evaluated in AFRBench yet; they are drawn with a dashed outline.

    python tools/readme_figures/paradigms.py            # writes docs/assets/paradigms.svg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.readme_figures.svgkit import INK, MUTED, PARADIGM_HUES, Svg, flow_chips, hsl, text_width
except ImportError:  # run as a script
    from svgkit import INK, MUTED, PARADIGM_HUES, Svg, flow_chips, hsl, text_width

HERE = Path(__file__).resolve().parent

PARADIGMS = [
    ("masks", "Personalized masks", "one learned mask per protectee, applied to any photo"),
    ("per_sample", "Per-sample perturbation optimization", "an adversarial perturbation optimized for each photo"),
    ("face", "Face synthesis", "a generator re-creates the face with the identity moved"),
    ("makeup", "Makeup synthesis", "adversarial makeup transferred onto the face"),
    ("accessory", "Accessory", "an adversarial patch, glasses or hat"),
]


def wrap(title: str, max_width: float, size: float = 14.5) -> list:
    """Break a paradigm title into lines no wider than ``max_width`` (greedy, on spaces)."""
    lines, line = [], ""
    for word in title.split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size, bold=True) > max_width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def label_height(title: str, max_width: float) -> float:
    """Height of a row's label block: the title lines and the count line."""
    return 18 * len(wrap(title, max_width)) + 26


def build(methods: list) -> str:
    width, label_w, pad = 960, 244, 22
    # Lay each paradigm row out first so the card height is known.
    rows, height = [], 74
    for key, title, blurb in PARADIGMS:
        members = sorted(
            (m for m in methods if m["paradigm"] == key), key=lambda m: (-m["year"], m["name"].lower())
        )
        chips = [(m["name"] + (" \u2020" if not m["evaluated"] else ""), str(m["year"]), not m["evaluated"])
                 for m in members]
        bottom = flow_chips(Svg(width, 10, "probe"), chips, 0, 0, width - label_w - 2 * pad, 0)
        rows.append((key, title, blurb, members, chips, bottom))
        height += max(bottom, label_height(title, label_w - 2 * pad - 14)) + 26
    height += 28

    svg = Svg(
        width, height, "Anti-face-recognition methods in AFRLib and AFRBench, by paradigm",
        "Twenty-six methods in five paradigms, newest first; the two marked with a dagger are not evaluated yet.",
    )
    svg.text(pad, 36, "26 anti-FR methods, 5 paradigms", size=19, bold=True)
    svg.text(pad, 57, "Newest first within each paradigm. Dashed outline (\u2020): implemented in AFRLib, "
                      "not evaluated in AFRBench yet.", size=12.5, fill=MUTED)
    y = 84
    for key, title, blurb, members, chips, bottom in rows:
        hue = PARADIGM_HUES[key]
        lines = wrap(title, label_w - 2 * pad - 14)
        row_h = max(bottom, label_height(title, label_w - 2 * pad - 14))
        svg.rect(pad - 8, y - 6, 4, row_h + 12, hsl(hue, 0.5, 0.5), radius=2)
        for k, line in enumerate(lines):
            svg.text(pad + 6, y + 15 + 18 * k, line, size=14.5, fill=hsl(hue, 0.55, 0.25), bold=True)
        count = sum(m["evaluated"] for m in members)
        svg.text(pad + 6, y + 33 + 18 * (len(lines) - 1),
                 f"{len(members)} methods" + (f", {count} evaluated" if count != len(members) else ""),
                 size=11.5, fill=MUTED)
        flow_chips(svg, chips, label_w + pad, y, width - label_w - 2 * pad, hue)
        y += row_h + 26
        if key != PARADIGMS[-1][0]:
            svg.line(pad - 8, y - 14, width - pad, y - 14, stroke="#eaeef2")
    return svg.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", type=Path, default=HERE / "methods.json")
    parser.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "assets" / "paradigms.svg")
    args = parser.parse_args()
    methods = json.loads(args.methods.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(methods))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
