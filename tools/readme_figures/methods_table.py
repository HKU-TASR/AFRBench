"""Write the README's method table from ``methods.json``: one small table per paradigm, newest first.

The README holds the table between ``<!-- methods-table:start -->`` and ``<!-- methods-table:end -->``.

    python tools/readme_figures/methods_table.py                  # print the tables
    python tools/readme_figures/methods_table.py --update README.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = "<!-- methods-table:start -->", "<!-- methods-table:end -->"

PARADIGMS = [
    ("masks", "Personalized masks"),
    ("per_sample", "Per-sample perturbation optimization"),
    ("face", "Face synthesis"),
    ("makeup", "Makeup synthesis"),
    ("accessory", "Accessory"),
]


def build(methods: list) -> str:
    blocks = []
    for key, title in PARADIGMS:
        members = sorted((m for m in methods if m["paradigm"] == key), key=lambda m: (-m["year"], m["name"].lower()))
        lines = [f"**{title}**", "", "| Method | Venue | Paper | Code |", "|---|---|---|---|"]
        for m in members:
            name = m["name"] + ("" if m["evaluated"] else " \u2020")
            code = f"[code]({m['code']})" if m["code"] else "\u2013"
            lines.append(f"| {name} | {m['venue']} | [paper]({m['paper']}) | {code} |")
        blocks.append("\n".join(lines))
    note = ("\u2020 Implemented in AFRLib but not part of the AFRBench-basic protocol. Per-stage inputs are listed in "
            "[docs/methods.md](docs/methods.md).")
    return "\n\n".join(blocks) + "\n\n" + note + "\n"


def update(readme: Path, table: str) -> bool:
    """Replace the region between the markers; returns whether the file changed."""
    text = readme.read_text()
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    if not rest or not tail and END not in rest:
        raise SystemExit(f"{readme} has no {START} ... {END} region")
    new = f"{head}{START}\n\n{table}\n{END}{tail}"
    changed = new != text
    if changed:
        readme.write_text(new)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", type=Path, default=HERE / "methods.json")
    parser.add_argument("--update", type=Path, default=None, help="README to rewrite in place")
    args = parser.parse_args()
    table = build(json.loads(args.methods.read_text()))
    if args.update:
        print(("updated " if update(args.update, table) else "already up to date: ") + str(args.update))
    else:
        print(table)


if __name__ == "__main__":
    main()
