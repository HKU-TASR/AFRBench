"""Aggregate eval outputs into a summary table."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

#: The query-mix sweep's sample points (r = fraction of protected queries).
MIX_RS = tuple(round(r * 0.1, 1) for r in range(11))

#: Columns that identify a row rather than measure it.
KEY_COLUMNS = ("run", "fr_system", "scenario", "query_id", "seen", "identity")

#: Columns every summary groups on.
INDEX_COLUMNS = ("run", "fr_system", "seen", "scenario")

#: Quality metrics that exist only for a whole run, carried through unaveraged.
SET_LEVEL_METRICS = ("fid", "n_images")


def summarize_experiment(eval_dirs: Sequence[Path], out_path: Path) -> Path:
    """Roll up the eval ``save_path`` directories into one summary CSV, with each metric's mean
    and variance.
    """
    rows: List[Dict[str, Any]] = []
    set_rows: List[Dict[str, Any]] = []
    for directory in eval_dirs:
        rows.extend(_read_rows(directory / "per_image.jsonl"))
        set_rows.extend(_set_level_rows(directory / "metrics.json"))
    if not rows and not set_rows:
        raise RuntimeError(f"No eval outputs found under {[str(d) for d in eval_dirs]}")

    table = pd.DataFrame(rows)
    if not table.empty:
        table = pd.concat(
            [table, _query_mix_rows(table), _gallery_mix_endpoints(table)],
            ignore_index=True,
        )
    summary = _aggregate(table) if not table.empty else pd.DataFrame()
    if set_rows:
        summary = summary.merge(
            pd.DataFrame(set_rows), how="left",
            on=[c for c in ("run", "fr_system", "scenario") if c in summary.columns],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return out_path


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _set_level_rows(path: Path) -> List[Dict[str, Any]]:
    """Run-level quality metrics from ``metrics.json``, which have no per-image value and so are
    carried alongside rather than averaged.
    """
    if not path.is_file():
        return []
    metrics = json.loads(path.read_text())
    out = []
    for run, values in (metrics.get("visual_quality") or {}).items():
        row = {"run": run, "fr_system": metrics.get("fr_system"),
               "scenario": "visual_quality"}
        row.update({k: v for k, v in values.items() if k in SET_LEVEL_METRICS})
        out.append(row)
    return out


def _aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and variance of every metric column per (run, fr_system, seen, scenario)."""
    group_cols = [c for c in INDEX_COLUMNS if c in table.columns]
    metric_cols = [
        c for c in table.columns
        if c not in group_cols and c not in KEY_COLUMNS
        and pd.api.types.is_numeric_dtype(table[c])
    ]
    grouped = table.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "var"])
    grouped.columns = [f"{metric}.{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _query_mix_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The query-mix sweep, derived per query from the 1A and 2A rows: each query takes its
    protected row with probability r, drawn from a seed on its id so the sweeps are nested.
    """
    if "scenario" not in table.columns:
        return pd.DataFrame()
    mixable = table[table["scenario"].isin(("1A", "2A"))]
    if mixable.empty:
        return pd.DataFrame()

    out: List[Dict[str, Any]] = []
    for (run, system), block in mixable.groupby(["run", "fr_system"], dropna=False):
        clean = block[block["scenario"] == "1A"].set_index("query_id")
        protected = block[block["scenario"] == "2A"].set_index("query_id")
        shared = [q for q in clean.index if q in protected.index]
        for r in MIX_RS:
            for query_id in shared:
                # Pair the adoption draw across methods and FR systems so every
                # run is compared on the same clean/protected query assignment.
                draw = random.Random(f"query-mix:{query_id}").random()
                source = protected if draw < r else clean
                row = dict(source.loc[query_id])
                row.update({
                    "run": run, "fr_system": system,
                    "scenario": f"mix-query-r{r}", "query_id": query_id,
                })
                out.append(row)
    return pd.DataFrame(out)


def _gallery_mix_endpoints(table: pd.DataFrame) -> pd.DataFrame:
    """The gallery-mix endpoints, copied from the already-scored 2B (r=0) and 2A (r=1)."""
    if "scenario" not in table.columns:
        return pd.DataFrame()
    out = []
    for kind, r in (("2B", 0.0), ("2A", 1.0)):
        block = table[table["scenario"] == kind].copy()
        if block.empty:
            continue
        block["scenario"] = f"mix-gallery-r{r}"
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
