"""Report AFRBench results by photo label: the camera or emotion of an AFRBench-DF frame.

``python -m afrbench.groups --dataset D --eval E --by camera`` joins the per-query rows that finished evals
write (``E/per_image.jsonl``) to the labels of the dataset's records, and averages every metric within each
value of the label. ``--by`` may be repeated to cross labels. The labels are those the records carry (see
``ImageRecord.labels``) and the record fields ``identity``, ``role``, and ``session``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from afrbench.dataset import BenchDataset, ImageRecord
from afrbench.evaluator.report import INDEX_COLUMNS, KEY_COLUMNS

RECORD_FIELDS = ("identity", "role", "session")
DEFAULT_METRICS = ("recall@25", "AP", "rank1_appearance", "psnr", "ssim", "lpips")


def label_value(record: ImageRecord, key: str) -> str:
    """The value of a label for one record, or ``""`` when the record does not carry it."""
    if key in record.labels:
        return str(record.labels[key])
    if key in RECORD_FIELDS:
        return str(getattr(record, key) or "")
    return ""


def label_keys(dataset: BenchDataset) -> List[str]:
    """Every key ``--by`` accepts for this dataset."""
    return sorted({key for record in dataset.records for key in record.labels} | set(RECORD_FIELDS))


def read_rows(eval_dirs: Sequence[Path]) -> pd.DataFrame:
    """The per-query rows of the given eval folders."""
    rows = []
    for directory in eval_dirs:
        path = Path(directory) / "per_image.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; is {directory} a finished eval folder?")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return pd.DataFrame(rows)


def grouped(
    dataset: BenchDataset,
    rows: pd.DataFrame,
    by: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean metrics per run, FR system, scenario, and value of the ``by`` labels.

    Args:
        dataset: The dataset the evals ran on.
        rows: The per-query rows (:func:`read_rows`).
        by: Label keys to group by; see :func:`label_keys`.
        metrics: Metric columns to report (default: those of :data:`DEFAULT_METRICS` the rows have).

    Returns:
        One row per group, with ``n`` (the rows it averages) and ``<metric>`` means.

    Raises:
        ValueError: If a key in ``by`` is not a label of this dataset.
    """
    unknown = sorted(set(by) - set(label_keys(dataset)))
    if unknown:
        raise ValueError(f"unknown label {unknown}; this dataset has {label_keys(dataset)}")
    table = rows.copy()
    for key in by:
        table[key] = [label_value(dataset.record(query_id), key) for query_id in table["query_id"]]
    present = [c for c in table.columns if c not in INDEX_COLUMNS and c not in KEY_COLUMNS and c not in by]
    wanted = [m for m in (metrics or DEFAULT_METRICS) if m in present]
    if metrics:
        missing = sorted(set(metrics) - set(present))
        if missing:
            raise ValueError(f"no metric {missing} in the rows; they have {sorted(present)}")
    index = [c for c in INDEX_COLUMNS if c in table.columns] + list(by)
    summary = table.groupby(index, dropna=False)[wanted].mean()
    summary.insert(0, "n", table.groupby(index, dropna=False).size())
    # A metric a scenario does not have (visual quality sits in its own rows) is a column of nothing.
    return summary.dropna(axis=1, how="all").reset_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrbench.groups", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, type=Path, help="the dataset the evals ran on")
    parser.add_argument("--eval", required=True, action="append", type=Path, metavar="DIR",
                        help="a finished eval folder, e.g. results/fawkes/eval/minimal (repeatable)")
    parser.add_argument("--by", required=True, action="append", metavar="LABEL",
                        help="label to group by, e.g. camera, emotion (repeatable)")
    parser.add_argument("--metrics", nargs="*", default=None, help=f"metric columns (default {', '.join(DEFAULT_METRICS)})")
    parser.add_argument("--scenario", nargs="*", default=None, help="only these scenarios, e.g. 1A 2A")
    parser.add_argument("--out", type=Path, default=None, help="CSV to write (default: groups.csv in the first eval folder)")
    args = parser.parse_args(argv)

    dataset = BenchDataset(args.dataset)
    rows = read_rows(args.eval)
    if args.scenario:
        rows = rows[rows["scenario"].isin(args.scenario)]
    try:
        table = grouped(dataset, rows, args.by, args.metrics)
    except ValueError as error:
        parser.error(str(error))
    out = args.out or args.eval[0] / "groups.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
