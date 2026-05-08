"""Aggregate per-run `metrics.json` files into a single CSV with mean/std
across seeds for each (run_group) — used to produce Table 1 of the paper."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

_RUN_PATTERN = re.compile(r"^(?P<group>.+)_seed(?P<seed>\d+)$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(args.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = _RUN_PATTERN.match(run_dir.name)
        if not m:
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text())
        if args.split not in data:
            continue
        for k, v in data[args.split].items():
            rows[m["group"]][k].append(float(v))

    records = []
    for group, metric_dict in rows.items():
        rec: dict[str, float | str | int] = {"group": group, "n_seeds": max(len(vs) for vs in metric_dict.values())}
        for metric, values in metric_dict.items():
            rec[f"{metric}_mean"] = mean(values)
            rec[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        records.append(rec)

    df = pd.DataFrame(records).sort_values("group")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
