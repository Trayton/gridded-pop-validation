"""Command-line interface for gridded-pop-validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .metrics import accuracy_summary, evaluate_by_group, evaluate_population_change


def _read_csv_columns(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gridded-pop-validate",
        description="Validate gridded population aggregates against census counts.",
    )
    ap.add_argument("input_csv", type=Path, help="Input CSV with observed and predicted population columns")
    ap.add_argument("--obs-col", default="census_pop", help="Census population column name (default: census_pop)")
    ap.add_argument("--pred-col", default="gridded_pop", help="Gridded population column name (default: gridded_pop)")
    ap.add_argument("--group-col", default=None, help="Optional grouping column (e.g., region or stratum)")
    ap.add_argument("--obs-t0-col", default=None, help="Optional baseline-period census column for change evaluation")
    ap.add_argument("--pred-t0-col", default=None, help="Optional baseline-period gridded column for change evaluation")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Optional output JSON report path")
    args = ap.parse_args(argv)

    rows = _read_csv_columns(args.input_csv)
    if not rows:
        raise SystemExit(f"Input CSV is empty: {args.input_csv}")

    obs = [float(r[args.obs_col]) for r in rows]
    pred = [float(r[args.pred_col]) for r in rows]

    report: dict[str, object] = {
        "input": str(args.input_csv),
        "overall": accuracy_summary(obs, pred),
    }
    if args.group_col:
        groups = [r[args.group_col] for r in rows]
        report["by_group"] = evaluate_by_group(obs, pred, groups)
    if args.obs_t0_col and args.pred_t0_col:
        obs0 = [float(r[args.obs_t0_col]) for r in rows]
        pred0 = [float(r[args.pred_t0_col]) for r in rows]
        report["change"] = evaluate_population_change(obs0, obs, pred0, pred)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
