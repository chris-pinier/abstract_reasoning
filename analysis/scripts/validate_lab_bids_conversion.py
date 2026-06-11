from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ar_analysis.analysis_config import Config as c
from ar_analysis.bids_converter.bids import BIDSdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that raw lab sessions were converted to BIDS outputs."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing original subj_* lab folders.",
    )
    parser.add_argument(
        "bids_root",
        type=Path,
        help="Generated BIDS dataset root.",
    )
    parser.add_argument(
        "--task-name",
        default=c.TASK_NAME,
        help="BIDS task label used in generated filenames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report output path. Supports .tsv, .csv, and .json.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all validation rows instead of only non-ok rows.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with code 1 when any row is not ok.",
    )
    return parser.parse_args()


def _write_report(report: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(exist_ok=True, parents=True)
    suffix = output.suffix.lower()
    if suffix == ".tsv":
        report.to_csv(output, sep="\t", index=False)
    elif suffix == ".csv":
        report.to_csv(output, index=False)
    elif suffix == ".json":
        report.to_json(output, orient="records", indent=2)
    else:
        raise ValueError("Report output must end with .tsv, .csv, or .json")


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Original data directory not found: {args.data_dir}")
    if not args.bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {args.bids_root}")

    report = BIDSdata.validate_bids_conversion(
        data_dir=args.data_dir,
        bids_root=args.bids_root,
        task_name=args.task_name,
    )
    problem_rows = report.query("status != 'ok'")

    summary = (
        report.groupby(["datatype", "status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    print(summary.to_string(index=False))

    rows_to_print = report if args.show_all else problem_rows
    if rows_to_print.empty:
        print("All expected source-backed BIDS outputs are present.")
    else:
        pd.set_option("display.max_colwidth", 160)
        print(rows_to_print.to_string(index=False))

    if args.output is not None:
        _write_report(report, args.output)
        print(f"Wrote validation report to {args.output}")

    if args.fail_on_missing and not problem_rows.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
