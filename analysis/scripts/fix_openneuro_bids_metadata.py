from __future__ import annotations

import argparse
from pathlib import Path

from ar_analysis.bids_converter.bids import BIDSdata
from ar_analysis.paths import PACKAGE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch generated BIDS metadata for OpenNeuro validation."
    )
    parser.add_argument(
        "bids_root",
        type=Path,
        help="Path to the generated BIDS dataset root.",
    )
    parser.add_argument(
        "--task-name",
        default="AbsPattComp",
        help="BIDS task label used in generated filenames.",
    )
    parser.add_argument(
        "--behav-meta-path",
        type=Path,
        default=PACKAGE_DIR / "bids_converter" / "behav_metadata.yml",
        help="YAML file containing behavioral TSV column metadata.",
    )
    parser.add_argument(
        "--et-meta-path",
        type=Path,
        default=PACKAGE_DIR / "bids_converter" / "et_metadata.yml",
        help="YAML file containing eye-tracking sidecar metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bids_root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {args.bids_root}")

    BIDSdata.patch_openneuro_metadata(
        bids_root=args.bids_root,
        task_name=args.task_name,
        et_meta_path=args.et_meta_path,
        behav_meta_path=args.behav_meta_path,
    )


if __name__ == "__main__":
    main()
