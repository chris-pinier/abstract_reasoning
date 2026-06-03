from __future__ import annotations

import argparse
from pathlib import Path

from abstract_reasoning_analysis.analysis_config import Config as c
from abstract_reasoning_analysis.bids_converter.bids import BIDSdata
from abstract_reasoning_analysis.paths import ANALYSIS_DIR, PACKAGE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the full lab dataset to BIDS."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing original subj_* lab folders.",
    )
    parser.add_argument(
        "bids_root",
        type=Path,
        help="Output BIDS dataset root. Prefer an empty or new directory.",
    )
    parser.add_argument(
        "--eye2bids-exe",
        type=Path,
        default=ANALYSIS_DIR / ".venv" / "bin" / "eye2bids",
        help="Path to the eye2bids executable.",
    )
    parser.add_argument(
        "--et-meta-path",
        type=Path,
        default=PACKAGE_DIR / "bids_converter" / "et_metadata.yml",
        help="YAML file containing eye-tracking metadata.",
    )
    parser.add_argument(
        "--behav-meta-path",
        type=Path,
        default=PACKAGE_DIR / "bids_converter" / "behav_metadata.yml",
        help="YAML file containing behavioral column metadata.",
    )
    parser.add_argument(
        "--task-name",
        default=c.TASK_NAME,
        help="BIDS task label.",
    )
    parser.add_argument(
        "--mne-verbose",
        default="WARNING",
        help="Verbosity passed to MNE functions.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    errors = BIDSdata.convert_all_subj_data_to_bids(
        data_dir=args.data_dir,
        eye2bids_exe=args.eye2bids_exe,
        et_meta_path=args.et_meta_path,
        behav_meta_path=args.behav_meta_path,
        bids_root=args.bids_root,
        task_name=args.task_name,
        mne_verbose=args.mne_verbose,
        pbar=not args.no_progress,
    )
    print(errors)


if __name__ == "__main__":
    main()
