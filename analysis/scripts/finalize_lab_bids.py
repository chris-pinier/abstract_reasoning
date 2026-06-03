from __future__ import annotations

import argparse
from pathlib import Path

from ar_analysis.analysis_config import Config as c
from ar_analysis.bids_converter.bids import BIDSdata
from ar_analysis.paths import ANALYSIS_DIR, PACKAGE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize lab BIDS metadata, optionally repairing one subject first."
    )
    parser.add_argument(
        "bids_root",
        type=Path,
        help="Path to the generated BIDS dataset root.",
    )
    parser.add_argument(
        "--repair-subject-dir",
        type=Path,
        help="Optional original subj_* directory to reconvert before finalizing.",
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
        help="Verbosity passed to MNE functions when repairing a subject.",
    )
    parser.add_argument(
        "--openneuro-compat",
        action="store_true",
        help="Patch metadata for the OpenNeuro validator, including .bidsignore.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.repair_subject_dir is not None:
        BIDSdata.convert_subj_data_to_bids(
            subj_dir=args.repair_subject_dir,
            eye2bids_exe=args.eye2bids_exe,
            et_meta_path=args.et_meta_path,
            behav_meta_path=args.behav_meta_path,
            bids_root=args.bids_root,
            task_name=args.task_name,
            mne_verbose=args.mne_verbose,
        )

    if args.openneuro_compat:
        BIDSdata.patch_openneuro_metadata(
            bids_root=args.bids_root,
            task_name=args.task_name,
            et_meta_path=args.et_meta_path,
            behav_meta_path=args.behav_meta_path,
        )
    else:
        BIDSdata.finalize_bids_dataset(
            bids_root=args.bids_root,
            task_name=args.task_name,
            et_meta_path=args.et_meta_path,
            behav_meta_path=args.behav_meta_path,
        )

    print(f"Finalized BIDS dataset at {args.bids_root}")


if __name__ == "__main__":
    main()
