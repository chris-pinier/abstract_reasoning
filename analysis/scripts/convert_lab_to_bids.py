from __future__ import annotations

import argparse
from pathlib import Path

from ar_analysis.analysis_config import Config as c
from ar_analysis.bids_converter.bids import BIDSdata
from ar_analysis.paths import ANALYSIS_DIR, PACKAGE_DIR


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
    parser.add_argument(
        "--include-sourcedata",
        action="store_true",
        help=(
            "Copy the original data directory into sourcedata/ after conversion. "
            "The original directory is left untouched."
        ),
    )
    parser.add_argument(
        "--sourcedata-dir",
        type=Path,
        help=(
            "Optional source directory to copy into sourcedata/. Defaults to data_dir "
            "when --include-sourcedata is used."
        ),
    )
    parser.add_argument(
        "--sourcedata-name",
        help="Destination folder name under sourcedata/. Defaults to source folder name.",
    )
    parser.add_argument(
        "--derivatives-dir",
        type=Path,
        help="Optional preprocessed/derived data directory to copy into derivatives/.",
    )
    parser.add_argument(
        "--pipeline-name",
        default="preprocessed",
        help="Destination folder name under derivatives/.",
    )
    parser.add_argument(
        "--pipeline-version",
        help="Optional derivative pipeline version.",
    )
    parser.add_argument(
        "--pipeline-description",
        help="Optional derivative pipeline description.",
    )
    parser.add_argument(
        "--derivative-source-url",
        default="../..",
        help=(
            "SourceDatasets URL for derivative metadata. Use 'none' to omit it. "
            "Defaults to ../.., the containing raw BIDS dataset."
        ),
    )
    parser.add_argument(
        "--overwrite-extra-data",
        action="store_true",
        help=(
            "Replace existing BIDS-side sourcedata/ or derivatives/ destination "
            "folders. This never modifies the original source directories."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_sourcedata = args.include_sourcedata or args.sourcedata_dir is not None
    if args.sourcedata_name is not None and not include_sourcedata:
        raise ValueError(
            "--sourcedata-name has no effect unless --include-sourcedata or "
            "--sourcedata-dir is provided."
        )
    derivative_source_url = (
        None
        if args.derivative_source_url.lower() == "none"
        else args.derivative_source_url
    )

    errors = BIDSdata.convert_all_subj_data_to_bids(
        data_dir=args.data_dir,
        eye2bids_exe=args.eye2bids_exe,
        et_meta_path=args.et_meta_path,
        behav_meta_path=args.behav_meta_path,
        bids_root=args.bids_root,
        task_name=args.task_name,
        mne_verbose=args.mne_verbose,
        pbar=not args.no_progress,
        include_sourcedata=include_sourcedata,
        sourcedata_dir=args.sourcedata_dir,
        sourcedata_name=args.sourcedata_name,
        derivatives_dir=args.derivatives_dir,
        pipeline_name=args.pipeline_name,
        pipeline_version=args.pipeline_version,
        pipeline_description=args.pipeline_description,
        derivative_source_url=derivative_source_url,
        overwrite_extra_data=args.overwrite_extra_data,
    )
    print(errors)


if __name__ == "__main__":
    main()
