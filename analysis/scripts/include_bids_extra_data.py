from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ar_analysis.bids_converter.bids import BIDSdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy non-BIDS source files and preprocessed outputs into the reserved "
            "sourcedata/ and derivatives/ folders of a BIDS dataset."
        )
    )
    parser.add_argument(
        "bids_root",
        type=Path,
        help="BIDS dataset root to augment.",
    )
    parser.add_argument(
        "--sourcedata-dir",
        type=Path,
        help="Original raw/source directory to copy into sourcedata/.",
    )
    parser.add_argument(
        "--sourcedata-name",
        help="Destination folder name under sourcedata/. Defaults to source folder name.",
    )
    parser.add_argument(
        "--derivatives-dir",
        type=Path,
        help="Preprocessed or derived data directory to copy into derivatives/.",
    )
    parser.add_argument(
        "--pipeline-name",
        default="preprocessed",
        help="Destination folder name under derivatives/.",
    )
    parser.add_argument(
        "--pipeline-version",
        help="Optional version for derivatives/<pipeline-name>/dataset_description.json.",
    )
    parser.add_argument(
        "--pipeline-description",
        help="Optional description for derivatives/<pipeline-name>/dataset_description.json.",
    )
    parser.add_argument(
        "--source-url",
        default="../..",
        help=(
            "SourceDatasets URL for derivative metadata. Use 'none' to omit it. "
            "Defaults to ../.., the containing raw BIDS dataset."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def _validate_source_dir(path: Path | None, label: str) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} path is not a directory: {path}")


def main() -> None:
    args = parse_args()

    if args.sourcedata_dir is None and args.derivatives_dir is None:
        raise ValueError("Provide --sourcedata-dir, --derivatives-dir, or both.")
    if not args.bids_root.exists():
        raise FileNotFoundError(f"BIDS root does not exist: {args.bids_root}")
    if not args.bids_root.is_dir():
        raise NotADirectoryError(f"BIDS root is not a directory: {args.bids_root}")

    _validate_source_dir(args.sourcedata_dir, "sourcedata")
    _validate_source_dir(args.derivatives_dir, "derivatives")

    source_url = None if args.source_url.lower() == "none" else args.source_url

    if args.sourcedata_dir is not None:
        sourcedata_name = BIDSdata._validate_extra_data_destination_name(
            args.sourcedata_name or args.sourcedata_dir.name, "sourcedata"
        )
        sourcedata_destination = args.bids_root / "sourcedata" / sourcedata_name
        if args.dry_run:
            print(f"Would copy {args.sourcedata_dir} -> {sourcedata_destination}")
        else:
            copied = BIDSdata.include_sourcedata(
                source_dir=args.sourcedata_dir,
                bids_root=args.bids_root,
                name=args.sourcedata_name,
                overwrite=args.overwrite,
            )
            print(f"Copied sourcedata to {copied}")

    if args.derivatives_dir is not None:
        pipeline_name = BIDSdata._validate_extra_data_destination_name(
            args.pipeline_name, "derivatives"
        )
        derivatives_destination = args.bids_root / "derivatives" / pipeline_name
        if args.dry_run:
            print(f"Would copy {args.derivatives_dir} -> {derivatives_destination}")
            print(
                "Would ensure derivative metadata at "
                f"{derivatives_destination / 'dataset_description.json'}"
            )
        else:
            copied = BIDSdata.include_derivatives(
                derivatives_dir=args.derivatives_dir,
                bids_root=args.bids_root,
                pipeline_name=args.pipeline_name,
                overwrite=args.overwrite,
                pipeline_version=args.pipeline_version,
                pipeline_description=args.pipeline_description,
                source_url=source_url,
            )
            print(f"Copied derivatives to {copied}")


if __name__ == "__main__":
    main()
