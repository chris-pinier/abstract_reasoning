from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ar_analysis.bids_converter.bids import BIDSdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove macOS metadata artifacts from a selected directory."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to clean recursively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files/directories that would be removed without deleting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {args.directory}")
    if not args.directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {args.directory}")

    artifacts = BIDSdata.iter_macos_metadata_files(args.directory)
    if args.dry_run:
        for artifact in artifacts:
            print(artifact)
        print(f"Found {len(artifacts)} macOS metadata artifact(s).")
        return

    removed = BIDSdata.clean_macos_metadata_files(args.directory)
    print(f"Removed {removed} macOS metadata artifact(s) from {args.directory}")


if __name__ == "__main__":
    main()
