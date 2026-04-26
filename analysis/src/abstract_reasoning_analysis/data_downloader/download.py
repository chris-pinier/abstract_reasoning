from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SubjectLike = int | str
PathLike = Path | str


MODALITY_ALIASES = {
    "beh": "beh",
    "behav": "beh",
    "behavior": "beh",
    "behavioral": "beh",
    "behaviour": "beh",
    "behavioural": "beh",
    "eeg": "eeg",
    "ieeg": "ieeg",
    "meg": "meg",
    "anat": "anat",
    "func": "func",
    "dwi": "dwi",
    "fmap": "fmap",
    "perf": "perf",
    "pet": "pet",
    "motion": "motion",
    "micr": "micr",
    "nirs": "nirs",
    "eye": "func",
    "eyetrack": "func",
    "eyetracking": "func",
    "eye-tracking": "func",
    "physio": "func",
}

ROOT_METADATA_PATTERNS = (
    "/*.json",
    "/*.tsv",
    "/*.bidsignore",
    "/README",
    "/CHANGES",
    "/LICENSE",
)


@dataclass(frozen=True)
class OpenNeuroDownloadPlan:
    dataset: str
    target_dir: Path | None
    tag: str | None
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    verify_hash: bool
    verify_size: bool
    max_retries: int
    max_concurrent_downloads: int
    metadata_timeout: float


def parse_openneuro_dataset(link: str) -> tuple[str, str | None]:
    """Return ``(dataset_id, tag)`` from an OpenNeuro dataset id or URL."""
    dataset_match = re.search(r"\b(ds\d{6,})\b", link)
    if dataset_match is None:
        raise ValueError(
            "`link` must be an OpenNeuro dataset id or URL containing an id "
            "like 'ds000001'."
        )

    tag_match = re.search(r"/versions/([^/?#]+)", link)
    return dataset_match.group(1), tag_match.group(1) if tag_match else None


def _as_tuple(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_entity(
    value: SubjectLike,
    *,
    prefix: str,
    width: int,
) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"Empty {prefix} value is not allowed.")

    if raw.startswith(f"{prefix}-"):
        return raw

    if raw.isdigit():
        raw = raw.zfill(max(width, len(raw)))

    return f"{prefix}-{raw}"


def _normalize_entities(
    values: SubjectLike | Sequence[SubjectLike] | None,
    *,
    prefix: str,
    width: int,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, int)):
        values = (values,)

    return tuple(
        _normalize_entity(value, prefix=prefix, width=width) for value in values
    )


def normalize_modalities(modalities: str | Iterable[str] | None) -> tuple[str, ...]:
    if modalities is None:
        return ()
    if isinstance(modalities, str):
        modalities = (modalities,)

    normalized = []
    for modality in modalities:
        key = modality.strip().lower()
        if not key:
            raise ValueError("Empty modality value is not allowed.")
        normalized.append(MODALITY_ALIASES.get(key, key))

    return tuple(dict.fromkeys(normalized))


def build_openneuro_include_patterns(
    *,
    subjects: SubjectLike | Sequence[SubjectLike] | None = None,
    modalities: str | Iterable[str] | None = None,
    sessions: SubjectLike | Sequence[SubjectLike] | None = None,
    include_metadata: bool = True,
    subject_width: int = 2,
    session_width: int = 2,
) -> tuple[str, ...]:
    """Build OpenNeuro glob include patterns for BIDS subjects/modalities.

    Patterns use openneuro-py's glob semantics. ``**`` matches across directory
    boundaries, so the same pattern handles both flat and sessioned BIDS layouts.
    """
    subject_ids = _normalize_entities(
        subjects, prefix="sub", width=subject_width
    ) or ("sub-*",)
    session_ids = _normalize_entities(sessions, prefix="ses", width=session_width)
    datatypes = normalize_modalities(modalities)

    patterns: list[str] = []
    if include_metadata:
        patterns.extend(ROOT_METADATA_PATTERNS)

    for subject in subject_ids:
        if include_metadata:
            patterns.extend((f"{subject}/*.json", f"{subject}/*.tsv"))

        if not datatypes and not session_ids:
            patterns.append(subject)
            continue

        if session_ids:
            for session in session_ids:
                if include_metadata:
                    patterns.extend(
                        (
                            f"{subject}/{session}/*.json",
                            f"{subject}/{session}/*.tsv",
                        )
                    )
                if datatypes:
                    patterns.extend(
                        f"{subject}/{session}/{datatype}/**" for datatype in datatypes
                    )
                else:
                    patterns.append(f"{subject}/{session}/**")
            continue

        for datatype in datatypes:
            patterns.extend(
                (
                    f"{subject}/{datatype}/**",
                    f"{subject}/**/{datatype}/**",
                )
            )

    return tuple(dict.fromkeys(patterns))


def make_openneuro_download_plan(
    link: str,
    download_dir: PathLike | None = None,
    *,
    subjects: SubjectLike | Sequence[SubjectLike] | None = None,
    modalities: str | Iterable[str] | None = None,
    sessions: SubjectLike | Sequence[SubjectLike] | None = None,
    tag: str | None = None,
    full_dataset: bool = False,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    include_metadata: bool = True,
    verify_hash: bool = True,
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 5,
    metadata_timeout: float = 15.0,
) -> OpenNeuroDownloadPlan:
    dataset, tag_from_link = parse_openneuro_dataset(link)
    tag = tag if tag is not None else tag_from_link

    if full_dataset and any((subjects, modalities, sessions, include)):
        raise ValueError(
            "`full_dataset=True` cannot be combined with subjects, modalities, "
            "sessions, or include patterns."
        )

    include_patterns = list(_as_tuple(include))
    if not full_dataset and any((subjects, modalities, sessions)):
        include_patterns.extend(
            build_openneuro_include_patterns(
                subjects=subjects,
                modalities=modalities,
                sessions=sessions,
                include_metadata=include_metadata,
            )
        )

    target_dir = Path(download_dir).expanduser() if download_dir is not None else None

    return OpenNeuroDownloadPlan(
        dataset=dataset,
        target_dir=target_dir,
        tag=tag,
        include=tuple(dict.fromkeys(include_patterns)),
        exclude=_as_tuple(exclude),
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        metadata_timeout=metadata_timeout,
    )


def download_bids_dataset(
    link: str,
    download_dir: PathLike | None = None,
    *,
    subjects: SubjectLike | Sequence[SubjectLike] | None = None,
    modalities: str | Iterable[str] | None = None,
    sessions: SubjectLike | Sequence[SubjectLike] | None = None,
    tag: str | None = None,
    full_dataset: bool = False,
    include: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    include_metadata: bool = True,
    verify_hash: bool = True,
    verify_size: bool = True,
    max_retries: int = 5,
    max_concurrent_downloads: int = 5,
    metadata_timeout: float = 15.0,
    dry_run: bool = False,
) -> OpenNeuroDownloadPlan:
    """Download a BIDS dataset from OpenNeuro.

    Examples
    --------
    Download the full dataset::

        download_bids_dataset("ds123456", "data/bids", full_dataset=True)

    Download only EEG and behavioral data for subjects 1 and 2::

        download_bids_dataset(
            "https://openneuro.org/datasets/ds123456",
            "data/bids",
            subjects=[1, 2],
            modalities=["eeg", "behavioral"],
        )
    """
    plan = make_openneuro_download_plan(
        link=link,
        download_dir=download_dir,
        subjects=subjects,
        modalities=modalities,
        sessions=sessions,
        tag=tag,
        full_dataset=full_dataset,
        include=include,
        exclude=exclude,
        include_metadata=include_metadata,
        verify_hash=verify_hash,
        verify_size=verify_size,
        max_retries=max_retries,
        max_concurrent_downloads=max_concurrent_downloads,
        metadata_timeout=metadata_timeout,
    )

    if dry_run:
        return plan

    try:
        import openneuro as on
    except ImportError as exc:
        raise ImportError(
            "openneuro-py is required. Install it in the analysis environment "
            "with `uv add openneuro-py`."
        ) from exc

    on.download(
        dataset=plan.dataset,
        tag=plan.tag,
        target_dir=plan.target_dir,
        include=plan.include or None,
        exclude=plan.exclude or None,
        verify_hash=plan.verify_hash,
        verify_size=plan.verify_size,
        max_retries=plan.max_retries,
        max_concurrent_downloads=plan.max_concurrent_downloads,
        metadata_timeout=plan.metadata_timeout,
    )
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download full or partial BIDS datasets from OpenNeuro."
    )
    parser.add_argument("link", help="OpenNeuro dataset id or URL, e.g. ds123456.")
    parser.add_argument(
        "--download-dir",
        "--target-dir",
        dest="download_dir",
        help="Directory to download into. Defaults to openneuro-py's dataset folder.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject labels or numbers, e.g. 1 2 sub-03.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        help="BIDS datatypes or aliases, e.g. eeg behavioral.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        help="Session labels or numbers, e.g. 1 2 ses-03.",
    )
    parser.add_argument("--tag", help="OpenNeuro snapshot tag/version.")
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Download the complete dataset. Cannot be combined with filters.",
    )
    parser.add_argument(
        "--include",
        action="append",
        help="Additional openneuro-py include pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help="Additional openneuro-py exclude pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not add root/session metadata include patterns for partial downloads.",
    )
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--no-verify-size", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-concurrent-downloads", type=int, default=5)
    parser.add_argument("--metadata-timeout", type=float, default=15.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned OpenNeuro download arguments without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = download_bids_dataset(
        link=args.link,
        download_dir=args.download_dir,
        subjects=args.subjects,
        modalities=args.modalities,
        sessions=args.sessions,
        tag=args.tag,
        full_dataset=args.full_dataset,
        include=args.include,
        exclude=args.exclude,
        include_metadata=not args.no_metadata,
        verify_hash=not args.no_verify_hash,
        verify_size=not args.no_verify_size,
        max_retries=args.max_retries,
        max_concurrent_downloads=args.max_concurrent_downloads,
        metadata_timeout=args.metadata_timeout,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(plan)


if __name__ == "__main__":
    main()
