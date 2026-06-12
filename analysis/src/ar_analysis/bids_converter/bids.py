# # * ########################################
# * IMPORTS
# * ########################################
from pathlib import Path
import math
import json
import mne
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
import subprocess
import re
import yaml
import shutil
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# from mne.preprocessing.eyetracking import read_eyelink_calibration
# from icecream import ic
# from pprint import pprint
from loguru import logger

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from ar_analysis.analysis_config import Config as c
from ar_analysis.paths import ANALYSIS_DIR, PACKAGE_DIR
from ar_analysis.utils.analysis_utils import list_contents, read_file

# TODO: add lab keyboard layout to metadata


# * ########################################
# * DATA CONVERTER
# * ########################################
@dataclass
class BIDSdata:
    """Convert lab EEG, behavior, and eye-tracking recordings into a BIDS tree."""

    BIDS_VERSION = "1.11.1"
    DATASET_NAME = "Abstract pattern completion EEG and eye-tracking dataset"
    ET_DATATYPE = "eeg"
    LEGACY_ET_DATATYPE = "func"
    CALIBRATION_ERROR_KEYS = ("AverageCalibrationError", "MaximalCalibrationError")
    TASK_DESCRIPTION = (
        "Participants completed abstract pattern completion problems. Each trial "
        "presented a sequence of abstract icons with one missing item and four "
        "candidate answer choices."
    )
    TASK_INSTRUCTIONS = (
        "Participants selected the answer choice that best completed the abstract "
        "icon sequence using the configured response keys."
    )
    EEG_EVENTS_METADATA = {
        "trial_type": {
            "Description": "Event label assigned during conversion from EEG trigger codes."
        },
        "value": {
            "Description": "Numeric trigger value recorded on the EEG status channel."
        },
        "sample": {
            "Description": "Sample index of the event in the EEG recording.",
            "Units": "samples",
        },
    }
    ET_PHYSIO_COLUMNS = ("timestamp", "x_coordinate", "y_coordinate", "pupil_size")
    ET_PHYSIO_COLUMN_METADATA = {
        "timestamp": {
            "Description": "Timestamp issued by the eye-tracker indexing the continuous recording.",
            "Units": "ms",
            "Origin": "Eye-tracker system startup",
        },
        "x_coordinate": {
            "LongName": "Gaze position (x)",
            "Description": "Gaze position x-coordinate of the recorded eye, in the coordinate units specified in this sidecar.",
        },
        "y_coordinate": {
            "LongName": "Gaze position (y)",
            "Description": "Gaze position y-coordinate of the recorded eye, in the coordinate units specified in this sidecar.",
        },
        "pupil_size": {
            "Description": "Pupil area of the recorded eye as calculated by the eye-tracker.",
            "Units": "arbitrary",
        },
    }
    ET_PHYSIOEVENTS_METADATA = {
        "blink": {
            "Description": "Eye status derived by the eye-tracker.",
            "Levels": {
                "0": "Eye open.",
                "1": "Eye closed.",
            },
        },
        "message": {
            "Description": "String messages logged by the eye-tracker.",
        },
        "trial_type": {
            "Description": "Event type identified by the eye-tracker model or experiment log.",
        },
    }
    BEHAV_NUMERIC_COLUMNS = (
        "trial_onset_time",
        "series_end_time",
        "choice_onset_time",
        "rt",
        "rt_global",
        "blockN",
        "iti",
    )
    OPENNEURO_BIDSIGNORE_PATTERNS = (
        "# eye2bids physiologic event files are BIDS 1.11+, but the OpenNeuro validator may not yet accept them.",
        "*_physioevents.tsv",
        "*_physioevents.tsv.gz",
        "*_physioevents.json",
        "# eye2bids continuous eye-tracking physio files are uploaded but skipped by the OpenNeuro validator.",
        "*_physio.tsv",
        "*_physio.tsv.gz",
        "*_physio.json",
    )

    @staticmethod
    def _format_bids_session_value(value: Any) -> Any:
        """Format session metadata values for writing into a BIDS sessions.tsv file."""
        if value in (None, "", []):
            # return "n/a"
            return ""
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                # return "n/a"
                return ""
            return ", ".join(str(v) for v in value)
        return value

    @staticmethod
    def _get_bids_session_row(sess_dir: Path, sess_id: str) -> dict[str, Any] | None:
        """Read one raw session info JSON file and convert it into a sessions.tsv row."""
        sess_info_files = sorted(sess_dir.glob("*sess_info.json"))
        if not sess_info_files:
            return None

        with open(sess_info_files[0], "r") as f:
            sess_info = json.load(f)

        eye_screen_dist = pd.to_numeric(
            pd.Series([sess_info.get("eye_screen_dist")]), errors="coerce"
        ).iloc[0]

        return {
            "session_id": f"ses-{sess_id}",
            "vision_correction": BIDSdata._format_bids_session_value(
                sess_info.get("vision_correction")
            ),
            "eye": BIDSdata._format_bids_session_value(sess_info.get("eye")),
            "eye_screen_dist": (
                eye_screen_dist if pd.notna(eye_screen_dist) else "n/a"
            ),
            "window_size": BIDSdata._format_bids_session_value(
                sess_info.get("window_size")
            ),
            "img_size": BIDSdata._format_bids_session_value(sess_info.get("img_size")),
            "notes": BIDSdata._format_bids_session_value(sess_info.get("Notes", "")),
        }

    @staticmethod
    def _write_bids_sessions_file(
        bids_root: Path, subj_id: str, session_rows: list[dict[str, Any]]
    ) -> None:
        """Write per-subject sessions.tsv and sessions.json metadata files."""
        if not session_rows:
            return

        subj_bids_dir = bids_root / f"sub-{subj_id}"
        subj_bids_dir.mkdir(exist_ok=True, parents=True)

        sessions_df = (
            pd.DataFrame(session_rows)
            .drop_duplicates(subset=["session_id"], keep="last")
            .sort_values("session_id")
        )

        sessions_tsv = subj_bids_dir / f"sub-{subj_id}_sessions.tsv"
        sessions_df.to_csv(sessions_tsv, sep="\t", index=False, na_rep="n/a")

        sessions_json = subj_bids_dir / f"sub-{subj_id}_sessions.json"
        sessions_meta = {
            "session_id": {"Description": "Session identifier for this participant."},
            "vision_correction": {
                "Description": "Vision correction worn by the participant during the session.",
                "Levels": {
                    "none": "No vision correction was worn during the session.",
                    "glasses": "Participant wore eyeglasses during the session.",
                    "contacts": "Participant wore contact lenses during the session.",
                },
            },
            "eye": {
                "Description": "Eye tracked during the session.",
                "Levels": {
                    "left": "Left eye",
                    "right": "Right eye",
                },
            },
            "eye_screen_dist": {
                "Description": "Distance from the tracked eye to the display during the session.",
                "Units": "mm",
            },
            "window_size": {
                "Description": "Stimulus presentation window size in pixels, encoded as WIDTHxHEIGHT."
            },
            "img_size": {
                "Description": "Stimulus image size in pixels, encoded as WIDTHxHEIGHT."
            },
            "notes": {
                "Description": "Free-text session notes with direct identifying information removed."
            },
        }

        with open(sessions_json, "w") as f:
            json.dump(sessions_meta, f, indent=4)

        # print(f"written at {sessions_tsv}")
        # display(sessions_df)

    @staticmethod
    def _is_nonfinite_number(value: Any) -> bool:
        """Return whether a value is a non-finite numeric scalar."""
        if isinstance(value, bool):
            return False
        try:
            return not math.isfinite(value)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        """Return whether a scalar value should be treated as missing metadata."""
        if value is None or BIDSdata._is_nonfinite_number(value):
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _clean_json_value(value: Any) -> Any:
        """Recursively remove missing values from JSON-serializable metadata."""
        if isinstance(value, dict):
            cleaned = {}
            for key, val in value.items():
                clean_val = BIDSdata._clean_json_value(val)
                if clean_val is not None:
                    cleaned[key] = clean_val
            return cleaned

        if isinstance(value, list):
            cleaned = []
            for val in value:
                clean_val = BIDSdata._clean_json_value(val)
                if clean_val is not None:
                    cleaned.append(clean_val)
            return cleaned

        if BIDSdata._is_missing_scalar(value):
            return None

        return value

    @staticmethod
    def _clean_calibration_error_value(value: Any) -> Any:
        """Clean nested calibration-error metadata while preserving non-empty values."""
        if isinstance(value, list):
            cleaned = []
            for val in value:
                clean_val = BIDSdata._clean_calibration_error_value(val)
                if clean_val is None:
                    continue
                if isinstance(clean_val, list) and len(clean_val) == 0:
                    continue
                cleaned.append(clean_val)
            return cleaned or None

        if BIDSdata._is_missing_scalar(value):
            return None

        return value

    @staticmethod
    def _clean_physio_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
        """Clean eye-tracking physio sidecar metadata before JSON serialization."""
        for key in BIDSdata.CALIBRATION_ERROR_KEYS:
            if key not in sidecar:
                continue

            cleaned_value = BIDSdata._clean_calibration_error_value(sidecar[key])
            if cleaned_value is None:
                sidecar.pop(key)
            else:
                sidecar[key] = cleaned_value

        return BIDSdata._clean_json_value(sidecar)

    @staticmethod
    def _write_json(path: Path, content: dict[str, Any]) -> None:
        """Write indented JSON with strict non-NaN serialization."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(content, f, indent=4, allow_nan=False)
            f.write("\n")

    @staticmethod
    def _call_with_lock(lock: Any | None, func, *args, **kwargs):
        """Run a function while holding an optional multiprocessing-compatible lock."""
        if lock is None:
            return func(*args, **kwargs)

        lock.acquire()
        try:
            return func(*args, **kwargs)
        finally:
            lock.release()

    @staticmethod
    def _write_dataset_description(
        bids_root: Path,
        task_name: str,
        dataset_name: str | None = None,
        bids_version: str | None = None,
    ) -> None:
        """Create or update dataset_description.json for the generated BIDS root."""
        bids_root.mkdir(exist_ok=True, parents=True)
        description_path = bids_root / "dataset_description.json"

        if description_path.exists():
            description = read_file(description_path)
        else:
            description = {}

        description.setdefault("Name", dataset_name or BIDSdata.DATASET_NAME)
        description["BIDSVersion"] = bids_version or BIDSdata.BIDS_VERSION
        description.setdefault("DatasetType", "raw")
        if not isinstance(description.get("GeneratedBy"), list):
            description["GeneratedBy"] = []

        generated_by = description["GeneratedBy"]
        if isinstance(generated_by, list) and not any(
            item.get("Name") == "abstract_reasoning_cleaned BIDS converter"
            for item in generated_by
            if isinstance(item, dict)
        ):
            generated_by.append(
                {
                    "Name": "abstract_reasoning_cleaned BIDS converter",
                    "Description": f"Converts raw EEG, behavioral, and eye-tracking data for task-{task_name}.",
                }
            )

        BIDSdata._write_json(description_path, description)

    @staticmethod
    def _write_openneuro_bidsignore(bids_root: Path) -> None:
        """Append OpenNeuro validator ignore patterns to .bidsignore."""
        bids_root.mkdir(exist_ok=True, parents=True)
        bidsignore_path = bids_root / ".bidsignore"

        if bidsignore_path.exists():
            existing_lines = bidsignore_path.read_text().splitlines()
        else:
            existing_lines = []

        lines = existing_lines.copy()
        for pattern in BIDSdata.OPENNEURO_BIDSIGNORE_PATTERNS:
            if pattern not in lines:
                lines.append(pattern)

        bidsignore_path.write_text("\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _task_metadata(task_name: str) -> dict[str, Any]:
        """Return reusable BIDS task metadata for EEG, behavior, and physio sidecars."""
        return {
            "TaskName": task_name,
            "TaskDescription": BIDSdata.TASK_DESCRIPTION,
            "Instructions": BIDSdata.TASK_INSTRUCTIONS,
        }

    @staticmethod
    def _add_task_metadata(sidecar: dict[str, Any], task_name: str) -> dict[str, Any]:
        """Add task-level metadata to a sidecar without overwriting existing values."""
        for key, value in BIDSdata._task_metadata(task_name).items():
            sidecar.setdefault(key, value)
        return sidecar

    @staticmethod
    def _format_stimulus_presentation(value: Any) -> Any:
        """Format nested stimulus-presentation metadata as a stable string."""
        if not isinstance(value, (dict, list)):
            return value

        if isinstance(value, list):
            return json.dumps(value, sort_keys=True)

        formatted = []
        for key, val in sorted(value.items()):
            if isinstance(val, (dict, list)):
                val = json.dumps(val, sort_keys=True)
            formatted.append(f"{key}: {val}")
        return "; ".join(formatted)

    @staticmethod
    def _is_nonempty_metadata_value(value: Any) -> bool:
        """Return whether a metadata value is present and meaningfully non-empty."""
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return False
        return True

    @staticmethod
    def _metadata_get(metadata: dict[str, Any], *keys: str) -> Any:
        """Return the first non-empty metadata value matching one of the provided keys."""
        for key in keys:
            if key in metadata and BIDSdata._is_nonempty_metadata_value(metadata[key]):
                return metadata[key]
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Convert a value to a finite float, returning None for missing/invalid values."""
        if value in (None, "", "n/a"):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isfinite(value):
            return value
        return None

    @staticmethod
    def _parse_pair(value: Any, scale: float = 1.0) -> list[float] | None:
        """Parse a two-number sequence or string and optionally scale both values."""
        if value in (None, "", "n/a"):
            return None
        if isinstance(value, str):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
            value = [float(number) for number in numbers]
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None

        first = BIDSdata._coerce_float(value[0])
        second = BIDSdata._coerce_float(value[1])
        if first is None or second is None:
            return None

        return [first * scale, second * scale]

    @staticmethod
    def _screen_origin_from_environment(value: Any) -> list[str] | None:
        """Convert environment coordinate metadata into a BIDS ScreenOrigin pair."""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [str(value[0]), str(value[1])]
        if not isinstance(value, str):
            return None

        normalized = value.lower().replace("_", "-").replace(" ", "-")
        parts = [part for part in normalized.split("-") if part]
        vertical = next((part for part in parts if part in {"top", "bottom"}), None)
        horizontal = next((part for part in parts if part in {"left", "right"}), None)
        if vertical and horizontal:
            return [vertical, horizontal]
        return None

    @staticmethod
    def _merge_nonempty_metadata(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge dictionaries while ignoring empty override values."""
        merged = dict(base)
        for key, value in override.items():
            if BIDSdata._is_nonempty_metadata_value(value):
                merged[key] = value
        return merged

    @staticmethod
    def _recorded_eye_from_context(
        sidecar: dict[str, Any],
        metadata: dict[str, Any],
        session_row: dict[str, Any] | None = None,
    ) -> str | None:
        """Infer the recorded eye from existing sidecar, static metadata, or session row."""
        for value in (
            sidecar.get("RecordedEye"),
            BIDSdata._metadata_get(metadata, "RecordedEye"),
            (session_row or {}).get("eye"),
        ):
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower()
            if normalized in {"left", "right", "cyclopean"}:
                return normalized

        return None

    @staticmethod
    def _stimulus_presentation_metadata(
        metadata: dict[str, Any],
        session_row: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build BIDS StimulusPresentation metadata from static and session metadata."""
        session_row = session_row or {}
        stimulus = BIDSdata._metadata_get(metadata, "StimulusPresentation") or {}
        if not isinstance(stimulus, dict):
            stimulus = {}

        screen_distance = BIDSdata._coerce_float(
            BIDSdata._metadata_get(metadata, "ScreenDistance")
        )
        if screen_distance is None:
            screen_distance_mm = BIDSdata._coerce_float(
                session_row.get("eye_screen_dist")
            )
            if screen_distance_mm is not None:
                screen_distance = screen_distance_mm / 1000
        if screen_distance is not None:
            stimulus.setdefault("ScreenDistance", screen_distance)

        screen_origin = BIDSdata._metadata_get(metadata, "ScreenOrigin")
        if screen_origin is None:
            screen_origin = BIDSdata._screen_origin_from_environment(
                BIDSdata._metadata_get(metadata, "EnvironmentCoordinates")
            )
        if screen_origin is not None:
            stimulus.setdefault("ScreenOrigin", screen_origin)

        refresh_rate = BIDSdata._coerce_float(
            BIDSdata._metadata_get(metadata, "ScreenRefreshRate")
        )
        if refresh_rate is not None:
            stimulus.setdefault("ScreenRefreshRate", refresh_rate)

        screen_resolution = BIDSdata._parse_pair(
            BIDSdata._metadata_get(metadata, "ScreenResolution")
            or session_row.get("window_size")
            or c.SCREEN_RESOLUTION
        )
        if screen_resolution is not None:
            stimulus.setdefault(
                "ScreenResolution",
                [
                    int(v) if float(v).is_integer() else v
                    for v in screen_resolution
                ],
            )

        screen_size = BIDSdata._parse_pair(
            BIDSdata._metadata_get(metadata, "ScreenSize")
            or BIDSdata._metadata_get(metadata, "ScreenSizeMeters")
        )
        if screen_size is None:
            screen_size = BIDSdata._parse_pair(
                BIDSdata._metadata_get(metadata, "ScreenSizeMillimeters"),
                scale=0.001,
            )
        if screen_size is not None:
            stimulus.setdefault("ScreenSize", screen_size)

        return stimulus

    @staticmethod
    def _patch_eeg_events_sidecar(
        events_json: Path,
        task_name: str,
        metadata: dict[str, Any] | None = None,
        session_row: dict[str, Any] | None = None,
        openneuro_compat: bool = False,
    ) -> None:
        """Patch an EEG events.json sidecar with task and stimulus metadata."""
        sidecar = read_file(events_json)
        metadata = metadata or {}

        for key, value in BIDSdata.EEG_EVENTS_METADATA.items():
            sidecar.setdefault(key, value)

        stimulus_presentation = BIDSdata._stimulus_presentation_metadata(
            metadata=metadata, session_row=session_row
        )
        if stimulus_presentation:
            existing = sidecar.get("StimulusPresentation")
            if isinstance(existing, dict):
                stimulus_presentation = BIDSdata._merge_nonempty_metadata(
                    stimulus_presentation, existing
                )
            sidecar["StimulusPresentation"] = stimulus_presentation

        if (
            BIDSdata._metadata_get(metadata, "SampleCoordinateSystem")
            == "gaze-on-screen"
        ):
            stimulus = sidecar.get("StimulusPresentation")
            missing = [
                key
                for key in (
                    "ScreenDistance",
                    "ScreenOrigin",
                    "ScreenResolution",
                    "ScreenSize",
                )
                if not isinstance(stimulus, dict)
                or not BIDSdata._is_nonempty_metadata_value(stimulus.get(key))
            ]
            if missing:
                raise ValueError(
                    f"{events_json} is missing required gaze-on-screen "
                    f"StimulusPresentation metadata: {', '.join(missing)}"
                )

        sidecar = BIDSdata._add_task_metadata(sidecar, task_name=task_name)
        BIDSdata._write_json(events_json, sidecar)

    @staticmethod
    def _patch_eeg_sidecar(eeg_json: Path, task_name: str) -> None:
        """Patch an EEG recording sidecar with task metadata."""
        sidecar = read_file(eeg_json)
        sidecar = BIDSdata._add_task_metadata(sidecar, task_name=task_name)
        BIDSdata._write_json(eeg_json, sidecar)

    @staticmethod
    def _patch_beh_sidecar(
        beh_json: Path, task_name: str, column_metadata: dict[str, Any] | None = None
    ) -> None:
        """Patch a behavioral JSON sidecar with task and column metadata."""
        sidecar = read_file(beh_json)

        columns = sidecar.pop("Columns", None)
        if isinstance(columns, dict):
            sidecar.update(columns)

        if column_metadata is not None:
            for key, value in column_metadata.items():
                sidecar.setdefault(key, value)

        if isinstance(sidecar.get("choice"), dict):
            sidecar["choice"].pop("Levels", None)
            sidecar["choice"][
                "Description"
            ] = "Participant's selected stimulus label, or a non-choice status for trials without a valid answer."

        if isinstance(sidecar.get("rt"), dict):
            sidecar["rt"][
                "Description"
            ] = "Response time relative to choice_onset_time. Timed-out or otherwise unavailable responses are encoded as n/a."

        sidecar = BIDSdata._add_task_metadata(sidecar, task_name=task_name)
        BIDSdata._write_json(beh_json, sidecar)

    @staticmethod
    def _prepare_beh_dataframe(raw_behav: pd.DataFrame) -> pd.DataFrame:
        """Normalize behavioral data for BIDS TSV output."""
        behav = raw_behav.copy()
        for column in BIDSdata.BEHAV_NUMERIC_COLUMNS:
            if column in behav.columns:
                behav[column] = pd.to_numeric(behav[column], errors="coerce")
        return behav

    @staticmethod
    def _patch_beh_tsv(beh_tsv: Path) -> None:
        """Rewrite an existing behavioral TSV using the converter's BIDS normalization."""
        behav = pd.read_csv(beh_tsv, sep="\t")
        behav = BIDSdata._prepare_beh_dataframe(behav)
        behav.to_csv(beh_tsv, sep="\t", index=False, na_rep="n/a")

    @staticmethod
    def _event_id_from_eeg_events(eeg_events: Any) -> tuple[dict[int, str], dict[str, int]]:
        """Build per-recording MNE event mappings from observed EEG trigger values."""
        observed_values = sorted({int(value) for value in eeg_events[:, 2]})
        event_desc = {
            value: c.VALID_EVENTS_INV.get(value, f"trigger_{value}")
            for value in observed_values
        }
        event_id = {description: value for value, description in event_desc.items()}
        return event_desc, event_id

    @staticmethod
    def _prepare_bids_root(
        bids_root: Path,
        task_name: str,
        bids_version: str | None = None,
        write_bidsignore: bool = False,
    ) -> None:
        """Ensure root-level BIDS metadata exists, optionally adding .bidsignore."""
        BIDSdata._write_dataset_description(
            bids_root=bids_root, task_name=task_name, bids_version=bids_version
        )
        if not write_bidsignore:
            return
        BIDSdata._write_openneuro_bidsignore(bids_root=bids_root)

    @staticmethod
    def _deduplicate_columns(columns: list[Any]) -> list[str]:
        """Return column names with repeated entries suffixed to make them unique."""
        seen = {}
        deduped = []

        for col in columns:
            name = str(col)
            count = seen.get(name, 0)
            seen[name] = count + 1
            deduped.append(name if count == 0 else f"{name}_{count + 1}")

        return deduped

    @staticmethod
    def _patch_physio_sidecar(
        physio_json: Path,
        metadata: dict[str, Any] | None = None,
        session_row: dict[str, Any] | None = None,
        task_name: str = "AbsPattComp",
    ) -> None:
        """Patch an eye-tracking physio sidecar to meet BIDS 1.11 metadata needs."""
        sidecar = read_file(physio_json)

        metadata = metadata or {}

        sidecar.setdefault("Manufacturer", "SR-Research")
        sidecar.setdefault("PhysioType", "eyetrack")
        sidecar.setdefault("SamplingFrequency", c.ET_SFREQ)
        sidecar.setdefault("StartTime", 0)
        sidecar.setdefault("Columns", list(BIDSdata.ET_PHYSIO_COLUMNS))

        if isinstance(sidecar["Columns"], list):
            sidecar["Columns"] = BIDSdata._deduplicate_columns(sidecar["Columns"])

        recorded_eye = BIDSdata._recorded_eye_from_context(
            sidecar=sidecar, metadata=metadata, session_row=session_row
        )
        if recorded_eye is not None:
            sidecar.setdefault("RecordedEye", recorded_eye)

        software_versions = BIDSdata._metadata_get(
            metadata, "SoftwareVersions", "SoftwareVersion"
        )
        if software_versions is not None:
            sidecar.setdefault("SoftwareVersions", software_versions)

        for key in (
            "ManufacturersModelName",
            "DeviceSerialNumber",
            "SampleCoordinateSystem",
            "EnvironmentCoordinates",
            "EyeTrackerDistance",
            "EyeTrackingMethod",
            "PupilFitMethod",
            "RawDataFilters",
        ):
            if key in metadata and BIDSdata._is_nonempty_metadata_value(metadata[key]):
                sidecar.setdefault(key, metadata[key])

        coordinate_units = (
            BIDSdata._metadata_get(metadata, "CoordinateUnits", "Units") or "pixel"
        )
        for column, column_metadata in BIDSdata.ET_PHYSIO_COLUMN_METADATA.items():
            sidecar.setdefault(column, {})
            if not isinstance(sidecar[column], dict):
                sidecar[column] = {"Description": str(sidecar[column])}
            for key, value in column_metadata.items():
                if not BIDSdata._is_nonempty_metadata_value(sidecar[column].get(key)):
                    sidecar[column][key] = value

        for column in ("x_coordinate", "y_coordinate"):
            if not BIDSdata._is_nonempty_metadata_value(sidecar[column].get("Units")):
                sidecar[column]["Units"] = coordinate_units

        sidecar = BIDSdata._clean_physio_sidecar(sidecar)
        required = (
            "SamplingFrequency",
            "StartTime",
            "Columns",
            "PhysioType",
            "RecordedEye",
            "SampleCoordinateSystem",
        )
        missing = [
            key
            for key in required
            if not BIDSdata._is_nonempty_metadata_value(sidecar.get(key))
        ]
        columns = sidecar.get("Columns")
        if (
            not isinstance(columns, list)
            or tuple(columns[:3]) != BIDSdata.ET_PHYSIO_COLUMNS[:3]
        ):
            missing.append("Columns[0:3]=timestamp,x_coordinate,y_coordinate")
        for column in ("x_coordinate", "y_coordinate"):
            if not isinstance(sidecar.get(column), dict) or not sidecar[column].get(
                "Units"
            ):
                missing.append(f"{column}.Units")
        if missing:
            raise ValueError(
                f"{physio_json} is missing required BIDS 1.11.1 eye-tracking "
                f"metadata: {', '.join(missing)}"
            )
        sidecar = BIDSdata._add_task_metadata(sidecar, task_name=task_name)
        BIDSdata._write_json(physio_json, sidecar)

    @staticmethod
    def _patch_physioevents_sidecar(physioevents_json: Path) -> None:
        """Patch an eye-tracking physioevents sidecar emitted by eye2bids."""
        sidecar = read_file(physioevents_json)

        sidecar.setdefault(
            "Columns", ["onset", "duration", "trial_type", "blink", "message"]
        )
        if isinstance(sidecar["Columns"], list):
            sidecar["Columns"] = BIDSdata._deduplicate_columns(sidecar["Columns"])
            sidecar["Columns"][0] = "onset"

        foreign_index_column = sidecar.pop("ForeignIndexColumn", None)
        sidecar.setdefault("OnsetSource", foreign_index_column or "timestamp")
        sidecar.setdefault(
            "Description", "Messages and model events logged by the eye-tracker."
        )
        for key, value in BIDSdata.ET_PHYSIOEVENTS_METADATA.items():
            sidecar.setdefault(key, value)

        BIDSdata._write_json(physioevents_json, sidecar)

    @staticmethod
    def _remove_uncompressed_eye2bids_tsvs(et_out_dir: Path, bids_base: str) -> None:
        """Remove uncompressed eye2bids TSVs when compressed equivalents are retained."""
        for pattern in (
            f"{bids_base}*_physio.tsv",
            f"{bids_base}*_physioevents.tsv",
        ):
            for path in et_out_dir.glob(pattern):
                path.unlink()

    @staticmethod
    def _remove_orphan_events_sidecar(et_out_dir: Path, bids_base: str) -> None:
        """Remove an events.json sidecar when no matching events.tsv file exists."""
        events_json = et_out_dir / f"{bids_base}_events.json"
        events_tsv = events_json.with_suffix(".tsv")
        events_tsv_gz = events_json.with_suffix(".tsv.gz")

        if (
            events_json.exists()
            and not events_tsv.exists()
            and not events_tsv_gz.exists()
        ):
            events_json.unlink()

    @staticmethod
    def iter_macos_metadata_files(root: Path) -> list[Path]:
        """Return macOS metadata artifacts found recursively under a directory."""
        if not root.exists():
            return []

        paths = []
        for pattern in ("._*", ".DS_Store"):
            for path in root.rglob(pattern):
                if path.is_file():
                    paths.append(path)

        for path in root.rglob("__MACOSX"):
            if path.is_dir():
                paths.append(path)

        return sorted(paths)

    @staticmethod
    def clean_macos_metadata_files(root: Path) -> int:
        """Delete macOS AppleDouble, .DS_Store, and __MACOSX files from a directory."""
        removed = 0
        for path in BIDSdata.iter_macos_metadata_files(root):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1

        if removed:
            logger.info(f"Removed {removed} macOS metadata artifact(s) from {root}")

        return removed

    @staticmethod
    def _remove_macos_metadata_files(bids_root: Path) -> int:
        """Delete macOS AppleDouble, .DS_Store, and __MACOSX files from a BIDS tree."""
        return BIDSdata.clean_macos_metadata_files(bids_root)

    @staticmethod
    def _copy_extra_data_tree(
        source_dir: Path,
        destination_dir: Path,
        overwrite: bool = False,
    ) -> Path:
        """Copy an extra-data directory into a BIDS-reserved destination."""
        source_dir = Path(source_dir)
        destination_dir = Path(destination_dir)

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

        resolved_source = source_dir.resolve()
        resolved_destination = destination_dir.resolve()
        if resolved_destination.is_relative_to(resolved_source):
            raise ValueError(
                f"Refusing to copy {source_dir} into its own subtree: {destination_dir}"
            )

        if destination_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination already exists: {destination_dir}. "
                    "Pass overwrite=True to replace it."
                )
            shutil.rmtree(destination_dir)

        destination_dir.parent.mkdir(exist_ok=True, parents=True)
        shutil.copytree(source_dir, destination_dir)
        BIDSdata.clean_macos_metadata_files(destination_dir)
        return destination_dir

    @staticmethod
    def _validate_extra_data_destination_name(name: str, label: str) -> str:
        """Validate a destination folder name under sourcedata/ or derivatives/."""
        if not name:
            raise ValueError(f"{label} destination name must not be empty.")

        name_path = Path(name)
        if name_path.is_absolute() or name_path.name != name or name in {".", ".."}:
            raise ValueError(
                f"{label} destination name must be a single folder name, got: {name}"
            )

        return name

    @staticmethod
    def include_sourcedata(
        source_dir: Path,
        bids_root: Path,
        name: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Copy original non-BIDS source files into ``sourcedata/``.

        BIDS reserves ``sourcedata/`` for files that predate BIDS conversion,
        such as the original lab EEG, eye-tracking, behavioral, and session-info
        files.
        """
        source_dir = Path(source_dir)
        bids_root = Path(bids_root)
        destination_name = BIDSdata._validate_extra_data_destination_name(
            name or source_dir.name, "sourcedata"
        )

        return BIDSdata._copy_extra_data_tree(
            source_dir=source_dir,
            destination_dir=bids_root / "sourcedata" / destination_name,
            overwrite=overwrite,
        )

    @staticmethod
    def _write_derivative_dataset_description(
        derivative_root: Path,
        pipeline_name: str,
        pipeline_version: str | None = None,
        pipeline_description: str | None = None,
        source_url: str | None = "../..",
    ) -> None:
        """Create or patch required BIDS derivative dataset metadata."""
        description_path = derivative_root / "dataset_description.json"
        if description_path.exists():
            with open(description_path) as f:
                description = json.load(f)
        else:
            description = {}

        description.setdefault("Name", f"{pipeline_name} derivatives")
        description["BIDSVersion"] = BIDSdata.BIDS_VERSION
        description["DatasetType"] = "derivative"

        generated_by = description.get("GeneratedBy")
        if not isinstance(generated_by, list):
            generated_by = []

        if generated_by and isinstance(generated_by[0], dict):
            generated_by[0].setdefault("Name", pipeline_name)
            if pipeline_version is not None:
                generated_by[0]["Version"] = pipeline_version
            if pipeline_description is not None:
                generated_by[0]["Description"] = pipeline_description
        else:
            entry = {"Name": pipeline_name}
            if pipeline_version is not None:
                entry["Version"] = pipeline_version
            if pipeline_description is not None:
                entry["Description"] = pipeline_description
            generated_by.insert(0, entry)

        description["GeneratedBy"] = generated_by
        if source_url is not None:
            source_datasets = description.get("SourceDatasets")
            if not isinstance(source_datasets, list):
                source_datasets = []
            if not any(
                isinstance(item, dict) and item.get("URL") == source_url
                for item in source_datasets
            ):
                source_datasets.append({"URL": source_url})
            description["SourceDatasets"] = source_datasets

        BIDSdata._write_json(description_path, description)

    @staticmethod
    def include_derivatives(
        derivatives_dir: Path,
        bids_root: Path,
        pipeline_name: str = "preprocessed",
        overwrite: bool = False,
        pipeline_version: str | None = None,
        pipeline_description: str | None = None,
        source_url: str | None = "../..",
    ) -> Path:
        """Copy preprocessed or otherwise derived files into ``derivatives/``.

        The copied directory is treated as one derivative dataset and receives a
        minimal ``dataset_description.json`` when one is missing.
        """
        pipeline_name = BIDSdata._validate_extra_data_destination_name(
            pipeline_name, "derivatives"
        )

        destination = BIDSdata._copy_extra_data_tree(
            source_dir=derivatives_dir,
            destination_dir=Path(bids_root) / "derivatives" / pipeline_name,
            overwrite=overwrite,
        )
        BIDSdata._write_derivative_dataset_description(
            derivative_root=destination,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_description=pipeline_description,
            source_url=source_url,
        )
        return destination

    @staticmethod
    def _move_legacy_eye_tracking_outputs(
        bids_root: Path, task_name: str = "AbsPattComp"
    ) -> None:
        """Move legacy func eye-tracking outputs into the EEG datatype directory."""
        for legacy_dir in bids_root.glob(
            f"sub-*/ses-*/{BIDSdata.LEGACY_ET_DATATYPE}"
        ):
            subj_dir = legacy_dir.parent.parent
            sess_dir = legacy_dir.parent
            if not subj_dir.name.startswith("sub-") or not sess_dir.name.startswith(
                "ses-"
            ):
                continue

            bids_base = f"{subj_dir.name}_{sess_dir.name}_task-{task_name}"
            et_dir = sess_dir / BIDSdata.ET_DATATYPE
            et_dir.mkdir(exist_ok=True, parents=True)

            for path in legacy_dir.glob(f"{bids_base}*_physio*"):
                target = et_dir / path.name
                if target.exists():
                    logger.warning(
                        f"Skipping legacy eye-tracking file migration because target exists: {target}"
                    )
                    continue
                path.replace(target)

    @staticmethod
    def _rename_eye2bids_outputs(
        et_out_dir: Path,
        et_file: Path,
        bids_base: str,
    ) -> None:
        """Rename eye2bids outputs from EDF-derived stems to BIDS stems."""
        for path in sorted(et_out_dir.glob(f"{et_file.stem}*")):
            suffix = path.name.removeprefix(et_file.stem)
            if not (suffix.startswith("_recording-") or suffix == "_events.json"):
                continue

            target = et_out_dir / f"{bids_base}{suffix}"
            if path != target:
                path.replace(target)

    @staticmethod
    def _postprocess_eye2bids_output(
        et_out_dir: Path,
        et_file: Path,
        subj_id: str,
        sess_id: str,
        task_name: str,
        metadata: dict[str, Any] | None = None,
        session_row: dict[str, Any] | None = None,
    ) -> None:
        """Rename, patch, and clean eye2bids outputs for one recording."""
        bids_base = f"sub-{subj_id}_ses-{sess_id}_task-{task_name}"

        BIDSdata._rename_eye2bids_outputs(
            et_out_dir=et_out_dir,
            et_file=et_file,
            bids_base=bids_base,
        )

        for physio_json in et_out_dir.glob(f"{bids_base}*_physio.json"):
            BIDSdata._patch_physio_sidecar(
                physio_json,
                metadata=metadata,
                session_row=session_row,
                task_name=task_name,
            )

        for physioevents_json in et_out_dir.glob(f"{bids_base}*_physioevents.json"):
            BIDSdata._patch_physioevents_sidecar(physioevents_json)

        BIDSdata._remove_uncompressed_eye2bids_tsvs(et_out_dir, bids_base=bids_base)
        BIDSdata._remove_orphan_events_sidecar(et_out_dir, bids_base=bids_base)

    @staticmethod
    def _find_eyelink_file(sess_dir: Path) -> Path:
        """Return the single EyeLink EDF file in a raw session directory."""
        et_files = sorted(
            path
            for path in list_contents(sess_dir, incl="file", recurs=False)
            if path.suffix.lower() == ".edf"
        )
        if not et_files:
            raise FileNotFoundError(f"No EyeLink EDF file found in {sess_dir}")
        if len(et_files) > 1:
            raise ValueError(
                "Multiple EyeLink EDF files found in "
                f"{sess_dir}: {', '.join(str(path) for path in et_files)}"
            )
        return et_files[0]

    @staticmethod
    def _read_bids_session_rows(
        bids_root: Path,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Read generated sessions.tsv files into a lookup keyed by subject/session."""
        session_rows = {}
        for sessions_tsv in bids_root.glob("sub-*/sub-*_sessions.tsv"):
            subj_id = sessions_tsv.parent.name
            try:
                sessions = pd.read_csv(sessions_tsv, sep="\t").replace({pd.NA: None})
            except Exception as exc:
                logger.warning(
                    f"Could not read sessions metadata from {sessions_tsv}: {exc}"
                )
                continue

            for row in sessions.to_dict("records"):
                session_id = row.get("session_id")
                if BIDSdata._is_missing_scalar(session_id):
                    continue
                session_id = str(session_id)
                if session_id.isdigit():
                    session_id = f"ses-{int(session_id):02d}"
                session_rows[(subj_id, session_id)] = {
                    key: (None if BIDSdata._is_missing_scalar(value) else value)
                    for key, value in row.items()
                }

        return session_rows

    @staticmethod
    def patch_bids_1_11_metadata(
        bids_root: Path,
        task_name: str = "AbsPattComp",
        et_meta_path: Path | None = None,
        behav_meta_path: Path | None = None,
        openneuro_compat: bool = False,
    ) -> None:
        """Patch an existing BIDS tree with BIDS 1.11 EEG/behavior/eye-tracking metadata."""
        BIDSdata._remove_macos_metadata_files(bids_root)
        BIDSdata._prepare_bids_root(
            bids_root=bids_root,
            task_name=task_name,
            bids_version=BIDSdata.BIDS_VERSION,
            write_bidsignore=openneuro_compat,
        )

        metadata = read_file(et_meta_path) if et_meta_path is not None else {}
        session_rows = BIDSdata._read_bids_session_rows(bids_root)

        for et_dir in bids_root.glob(f"sub-*/ses-*/{BIDSdata.ET_DATATYPE}"):
            subj_dir = et_dir.parent.parent
            sess_dir = et_dir.parent
            if not subj_dir.name.startswith("sub-") or not sess_dir.name.startswith(
                "ses-"
            ):
                continue

            bids_base = f"{subj_dir.name}_{sess_dir.name}_task-{task_name}"
            session_row = session_rows.get((subj_dir.name, sess_dir.name))

            for physio_json in et_dir.glob(f"{bids_base}*_physio.json"):
                BIDSdata._patch_physio_sidecar(
                    physio_json,
                    metadata=metadata,
                    session_row=session_row,
                    task_name=task_name,
                )

            for physioevents_json in et_dir.glob(f"{bids_base}*_physioevents.json"):
                BIDSdata._patch_physioevents_sidecar(physioevents_json)

            for events_json in et_dir.glob(f"{bids_base}_events.json"):
                BIDSdata._patch_eeg_events_sidecar(
                    events_json,
                    task_name=task_name,
                    metadata=metadata,
                    session_row=session_row,
                    openneuro_compat=openneuro_compat,
                )

            for eeg_json in et_dir.glob(f"{bids_base}_eeg.json"):
                BIDSdata._patch_eeg_sidecar(eeg_json, task_name=task_name)

        behav_metadata = read_file(behav_meta_path) if behav_meta_path else None
        for beh_tsv in bids_root.glob(f"sub-*/ses-*/beh/*_task-{task_name}_beh.tsv"):
            BIDSdata._patch_beh_tsv(beh_tsv)

        for beh_json in bids_root.glob(f"sub-*/ses-*/beh/*_task-{task_name}_beh.json"):
            BIDSdata._patch_beh_sidecar(
                beh_json, task_name=task_name, column_metadata=behav_metadata
            )

    @staticmethod
    def patch_openneuro_metadata(
        bids_root: Path,
        task_name: str = "AbsPattComp",
        et_meta_path: Path | None = None,
        behav_meta_path: Path | None = None,
    ) -> None:
        """Patch metadata and .bidsignore for OpenNeuro upload compatibility."""
        BIDSdata.patch_bids_1_11_metadata(
            bids_root=bids_root,
            task_name=task_name,
            et_meta_path=et_meta_path,
            behav_meta_path=behav_meta_path,
            openneuro_compat=True,
        )

    @staticmethod
    def finalize_bids_dataset(
        bids_root: Path,
        task_name: str = "AbsPattComp",
        et_meta_path: Path | None = None,
        behav_meta_path: Path | None = None,
    ) -> None:
        """Run final whole-dataset cleanup and metadata patching after conversion."""
        BIDSdata._remove_macos_metadata_files(bids_root)
        BIDSdata._prepare_bids_root(bids_root=bids_root, task_name=task_name)
        BIDSdata._move_legacy_eye_tracking_outputs(
            bids_root=bids_root, task_name=task_name
        )

        BIDSdata.patch_bids_1_11_metadata(
            bids_root=bids_root,
            task_name=task_name,
            et_meta_path=et_meta_path,
            behav_meta_path=behav_meta_path,
        )

        for et_dir in bids_root.glob(f"sub-*/ses-*/{BIDSdata.ET_DATATYPE}"):
            subj_dir = et_dir.parent.parent
            sess_dir = et_dir.parent
            if not subj_dir.name.startswith("sub-") or not sess_dir.name.startswith(
                "ses-"
            ):
                continue

            bids_base = f"{subj_dir.name}_{sess_dir.name}_task-{task_name}"

            BIDSdata._remove_uncompressed_eye2bids_tsvs(et_dir, bids_base=bids_base)
            BIDSdata._remove_orphan_events_sidecar(et_dir, bids_base=bids_base)

        BIDSdata._remove_macos_metadata_files(bids_root)

    @staticmethod
    def convert_subj_data_to_bids(
        subj_dir: Path,
        eye2bids_exe: Path,
        et_meta_path: Path,
        behav_meta_path: Path,
        bids_root: Path,
        task_name="AbsPattComp",
        mne_verbose: str = "WARNING",
        pbar: bool = True,
        bids_write_lock: Any | None = None,
    ):
        """Convert all available sessions for one raw lab subject directory.

        EEG BDF files are written with MNE-BIDS, behavioral CSV files are normalized
        to BIDS TSV/JSON pairs, and EyeLink EDF files are converted through
        eye2bids. Conversion continues across modalities/sessions and returns a
        list of failed subject-session modality tuples.
        """
        # * --- CONFIGURATION PATHS ---
        if not et_meta_path.exists():
            raise FileNotFoundError(
                f"Eye-tracking BIDS metadata file not found at '{et_meta_path}'"
            )
        if not behav_meta_path.exists():
            raise FileNotFoundError(
                f"Behavioral BIDS metadata file not found at '{behav_meta_path}'"
            )
        if not eye2bids_exe.exists():
            raise FileNotFoundError(
                f"eye2bids executable not found at '{eye2bids_exe}'"
            )
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject directory not found at '{subj_dir}'")

        subj_id = subj_dir.name.split("_")[1]
        sess_dirs = list_contents(subj_dir, recurs=False, incl="folder")
        eye_metadata = read_file(et_meta_path)
        behav_metadata = read_file(behav_meta_path)

        BIDSdata._call_with_lock(
            bids_write_lock,
            BIDSdata._prepare_bids_root,
            bids_root=bids_root,
            task_name=task_name,
        )

        errors = []
        session_rows = []

        # TODO: add logger and hide outputs of eye2bids from the console
        # * Added tqdm back to the loop so tqdm.write formats nicely
        for sess_dir in tqdm(
            sess_dirs, desc=f"Converting Subj {subj_id}", disable=not pbar
        ):
            sess_N = int(sess_dir.name.split("_")[1])
            sess_id = f"{sess_N:02}"

            session_row = BIDSdata._get_bids_session_row(
                sess_dir=sess_dir, sess_id=sess_id
            )
            if session_row is not None:
                session_rows.append(session_row)

            # * ########################################################################
            # * EEG
            # * ########################################################################
            try:
                eeg_fpath = list_contents(sess_dir, reg=r".+\.bdf$")[0]
                raw_eeg = mne.io.read_raw_bdf(eeg_fpath)
                ch_names = [ch for ch in raw_eeg.ch_names if "status" not in ch.lower()]

                ch_types = []
                for ch in ch_names:
                    _ch = ch.lower()
                    if "emg" in _ch:
                        ch_types.append("emg")
                    elif "eog" in _ch:
                        ch_types.append("eog")
                    else:
                        ch_types.append("eeg")

                raw_eeg.set_channel_types(dict(zip(ch_names, ch_types)))
                # raw_eeg.set_montage(c.EEG_MONTAGE)

                event_id = None
                try:
                    eeg_events = mne.find_events(
                        raw=raw_eeg,
                        min_duration=0,
                        initial_event=False,
                        shortest_event=1,
                        uint_cast=True,
                        verbose=mne_verbose,
                    )
                except ValueError as exc:
                    if "Could not find any of the events" not in str(exc):
                        raise
                    tqdm.write(
                        f"No EEG trigger events found for subj_{subj_id} "
                        f"sess_{sess_id}; writing EEG without events.tsv."
                    )
                    eeg_events = None

                if eeg_events is not None and len(eeg_events) > 0:
                    event_desc, event_id = BIDSdata._event_id_from_eeg_events(
                        eeg_events
                    )
                    annotations = mne.annotations_from_events(
                        events=eeg_events,
                        sfreq=raw_eeg.info["sfreq"],
                        event_desc=event_desc,
                        verbose=mne_verbose,
                    )
                    raw_eeg.set_annotations(annotations, verbose=mne_verbose)

                eeg_bids_path = BIDSPath(
                    task=task_name,
                    subject=subj_id,
                    session=sess_id,
                    root=bids_root,
                    datatype="eeg",
                    suffix="eeg",
                )

                BIDSdata._call_with_lock(
                    bids_write_lock,
                    write_raw_bids,
                    raw_eeg,
                    eeg_bids_path,
                    event_id=event_id,
                    overwrite=True,
                )
                # raw_eeg, eeg_bids_path, event_id=c.VALID_EVENTS, overwrite=True
                # TODO: consider filling the following missing fields in .+_eeg.json
                # "PowerLineFrequency": "n/a",
                # "SoftwareFilters": "n/a",
                # "EEGReference": "n/a",
                # "EEGGround": "n/a",
                events_json = (
                    eeg_bids_path.copy()
                    .update(suffix="events", extension=".json")
                    .fpath
                )
                if events_json.exists():
                    BIDSdata._patch_eeg_events_sidecar(
                        events_json,
                        task_name=task_name,
                        metadata=eye_metadata,
                        session_row=session_row,
                    )

                eeg_json = (
                    eeg_bids_path.copy().update(suffix="eeg", extension=".json").fpath
                )
                if eeg_json.exists():
                    BIDSdata._patch_eeg_sidecar(eeg_json, task_name=task_name)

            except Exception as e:
                errors.append((f"sub-{subj_id}_ses{sess_id}", "eeg"))
                tqdm.write(
                    f"An error occurred; Skipping EEG BIDS conversion for subj_{subj_id} sess_{sess_id}.\n{e}"
                )

            # * ########################################################################
            # * Behavioral
            # * ########################################################################
            try:
                behav_fpath = list_contents(sess_dir, reg=r".+-behav\.csv$")[0]
                raw_behav = pd.read_csv(behav_fpath, index_col=0)

                beh_bids_path = BIDSPath(
                    task=task_name,
                    subject=subj_id,
                    session=sess_id,
                    root=bids_root,
                    suffix="beh",
                    datatype="beh",
                    extension=".tsv",
                )
                beh_bids_path.directory.mkdir(exist_ok=True, parents=True)
                behav = BIDSdata._prepare_beh_dataframe(raw_behav)
                behav.to_csv(beh_bids_path.fpath, sep="\t", index=False, na_rep="n/a")

                # Write a JSON sidecar describing your columns
                metadata = BIDSdata._task_metadata(task_name)
                metadata.update(behav_metadata)
                json_path = beh_bids_path.copy().update(extension=".json").fpath

                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                    f.write("\n")
            except Exception as e:
                errors.append((f"sub-{subj_id}_ses{sess_id}", "behav"))
                tqdm.write(
                    f"An error occurred; Skipping Behav BIDS conversion for subj_{subj_id} sess_{sess_id}.\n{e}"
                )

            # * ########################################################################
            # * Eye Tracking
            # * ########################################################################
            try:
                et_file = BIDSdata._find_eyelink_file(sess_dir)

                # Eye tracking is stored as physio data alongside the associated EEG recording.
                et_out_dir = (
                    bids_root
                    / f"sub-{subj_id}"
                    / f"ses-{sess_id}"
                    / BIDSdata.ET_DATATYPE
                )
                et_out_dir.mkdir(exist_ok=True, parents=True)

                command = [
                    str(eye2bids_exe),
                    "--input_file",
                    str(et_file),
                    "--metadata_file",
                    str(et_meta_path),
                    "--output_dir",
                    str(et_out_dir),
                ]

                # Run conversion
                result = subprocess.run(command, capture_output=True, text=True)
                is_success = result.returncode == 0

                if not is_success:
                    # If the command failed, log it and manually trigger the except block
                    stdout = result.stdout.strip()
                    stderr = result.stderr.strip()
                    details = "\n".join(
                        part
                        for part in (
                            f"Command: {' '.join(command)}",
                            f"stdout:\n{stdout}" if stdout else "",
                            f"stderr:\n{stderr}" if stderr else "",
                        )
                        if part
                    )
                    raise RuntimeError(f"eye2bids failed:\n{details}")

                tqdm.write(
                    f"Successfully converted ET data for subj_{subj_id} "
                    f"sess_{sess_id}: {et_file.name}"
                )

                # Clean up intermediate .asc files ONLY if successful
                intermediate_files = list_contents(
                    sess_dir, reg=r"(_events|_samples)\.asc$"
                )
                for f in intermediate_files:
                    f.unlink()

                BIDSdata._postprocess_eye2bids_output(
                    et_out_dir=et_out_dir,
                    et_file=et_file,
                    subj_id=subj_id,
                    sess_id=sess_id,
                    task_name=task_name,
                    metadata=eye_metadata,
                    session_row=session_row,
                )

            except Exception as e:
                errors.append((f"sub-{subj_id}_ses{sess_id}", "et"))
                tqdm.write(
                    f"An error occurred; Skipping ET BIDS conversion for subj_{subj_id} sess_{sess_id}.\n{e}"
                )

        BIDSdata._write_bids_sessions_file(
            bids_root=bids_root, subj_id=subj_id, session_rows=session_rows
        )
        BIDSdata._remove_macos_metadata_files(
            bids_root=bids_root / f"sub-{subj_id}"
        )

        if len(errors) > 0:
            pass  # TODO

        # print(session_rows)

        return errors

    @staticmethod
    def _convert_subj_data_to_bids_captured(
        subj_dir: Path,
        eye2bids_exe: Path,
        et_meta_path: Path,
        behav_meta_path: Path,
        bids_root: Path,
        task_name: str,
        mne_verbose: str,
        bids_write_lock: Any,
    ) -> tuple[list, str, str]:
        """Convert one subject while capturing noisy child-process console output."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            errors = BIDSdata.convert_subj_data_to_bids(
                subj_dir,
                eye2bids_exe=eye2bids_exe,
                et_meta_path=et_meta_path,
                behav_meta_path=behav_meta_path,
                bids_root=bids_root,
                task_name=task_name,
                mne_verbose=mne_verbose,
                pbar=False,
                bids_write_lock=bids_write_lock,
            )

        return errors, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _write_worker_output(subj_name: str, stdout: str, stderr: str) -> None:
        """Print captured worker output as one grouped block."""
        output = "\n".join(part.rstrip() for part in (stdout, stderr) if part.strip())
        if output:
            tqdm.write(f"--- Captured output for {subj_name} ---\n{output}")

    @staticmethod
    def convert_all_subj_data_to_bids(
        data_dir: Path,
        eye2bids_exe: Path,
        et_meta_path: Path,
        behav_meta_path: Path,
        bids_root: Path,
        task_name="AbsPattComp",
        mne_verbose: str = "WARNING",
        pbar: bool = True,
        include_sourcedata: bool = False,
        sourcedata_dir: Path | None = None,
        sourcedata_name: str | None = None,
        derivatives_dir: Path | None = None,
        pipeline_name: str = "preprocessed",
        pipeline_version: str | None = None,
        pipeline_description: str | None = None,
        derivative_source_url: str | None = "../..",
        overwrite_extra_data: bool = False,
        n_jobs: int = 1,
        openneuro_compat: bool = False,
    ):
        """Convert every subj_* directory and optionally attach extra BIDS data.

        ``include_sourcedata`` copies the original source directory into
        ``sourcedata/`` after conversion. ``derivatives_dir`` copies a
        preprocessed output directory into ``derivatives/<pipeline_name>/``.
        ``n_jobs`` controls subject-level multiprocessing; shared BIDS writes
        are locked when ``n_jobs`` is greater than one. ``openneuro_compat``
        adds validator compatibility metadata such as .bidsignore patterns.
        """
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")
        if not eye2bids_exe.exists():
            raise FileNotFoundError(
                f"eye2bids executable not found at '{eye2bids_exe}'. "
                "Install eye2bids in the active environment or pass --eye2bids-exe."
            )

        subj_dirs = list_contents(data_dir, reg="subj.+", recurs=False)
        logger.info(f"Using eye2bids executable: {eye2bids_exe}")
        BIDSdata._prepare_bids_root(bids_root=bids_root, task_name=task_name)

        errors = {}
        if not subj_dirs:
            logger.warning(f"No subject directories found in {data_dir}")

        if n_jobs == 1 or not subj_dirs:
            for subj_dir in tqdm(subj_dirs, disable=not pbar):
                subj_errors = BIDSdata.convert_subj_data_to_bids(
                    subj_dir,
                    eye2bids_exe=eye2bids_exe,
                    et_meta_path=et_meta_path,
                    behav_meta_path=behav_meta_path,
                    bids_root=bids_root,
                    task_name=task_name,
                    mne_verbose=mne_verbose,
                    pbar=pbar,
                )
                errors[subj_dir.name] = subj_errors
        else:
            max_workers = min(n_jobs, len(subj_dirs))
            with Manager() as manager:
                bids_write_lock = manager.Lock()
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            BIDSdata._convert_subj_data_to_bids_captured,
                            subj_dir,
                            eye2bids_exe=eye2bids_exe,
                            et_meta_path=et_meta_path,
                            behav_meta_path=behav_meta_path,
                            bids_root=bids_root,
                            task_name=task_name,
                            mne_verbose=mne_verbose,
                            bids_write_lock=bids_write_lock,
                        ): subj_dir.name
                        for subj_dir in subj_dirs
                    }

                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        disable=not pbar,
                        desc="Converting subjects",
                    ):
                        subj_name = futures[future]
                        try:
                            subj_errors, stdout, stderr = future.result()
                            errors[subj_name] = subj_errors
                            if subj_errors:
                                BIDSdata._write_worker_output(
                                    subj_name, stdout=stdout, stderr=stderr
                                )
                        except Exception as exc:
                            errors[subj_name] = [(subj_name, "subject", str(exc))]
                            tqdm.write(
                                f"An error occurred; Skipping subject conversion for {subj_name}.\n{exc}"
                            )

        BIDSdata.finalize_bids_dataset(
            bids_root=bids_root,
            task_name=task_name,
            et_meta_path=et_meta_path,
            behav_meta_path=behav_meta_path,
        )

        if openneuro_compat:
            BIDSdata.patch_openneuro_metadata(
                bids_root=bids_root,
                task_name=task_name,
                et_meta_path=et_meta_path,
                behav_meta_path=behav_meta_path,
            )

        if include_sourcedata:
            BIDSdata.include_sourcedata(
                source_dir=sourcedata_dir or data_dir,
                bids_root=bids_root,
                name=sourcedata_name,
                overwrite=overwrite_extra_data,
            )

        if derivatives_dir is not None:
            BIDSdata.include_derivatives(
                derivatives_dir=derivatives_dir,
                bids_root=bids_root,
                pipeline_name=pipeline_name,
                overwrite=overwrite_extra_data,
                pipeline_version=pipeline_version,
                pipeline_description=pipeline_description,
                source_url=derivative_source_url,
            )

        return errors

    @staticmethod
    def _format_validation_paths(paths: list[Path | str]) -> str:
        """Format validation paths for a compact tabular report."""
        return "; ".join(str(path) for path in paths)

    @staticmethod
    def _session_row_exists(sessions_tsv: Path, session_id: str) -> bool:
        """Return whether a BIDS sessions.tsv contains the expected session row."""
        if not sessions_tsv.exists():
            return False
        try:
            sessions = pd.read_csv(sessions_tsv, sep="\t", dtype=str)
        except Exception:
            return False
        if "session_id" not in sessions.columns:
            return False
        return session_id in set(sessions["session_id"].dropna())

    @staticmethod
    def validate_bids_conversion(
        data_dir: Path,
        bids_root: Path,
        task_name: str = "AbsPattComp",
    ) -> pd.DataFrame:
        """Validate that each raw lab session has the expected converted BIDS files.

        The validation is source-driven: every ``subj_* / sess_*`` directory in
        ``data_dir`` is checked for source EEG, behavioral, eye-tracking, and
        session-info files, then compared against the expected BIDS outputs.
        """
        logger.info(
            "Running source-to-BIDS data type mapping validation "
            f"for {data_dir} -> {bids_root}"
        )
        rows = []
        subj_dirs = sorted(
            path
            for path in Path(data_dir).glob("subj_*")
            if path.is_dir() and re.fullmatch(r"subj_\d+", path.name)
        )

        for subj_dir in subj_dirs:
            subj_id = f"{int(subj_dir.name.split('_')[1]):02}"
            sess_dirs = sorted(
                path
                for path in subj_dir.glob("sess_*")
                if path.is_dir() and re.fullmatch(r"sess_\d+", path.name)
            )

            for sess_dir in sess_dirs:
                sess_id = f"{int(sess_dir.name.split('_')[1]):02}"
                bids_subj_dir = Path(bids_root) / f"sub-{subj_id}"
                bids_sess_dir = bids_subj_dir / f"ses-{sess_id}"
                bids_base = f"sub-{subj_id}_ses-{sess_id}_task-{task_name}"

                checks = {
                    "behavior": {
                        "source": sorted(sess_dir.glob("*-behav.csv")),
                        "required": [
                            bids_sess_dir / "beh" / f"{bids_base}_beh.tsv",
                            bids_sess_dir / "beh" / f"{bids_base}_beh.json",
                        ],
                        "patterns": [],
                    },
                    "eeg": {
                        "source": sorted(sess_dir.glob("*.bdf"))
                        + sorted(sess_dir.glob("*.BDF")),
                        "required": [
                            bids_sess_dir / "eeg" / f"{bids_base}_eeg.bdf",
                            bids_sess_dir / "eeg" / f"{bids_base}_eeg.json",
                            bids_sess_dir / "eeg" / f"{bids_base}_channels.tsv",
                            bids_sess_dir / f"sub-{subj_id}_ses-{sess_id}_scans.tsv",
                        ],
                        "patterns": [],
                    },
                    "eyetracking": {
                        "source": sorted(sess_dir.glob("*.edf"))
                        + sorted(sess_dir.glob("*.EDF")),
                        "required": [],
                        "patterns": [
                            bids_sess_dir
                            / "eeg"
                            / f"{bids_base}_recording-eye*_physio.tsv.gz",
                            bids_sess_dir
                            / "eeg"
                            / f"{bids_base}_recording-eye*_physio.json",
                            bids_sess_dir
                            / "eeg"
                            / f"{bids_base}_recording-eye*_physioevents.tsv.gz",
                            bids_sess_dir
                            / "eeg"
                            / f"{bids_base}_recording-eye*_physioevents.json",
                        ],
                    },
                    "session_info": {
                        "source": sorted(sess_dir.glob("*sess_info.json")),
                        "required": [
                            bids_subj_dir / f"sub-{subj_id}_sessions.tsv",
                            bids_subj_dir / f"sub-{subj_id}_sessions.json",
                        ],
                        "patterns": [],
                    },
                }

                for datatype, check in checks.items():
                    source_files = check["source"]
                    present_bids_files = [
                        path for path in check["required"] if path.exists()
                    ]
                    missing_bids = [
                        path for path in check["required"] if not path.exists()
                    ]

                    for pattern in check["patterns"]:
                        matches = sorted(pattern.parent.glob(pattern.name))
                        present_bids_files.extend(matches)
                        if not matches:
                            missing_bids.append(str(pattern))

                    if datatype == "session_info":
                        sessions_tsv = bids_subj_dir / f"sub-{subj_id}_sessions.tsv"
                        if not BIDSdata._session_row_exists(
                            sessions_tsv, f"ses-{sess_id}"
                        ):
                            missing_bids.append(f"{sessions_tsv}: row ses-{sess_id}")

                    if not source_files:
                        status = "source_missing"
                    elif missing_bids:
                        status = "missing_bids"
                    else:
                        status = "ok"

                    rows.append(
                        {
                            "subject": f"sub-{subj_id}",
                            "session": f"ses-{sess_id}",
                            "datatype": datatype,
                            "status": status,
                            "source_count": len(source_files),
                            "bids_count": len(present_bids_files),
                            "source_files": BIDSdata._format_validation_paths(
                                source_files
                            ),
                            "bids_files": BIDSdata._format_validation_paths(
                                present_bids_files
                            ),
                            "missing_bids": BIDSdata._format_validation_paths(
                                missing_bids
                            ),
                        }
                    )

        columns = [
            "subject",
            "session",
            "datatype",
            "status",
            "source_count",
            "bids_count",
            "source_files",
            "bids_files",
            "missing_bids",
        ]
        report = pd.DataFrame(rows, columns=columns)
        problem_count = int((report["status"] != "ok").sum()) if not report.empty else 0
        outcome = "PASSED" if problem_count == 0 else "FAILED"
        logger.info(
            f"Data type mapping validation {outcome}: "
            f"{len(report)} source-backed data type rows checked, "
            f"{problem_count} non-ok row(s)."
        )
        return report

    @staticmethod
    def read_bids():
        """
        Scratch helper for manually reading and inspecting a generated BIDS dataset.

        See: https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
        """
        from mne_bids import (
            BIDSPath,
            find_matching_paths,
            get_entity_vals,
            make_report,
            print_dir_tree,
            read_raw_bids,
        )

        bids_root = "/Volumes/SSD-512Go/PhD Data/experiment1/data/Lab-BIDS2"

        sessions = get_entity_vals(bids_root, "session", ignore_sessions="on")

        datatype = "eeg"
        extensions = [".bdf", ".tsv"]  # ignore .json files
        bids_paths = find_matching_paths(
            bids_root, datatypes=datatype, sessions=sessions, extensions=extensions
        )
        session = "05"
        bids_path = BIDSPath(root=bids_root, session=session, datatype=datatype)

        task = "AbsPattComp"
        # suffix = "eeg"
        suffix = None
        subject = "01"

        bids_path = bids_path.update(subject=subject, task=task, suffix=suffix)

        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw2 = mne.io.read_raw_bdf(bids_path.fpath)

        # raw3 = mne.io.read_raw_bdf(
        #     "/Volumes/SSD-512Go/PhD Data/experiment1/data/Lab/subj_01/sess_01/cp0101.bdf"
        # )
        eeg_events_files = list_contents(
            Path(bids_root), incl="file", reg=r"eeg/.+_events.tsv"
        )
        # eeg_events_files
        eeg_events = pd.concat([pd.read_csv(f, sep="\t") for f in eeg_events_files])
        eeg_events[["trial_type", "value"]].value_counts()

        # set(list(zip(eeg_events['trial_type'].tolist(), eeg_events['value'].tolist())))

        # self = HumanSessData(
        #     subj_N=1,
        #     sess_N=1,
        #     data_dir="/Volumes/SSD-512Go/PhD Data/experiment1/data/Lab-BIDS2",
        #     preprocessed_dir="./TEST/prepro",
        #     export_dir="./TEST/export",
        # )

    @staticmethod
    def validation():
        """Placeholder for manual validation checks comparing raw and BIDS trees."""
        # extract_subj_N = lambda x: int(re.search(r"sub.*(\d{2})", str(x))[1])

        # subj_folders = list_contents(data_dir, incl="folder", recurs=False)
        # counts_original = {}
        # for subj_folder in subj_folders:
        #     counts_original[extract_subj_N(subj_folder)] = len(
        #         list_contents(subj_folder, incl="file")
        #     )

        # subj_folders = list_contents(bids_root, incl="folder", recurs=False)
        # counts_bids = {}
        # for subj_folder in subj_folders:
        #     counts_bids[extract_subj_N(subj_folder)] = len(
        #         list_contents(subj_folder, incl="file")
        #     )

        # counts_bids = pd.DataFrame.from_dict(
        #     counts_bids, orient="index", columns=["bids"]
        # )
        # counts_original = pd.DataFrame.from_dict(
        #     counts_original, orient="index", columns=["original"]
        # )
        # counts_df = pd.concat([counts_bids, counts_original], axis=1)

        # counts_df["bids"] = round(counts_df["bids"] / counts_df["bids"].max(), 2) * 100
        # counts_df["original"] = (
        #     round(counts_df["original"] / counts_df["original"].max(), 2) * 100
        # )

        # counts_df["same"] = counts_df["original"] == counts_df["bids"]

        # # def highlight_false(x, color="red"):
        # #     return np.where(x == False, f"color: {color};", None)

        # # counts_df.style.apply(highlight_false)

        raise NotImplementedError


def main():
    """Run the legacy hard-coded conversion entry point."""
    task_name = c.TASK_NAME

    MAIN_SAVE_DIR = Path("/Volumes/SSD-512Go/PhD Data/experiment1/")
    data_dir = MAIN_SAVE_DIR / "data/Lab-OLD"
    bids_root = MAIN_SAVE_DIR / "data/Lab/raw2"

    eye2bids_exe = ANALYSIS_DIR / ".venv/bin/eye2bids"
    et_meta_path = PACKAGE_DIR / "bids_converter/et_metadata.yml"
    behav_meta_path = PACKAGE_DIR / "bids_converter/behav_metadata.yml"

    mne_verbose = "WARNING"
    pbar = True

    errors = BIDSdata.convert_all_subj_data_to_bids(
        data_dir=data_dir,
        eye2bids_exe=eye2bids_exe,
        et_meta_path=et_meta_path,
        behav_meta_path=behav_meta_path,
        bids_root=bids_root,
        task_name=task_name,
        mne_verbose=mne_verbose,
        pbar=pbar,
    )

    print(errors)
    
if __name__ == "__main__":
    # main()
    pass
