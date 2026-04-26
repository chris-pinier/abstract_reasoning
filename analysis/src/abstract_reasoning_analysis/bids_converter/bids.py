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

# from mne.preprocessing.eyetracking import read_eyelink_calibration
# from icecream import ic
# from pprint import pprint
from loguru import logger

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from abstract_reasoning_analysis.analysis_config import Config as c
from abstract_reasoning_analysis.paths import ANALYSIS_DIR, PACKAGE_DIR
from abstract_reasoning_analysis.utils.analysis_utils import list_contents, read_file

# TODO: add lab keyboard layout to metadata


# * ########################################
# * DATA CONVERTER
# * ########################################
@dataclass
class BIDSdata:
    BIDS_VERSION = "1.11.1"
    DATASET_NAME = "Abstract pattern completion EEG and eye-tracking dataset"
    ET_DATATYPE = "eeg"
    LEGACY_ET_DATATYPE = "func"
    CALIBRATION_ERROR_KEYS = ("AverageCalibrationError", "MaximalCalibrationError")
    OPENNEURO_BIDSIGNORE_PATTERNS = (
        "# eye2bids physiologic event files are BIDS 1.11+, but the OpenNeuro validator may not yet accept them.",
        "*_physioevents.tsv",
        "*_physioevents.tsv.gz",
        "*_physioevents.json",
        "# eye2bids continuous physio files must remain compressed in BIDS.",
        "*_physio.tsv",
    )

    @staticmethod
    def _format_bids_session_value(value: Any) -> Any:
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
        if isinstance(value, bool):
            return False
        try:
            return not math.isfinite(value)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        if value is None or BIDSdata._is_nonfinite_number(value):
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _clean_json_value(value: Any) -> Any:
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
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(content, f, indent=4, allow_nan=False)
            f.write("\n")

    @staticmethod
    def _write_dataset_description(
        bids_root: Path,
        task_name: str,
        dataset_name: str | None = None,
    ) -> None:
        bids_root.mkdir(exist_ok=True, parents=True)
        description_path = bids_root / "dataset_description.json"

        if description_path.exists():
            description = read_file(description_path)
        else:
            description = {}

        description.setdefault("Name", dataset_name or BIDSdata.DATASET_NAME)
        description.setdefault("BIDSVersion", BIDSdata.BIDS_VERSION)
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
    def _prepare_bids_root(bids_root: Path, task_name: str) -> None:
        BIDSdata._write_dataset_description(bids_root=bids_root, task_name=task_name)
        BIDSdata._write_openneuro_bidsignore(bids_root=bids_root)

    @staticmethod
    def _deduplicate_columns(columns: list[Any]) -> list[str]:
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
        physio_json: Path, metadata: dict[str, Any] | None = None
    ) -> None:
        sidecar = read_file(physio_json)

        metadata = metadata or {}

        sidecar.setdefault("Manufacturer", "SR-Research")
        sidecar.setdefault("PhysioType", "eyetrack")
        sidecar.setdefault("SamplingFrequency", c.ET_SFREQ)
        sidecar.setdefault("StartTime", 0)
        sidecar.setdefault(
            "Columns", ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"]
        )

        if isinstance(sidecar["Columns"], list):
            sidecar["Columns"] = BIDSdata._deduplicate_columns(sidecar["Columns"])

        for key in (
            "Units",
            "SampleCoordinateSystem",
            "EnvironmentCoordinates",
            "SoftwareVersion",
            "ScreenAOIDefinition",
            "EyeCameraSettings",
            "EyeTrackerDistance",
            "FeatureDetectionSettings",
            "GazeMappingSettings",
            "RawDataFilters",
        ):
            if key in metadata and metadata[key] not in (None, ""):
                sidecar.setdefault(key, metadata[key])

        sidecar = BIDSdata._clean_physio_sidecar(sidecar)
        BIDSdata._write_json(physio_json, sidecar)

    @staticmethod
    def _patch_physioevents_sidecar(physioevents_json: Path) -> None:
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

        BIDSdata._write_json(physioevents_json, sidecar)

    @staticmethod
    def _remove_uncompressed_eye2bids_tsvs(et_out_dir: Path, bids_base: str) -> None:
        for pattern in (
            f"{bids_base}*_physio.tsv",
            f"{bids_base}*_physioevents.tsv",
        ):
            for path in et_out_dir.glob(pattern):
                path.unlink()

    @staticmethod
    def _remove_orphan_events_sidecar(et_out_dir: Path, bids_base: str) -> None:
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
    def _move_legacy_eye_tracking_outputs(
        bids_root: Path, task_name: str = "AbsPattComp"
    ) -> None:
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
    ) -> None:
        bids_base = f"sub-{subj_id}_ses-{sess_id}_task-{task_name}"

        BIDSdata._rename_eye2bids_outputs(
            et_out_dir=et_out_dir,
            et_file=et_file,
            bids_base=bids_base,
        )

        for physio_json in et_out_dir.glob(f"{bids_base}*_physio.json"):
            BIDSdata._patch_physio_sidecar(physio_json, metadata=metadata)

        for physioevents_json in et_out_dir.glob(f"{bids_base}*_physioevents.json"):
            BIDSdata._patch_physioevents_sidecar(physioevents_json)

        BIDSdata._remove_uncompressed_eye2bids_tsvs(et_out_dir, bids_base=bids_base)
        BIDSdata._remove_orphan_events_sidecar(et_out_dir, bids_base=bids_base)

    @staticmethod
    def finalize_bids_dataset(
        bids_root: Path,
        task_name: str = "AbsPattComp",
        et_meta_path: Path | None = None,
    ) -> None:
        BIDSdata._prepare_bids_root(bids_root=bids_root, task_name=task_name)
        BIDSdata._move_legacy_eye_tracking_outputs(
            bids_root=bids_root, task_name=task_name
        )

        metadata = read_file(et_meta_path) if et_meta_path is not None else {}

        for et_dir in bids_root.glob(f"sub-*/ses-*/{BIDSdata.ET_DATATYPE}"):
            subj_dir = et_dir.parent.parent
            sess_dir = et_dir.parent
            if not subj_dir.name.startswith("sub-") or not sess_dir.name.startswith(
                "ses-"
            ):
                continue

            bids_base = f"{subj_dir.name}_{sess_dir.name}_task-{task_name}"

            for physio_json in et_dir.glob(f"{bids_base}*_physio.json"):
                BIDSdata._patch_physio_sidecar(physio_json, metadata=metadata)

            for physioevents_json in et_dir.glob(f"{bids_base}*_physioevents.json"):
                BIDSdata._patch_physioevents_sidecar(physioevents_json)

            BIDSdata._remove_uncompressed_eye2bids_tsvs(et_dir, bids_base=bids_base)
            BIDSdata._remove_orphan_events_sidecar(et_dir, bids_base=bids_base)

    @staticmethod
    def convert_subj_data_to_bids(
        subj_dir: Path,
        eye2bids_exe: Path,
        et_meta_path: Path,
        behav_meta_path: Path,
        bids_root: Path,
        task_name="AbsPattComp",
        mne_verbose: str = "WARNING",
    ):
        # * --- CONFIGURATION PATHS ---
        assert et_meta_path.exists(), (
            f"Eye-tracking BIDS metadata file not found at '{et_meta_path}'"
        )
        assert behav_meta_path.exists(), (
            f"Behavioral BIDS metadata file not found at '{behav_meta_path}'"
        )
        assert eye2bids_exe.exists(), (
            f"eye2bids executable not found at '{eye2bids_exe}'"
        )
        assert subj_dir.exists(), f"Subject directory not found at '{subj_dir}'"

        subj_id = subj_dir.name.split("_")[1]
        sess_dirs = list_contents(subj_dir, recurs=False, incl="folder")
        eye_metadata = read_file(et_meta_path)
        behav_metadata = read_file(behav_meta_path)

        BIDSdata._prepare_bids_root(bids_root=bids_root, task_name=task_name)

        errors = []
        session_rows = []

        # TODO: add logger and hide outputs of eye2bids from the console
        # * Added tqdm back to the loop so tqdm.write formats nicely
        for sess_dir in tqdm(sess_dirs, desc=f"Converting Subj {subj_id}"):
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

                # Detecting events
                eeg_events = mne.find_events(
                    raw=raw_eeg,
                    min_duration=0,
                    initial_event=False,
                    shortest_event=1,
                    uint_cast=True,
                    verbose=mne_verbose,
                )

                # Get annotations from events and add them to the raw data
                annotations = mne.annotations_from_events(
                    events=eeg_events,
                    sfreq=raw_eeg.info["sfreq"],
                    event_desc=c.VALID_EVENTS_INV,
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

                write_raw_bids(
                    raw_eeg,
                    eeg_bids_path,
                    event_id=None,
                    overwrite=True,
                    # raw_eeg, eeg_bids_path, event_id=c.VALID_EVENTS, overwrite=True
                )
                # TODO: consider filling the following missing fields in .+_eeg.json
                # "PowerLineFrequency": "n/a",
                # "SoftwareFilters": "n/a",
                # "EEGReference": "n/a",
                # "EEGGround": "n/a",

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
                raw_behav.to_csv(beh_bids_path.fpath, sep="\t", index=False)

                # Write a JSON sidecar describing your columns
                metadata = {
                    "TaskName": task_name,
                    "Columns": behav_metadata,
                }
                json_path = beh_bids_path.copy().update(extension=".json").fpath

                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as e:
                errors.append((f"sub-{subj_id}_ses{sess_id}", "behav"))
                tqdm.write(
                    f"An error occurred; Skipping Behav BIDS conversion for subj_{subj_id} sess_{sess_id}.\n{e}"
                )

            # * ########################################################################
            # * Eye Tracking
            # * ########################################################################
            try:
                et_file = list_contents(sess_dir, reg=r".+\.EDF$")[0]

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
                    raise RuntimeError(f"eye2bids failed: {result.stderr}")

                tqdm.write(f"Successfully converted ET data for {et_file.name}")

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
                )

            except Exception as e:
                errors.append((f"sub-{subj_id}_ses{sess_id}", "et"))
                tqdm.write(
                    f"An error occurred; Skipping ET BIDS conversion for subj_{subj_id} sess_{sess_id}.\n{e}"
                )

        BIDSdata._write_bids_sessions_file(
            bids_root=bids_root, subj_id=subj_id, session_rows=session_rows
        )

        if len(errors) > 0:
            pass  # TODO

        # print(session_rows)

        return errors

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
    ):

        subj_dirs = list_contents(data_dir, reg="subj.+", recurs=False)
        BIDSdata._prepare_bids_root(bids_root=bids_root, task_name=task_name)

        errors = {}

        for subj_dir in tqdm(subj_dirs, disable=not pbar):
            subj_errors = BIDSdata.convert_subj_data_to_bids(
                subj_dir,
                eye2bids_exe=eye2bids_exe,
                et_meta_path=et_meta_path,
                behav_meta_path=behav_meta_path,
                bids_root=bids_root,
                task_name=task_name,
                mne_verbose=mne_verbose,
            )
            errors[subj_dir.name] = subj_errors

        BIDSdata.finalize_bids_dataset(
            bids_root=bids_root,
            task_name=task_name,
            et_meta_path=et_meta_path,
        )

        return errors

    @staticmethod
    def read_bids():
        """
        see: https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html
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
