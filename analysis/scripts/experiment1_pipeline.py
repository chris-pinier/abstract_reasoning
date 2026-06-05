from __future__ import annotations

from pathlib import Path
from datetime import datetime

import json
import os
import sys
import pickle


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if path.name == "abstract_reasoning" and (path / "analysis").exists():
            return path
    raise FileNotFoundError("Could not locate the abstract_reasoning repo root.")


ROOT = find_repo_root()
ANALYSIS_DIR = ROOT / "analysis"
SRC_DIR = ANALYSIS_DIR / "src"

for path in [SRC_DIR, ANALYSIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".temp/matplotlib"))

import argparse
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import mne
from tqdm.auto import tqdm

from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, RDMs

from ar_analysis.analysis_config import Config as c
from ar_analysis.data_loader.human import HumanSubjData, HumanSessData
from ar_analysis.utils.analysis_utils import save_pickle

mne.set_log_level("WARNING")

# -----------------------------
# User-editable configuration
# -----------------------------
DATA_ROOT = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data")
HUMAN_DATA_DIR = DATA_ROOT / "Lab/raw-BIDS3"
PREPROCESSED_DIR = DATA_ROOT / "Lab/preprocessed"

# Change this for full runs, e.g. DATA_ROOT / "Lab/analyzed/experiments".
OUTPUT_BASE_DIR = ROOT / ".temp/experiments"

# Use None to process all discovered participants. Use a list for prototyping, e.g. [1].
SUBJ_NS: list[int] | None = [2]

DATA_FMT = "bids"
RNG_SEED = 0

# Global experiment constraints.
FRONTAL_CHANS = list(c.EEG_CHAN_GROUPS.frontal)
NO_BASELINE = None

# All windows are 600 ms. FRP/control are post-onset; response/rest are pre-event.
WINDOW_S = 0.600
FRP_TMIN, FRP_TMAX = 0.0, WINDOW_S
RESPONSE_TMIN, RESPONSE_TMAX = -WINDOW_S, 0.0
REST_TMIN, REST_TMAX = -WINDOW_S, 0.0

RESPONSE_EVENTS = ["a", "x", "m", "l"]
REST_EVENT = "trial_start"
FRP_STIM_SCOPE = "sequence"

# Random-control windows are sampled within each trial's true trial interval.
RANDOM_CONTROL_START_EVENT = "stim-all_stim"
RANDOM_CONTROL_END_EVENT = "trial_end"
RANDOM_CONTROL_MIN_ONSET_DIFF_S = 0.300

# Dataset/RDM export settings.
DISSIMILARITY_METRIC = "correlation"
DATASET_LEVELS = ["trial_lvl", "pattern_lvl"]
EXPORT_EVOKED_OBJECTS = False

def create_run_dirs(base_dir: Path, experiment_name: str = "experiment1") -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{experiment_name}_{timestamp}"
    dirs = {
        "run": run_dir,
        "trial_data": run_dir / "trial_data",
        "datasets": run_dir / "rsatoolbox_datasets",
        "rdms": run_dir / "rdms",
        "logs": run_dir / "logs",
        "metadata": run_dir / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=False if path == run_dir else True)
    return dirs



def dirs_from_run_dir(run_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "run": run_dir,
        "trial_data": run_dir / "trial_data",
        "datasets": run_dir / "rsatoolbox_datasets",
        "rdms": run_dir / "rdms",
        "logs": run_dir / "logs",
        "metadata": run_dir / "metadata",
    }


def apply_runtime_config(config: dict) -> None:
    global DATA_ROOT, HUMAN_DATA_DIR, PREPROCESSED_DIR, OUTPUT_BASE_DIR
    global SUBJ_NS, DATA_FMT, RNG_SEED, RANDOM_CONTROL_START_EVENT
    global RANDOM_CONTROL_END_EVENT, RANDOM_CONTROL_MIN_ONSET_DIFF_S
    global EXPORT_EVOKED_OBJECTS, DATASET_LEVELS

    DATA_ROOT = Path(config["data_root"])
    HUMAN_DATA_DIR = Path(config["human_data_dir"])
    PREPROCESSED_DIR = Path(config["preprocessed_dir"])
    OUTPUT_BASE_DIR = Path(config["output_base_dir"])
    SUBJ_NS = config["subj_Ns"]
    DATA_FMT = config["data_fmt"]
    RNG_SEED = int(config["rng_seed"])
    RANDOM_CONTROL_START_EVENT = config["random_control_start_event"]
    RANDOM_CONTROL_END_EVENT = config["random_control_end_event"]
    RANDOM_CONTROL_MIN_ONSET_DIFF_S = float(config["random_control_min_onset_diff_s"])
    EXPORT_EVOKED_OBJECTS = bool(config["export_evoked_objects"])
    DATASET_LEVELS = list(config["dataset_levels"])


def build_config(run_dirs: dict[str, Path]) -> dict:
    return {
        "data_root": str(DATA_ROOT),
        "human_data_dir": str(HUMAN_DATA_DIR),
        "preprocessed_dir": str(PREPROCESSED_DIR),
        "output_base_dir": str(OUTPUT_BASE_DIR),
        "run_dir": str(run_dirs["run"]),
        "subj_Ns": SUBJ_NS,
        "data_fmt": DATA_FMT,
        "rng_seed": RNG_SEED,
        "channels": FRONTAL_CHANS,
        "baseline": None,
        "window_s": WINDOW_S,
        "frp_stim_scope": FRP_STIM_SCOPE,
        "response_events": RESPONSE_EVENTS,
        "rest_event": REST_EVENT,
        "random_control_start_event": RANDOM_CONTROL_START_EVENT,
        "random_control_end_event": RANDOM_CONTROL_END_EVENT,
        "random_control_min_onset_diff_s": RANDOM_CONTROL_MIN_ONSET_DIFF_S,
        "dissimilarity_metric": DISSIMILARITY_METRIC,
        "dataset_levels": DATASET_LEVELS,
        "export_evoked_objects": EXPORT_EVOKED_OBJECTS,
    }

def trial_events(raw_trial: mne.io.Raw) -> pd.DataFrame:
    """Return annotation-derived events with sample indices relative to raw_trial data."""
    events, event_id = mne.events_from_annotations(raw_trial, verbose="WARNING")
    inv_event_id = {v: k for k, v in event_id.items()}
    rows = []
    for sample_abs, _, code in events:
        rows.append(
            {
                "sample": int(sample_abs - raw_trial.first_samp),
                "description": inv_event_id[int(code)],
            }
        )
    return pd.DataFrame(rows)


def n_samples_for_window(raw_trial: mne.io.Raw, tmin: float, tmax: float) -> int:
    return int(np.ceil((tmax - tmin) * raw_trial.info["sfreq"])) + 1


def selected_info(raw_trial: mne.io.Raw, selected_chans: list[str]) -> mne.Info:
    return raw_trial.copy().pick(selected_chans).info


def make_evoked_from_array(
    data: np.ndarray,
    info: mne.Info,
    tmin: float,
    comment: str,
) -> mne.Evoked:
    return mne.EvokedArray(data, info, tmin=tmin, nave=1, comment=comment)


def extract_event_locked_window(
    raw_trial: mne.io.Raw,
    event_names: list[str],
    tmin: float,
    tmax: float,
    selected_chans: list[str],
    comment: str,
) -> mne.Evoked | None:
    """Extract one no-baseline event-locked window from a trial."""
    events = trial_events(raw_trial)
    if events.empty:
        return None

    event_rows = events.query("description in @event_names")
    if event_rows.empty:
        return None

    event_sample = int(event_rows.iloc[0]["sample"])
    sfreq = raw_trial.info["sfreq"]
    n_samples = n_samples_for_window(raw_trial, tmin, tmax)
    start_sample = event_sample + int(np.round(tmin * sfreq))
    stop_sample = start_sample + n_samples

    data = raw_trial.get_data(picks=selected_chans)
    if start_sample < 0 or stop_sample > data.shape[1]:
        return None

    return make_evoked_from_array(
        data[:, start_sample:stop_sample],
        selected_info(raw_trial, selected_chans),
        tmin=tmin,
        comment=comment,
    )


def extract_response_epoch(raw_trial: mne.io.Raw) -> mne.Evoked | None:
    return extract_event_locked_window(
        raw_trial=raw_trial,
        event_names=RESPONSE_EVENTS,
        tmin=RESPONSE_TMIN,
        tmax=RESPONSE_TMAX,
        selected_chans=FRONTAL_CHANS,
        comment="response_pre600_no_baseline",
    )


def extract_rest_epoch(raw_trial: mne.io.Raw) -> mne.Evoked | None:
    return extract_event_locked_window(
        raw_trial=raw_trial,
        event_names=[REST_EVENT],
        tmin=REST_TMIN,
        tmax=REST_TMAX,
        selected_chans=FRONTAL_CHANS,
        comment="rest_pre_trial_start_600_no_baseline",
    )

def trial_period_samples(
    raw_trial: mne.io.Raw,
    start_event: str = RANDOM_CONTROL_START_EVENT,
    end_event: str = RANDOM_CONTROL_END_EVENT,
) -> tuple[int, int] | None:
    """Return sample bounds for the true trial interval inside a cropped trial Raw."""
    events = trial_events(raw_trial)
    if events.empty:
        return None

    start_rows = events.query("description == @start_event")
    end_rows = events.query("description == @end_event")
    if start_rows.empty or end_rows.empty:
        return None

    start_sample = int(start_rows.iloc[0]["sample"])
    end_sample = int(end_rows.iloc[-1]["sample"])
    if end_sample <= start_sample:
        return None
    return start_sample, end_sample


def sample_spaced_random_starts(
    possible_starts: np.ndarray,
    n_windows: int,
    min_onset_diff_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 1_000,
) -> np.ndarray | None:
    """Sample unique onset samples separated by at least min_onset_diff_samples."""
    if n_windows <= 0:
        return np.asarray([], dtype=int)
    if n_windows > possible_starts.size:
        return None

    min_onset_diff_samples = max(1, int(min_onset_diff_samples))
    for _ in range(max_attempts):
        selected: list[int] = []
        for start in rng.permutation(possible_starts):
            if all(abs(int(start) - other) >= min_onset_diff_samples for other in selected):
                selected.append(int(start))
                if len(selected) == n_windows:
                    return np.sort(np.asarray(selected, dtype=int))
    return None


def extract_random_control_frp(
    raw_trial: mne.io.Raw,
    n_windows: int,
    rng: np.random.Generator,
    selected_chans: list[str] = FRONTAL_CHANS,
    window_s: float = WINDOW_S,
    min_onset_diff_s: float = RANDOM_CONTROL_MIN_ONSET_DIFF_S,
) -> tuple[mne.Evoked | None, list[int]]:
    """Generate an average of n random no-baseline windows in a trial."""
    if n_windows <= 0:
        return None, []

    period = trial_period_samples(raw_trial)
    if period is None:
        return None, []

    start_bound, end_bound = period
    sfreq = raw_trial.info["sfreq"]
    n_samples = int(np.ceil(window_s * sfreq)) + 1
    max_start = end_bound - n_samples
    if max_start < start_bound:
        return None, []

    possible_starts = np.arange(start_bound, max_start + 1, dtype=int)
    min_onset_diff_samples = int(np.ceil(min_onset_diff_s * sfreq))
    starts = sample_spaced_random_starts(
        possible_starts=possible_starts,
        n_windows=n_windows,
        min_onset_diff_samples=min_onset_diff_samples,
        rng=rng,
    )
    if starts is None:
        return None, []

    data = raw_trial.get_data(picks=selected_chans)
    windows = np.stack([data[:, start : start + n_samples] for start in starts])
    avg_window = windows.mean(axis=0)

    evoked = make_evoked_from_array(
        avg_window,
        selected_info(raw_trial, selected_chans),
        tmin=0.0,
        comment="frp_random_control_post600_no_baseline",
    )
    return evoked, starts.tolist()


def extract_true_frp(
    sess: HumanSessData,
    eeg_trial: mne.io.Raw,
    et_trial: mne.io.Raw,
    behav: pd.DataFrame,
    trial_N: int,
) -> tuple[mne.Evoked | None, int]:
    """Extract the true sequence-FRP and return the number of valid fixation epochs."""
    frp, fixation_epochs = sess.get_trial_frp(
        eeg_trial=eeg_trial,
        et_trial=et_trial,
        raw_behav=behav,
        trial_N=trial_N,
        stim_scope=FRP_STIM_SCOPE,
        tmin=FRP_TMIN,
        tmax=FRP_TMAX,
        baseline=NO_BASELINE,
        selected_chans=FRONTAL_CHANS,
        return_epochs=True,
    )
    n_fixations = 0 if fixation_epochs is None else len(fixation_epochs)
    return frp, n_fixations

CONDITIONS = ["frp", "frp_control", "response", "rest"]


def trial_uid(subj_N: int, sess_N: int, trial_N: int) -> str:
    return f"sub-{subj_N:02}_ses-{sess_N:02}_trial-{trial_N:03}"


def row_value(row: pd.Series, key: str, default=None):
    return row[key] if key in row.index else default


def process_subject_trials(
    subj: HumanSubjData,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, mne.Evoked]], dict[str, dict], list[dict]]:
    condition_data: dict[str, dict[str, mne.Evoked]] = {condition: {} for condition in CONDITIONS}
    trial_records: dict[str, dict] = {}
    random_window_rows: list[dict] = []

    for sess_N, sess in tqdm(
        subj.sessions.items(),
        desc=f"subj {subj.subj_N:02} sessions",
        leave=False,
    ):
        behav, et_trials, eeg_trials = sess.get_trials_data(
            preprocessed_dir=PREPROCESSED_DIR,
            raise_error=False,
            eeg_incomplete="skip",
        )
        if behav is None or et_trials is None or eeg_trials is None:
            continue

        et_trials = list(et_trials)
        eeg_trials = list(eeg_trials)
        n_trials = min(len(behav), len(et_trials), len(eeg_trials))

        for trial_pos, ((behav_index, behav_row), et_trial, eeg_trial) in enumerate(
            zip(behav.iterrows(), et_trials, eeg_trials)
        ):
            if trial_pos >= n_trials:
                break

            # get_trial_info expects the behavioral DataFrame index used by the session loader.
            trial_N = int(behav_index)
            uid = trial_uid(subj.subj_N, sess_N, trial_N)

            record = {
                "trial_uid": uid,
                "subj_N": subj.subj_N,
                "sess_N": sess_N,
                "trial_N": trial_N,
                "trial_pos": trial_pos,
                "item_id": row_value(behav_row, "item_id"),
                "pattern": row_value(behav_row, "pattern"),
            }
            trial_records[uid] = record

            frp, n_fixations = extract_true_frp(sess, eeg_trial, et_trial, behav, trial_N)
            if frp is not None:
                condition_data["frp"][uid] = frp

            control, random_starts = extract_random_control_frp(eeg_trial, n_fixations, rng)
            if control is not None:
                condition_data["frp_control"][uid] = control
            for start in random_starts:
                random_window_rows.append(
                    {
                        "trial_uid": uid,
                        "subj_N": subj.subj_N,
                        "sess_N": sess_N,
                        "trial_N": trial_N,
                        "random_onset_sample": start,
                        "random_onset_s": start / eeg_trial.info["sfreq"],
                        "min_onset_diff_s": RANDOM_CONTROL_MIN_ONSET_DIFF_S,
                    }
                )

            response = extract_response_epoch(eeg_trial)
            if response is not None:
                condition_data["response"][uid] = response

            rest = extract_rest_epoch(eeg_trial)
            if rest is not None:
                condition_data["rest"][uid] = rest

    return condition_data, trial_records, random_window_rows


def dataset_trial_uids(trial_records: dict[str, dict]) -> list[str]:
    """Return all trials observed for a participant; missing condition data is filled with NaNs later."""
    return sorted(trial_records)


def complete_trial_uids(condition_data: dict[str, dict[str, mne.Evoked]]) -> list[str]:
    """Return trials with valid data in every condition, useful for summaries/checks."""
    trial_sets = [set(condition_data[condition].keys()) for condition in CONDITIONS]
    return sorted(set.intersection(*trial_sets))



def evoked_to_feature_vector(evoked: mne.Evoked) -> np.ndarray:
    """Flatten frontal channels x time into one observation feature vector."""
    return evoked.get_data().reshape(-1)


def sorted_aligned_uids(aligned_uids: list[str], trial_records: dict[str, dict]) -> list[str]:
    """Sort aligned trials by pattern first and item_id second."""
    manifest = pd.DataFrame([trial_records[uid] for uid in aligned_uids]).copy()
    manifest["_uid"] = aligned_uids
    manifest = manifest.sort_values(
        ["pattern", "item_id", "sess_N", "trial_N"],
        kind="mergesort",
        na_position="last",
    )
    return manifest["_uid"].tolist()


def infer_feature_shape(condition_trials: dict[str, dict[str, mne.Evoked]]) -> tuple[int, ...]:
    """Find a valid trial feature shape for NaN-filled missing observations."""
    for trials in condition_trials.values():
        for evoked in trials.values():
            return evoked_to_feature_vector(evoked).shape
    raise ValueError("Could not infer feature shape: no valid condition data found.")


def dataset_descriptors(subj_N: int, condition: str, level: str) -> dict:
    return {
        "subj_N": subj_N,
        "condition": condition,
        "level": level,
        "channels": "frontal",
        "baseline": "None",
        "window_s": WINDOW_S,
        "missing_observations": "np.nan",
    }


def build_trial_dataset(
    subj_N: int,
    condition: str,
    condition_trials: dict[str, mne.Evoked],
    trial_records: dict[str, dict],
    aligned_uids: list[str],
    feature_shape: tuple[int, ...],
) -> tuple[Dataset, np.ndarray, pd.DataFrame]:
    sorted_uids = sorted_aligned_uids(aligned_uids, trial_records)
    measurement_rows = []
    has_data = []
    for uid in sorted_uids:
        evoked = condition_trials.get(uid)
        if evoked is None:
            measurement_rows.append(np.full(feature_shape, np.nan))
            has_data.append(False)
        else:
            measurement_rows.append(evoked_to_feature_vector(evoked))
            has_data.append(True)

    measurements = np.stack(measurement_rows)
    manifest = pd.DataFrame([trial_records[uid] for uid in sorted_uids]).copy()
    manifest.insert(0, "overall_trial_N", np.arange(len(manifest)))
    manifest.insert(1, "condition", condition)
    manifest.insert(2, "level", "trial_lvl")
    manifest["has_data"] = has_data

    obs_descriptors = {
        "trial_uid": manifest["trial_uid"].tolist(),
        "overall_trial_N": manifest["overall_trial_N"].tolist(),
        "sess_N": manifest["sess_N"].tolist(),
        "trial_N": manifest["trial_N"].tolist(),
        "item_id": manifest["item_id"].tolist(),
        "pattern": manifest["pattern"].tolist(),
        "has_data": manifest["has_data"].tolist(),
    }
    dataset = Dataset(
        measurements=measurements,
        descriptors=dataset_descriptors(subj_N, condition, "trial_lvl"),
        obs_descriptors=obs_descriptors,
    )
    return dataset, measurements, manifest


def build_pattern_dataset(
    subj_N: int,
    condition: str,
    trial_measurements: np.ndarray,
    trial_manifest: pd.DataFrame,
    feature_shape: tuple[int, ...],
) -> tuple[Dataset, np.ndarray, pd.DataFrame]:
    pattern_rows = []
    manifest_rows = []
    for pattern in sorted(trial_manifest["pattern"].dropna().unique()):
        pattern_mask = trial_manifest["pattern"].to_numpy() == pattern
        valid_mask = pattern_mask & trial_manifest["has_data"].to_numpy(dtype=bool)
        n_valid = int(valid_mask.sum())
        n_total = int(pattern_mask.sum())
        if n_valid == 0:
            pattern_rows.append(np.full(feature_shape, np.nan))
            has_data = False
        else:
            pattern_rows.append(np.nanmean(trial_measurements[valid_mask], axis=0))
            has_data = True
        manifest_rows.append(
            {
                "condition": condition,
                "level": "pattern_lvl",
                "pattern": pattern,
                "has_data": has_data,
                "n_valid_items": n_valid,
                "n_total_items": n_total,
            }
        )

    measurements = np.stack(pattern_rows)
    manifest = pd.DataFrame(manifest_rows)
    obs_descriptors = {
        "pattern": manifest["pattern"].tolist(),
        "has_data": manifest["has_data"].tolist(),
        "n_valid_items": manifest["n_valid_items"].tolist(),
        "n_total_items": manifest["n_total_items"].tolist(),
    }
    dataset = Dataset(
        measurements=measurements,
        descriptors=dataset_descriptors(subj_N, condition, "pattern_lvl"),
        obs_descriptors=obs_descriptors,
    )
    return dataset, measurements, manifest


def dataset_without_nan_observations(dataset: Dataset) -> tuple[Dataset, np.ndarray]:
    """Return a copy of a Dataset restricted to observations with finite measurements."""
    measurements = np.asarray(dataset.measurements)
    valid_mask = np.isfinite(measurements.reshape(measurements.shape[0], -1)).all(axis=1)
    obs_descriptors = {
        key: np.asarray(value)[valid_mask].tolist()
        for key, value in dataset.obs_descriptors.items()
    }
    valid_dataset = Dataset(
        measurements=measurements[valid_mask],
        descriptors={**dataset.descriptors, "nan_observations_dropped_for_rdm": True},
        obs_descriptors=obs_descriptors,
        channel_descriptors=dataset.channel_descriptors,
    )
    return valid_dataset, valid_mask


def calculate_rdm(dataset: Dataset) -> tuple[RDMs, np.ndarray]:
    """Calculate an RDM and expand it back to the dataset observation grid."""
    valid_dataset, valid_mask = dataset_without_nan_observations(dataset)
    n_observations = dataset.measurements.shape[0]
    full_matrix = np.full((n_observations, n_observations), np.nan, dtype=float)
    valid_indices = np.flatnonzero(valid_mask)

    if valid_dataset.measurements.shape[0] == 1:
        full_matrix[valid_indices[0], valid_indices[0]] = 0.0
    elif valid_dataset.measurements.shape[0] >= 2:
        valid_rdm = calc_rdm(valid_dataset, method=DISSIMILARITY_METRIC)
        valid_matrix = valid_rdm.get_matrices()[0]
        full_matrix[np.ix_(valid_indices, valid_indices)] = valid_matrix

    descriptors = {
        **dataset.descriptors,
        "nan_observations_preserved_in_rdm": True,
        "n_nan_observations": int((~valid_mask).sum()),
    }
    rdm_descriptors = {
        "subj_N": [dataset.descriptors.get("subj_N")],
        "condition": [dataset.descriptors.get("condition")],
        "level": [dataset.descriptors.get("level")],
    }
    full_rdm = RDMs(
        dissimilarities=full_matrix[None, :, :],
        dissimilarity_measure=DISSIMILARITY_METRIC,
        descriptors=descriptors,
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=dataset.obs_descriptors,
    )
    return full_rdm, valid_mask


def save_level_outputs(
    subj_label: str,
    condition: str,
    level: str,
    dataset: Dataset,
    measurements: np.ndarray,
    manifest: pd.DataFrame,
) -> dict:
    rdm, rdm_valid_mask = calculate_rdm(dataset)
    base = f"{subj_label}-{level}-{condition}-frontal-no_baseline"

    np.save(RUN_DIRS["trial_data"] / f"measurements-{base}.npy", measurements)
    manifest.to_csv(RUN_DIRS["metadata"] / f"manifest-{base}.csv", index=False)
    dataset.save(RUN_DIRS["datasets"] / f"dataset-{base}.hdf5", file_type="hdf5", overwrite=True)
    np.save(RUN_DIRS["metadata"] / f"rdm_valid_mask-{base}.npy", rdm_valid_mask)

    rdm_path = RUN_DIRS["rdms"] / f"rdm-{base}.hdf5"
    rdm_npy_path = RUN_DIRS["rdms"] / f"rdm-{base}.npy"
    rdm.save(rdm_path, file_type="hdf5", overwrite=True)
    rdm_matrix = rdm.get_matrices()
    np.save(rdm_npy_path, rdm_matrix)

    return {
        "level": level,
        "n_dataset_observations": measurements.shape[0],
        "n_valid_for_condition": int(np.sum(manifest["has_data"])),
        "n_nan_observations": int((~manifest["has_data"]).sum()),
        "n_rdm_observations": int(np.sum(rdm_valid_mask)),
        "n_nan_rdm_cells": int(np.isnan(rdm_matrix).sum()),
        "measurements_shape": tuple(measurements.shape),
        "rdm_shape": tuple(rdm_matrix.shape),
        "dataset": str(RUN_DIRS["datasets"] / f"dataset-{base}.hdf5"),
        "rdm": str(rdm_path),
        "rdm_numpy": str(rdm_npy_path),
    }


def save_subject_outputs(
    subj_N: int,
    condition_data: dict[str, dict[str, mne.Evoked]],
    trial_records: dict[str, dict],
    random_window_rows: list[dict],
    aligned_uids: list[str],
) -> dict[str, dict]:
    subj_label = f"subj_{subj_N:02}"
    output_summary: dict[str, dict] = {}
    feature_shape = infer_feature_shape(condition_data)
    sorted_uids = sorted_aligned_uids(aligned_uids, trial_records)

    if EXPORT_EVOKED_OBJECTS:
        save_pickle(
            {
                "condition_data": condition_data,
                "trial_records": trial_records,
                "aligned_trial_uids": sorted_uids,
                "missing_condition_data": "stored as np.nan in datasets/measurements",
            },
            RUN_DIRS["trial_data"] / f"{subj_label}-condition_evokeds.pkl",
        )

    pd.DataFrame(random_window_rows).to_csv(
        RUN_DIRS["trial_data"] / f"{subj_label}-frp_control_random_windows.csv",
        index=False,
    )

    for condition in CONDITIONS:
        trial_dataset, trial_measurements, trial_manifest = build_trial_dataset(
            subj_N=subj_N,
            condition=condition,
            condition_trials=condition_data[condition],
            trial_records=trial_records,
            aligned_uids=sorted_uids,
            feature_shape=feature_shape,
        )
        output_summary[f"trial_lvl/{condition}"] = save_level_outputs(
            subj_label,
            condition,
            "trial_lvl",
            trial_dataset,
            trial_measurements,
            trial_manifest,
        )

        pattern_dataset, pattern_measurements, pattern_manifest = build_pattern_dataset(
            subj_N=subj_N,
            condition=condition,
            trial_measurements=trial_measurements,
            trial_manifest=trial_manifest,
            feature_shape=feature_shape,
        )
        output_summary[f"pattern_lvl/{condition}"] = save_level_outputs(
            subj_label,
            condition,
            "pattern_lvl",
            pattern_dataset,
            pattern_measurements,
            pattern_manifest,
        )

    pd.DataFrame([trial_records[uid] for uid in sorted_uids]).to_csv(
        RUN_DIRS["metadata"] / f"{subj_label}-aligned_trials.csv",
        index=False,
    )
    return output_summary



def discover_subject_numbers(human_data_dir: Path, data_fmt: str) -> list[int]:
    human_data_dir = Path(human_data_dir)
    prefix = "sub-" if data_fmt == "bids" else "subj_"
    subj_ns = []
    for path in sorted(human_data_dir.glob(f"{prefix}*")):
        if not path.is_dir():
            continue
        match = re.search(r"(\d{2})", path.name)
        if match:
            subj_ns.append(int(match.group(1)))
    if not subj_ns:
        raise FileNotFoundError(f"No subjects found in {human_data_dir}")
    return subj_ns


def parse_subjects(value: str | None) -> list[int] | None:
    if value is None or value.lower() in {"none", "all"}:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def run_subject_experiment(subj_N: int, run_dir: str | Path, config: dict) -> dict:
    global RUN_DIRS
    apply_runtime_config(config)
    RUN_DIRS = dirs_from_run_dir(Path(run_dir))

    rng = np.random.default_rng(RNG_SEED + int(subj_N))
    subj = HumanSubjData(
        data_dir=HUMAN_DATA_DIR,
        preprocessed_dir=PREPROCESSED_DIR,
        export_dir=RUN_DIRS["run"],
        subj_N=int(subj_N),
        data_fmt=DATA_FMT,
    )

    condition_data, trial_records, random_window_rows = process_subject_trials(subj, rng)
    aligned_uids = dataset_trial_uids(trial_records)
    complete_uids = complete_trial_uids(condition_data)

    status_row = {
        "subj_N": int(subj_N),
        "n_trials_total": len(trial_records),
        "n_frp": len(condition_data["frp"]),
        "n_frp_control": len(condition_data["frp_control"]),
        "n_response": len(condition_data["response"]),
        "n_rest": len(condition_data["rest"]),
        "n_complete": len(complete_uids),
        "n_dataset_trials": len(aligned_uids),
        "status": "ok",
    }

    if len(aligned_uids) == 0:
        status_row["status"] = "no_trials"
        return status_row

    output_summary = save_subject_outputs(
        subj_N=int(subj_N),
        condition_data=condition_data,
        trial_records=trial_records,
        random_window_rows=random_window_rows,
        aligned_uids=aligned_uids,
    )
    status_row["outputs"] = json.dumps(output_summary)
    return status_row


def run_subject_safe(subj_N: int, run_dir: str | Path, config: dict) -> tuple[dict, dict | None]:
    try:
        return run_subject_experiment(subj_N, run_dir, config), None
    except Exception as exc:
        error = {
            "subj_N": int(subj_N),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        status = {
            "subj_N": int(subj_N),
            "n_trials_total": np.nan,
            "n_frp": np.nan,
            "n_frp_control": np.nan,
            "n_response": np.nan,
            "n_rest": np.nan,
            "n_complete": np.nan,
            "n_dataset_trials": np.nan,
            "status": "error",
            "error": repr(exc),
        }
        return status, error


def write_run_outputs(run_dirs: dict[str, Path], config: dict, summary_rows: list[dict], errors: list[dict]) -> pd.DataFrame:
    summary_df = pd.DataFrame(summary_rows).sort_values("subj_N")
    errors_df = pd.DataFrame(errors)
    summary_df.to_csv(run_dirs["logs"] / "experiment1_summary.csv", index=False)
    errors_df.to_csv(run_dirs["logs"] / "experiment1_errors.csv", index=False)
    with open(run_dirs["metadata"] / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return summary_df


def run_experiment(subj_ns: list[int], n_workers: int, config: dict, run_dirs: dict[str, Path]) -> pd.DataFrame:
    summary_rows: list[dict] = []
    errors: list[dict] = []

    if n_workers <= 1:
        for subj_N in tqdm(subj_ns, desc="Experiment 1 participants"):
            status, error = run_subject_safe(subj_N, run_dirs["run"], config)
            summary_rows.append(status)
            if error is not None:
                errors.append(error)
            write_run_outputs(run_dirs, config, summary_rows, errors)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_subject_safe, subj_N, run_dirs["run"], config): subj_N
                for subj_N in subj_ns
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Experiment 1 participants"):
                status, error = future.result()
                summary_rows.append(status)
                if error is not None:
                    errors.append(error)
                write_run_outputs(run_dirs, config, summary_rows, errors)

    return write_run_outputs(run_dirs, config, summary_rows, errors)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 1 EEG RSA export pipeline.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--human-data-dir", type=Path, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--output-base-dir", type=Path, default=OUTPUT_BASE_DIR)
    parser.add_argument("--subj-ns", type=str, default=None, help="Comma-separated subject numbers, or 'all'.")
    parser.add_argument("--data-fmt", type=str, default=DATA_FMT, choices=["bids", "original"])
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--random-control-start-event", type=str, default=RANDOM_CONTROL_START_EVENT)
    parser.add_argument("--random-control-end-event", type=str, default=RANDOM_CONTROL_END_EVENT)
    parser.add_argument("--random-control-min-onset-diff-s", type=float, default=RANDOM_CONTROL_MIN_ONSET_DIFF_S)
    parser.add_argument("--export-evoked-objects", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_arg_parser().parse_args(argv)

    global DATA_ROOT, HUMAN_DATA_DIR, PREPROCESSED_DIR, OUTPUT_BASE_DIR
    global SUBJ_NS, DATA_FMT, RNG_SEED, RANDOM_CONTROL_START_EVENT
    global RANDOM_CONTROL_END_EVENT, RANDOM_CONTROL_MIN_ONSET_DIFF_S, EXPORT_EVOKED_OBJECTS
    global RUN_DIRS

    DATA_ROOT = Path(args.data_root)
    HUMAN_DATA_DIR = Path(args.human_data_dir) if args.human_data_dir is not None else DATA_ROOT / "Lab/raw-BIDS3"
    PREPROCESSED_DIR = Path(args.preprocessed_dir) if args.preprocessed_dir is not None else DATA_ROOT / "Lab/preprocessed"
    OUTPUT_BASE_DIR = Path(args.output_base_dir)
    SUBJ_NS = parse_subjects(args.subj_ns)
    DATA_FMT = args.data_fmt
    RNG_SEED = args.rng_seed
    RANDOM_CONTROL_START_EVENT = args.random_control_start_event
    RANDOM_CONTROL_END_EVENT = args.random_control_end_event
    RANDOM_CONTROL_MIN_ONSET_DIFF_S = args.random_control_min_onset_diff_s
    EXPORT_EVOKED_OBJECTS = bool(args.export_evoked_objects)

    subj_ns = SUBJ_NS if SUBJ_NS is not None else discover_subject_numbers(HUMAN_DATA_DIR, DATA_FMT)
    RUN_DIRS = create_run_dirs(OUTPUT_BASE_DIR, "experiment1_frontal_no_baseline")
    config = build_config(RUN_DIRS)

    print(f"Run dir:          {RUN_DIRS['run']}")
    print(f"HUMAN_DATA_DIR:   {HUMAN_DATA_DIR} exists={HUMAN_DATA_DIR.exists()}")
    print(f"PREPROCESSED_DIR: {PREPROCESSED_DIR} exists={PREPROCESSED_DIR.exists()}")
    print(f"Subjects:         {subj_ns}")
    print(f"Workers:          {args.n_workers}")

    summary_df = run_experiment(subj_ns, max(1, args.n_workers), config, RUN_DIRS)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    main()
