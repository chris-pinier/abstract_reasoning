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
from ar_analysis.analysis_plotting import get_gaze_heatmap
from ar_analysis.data_loader.human import HumanSubjData, HumanSessData
from ar_analysis.utils.analysis_utils import get_trial_info, save_pickle

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
CHANNEL_GROUP = "frontal"
SELECTED_CHANS = list(c.EEG_CHAN_GROUPS[CHANNEL_GROUP])

# All windows are 600 ms. FRP/control are post-onset; response/rest are pre-event.
WINDOW_S = 0.600
FRP_TMIN, FRP_TMAX = 0.0, WINDOW_S
RESPONSE_TMIN, RESPONSE_TMAX = -WINDOW_S, 0.0
REST_TMIN, REST_TMAX = -WINDOW_S, 0.0
FRP_BASELINE = None
RESPONSE_BASELINE = None
REST_BASELINE = None

RESPONSE_EVENTS = ["a", "x", "m", "l"]
REST_EVENT = "trial_start"
FRP_STIM_SCOPE = "sequence"

# Random-control windows are sampled within each trial's true trial interval.
FRP_CONTROL_METHOD = "circular_shift"
RANDOM_CONTROL_START_EVENT = "stim-all_stim"
RANDOM_CONTROL_END_EVENT = "trial_end"
RANDOM_CONTROL_MIN_ONSET_DIFF_S = 0.200
SHIFT_CONTROL_MAX_ATTEMPTS = 1_000
HEATMAP_BIN_SIZE = 50

# Dataset/RDM export settings.
DISSIMILARITY_METRIC = "correlation"
DATASET_LEVELS = ["trial_lvl", "pattern_lvl"]
EXPORT_EVOKED_OBJECTS = False


def resolve_channel_group(channel_group: str) -> list[str] | str:
    """Resolve an EEG channel group name or comma-separated channel list."""
    if channel_group == "eeg":
        return "eeg"
    if channel_group in c.EEG_CHAN_GROUPS:
        return list(c.EEG_CHAN_GROUPS[channel_group])
    if "," in channel_group:
        return [chan.strip() for chan in channel_group.split(",") if chan.strip()]
    raise ValueError(
        f"Unknown channel group `{channel_group}`. "
        f"Use one of {list(c.EEG_CHAN_GROUPS.keys())}, `eeg`, or a comma-separated channel list."
    )


def parse_baseline(value: str | None) -> tuple[float | None, float | None] | None:
    """Parse an MNE-style baseline string: none, -0.1,0, None,0."""
    if value is None or value.strip().lower() in {"none", "null", "false", ""}:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(
            "Baseline must be `none` or two comma-separated values, e.g. `-0.1,0`."
        )

    def parse_part(part: str) -> float | None:
        return None if part.lower() in {"none", "null"} else float(part)

    return parse_part(parts[0]), parse_part(parts[1])


def baseline_tag(baseline: tuple[float | None, float | None] | None) -> str:
    if baseline is None:
        return "no_baseline"
    return "baseline_" + "_".join(
        "None" if val is None else f"{val:g}" for val in baseline
    )


def baseline_tag_for_condition(condition: str) -> str:
    if condition in {"frp", "frp_control"}:
        return baseline_tag(FRP_BASELINE)
    if condition == "response":
        return baseline_tag(RESPONSE_BASELINE)
    if condition == "rest":
        return baseline_tag(REST_BASELINE)
    return "no_baseline"


def channel_tag() -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", CHANNEL_GROUP).strip("_")


def create_run_dirs(
    base_dir: Path, experiment_name: str = "experiment1"
) -> dict[str, Path]:
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
    global EXPORT_EVOKED_OBJECTS, DATASET_LEVELS, CHANNEL_GROUP, SELECTED_CHANS
    global FRP_BASELINE, RESPONSE_BASELINE, REST_BASELINE, FRP_CONTROL_METHOD
    global HEATMAP_BIN_SIZE

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
    CHANNEL_GROUP = config["channel_group"]
    SELECTED_CHANS = resolve_channel_group(CHANNEL_GROUP)
    FRP_BASELINE = (
        tuple(config["frp_baseline"]) if config["frp_baseline"] is not None else None
    )
    RESPONSE_BASELINE = (
        tuple(config["response_baseline"])
        if config["response_baseline"] is not None
        else None
    )
    REST_BASELINE = (
        tuple(config["rest_baseline"]) if config["rest_baseline"] is not None else None
    )
    FRP_CONTROL_METHOD = config["frp_control_method"]
    HEATMAP_BIN_SIZE = int(config["heatmap_bin_size"])


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
        "channel_group": CHANNEL_GROUP,
        "channels": SELECTED_CHANS,
        "frp_baseline": FRP_BASELINE,
        "response_baseline": RESPONSE_BASELINE,
        "rest_baseline": REST_BASELINE,
        "window_s": WINDOW_S,
        "frp_stim_scope": FRP_STIM_SCOPE,
        "response_events": RESPONSE_EVENTS,
        "rest_event": REST_EVENT,
        "frp_control_method": FRP_CONTROL_METHOD,
        "random_control_start_event": RANDOM_CONTROL_START_EVENT,
        "random_control_end_event": RANDOM_CONTROL_END_EVENT,
        "random_control_min_onset_diff_s": RANDOM_CONTROL_MIN_ONSET_DIFF_S,
        "heatmap_bin_size": HEATMAP_BIN_SIZE,
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


def selected_info(raw_trial: mne.io.Raw, selected_chans: list[str] | str) -> mne.Info:
    return raw_trial.copy().pick(selected_chans).info


def make_evoked_from_array(
    data: np.ndarray,
    info: mne.Info,
    tmin: float,
    comment: str,
    baseline: tuple[float | None, float | None] | None = None,
) -> mne.Evoked:
    evoked = mne.EvokedArray(data, info, tmin=tmin, nave=1, comment=comment)
    if baseline is not None:
        evoked.apply_baseline(baseline)
    return evoked


def extract_event_locked_window(
    raw_trial: mne.io.Raw,
    event_names: list[str],
    tmin: float,
    tmax: float,
    selected_chans: list[str] | str,
    comment: str,
    baseline: tuple[float | None, float | None] | None = None,
) -> mne.Evoked | None:
    """Extract one event-locked window from a trial."""
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
        baseline=baseline,
    )


def extract_response_epoch(raw_trial: mne.io.Raw) -> mne.Evoked | None:
    return extract_event_locked_window(
        raw_trial=raw_trial,
        event_names=RESPONSE_EVENTS,
        tmin=RESPONSE_TMIN,
        tmax=RESPONSE_TMAX,
        selected_chans=SELECTED_CHANS,
        comment=f"response_pre600_{baseline_tag(RESPONSE_BASELINE)}",
        baseline=RESPONSE_BASELINE,
    )


def extract_rest_epoch(raw_trial: mne.io.Raw) -> mne.Evoked | None:
    return extract_event_locked_window(
        raw_trial=raw_trial,
        event_names=[REST_EVENT],
        tmin=REST_TMIN,
        tmax=REST_TMAX,
        selected_chans=SELECTED_CHANS,
        comment=f"rest_pre_trial_start_600_{baseline_tag(REST_BASELINE)}",
        baseline=REST_BASELINE,
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
            if all(
                abs(int(start) - other) >= min_onset_diff_samples for other in selected
            ):
                selected.append(int(start))
                if len(selected) == n_windows:
                    return np.sort(np.asarray(selected, dtype=int))
    return None


def extract_random_control_frp(
    raw_trial: mne.io.Raw,
    n_windows: int,
    rng: np.random.Generator,
    selected_chans: list[str] | str | None = None,
    window_s: float = WINDOW_S,
    min_onset_diff_s: float = RANDOM_CONTROL_MIN_ONSET_DIFF_S,
) -> tuple[mne.Evoked | None, list[int]]:
    """Generate an average of n random windows in a trial."""
    selected_chans = SELECTED_CHANS if selected_chans is None else selected_chans
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
        comment=f"frp_spaced_random_control_post600_{baseline_tag(FRP_BASELINE)}",
        baseline=FRP_BASELINE,
    )
    return evoked, starts.tolist()


def valid_sequence_fixation_onsets_and_gaze(
    sess: HumanSessData,
    eeg_trial: mne.io.Raw,
    et_trial: mne.io.Raw,
    behav: pd.DataFrame,
    trial_N: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Return valid sequence-fixation EEG onset samples and gaze traces."""
    cropped_et_trial, et_annotations, time_bounds = sess.crop_et_trial(et_trial)
    (
        stimulus_positions,
        _,
        sequence_items,
        choice_items,
        *_,
    ) = get_trial_info(
        trial_N,
        behav,
        c.X_POS_STIM,
        c.Y_POS_CHOICES,
        c.Y_POS_SEQUENCE,
        c.SCREEN_RESOLUTION,
        c.IMG_SIZE,
    )
    selected_stim_inds = set(
        sess._stim_inds_for_scope(FRP_STIM_SCOPE, sequence_items, choice_items)
    )

    et_data = cropped_et_trial.get_data()
    et_times = cropped_et_trial.times
    sfreq = float(eeg_trial.info["sfreq"])
    n_epoch_samples = n_samples_for_window(eeg_trial, FRP_TMIN, FRP_TMAX)
    window_offset_samples = int(np.round(FRP_TMIN * sfreq))
    onsets = []
    gaze_traces = []

    for fixation_ind in et_annotations.query("description == 'fixation'").index:
        fixation = et_annotations.loc[fixation_ind]
        fixation_onset = float(fixation["onset"])
        fixation_duration = float(fixation["duration"])
        if fixation_duration < c.MIN_FIXATION_DURATION:
            continue

        fixation_offset = min(fixation_onset + fixation_duration, et_times[-1])
        et_start_sample = int(np.searchsorted(et_times, fixation_onset, side="left"))
        et_stop_sample = int(np.searchsorted(et_times, fixation_offset, side="right"))
        if et_stop_sample <= et_start_sample:
            continue

        gaze_x, gaze_y = et_data[:2, et_start_sample:et_stop_sample]
        on_target, stim_ind = sess.is_fixation_on_target(
            gaze_x, gaze_y, stimulus_positions
        )
        if not on_target or stim_ind not in selected_stim_inds:
            continue

        fixation_onset_in_trial = time_bounds[0] + fixation_onset
        onset_sample = int(np.round(fixation_onset_in_trial * sfreq))
        eeg_window_start = onset_sample + window_offset_samples
        eeg_window_stop = eeg_window_start + n_epoch_samples
        if eeg_window_start < 0 or eeg_window_stop > eeg_trial.n_times:
            continue

        onsets.append(onset_sample)
        gaze_traces.append(np.stack([gaze_x, gaze_y], axis=0))

    return onsets, gaze_traces


def circular_shift_onsets(
    onsets: list[int],
    min_onset: int,
    max_onset: int,
    rng: np.random.Generator,
    max_attempts: int = SHIFT_CONTROL_MAX_ATTEMPTS,
) -> tuple[np.ndarray | None, int | None]:
    """Circularly shift fixation onsets within inclusive onset bounds."""
    if not onsets:
        return None, None

    onsets_arr = np.asarray(onsets, dtype=int)
    onsets_arr = onsets_arr[(onsets_arr >= min_onset) & (onsets_arr <= max_onset)]
    if onsets_arr.size == 0:
        return None, None

    period_len = max_onset - min_onset + 1
    if period_len <= 1:
        return None, None

    for _ in range(max_attempts):
        shift = int(rng.integers(1, period_len))
        shifted = min_onset + ((onsets_arr - min_onset + shift) % period_len)
        if not np.array_equal(np.sort(shifted), np.sort(onsets_arr)):
            return np.sort(shifted), shift
    return None, None


def extract_shifted_control_frp(
    raw_trial: mne.io.Raw,
    true_onset_samples: list[int],
    rng: np.random.Generator,
    selected_chans: list[str] | str | None = None,
    tmin: float = FRP_TMIN,
    tmax: float = FRP_TMAX,
) -> tuple[mne.Evoked | None, list[int], int | None]:
    """Generate a fake FRP by circularly shifting true fixation onsets in the trial."""
    selected_chans = SELECTED_CHANS if selected_chans is None else selected_chans
    if not true_onset_samples:
        return None, [], None

    period = trial_period_samples(raw_trial)
    if period is None:
        return None, [], None

    start_bound, end_bound = period
    sfreq = float(raw_trial.info["sfreq"])
    n_samples = n_samples_for_window(raw_trial, tmin, tmax)
    min_onset = start_bound - int(np.round(tmin * sfreq))
    max_onset = end_bound - n_samples - int(np.round(tmin * sfreq))
    if max_onset < min_onset:
        return None, [], None

    shifted_onsets, shift = circular_shift_onsets(
        true_onset_samples,
        min_onset=min_onset,
        max_onset=max_onset,
        rng=rng,
    )
    if shifted_onsets is None:
        return None, [], None

    data = raw_trial.get_data(picks=selected_chans)
    start_offsets = shifted_onsets + int(np.round(tmin * sfreq))
    windows = np.stack([data[:, start : start + n_samples] for start in start_offsets])
    avg_window = windows.mean(axis=0)
    evoked = make_evoked_from_array(
        avg_window,
        selected_info(raw_trial, selected_chans),
        tmin=tmin,
        comment=f"frp_circular_shift_control_post600_{baseline_tag(FRP_BASELINE)}",
        baseline=FRP_BASELINE,
    )
    return evoked, shifted_onsets.tolist(), shift


def extract_sequence_heatmap(
    sess: HumanSessData,
    eeg_trial: mne.io.Raw,
    et_trial: mne.io.Raw,
    behav: pd.DataFrame,
    trial_N: int,
) -> np.ndarray | None:
    """Create one flattened gaze heatmap from valid sequence-icon fixations."""
    _, gaze_traces = valid_sequence_fixation_onsets_and_gaze(
        sess,
        eeg_trial,
        et_trial,
        behav,
        trial_N,
    )
    if not gaze_traces:
        return None

    gaze = np.concatenate(gaze_traces, axis=1)
    heatmap, _, _ = get_gaze_heatmap(
        gaze[0],
        gaze[1],
        screen_res=c.SCREEN_RESOLUTION,
        bin_size=HEATMAP_BIN_SIZE,
        show=False,
        normalize=True,
    )
    if not np.isfinite(heatmap).all():
        return None
    return heatmap.reshape(-1)


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
        baseline=FRP_BASELINE,
        selected_chans=SELECTED_CHANS,
        return_epochs=True,
    )
    n_fixations = 0 if fixation_epochs is None else len(fixation_epochs)
    return frp, n_fixations


EEG_CONDITIONS = ["frp", "frp_control", "response", "rest"]
NON_EEG_CONDITIONS = ["sequence_heatmap"]
CONDITIONS = EEG_CONDITIONS + NON_EEG_CONDITIONS


def trial_uid(subj_N: int, sess_N: int, trial_N: int) -> str:
    return f"sub-{subj_N:02}_ses-{sess_N:02}_trial-{trial_N:03}"


def row_value(row: pd.Series, key: str, default=None):
    return row[key] if key in row.index else default


def process_subject_trials(
    subj: HumanSubjData,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, mne.Evoked | np.ndarray]], dict[str, dict], list[dict]]:
    condition_data: dict[str, dict[str, mne.Evoked | np.ndarray]] = {
        condition: {} for condition in CONDITIONS
    }
    trial_records: dict[str, dict] = {}
    control_window_rows: list[dict] = []

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

            true_onset_samples, _ = valid_sequence_fixation_onsets_and_gaze(
                sess,
                eeg_trial,
                et_trial,
                behav,
                trial_N,
            )

            frp, n_fixations = extract_true_frp(
                sess, eeg_trial, et_trial, behav, trial_N
            )
            if frp is not None:
                condition_data["frp"][uid] = frp

            if FRP_CONTROL_METHOD == "circular_shift":
                control, control_starts, shift = extract_shifted_control_frp(
                    eeg_trial,
                    true_onset_samples,
                    rng,
                )
            elif FRP_CONTROL_METHOD == "spaced_random":
                control, control_starts = extract_random_control_frp(
                    eeg_trial,
                    n_fixations,
                    rng,
                )
                shift = None
            else:
                raise ValueError(
                    "FRP_CONTROL_METHOD must be one of: circular_shift, spaced_random"
                )
            if control is not None:
                condition_data["frp_control"][uid] = control
            for start in control_starts:
                control_window_rows.append(
                    {
                        "trial_uid": uid,
                        "subj_N": subj.subj_N,
                        "sess_N": sess_N,
                        "trial_N": trial_N,
                        "control_method": FRP_CONTROL_METHOD,
                        "control_onset_sample": start,
                        "control_onset_s": start / eeg_trial.info["sfreq"],
                        "circular_shift_samples": shift,
                        "circular_shift_s": (
                            None if shift is None else shift / eeg_trial.info["sfreq"]
                        ),
                        "min_onset_diff_s": RANDOM_CONTROL_MIN_ONSET_DIFF_S,
                    }
                )

            response = extract_response_epoch(eeg_trial)
            if response is not None:
                condition_data["response"][uid] = response

            rest = extract_rest_epoch(eeg_trial)
            if rest is not None:
                condition_data["rest"][uid] = rest

            heatmap = extract_sequence_heatmap(
                sess, eeg_trial, et_trial, behav, trial_N
            )
            if heatmap is not None:
                condition_data["sequence_heatmap"][uid] = heatmap

    return condition_data, trial_records, control_window_rows


def dataset_trial_uids(trial_records: dict[str, dict]) -> list[str]:
    """Return all trials observed for a participant; missing condition data is filled with NaNs later."""
    return sorted(trial_records)


def complete_trial_uids(
    condition_data: dict[str, dict[str, mne.Evoked | np.ndarray]],
) -> list[str]:
    """Return trials with valid data in every condition, useful for summaries/checks."""
    trial_sets = [set(condition_data[condition].keys()) for condition in CONDITIONS]
    return sorted(set.intersection(*trial_sets))


def condition_to_feature_vector(data: mne.Evoked | np.ndarray) -> np.ndarray:
    """Flatten one condition observation into one feature vector."""
    if isinstance(data, mne.Evoked):
        return data.get_data().reshape(-1)
    return np.asarray(data).reshape(-1)


def sorted_aligned_uids(
    aligned_uids: list[str], trial_records: dict[str, dict]
) -> list[str]:
    """Sort aligned trials by pattern first and item_id second."""
    manifest = pd.DataFrame([trial_records[uid] for uid in aligned_uids]).copy()
    manifest["_uid"] = aligned_uids
    manifest = manifest.sort_values(
        ["pattern", "item_id", "sess_N", "trial_N"],
        kind="mergesort",
        na_position="last",
    )
    return manifest["_uid"].tolist()


def infer_feature_shape(
    condition_trials: dict[str, mne.Evoked | np.ndarray],
) -> tuple[int, ...]:
    """Find a valid trial feature shape for NaN-filled missing observations."""
    for data in condition_trials.values():
        return condition_to_feature_vector(data).shape
    raise ValueError("Could not infer feature shape: no valid condition data found.")


def heatmap_feature_shape() -> tuple[int, ...]:
    screen_width, screen_height = c.SCREEN_RESOLUTION
    return ((screen_width // HEATMAP_BIN_SIZE) * (screen_height // HEATMAP_BIN_SIZE),)


def infer_eeg_feature_shape(
    condition_data: dict[str, dict[str, mne.Evoked | np.ndarray]],
) -> tuple[int, ...]:
    for condition in EEG_CONDITIONS:
        for data in condition_data[condition].values():
            return condition_to_feature_vector(data).shape
    raise ValueError(
        "Could not infer EEG feature shape: no valid EEG condition data found."
    )


def feature_shape_for_condition(
    condition: str,
    condition_data: dict[str, dict[str, mne.Evoked | np.ndarray]],
) -> tuple[int, ...]:
    if condition_data[condition]:
        return infer_feature_shape(condition_data[condition])
    if condition == "sequence_heatmap":
        return heatmap_feature_shape()
    return infer_eeg_feature_shape(condition_data)


def dataset_descriptors(subj_N: int, condition: str, level: str) -> dict:
    is_eeg = condition in EEG_CONDITIONS
    return {
        "subj_N": subj_N,
        "condition": condition,
        "level": level,
        "modality": "eeg" if is_eeg else "gaze",
        "channel_group": CHANNEL_GROUP if is_eeg else "none",
        "channels": ",".join(SELECTED_CHANS)
        if is_eeg and isinstance(SELECTED_CHANS, list)
        else str(SELECTED_CHANS if is_eeg else "none"),
        "frp_baseline": baseline_tag(FRP_BASELINE),
        "response_baseline": baseline_tag(RESPONSE_BASELINE),
        "rest_baseline": baseline_tag(REST_BASELINE),
        "window_s": WINDOW_S,
        "missing_observations": "np.nan",
    }


def build_trial_dataset(
    subj_N: int,
    condition: str,
    condition_trials: dict[str, mne.Evoked | np.ndarray],
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
            measurement_rows.append(condition_to_feature_vector(evoked))
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
    valid_mask = np.isfinite(measurements.reshape(measurements.shape[0], -1)).all(
        axis=1
    )
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
    base = f"{subj_label}-{level}-{condition}-{channel_tag()}-{baseline_tag_for_condition(condition)}"

    np.save(RUN_DIRS["trial_data"] / f"measurements-{base}.npy", measurements)
    manifest.to_csv(RUN_DIRS["metadata"] / f"manifest-{base}.csv", index=False)
    dataset.save(
        RUN_DIRS["datasets"] / f"dataset-{base}.hdf5", file_type="hdf5", overwrite=True
    )
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
    condition_data: dict[str, dict[str, mne.Evoked | np.ndarray]],
    trial_records: dict[str, dict],
    control_window_rows: list[dict],
    aligned_uids: list[str],
) -> dict[str, dict]:
    subj_label = f"subj_{subj_N:02}"
    output_summary: dict[str, dict] = {}
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

    pd.DataFrame(control_window_rows).to_csv(
        RUN_DIRS["trial_data"] / f"{subj_label}-frp_control_windows.csv",
        index=False,
    )

    for condition in CONDITIONS:
        feature_shape = feature_shape_for_condition(condition, condition_data)
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

    condition_data, trial_records, control_window_rows = process_subject_trials(
        subj, rng
    )
    aligned_uids = dataset_trial_uids(trial_records)
    complete_uids = complete_trial_uids(condition_data)

    status_row = {
        "subj_N": int(subj_N),
        "n_trials_total": len(trial_records),
        "n_frp": len(condition_data["frp"]),
        "n_frp_control": len(condition_data["frp_control"]),
        "n_response": len(condition_data["response"]),
        "n_rest": len(condition_data["rest"]),
        "n_sequence_heatmap": len(condition_data["sequence_heatmap"]),
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
        control_window_rows=control_window_rows,
        aligned_uids=aligned_uids,
    )
    status_row["outputs"] = json.dumps(output_summary)
    return status_row


def run_subject_safe(
    subj_N: int, run_dir: str | Path, config: dict
) -> tuple[dict, dict | None]:
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
            "n_sequence_heatmap": np.nan,
            "n_complete": np.nan,
            "n_dataset_trials": np.nan,
            "status": "error",
            "error": repr(exc),
        }
        return status, error


def write_run_outputs(
    run_dirs: dict[str, Path],
    config: dict,
    summary_rows: list[dict],
    errors: list[dict],
) -> pd.DataFrame:
    summary_df = pd.DataFrame(summary_rows).sort_values("subj_N")
    errors_df = pd.DataFrame(errors)
    summary_df.to_csv(run_dirs["logs"] / "experiment1_summary.csv", index=False)
    errors_df.to_csv(run_dirs["logs"] / "experiment1_errors.csv", index=False)
    with open(run_dirs["metadata"] / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return summary_df


def run_experiment(
    subj_ns: list[int], n_workers: int, config: dict, run_dirs: dict[str, Path]
) -> pd.DataFrame:
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
                executor.submit(
                    run_subject_safe, subj_N, run_dirs["run"], config
                ): subj_N
                for subj_N in subj_ns
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Experiment 1 participants",
            ):
                status, error = future.result()
                summary_rows.append(status)
                if error is not None:
                    errors.append(error)
                write_run_outputs(run_dirs, config, summary_rows, errors)

    return write_run_outputs(run_dirs, config, summary_rows, errors)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 1 EEG RSA export pipeline."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--human-data-dir", type=Path, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--output-base-dir", type=Path, default=OUTPUT_BASE_DIR)
    parser.add_argument(
        "--subj-ns",
        type=str,
        default=None,
        help="Comma-separated subject numbers, or 'all'.",
    )
    parser.add_argument(
        "--data-fmt", type=str, default=DATA_FMT, choices=["bids", "original"]
    )
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--channel-group",
        type=str,
        default=CHANNEL_GROUP,
        help="EEG channel group name, `eeg`, or comma-separated channel names.",
    )
    parser.add_argument(
        "--frp-baseline",
        type=str,
        default="none",
        help="MNE-style FRP baseline: none, -0.1,0, None,0, etc.",
    )
    parser.add_argument(
        "--response-baseline",
        type=str,
        default="none",
        help="MNE-style response baseline: none, -0.6,-0.4, etc.",
    )
    parser.add_argument(
        "--rest-baseline",
        type=str,
        default="none",
        help="MNE-style rest baseline: none, -0.6,-0.4, etc.",
    )
    parser.add_argument(
        "--frp-control-method",
        type=str,
        default=FRP_CONTROL_METHOD,
        choices=["circular_shift", "spaced_random"],
    )
    parser.add_argument(
        "--random-control-start-event", type=str, default=RANDOM_CONTROL_START_EVENT
    )
    parser.add_argument(
        "--random-control-end-event", type=str, default=RANDOM_CONTROL_END_EVENT
    )
    parser.add_argument(
        "--random-control-min-onset-diff-s",
        type=float,
        default=RANDOM_CONTROL_MIN_ONSET_DIFF_S,
    )
    parser.add_argument("--heatmap-bin-size", type=int, default=HEATMAP_BIN_SIZE)
    parser.add_argument("--export-evoked-objects", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_arg_parser().parse_args(argv)

    global DATA_ROOT, HUMAN_DATA_DIR, PREPROCESSED_DIR, OUTPUT_BASE_DIR
    global SUBJ_NS, DATA_FMT, RNG_SEED, RANDOM_CONTROL_START_EVENT
    global \
        RANDOM_CONTROL_END_EVENT, \
        RANDOM_CONTROL_MIN_ONSET_DIFF_S, \
        EXPORT_EVOKED_OBJECTS
    global CHANNEL_GROUP, SELECTED_CHANS, FRP_BASELINE, RESPONSE_BASELINE, REST_BASELINE
    global FRP_CONTROL_METHOD, HEATMAP_BIN_SIZE
    global RUN_DIRS

    DATA_ROOT = Path(args.data_root)
    HUMAN_DATA_DIR = (
        Path(args.human_data_dir)
        if args.human_data_dir is not None
        else DATA_ROOT / "Lab/raw-BIDS3"
    )
    PREPROCESSED_DIR = (
        Path(args.preprocessed_dir)
        if args.preprocessed_dir is not None
        else DATA_ROOT / "Lab/preprocessed"
    )
    OUTPUT_BASE_DIR = Path(args.output_base_dir)
    SUBJ_NS = parse_subjects(args.subj_ns)
    DATA_FMT = args.data_fmt
    RNG_SEED = args.rng_seed
    CHANNEL_GROUP = args.channel_group
    SELECTED_CHANS = resolve_channel_group(CHANNEL_GROUP)
    FRP_BASELINE = parse_baseline(args.frp_baseline)
    RESPONSE_BASELINE = parse_baseline(args.response_baseline)
    REST_BASELINE = parse_baseline(args.rest_baseline)
    FRP_CONTROL_METHOD = args.frp_control_method
    RANDOM_CONTROL_START_EVENT = args.random_control_start_event
    RANDOM_CONTROL_END_EVENT = args.random_control_end_event
    RANDOM_CONTROL_MIN_ONSET_DIFF_S = args.random_control_min_onset_diff_s
    HEATMAP_BIN_SIZE = args.heatmap_bin_size
    EXPORT_EVOKED_OBJECTS = bool(args.export_evoked_objects)

    subj_ns = (
        SUBJ_NS
        if SUBJ_NS is not None
        else discover_subject_numbers(HUMAN_DATA_DIR, DATA_FMT)
    )
    RUN_DIRS = create_run_dirs(
        OUTPUT_BASE_DIR,
        f"experiment1_{channel_tag()}_{baseline_tag(FRP_BASELINE)}",
    )
    config = build_config(RUN_DIRS)

    print(f"Run dir:          {RUN_DIRS['run']}")
    print(f"HUMAN_DATA_DIR:   {HUMAN_DATA_DIR} exists={HUMAN_DATA_DIR.exists()}")
    print(f"PREPROCESSED_DIR: {PREPROCESSED_DIR} exists={PREPROCESSED_DIR.exists()}")
    print(f"Subjects:         {subj_ns}")
    print(f"Workers:          {args.n_workers}")
    print(
        f"Channel group:    {CHANNEL_GROUP} ({len(SELECTED_CHANS) if isinstance(SELECTED_CHANS, list) else SELECTED_CHANS})"
    )
    print(f"FRP control:      {FRP_CONTROL_METHOD}")
    print(
        f"Baselines:        frp={FRP_BASELINE}, response={RESPONSE_BASELINE}, rest={REST_BASELINE}"
    )

    summary_df = run_experiment(subj_ns, max(1, args.n_workers), config, RUN_DIRS)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    main()
