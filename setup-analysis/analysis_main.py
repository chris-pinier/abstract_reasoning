# *
# ! TEMP
import os
from pathlib import Path

WD = Path(__file__).parent
os.chdir(WD)
# ! TEMP

import base64
import io
import json
import os
import pickle
import re
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from pprint import pprint
from string import ascii_letters
from typing import Any, Dict, Final, List, Optional, Tuple, Union
import contextlib

import hmp
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import mne.baseline
from mne_icalabel import label_components
import numpy as np
import pandas as pd
import pendulum
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.subplots as ps
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorpac
import tomllib
from analysis_plotting import (
    get_gaze_heatmap,
    plot_eeg,
    plot_eeg_and_gaze_fixations,
    plot_matrix,
    prepare_eeg_data_for_plot,
    show_ch_groups,
)
from analysis_utils import (
    check_ch_groups,
    get_stim_coords,
    get_trial_info,
    locate_trials,
    normalize,
    resample_and_handle_nans,
    resample_eye_tracking_data,
    set_eeg_montage,
    save_pickle,
    load_pickle,
)
from box import Box
from IPython.display import display
from loguru import logger
from mne.preprocessing.eyetracking import read_eyelink_calibration
from PIL import Image
from pyprep.find_noisy_channels import NoisyChannels
from rich import print as rprint
from rich.console import Console as richConsole
from rich.table import Table as richTable
from rich.theme import Theme as richTheme
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from scipy import signal as signal
from scipy.stats import pearsonr, spearmanr
from tensorpac import EventRelatedPac, Pac
from tensorpac.methods import *
from tensorpac.signals import pac_signals_wavelet
from tqdm.auto import tqdm

import logging

# Get the tensorpac logger
tensorpac_logger = logging.getLogger("tensorpac")

# Set the logging level to WARNING or ERROR to suppress INFO messages
tensorpac_logger.setLevel(logging.WARNING)  # Or logging.ERROR

# from autoreject import AutoReject

# TODO: WARNING: unknown channels detected. Dropping:  ['EMG1', 'EMG2', 'EMG3', 'EMG4']: modify appropriate code to adjust when first slot was used instead of 7
# * Packages to look into:
# import mplcursors
# import polars as pl
# import pylustrator

# * ####################################################################################
# * LOADING FILES AND SET GLOBAL VARS
# * ####################################################################################

WD = Path(__file__).parent
os.chdir(WD)

EXP_CONFIG_FILE = WD.parent / "config/experiment_config.toml"
ANAYSIS_CONFIG_FILE = WD.parent / "config/analysis_config.toml"

DATA_DIR = Path("/Users/chris/Documents/PhD-Local/PhD Data/experiment1/data/Lab")

TIMEZONE = "Europe/Amsterdam"
TIMESTAMP = pendulum.now(TIMEZONE).format("YYYYMMDD_HHmmss")

LOG_DIR = WD / "analysis_logs"
LOG_DIR.mkdir(exist_ok=True)

# * Load experiment config
with open(EXP_CONFIG_FILE, "rb") as f:
    EXP_CONFIG = Box(tomllib.load(f))

# * Load analysis config
with open(ANAYSIS_CONFIG_FILE, "rb") as f:
    ANALYSIS_CONFIG = Box(tomllib.load(f))

# * Create an empty notes file
# NOTES_FILE = WD / "notes.json"
# with open(NOTES_FILE, "w") as f:
#     json.dump({}, f)

# * Set backend for MNE and Matplotlib
MNE_BROWSER_BACKEND = "qt"
MPL_BACKEND = "module://matplotlib_inline.backend_inline"  # "ipympl"

# * Random seed
RAND_SEED = 0

# * EEG Montage and Channel Groups
EEG_MONTAGE: mne.channels.DigMontage = mne.channels.make_standard_montage("biosemi64")
EEG_CHAN_GROUPS = ANALYSIS_CONFIG.eeg.ch_groups
ALL_BAD_CHANS = ANALYSIS_CONFIG.eeg.bad_channels
EOG_CHANS = ANALYSIS_CONFIG.eeg.chans.eog
STIM_CHAN = ANALYSIS_CONFIG.eeg.chans.stim
NON_EEG_CHANS = EOG_CHANS + [STIM_CHAN]


# * Sampling Frequencies for EEG and Eye Tracking
EEG_SFREQ: int = 2048
ET_SFREQ: int = 2000

# * Getting Valid Event IDs
VALID_EVENTS = EXP_CONFIG["lab"]["event_IDs"]
VALID_EVENTS_INV = {v: k for k, v in VALID_EVENTS.items()}

# * Number of sequences per session
N_SEQ_PER_SESS = 80

# * Loading images
ICON_IMAGES_DIR = WD.parent / "experiment-Lab/images"
ICON_IMAGES = {img.stem: mpimg.imread(img) for img in ICON_IMAGES_DIR.glob("*.png")}

IMG_SIZE = (256, 256)
SCREEN_RESOLUTION = (2560, 1440)

# * Stimulus positions
Y_POS_CHOICES, Y_POS_SEQUENCE = [-IMG_SIZE[1], IMG_SIZE[1]]
X_POS_STIM = ANALYSIS_CONFIG["stim"]["x_pos_stim"]

PATTERNS = [
    "ABCAABCA",
    "ABBAABBA",
    "ABBACDDC",
    "ABCDEEDC",
    "ABBCABBC",
    "ABCDDCBA",
    "AAABAAAB",
    "ABABCDCD",
]

# * Time bounds (seconds) for separating trials
# * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
PRE_TRIAL_TIME = 1
POST_TRIAL_TIME = 1

# * ####################################################################################
mne.viz.set_browser_backend(MNE_BROWSER_BACKEND)
plt.switch_backend(MPL_BACKEND)
pd.set_option("future.no_silent_downcasting", True)

log_files = list(LOG_DIR.glob("*.log"))

if len(log_files) > 0:
    last_log_file = sorted(log_files)[-1]
    last_log_file_N = int(last_log_file.stem.split("-")[1])
else:
    last_log_file_N = -1

LOG_FILE = LOG_DIR / f"anlysis_log-{last_log_file_N + 1:03}-{TIMESTAMP}.log"
logger.add(LOG_FILE)


# custom_theme = richTheme(
#     {"info": "green", "warning": "bright_white on red1", "danger": "bold red"}
# )
# console = richConsole(theme=custom_theme)

# * ####################################################################################
# * DATA PREPROCESSING AND LOADING
# * ####################################################################################


def load_and_clean_behav_data(data_dir: Union[Path, str], subj_N: int, sess_N: int):
    data_dir = Path(data_dir)

    try:
        behav_file = next(
            data_dir.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )
    except StopIteration:
        # logger.error(f"Behavioral data file not found for subj {subj_N}, sess {sess_N}")
        raise FileNotFoundError(
            f"Behavioral data file not found for subj {subj_N}, sess {sess_N}"
        )

    # assert behav_file.exists(), "Behavioral data file not found"
    behav_data = pd.read_csv(behav_file, index_col=0)

    # sequences_file = wd.parent / f"experiment-Lab/sequences/session_{sess_N}.csv"
    # assert sequences_file.exists(), "Sequences file not found"
    # sequences = pd.read_csv(sequences_file, dtype={"choice_order": str, "seq_order": str})

    sess_date = pendulum.from_format(
        behav_file.stem.split("-")[2], "YYYYMMDD_HHmmss", tz="Europe/Amsterdam"
    )

    behav_data.rename(columns={"subj_id": "subj_N"}, inplace=True)
    behav_data.insert(1, "sess_N", int(sess_N))
    behav_data.insert(2, "trial_N", list(range(len(behav_data))))
    behav_data.insert(3, "block_N", behav_data["blockN"])
    behav_data.insert(behav_data.shape[1], "sess_date", sess_date)

    cols_to_drop = [
        "blockN",
        "trial_type",
        "trial_onset_time",
        "series_end_time",
        "choice_onset_time",
        "rt_global",
    ]
    behav_data.drop(columns=cols_to_drop, inplace=True)

    # * Identify timeout trials and mark them as incorrect
    timeout_trials = behav_data.query("rt=='timeout'")
    behav_data.loc[timeout_trials.index, "correct"] = False
    behav_data.loc[timeout_trials.index, "rt"] = np.nan
    # behav_data.loc[timeout_trials.index, ['choice_key', "choice"]] = "invalid"

    behav_data["rt"] = behav_data["rt"].astype(float)
    behav_data["correct"] = behav_data["correct"].replace(
        {"invalid": False, "True": True, "False": False}
    )
    behav_data["correct"] = behav_data["correct"].astype(bool)

    assert (
        behav_data["correct"].mean()
        == behav_data.query("choice==solution").shape[0] / behav_data.shape[0]
    ), "Error with cleaning of 'Correct' column"

    return behav_data


def load_raw_data(subj_N, sess_N, data_dir, montage, bad_chans=None, logger=None):
    sess_dir = data_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"

    # * File paths
    et_fpath = [f for f in sess_dir.glob("*.asc")][0]
    eeg_fpath = [f for f in sess_dir.glob("*.bdf")][0]
    # behav_fpath = [f for f in sess_dir.glob("*behav*.csv")][0]
    sess_info_file = [f for f in sess_dir.glob("*sess_info.json")][0]
    sequences_file = WD.parent / f"experiment-Lab/sequences/session_{sess_N}.csv"

    # * Load data
    sess_info = json.load(open(sess_info_file))

    if len(sess_info["Notes"]) > 0:
        pass
        # logger.warning(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")
        # with open(NOTES_FILE, "r") as f:
        #     notes = json.load(f)

        # notes.update({f"subj_{subj_N:02}-sess_{sess_N:02}": sess_info["Notes"]})

        # with open(NOTES_FILE, "w") as f:
        #     json.dump(notes, f)

        # print(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")

    sequences = pd.read_csv(
        sequences_file, dtype={"choice_order": str, "seq_order": str}
    )

    # raw_behav = pd.read_csv(behav_fpath).merge(sequences, on="item_id")
    raw_behav = load_and_clean_behav_data(subj_N, sess_N).merge(sequences, on="item_id")

    raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=False, verbose="WARNING")

    # set_eeg_montage(subj_N, sess_N, raw_eeg, montage, eog_chans, bad_chans)

    set_eeg_montage(
        raw_eeg,
        montage,
        EOG_CHANS,
        NON_EEG_CHANS,
        verbose=True,
    )

    raw_eeg.info["bads"] = bad_chans

    raw_et = mne.io.read_raw_eyelink(et_fpath, verbose="WARNING")

    with contextlib.redirect_stdout(io.StringIO()):
        et_cals = read_eyelink_calibration(et_fpath)

    # * Drop unnecessary columns
    # raw_behav.drop(columns=["Unnamed: 0"], inplace=True)

    raw_behav.drop(columns=["pattern_x", "solution_x"], inplace=True)

    raw_behav.rename(
        columns={
            "pattern_y": "pattern",
            "solution_y": "solution",
        },
        inplace=True,
    )

    return sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals


# * ####################################################################################
# * BEHAVIORAL ANALYSIS
# * ####################################################################################


def behav_analysis(subj_N, sess_N, return_raw=False):
    behav_data = load_and_clean_behav_data(subj_N, sess_N)

    acc_by_pattern = behav_data.groupby("pattern")["correct"].mean()
    acc_by_pattern.name = "accuracy"

    rt_by_pattern = behav_data.groupby("pattern")["rt"].mean()

    res_by_pattern = pd.concat([acc_by_pattern, rt_by_pattern], axis=1)

    overall_acc = behav_data["correct"].mean()
    overall_rt = behav_data["rt"].mean()

    rt_by_correct = behav_data.groupby("correct")["rt"].mean()

    rt_by_correct_and_pattern = behav_data.groupby(["pattern", "correct"])["rt"].mean()
    rt_by_correct_and_pattern = rt_by_correct_and_pattern.reset_index()

    res = [
        res_by_pattern,
        overall_acc,
        overall_rt,
        rt_by_correct,
        rt_by_correct_and_pattern,
    ]

    if return_raw:
        res.append(behav_data)

    return res


def behav_analysis_all(return_raw=False):
    behav_files = sorted(DATA_DIR.rglob("*behav*.csv"))

    (
        res_by_pattern,
        overall_acc,
        overall_rt,
        rt_by_correct,
        rt_by_correct_and_pattern,
    ) = [], [], [], [], []

    raw_data = []

    for f in behav_files:
        sess_N, subj_N = [int(d.stem.split("_")[1]) for d in f.parents[:2]]
        # print(subj_N, sess_N)

        res = behav_analysis(subj_N, sess_N, return_raw=return_raw)
        if return_raw:
            raw_data.append(res.pop())

        (
            this_res_by_pattern,
            this_overall_acc,
            this_overall_rt,
            this_rt_by_correct,
            this_rt_by_correct_and_pattern,
        ) = res

        this_res_by_pattern["subj_N"] = subj_N
        this_res_by_pattern["sess_N"] = sess_N

        this_overall_acc = pd.DataFrame(
            [[this_overall_acc, subj_N, sess_N]],
            columns=["accuracy", "subj_N", "sess_N"],
        )

        this_overall_rt = pd.DataFrame(
            [[this_overall_rt, subj_N, sess_N]], columns=["rt", "subj_N", "sess_N"]
        )

        this_rt_by_correct = pd.DataFrame(
            [[this_rt_by_correct[False], this_rt_by_correct[True], subj_N, sess_N]],
            columns=["rt_incorrect", "rt_correct", "subj_N", "sess_N"],
        )

        this_rt_by_correct_and_pattern["subj_N"] = subj_N
        this_rt_by_correct_and_pattern["sess_N"] = sess_N

        res_by_pattern.append(this_res_by_pattern)
        overall_acc.append(this_overall_acc)
        overall_rt.append(this_overall_rt)
        rt_by_correct.append(this_rt_by_correct)
        rt_by_correct_and_pattern.append(this_rt_by_correct_and_pattern)

    res_by_pattern = pd.concat(res_by_pattern).reset_index(drop=False)
    overall_acc = pd.concat(overall_acc).reset_index(drop=True)
    overall_rt = pd.concat(overall_rt).reset_index(drop=True)
    rt_by_correct = pd.concat(rt_by_correct).reset_index(drop=True)
    rt_by_correct_and_pattern = pd.concat(rt_by_correct_and_pattern).reset_index(
        drop=True
    )

    res = [
        res_by_pattern,
        overall_acc,
        overall_rt,
        rt_by_correct,
        rt_by_correct_and_pattern,
    ]

    if return_raw:
        raw_data = pd.concat(raw_data).reset_index(drop=True)
        res.append(raw_data)

    return res


# * ####################################################################################
# * EYE TRACKING ANALYSIS
# * ####################################################################################
def preprocess_et_data(raw_et, et_cals):
    fpath = Path(raw_et.filenames[0])
    subj_N = int(fpath.parents[1].name.split("_")[1])
    sess_N = int(fpath.parents[0].name.split("_")[1])

    if not et_cals:
        print(f"WARNING: NO CALIBRATION FOUND FOR SUBJ {subj_N}, SESS {sess_N}")

    # * Read events from annotations
    et_events, et_events_dict = mne.events_from_annotations(raw_et, verbose="WARNING")

    # * Convert keys to strings (if they aren't already)
    et_events_dict = {str(k): v for k, v in et_events_dict.items()}

    if et_events_dict.get("exp_start"):
        et_events_dict["experiment_start"] = et_events_dict.pop("exp_start")

    # * Create a mapping from old event IDs to new event IDs
    # * that is, adding key-value pairs for events exracted from the eye tracker
    # * i.e., fixation, saccade, blink, etc.

    id_mapping = {}
    eye_events_idx = 60

    for event_name, event_id in et_events_dict.items():
        if event_name in VALID_EVENTS:
            new_id = VALID_EVENTS[event_name]
        else:
            eye_events_idx += 1
            new_id = eye_events_idx
        id_mapping[event_id] = new_id

    # # * Update event IDs in et_events
    for i in range(et_events.shape[0]):
        old_id = et_events[i, 2]
        if old_id in id_mapping:
            et_events[i, 2] = id_mapping[old_id]

    # * Update et_events_dict with new IDs
    et_events_dict = {k: id_mapping[v] for k, v in et_events_dict.items()}
    et_events_dict = {
        k: v for k, v in sorted(et_events_dict.items(), key=lambda x: x[1])
    }
    et_events_dict_inv = {v: k for k, v in et_events_dict.items()}

    inds_responses = np.where(np.isin(et_events[:, 2], [10, 11, 12, 13, 14, 15, 16]))
    choice_key_et = [VALID_EVENTS_INV[i] for i in et_events[inds_responses, 2][0]]

    et_events_df = pd.DataFrame(et_events, columns=["sample_nb", "prev", "event_id"])
    et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

    # print("Eye tracking event counts:")
    # display(et_events_df["event_id"].value_counts())

    et_trial_bounds, et_trial_events_df = locate_trials(et_events, et_events_dict)

    # * Remove practice trials
    if sess_N == 1:
        choice_key_et = choice_key_et[3:]
        et_trial_bounds = et_trial_bounds[3:]
        et_trial_events_df = et_trial_events_df.query("trial_id >= 3").copy()
        et_trial_events_df["trial_id"] -= 3

    manual_et_epochs = []
    for start, end in tqdm(et_trial_bounds, desc="Creating ET epochs"):
        # * Get start and end times in seconds
        start_time = (et_events[start, 0] / raw_et.info["sfreq"]) - PRE_TRIAL_TIME
        end_time = et_events[end, 0] / raw_et.info["sfreq"] + POST_TRIAL_TIME

        # * Crop the raw data to this time window
        epoch_data = raw_et.copy().crop(tmin=start_time, tmax=end_time)

        # * Add this epoch to our list
        manual_et_epochs.append(epoch_data)

    # * Print some information about our epochs
    # print(f"Number of epochs created: {len(manual_et_epochs)}")
    # for i, epoch in enumerate(manual_et_epochs):
    #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

    # assert (
    #     len(manual_et_epochs) == 80
    # ), "Incorrect number of epochs created, should be 80"

    manual_et_epochs = (et_trial for et_trial in manual_et_epochs)

    return (
        manual_et_epochs,
        et_events_dict,
        et_events_dict_inv,
        et_trial_bounds,
        et_trial_events_df,
    )


# * ####################################################################################
# * EEG ANALYSIS
# * ####################################################################################


def get_response_ERP(eeg_data, show=False):
    raise NotImplementedError
    # # eeg_chan_inds = mne.pick_types(prepro_eeg.info, eeg=True, eog=False, stim=False)
    # # eeg_chans = [prepro_eeg.ch_names[i] for i in eeg_chan_inds]
    # # bad_eeg_chans = prepro_eeg.info["bads"]

    # flash_duration = 0.6

    # events, event_ids = mne.events_from_annotations(eeg_data, verbose="WARNING")

    # flash_event_ids = [
    #     event_ids[event] for event in ["stim-flash_sequence", "stim-flash_choices"]
    # ]

    # tmin = 0.1
    # tmax = flash_duration
    # baseline = None

    # epochs_flashes = mne.Epochs(
    #     eeg_data,
    #     events=events,
    #     event_id=flash_event_ids,
    #     tmin=tmin,
    #     tmax=tmax,
    #     baseline=baseline,
    # )

    # evoked_flashes = epochs_flashes.average()

    # if show:
    #     evoked_flashes.plot_joint(times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    #     evoked_flashes.plot()

    # return evoked_flashes


def get_flashes_ERP(eeg_data, show=False):
    # eeg_chan_inds = mne.pick_types(prepro_eeg.info, eeg=True, eog=False, stim=False)
    # eeg_chans = [prepro_eeg.ch_names[i] for i in eeg_chan_inds]
    # bad_eeg_chans = prepro_eeg.info["bads"]

    flash_duration = 0.6

    events, event_ids = mne.events_from_annotations(eeg_data, verbose="WARNING")

    flash_event_ids = [
        event_ids[event] for event in ["stim-flash_sequence", "stim-flash_choices"]
    ]

    tmin = 0.1
    tmax = flash_duration
    baseline = None

    epochs_flashes = mne.Epochs(
        eeg_data,
        events=events,
        event_id=flash_event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
    )

    evoked_flashes = epochs_flashes.average()

    if show:
        evoked_flashes.plot_joint(times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        evoked_flashes.plot()

    return evoked_flashes


def reject_eeg_chans_procedure(raw_eeg):
    eeg_chan_inds = mne.pick_types(raw_eeg.info, eeg=True)
    eeg_chans = [raw_eeg.ch_names[i] for i in eeg_chan_inds]
    # bad_chans = raw_eeg.info["bads"]
    # good_chans = [c for c in eeg_chans if c not in bad_chans]

    evoked = get_flashes_ERP(raw_eeg)
    evoked_data = evoked.get_data(eeg_chans)

    maybe_bad_chans = []

    for chan_ind in range(evoked_data.shape[0]):
        chan_data = evoked_data[chan_ind]
        other_chans_data = np.delete(evoked_data, chan_ind, axis=0)

        min_chan = chan_data.min(axis=0)
        max_chan = chan_data.max(axis=0)

        min_other_chans = other_chans_data.min(axis=0)
        max_other_chans = other_chans_data.max(axis=0)

        # * Check if the channel is suspicious
        # * i.e, if it has a higher or lower amplitude than other channels on 90% of the time points
        suspicious_high = np.mean(min_chan < min_other_chans) > 0.9
        suspicious_low = np.mean(max_chan > max_other_chans) > 0.9

        # * If the channel is suspicious, add it to the list of maybe bad channels
        if suspicious_high or suspicious_low:
            maybe_bad_chans.append(eeg_chans[chan_ind])

    if len(maybe_bad_chans) > 0:
        print("Maybe bad channels:")
        print(maybe_bad_chans)

    # fig, ax = plt.subplot()
    # maybe_bad_chan_inds = [eeg_chans.index(c) for c in maybe_bad_chans]

    # evoked.plot()
    # evoked.plot(picks=[c for c in eeg_chans if c not in maybe_bad_chans])
    # evoked.plot(picks=maybe_bad_chans)

    return maybe_bad_chans


def preprocess_eeg_data(raw_eeg, eeg_chan_groups, raw_behav, preprocessed_dir=None):
    # ! TEMP
    # eeg_chan_groups = EEG_CHAN_GROUPS
    # ! TEMP

    fpath = Path(raw_eeg.filenames[0])
    subj_N = int(fpath.parents[1].name.split("_")[1])
    sess_dir = fpath.parents[0]
    sess_N = int(sess_dir.name.split("_")[1])

    preprocessed_dir = WD / "results/preprocessed_data/"
    preprocessed_dir.mkdir(exist_ok=True)

    preprocessed_raw_fpath = (
        preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_preprocessed-raw.fif"
    )

    if not preprocessed_raw_fpath.exists():
        print("Preprocessing raw data...")

        # * Setting EOG channels
        raw_eeg.load_data(verbose="WARNING")

        # * Detecting events
        eeg_events = mne.find_events(
            raw_eeg,
            min_duration=0,
            initial_event=False,
            shortest_event=1,
            uint_cast=True,
            verbose="WARNING",
        )

        # # ! TEMP
        # df = pd.DataFrame(eeg_events, columns=["sample_nb", "prev", "event_id"])
        # df["event_id"] = df["event_id"].replace(VALID_EVENTS_INV)
        # ! TEMP

        # * Get annotations from events and add them to the raw data
        annotations = mne.annotations_from_events(
            eeg_events,
            raw_eeg.info["sfreq"],
            event_desc=VALID_EVENTS_INV,
            verbose="WARNING",
        )

        raw_eeg.set_annotations(annotations, verbose="WARNING")

        bad_chans = raw_eeg.info["bads"]

        manually_set_bad_chans = ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(
            f"sess_{sess_N}"
        )

        if not bad_chans == manually_set_bad_chans:
            print(
                "WARNING: raw EEG bad channels do not match expected bad channels, combining them"
            )

        bad_chans = list(set(bad_chans) | set(manually_set_bad_chans))

        raw_eeg.info["bads"] = bad_chans

        # raw_eeg.drop_channels(bad_chans)

        # * Check if channel groups include all channels present in the montage
        # * i.e., that there are no "orphan" channels
        check_ch_groups(raw_eeg.get_montage(), eeg_chan_groups)

        # * Average Reference
        raw_eeg = raw_eeg.set_eeg_reference(ref_channels="average", verbose="WARNING")

        # * ############################################################################
        # * EOG artifact rejection using ICA
        # * ############################################################################

        ica_fpath = preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_fitted-ica.fif"

        if ica_fpath.exists():
            # * if ICA file exists, load it
            ica = mne.preprocessing.read_ica(ica_fpath)

        else:
            # * Create a copy of the raw data to hihg-pass filter at 1Hz before ICA
            # * as recommended by MNE: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
            # raw_eeg_copy_for_ica = raw_eeg.copy()
            raw_eeg.filter(l_freq=1, h_freq=100, verbose="WARNING")

            ica = mne.preprocessing.ICA(
                n_components=None,
                noise_cov=None,
                random_state=RAND_SEED,
                # method="fastica",
                method="infomax",
                fit_params=dict(extended=True),
                max_iter="auto",
                verbose="WARNING",
            )

            ica.fit(raw_eeg, verbose="WARNING")

            ica.save(ica_fpath, verbose="WARNING")

        # eog_inds, eog_scores = ica.find_bads_eog(raw_eeg)
        # ica.exclude = eog_inds

        # # * Label components using IClabel
        ic_labels = label_components(raw_eeg, ica, method="iclabel")

        # df_labels = pd.DataFrame(
        #     list(zip(list(ic_labels["y_pred_proba"]), ic_labels["labels"])), columns=['prob', 'label']
        # ).sort_values(by='prob', ascending=False)

        # # * Get indices of components labeled as 'brain'
        brain_ic_indices = [
            idx for idx, label in enumerate(ic_labels["labels"]) if label == "brain"
        ]

        # # * Keep only brain components, effectively rejecting artifactual ones
        ica.exclude = [
            idx for idx in range(ica.n_components_) if idx not in brain_ic_indices
        ]

        # * Apply ICA to raw data
        raw_eeg = ica.apply(raw_eeg, verbose="WARNING")

        # TODO: automatically remove bad channels (e.g., amplitude cutoff)

        # * Bandpass Filter: 0.1 - 100 Hz
        # raw_eeg.filter(l_freq=0.1, h_freq=100, verbose="WARNING")
        raw_eeg.notch_filter(freqs=50, verbose="WARNING")
        raw_eeg.notch_filter(freqs=100, verbose="WARNING")

        # * Save preprocessed raw data
        raw_eeg.save(preprocessed_raw_fpath, overwrite=True, verbose="WARNING")

        del raw_eeg

    raw_eeg = mne.io.read_raw_fif(
        preprocessed_raw_fpath, preload=False, verbose="WARNING"
    )
    eeg_events, _ = mne.events_from_annotations(
        raw_eeg, VALID_EVENTS, verbose="WARNING"
    )

    choice_key_eeg = [
        VALID_EVENTS_INV[i]
        for i in eeg_events[:, 2]
        if i in [10, 11, 12, 13, 14, 15, 16]
    ]

    eeg_trial_bounds, eeg_events_df = locate_trials(eeg_events, VALID_EVENTS)

    # * Remove practice trials
    if sess_N == 1:
        choice_key_eeg = choice_key_eeg[3:]
        eeg_trial_bounds = eeg_trial_bounds[3:]

    if not len(choice_key_eeg) == len(eeg_trial_bounds) == 80:
        raise ValueError(
            "Error with EEG events: incorrect number of trials.\n"
            f"{len(choice_key_eeg) = }\n{len(eeg_trial_bounds) = }"
        )

    raw_behav["choice_key_eeg"] = choice_key_eeg
    raw_behav["same"] = raw_behav["choice_key"] == raw_behav["choice_key_eeg"]

    manual_eeg_trials = []

    # * Loop through each trial
    for start, end in tqdm(eeg_trial_bounds, "Creating EEG epochs"):
        # * Get start and end times in seconds
        start_time = (eeg_events[start, 0] / raw_eeg.info["sfreq"]) - PRE_TRIAL_TIME
        end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + POST_TRIAL_TIME

        # * Crop the raw data to this time window
        epoch_data = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)

        # * Add this epoch to our list
        manual_eeg_trials.append(epoch_data)

    # * Print some information about our epochs
    # print(f"Number of epochs created: {len(manual_eeg_trials)}")
    # for i, epoch in enumerate(manual_eeg_trials):
    #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")
    manual_eeg_trials = (trial for trial in manual_eeg_trials)

    return manual_eeg_trials, eeg_trial_bounds, eeg_events, eeg_events_df


# * ####################################################################################
# * CREATE RDMs
# * ####################################################################################


def get_neg_erp_peak(
    evoked: mne.Evoked,
    time_window: tuple,
    selected_chans: Optional[List[str]] = None,
    unit: str = "uV",
    plot: bool = False,
) -> Tuple[float, float]:
    """_summary_

    Args:
        evoked (mne.Evoked): _description_
        time_window (tuple): _description_
        selected_chans (Optional[List[str]], optional): _description_. Defaults to None.
        unit (str, optional): _description_. Defaults to "uV".
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[float, float]: peak_latency, peak_amplitude
    """

    # * Get data for the time window
    times = evoked.times
    chans = evoked.ch_names

    # * if no channels are selected, use all channels
    selected_chans = selected_chans if selected_chans else chans
    selected_chans = [c for c in chans if c in selected_chans]

    # * Remove bad channels
    selected_chans = [c for c in selected_chans if c not in evoked.info["bads"]]

    # *
    time_mask = (times >= time_window[0]) & (times <= time_window[1])

    # *
    data = evoked.get_data(picks=selected_chans, units=unit).mean(axis=0)

    # * Find the first negative peak
    negative_peak_idx = np.argmin(data[time_mask])
    peak_latency = times[time_mask][negative_peak_idx]
    peak_amplitude = data[time_mask][negative_peak_idx]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(times, data, lw=2, label="Selected Channels")
        ax.plot(
            times,
            evoked.get_data(
                picks=[c for c in chans if c not in selected_chans], units=unit
            ).mean(axis=0),
            color="gray",
            alpha=0.8,
            label="Other Channels",
        )

        ax.axvline(peak_latency, color="r", ls="--", lw=1.25)
        ax.axhline(peak_amplitude, color="r", ls="--", lw=1.25)
        ax.vlines(time_window, data.min(), data.max(), color="k", ls="--", lw=1.25)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Amplitude ({unit})")
        ax.set_title(
            f"Peak Latency: {peak_latency:.3f}s, Peak Amplitude: {peak_amplitude:.3f}uV"
        )
        ax.legend(bbox_to_anchor=(1.005, 1), loc="upper left", borderaxespad=0)

        plt.tight_layout()
        plt.show()
        plt.close()

    return peak_latency, peak_amplitude


def compare_rdms(rdm1, rdm2, method="pearson"):
    avail_methods = ["pearson", "spearman"]
    if method not in avail_methods:
        raise ValueError(f"Method should be either {avail_methods}")

    # * Extract the upper triangle (excluding the diagonal)
    rdm1_flattened = rdm1[np.triu_indices_from(rdm1, k=1)]
    rdm2_flattened = rdm2[np.triu_indices_from(rdm2, k=1)]

    if method == "pearson":
        # * Pearson correlation
        corr, _ = pearsonr(rdm1_flattened, rdm2_flattened)
    else:
        # * Spearman correlation
        corr, _ = spearmanr(rdm1_flattened, rdm2_flattened)

    return corr


def get_rdms_negative_peak_eeg(
    erp_data,
    selected_chans,
    time_window=(0, 0.2),
    unit="uV",
    method="euclidean",
    show_plots=True,
):
    # method = 'euclidean' # ! TEMP
    # erp_data = subj_pattern_erps.copy() # ! TEMP

    participants = list(erp_data.keys())
    patterns = list(erp_data[participants[0]].keys())

    # * Sorting by pattern name in alaphabetical order
    for participant, patterns_data in erp_data.items():
        erp_data[participant] = dict(sorted(patterns_data.items(), key=lambda x: x[0]))

    # * Extract latency and amplitude features for each pattern type and participant
    latency_features = {}
    amplitude_features = {}

    for participant, patterns_data in erp_data.items():
        latency_features[participant] = []
        amplitude_features[participant] = []

        for pattern_name, evoked in patterns_data.items():
            peak_latency, peak_amplitude = get_neg_erp_peak(
                evoked, time_window, selected_chans, unit
            )

            # * Append results
            latency_features[participant].append(peak_latency)
            amplitude_features[participant].append(peak_amplitude)

    combined_features = np.vstack(
        [(latency_features[p], amplitude_features[p]) for p in participants]
    ).T

    # * Transpose to have patterns as rows and features as columns

    # * Compute RDMs separately for latency and amplitude
    latency_rdms = {}
    amplitude_rdms = {}
    combined_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        latency_dataset = Dataset(
            measurements=np.array(latency_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        amplitude_dataset = Dataset(
            measurements=np.array(amplitude_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Convert to Dataset for RDM calculation
        combined_dataset = Dataset(
            measurements=combined_features, obs_descriptors={"pattern": patterns}
        )

        # * Compute RDMs
        latency_rdms[participant] = calc_rdm(latency_dataset, method=method)
        amplitude_rdms[participant] = calc_rdm(amplitude_dataset, method=method)
        combined_rdms[participant] = calc_rdm(combined_dataset, method=method)

    if show_plots:
        for participant in latency_rdms.keys():
            latency_rdm = latency_rdms[participant].get_matrices()[0]
            amplitude_rdm = amplitude_rdms[participant].get_matrices()[0]
            combined_rdm = combined_rdms[participant].get_matrices()[0]

            titles = [
                f"Latency RDM for Participant {participant}",
                f"Amplitude RDM for Participant {participant}",
                f"Combined RDM for Participant {participant}",
            ]

            data = [latency_rdm, amplitude_rdm, combined_rdm]

            fig, axes = plt.subplots(1, 3, figsize=(12, 10))
            for ax, title, data in zip(axes, titles, data):
                plot_matrix(
                    data,
                    labels=patterns,
                    title=title,
                    show_values=True,
                    norm="max",
                    as_pct=True,
                    ax=ax,
                )

            axes[1].set_yticklabels([])
            axes[2].set_yticklabels([])

            plt.show()
            plt.tight_layout()
            plt.close()

    return latency_rdms, amplitude_rdms, combined_rdms


def get_rdms_behavior_pattern_groups(
    behav_data: pd.DataFrame, method="euclidean", show_plots=True
):
    """_summary_

    Args:
        behav_data (pd.DataFrame): _description_
        method (str, optional): _description_. Defaults to "euclidean".
    """

    cleaned_behav_data = behav_data.copy()

    timeout_trials = cleaned_behav_data.query("choice == 'timeout'")
    cleaned_behav_data.loc[timeout_trials.index, "correct"] = "False"
    cleaned_behav_data.drop(
        cleaned_behav_data.query("correct == 'invalid'").index, inplace=True
    )
    cleaned_behav_data["correct"].replace({"True": 1, "False": 0}, inplace=True)

    participants = [int(i) for i in behav_data["subj_N"].unique()]
    patterns = list(behav_data["pattern"].unique())

    cleaned_behav_data = (
        cleaned_behav_data.groupby(["subj_N", "pattern"])["correct"]
        .mean()
        .unstack()
        .T.sort_index()
        .to_dict()
    )

    behav_features: Dict[int, List[float]] = {}

    for participant in participants:
        behav_features[participant] = []
        for pattern, mean_accuracy in cleaned_behav_data[participant].items():
            behav_features[participant].append(mean_accuracy)

    behav_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        behav_dataset = Dataset(
            measurements=np.array(behav_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Compute RDMs
        behav_rdms[participant] = calc_rdm(behav_dataset, method=method)

        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            rdm = behav_rdms[participant].get_matrices()[0]
            plot_matrix(
                rdm,
                labels=patterns,
                title=f"Gaze RDM for Participant {participant}",
                show_values=True,
                norm="max",
                as_pct=True,
                ax=ax,
            )

            ax.set_yticklabels([])
            plt.show()
            plt.tight_layout()
            plt.close()

    return behav_rdms


def get_rdms_eye_gaze(gaze_data: pd.DataFrame, method="euclidean", show_plots=True):
    gaze_duration_for_sequences = (
        gaze_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern"])["mean_duration"]
        .mean()
        .unstack()
    ).T.to_dict()

    participants = list(gaze_duration_for_sequences.keys())
    patterns = list(gaze_duration_for_sequences[participants[0]].keys())

    gaze_features: dict = {}
    for participant in participants:
        gaze_features[participant] = []
        for pattern, mean_duration in gaze_duration_for_sequences[participant].items():
            gaze_features[participant].append(mean_duration)

    # * Normalize
    # normalized_gaze_features = {p: normalize(feat) for p, feat in gaze_features.items()}
    # gaze_features = normalized_gaze_features

    gaze_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        behav_dataset = Dataset(
            measurements=np.array(gaze_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Compute RDMs
        gaze_rdms[participant] = calc_rdm(behav_dataset, method=method)

        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            rdm = gaze_rdms[participant].get_matrices()[0]
            plot_matrix(
                rdm,
                labels=patterns,
                title=f"Gaze RDM for Participant {participant}",
                show_values=True,
                norm="max",
                as_pct=True,
                ax=ax,
            )
            ax.set_yticklabels([])
            plt.show()
            plt.tight_layout()
            plt.close()

    return gaze_rdms


def get_rdms_eye_pupil_diam(
    gaze_data: pd.DataFrame, method="euclidean", show_plots=True
):
    pupil_diam_for_sequences = (
        gaze_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern"])["mean_pupil_diam"]
        .mean()
        .unstack()
    ).T.to_dict()

    participants = list(pupil_diam_for_sequences.keys())
    patterns = list(pupil_diam_for_sequences[participants[0]].keys())

    pupil_diam_features: dict = {}
    for participant in participants:
        pupil_diam_features[participant] = []
        for pattern, mean_duration in pupil_diam_for_sequences[participant].items():
            pupil_diam_features[participant].append(mean_duration)

    # * Normalize
    # normalized_gaze_features = {p: normalize(feat) for p, feat in gaze_features.items()}
    # gaze_features = normalized_gaze_features

    pupil_diam_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        behav_dataset = Dataset(
            measurements=np.array(pupil_diam_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Compute RDMs
        pupil_diam_rdms[participant] = calc_rdm(behav_dataset, method=method)

        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            rdm = pupil_diam_rdms[participant].get_matrices()[0]

            plot_matrix(
                rdm,
                labels=patterns,
                title=f"Pupil Diam RDM for Participant {participant}",
                show_values=True,
                norm="max",
                as_pct=True,
                ax=ax,
            )
            ax.set_yticklabels([])
            plt.show()
            plt.tight_layout()
            plt.close()

    return pupil_diam_rdms


def get_rdms_rt(behav_data: pd.DataFrame, method="euclidean", show_plots=True):
    rt_for_sequences = behav_data.copy()

    # * Identify timeout trials and mark them as incorrect
    # TODO: this is repeated from load_and_clean_behav_data(), should be refactored
    timeout_trials = rt_for_sequences.query("rt=='timeout'")
    rt_for_sequences.loc[timeout_trials.index, "correct"] = False
    rt_for_sequences.loc[timeout_trials.index, "rt"] = np.nan
    # rt_for_sequences.loc[timeout_trials.index, ['choice_key', "choice"]] = "invalid"

    rt_for_sequences["rt"] = rt_for_sequences["rt"].astype(float)
    rt_for_sequences["correct"] = rt_for_sequences["correct"].replace(
        {"invalid": False, "True": True, "False": False}
    )
    rt_for_sequences["correct"] = rt_for_sequences["correct"].astype(bool)

    rt_for_sequences = (
        rt_for_sequences.groupby(["subj_N", "pattern"])["rt"]
        .mean()
        .unstack()
        .T.to_dict()
    )

    participants = list(rt_for_sequences.keys())
    patterns = list(rt_for_sequences[participants[0]].keys())

    # rt_features = rt_for_sequences
    rt_features: dict = {}
    for participant in participants:
        rt_features[participant] = []
        for pattern, mean_rt in rt_for_sequences[participant].items():
            rt_features[participant].append(mean_rt)

    # * Normalize
    # normalized_gaze_features = {p: normalize(feat) for p, feat in gaze_features.items()}
    # gaze_features = normalized_gaze_features

    rt_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        rt_dataset = Dataset(
            measurements=np.array(rt_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Compute RDMs
        rt_rdms[participant] = calc_rdm(rt_dataset, method=method)

        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            rdm = rt_rdms[participant].get_matrices()[0]

            plot_matrix(
                rdm,
                labels=patterns,
                title=f"Pupil Diam RDM for Participant {participant}",
                show_values=True,
                norm="max",
                as_pct=True,
                ax=ax,
            )
            ax.set_yticklabels([])
            plt.show()
            plt.tight_layout()
            plt.close()

    return rt_rdms


def get_rdms_pac():
    # TODO: implement
    raise NotImplementedError


def get_rdm_negative_peak_eeg(
    erp_data,
    selected_chans,
    time_window=(0, 0.2),
    unit="uV",
    method="euclidean",
    show_plots=True,
):
    # method = 'euclidean' # ! TEMP
    # erp_data = subj_pattern_erps.copy() # ! TEMP

    # * Sorting by pattern name in alaphabetical order
    for participant, patterns_data in erp_data.items():
        erp_data[participant] = dict(sorted(patterns_data.items(), key=lambda x: x[0]))

    participants = list(erp_data.keys())
    patterns = list(erp_data[participants[0]].keys())

    # * Extract latency and amplitude features for each pattern type and participant
    latency_features = {}
    amplitude_features = {}

    for participant, patterns_data in erp_data.items():
        latency_features[participant] = []
        amplitude_features[participant] = []

        for pattern_name, evoked in patterns_data.items():
            peak_latency, peak_amplitude = get_neg_erp_peak(
                evoked, time_window, selected_chans, unit
            )

            # * Append results
            latency_features[participant].append(peak_latency)
            amplitude_features[participant].append(peak_amplitude)

    combined_features = np.vstack(
        [(latency_features[p], amplitude_features[p]) for p in participants]
    ).T

    # * Transpose to have patterns as rows and features as columns

    # * Compute RDMs separately for latency and amplitude
    latency_rdms = {}
    amplitude_rdms = {}
    combined_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        latency_dataset = Dataset(
            measurements=np.array(latency_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        amplitude_dataset = Dataset(
            measurements=np.array(amplitude_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Convert to Dataset for RDM calculation
        combined_dataset = Dataset(
            measurements=combined_features, obs_descriptors={"pattern": patterns}
        )

        # * Compute RDMs
        latency_rdms[participant] = calc_rdm(latency_dataset, method=method)
        amplitude_rdms[participant] = calc_rdm(amplitude_dataset, method=method)
        combined_rdms[participant] = calc_rdm(combined_dataset, method=method)

    if show_plots:
        for participant in latency_rdms.keys():
            latency_rdm = latency_rdms[participant].get_matrices()[0]
            amplitude_rdm = amplitude_rdms[participant].get_matrices()[0]
            combined_rdm = combined_rdms[participant].get_matrices()[0]

            titles = [
                f"Latency RDM for Participant {participant}",
                f"Amplitude RDM for Participant {participant}",
                f"Combined RDM for Participant {participant}",
            ]

            data = [latency_rdm, amplitude_rdm, combined_rdm]

            fig, axes = plt.subplots(1, 3, figsize=(12, 10))
            for ax, title, data in zip(axes, titles, data):
                plot_matrix(
                    data,
                    labels=patterns,
                    title=title,
                    show_values=True,
                    norm="max",
                    as_pct=True,
                    ax=ax,
                )

            axes[1].set_yticklabels([])
            axes[2].set_yticklabels([])

            plt.show()
            plt.tight_layout()
            plt.close()

    return latency_rdms, amplitude_rdms, combined_rdms


# * ####################################################################################
# * EEG ANALYSIS
# * ####################################################################################
def analyze_flash_period(et_epoch, eeg_epoch, raw_behav, epoch_N):
    # ! IMPORTANT
    # TODO: can't remember, something related to eeg baseline ?
    # * eeg_baseline

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target
        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

        # * Convert to seconds
        start_time = (
            annotations.iloc[: first_flash.name]
            .query("description == 'fixation'")
            .iloc[-1]["onset"]
        )
        end_time = all_stim_pres["onset"]

        # * Crop the data
        epoch = epoch.copy().crop(start_time, end_time)

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        time_bounds = (start_time, end_time)

        return epoch, annotations, time_bounds

    et_epoch, et_annotations, time_bounds = crop_et_epoch(et_epoch)

    eeg_epoch = eeg_epoch.copy().crop(*time_bounds)

    # * Get channel positions for topomap
    info = eeg_epoch.info

    chans_pos_xy = np.array(
        list(info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    # * Get channel indices for each region
    ch_group_inds = {
        group_name: [i for i, ch in enumerate(eeg_epoch.ch_names) if ch in group_chans]
        for group_name, group_chans in EEG_CHAN_GROUPS.items()
    }

    # * Get positions and presentation order of stimuli
    trial_info = trial_info = get_trial_info(
        epoch_N,
        raw_behav,
        X_POS_STIM,
        Y_POS_CHOICES,
        Y_POS_SEQUENCE,
        SCREEN_RESOLUTION,
        IMG_SIZE,
    )
    stim_pos, stim_order = trial_info[:2]

    flash_event_ids = ["stim-flash_sequence", "stim-flash_choices"]

    fixation_inds = et_annotations.query("description == 'fixation'").index

    fixation_data = {i: [] for i in stim_order}
    eeg_fixation_data = {i: [] for i in stim_order}

    for fixation_ind in fixation_inds:
        # * Get number of flash events before the current fixation; -1 to get the index
        flash_events = et_annotations.iloc[:fixation_ind].query(
            f"description.isin({flash_event_ids})"
        )

        n_flash_events = flash_events.shape[0]
        stim_flash_ind = n_flash_events - 1

        # * If first fixation is before the first flash, stim_flash_ind will be -1
        # * We set it to 0 in this case, and check if fixation is already on target
        stim_flash_ind = max(stim_flash_ind, 0)

        # * Get target location
        target_grid_loc = stim_order[stim_flash_ind]
        target_name, target_coords = stim_pos[target_grid_loc]
        targ_left, targ_right, targ_bottom, targ_top = target_coords

        fixation = et_annotations.loc[fixation_ind]

        fixation_start = fixation["onset"]
        fixation_duration = fixation["duration"]
        fixation_stop = fixation_start + fixation_duration

        end_time = min(fixation_stop, et_epoch.times[-1])

        gaze_x, gaze_y, pupil_diam = (
            et_epoch.copy().crop(fixation_start, end_time).get_data()
        )

        mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

        on_target = (targ_left <= mean_gaze_x <= targ_right) and (
            targ_bottom <= mean_gaze_y <= targ_top
        )

        end_time = min(fixation_stop, eeg_epoch.times[-1])

        eeg_slice = eeg_epoch.copy().crop(fixation_start, end_time)

        # eeg_annotations = eeg_slice.annotations.to_data_frame(time_format="ms")
        # mne_events, _ = mne.events_from_annotations(eeg_slice)
        # eeg_annotations.insert(1, 'sample_nb', mne_events[:, 0])

        eeg_fixation_data[stim_flash_ind].append(eeg_slice)

        eeg_slice = eeg_slice.copy().pick(["eeg"]).get_data()

        if fixation_duration >= 0.2 and on_target:
            discarded = False
            fixation_data[stim_flash_ind].append(np.array([gaze_x, gaze_y]))
        else:
            discarded = True

        title = f"FLASH-{stim_flash_ind} ({fixation_duration * 1000:.0f} ms)"
        title += " - " + ("DISCARDED" if discarded else "SAVED")

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[1, 1])
        ax_et = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_eeg = fig.add_subplot(gs[1, :])
        ax_eeg_group = fig.add_subplot(gs[2, :], sharex=ax_eeg)
        ax_eeg_avg = fig.add_subplot(gs[3, :], sharex=ax_eeg)

        ax_et.set_xlim(0, SCREEN_RESOLUTION[0])
        ax_et.set_ylim(SCREEN_RESOLUTION[1], 0)
        ax_et.set_title(title)

        # * Plot target icon
        ax_et.imshow(
            ICON_IMAGES[target_name],
            extent=[targ_left, targ_right, targ_bottom, targ_top],
            origin="lower",
        )

        mne.viz.plot_topomap(
            eeg_slice.mean(axis=1),
            chans_pos_xy,
            ch_type="eeg",
            sensors=True,
            contours=0,
            outlines="head",
            sphere=None,
            image_interp="cubic",
            extrapolate="auto",
            border="mean",
            res=640,
            size=1,
            cmap=None,
            vlim=(None, None),
            cnorm=None,
            axes=ax_topo,
            show=False,
        )

        # * Plot rectangle around target, with dimensions == img_size
        rectangle = mpatches.Rectangle(
            (targ_left, targ_bottom),
            IMG_SIZE[0],
            IMG_SIZE[1],
            linewidth=1,
            linestyle="--",
            edgecolor="black",
            facecolor="none",
        )
        ax_et.add_patch(rectangle)

        # * Plot gaze data
        ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
        ax_et.scatter(mean_gaze_x, mean_gaze_y, c="yellow", s=3)

        # * Plot EEG data
        ax_eeg.plot(eeg_slice.T)

        # ax_eeg.vlines(eeg_annotations["sample_nb"], eeg_slice.T.min(), eeg_slice.T.max())

        ax_eeg_avg.plot(eeg_slice.mean(axis=0))

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["occipital"]].mean(axis=0),
            color="red",
            label="occipital",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["parietal"]].mean(axis=0),
            color="green",
            label="parietal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["centro-parietal"]].mean(axis=0),
            color="purple",
            label="centro-parietal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["temporal"]].mean(axis=0),
            color="orange",
            label="temporal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["frontal"]].mean(axis=0),
            color="blue",
            label="frontal",
        )

        ax_eeg_group.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg.set_xlim(0, eeg_slice.shape[1])
        xticks = np.arange(0, eeg_slice.shape[1], 100)
        ax_eeg.set_xticks(xticks, ((xticks / EEG_SFREQ) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    return fixation_data, eeg_fixation_data


def is_fixation_on_target(
    gaze_x: np.ndarray, gaze_y: np.ndarray, targets_pos: List
) -> Tuple[bool, int | None]:
    """Check if the fixation is on target, and return the target index

    Args:
        gaze_x (np.ndarray): #TODO: _description_
        gaze_y (np.ndarray): #TODO: _description_
        targets_pos (list[str, list[float]]): #TODO: _description_

    Returns:
        tuple(bool, [int|None]): #TODO: _description_
    """

    # * Determine if fixation is on target
    mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

    on_target = False

    for target_ind, (target_name, target_pos) in enumerate(targets_pos):
        targ_left, targ_right, targ_bottom, targ_top = target_pos

        if (
            targ_left <= mean_gaze_x <= targ_right
            and targ_bottom <= mean_gaze_y <= targ_top
        ):
            on_target = True
            return (on_target, target_ind)

    return (on_target, None)


def crop_et_trial(epoch: mne.Epochs):
    # ! WARNING: We may not be capturing the first fixation if it is already on target
    # epoch = et_trial.copy()

    # * Get annotations, convert to DataFrame, and adjust onset times
    annotations = epoch.annotations.to_data_frame(time_format="ms")
    annotations["onset"] -= annotations["onset"].iloc[0]
    annotations["onset"] /= 1000

    # first_flash = annotations.query("description.str.contains('flash')").iloc[0]
    all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

    response_ids = EXP_CONFIG.lab.allowed_keys + ["timeout", "invalid"]
    response = annotations.query(f"description.isin({response_ids})").iloc[0]

    # * Convert to seconds
    start_time = all_stim_pres["onset"]
    end_time = response["onset"]

    # * Crop the data
    # epoch = epoch.copy().crop(first_flash["onset"], last_fixation["onset"])
    epoch = epoch.copy().crop(start_time, end_time)

    # * Get annotations, convert to DataFrame, and adjust onset times
    annotations = epoch.annotations.to_data_frame(time_format="ms")
    annotations["onset"] -= annotations["onset"].iloc[0]
    annotations["onset"] /= 1000

    all_stim_idx = annotations.query("description == 'stim-all_stim'").index[0]
    annotations = annotations.iloc[all_stim_idx:]
    annotations.reset_index(drop=True, inplace=True)

    time_bounds = [start_time, end_time]

    return epoch, annotations, time_bounds


def analyze_phase_coupling(
    eeg_data: np.ndarray,
    sfreq: int | float,
    f_pha,
    f_amp,
    idpac=(1, 2, 3),
    dcomplex="wavelet",
):
    """
    Analyze phase-amplitude coupling using Tensorpac.
    see: https://etiennecmb.github.io/tensorpac/auto_examples/erpac/plot_erpac.html#sphx-glr-auto-examples-erpac-plot-erpac-py
    Parameters:
    -----------
    eeg_data : np.ndarray
    f_pha : #TODO: write the description
    f_amp : #TODO: write the description

    Returns:
    --------
    pac : ndarray
        Phase-amplitude coupling results.
    p_obj : tensorpac.Pac
        PAC object with results.
    """

    # * alpha (8–13 Hz), beta (13–30 Hz), delta (0.5–4 Hz), and theta (4–7 Hz)
    # * Extract data and sampling frequency
    data = eeg_data.mean(axis=0)

    # * Suppress printed output
    with contextlib.redirect_stdout(io.StringIO()):
        p_obj = tensorpac.Pac(idpac=idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex)

        # * Extract phase and amplitude
        # * filter func :
        # *     - expect x as array of data of shape (n_epochs, n_times)
        # *     - returns the filtered data of shape (n_freqs, n_epochs, n_times)
        pha_p = p_obj.filter(sf=sfreq, x=data, ftype="phase")
        amp_p = p_obj.filter(sfreq, data, ftype="amplitude")

        # * Compute PAC
        pac = p_obj.fit(pha_p, amp_p)

    return pac, p_obj


def analyze_trial_decision_period(
    eeg_trial: mne.io.Raw,
    et_trial: mne.io.eyelink.eyelink.RawEyelink,
    raw_behav: pd.DataFrame,
    trial_N: int,
    eeg_baseline: float = 0.100,
    eeg_window: float = 0.600,
    show_plots: bool = True,
    pbar_off=True,
):
    """
    This function uses the Eye Tracker's label to identify fixation events
    """

    # ! TEMP
    # trial_N = 0
    # eeg_baseline: float = 0.100
    # eeg_window: float = 0.600
    # show_plots = False
    # pbar_off =False
    # ! TEMP

    # * ################################################################################
    # * "GLOBAL" VARIABLES

    # * Define frequency bands for Phase-Amplitude Coupling (PAC) analysis
    theta_band = [4, 7]  # Theta band: 4-7 Hz
    alpha_band = [8, 13]  # Alpha band: 8-13 Hz

    # * Get channel indices for PAC analysis
    picked_chs_pac = mne.pick_channels(
        eeg_trial.ch_names,
        include=EEG_CHAN_GROUPS.frontal,
        exclude=eeg_trial.info["bads"],
    )

    # * ################################################################################

    assert EEG_SFREQ == eeg_trial.info["sfreq"], "EEG data has incorrect sampling rate"

    # * Crop the data
    et_trial, et_annotations, time_bounds = crop_et_trial(et_trial)

    # * Adjust time bounds for EEG baseline and window
    # * Cropping with sample bounds
    trial_duration = (time_bounds[1] + eeg_window + eeg_baseline) - time_bounds[0]
    sample_bounds = [0, 0]
    sample_bounds[0] = int(time_bounds[0] * EEG_SFREQ)
    sample_bounds[1] = sample_bounds[0] + int(np.ceil(trial_duration * EEG_SFREQ))

    eeg_trial = eeg_trial.copy().crop(
        eeg_trial.times[sample_bounds[0]], eeg_trial.times[sample_bounds[1]]
    )  # TODO: Select only good channels here?

    # * Get channel positions for topomap
    eeg_info = eeg_trial.info

    chans_pos_xy = np.array(
        list(eeg_info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    # * Get info on the current trial
    stim_pos, stim_order, sequence, choices, _, solution, _ = get_trial_info(
        trial_N,
        raw_behav,
        X_POS_STIM,
        Y_POS_CHOICES,
        Y_POS_SEQUENCE,
        SCREEN_RESOLUTION,
        IMG_SIZE,
    )

    solution_ind = {v: k for k, v in choices.items()}[solution]

    # * Get the onset of the response event
    response_onset = et_annotations.query(
        "description.isin(['a', 'x', 'm', 'l', 'timeout', 'invalid'])"
    ).iloc[0]["onset"]

    seq_and_choices = sequence.copy()
    seq_and_choices.update({k + len(sequence): v for k, v in choices.items()})

    # * Get the indices of the icons, reindex choices to simplify analysis
    sequence_icon_inds = list(sequence.keys())
    choice_icon_inds = [i + len(sequence) for i in choices.keys()]
    solution_ind += len(sequence)
    # wrong_choice_icon_inds = [i for i in choice_icon_inds if i != solution_ind]

    # * Get the types of stimuli (e.g., choice related or unrelated to the sequence)
    stim_types = {}
    for i, icon_name in seq_and_choices.items():
        if i < 7:
            stim_types[i] = "sequence"
        elif i == 7:
            stim_types[i] = "question_mark"
        else:
            if i == solution_ind:
                stim_types[i] = "choice_correct"
            else:
                if icon_name in sequence.values():
                    stim_types[i] = "choice_incorrect_related"
                else:
                    stim_types[i] = "choice_incorrect_unrelated"

    # * Indices of every gaze fixation event
    fixation_inds = et_annotations.query("description == 'fixation'").index

    # * Initialize data containers
    gaze_target_fixation_sequence = []
    fixation_data: dict = {i: [] for i in range(len(stim_order))}
    eeg_fixation_data: dict = {i: [] for i in range(len(stim_order))}
    # eeg_fixation_pac_data: dict = {i: [] for i in range(len(stim_order))}

    # TODO: get heatmap of fixation data
    # * Loop through each fixation event
    pbar = tqdm(fixation_inds, leave=False, disable=pbar_off)

    for idx_fix, fixation_ind in enumerate(pbar):
        # * Get number of flash events before the current fixation; -1 to get the index
        fixation = et_annotations.loc[fixation_ind]

        # * Get fixation start and end time, and duration
        fixation_start = fixation["onset"]
        fixation_duration = fixation["duration"]
        fixation_end = fixation_start + fixation_duration

        # * Make sure we don't go beyond the end of the trial, crop the data if needed
        end_time = min(fixation_end, et_trial.times[-1])

        # * Get gaze positions and pupil diameter during the fixation period
        gaze_x, gaze_y, pupil_diam = (
            et_trial.copy().crop(fixation_start, end_time).get_data()
        )

        # * Determine if gaze is on target
        on_target, target_ind = is_fixation_on_target(gaze_x, gaze_y, stim_pos)

        # * Get EEG data during the fixation period
        # * Convert time bounds to sample bounds
        eeg_start_sample = int(fixation_start * EEG_SFREQ)
        eeg_end_sample = eeg_start_sample + int(
            np.ceil((eeg_window + eeg_baseline) * EEG_SFREQ)
        )

        # * Convert EEG sample bounds back to time bounds
        eeg_start_time = eeg_trial.times[eeg_start_sample]
        eeg_end_time = eeg_trial.times[eeg_end_sample]

        # ! Epoching on eeg_tril as MNE object results in a lot of epochs being dropped
        # ! automatically by MNE, so we'll use the raw data and crop it manually
        # ! then convert it to an EpochsArray object
        # * Crop by sample bounds
        eeg_slice = eeg_trial.copy().crop(eeg_start_time, eeg_end_time).get_data()

        eeg_slice = mne.EpochsArray(
            [eeg_slice], eeg_trial.info, tmin=-eeg_baseline, verbose="WARNING"
        )

        # * Apply baseline correction and detrend
        eeg_slice = eeg_slice.apply_baseline(baseline=(None, 0), verbose="WARNING")
        eeg_slice.detrend = 1

        # * Check if fixation is on target and longer than 100 ms
        if fixation_duration >= 0.1 and on_target:
            # if on_target:
            discarded = False

            fixation_data[target_ind].append(np.array([gaze_x, gaze_y]))

            gaze_target_fixation_sequence.append(
                [target_ind, fixation_start, fixation_duration, pupil_diam.mean()]
            )

            eeg_fixation_data[target_ind].append(eeg_slice)

        else:
            # * Only for visualization purposes
            discarded = True

        if show_plots:
            # * Select EEG channel groups to plot
            ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
            ch_group_colors = ["red", "green", "blue", "pink", "orange"]

            selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
                prepare_eeg_data_for_plot(
                    EEG_CHAN_GROUPS,
                    EEG_MONTAGE,
                    NON_EEG_CHANS,
                    eeg_trial.info["bads"],
                    ch_group_names,
                    ch_group_colors,
                )
            )

            title = f"ICON-{target_ind}" if on_target else "OFF-TARGET"
            title += f" ({fixation_duration * 1000:.0f} ms)"
            title += " - " + ("DISCARDED" if discarded else "SAVED")

            # fig = plot_eeg_and_gaze_fixations_plotly(
            plot_eeg_and_gaze_fixations(
                # * Convert to microvolts
                # eeg_data=eeg_slice * 1e6,
                eeg_data=eeg_slice.get_data(picks="eeg", units="uV")[0],  # * 1e6,
                eeg_sfreq=EEG_SFREQ,
                et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                eeg_baseline=eeg_baseline,
                response_onset=response_onset,
                eeg_start_time=eeg_start_time,
                eeg_end_time=eeg_end_time,
                icon_images=ICON_IMAGES,
                img_size=IMG_SIZE,
                stim_pos=stim_pos,
                chans_pos_xy=chans_pos_xy,
                ch_group_inds=ch_group_inds,
                group_colors=group_colors,
                screen_resolution=SCREEN_RESOLUTION,
                title=title,
                vlines=[
                    eeg_baseline * EEG_SFREQ,
                    eeg_baseline * EEG_SFREQ + fixation_duration * EEG_SFREQ,
                ],
            )

            # plt.savefig(
            #     wd
            #     / f"subj_{subj_N:02}-sess_{sess_N:02}-trial_{epoch_N:02}-fixation{idx_fix:02}.png"
            # )

    plt.close("all")

    # * Getting Fixation Related Potentials (FRPs)
    # * FRPs here correspond to the average EEG signal during fixations on each icon

    # * Concatenate all fixations on each icon
    eeg_fixation_data = {
        target_ind: mne.concatenate_epochs(data, verbose="WARNING")
        for target_ind, data in eeg_fixation_data.items()
        if len(data) > 0
    }

    # * Concat all fixations on each icon from the sequence (top row in experiment)
    eeg_fixations_sequence = {
        k: v
        for k, v in eeg_fixation_data.items()
        if k in sequence_icon_inds and len(v) > 0
    }

    # * Concat all fixations on each icon from the choices (bottom row in experiment)
    eeg_fixations_choices = {
        k: v
        for k, v in eeg_fixation_data.items()
        if k in choice_icon_inds and len(v) > 0
    }

    # * Calculate ERPs to fixations on sequence icons
    if len(eeg_fixations_sequence) > 0:
        fixations_sequence_erp = mne.concatenate_epochs(
            list(eeg_fixations_sequence.values()), verbose="WARNING"
        ).average()
    else:
        # fixations_sequence_erp = np.array([])
        fixations_sequence_erp = None

    # * Calculate ERPs to fixations on choice icons
    if len(eeg_fixations_choices) > 0:
        fixations_choices_erp = mne.concatenate_epochs(
            list(eeg_fixations_choices.values()), verbose="WARNING"
        ).average()
    else:
        # fixations_choices_erp = np.array([])
        fixations_choices_erp = None

    # * Compute Phase-Amplitude Coupling (PAC)
    eeg_fixation_pac_data = {}
    for target_ind, mne_data in eeg_fixation_data.items():
        # TODO: check which one to use for phase and amplitude
        pac, _ = analyze_phase_coupling(
            mne_data.get_data(picks=picked_chs_pac),
            sfreq=EEG_SFREQ,
            f_pha=theta_band,
            f_amp=alpha_band,
        )
        eeg_fixation_pac_data[target_ind] = pac

    # * ################################################################################
    # * GAZE ANALYSIS
    # * ################################################################################

    gaze_target_fixation_sequence_df = pd.DataFrame(
        gaze_target_fixation_sequence,
        columns=["target_ind", "onset", "duration", "pupil_diam"],
    )
    del gaze_target_fixation_sequence

    gaze_target_fixation_sequence_df["trial_N"] = trial_N

    gaze_target_fixation_sequence_df["stim_name"] = gaze_target_fixation_sequence_df[
        "target_ind"
    ].replace(seq_and_choices)

    gaze_target_fixation_sequence_df["pupil_diam"] = gaze_target_fixation_sequence_df[
        "pupil_diam"
    ].round(2)

    gaze_target_fixation_sequence_df["stim_type"] = gaze_target_fixation_sequence_df[
        "target_ind"
    ].replace(stim_types)

    first_fixation_order = (
        gaze_target_fixation_sequence_df.sort_values("onset")
        .groupby("target_ind")
        .first()["onset"]
        .rank()
        .astype(int)
    )

    first_fixation_order.name = "first_fix_order"

    mean_duration_per_target = (
        gaze_target_fixation_sequence_df.groupby("target_ind")["duration"]
        .mean()
        .round(2)
    )

    mean_diam_per_target = (
        gaze_target_fixation_sequence_df.groupby("target_ind")["pupil_diam"]
        .mean()
        .round()
    )

    fix_counts_per_target = gaze_target_fixation_sequence_df[
        "target_ind"
    ].value_counts()

    total_fix_duration_per_target = gaze_target_fixation_sequence_df.groupby(
        "target_ind"
    )["duration"].sum()

    mean_duration_per_target.name = "mean_duration"
    mean_diam_per_target.name = "mean_pupil_diam"
    total_fix_duration_per_target.name = "total_duration"

    mean_diam_per_target.sort_values(ascending=False, inplace=True)
    fix_counts_per_target.sort_values(ascending=False, inplace=True)
    total_fix_duration_per_target.sort_values(ascending=False, inplace=True)

    gaze_info = pd.concat(
        [
            fix_counts_per_target,
            first_fixation_order,
            total_fix_duration_per_target,
            mean_duration_per_target,
            mean_diam_per_target,
        ],
        axis=1,
    ).reset_index()

    gaze_info["stim_name"] = gaze_info["target_ind"].replace(seq_and_choices)
    gaze_info["trial_N"] = trial_N
    gaze_info["stim_type"] = gaze_info["target_ind"].replace(stim_types)
    gaze_info.sort_values("target_ind", inplace=True)

    # gaze_info.query("target_ind in @sequence_icon_inds")
    # gaze_info.query("target_ind == @choice_icon_inds")
    # gaze_info.query("target_ind == @wrong_choice_icon_inds")
    # gaze_info.query("target_ind == @solution_ind")

    # gaze_info

    # gaze_target_fixation_sequence.query("target_ind == @sequence_icon_inds").groupby(
    #     "target_ind"
    # )["duration"].mean().plot(kind="bar")
    # gaze_target_fixation_sequence["duration"].plot()
    # gaze_target_fixation_sequence["pupil_diam"].plot()
    # gaze_target_fixation_sequence.groupby("target_ind")["pupil_diam"].plot()

    return (
        fixation_data,
        eeg_fixation_data,
        eeg_fixation_pac_data,
        gaze_target_fixation_sequence_df,
        gaze_info,
        fixations_sequence_erp,
        fixations_choices_erp,
    )


# * ####################################################################################
# * MAIN FUNCTIONS
# * ####################################################################################


def analyze_session(subj_N: int, sess_N: int, save_dir: Path):
    """ """
    # ! TEMP
    # subj_N = 4
    # sess_N = 1
    # ! TEMP

    bad_chans = ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])

    # * Load the data
    sess_info, _, raw_behav, raw_eeg, raw_et, et_cals = load_raw_data(
        subj_N, sess_N, DATA_DIR, EEG_MONTAGE, bad_chans
    )

    if notes := sess_info["Notes"]:
        print(f"SESSION NOTES:\n{notes}")

    # sess_screen_resolution = sess_info["window_size"]
    # sess_img_size = sess_info["img_size"]
    # et_sfreq = raw_et.info["sfreq"]
    # tracked_eye = sess_info["eye"]
    # vision_correction = sess_info["vision_correction"]
    # eye_screen_distance = sess_info["eye_screen_dist"]

    if not ET_SFREQ == raw_et.info["sfreq"]:
        raise ValueError("Eye-tracking data has incorrect sampling rate")

    if not EEG_SFREQ == raw_eeg.info["sfreq"]:
        raise ValueError("EEG data has incorrect sampling rate")

    (
        manual_et_trials,
        *_,
        # et_events_dict,
        # et_events_dict_inv,
        # et_trial_bounds,
        # et_trial_events_df,
    ) = preprocess_et_data(raw_et, et_cals)

    (
        manual_eeg_trials,
        *_,
        # eeg_trial_bounds,
        # eeg_events,
        # eeg_events_df,
    ) = preprocess_eeg_data(raw_eeg, EEG_CHAN_GROUPS, raw_behav)

    bad_chans = raw_eeg.info["bads"]

    # * Initialize data containers
    sess_frps: Dict[str, List] = {"sequence": [], "choices": []}
    fixation_data_all = []
    eeg_fixation_data_all = []
    gaze_info_all = []
    gaze_target_fixation_sequence_all = []
    eeg_fixation_pac_data_all = []

    for trial_N in tqdm(raw_behav.index, desc="Analyzing every trial", leave=False):
        # * Get the EEG and ET data for the current trial
        eeg_trial = next(manual_eeg_trials)
        et_trial = next(manual_et_trials)

        (
            fixation_data,
            eeg_fixation_data,
            eeg_fixation_pac_data,
            gaze_target_fixation_sequence,
            gaze_info,
            fixations_sequence_erp,
            fixations_choices_erp,
        ) = analyze_trial_decision_period(
            eeg_trial,
            et_trial,
            raw_behav,
            trial_N,
            eeg_baseline=0.100,
            eeg_window=0.600,
            show_plots=False,
        )

        sess_frps["sequence"].append(fixations_sequence_erp)
        sess_frps["choices"].append(fixations_choices_erp)
        fixation_data_all.append(fixation_data)
        eeg_fixation_data_all.append(eeg_fixation_data)
        gaze_info_all.append(gaze_info)
        gaze_target_fixation_sequence_all.append(gaze_target_fixation_sequence)
        eeg_fixation_pac_data_all.append(eeg_fixation_pac_data)

    # * Concatenate the gaze data
    gaze_info = pd.concat([df for df in gaze_info_all if df.shape[0] > 0])
    gaze_info.reset_index(drop=True, inplace=True)

    gaze_target_fixation_sequence = pd.concat(
        [df for df in gaze_target_fixation_sequence_all if df.shape[0] > 0]
    )
    gaze_target_fixation_sequence.reset_index(
        drop=False, inplace=True, names=["fixation_N"]
    )

    valid_frps = dict(
        subj_N=subj_N,
        sess_N=sess_N,
        n_seq_frps=len(sess_frps["sequence"]) - sess_frps["sequence"].count(None),
        n_choices_frps=len(sess_frps["choices"]) - sess_frps["choices"].count(None),
    )

    # * Save the data to pickle files
    pd.DataFrame([valid_frps]).to_csv(save_dir / "valid_frps.csv", index=False)
    save_pickle(sess_frps, save_dir / "sess_frps.pkl")
    save_pickle(fixation_data_all, save_dir / "fixation_data.pkl")
    save_pickle(eeg_fixation_data_all, save_dir / "eeg_fixation_data.pkl")
    save_pickle(gaze_info, save_dir / "gaze_info.pkl")
    save_pickle(
        gaze_target_fixation_sequence, save_dir / "gaze_target_fixation_sequence.pkl"
    )
    save_pickle(eeg_fixation_pac_data_all, save_dir / "eeg_fixation_pac_data.pkl")

    return (
        sess_frps,
        fixation_data,
        eeg_fixation_data,
        gaze_info,
        gaze_target_fixation_sequence,
        # eeg_fixation_pac_data_all,
    )


def main():
    # res_dir = WD / "results/analyzed"
    res_dir = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis")
    res_dir.mkdir(exist_ok=True, parents=True)

    n_subjs = len(list(DATA_DIR.glob("subj_*")))

    errors = []
    for subj_N in tqdm(
        range(1, n_subjs + 1),
        desc="Analyzing data of every subjects",
    ):
        # # ! TEMP
        # subj_N = 1
        # sess_N = 3
        # trial_N = 1
        # # ! TEMP

        subj_dir = DATA_DIR / f"subj_{subj_N:02}"

        n_sessions = len(list(subj_dir.glob("sess_*")))

        for sess_N in range(1, n_sessions + 1):
            save_dir = res_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"
            save_dir.mkdir(exist_ok=True, parents=True)

            if save_dir.exists() and len(list(save_dir.glob("*.pkl"))) > 0:
                continue
            else:
                try:
                    (
                        sess_erps,
                        fixation_data,
                        eeg_fixation_data,
                        gaze_info,
                        gaze_target_fixation_sequence,
                    ) = analyze_session(subj_N, sess_N, save_dir)
                except Exception as e:
                    print(f"Error in subj_{subj_N:02}-sess_{sess_N:02}: {e}")
                    errors.append((subj_N, sess_N, e))
                    continue

            # * ########################################################################
            # sess_bad_chans = raw_eeg.info["bads"]

            # sess_bad_chans = ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(
            #     f"sess_{sess_N}", []
            # )
            # ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
            # ch_group_colors = ["red", "green", "blue", "pink", "orange"]

            # selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            #     prepare_eeg_data_for_plot(
            #         EEG_CHAN_GROUPS,
            #         EEG_MONTAGE,
            #         NON_EEG_CHANS,
            #         sess_bad_chans,
            #         ch_group_names,
            #         ch_group_colors,
            #     )
            # )

            # frp_ind = 20
            # frp_data = [sess_erps[type][frp_ind] for type in ["sequence", "choices"]]
            # fig_titles = ["FRP - Sequence Icons", "FRP - Choice Icons"]
            # for data, title in zip(frp_data, fig_titles):
            #     fig = plot_eeg(
            #         data.get_data(picks=selected_chans_names) * 1e6,
            #         chans_pos_xy,
            #         ch_group_inds,
            #         group_colors,
            #         EEG_SFREQ,
            #         eeg_baseline=0.1,  # TODO: check what eeg_baseline does & if 0.1 is correct
            #         vlines=0.1,
            #         title=title,
            #     )

            # gaze_target_fixation_sequence.groupby(["trial_N", "target_ind"])[
            #     "duration"
            # ].mean()

            # fig, ax = plt.subplots()
            # for trial_N in gaze_target_fixation_sequence["trial_N"].unique():
            #     temp = gaze_target_fixation_sequence.query("trial_N == @trial_N")
            #     temp["pupil_diam"].reset_index(drop=True).plot(ax=ax)
            # ax.set_title("Pupil diameter over time")
            # plt.show()
            # plt.close()

            # fig, ax = plt.subplots()
            # for trial_N in gaze_target_fixation_sequence["trial_N"].unique():
            #     temp = gaze_target_fixation_sequence.query("trial_N == @trial_N")
            #     temp["duration"].reset_index(drop=True).plot(ax=ax)
            # ax.set_title("Fixation duration over time")
            # plt.show()
            # plt.close("all")

    # print("\n".join(errors))
    # pprint(errors)


def inspect_results():
    # plt.close("all")
    # plt.get_backend()
    # plt.switch_backend("webagg")
    # plt.switch_backend(mpl_backend)

    res_dir = WD / "results/analyzed"  # /Oct27-Seq_and_choices"

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    erps = {}
    erp_files = sorted(res_dir.glob("sub*/sess*/*erps.pkl"))
    behav_files = sorted(DATA_DIR.glob("sub*/sess*/*behav.csv"))
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    behav_files = list(DATA_DIR.rglob("*behav.csv"))

    gaze_data = []
    behav_data = []
    for i, file in enumerate(gaze_info_files):
        subj_N = int(file.parents[1].stem.split("_")[-1])
        sess_N = int(file.parents[0].stem.split("_")[-1])

        behav_file = [
            f for f in behav_files if f"subj_{subj_N:02}/sess_{sess_N:02}" in str(f)
        ][0]
        behav_df = pd.read_csv(behav_file, index_col=0)
        behav_df.reset_index(names="trial_N", inplace=True)
        behav_df.insert(1, "sess_N", sess_N)
        behav_df = behav_df[
            [
                "subj_id",
                "sess_N",
                "trial_N",
                "rt",
                "choice",
                "correct",
                "pattern",
                "item_id",
            ]
        ]
        behav_df.rename(columns={"subj_id": "subj_N"}, inplace=True)

        behav_data.append(behav_df)

        gaze_df = pd.read_pickle(file)
        gaze_df["subj_N"] = subj_N
        gaze_df["sess_N"] = sess_N
        gaze_data.append(gaze_df)

    behav_data = pd.concat(behav_data, axis=0)

    # behav_data = behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
    behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
    behav_data.reset_index(drop=True, inplace=True)

    gaze_data = pd.concat(gaze_data, axis=0)

    gaze_data = gaze_data.merge(
        behav_data[["subj_N", "sess_N", "trial_N", "pattern", "item_id"]],
        on=["subj_N", "sess_N", "trial_N"],
    )

    gaze_data = gaze_data.sort_values(
        ["subj_N", "sess_N", "trial_N", "target_ind"]
    ).reset_index(drop=True)

    gaze_data["mean_duration"].mean()

    gaze_data.groupby(["subj_N", "sess_N"])["mean_pupil_diam"].mean()
    gaze_data.groupby(["subj_N"])["mean_pupil_diam"].mean()

    pupil_diam_by_subj_pattern = (
        gaze_data.groupby(["subj_N", "pattern"])["mean_pupil_diam"].mean().reset_index()
    )

    fix_duration_by_subj_pattern = (
        gaze_data.groupby(["subj_N", "pattern"])["mean_duration"].mean().reset_index()
    )

    # fig, ax = plt.subplots()
    # sns.barplot(
    #     pupil_diam_by_subj_pattern,
    #     x="subj_N",
    #     y="mean_pupil_diam",
    #     hue="pattern",
    #     ax=ax,
    # )
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # fig, ax = plt.subplots()
    # sns.barplot(
    #     pupil_diam_by_subj_pattern,
    #     x="pattern",
    #     y="mean_pupil_diam",
    #     ax=ax,
    # )
    # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # fig, ax = plt.subplots()
    # sns.barplot(
    #     pupil_diam_by_subj_pattern,
    #     x="pattern",
    #     y="mean_pupil_diam",
    #     ax=ax,
    # )
    # xticks = ax.get_xticks()
    # xticklabels = ax.get_xticklabels()
    # ax.set_xticks(xticks, xticklabels, rotation=45)
    # # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # fig, ax = plt.subplots()
    # sns.barplot(
    #     fix_duration_by_subj_pattern,
    #     x="pattern",
    #     y="mean_duration",
    #     ax=ax,
    # )
    # xticks = ax.get_xticks()
    # xticklabels = ax.get_xticklabels()
    # ax.set_xticks(xticks, xticklabels, rotation=45)
    # # ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # * ################################################################################
    # * Inspect ERP data
    # * ################################################################################

    # * baseline correction period in seconds, used for plotting here
    eeg_baseline = 0.1
    ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
    ch_group_colors = ["red", "green", "blue", "pink", "orange"]

    patterns = [
        "ABCAABCA",
        "ABBAABBA",
        "ABBACDDC",
        "ABCDEEDC",
        "ABBCABBC",
        "ABCDDCBA",
        "AAABAAAB",
        "ABABCDCD",
    ]
    # TODO: change to line below, but make sure it doesn't change the rest of the analysis:
    # patterns = behav_data["pattern"].unique()

    # * Create containers for ERPs
    group_seq_erps = []
    group_choices_erps = []
    group_overall_erps = []

    subj_pattern_frps = {}
    group_pattern_frps = {p: [] for p in patterns}

    # * Load ERPs
    for erp_file in sorted(erp_files):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        subj_pattern_frps[subj_N] = {p: [] for p in patterns}

    for erp_file in tqdm(sorted(erp_files)):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        sess_N = int(erp_file.parents[0].stem.split("_")[-1])
        print(f"Analyzing subj_{subj_N:02}-sess_{sess_N:02}")

        behav_file = list(
            DATA_DIR.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )[0]
        raw_behav = pd.read_csv(behav_file, index_col=0)

        sess_bad_chans = ALL_BAD_CHANS[f"subj_{subj_N}"][f"sess_{sess_N}"]

        selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            prepare_eeg_data_for_plot(
                EEG_CHAN_GROUPS,
                EEG_MONTAGE,
                NON_EEG_CHANS,
                sess_bad_chans,
                ch_group_names,
                ch_group_colors,
            )
        )

        # with open(erp_file, "rb") as f:
        # erp_data = pickle.load(f)
        erp_data = load_pickle(erp_file)

        sequence_erps = erp_data["sequence"]
        choice_erps = erp_data["choices"]

        for pattern, this_pattern_erp in list(zip(raw_behav["pattern"], sequence_erps)):
            if isinstance(this_pattern_erp, mne.EvokedArray):
                subj_pattern_frps[subj_N][pattern].append(this_pattern_erp)
                # group_pattern_frps[pattern].extend(erp_data["sequence"])

        sequence_erps = [
            erp for erp in sequence_erps if isinstance(erp, mne.EvokedArray)
        ]

        choice_erps = [erp for erp in choice_erps if isinstance(erp, mne.EvokedArray)]

        # * Get Group averaged ERPs
        mean_sequence_erp = mne.combine_evoked(sequence_erps, "equal")
        mean_choices_erp = mne.combine_evoked(choice_erps, "equal")
        mean_overall_erp = mne.combine_evoked(sequence_erps + choice_erps, "equal")

        group_seq_erps.append(mean_sequence_erp)
        group_choices_erps.append(mean_choices_erp)
        group_overall_erps.append(mean_overall_erp)

        show_plots = False

        if show_plots:
            for eeg_data, title in zip(
                [mean_sequence_erp, mean_choices_erp, mean_overall_erp],
                ["Sequence ERP", "Choices ERP", "Overall ERP"],
            ):
                fig = plot_eeg(
                    eeg_data.get_data(picks=selected_chans_names) * 1e6,
                    chans_pos_xy,
                    {
                        k: v
                        for k, v in ch_group_inds.items()
                        if k in ["frontal", "occipital"]
                    },
                    # ch_group_inds,
                    group_colors,
                    EEG_SFREQ,
                    eeg_baseline,
                    vlines=None,
                    title=f"Subj{subj_N}-Sess{sess_N}\n" + title,
                )
                plt.show()
                plt.close("all")

            for trial_N in range(len(erps[subj_N][sess_N]["sequence"])):
                eeg_data = erps[subj_N][sess_N]["sequence"][trial_N]
                eeg_data = eeg_data.get_data(picks=selected_chans_names) * 1e6

                fig = plot_eeg(
                    eeg_data,
                    chans_pos_xy,
                    {
                        k: v
                        for k, v in ch_group_inds.items()
                        if k in ["frontal", "occipital"]
                    },
                    # ch_group_inds,
                    group_colors,
                    EEG_SFREQ,
                    eeg_baseline,
                    vlines=None,
                    title=f"Subj{subj_N}-Sess{sess_N}-Trial{trial_N}",
                    chan_names=selected_chans_names,
                    # plot_topo=False,
                    plot_eeg=False,
                    # plot_eeg_group=False,
                )

                plt.show()
                plt.close("all")

    # [
    #     [len(pat_frp) for pat_frp in subj_frps.values()]
    #     for subj_frps in subj_pattern_frps.values()
    # ]

    # missing_frps = {}
    # for subj_N in subj_pattern_frps.keys():
    #     for pattern, frps in subj_pattern_frps[subj_N].items():
    #         missing = [i for i, frp in enumerate(frps)]
    #         print(missing)
    #         # if not isinstance(frp, mne.Evoked)]
    #         if len(missing) > 0:
    #             missing_frps[(subj_N, pattern)] = missing

    for subj_N in subj_pattern_frps.keys():
        print(f"Subject {subj_N}")
        for i, (pattern, this_pattern_erps) in enumerate(
            subj_pattern_frps[subj_N].items()
        ):
            print(f"\t{i + 1}. Pattern {pattern}: {len(this_pattern_erps)} trials")

    for subj_N in tqdm(subj_pattern_frps.keys()):
        for pattern, this_pattern_erps in subj_pattern_frps[subj_N].items():
            if len(this_pattern_erps) > 0:
                mean_pattern_erp = mne.combine_evoked(this_pattern_erps, "equal")
                subj_pattern_frps[subj_N][pattern] = mean_pattern_erp

    for pattern in patterns:
        temp = []
        for subj_N in subj_pattern_frps.keys():
            temp.append(subj_pattern_frps[subj_N][pattern])

        group_pattern_frps[pattern] = mne.combine_evoked(temp, "equal")

    # * Plot ERPs
    for patt_erp in group_pattern_frps.values():
        patt_erp.plot()
        patt_erp.apply_baseline((None, 0)).plot()

    group_seq_erps = mne.combine_evoked(group_seq_erps, "equal")
    group_choices_erps = mne.combine_evoked(group_choices_erps, "equal")
    group_overall_erps = mne.combine_evoked(group_overall_erps, "equal")

    # group_pattern_frps = {
    #     k: mne.combine_evoked(v, "equal") for k, v in group_pattern_frps.items()
    # }

    all_subj_bad_chans = group_seq_erps.info["bads"]

    selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
        prepare_eeg_data_for_plot(
            EEG_CHAN_GROUPS,
            EEG_MONTAGE,
            NON_EEG_CHANS,
            all_subj_bad_chans,
            ch_group_names,
            ch_group_colors,
        )
    )

    for eeg_data, title in zip(
        [group_seq_erps, group_choices_erps, group_overall_erps],
        ["Sequence ERP", "Choices ERP", "Overall ERP"],
    ):
        _eeg_data = (
            eeg_data.copy().crop(-0.1, 0.3).get_data(picks=selected_chans_names) * 1e6
        )

        fig = plot_eeg(
            _eeg_data,
            chans_pos_xy,
            {k: v for k, v in ch_group_inds.items() if k in ["frontal", "occipital"]},
            # ch_group_inds,
            group_colors,
            EEG_SFREQ,
            eeg_baseline,
            vlines=None,
            title=title,
        )

    for pattern, eeg_data in group_pattern_frps.items():
        _eeg_data = (
            eeg_data.copy().crop(-0.1, 0.3).get_data(picks=selected_chans_names) * 1e6
        )

        # plt.get_backend()
        # plt.switch_backend("webagg")
        # plt.switch_backend(mpl_backend)

        plot_eeg(
            _eeg_data,
            chans_pos_xy,
            # {k: v for k, v in ch_group_inds.items() if k in ["frontal", "occipital"]},
            ch_group_inds,
            group_colors,
            EEG_SFREQ,
            eeg_baseline,
            vlines=None,
            title="All Subjects\n" + f"Pattern: {pattern}",
        )
        plt.show()

    # * ################################################################################
    # * Get RDMs
    # * ################################################################################

    subj_neg_peaks = pd.DataFrame()

    for subj in subj_pattern_frps.keys():
        for pattern, erp in subj_pattern_frps[subj].items():
            latency, amplitude = get_neg_erp_peak(
                erp, (0, 0.2), EEG_CHAN_GROUPS["occipital"]
            )
            subj_neg_peaks = pd.concat(
                [
                    subj_neg_peaks,
                    pd.DataFrame(
                        [
                            {
                                "subj": subj,
                                "pattern": pattern,
                                "latency": latency,
                                "amplitude": amplitude,
                            }
                        ]
                    ),
                ]
            )
    subj_neg_peaks.reset_index(drop=True, inplace=True)
    subj_neg_peaks.to_csv(WD / "results/subj_neg_peaks.csv", index=False)

    rdm_method = "euclidean"

    (
        rdms_erp_latency,
        rdms_erp_amplitude,
        rdms_erp_combined,
    ) = get_rdms_negative_peak_eeg(
        subj_pattern_frps,
        EEG_CHAN_GROUPS["occipital"],
        time_window=(0, 0.2),
        method=rdm_method,
        show_plots=False,
    )

    rdms_accuracy = get_rdms_behavior_pattern_groups(
        behav_data, method=rdm_method, show_plots=False
    )

    rdms_gaze_duration = get_rdms_eye_gaze(
        gaze_data, method=rdm_method, show_plots=False
    )

    rdms_pupil_diam = get_rdms_eye_pupil_diam(
        gaze_data, method=rdm_method, show_plots=False
    )

    rdms_rt = get_rdms_rt(behav_data, method=rdm_method, show_plots=False)

    rdms = dict(
        rdms_erp_latency=rdms_erp_latency,
        rdms_erp_amplitude=rdms_erp_amplitude,
        rdms_erp_combined=rdms_erp_combined,
        rdms_accuracy=rdms_accuracy,
        rdms_rt=rdms_rt,
        rdms_gaze_duration=rdms_gaze_duration,
        rdms_pupil_diam=rdms_pupil_diam,
    )

    labels = list(rdms_erp_latency[1].pattern_descriptors.values())[0]

    def get_group_rdms(rdms, labels):
        all_subj_rdms = {k: None for k in rdms.keys()}

        for rdm_type, subjects_rdm in rdms.items():
            # ! TEMP
            # for subj, subj_rdm in subjects_rdm.items():
            #     fig, ax = plt.subplots()
            #     plot_matrix(
            #         subj_rdm.get_matrices()[0],
            #         labels=labels,
            #         title=f"{rdm_type}",
            #         show_values=True,
            #         norm="max",
            #         as_pct=True,
            #         ax=ax,
            #     )
            #     fig.savefig(
            #         wd / "results" / f"subj_{subj:02}-{rdm_type}.png",
            #         dpi=200,
            #         bbox_inches="tight",
            #     )
            # ! TEMP

            all_subj_rdms[rdm_type] = np.concat(
                [rdm.get_matrices()[0][None, :, :] for rdm in subjects_rdm.values()]
            ).mean(axis=0)

            title = (
                rdm_type.replace("_", " ")
                .title()
                .replace("Rdms", "RDM")
                .replace("Erp", "ERP")
            )

            fig, ax = plt.subplots()
            plot_matrix(
                all_subj_rdms[rdm_type],
                labels=labels,
                title=title,
                show_values=True,
                norm="max",
                as_pct=True,
                ax=ax,
            )
            # ! TEMP
            # fig.savefig(
            #     wd / "results" / f"{title}-Group_lvl.png",
            #     dpi=200,
            #     bbox_inches="tight",
            # )
            # ! TEMP

        # * Get the list of RDM types
        rdm_types = list(all_subj_rdms.keys())

        # * Initialize square DataFrames for Pearson and Spearman correlations
        df_pearson_corr_all_subj = pd.DataFrame(
            np.nan, index=rdm_types, columns=rdm_types
        )
        df_spearman_corr_all_subj = pd.DataFrame(
            np.nan, index=rdm_types, columns=rdm_types
        )

        # * Compute pairwise correlations and fill the matrices
        for comb in combinations(rdm_types, 2):
            rdm1 = all_subj_rdms[comb[0]]
            rdm2 = all_subj_rdms[comb[1]]

            # * Fill the matrices symmetrically
            pearson_corr = compare_rdms(rdm1, rdm2, method="pearson")
            df_pearson_corr_all_subj.loc[comb[0], comb[1]] = pearson_corr
            df_pearson_corr_all_subj.loc[comb[1], comb[0]] = pearson_corr

            spearman_corr = compare_rdms(rdm1, rdm2, method="spearman")
            df_spearman_corr_all_subj.loc[comb[0], comb[1]] = spearman_corr
            df_spearman_corr_all_subj.loc[comb[1], comb[0]] = spearman_corr

        # * Fill the diagonal with 1s (self-correlation)
        np.fill_diagonal(df_pearson_corr_all_subj.values, 1)
        np.fill_diagonal(df_spearman_corr_all_subj.values, 1)

        dfs_corr = [df_pearson_corr_all_subj, df_spearman_corr_all_subj]
        titles = [f"RSA - {m}\nGroup Level" for m in ["Pearson", "Spearman"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for df, title, ax in zip(dfs_corr, titles, axes):
            # print(df.values)
            plot_matrix(
                df.values,
                labels=[c.replace("rdms_", "").replace("_", " ") for c in df.columns],
                show_values=True,
                title=title,
                cmap="RdBu_r",
                ax=ax,
                as_pct=True,
            )
        plt.tight_layout()

        # ! TEMP
        # fig.savefig(
        #     wd / "results" / f"RDMs-Correlation-Group_lvl-{title}.png",
        #     dpi=200,
        #     bbox_inches="tight",
        # )
        # ! TEMP

        plt.show()

        return all_subj_rdms, dfs_corr

    group_rdms, correlations = get_group_rdms(rdms, labels)

    return group_pattern_frps


def inspect_results2():
    res_dir = WD / "results/analyzed"  # /Oct27-Seq_and_choices"

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    frps = {}
    frp_files = sorted(res_dir.glob("sub*/sess*/*frps.pkl"))
    behav_files = sorted(DATA_DIR.glob("sub*/sess*/*behav.csv"))
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    behav_files = list(DATA_DIR.rglob("*behav.csv"))

    subjs = sorted([int(f.stem.split("_")[1]) for f in res_dir.glob("subj*")])

    # * baseline correction period in seconds, used for plotting here
    eeg_baseline = 0.1
    ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
    ch_group_colors = ["red", "green", "blue", "pink", "orange"]

    def get_sess_frps(res_dir, subj_N, sess_N):
        frp_file = res_dir / f"subj_{subj_N:02}/sess_{sess_N:02}/sess_frps.pkl"

        behav_file = list(
            DATA_DIR.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )[0]

        raw_behav = pd.read_csv(behav_file, index_col=0)

        sess_patterns = raw_behav["pattern"].to_list()
        sess_item_ids = raw_behav["item_id"].to_list()

        frp_data = load_pickle(frp_file)

        sequence_frps = frp_data["sequence"]
        choice_frps = frp_data["choices"]

        results = dict(
            sess_patterns=sess_patterns,
            sess_item_ids=sess_item_ids,
            sequence_frps=sequence_frps,
            choice_frps=choice_frps,
        )

        return results

    def get_subj_frps(res_dir, subj_N):
        subj_dir = res_dir / f"subj_{subj_N:02}"
        subj_N = int(subj_dir.stem.split("_")[-1])

        pattern_seq_frps = {p: [] for p in PATTERNS}
        pattern_choices_frps = {p: [] for p in PATTERNS}

        for sess_dir in sorted(subj_dir.glob("sess_*")):
            sess_N = int(sess_dir.stem.split("_")[-1])

            if len([f for f in sess_dir.glob("*.pkl")]) > 0:
                (
                    sess_patterns,
                    sess_item_ids,
                    sequence_frps,
                    choice_frps,
                ) = get_sess_frps(res_dir, subj_N, sess_N).values()

                for p in sess_patterns:
                    pattern_inds = [i for i, _p in enumerate(sess_patterns) if _p == p]

                    sess_pattern_seq_frps = [sequence_frps[i] for i in pattern_inds]
                    sess_pattern_seq_frps = [
                        frp for frp in sess_pattern_seq_frps if frp is not None
                    ]
                    pattern_seq_frps[p].extend(sess_pattern_seq_frps)

                    sess_pattern_choices_frps = [choice_frps[i] for i in pattern_inds]
                    sess_pattern_choices_frps = [
                        frp for frp in sess_pattern_choices_frps if frp is not None
                    ]
                    pattern_choices_frps[p].extend(sess_pattern_choices_frps)

        avg_pattern_seq_frps = {
            p: mne.combine_evoked(frps, "equal") for p, frps in pattern_seq_frps.items()
        }

        avg_pattern_choices_frps = {
            p: mne.combine_evoked(frps, "equal")
            for p, frps in pattern_choices_frps.items()
        }

        return (
            pattern_seq_frps,
            pattern_choices_frps,
            avg_pattern_seq_frps,
            avg_pattern_choices_frps,
        )

    group_avg_pattern_seq_frps = {p: [] for p in PATTERNS}
    group_avg_pattern_choices_frps = {p: [] for p in PATTERNS}

    # * Load FRPs
    for subj_N in tqdm(subjs):
        (
            pattern_seq_frps,
            pattern_choices_frps,
            avg_pattern_seq_frps,
            avg_pattern_choices_frps,
        ) = get_subj_frps(res_dir, subj_N)

        for p in PATTERNS:
            group_avg_pattern_seq_frps[p].append(avg_pattern_seq_frps[p])
            group_avg_pattern_choices_frps[p].append(avg_pattern_choices_frps[p])

            selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
                prepare_eeg_data_for_plot(
                    EEG_CHAN_GROUPS,
                    EEG_MONTAGE,
                    NON_EEG_CHANS,
                    avg_pattern_seq_frps[p].info["bads"],
                    ch_group_names,
                    ch_group_colors,
                )
            )

            peak_latency, peak_amplitude = get_neg_erp_peak(
                avg_pattern_seq_frps[p], (0, 0.2), EEG_CHAN_GROUPS["occipital"]
            )

            fig = plot_eeg(
                avg_pattern_seq_frps[p].get_data(
                    units="uV", picks=selected_chans_names
                ),
                chans_pos_xy,
                ch_group_inds,
                group_colors,
                EEG_SFREQ,
                eeg_baseline,
                vlines=[peak_latency * 1000],
            )
            plt.show()
            plt.close()

    # * Combine the FRPs of all subjects
    for p in PATTERNS:
        group_avg_pattern_seq_frps[p] = mne.combine_evoked(
            group_avg_pattern_seq_frps[p], "equal"
        )
        group_avg_pattern_choices_frps[p] = mne.combine_evoked(
            group_avg_pattern_choices_frps[p], "equal"
        )

    for pattern, frp in group_avg_pattern_seq_frps.items():
        # frp.plot()
        selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            prepare_eeg_data_for_plot(
                EEG_CHAN_GROUPS,
                EEG_MONTAGE,
                NON_EEG_CHANS,
                frp.info["bads"],
                ch_group_names,
                ch_group_colors,
            )
        )

        peak_latency, peak_amplitude = get_neg_erp_peak(
            frp, (0, 0.2), EEG_CHAN_GROUPS["occipital"]
        )

        fig = plot_eeg(
            frp.get_data(units="uV", picks=selected_chans_names),
            chans_pos_xy,
            ch_group_inds,
            group_colors,
            EEG_SFREQ,
            eeg_baseline,
            vlines=[peak_latency * 1000],
        )

        # fixation_data = load_pickle(
        #     "/Users/chris/Documents/PhD-Local/abstract_reasoning/setup-analysis/results/analyzed/subj_14/sess_01/fixation_data.pkl"
        # )
        # len(fixation_data)

        # # * Set up the figure
        # fig, ax = plt.subplots(fig)
        # ax.set_xlim(0, screen_resolution[0])
        # ax.set_ylim(screen_resolution[1], 0)
        # ax.set_xticks([])
        # ax.set_yticks([])

        # frp.apply_baseline((None, 0)).plot()

    valid_frps = pd.concat([pd.read_csv(f) for f in res_dir.rglob("*valid_frps.csv")])
    valid_frps.reset_index(drop=True, inplace=True)
    valid_frps.sort_values(["subj_N", "sess_N"]).reset_index(drop=True)


def inspect_behav():
    # *

    res = behav_analysis_all(return_raw=True)

    (
        res_by_pattern,
        overall_acc,
        overall_rt,
        rt_by_correct,
        rt_by_correct_and_pattern,
        raw_behav,
    ) = res

    del res

    # raw_data

    fig_params = {
        "figsize": (9, 6),
        "tight_layout": True,
    }
    rc_params = {
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }

    def init_fig(fig_params):
        with plt.rc_context(rc=rc_params):
            fig, ax = plt.subplots(**fig_params)
        return fig, ax

    def custom_behav_plot(
        data,
        plot_kind,
        fig_params,
        title=None,
        xlabel=None,
        ylabel=None,
        legend=None,
        hlines: Optional[List[Dict] | None] = None,
        grid: Optional[Dict | None] = None,
        show=True,
    ):
        # TODO: finish implementing legend logic
        legend = "auto" if legend is None else legend
        assert legend in [True, False, "auto"]

        fig, ax = init_fig(fig_params)

        data.plot(
            kind=plot_kind,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        if hlines:
            for hline_params in hlines:
                hline_y = hline_params["y"]
                hline_kwargs = {k: v for k, v in hline_params.items() if k != "y"}
                ax.axhline(hline_y, **hline_kwargs)
        if grid:
            ax.grid(**grid)

        if legend is True:
            ax.legend()
            _legend = ax.get_legend()

            if legend:
                ax.legend(
                    title=_legend.get_title().get_text(),
                    bbox_to_anchor=(1.005, 1),
                    loc="upper left",
                )
        elif legend is False:
            ax.get_legend().remove()
        else:
            _legend = ax.get_legend()

            ax.legend(
                title=_legend.get_title().get_text(),
                bbox_to_anchor=(1.005, 1),
                loc="upper left",
            )

        if show:
            plt.show()

        return fig

    # * ################################################################################
    # * ACCURACY PLOTS
    # * ################################################################################

    # * Plot accuracy by pattern and session for each subject
    for subj_N in res_by_pattern["subj_N"].unique():
        data = (
            res_by_pattern.query(f"subj_N == {subj_N}")
            .groupby(["sess_N", "pattern"])["accuracy"]
            .mean()
            .unstack()
            .T
        )

        fig = custom_behav_plot(
            data,
            "bar",
            fig_params,
            title=f"Subject {subj_N} - Accuracy\nby Pattern and Session",
            xlabel=None,
            ylabel="Accuracy",
            # legend=None,
            grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
        )

    # * Plot accuracy by subject and session
    data = overall_acc.groupby(["subj_N", "sess_N"])["accuracy"].mean().unstack()

    fig = custom_behav_plot(
        data,
        "bar",
        fig_params,
        title="Accuracy\nby Subject and Session",
        xlabel="Subject",
        ylabel="Accuracy",
        legend=None,
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    # * Plot accuracy by pattern and subject (line plot)
    data = res_by_pattern.groupby(["pattern", "subj_N"])["accuracy"].mean().unstack()
    fig = custom_behav_plot(
        data,
        "line",
        fig_params,
        title="Accuracy\nby Pattern and Subject",
        xlabel="Pattern",
        ylabel="Accuracy",
        # legend=None,
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    # * Plot accuracy by subject
    data = overall_acc.groupby(["subj_N"])["accuracy"].mean()
    fig = custom_behav_plot(
        data,
        "bar",
        fig_params,
        title="Overall Accuracy\nBy Subject",
        xlabel="Subject",
        ylabel="Accuracy",
        legend=True,
        hlines=[dict(y=data.mean(), color="red", linestyle="--", lw=0.9, label="Mean")],
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    # * Make a summary table
    summary_table = (
        overall_acc.groupby(["subj_N", "sess_N"])["accuracy"].mean().unstack()
    )

    display(summary_table.fillna(""))
    # print(summary_table.fillna("").to_markdown())

    display(raw_behav.groupby(["subj_N", "item_id"])["correct"].mean().unstack().T)

    # * ################################################################################
    # * RT PLOTS
    # * ################################################################################
    data = raw_behav.groupby(["subj_N"])["rt"].mean()

    # * Plot RT by subject and session
    data = raw_behav.groupby(["subj_N", "sess_N"])["rt"].mean().unstack()

    fig = custom_behav_plot(
        data,
        "bar",
        fig_params,
        title="Response Time\nby Subject and Session",
        xlabel="Subject",
        ylabel="RT",
        legend=None,
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    # * Plot RT by pattern and subject (line plot)
    data = raw_behav.groupby(["pattern", "subj_N"])["rt"].mean().unstack()

    fig = custom_behav_plot(
        data,
        "line",
        fig_params,
        title="Response Time\nby Pattern and Subject",
        xlabel="Pattern",
        ylabel="RT",
        # legend=None,
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    # * Plot accuracy by subject
    data = raw_behav.groupby(["subj_N"])["rt"].mean()
    fig = custom_behav_plot(
        data,
        "bar",
        fig_params,
        title="Mean Response Time\nBy Subject",
        xlabel="Subject",
        ylabel="RT",
        legend=True,
        hlines=[dict(y=data.mean(), color="red", linestyle="--", lw=0.9, label="Mean")],
        grid=dict(axis="y", ls="--", lw=0.7, c="black", alpha=0.7),
    )

    del data

    # * ################################################################################
    # * Stats
    # * ################################################################################

    mask = raw_behav[~raw_behav["rt"].isna()].index
    # raw_behav['rt'].dtype
    linear_reg1 = sm.OLS(
        raw_behav.loc[mask, "correct"], raw_behav.loc[mask, "rt"]
    ).fit()
    print(linear_reg1.summary())

    data = raw_behav.copy().loc[mask, ["pattern", "rt"]]
    data = pd.get_dummies(data, columns=["pattern"], drop_first=True)
    pattern_cols = data.columns[1:]
    # pattern_cols.astype(int)

    formula = "rt ~ " + " + ".join(pattern_cols)
    linear_reg2 = smf.ols(formula, data=data).fit()
    print(linear_reg2.summary())

    data = raw_behav[["pattern", "rt", "correct"]].copy()  # .groupby('pattern').mean()
    data["correct"] = data["correct"].astype(int)
    data = data.dropna()

    # * Perform one-way ANOVA
    model = smf.ols("rt ~ pattern", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # * Perform one-way ANOVA
    model = smf.ols("correct ~ pattern", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # * Linear regression with negative peaks
    subj_neg_peaks = pd.read_csv(WD / "results/subj_neg_peaks.csv")
    group_neg_peaks = subj_neg_peaks.groupby("pattern")[["latency", "amplitude"]].mean()

    rt_by_pattern = (
        raw_behav.query("~rt.isna() and subj_N<6")
        .groupby("pattern")["rt"]
        .mean()
        .reset_index()
    )
    rt_by_pattern = rt_by_pattern.merge(group_neg_peaks, on="pattern")
    rt_by_pattern

    linear_reg3 = sm.OLS(
        rt_by_pattern["rt"], rt_by_pattern[["latency", "amplitude"]]
    ).fit()
    # res = linear_reg3.fit()
    print(linear_reg3.summary())

    test_set = (
        raw_behav.query("~rt.isna() and subj_N>=6")
        .groupby("pattern")["rt"]
        .mean()
        .reset_index()
        .merge(group_neg_peaks, on="pattern")
    )

    test_set.insert(
        2, "prediction", linear_reg3.predict(test_set[["latency", "amplitude"]])
    )
    test_set.insert(3, "residual", test_set["rt"] - test_set["prediction"])
    test_set

    dir(linear_reg3)


# clear_jupyter_artifacts()
# memory_usage = get_memory_usage()
# memory_usage.head(50)

if __name__ == "__main__":
    pass

    # # ! TEMP
    # temp = pd.merge(gaze_data, behav_data, on=["subj_N", "sess_N", "trial_N"])
    # temp.drop(columns=["pattern_y", "item_id_y"], inplace=True)
    # temp.rename(columns={"pattern_x": "pattern", "item_id_x": "item_id"}, inplace=True)
    # temp["correct"].unique()
    # temp["correct"].replace("invalid", False, inplace=True)
    # temp["correct"].replace("False", False, inplace=True)
    # temp["correct"].replace("True", True, inplace=True)
    # temp["correct"].unique()
    # temp["rt"].replace("timeout", np.nan, inplace=True)
    # temp["rt"] = temp["rt"].astype(float)

    # df_agg = pd.concat(
    #     [
    #         temp.groupby(["pattern"])["correct"].mean().sort_index(),
    #         temp.groupby(["pattern"])["mean_pupil_diam"].mean().sort_index(),
    #         temp.groupby(["pattern"])["mean_duration"].mean().sort_index(),
    #         temp.groupby(["pattern"])["rt"].mean().sort_index(),
    #     ],
    #     axis=1,
    # )
    # df_agg_corr = df_agg.corr().to_numpy()
    # plot_matrix(
    #     df_agg_corr,
    #     labels=df_agg.columns,
    #     show_values=True,
    #     as_pct=True,
    #     cmap="coolwarm",
    # )
    # # ! TEMP
