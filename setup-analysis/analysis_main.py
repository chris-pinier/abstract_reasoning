# *
import base64
import io
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from pprint import pprint
from string import ascii_letters
from typing import Any, Dict, Final, List, Optional, Tuple, Union

import hmp
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import mne.baseline

import numpy as np
import pandas as pd
import pendulum
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.subplots as ps
import tomllib
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
from tensorpac import EventRelatedPac
from tensorpac.signals import pac_signals_wavelet
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
from analysis_utils import (
    normalize,
    get_stim_coords,
    get_trial_info,
    locate_trials,
    resample_eye_tracking_data,
    resample_and_handle_nans,
    check_ch_groups,
)
from analysis_plotting import (
    plot_matrix,
    plot_eeg_and_gaze_fixations,
    plot_eeg,
    get_gaze_heatmap,
    prepare_eeg_data_for_plot,
)
# * Packages to look into:
# import mplcursors
# import polars as pl
# import pylustrator

# * ####################################################################################
# * LOADING FILES
# * ####################################################################################

# wd = Path(__file__).parent
wd = Path("/Users/chris/Documents/PhD-Local/abstract_reasoning/setup-analysis")
os.chdir(wd)

exp_config_file = wd.parent / "config/experiment_config.toml"
anaysis_config_file = wd.parent / "config/analysis_config.toml"

notes_file = wd / "notes.json"

with open(notes_file, "w") as f:
    json.dump({}, f)

data_dir = Path("/Users/chris/Documents/PhD-Local/PhD Data/experiment1/data/Lab")


timestamp = pendulum.now().format("YYYYMMDD_HHmmss")
log_dir = wd / "analysis_logs"
log_dir.mkdir(exist_ok=True)
log_files = list(log_dir.glob("*.log"))

if len(log_files) > 0:
    # last_log_file = sorted(log_files, key=os.path.getctime)[-1]
    last_log_file = sorted(log_files)[-1]  # , key=os.path.getctime)[-1]
    last_log_file_N = int(last_log_file.stem.split("-")[1])
else:
    last_log_file_N = -1

log_file = log_dir / f"anlysis_log-{last_log_file_N + 1:03}-{timestamp}.log"
logger.add(log_file)


# * Load experiment config
with open(exp_config_file, "rb") as f:
    # with open(exp_config_file) as f:
    # exp_config = json.load(f)
    exp_config = Box(tomllib.load(f))

# * Load analysis config
with open(anaysis_config_file, "rb") as f:
    analysis_config = Box(tomllib.load(f))


# * ####################################################################################
# * GLOBAL VARS
# * ####################################################################################
pd.set_option("future.no_silent_downcasting", True)

mne_browser_backend = "qt"
mne.viz.set_browser_backend(mne_browser_backend)

# mpl_backend = "ipympl"
mpl_backend = "module://matplotlib_inline.backend_inline"
# mpl_backend = plt.get_backend()
plt.switch_backend(mpl_backend)

rand_seed = 0

eeg_montage = mne.channels.make_standard_montage("biosemi64")

eeg_chan_groups = analysis_config.eeg.ch_groups
all_bad_chans = analysis_config.eeg.bad_channels
eog_chans = analysis_config.eeg.chans.eog
stim_chan = analysis_config.eeg.chans.stim
non_eeg_chans = eog_chans + [stim_chan]

eeg_sfreq: int = 2048
et_sfreq: int = 2000

# * Loading images
icon_images_dir = wd.parent / "experiment-Lab/images"
icon_images = {img.stem: mpimg.imread(img) for img in icon_images_dir.glob("*.png")}

x_pos_stim = analysis_config["stim"]["x_pos_stim"]

img_size = (256, 256)
screen_resolution = (2560, 1440)


y_pos_choices, y_pos_sequence = [-img_size[1], img_size[1]]

# * Getting Valid Event IDs
valid_events = exp_config["lab"]["event_IDs"]
valid_events_inv = {v: k for k, v in valid_events.items()}

# * Time bounds (seconds) for separating trials
# * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
pre_trial_time = 1
post_trial_time = 1


custom_theme = richTheme(
    {"info": "green", "warning": "bright_white on red1", "danger": "bold red"}
)
console = richConsole(theme=custom_theme)


# * ####################################################################################
# * DATA PREPROCESSING AND LOADING
# * ####################################################################################


def load_and_clean_behav_data(subj_N, sess_N):
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


def load_raw_data(subj_N, sess_N, data_dir):
    sess_dir = data_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"

    # * File paths
    et_fpath = [f for f in sess_dir.glob("*.asc")][0]
    eeg_fpath = [f for f in sess_dir.glob("*.bdf")][0]
    # behav_fpath = [f for f in sess_dir.glob("*behav*.csv")][0]
    sess_info_file = [f for f in sess_dir.glob("*sess_info.json")][0]
    sequences_file = wd.parent / f"experiment-Lab/sequences/session_{sess_N}.csv"

    # * Load data
    sess_info = json.load(open(sess_info_file))

    if len(sess_info["Notes"]) > 0:
        logger.warning(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")
        with open(notes_file, "r") as f:
            notes = json.load(f)

        notes.update({f"subj_{subj_N:02}-sess_{sess_N:02}": sess_info["Notes"]})

        with open(notes_file, "w") as f:
            json.dump(notes, f)

        # print(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")

    sequences = pd.read_csv(
        sequences_file, dtype={"choice_order": str, "seq_order": str}
    )

    # raw_behav = pd.read_csv(behav_fpath).merge(sequences, on="item_id")
    raw_behav = load_and_clean_behav_data(subj_N, sess_N).merge(sequences, on="item_id")

    raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=False)

    raw_et = mne.io.read_raw_eyelink(et_fpath)
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
    behav_files = sorted(data_dir.rglob("*behav*.csv"))

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
    subj = int(fpath.parents[1].name.split("_")[1])
    sess_N = int(fpath.parents[0].name.split("_")[1])

    try:
        # let's access the first calibration
        first_cal = et_cals[0]
        # print(f"number of calibrations: {len(et_cals)}")
        # print(first_cal)
    except Exception as e:
        print("WARNING: NO CALIBRATION FOUND", e)

    # raw_et.plot(duration=0.5, scalings=dict(eyegaze=1e3))
    # raw_et.annotations

    # print(f"{raw_et.ch_names = }")

    # chan_xpos, chan_ypos, chan_pupil = raw_et.ch_names
    # x_pos = raw_et[chan_xpos][0][0]
    # y_pos = raw_et[chan_ypos][0][0]

    # * Read events from annotations
    et_events, et_events_dict = mne.events_from_annotations(raw_et, verbose=False)

    # * Convert keys to strings (if they aren't already)
    et_events_dict = {str(k): v for k, v in et_events_dict.items()}

    if et_events_dict.get("exp_start"):
        et_events_dict["experiment_start"] = et_events_dict.pop("exp_start")

    # print("Unique event IDs before update:", np.unique(et_events[:, 2]))

    # * Create a mapping from old event IDs to new event IDs
    id_mapping = {}
    eye_events_idx = 60

    for event_name, event_id in et_events_dict.items():
        if event_name in valid_events:
            new_id = valid_events[event_name]
        else:
            eye_events_idx += 1
            new_id = eye_events_idx
        id_mapping[event_id] = new_id

    # # * Update event IDs in et_events
    for i in range(et_events.shape[0]):
        old_id = et_events[i, 2]
        if old_id in id_mapping:
            et_events[i, 2] = id_mapping[old_id]

    # print("Unique event IDs after update:", np.unique(et_events[:, 2]))

    # * Update et_events_dict with new IDs
    et_events_dict = {k: id_mapping[v] for k, v in et_events_dict.items()}
    et_events_dict = {
        k: v for k, v in sorted(et_events_dict.items(), key=lambda x: x[1])
    }
    et_events_dict_inv = {v: k for k, v in et_events_dict.items()}

    inds_responses = np.where(np.isin(et_events[:, 2], [10, 11, 12, 13, 14, 15, 16]))
    choice_key_et = [valid_events_inv[i] for i in et_events[inds_responses, 2][0]]

    et_events_df = pd.DataFrame(et_events, columns=["sample_nb", "prev", "event_id"])
    et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

    # print("Eye tracking event counts:")
    # display(et_events_df["event_id"].value_counts())

    et_trial_bounds, et_trial_events_df = locate_trials(et_events, et_events_dict)

    # * Remove practice trials
    if sess_N == 1:
        choice_key_et = choice_key_et[3:]
        et_trial_bounds = et_trial_bounds[3:]
        et_trial_events_df = et_trial_events_df.query("trial_id >= 3")
        et_trial_events_df["trial_id"] -= 3

    manual_et_epochs = []

    for start, end in tqdm(et_trial_bounds, desc="Creating ET epochs"):
        # * Get start and end times in seconds
        start_time = (et_events[start, 0] / raw_et.info["sfreq"]) - pre_trial_time
        end_time = et_events[end, 0] / raw_et.info["sfreq"] + post_trial_time

        # * Crop the raw data to this time window
        epoch_data = raw_et.copy().crop(tmin=start_time, tmax=end_time)

        # * Add this epoch to our list
        manual_et_epochs.append(epoch_data)

    # * Print some information about our epochs
    # print(f"Number of epochs created: {len(manual_et_epochs)}")
    # for i, epoch in enumerate(manual_et_epochs):
    #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

    return (
        et_events_dict,
        et_events_dict_inv,
        et_trial_bounds,
        et_trial_events_df,
        manual_et_epochs,
    )


# * ####################################################################################
# * EEG ANALYSIS
# * ####################################################################################


def preprocess_eeg_data(raw_eeg, eeg_chan_groups, raw_behav):
    fpath = Path(raw_eeg.filenames[0])
    subj_N = int(fpath.parents[1].name.split("_")[1])
    sess_dir = fpath.parents[0]
    sess_N = int(sess_dir.name.split("_")[1])

    preprocessed_dir = wd / "results/preprocessed_data/"
    preprocessed_dir.mkdir(exist_ok=True)

    preprocessed_raw_fpath = (
        preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_preprocessed-raw.fif"
    )

    if preprocessed_raw_fpath.is_file():
        raw_eeg = mne.io.read_raw_fif(preprocessed_raw_fpath, preload=False)
        eeg_events, _ = mne.events_from_annotations(raw_eeg, valid_events)

    else:
        # * Setting EOG channels
        raw_eeg.load_data()

        bad_chans = all_bad_chans.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
        raw_eeg.set_channel_types({ch: "eog" for ch in eog_chans})

        montage = mne.channels.make_standard_montage("biosemi64")

        if other_chans := [
            ch for ch in raw_eeg.ch_names if ch not in montage.ch_names + non_eeg_chans
        ]:
            print("WARNING: unknown channels detected. Dropping: ", other_chans)
            logger.warning(
                f"subj_{subj_N} - sess_{sess_N} - Unknown channels found: {other_chans}"
            )
            raw_eeg.drop_channels(other_chans)

        raw_eeg.set_montage(montage)
        raw_eeg.info["bads"] = bad_chans

        # raw_eeg.drop_channels(bad_chans)

        # # * Check channel groups
        check_ch_groups(raw_eeg.get_montage(), eeg_chan_groups)

        # * Average Reference
        raw_eeg.set_eeg_reference(ref_channels="average")

        # * drop bad channels
        # bad_chans = mne.preprocessing.find_bad_channels_lof(raw_eeg)
        # raw_eeg.plot()

        # raw_eeg.info["bads"] = bad_chans
        # raw_eeg.drop_channels(bad_chans)
        # logger.info(f"subj_{subj_N} - sess_{sess_N} - Bad channels: {bad_chans}")

        # * drop bad channels automatically
        # epochs = mne.Epochs(
        #     raw_eeg,
        #     eeg_events,
        #     valid_events["stim-all_stim"],
        #     tmin=-0.1,
        #     tmax=0.6,
        #     baseline=(-0.1, 0),
        # )
        # evoked = epochs.average()

        # chans_95 = np.percentile(evoked.data, [2.5, 97.5], axis=0)[0]
        # low = np.percentile(evoked.data, [2.5, 97.5], axis=0)[0].min()
        # high = np.percentile(evoked.data, [2.5, 97.5], axis=0)[1].max()
        # bad_chans = list(set(np.where((evoked.data < low) | (evoked.data > high))[0]))
        # bad_chans = [raw_eeg.ch_names[i] for i in bad_chans]
        # raw_eeg.info["bads"] = bad_chans

        # evoked.plot()
        # evoked.plot(exclude=bad_chans)
        # * EOG artifact rejection using ICA *
        ica_fpath = preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_fitted-ica.fif"

        if ica_fpath.is_file():
            ica = mne.preprocessing.read_ica(ica_fpath)

        else:
            raw_eeg.filter(
                l_freq=1, verbose=False
            )  # TODO: make that a copy to filter before ICA

            ica = mne.preprocessing.ICA(
                n_components=None,
                noise_cov=None,
                random_state=rand_seed,
                method="fastica",
                max_iter="auto",
            )
            ica.fit(raw_eeg)

            ica.save(ica_fpath)

        eog_inds, eog_scores = ica.find_bads_eog(raw_eeg)

        ica.exclude = eog_inds

        raw_eeg = ica.apply(raw_eeg)

        # TODO: look at IClabel to clean up data further
        # TODO: automatically remove bad channels
        # * e.g., amplitude cutoff

        # * Bandpass Filter: 0.1 - 100 Hz
        raw_eeg.filter(l_freq=0.1, h_freq=100, verbose=False)
        raw_eeg.notch_filter(freqs=50, verbose=False)
        raw_eeg.notch_filter(freqs=100, verbose=False)

        # * Detecting events
        eeg_events = mne.find_events(
            raw_eeg,
            min_duration=0,
            initial_event=False,
            shortest_event=1,
            uint_cast=True,
        )

        annotations = mne.annotations_from_events(
            eeg_events, raw_eeg.info["sfreq"], event_desc=valid_events_inv
        )
        raw_eeg.set_annotations(annotations)

        raw_eeg.save(preprocessed_raw_fpath, overwrite=True)

        del raw_eeg

        raw_eeg = mne.io.read_raw_fif(preprocessed_raw_fpath, preload=False)

    choice_key_eeg = [
        valid_events_inv[i]
        for i in eeg_events[:, 2]
        if i in [10, 11, 12, 13, 14, 15, 16]
    ]

    print("\n\nLOCATING EEG EVENTS\n\n")
    print(f"{eeg_events = }")
    print(f"{valid_events = }")
    eeg_trial_bounds, eeg_events_df = locate_trials(eeg_events, valid_events)

    # * Remove practice trials
    if sess_N == 1:
        choice_key_eeg = choice_key_eeg[3:]
        eeg_trial_bounds = eeg_trial_bounds[3:]

    raw_behav["choice_key_eeg"] = choice_key_eeg
    raw_behav["same"] = raw_behav["choice_key"] == raw_behav["choice_key_eeg"]

    manual_eeg_trials = []

    # * Loop through each trial
    for start, end in tqdm(eeg_trial_bounds, "Creating EEG epochs"):
        # * Get start and end times in seconds
        start_time = (eeg_events[start, 0] / raw_eeg.info["sfreq"]) - pre_trial_time
        end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + post_trial_time

        # * Crop the raw data to this time window
        epoch_data = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)

        # * Add this epoch to our list
        manual_eeg_trials.append(epoch_data)

    # * Print some information about our epochs
    # print(f"Number of epochs created: {len(manual_eeg_trials)}")
    # for i, epoch in enumerate(manual_eeg_trials):
    #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

    return raw_eeg, manual_eeg_trials, eeg_trial_bounds, eeg_events, eeg_events_df


# * ####################################################################################
# * CREATE RDMs
# * ####################################################################################


def calc_rdm_manual(cleaned_behav_data):
    """Calculate the RDM manually using squared Euclidean distance.

    Args:
        cleaned_behav_data (dict): A dictionary where keys are patterns and values
        are their mean accuracies.

    Returns:
        pd.DataFrame: A DataFrame representing the RDM with patterns as labels.
    """
    patterns = list(cleaned_behav_data.keys())
    accuracies = np.array(list(cleaned_behav_data.values()))

    # * Initialize an empty matrix
    n = len(patterns)
    rdm = np.zeros((n, n))

    # * Compute squared Euclidean distance for each pair of patterns
    for i in range(n):
        for j in range(n):
            rdm[i, j] = (accuracies[i] - accuracies[j]) ** 2

    # * Convert to DataFrame for readability
    rdm_df = pd.DataFrame(rdm, index=patterns, columns=patterns)

    return rdm_df


def compare_rdms(rdm1, rdm2, method="pearson"):
    avail_methods = ["pearson", "spearman"]
    if method not in avail_methods:
        raise ValueError(f"Method should be either {avail_methods}")

    # Extract the upper triangle (excluding the diagonal)
    rdm1_flattened = rdm1[np.triu_indices_from(rdm1, k=1)]
    rdm2_flattened = rdm2[np.triu_indices_from(rdm2, k=1)]

    if method == "pearson":
        # Pearson correlation
        corr, _ = pearsonr(rdm1_flattened, rdm2_flattened)
    else:
        # Spearman correlation
        corr, _ = spearmanr(rdm1_flattened, rdm2_flattened)

    return corr


def get_rdms_negative_peak_eeg(
    erp_data, time_window=(0, 0.2), method="euclidean", show_plots=True
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
            # * Get data for the time window
            times = evoked.times
            chans = evoked.ch_names
            selected_chans = [c for c in chans if c in eeg_chan_groups["occipital"]]
            selected_chans = [c for c in selected_chans if c not in evoked.info["bads"]]

            time_mask = (times >= time_window[0]) & (times <= time_window[1])
            # data = evoked.get_data(picks=selected_chans).mean(axis=0)
            data = evoked.get_data().mean(axis=0)

            # * Find the first negative peak
            negative_peak_idx = np.argmin(data[time_mask])
            peak_latency = times[time_mask][negative_peak_idx]
            peak_amplitude = data[time_mask][negative_peak_idx]

            # * Append results
            latency_features[participant].append(peak_latency)
            amplitude_features[participant].append(peak_amplitude)

    # * Normalize latency and amplitude separately
    # normalized_latency_features = {
    #     p: normalize(latencies) for p, latencies in latency_features.items()
    # }
    # normalized_amplitude_features = {
    #     p: normalize(amplitudes) for p, amplitudes in amplitude_features.items()
    # }

    # latency_features = normalized_latency_features
    # amplitude_features = normalized_amplitude_features

    # combined_features = np.vstack((
    #     normalized_latency_features[participant],
    #     normalized_amplitude_features[participant]
    # )).T

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


# * ####################################################################################
# * ANALYZE FLASH PERIOD
# * ####################################################################################
def analyze_flash_period(et_epoch, eeg_epoch, epoch_N):
    # ! IMPORTANT
    # TODO
    # * eeg_baseline
    # TODO

    # * Extract epoch data
    # et_epoch = manual_et_epochs[epoch_N]
    # eeg_epoch = manual_eeg_epochs[epoch_N]

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]
        # last_flash = annotations.query("description.str.contains('flash')").iloc[-1]
        # last_fixation = (
        #     annotations.iloc[: all_stim_pres.name]
        #     .query("description == 'fixation'")
        #     .iloc[-1]
        # )

        # * Convert to seconds
        # end_time = last_fixation["onset"] + last_fixation["duration"]
        start_time = (
            annotations.iloc[: first_flash.name]
            .query("description == 'fixation'")
            .iloc[-1]["onset"]
        )
        end_time = all_stim_pres["onset"]

        # * Crop the data
        # epoch = epoch.copy().crop(first_flash["onset"], last_fixation["onset"])
        epoch = epoch.copy().crop(start_time, end_time)

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # time_bounds = (first_flash["onset"], end_time)
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
        for group_name, group_chans in eeg_chan_groups.items()
    }

    # * Get positions and presentation order of stimuli
    trial_info = trial_info = get_trial_info(
        epoch_N,
        raw_behav,
        x_pos_stim,
        y_pos_choices,
        y_pos_sequence,
        screen_resolution,
        img_size,
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

        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_title(title)

        # * Plot target icon
        ax_et.imshow(
            icon_images[target_name],
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
            img_size[0],
            img_size[1],
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
        ax_eeg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    return fixation_data, eeg_fixation_data


# * ####################################################################################
# * ANALYZE DECISION PERIOD
# * ####################################################################################


def analyze_decision_period(
    manual_eeg_trials: List[mne.io.Raw],
    manual_et_trials: List[mne.io.eyelink.eyelink.RawEyelink],
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
    # ! TEMP

    # * Extract epoch data
    et_trial = manual_et_trials[trial_N].copy()
    eeg_trial = manual_eeg_trials[trial_N].copy()

    assert eeg_sfreq == eeg_trial.info["sfreq"], "EEG data has incorrect sampling rate"

    def crop_et_trial(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target
        # epoch = et_trial.copy()

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

        response_ids = exp_config.lab.allowed_keys + ["timeout", "invalid"]
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

    # * Crop the data
    et_trial, et_annotations, time_bounds = crop_et_trial(et_trial)

    # * Adjust time bounds for EEG baseline and window
    # * Cropping with sample bounds
    trial_duration = (time_bounds[1] + eeg_window + eeg_baseline) - time_bounds[0]
    sample_bounds = [None, None]
    sample_bounds[0] = int(time_bounds[0] * eeg_sfreq)
    sample_bounds[1] = sample_bounds[0] + int(np.ceil(trial_duration * eeg_sfreq))

    eeg_trial = eeg_trial.copy().crop(
        eeg_trial.times[sample_bounds[0]], eeg_trial.times[sample_bounds[1]]
    )

    # * Get channel positions for topomap
    eeg_info = eeg_trial.info

    chans_pos_xy = np.array(
        list(eeg_info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    # * Get info on the current trial
    trial_info = get_trial_info(
        trial_N,
        raw_behav,
        x_pos_stim,
        y_pos_choices,
        y_pos_sequence,
        screen_resolution,
        img_size,
    )

    stim_pos, stim_order, sequence, choices, response_ind, solution, rt = trial_info
    solution_ind = {v: k for k, v in choices.items()}[solution]
    # response = choices.get(response_ind, "timeout")
    # correct = response == solution

    # * Get the onset of the response event
    response_event = et_annotations.query(
        "description.isin(['a', 'x', 'm', 'l', 'timeout', 'invalid'])"
    ).iloc[0]

    response_onset = response_event["onset"]

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

    for idx_fix, fixation_ind in enumerate(
        tqdm(fixation_inds, leave=False, disable=pbar_off)
    ):
        # * Get number of flash events before the current fixation; -1 to get the index
        fixation = et_annotations.loc[fixation_ind]

        fixation_start = fixation["onset"]
        fixation_duration = fixation["duration"]
        fixation_end = fixation_start + fixation_duration

        end_time = min(fixation_end, et_trial.times[-1])

        gaze_x, gaze_y, pupil_diam = (
            et_trial.copy().crop(fixation_start, end_time).get_data()
        )

        # * Determine if fixation is on target
        mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

        on_target = False

        for i, (icon_name, pos) in enumerate(stim_pos):
            targ_left, targ_right, targ_bottom, targ_top = pos
            if (
                targ_left <= mean_gaze_x <= targ_right
                and targ_bottom <= mean_gaze_y <= targ_top
            ):
                on_target = True
                stim_ind = i
                break

        # * Crop by sample bounds
        eeg_start_sample = int(fixation_start * eeg_sfreq)
        eeg_end_sample = eeg_start_sample + int(
            np.ceil((eeg_window + eeg_baseline) * eeg_sfreq)
        )
        eeg_start_time = eeg_trial.times[eeg_start_sample]

        eeg_end_time = eeg_trial.times[eeg_end_sample]
        eeg_slice = eeg_trial.copy().crop(eeg_start_time, eeg_end_time)

        # fig, axes = plt.subplots(2, 1, figsize=(10, 4))
        # ax1, ax2 = axes
        # ax1.plot(eeg_slice.get_data(picks="eeg").T)
        # ax1.set_title(f"Fixation {idx_fix}")

        # slice_annotations = eeg_slice.annotations
        # mne.events_from_annotations(eeg_slice)

        # times = eeg_slice.times
        eeg_slice = eeg_slice.get_data()
        # eeg_slice = mne.baseline.rescale(eeg_slice, times, baseline=(None, eeg_baseline), mode="mean")

        ep_ = mne.EpochsArray(
            [eeg_slice], eeg_trial.info, tmin=-eeg_baseline, verbose=False
        )

        # , baseline=(None, 0), verbose=False)

        ep_ = ep_.apply_baseline(baseline=(None, 0), verbose=False)
        ep_.detrend = 1

        eeg_slice = ep_

        # ax2.plot(eeg_slice.average().get_data(picks="eeg").T)
        # ax2.plot(eeg_slice.get_data(picks="eeg")[0].T)
        # ax2.set_title(f"Fixation {idx_fix} - Baseline corrected")
        # ax2.sharex(ax1)
        # ax2.set_xlim(0, eeg_slice.shape[1])
        # ax2.set_xticks(np.arange(0, eeg_slice.shape[1], 100), ((np.arange(0, eeg_slice.shape[1], 100) / eeg_sfreq) * 1000).astype(int))
        # plt.tight_layout()
        # plt.show()

        # * Check if fixation is on target and longer than 100 ms
        if fixation_duration >= 0.1 and on_target:
            # if on_target:
            discarded = False

            fixation_data[stim_ind].append(np.array([gaze_x, gaze_y]))

            gaze_target_fixation_sequence.append(
                [stim_ind, fixation_start, fixation_duration, pupil_diam.mean()]
            )

            eeg_fixation_data[stim_ind].append(eeg_slice)

        else:
            # * Only for visualization purposes
            discarded = True

        if show_plots:
            # * Select EEG channel groups to plot
            selected_chan_groups = {
                k: v
                for k, v in eeg_chan_groups.items()
                if k
                in [
                    "frontal",
                    "parietal",
                    "central",
                    "temporal",
                    "occipital",
                ]
            }

            group_colors = dict(
                zip(
                    selected_chan_groups.keys(),
                    ["red", "green", "blue", "pink", "orange"],
                )
            )

            # * Get channel indices for each channel group
            ch_group_inds = {
                group_name: [
                    i for i, ch in enumerate(eeg_info.ch_names) if ch in group_chans
                ]
                for group_name, group_chans in selected_chan_groups.items()
            }

            title = f"ICON-{stim_ind}" if on_target else "OFF-TARGET"
            title += f" ({fixation_duration * 1000:.0f} ms)"
            title += " - " + ("DISCARDED" if discarded else "SAVED")

            # fig = plot_eeg_and_gaze_fixations_plotly(
            plot_eeg_and_gaze_fixations(
                # * Convert to microvolts
                # eeg_data=eeg_slice * 1e6,
                eeg_data=eeg_slice.get_data(picks="eeg") * 1e6,
                eeg_sfreq=eeg_sfreq,
                et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                eeg_baseline=eeg_baseline,
                response_onset=response_onset,
                eeg_start_time=eeg_start_time,
                eeg_end_time=eeg_end_time,
                icon_images=icon_images,
                img_size=img_size,
                stim_pos=stim_pos,
                chans_pos_xy=chans_pos_xy,
                ch_group_inds=ch_group_inds,
                group_colors=group_colors,
                screen_resolution=screen_resolution,
                title=title,
                vlines=[
                    eeg_baseline * eeg_sfreq,
                    eeg_baseline * eeg_sfreq + fixation_duration * eeg_sfreq,
                ],
            )

            # plt.savefig(
            #     wd
            #     / f"subj_{subj_N:02}-sess_{sess_N:02}-trial_{epoch_N:02}-fixation{idx_fix:02}.png"
            # )

    # plt.close('all')

    # * GETTING ERPs
    eeg_fixation_data = {
        fixation_ind: mne.concatenate_epochs(data, verbose=False)
        for fixation_ind, data in eeg_fixation_data.items()
        if len(data) > 0
    }

    eeg_fixations_sequence = {
        k: v
        for k, v in eeg_fixation_data.items()
        if k in sequence_icon_inds and len(v) > 0
    }

    eeg_fixations_choices = {
        k: v
        for k, v in eeg_fixation_data.items()
        if k in choice_icon_inds and len(v) > 0
    }

    # * Calculate ERPs to fixations on sequence icons
    if len(eeg_fixations_sequence) > 0:
        fixations_sequence_erp = mne.concatenate_epochs(
            list(eeg_fixations_sequence.values()), verbose=False
        ).average()
    else:
        # fixations_sequence_erp = np.array([])
        fixations_sequence_erp = None

    # * Calculate ERPs to fixations on choice icons
    if len(eeg_fixations_choices) > 0:
        fixations_choices_erp = mne.concatenate_epochs(
            list(eeg_fixations_choices.values()), verbose=False
        ).average()
    else:
        # fixations_choices_erp = np.array([])
        fixations_choices_erp = None

    # fixations_choices_erp = fixations_choices_erp.apply_baseline((None, 0))
    # fixations_choices_erp.detrend=1
    # fixations_choices_erp.plot()
    # t = eeg_fixations_sequence[0].average().apply_baseline((None, 0)).plot()
    # t = t.plot()

    # * ################################################################################
    # * GAZE ANALYSIS
    gaze_target_fixation_sequence = pd.DataFrame(
        gaze_target_fixation_sequence,
        columns=["stim_ind", "onset", "duration", "pupil_diam"],
    )

    gaze_target_fixation_sequence["trial_N"] = trial_N

    gaze_target_fixation_sequence["stim_name"] = gaze_target_fixation_sequence[
        "stim_ind"
    ].replace(seq_and_choices)

    gaze_target_fixation_sequence["pupil_diam"] = gaze_target_fixation_sequence[
        "pupil_diam"
    ].round(2)
    gaze_target_fixation_sequence["stim_type"] = gaze_target_fixation_sequence[
        "stim_ind"
    ].replace(stim_types)

    first_fixation_order = (
        gaze_target_fixation_sequence.sort_values("onset")
        .groupby("stim_ind")
        .first()["onset"]
        .rank()
        .astype(int)
    )

    first_fixation_order.name = "first_fix_order"

    mean_duration_per_target = (
        gaze_target_fixation_sequence.groupby("stim_ind")["duration"].mean().round(2)
    )

    mean_diam_per_target = (
        gaze_target_fixation_sequence.groupby("stim_ind")["pupil_diam"].mean().round()
    )

    fix_counts_per_target = gaze_target_fixation_sequence["stim_ind"].value_counts()

    total_fix_duration_per_target = gaze_target_fixation_sequence.groupby("stim_ind")[
        "duration"
    ].sum()

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

    gaze_info["stim_name"] = gaze_info["stim_ind"].replace(seq_and_choices)
    gaze_info["trial_N"] = trial_N
    gaze_info["stim_type"] = gaze_info["stim_ind"].replace(stim_types)
    gaze_info.sort_values("stim_ind", inplace=True)

    # * ################################################################################
    # gaze_info.query("stim_ind in @sequence_icon_inds")
    # gaze_info.query("stim_ind == @choice_icon_inds")
    # gaze_info.query("stim_ind == @wrong_choice_icon_inds")
    # gaze_info.query("stim_ind == @solution_ind")

    # gaze_info

    # gaze_target_fixation_sequence.query("stim_ind == @sequence_icon_inds").groupby(
    #     "stim_ind"
    # )["duration"].mean().plot(kind="bar")
    # gaze_target_fixation_sequence["duration"].plot()
    # gaze_target_fixation_sequence["pupil_diam"].plot()
    # gaze_target_fixation_sequence.groupby("stim_ind")["pupil_diam"].plot()

    return (
        fixation_data,
        eeg_fixation_data,
        gaze_target_fixation_sequence,
        gaze_info,
        fixations_sequence_erp,
        fixations_choices_erp,
    )


def analyze_phase_coupling():
    # see: https://etiennecmb.github.io/tensorpac/auto_examples/erpac/plot_erpac.html#sphx-glr-auto-examples-erpac-plot-erpac-py

    # * alpha (8–13 Hz), beta (13–30 Hz), delta (0.5–4 Hz), and theta (4–7 Hz)

    # * Example phase-amplitude coupling analysis using tensorpac
    def pac_analysis(epochs, low_freqs=(2, 20, 10), high_freqs=(60, 150, 30)):
        """
        Analyze phase-amplitude coupling using tensorpac.

        Parameters:
        -----------
        epochs : mne.Epochs
            MNE epochs object containing the data
        low_freqs : tuple
            (start, stop, n_freqs) for phase frequencies
        high_freqs : tuple
            (start, stop, n_freqs) for amplitude frequencies

        Returns:
        --------
        pac : tensorpac.Pac
            PAC object containing coupling results
        """
        import tensorpac

        # * Extract data and sampling frequency
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]

        # * Create phase and amplitude frequency vectors
        f_pha = np.linspace(low_freqs[0], low_freqs[1], low_freqs[2])
        f_amp = np.linspace(high_freqs[0], high_freqs[1], high_freqs[2])
        # f_pha =
        # f_amp =
        # * Initialize PAC object
        p = tensorpac.Pac(idpac=(6, 2, 3), f_pha=f_pha, f_amp=f_amp, dcomplex="wavelet")

        # * Compute PAC
        pac = p.filterfit(data, sfreq, n_jobs=-1)

        return pac, p, f_pha, f_amp

    pass


# * ####################################################################################
# * MAIN FUNCTIONS
# * ####################################################################################


def main():
    res_dir = wd / "results/analyzed"
    res_dir.mkdir(exist_ok=True, parents=True)

    n_subjs = len(list(data_dir.glob("subj_*")))

    for subj_N in tqdm(
        range(1, n_subjs + 1),
        desc="Analyzing data of every subjects",
    ):
        # ! TEMP
        # subj_N = 6
        # sess_N = 1
        # trial_N = 47
        # ! TEMP

        subj_dir = data_dir / f"subj_{subj_N:02}"

        n_sessions = len(list(subj_dir.glob("sess_*")))

        for sess_N in range(1, n_sessions + 1):
            # for sess_N in [4]:
            save_dir = res_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"
            save_dir.mkdir(exist_ok=True, parents=True)

            if save_dir.exists() and len(list(save_dir.glob("*.pkl"))) > 0:
                continue
            else:
                sess_erps = {"sequence": [], "choices": []}

                sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals = (
                    load_raw_data(subj_N, sess_N, data_dir)
                )

                # sess_screen_resolution = sess_info["window_size"]
                # sess_img_size = sess_info["img_size"]
                # et_sfreq = raw_et.info["sfreq"]

                assert (
                    et_sfreq == raw_et.info["sfreq"]
                ), "Eye-tracking data has incorrect sampling rate"

                assert (
                    eeg_sfreq == raw_eeg.info["sfreq"]
                ), "EEG data has incorrect sampling rate"

                tracked_eye = sess_info["eye"]
                vision_correction = sess_info["vision_correction"]
                eye_screen_distance = sess_info["eye_screen_dist"]

                (
                    et_events_dict,
                    et_events_dict_inv,
                    et_trial_bounds,
                    et_trial_events_df,
                    manual_et_trials,
                ) = preprocess_et_data(raw_et, et_cals)

                (
                    raw_eeg,
                    manual_eeg_trials,
                    eeg_trial_bounds,
                    eeg_events,
                    eeg_events_df,
                ) = preprocess_eeg_data(raw_eeg, eeg_chan_groups, raw_behav)

                bad_chans = raw_eeg.info["bads"]
                del raw_eeg

                gaze_info_all = []
                gaze_target_fixation_sequence_all = []

                for trial_N in tqdm(
                    raw_behav.index, desc="Analyzing every trial", leave=False
                ):
                    (
                        fixation_data,
                        eeg_fixation_data,
                        gaze_target_fixation_sequence,
                        gaze_info,
                        fixations_sequence_erp,
                        fixations_choices_erp,
                    ) = analyze_decision_period(
                        manual_eeg_trials,
                        manual_et_trials,
                        raw_behav,
                        trial_N,
                        eeg_baseline=0.100,
                        eeg_window=0.600,
                        show_plots=False,
                    )
                    sess_erps["sequence"].append(fixations_sequence_erp)
                    sess_erps["choices"].append(fixations_choices_erp)

                    gaze_info_all.append(gaze_info)
                    gaze_target_fixation_sequence_all.append(
                        gaze_target_fixation_sequence
                    )

                # [f.plot() for f in sess_erps["sequence"]]
                # mne.combine_evoked([e for e in sess_erps["sequence"] if type(e)!=np.ndarray], "equal").plot()
                # mne.combine_evoked([e for e in sess_erps["choices"] if type(e)!=np.ndarray], "equal").plot()

                # plt.plot(np.mean([erp.get_data(picks='eeg') for erp in sess_erps["sequence"]], axis=0).T)

                def get_avg_erps():
                    pass
                    # sess_erps["choices"] = [
                    #     erp for erp in sess_erps["choices"] if len(erp) > 0
                    # ]
                    # sess_erps["sequence"] = [
                    #     erp for erp in sess_erps["sequence"] if len(erp) > 0
                    # ]

                    # avg_choices_erp = np.stack(sess_erps["choices"]).mean(axis=0)
                    # avg_sequence_erp = np.stack(sess_erps["sequence"]).mean(axis=0)

                    # avg_erp_all_stim = np.stack(
                    #     sess_erps["choices"] + sess_erps["sequence"]
                    # ).mean(axis=0)
                    # mne.concatenate_epochs(sess_erps["choices"])

                    # for data in [avg_choices_erp, avg_sequence_erp, avg_erp_all_stim]:
                    #     plt.plot(data.T)
                    #     plt.show()
                    #     plt.plot(data[-10:].mean(axis=0).T)
                    #     plt.show()

                    # return avg_choices_erp, avg_sequence_erp, avg_erp_all_stim

                gaze_info = pd.concat(gaze_info_all)

                gaze_target_fixation_sequence = pd.concat(
                    gaze_target_fixation_sequence_all
                )

                gaze_target_fixation_sequence.reset_index(
                    drop=False, inplace=True, names=["fixation_N"]
                )

                with open(save_dir / "fixation_data.pkl", "wb") as f:
                    pickle.dump(fixation_data, f)

                with open(save_dir / "eeg_fixation_data.pkl", "wb") as f:
                    pickle.dump(eeg_fixation_data, f)

                with open(save_dir / "gaze_info.pkl", "wb") as f:
                    pickle.dump(gaze_info, f)

                with open(
                    save_dir / "gaze_target_fixation_sequence.pkl",
                    "wb",
                ) as f:
                    pickle.dump(gaze_target_fixation_sequence, f)

                with open(
                    save_dir / "sess_erps.pkl",
                    "wb",
                ) as f:
                    pickle.dump(sess_erps, f)

    # * ################################################################################
    # sess_bad_chans = raw_eeg.info["bads"]
    sess_bad_chans = all_bad_chans.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
    ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
    ch_group_colors = ["red", "green", "blue", "pink", "orange"]

    selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
        prepare_eeg_data_for_plot(
            eeg_chan_groups,
            eeg_montage,
            non_eeg_chans,
            sess_bad_chans,
            ch_group_names,
            ch_group_colors,
        )
    )

    fig_titles = ["ERP - Sequence Icons", "ERP - Choice Icons"]

    for eeg_data, title in zip(
        [fixations_sequence_erp, fixations_choices_erp], fig_titles
    ):
        fig = plot_eeg(
            eeg_data.get_data(picks=selected_chans_names) * 1e6,
            chans_pos_xy,
            ch_group_inds,
            group_colors,
            eeg_sfreq,
            eeg_baseline=0.1,  # TODO: check what eeg_baseline does & if 0.1 is correct
            vlines=0.1,
            title=title,
        )

    # gaze_target_fixation_sequence.groupby(["trial_N", "stim_ind"])[
    #     "duration"
    # ].mean()

    # fig, ax = plt.subplots()
    # for trial_N in gaze_target_fixation_sequence["trial_N"].unique():
    #     temp = gaze_target_fixation_sequence.query("trial_N == @trial_N")
    #     temp["pupil_diam"].reset_index(drop=True).plot(ax=ax)
    # ax.set_title("Pupil diameter over time")
    # # plt.show()
    # # plt.close()

    # fig, ax = plt.subplots()
    # for trial_N in gaze_target_fixation_sequence["trial_N"].unique():
    #     temp = gaze_target_fixation_sequence.query("trial_N == @trial_N")
    #     temp["duration"].reset_index(drop=True).plot(ax=ax)
    # ax.set_title("Fixation duration over time")
    # plt.show()
    # plt.close("all")


def inspect_results():
    # plt.close("all")
    # plt.get_backend()
    # plt.switch_backend("webagg")
    # plt.switch_backend(mpl_backend)

    res_dir = wd / "results/analyzed-banpdass[0.1-100]"  # /Oct27-Seq_and_choices"

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    erps = {}
    erp_files = sorted(res_dir.glob("sub*/sess*/*erps.pkl"))
    behav_files = sorted(data_dir.glob("sub*/sess*/*behav.csv"))
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    behav_files = list(data_dir.rglob("*behav.csv"))

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

    behav_data = behav_data.sort_values(["subj_N", "sess_N", "trial_N"]).reset_index(
        drop=True
    )
    gaze_data = pd.concat(gaze_data, axis=0)
    gaze_data = gaze_data.merge(
        behav_data[["subj_N", "sess_N", "trial_N", "pattern", "item_id"]],
        on=["subj_N", "sess_N", "trial_N"],
    )

    gaze_data = gaze_data.sort_values(
        ["subj_N", "sess_N", "trial_N", "stim_ind"]
    ).reset_index(drop=True)

    gaze_data["mean_duration"].mean()

    gaze_data.groupby(["subj_N", "sess_N"])["mean_pupil_diam"].mean()
    gaze_data.groupby(["subj_N"])["mean_pupil_diam"].mean()

    # * ################################################################################
    # * Inspect ERP data
    # * ################################################################################

    eeg_baseline = 0.1  # baseline correction period in seconds, used for plotting here
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
    # change to line below, but make sure it doesn't change the rest of the analysis:
    # patterns = behav_data["pattern"].unique()

    all_subj_seq_erps = []
    all_subj_choices_erps = []
    all_subj_overall_erps = []

    subj_pattern_erps = {}
    all_subj_pattern_erps = {p: [] for p in patterns}

    for erp_file in sorted(erp_files):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        subj_pattern_erps[subj_N] = {p: [] for p in patterns}

    for erp_file in tqdm(sorted(erp_files)):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        sess_N = int(erp_file.parents[0].stem.split("_")[-1])

        behav_file = list(
            data_dir.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )[0]
        raw_behav = pd.read_csv(behav_file, index_col=0)

        sess_bad_chans = all_bad_chans[f"subj_{subj_N}"][f"sess_{sess_N}"]

        selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            prepare_eeg_data_for_plot(
                eeg_chan_groups,
                eeg_montage,
                non_eeg_chans,
                sess_bad_chans,
                ch_group_names,
                ch_group_colors,
            )
        )

        with open(erp_file, "rb") as f:
            erp_data = pickle.load(f)

        sequence_erps = erp_data["sequence"]
        choice_erps = erp_data["choices"]

        for pattern, this_pattern_erp in list(zip(raw_behav["pattern"], sequence_erps)):
            if isinstance(this_pattern_erp, mne.EvokedArray):
                subj_pattern_erps[subj_N][pattern].append(this_pattern_erp)
                # all_subj_pattern_erps[pattern].extend(erp_data["sequence"])

        sequence_erps = [
            erp for erp in sequence_erps if isinstance(erp, mne.EvokedArray)
        ]

        choice_erps = [erp for erp in choice_erps if isinstance(erp, mne.EvokedArray)]

        mean_sequence_erp = mne.combine_evoked(sequence_erps, "equal")
        mean_choices_erp = mne.combine_evoked(choice_erps, "equal")
        mean_overall_erp = mne.combine_evoked(sequence_erps + choice_erps, "equal")

        all_subj_seq_erps.append(mean_sequence_erp)
        all_subj_choices_erps.append(mean_choices_erp)
        all_subj_overall_erps.append(mean_overall_erp)

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
                    eeg_sfreq,
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
                    eeg_sfreq,
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

    for subj_N in subj_pattern_erps.keys():
        print(f"Subject {subj_N}")
        for i, (pattern, this_pattern_erps) in enumerate(
            subj_pattern_erps[subj_N].items()
        ):
            print(f"\t{i+1}. Pattern {pattern}: {len(this_pattern_erps)} trials")

    for subj_N in tqdm(subj_pattern_erps.keys()):
        for pattern, this_pattern_erps in subj_pattern_erps[subj_N].items():
            if len(this_pattern_erps) > 0:
                mean_pattern_erp = mne.combine_evoked(this_pattern_erps, "equal")
                subj_pattern_erps[subj_N][pattern] = mean_pattern_erp

    for pattern in patterns:
        temp = []
        for subj_N in subj_pattern_erps.keys():
            temp.append(subj_pattern_erps[subj_N][pattern])

        all_subj_pattern_erps[pattern] = mne.combine_evoked(temp, "equal")

    # * Plot ERPs
    for patt_erp in all_subj_pattern_erps.values():
        patt_erp.plot()
        patt_erp.apply_baseline((None, 0)).plot()

    all_subj_seq_erps = mne.combine_evoked(all_subj_seq_erps, "equal")
    all_subj_choices_erps = mne.combine_evoked(all_subj_choices_erps, "equal")
    all_subj_overall_erps = mne.combine_evoked(all_subj_overall_erps, "equal")

    # all_subj_pattern_erps = {
    #     k: mne.combine_evoked(v, "equal") for k, v in all_subj_pattern_erps.items()
    # }

    all_subj_bad_chans = all_subj_seq_erps.info["bads"]

    selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
        prepare_eeg_data_for_plot(
            eeg_chan_groups,
            eeg_montage,
            non_eeg_chans,
            all_subj_bad_chans,
            ch_group_names,
            ch_group_colors,
        )
    )

    for eeg_data, title in zip(
        [all_subj_seq_erps, all_subj_choices_erps, all_subj_overall_erps],
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
            eeg_sfreq,
            eeg_baseline,
            vlines=None,
            title=title,
        )

    for pattern, eeg_data in all_subj_pattern_erps.items():
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
            eeg_sfreq,
            eeg_baseline,
            vlines=None,
            title="All Subjects\n" + f"Pattern: {pattern}",
        )
        plt.show()

    # * ################################################################################
    # * Get RDMs
    # * ################################################################################

    rdm_method = "euclidean"

    rdms_erp_latency, rdms_erp_amplitude, rdms_erp_combined = (
        get_rdms_negative_peak_eeg(
            subj_pattern_erps, method=rdm_method, show_plots=False
        )
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

    rdms = dict(
        rdms_erp_latency=rdms_erp_latency,
        rdms_erp_amplitude=rdms_erp_amplitude,
        rdms_erp_combined=rdms_erp_combined,
        rdms_accuracy=rdms_accuracy,
        rdms_gaze_duration=rdms_gaze_duration,
        rdms_pupil_diam=rdms_pupil_diam,
    )

    labels = list(rdms_erp_latency[1].pattern_descriptors.values())[0]

    def get_group_rdms(rdms, labels):
        all_subj_rdms = {k: None for k in rdms.keys()}

        for rdm_type, subjects_rdm in rdms.items():
            # ! TEMP
            for subj, subj_rdm in subjects_rdm.items():
                fig, ax = plt.subplots()
                plot_matrix(
                    subj_rdm.get_matrices()[0],
                    labels=labels,
                    title=f"{rdm_type}",
                    show_values=True,
                    norm="max",
                    as_pct=True,
                    ax=ax,
                )
                fig.savefig(
                    wd / "results" / f"subj_{subj:02}-{rdm_type}.png",
                    dpi=200,
                    bbox_inches="tight",
                )
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
            fig.savefig(
                wd / "results" / f"{title}-Group_lvl.png",
                dpi=200,
                bbox_inches="tight",
            )
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
        fig.savefig(
            wd / "results" / f"RDMs-Correlation-Group_lvl-{title}.png",
            dpi=200,
            bbox_inches="tight",
        )
        # ! TEMP

        plt.show()

        return all_subj_rdms, dfs_corr

    group_rdms, correlations = get_group_rdms(rdms, labels)
    # TODO: get RDMs for RT

    correlations[0]
    correlations[1]

    return all_subj_pattern_erps


def inspect_behav():
    res = behav_analysis_all(return_raw=True)

    (
        res_by_pattern,
        overall_acc,
        overall_rt,
        rt_by_correct,
        rt_by_correct_and_pattern,
        raw_data,
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

    for subj_N in res_by_pattern["subj_N"].unique():
        data = (
            res_by_pattern.query(f"subj_N == {subj_N}")
            .groupby(["sess_N", "pattern"])["accuracy"]
            .mean()
            .unstack()
            .T
        )
        fig, ax = init_fig(fig_params)
        data.plot(kind="bar", ax=ax, title=f"Subject {subj_N}")
        # data.plot(kind='box', ax=ax, title=f"Subject {subj_N}")
        legend = ax.get_legend()
        ax.legend(
            title=legend.get_title().get_text(),
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
        )
        ax.set_ylabel("Accuracy")
        plt.show()

        data = (
            overall_acc.query(f"subj_N == {subj_N}")
            .groupby("sess_N")["accuracy"]
            .mean()
        )
        fig, ax = init_fig(fig_params)
        data.plot(kind="bar", ax=ax, title=f"Subject {subj_N}")
        ax.set_ylabel("Accuracy")
        plt.show()

    data = overall_acc.groupby(["subj_N", "sess_N"])["accuracy"].mean().unstack()
    fig, ax = init_fig(fig_params)
    data.plot(
        kind="bar",
        ax=ax,
        title="Overall by Session",
        xlabel="Subject",
        ylabel="Accuracy",
    )
    legend = ax.get_legend()
    ax.legend(
        title=legend.get_title().get_text(), bbox_to_anchor=(1.005, 1), loc="upper left"
    )
    plt.show()

    data = res_by_pattern.groupby(["pattern", "subj_N"])["accuracy"].mean().unstack()
    fig, ax = init_fig(fig_params)
    data.plot(kind="line", ax=ax, title="Accuracy by Pattern", ylabel="Accuracy")
    ax.plot(data.mean(axis=1), color="black", linestyle="--", label="Mean")
    legend = ax.get_legend()
    ax.legend(
        title=legend.get_title().get_text(), bbox_to_anchor=(1.005, 1), loc="upper left"
    )
    ax.tick_params(axis="x", rotation=70)
    plt.show()

    data = overall_acc.groupby(["subj_N"])["accuracy"].mean()
    fig, ax = init_fig(fig_params)
    data.plot(
        kind="bar", ax=ax, title="Overall Accuracy", xlabel="Subject", ylabel="Accuracy"
    )
    ax.axhline(data.mean(), color="black", linestyle="--", label="Mean")
    plt.show()

    overall_acc.groupby(["subj_N", "sess_N"])["accuracy"].mean().unstack()


# clear_jupyter_artifacts()
# memory_usage = get_memory_usage()
# memory_usage.head(50)

if __name__ == "__main__":
    pass
