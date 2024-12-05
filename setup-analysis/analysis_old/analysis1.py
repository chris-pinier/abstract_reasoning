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
import mplcursors
import numpy as np
import pandas as pd
import pendulum
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.subplots as ps
import polars as pl
import pylustrator
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
from scipy.interpolate import interp1d
from tensorpac import EventRelatedPac
from tensorpac.signals import pac_signals_wavelet
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from itertools import combinations


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
et_sfreq: int = 1000

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
# * UTILS
# * ####################################################################################


def get_stim_coords(
    x_pos_stim, y_pos_choices, y_pos_sequence, screen_resolution, img_size
):
    # # * Create the figure
    # fig, ax = plt.subplots(figsize=(12, 8))

    # # * Set the limits of the plot to match the screen resolution
    # ax.set_xlim(0, screen_resolution[0])
    # ax.set_ylim(0, screen_resolution[1])
    # ax.invert_yaxis()

    targets = {
        f"seq{i}": center_x for i, center_x in enumerate(x_pos_stim["items_set"])
    }
    targets.update(
        {
            f"choice{i}": center_x
            for i, center_x in enumerate(x_pos_stim["avail_choice"])
        }
    )

    for target_name, x in targets.items():
        x_ = x + screen_resolution[0] / 2

        x_left = x_ - img_size[0] / 2
        x_right = x_ + img_size[0] / 2

        y_ = y_pos_sequence if target_name.startswith("seq") else y_pos_choices
        y_ = -y_ + screen_resolution[1] / 2

        y_top = y_ - img_size[1] / 2
        y_bottom = y_ + img_size[1] / 2

        targets[target_name] = [[x_left, x_right], [y_top, y_bottom]]

        # rect = Rectangle(
        #     xy=(x_left, y_top),
        #     width=img_size[0],
        #     height=img_size[1],
        #     fill=False,
        #     edgecolor="blue",
        #     linewidth=2,
        # )
        # ax.add_patch(rect)

        # ax.text(
        #         x_,
        #         y_,
        #         target_name,
        #         ha="center",
        #         va="center",
        #         fontweight="bold",
        #         # color=color_map[target_name],
        #         alpha=0.5,
        #     )

        # * Can use the following to get the position of the rectangle
        # rect.get_xy()
        # rect.get_width()
        # rect.get_height()
        # ax.scatter(x_, y_, c="blue")
        # ax.scatter(x_left, y_, c='blue')
        # ax.scatter(x_right, y_, c="blue")
        # ax.scatter(x_, y_top, c="blue")
        # ax.scatter(x_, y_bottom, c="blue")
    return targets


def get_trial_info(epoch_N, raw_behav):
    # TODO: Finish this function
    trial_behav = raw_behav.iloc[epoch_N]

    trial_seq = {i: trial_behav[f"figure{i+1}"] for i in range(8)}
    trial_seq[trial_behav["masked_idx"]] = "question-mark"

    trial_solution = trial_behav["solution"]
    trial_choices = {i: trial_behav[f"choice{i+1}"] for i in range(4)}
    trial_response = trial_behav["choice"]
    rt = trial_behav["rt"]

    if trial_response in ["timeout", "invalid"]:
        choice_ind = None
    else:
        choice_ind = [k for k, v in trial_choices.items() if v == trial_response][0]

    # correct = trial_response == trial_solution
    trial_seq_order = [int(i) for i in str(trial_behav["seq_order"])]
    trial_choice_order = [int(i) for i in str(trial_behav["choice_order"])]

    icons_order = trial_seq_order + [i + 8 for i in trial_choice_order]

    icons_coords = get_stim_coords(
        x_pos_stim, y_pos_choices, y_pos_sequence, screen_resolution, img_size
    )

    icons_coords = [i[0] + i[1] for i in icons_coords.values()]

    trial_icons = list(trial_seq.values()) + list(trial_choices.values())

    icons_coords = [[trial_icons[i], icons_coords[i]] for i in range(len(trial_icons))]

    return (
        icons_coords,
        icons_order,
        trial_seq,
        trial_choices,
        choice_ind,
        trial_solution,
        rt,
    )


def get_memory_usage():
    import types

    from pympler import asizeof

    # List of variable names to exclude, particularly Jupyter artifacts
    exclude_vars = {
        "quit",
        "exit",
        "Out",
        "_oh",
        "_dh",
        "_",
        "__",
        "___",
        "get_ipython",
        "logger",
        "globals_snapshot",
    }

    # Snapshot of globals to prevent modification during iteration
    globals_snapshot = {
        name: value for name, value in globals().items() if name not in exclude_vars
    }

    # Filter globals for only objects that pympler can measure
    variables = {}
    for name, value in tqdm(globals_snapshot.items()):
        # Skip modules, functions, and classes to avoid issues
        if isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        # Attempt to get the memory size and skip if it fails
        try:
            variables[name] = asizeof.asizeof(value)
        except (ValueError, TypeError):
            pass  # Ignore variables that raise errors

    df = pd.DataFrame(
        [(name, size / 125000) for name, size in variables.items()],
        columns=["Variable", "Usage (Megabits)"],
    ).sort_values(by="Usage (Megabits)", ascending=False)

    df["Usage (Megabits)"] = df["Usage (Megabits)"].round(2)

    # Display the DataFrame
    df.reset_index(drop=True, inplace=True)

    return df


def clear_jupyter_artifacts():
    # List of known Jupyter and system variables to avoid deleting
    protected_vars = {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__file__",
        "__cached__",
        "__annotations__",
        "__IPYTHON__",
        "__IPYTHON__.config",
    }

    # Loop through all variables in the global namespace
    for name in list(globals().keys()):
        # Delete only numbered underscore variables that are not in the protected list
        if name.startswith("_") and name[1:].isdigit() and name not in protected_vars:
            del globals()[name]


def console_log(msg, level=None):
    console.print(msg)
    dir(console)

    # console.print("This is information", style="info")
    # console.print("[warning]WARNING[/warning]:The pod bay doors are locked")
    # console.print("Something terrible happened!", style="danger")


def check_notes():
    sess_info_files = sorted(data_dir.rglob("*sess_info.json"))
    notes = {}
    for f in sess_info_files:
        subj_N, sess_N = [int(d) for d in f.name.split("-")[:2]]

        with open(f, "r") as file:
            sess_info = json.load(file)

        if len(sess_info["Notes"]) > 0:
            notes[f"subj_{subj_N:02}-sess_{sess_N:02}"] = sess_info["Notes"]
    pprint(notes)


def check_et_calibrations():
    et_files = sorted(data_dir.rglob("*.asc"))
    et_cals = {}
    for f in et_files:
        subj_N = int(f.parents[1].name.split("_")[1])
        sess_N = int(f.parents[0].name.split("_")[1])

        cals = read_eyelink_calibration(f)

        cals = [dict(cal) for cal in cals]
        # pprint([i for i in dir(cal[0]) if not i.startswith("_")])
        # pprint(list(cal[0].keys()))

        et_cals[f"subj_{subj_N:02}-sess_{sess_N:02}"] = cals

    for subj_sess, cals in et_cals.items():
        subj, sess = subj_sess.split("-")
        if len(cals) == 0:
            msg = f"[warning] WARNING [/warning] No calibration found for {subj}-{sess}"
            console_log(msg)

    # pprint(et_cals)


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
    behav_fpath = [f for f in sess_dir.glob("*behav*.csv")][0]
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
        first_cal = et_cals[
            0
        ]  # let's access the first (and only in this case) calibration
        print(f"number of calibrations: {len(et_cals)}")
        print(first_cal)
    except:
        print("WARNING: NO CALIBRATION FOUND")

    # raw_et.plot(duration=0.5, scalings=dict(eyegaze=1e3))
    # raw_et.annotations

    print(f"{raw_et.ch_names = }")
    chan_xpos, chan_ypos, chan_pupil = raw_et.ch_names
    x_pos = raw_et[chan_xpos][0][0]
    y_pos = raw_et[chan_ypos][0][0]

    # * Read events from annotations
    et_events, et_events_dict = mne.events_from_annotations(raw_et)

    # * Convert keys to strings (if they aren't already)
    et_events_dict = {str(k): v for k, v in et_events_dict.items()}

    if et_events_dict.get("exp_start"):
        et_events_dict["experiment_start"] = et_events_dict.pop("exp_start")

    print("Unique event IDs before update:", np.unique(et_events[:, 2]))

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

    print("Unique event IDs after update:", np.unique(et_events[:, 2]))

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

    print("Eye tracking event counts:")
    display(et_events_df["event_id"].value_counts())

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
    print(f"Number of epochs created: {len(manual_et_epochs)}")
    for i, epoch in enumerate(manual_et_epochs):
        print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

    return (
        et_events_dict,
        et_events_dict_inv,
        et_trial_bounds,
        et_trial_events_df,
        manual_et_epochs,
    )


def locate_trials(events, valid_events):
    # valid_events_inv = {v: k for k, v in valid_events.items()}

    # * Find trial start and end events
    trial_start_inds = np.where(events[:, 2] == valid_events["trial_start"])[0]
    events_df = pd.DataFrame(events[:, [0, 2]], columns=["sample_nb", "event_id"])
    events_df["event_id"] = events_df["event_id"].replace(valid_events_inv)

    trial_bounds = trial_start_inds.tolist()
    trial_bounds = trial_bounds + [events.shape[0] - 1]
    trial_bounds = list(zip(trial_bounds[:-1], trial_bounds[1:]))

    trial_end_inds = np.array([], dtype=int)
    trial_n = np.zeros(len(events_df), dtype=int) - 1

    for i, bound in enumerate(trial_bounds):
        temp = events_df.iloc[bound[0] : bound[1]]
        # if "trial_aborted" in temp["event_id"].values:
        #     trial_bounds.pop(i)
        # elif "trial_end" in temp["event_id"].values:
        if "trial_end" in temp["event_id"].values:
            trial_end_ind = temp[temp["event_id"] == "trial_end"].index[-1]
            trial_end_inds = np.append(trial_end_inds, trial_end_ind)
            trial_n[bound[0] : trial_end_ind + 1] = i

    events_df["trial_id"] = trial_n

    trial_bounds = np.array([i for i in zip(trial_start_inds, trial_end_inds)])

    assert all(events_df.loc[trial_bounds[:, 0]]["event_id"] == "trial_start")
    assert all(events_df.loc[trial_bounds[:, 1]]["event_id"] == "trial_end")

    events_df.query("trial_id != -1", inplace=True)

    return trial_bounds, events_df


def export_trials(events_df, save=False):
    # * Function to highlight rows
    def highlight_trials(row):
        if "trial_start" in str(row["event_id"]):
            return ["background-color: green" for col in row]
        elif "trial_end" in str(row["event_id"]):
            return ["background-color: red" for col in row]
        else:
            return ["" for col in row]

    # * Apply the function to the DataFrame
    styled_df = events_df.style.apply(highlight_trials, axis=1)
    if save:
        styled_df.to_excel("trials.xlsx", engine="openpyxl", index=False)
    return styled_df


def get_gaze_heatmap(
    x_gaze, y_gaze, screen_res, bin_size=50, show=False, normalize=True
):
    """
    Generate a heatmap from gaze data.
    Parameters:
    x_gaze (array-like): Array of x-coordinates of gaze points.
    y_gaze (array-like): Array of y-coordinates of gaze points.
    screen_res (tuple): Screen resolution as (width, height).
    bin_size (int, optional): Size of the bins for the histogram. Default is 50.
    show (bool, optional): If True, display the heatmap. Default is False.
    normalize (bool, optional): If True, normalize the heatmap. Default is True.
    Returns:
    tuple: A tuple containing:
        - heatmap (2D array): The generated heatmap.
        - xedges (array): The bin edges along the x-axis.
        - yedges (array): The bin edges along the y-axis.
    """
    # TODO: Add possibility to show the heatmap over an existing plot

    screen_width, screen_height = screen_resolution

    valid_mask = ~np.isnan(x_gaze) & ~np.isnan(y_gaze)

    heatmap_gaze_x = x_gaze[valid_mask]
    heatmap_gaze_y = screen_height - y_gaze[valid_mask]

    # * Generate the heatmap using 2D histogram
    num_bins_x = screen_width // bin_size
    num_bins_y = screen_height // bin_size

    heatmap, xedges, yedges = np.histogram2d(
        heatmap_gaze_x,
        heatmap_gaze_y,
        bins=[num_bins_x, num_bins_y],
        range=[[0, screen_width], [0, screen_height]],
    )

    # * Optionally normalize the heatmap
    if normalize == True:
        heatmap = heatmap / np.max(heatmap)

    # * Plot the heatmap
    if show == True:
        plt.figure(figsize=(12, 8))
        plt.imshow(
            heatmap.T,
            extent=[0, screen_width, 0, screen_height],
            origin="lower",
            cmap="hot",
            aspect="auto",
        )
        plt.colorbar(label="Normalized Gaze Density")
        plt.xlabel("X Position (pixels)")
        plt.ylabel("Y Position (pixels)")
        plt.title("Eye-Tracking Heatmap")
        plt.show()
        plt.close("all")

    return heatmap, xedges, yedges


# * ####################################################################################
# * EEG ANALYSIS
# * ####################################################################################


def determine_bad_channels(raw_eeg):
    pass
    # TODO: Implement this function


def check_ch_groups(
    montage: mne.channels.DigMontage,
    ch_names: List[str],
    ch_groups: Dict[str, list[str]],
):
    montage_chans = set(montage.ch_names)
    groupped_chans = set([ch for ch_group in ch_groups.values() for ch in ch_group])

    orphan_chans = montage_chans - groupped_chans

    if orphan_chans:
        print("WARNING: orphan channels found in ch_groups:")
        print(orphan_chans)


def show_ch_groups(montage, ch_groups: Dict):
    """
    Display channel groups on a montage.
    Parameters:
    montage : mne.channels.DigMontage
        The montage object containing the electrode positions.
    ch_groups : dict
        A dictionary where keys are group names and values are lists of channel names to be displayed.
    Returns:
    None
    """

    for group_name, chans in ch_groups.items():
        fix, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(group_name)
        montage.plot(show_names=chans, axes=ax, show=False)
        plt.show()


def custom_plot_montage(
    montage: mne.channels.DigMontage,
    ch_groups: Dict[str, List[str]],
    group_colors: Dict[str, str],
    show_names: bool = True,
    show_legend: bool = True,
    show: bool = True,
):
    # ! TEMP
    # ch_groups = ch_groups1.copy()
    # group_colors = ch_groups1_colors.copy()
    # ! TEMP

    # * Ensure channel names in ch_groups are non-overlapping
    all_chans = [ch for group in ch_groups.values() for ch in group]

    if len(all_chans) != len(set(all_chans)):
        raise ValueError(
            "Channel names in ch_groups must be non-overlapping in between groups."
        )

    chans_info = {ch: k for k, v in ch_groups.items() for ch in v}
    chans_info = {ch: chans_info.get(ch, "unassigned") for ch in montage.ch_names}
    ch_colors = {
        ch: group_colors.get(chans_info[ch], "black") for ch in montage.ch_names
    }
    ch_colors = list(ch_colors.values())

    # * Plot initially to capture the ax.lines and ax.texts (ch names)
    fig = montage.plot(show_names=True, show=False)
    ax = fig.get_axes()[0]
    plt.close()
    [group for group, chans in ch_groups.items() if "Cz" in chans]

    # * Get the lines (head, nose, ears) and positions from the initial plot
    head_lines = [line.get_xydata() for line in ax.lines]
    chans_plot_pos = ax.collections[0].get_offsets().data  # .T
    # chans_plot_pos_x, chans_plot_pos_y = chans_plot_pos
    chan_names = [i for i in ax.texts if i.get_text() in montage.ch_names]

    # * Now, replicate the plot, including the head, nose, and ears from `ax.lines`
    fig, ax = plt.subplots(figsize=(10, 10))

    # * Plot the head, nose, and ears using the lines stored in `ax.lines`
    for line_data in head_lines:
        ax.plot(line_data[:, 0], line_data[:, 1], color="black", lw=2)

    # * Plot the channel points
    ch_color_inds = pd.Series(ch_colors).groupby(ch_colors).groups

    for color, inds in ch_color_inds.items():
        # for i, color in enumerate(ch_colors):
        ax.scatter(
            chans_plot_pos[inds, 0],
            chans_plot_pos[inds, 1],
            color=color,
            s=100,
            zorder=2,
            label={v: k for k, v in group_colors.items()}.get(color, "unassigned"),
        )

    # * Plot the channel names
    if show_names:
        for i, ch_name in enumerate(chan_names):
            ax.text(
                ch_name.get_position()[0],
                ch_name.get_position()[1],
                ch_name.get_text(),
                fontsize=9,
                ha="left",
                va="center",
            )

    # * Set the aspect ratio, remove axes, and show the plot
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    if show:
        plt.show()
    plt.close()

    chans_info = [
        [i, ch_name, ch_group, ch_colors[i]]
        for i, (ch_name, ch_group) in enumerate(chans_info.items())
    ]

    return fig, chans_info


def custom_plot_montage_plotly(
    montage,  #: DigMontage,
    ch_groups: Dict[str, List[str]],
    group_colors: Dict[str, str],
    show_names: bool = True,
    show_legend: bool = True,
    show: bool = True,
):
    from typing import Dict, List

    import numpy as np
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.offline as pyo
    from mne.channels import DigMontage

    # * Ensure channel names in ch_groups are non-overlapping
    all_chans = [ch for group in ch_groups.values() for ch in group]

    if len(all_chans) != len(set(all_chans)):
        raise ValueError(
            "Channel names in ch_groups must be non-overlapping between groups."
        )

    # * Map each channel to its group
    chans_info = {ch: k for k, v in ch_groups.items() for ch in v}
    chans_info = {ch: chans_info.get(ch, "unassigned") for ch in montage.ch_names}

    # * Map each channel to its color
    ch_colors = {
        ch: group_colors.get(chans_info[ch], "black") for ch in montage.ch_names
    }

    # * Get the 3D positions of the channels
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]

    # * Ensure the positions are in the same order as montage.ch_names
    pos_3d = np.array([ch_pos[ch] for ch in montage.ch_names])

    # * Convert 3D positions to 2D using azimuthal projection
    def _cart_to_sph(coords):
        x, y, z = coords.T
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.column_stack((azimuth, elevation, r))

    pos_sph = _cart_to_sph(pos_3d)
    xy = pos_sph[:, :2]  # Use azimuth and elevation as x and y

    # * Prepare the head outline (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    head_x = 0.5 * np.cos(theta)
    head_y = 0.5 * np.sin(theta)

    # * Define nose and ears
    nose = np.array([[0.0, 0.5], [-0.1, 0.6], [0.1, 0.6], [0.0, 0.5]])
    left_ear = np.array([[-0.5, 0.0], [-0.6, 0.1], [-0.6, -0.1], [-0.5, 0.0]])
    right_ear = np.array([[0.5, 0.0], [0.6, 0.1], [0.6, -0.1], [0.5, 0.0]])

    # * Create Plotly traces
    traces = []

    # * Head outline
    traces.append(
        go.Scatter(
            x=head_x,
            y=head_y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    # * Nose and ears
    traces.append(
        go.Scatter(
            x=nose[:, 0],
            y=nose[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )
    traces.append(
        go.Scatter(
            x=left_ear[:, 0],
            y=left_ear[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )
    traces.append(
        go.Scatter(
            x=right_ear[:, 0],
            y=right_ear[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    # * Group channels by color
    ch_color_series = pd.Series(ch_colors)
    ch_color_groups = ch_color_series.groupby(ch_colors).groups

    # * Plot channel points
    for color, inds in ch_color_groups.items():
        group_name = {v: k for k, v in group_colors.items()}.get(color, "unassigned")
        traces.append(
            go.Scatter(
                x=xy[inds, 0],
                y=xy[inds, 1],
                mode="markers",
                marker=dict(size=10, color=color),
                name=group_name if show_legend else "",
                showlegend=show_legend,
            )
        )

    # * Plot channel names
    if show_names:
        for i, ch_name in enumerate(montage.ch_names):
            traces.append(
                go.Scatter(
                    x=[xy[i, 0]],
                    y=[xy[i, 1]],
                    mode="text",
                    text=[ch_name],
                    textposition="top center",
                    showlegend=False,
                )
            )

    # * Create layout
    layout = go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.7, 0.7]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.7, 0.7]),
        width=600,
        height=600,
        showlegend=show_legend,
        legend=dict(x=1, y=1),
        margin=dict(t=20, b=20, l=20, r=20),
    )

    fig = go.Figure(data=traces, layout=layout)

    if show:
        fig.show()

    chans_info = [
        [i, ch_name, chans_info[ch_name], ch_colors[ch_name]]
        for i, ch_name in enumerate(montage.ch_names)
    ]

    return fig, chans_info


def preprocess_eeg_data(raw_eeg, eeg_chan_groups, raw_behav):
    fpath = Path(raw_eeg.filenames[0])
    subj_N = int(fpath.parents[1].name.split("_")[1])
    sess_dir = fpath.parents[0]
    sess_N = int(sess_dir.name.split("_")[1])

    preprocessed_dir = wd / f"results/preprocessed_data/"
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
        check_ch_groups(raw_eeg.get_montage(), raw_eeg.ch_names, eeg_chan_groups)

        # ! Demonstration purposes
        # # * Show channel groups
        # show_ch_groups(raw_eeg.get_montage(), eeg_chan_groups)

        # ch_groups1 = {
        #     g: chans
        #     for g, chans in eeg_chan_groups.items()
        #     if g
        #     in ["frontal", "parietal", "occipital", "temporal", "central", "unassigned"]
        # }
        # ch_groups1_colors = dict(
        #     zip(ch_groups1.keys(), ["red", "green", "blue", "purple", "orange"])
        # )
        # ch_groups2 = {
        #     g: chans
        #     for g, chans in eeg_chan_groups.items()
        #     if g in ["frontal", "occipital"]
        # }

        # ch_groups2_colors = dict(
        #     zip(
        #         ch_groups2.keys(),
        #         [
        #             "red",
        #             "green",
        #         ],
        #     )
        # )

        # ch_groups1_montage, _ = custom_plot_montage(
        #     montage, ch_groups1, ch_groups1_colors, show_names=False
        # )

        # ch_groups2_montage, _ = custom_plot_montage(
        #     montage,
        #     ch_groups2,
        #     ch_groups2_colors,
        #     show_names=True,
        # )

        # * Average Reference
        raw_eeg.set_eeg_reference(ref_channels="average")

        # * drop bad channels
        # bad_chans = mne.preprocessing.find_bad_channels_lof(raw_eeg)
        # raw_eeg.plot()

        # raw_eeg.info["bads"] = bad_chans
        # raw_eeg.drop_channels(bad_chans)
        # logger.info(f"subj_{subj_N} - sess_{sess_N} - Bad channels: {bad_chans}")

        # * EOG artifact rejection using ICA *
        ica_fpath = preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_fitted-ica.fif"

        if ica_fpath.is_file():
            ica = mne.preprocessing.read_ica(ica_fpath)

        else:
            raw_eeg.filter(l_freq=1, verbose=False)

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
        # raw_eeg.set_annotations(annotations)

        raw_eeg.save(preprocessed_raw_fpath, overwrite=True)

        del raw_eeg

        raw_eeg = mne.io.read_raw_fif(preprocessed_raw_fpath, preload=False)

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

    choice_key_eeg = [
        valid_events_inv[i]
        for i in eeg_events[:, 2]
        if i in [10, 11, 12, 13, 14, 15, 16]
    ]

    eeg_trial_bounds, eeg_events_df = locate_trials(eeg_events, valid_events)

    # * Remove practice trials
    if sess_N == 1:
        choice_key_eeg = choice_key_eeg[3:]
        eeg_trial_bounds = eeg_trial_bounds[3:]

    raw_behav["choice_key_eeg"] = choice_key_eeg
    raw_behav["same"] = raw_behav["choice_key"] == raw_behav["choice_key_eeg"]

    manual_eeg_trials = []

    # * Loop through each trial
    for start, end in tqdm(eeg_trial_bounds):
        # * Get start and end times in seconds
        start_time = (eeg_events[start, 0] / raw_eeg.info["sfreq"]) - pre_trial_time
        end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + post_trial_time

        # * Crop the raw data to this time window
        epoch_data = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)

        # * Add this epoch to our list
        manual_eeg_trials.append(epoch_data)

    # * Print some information about our epochs
    print(f"Number of epochs created: {len(manual_eeg_trials)}")
    for i, epoch in enumerate(manual_eeg_trials):
        print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

    return raw_eeg, manual_eeg_trials, eeg_trial_bounds, eeg_events, eeg_events_df


# *  ERPs
def erp_analysis(raw_eeg, eeg_events, valid_events):
    # * ## SEQUENCE FLASHES ##
    epochs_stim_flash_seq = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=valid_events["stim-flash_sequence"],
        baseline=(-0.1, 0),
    )
    evoked_stim_flash_seq = epochs_stim_flash_seq.average()

    evoked_stim_flash_seq.plot()
    plt.show()

    # * ## CHOICE FLASHES ##
    epochs_stim_flash_choices = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=valid_events["stim-flash_choices"],
        baseline=(-0.1, 0),
    )
    evoked_stim_flash_choices = epochs_stim_flash_choices.average()

    evoked_stim_flash_choices.plot()
    plt.show()

    # * ## SEQUENCE + CHOICE FLASHES ##
    epochs_stim_flash_all = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=[
            valid_events[i] for i in ["stim-flash_choices", "stim-flash_choices"]
        ],
        baseline=(-0.1, 0),
    )

    evoked_stim_flash_all = epochs_stim_flash_all.average()

    evoked_stim_flash_all.plot()
    plt.show()

    # * ## ALL STIM PRESENTAION ##
    epochs_stim_flash_all = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=1,
        event_id=valid_events["stim-all_stim"],
        baseline=(-0.1, 0),
    )

    evoked_stim_flash_all = epochs_stim_flash_all.average()

    evoked_stim_flash_all.plot()
    plt.show()


# * ## ALL STIM PRESENTATION PER PATTERN TYPE ##
# * ####################################################################################
# * PLOTTING EEG AND EYE-TRACKING DATA
# * ####################################################################################


def resample_eye_tracking_data(et_epoch, tracked_eye, et_sfreq_original, eeg_sfreq):
    x_gaze = et_epoch[f"xpos_{tracked_eye}"][0][0]
    y_gaze = et_epoch[f"ypos_{tracked_eye}"][0][0]
    x_gaze_resampled, y_gaze_resampled = resample_and_handle_nans(
        x_gaze, y_gaze, et_sfreq_original, eeg_sfreq
    )
    return x_gaze_resampled, y_gaze_resampled


def resample_and_handle_nans(x_gaze, y_gaze, et_sfreq_original, eeg_sfreq):
    et_time = np.arange(len(x_gaze)) / et_sfreq_original
    new_et_time = np.arange(et_time[0], et_time[-1], 1 / eeg_sfreq)
    x_gaze_resampled = np.full_like(new_et_time, np.nan)
    y_gaze_resampled = np.full_like(new_et_time, np.nan)
    valid_mask = ~np.isnan(x_gaze)
    valid_indices = np.where(valid_mask)[0]

    from itertools import groupby
    from operator import itemgetter

    def get_valid_segments(valid_indices):
        segments = []
        for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            segments.append((group[0], group[-1]))
        return segments

    valid_segments = get_valid_segments(valid_indices)

    for start_idx, end_idx in valid_segments:
        segment_time = et_time[start_idx : end_idx + 1]
        x_segment = x_gaze[start_idx : end_idx + 1]
        y_segment = y_gaze[start_idx : end_idx + 1]
        x_interp = interp1d(
            segment_time,
            x_segment,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        y_interp = interp1d(
            segment_time,
            y_segment,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        segment_new_time_mask = (new_et_time >= segment_time[0]) & (
            new_et_time <= segment_time[-1]
        )
        segment_new_time = new_et_time[segment_new_time_mask]
        x_gaze_resampled[segment_new_time_mask] = x_interp(segment_new_time)
        y_gaze_resampled[segment_new_time_mask] = y_interp(segment_new_time)

    return x_gaze_resampled, y_gaze_resampled


def create_base_figure(screen_resolution, targets, ax_et):
    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticklabels([])
    ax_et.set_yticklabels([])
    ax_et.set_xticks([])
    ax_et.set_yticks([])
    ax_et.set_aspect("equal", adjustable="box")
    for target in targets:
        ax_et.plot(target[0], target[1], "ko")


def create_video_from_frames(eeg_frames_dir, output_file, fps, zfill_len):
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        str(eeg_frames_dir / f"frame_%0{zfill_len}d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(eeg_frames_dir / output_file),
    ]
    print("Creating video with FFmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video created: {output_file}")


# * ####################################################################################
# * GENERATE EEG + EYE TRACKING + STIMULI VIDEO
# * ####################################################################################


def generate_trial_video(manual_et_epochs, manual_eeg_epochs, raw_behav, epoch_N: int):
    # * Extract epoch data
    et_epoch = manual_et_epochs[epoch_N]
    eeg_epoch = manual_eeg_epochs[epoch_N]

    fname = Path(eeg_epoch.filenames[0]).stem
    identifier = re.findall(r"subj_(\d+)", fname)[0]
    subj_N = int(identifier[:2])
    sess_N = int(identifier[2:])

    tracked_eye = et_epoch.ch_names[0].split("_")[1]
    # et_sfreq = et_epoch.info["sfreq"]
    # eeg_sfreq = eeg_epoch.info["sfreq"]
    assert (
        et_sfreq == et_epoch.info["sfreq"]
    ), "Eye-tracking data has incorrect sampling rate"
    assert eeg_sfreq == eeg_epoch.info["sfreq"], "EEG data has incorrect sampling rate"

    sess_bad_chans = all_bad_chans.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
    montage = eeg_epoch.get_montage()

    selected_chans = [
        i
        for i, ch in enumerate(montage.ch_names)
        if ch not in non_eeg_chans + sess_bad_chans
    ]
    selected_chans_names = [montage.ch_names[i] for i in selected_chans]

    chans_pos_xy = np.array(
        [
            v
            for k, v in montage.get_positions()["ch_pos"].items()
            if k in selected_chans_names
        ]
    )[:, :2]

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
        zip(selected_chan_groups.keys(), ["red", "green", "blue", "purple", "orange"])
    )

    # * Get channel indices for each channel group
    ch_group_inds = {
        group_name: [
            i for i, ch in enumerate(selected_chans_names) if ch in group_chans
        ]
        for group_name, group_chans in selected_chan_groups.items()
    }

    # * Get channel positions for topomap
    eeg_info = eeg_epoch.info

    eeg_info = mne.pick_info(
        eeg_info,
        [
            i
            for i, ch in enumerate(eeg_info.ch_names)
            if ch not in non_eeg_chans + eeg_info["bads"]
        ],
    )

    chans_pos_xy = np.array(
        list(eeg_info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    trial_info = get_trial_info(epoch_N, raw_behav)
    stim_pos, stim_order = trial_info[:2]

    # * Resample eye-tracking data for the current trial
    x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
        et_epoch, tracked_eye, et_sfreq, eeg_sfreq
    )

    # * Synchronize data lengths
    eeg_data = eeg_epoch.get_data(picks=selected_chans_names)

    epoch_evts = pd.Series(eeg_epoch.get_data(picks=[stim_chan])[0])

    # * Find indices where consecutive events are different
    diff_indices = np.where(epoch_evts.diff() != 0)[0]
    epoch_evts = epoch_evts[diff_indices]
    epoch_evts = epoch_evts[epoch_evts != 0]
    epoch_evts = epoch_evts.replace(valid_events_inv)

    # * Drop EOG & Status channels
    # eeg_data = eeg_data[:-5, :]

    # * Ensure that the data arrays have the same length
    min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
    eeg_data = eeg_data[:, :min_length]
    eeg_data *= 1e6  # Convert to microvolts
    # avg_eeg_data = eeg_data.mean(axis=0)

    x_gaze_resampled = x_gaze_resampled[:min_length]
    y_gaze_resampled = y_gaze_resampled[:min_length]

    # * y-axis limits for the EEG plots
    eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
    # avg_eeg_min, avg_eeg_max = avg_eeg_data.min(), avg_eeg_data.max()

    y_eeg_min, y_eeg_max = eeg_min * 1.1, eeg_max * 1.1
    # y_eeg_avg_min, y_eeg_avg_max = avg_eeg_min * 1.1, avg_eeg_max * 1.1

    # * Heatmap of gaze data
    all_stim_onset = epoch_evts[epoch_evts == "stim-all_stim"].index[0]
    trial_end = epoch_evts[epoch_evts == "trial_end"].index[0]

    heatmap, _, _ = get_gaze_heatmap(
        x_gaze_resampled[all_stim_onset:trial_end],
        y_gaze_resampled[all_stim_onset:trial_end],
        screen_resolution,
        bin_size=20,
        show=True,
    )

    # * Now you have resampled eye-tracking data that matches the EEG data in sampling rate and length
    # * Proceed with your analysis or visualization
    # * For example, you can plot the data or save it for later use
    samples_per_1ms = eeg_sfreq / 1000
    samples_per_100ms = round(samples_per_1ms * 100)

    # sample_window_ms = 50
    # samples_per_window = round(samples_per_1ms * sample_window_ms)

    # TODO: comment
    leftover = eeg_data.shape[1] % samples_per_100ms
    # n_splits = eeg_data.shape[1] // samples_per_100ms

    inds = np.arange(0, eeg_data.shape[1], samples_per_100ms)

    if leftover > 0:
        inds = np.append(inds, inds[-1] + leftover)

    # TODO: comment
    step_size = samples_per_100ms
    steps = np.diff(inds)
    inds = np.array(list(zip(inds[:-1], inds[1:])))

    zfill_len = len(str(steps.shape[0]))

    # * Set up the directory to save the frames
    eeg_frames_dir = Path.cwd() / f"eeg_frames-{subj_N}-{sess_N:02d}-ep{epoch_N:02d}"

    if eeg_frames_dir.exists():
        shutil.rmtree(eeg_frames_dir)
    eeg_frames_dir.mkdir()

    # * Adjust figure size and subplot layout
    fig = plt.figure(figsize=(25.6, 14.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 3], width_ratios=[3, 2])

    ax_eeg1 = fig.add_subplot(gs[0, :])  # * EEG traces plot (spans both columns)
    ax_eeg2 = fig.add_subplot(gs[1, :], sharex=ax_eeg1)  # * Avg EEG plot (both cols)
    ax_topo = fig.add_subplot(gs[2, 0])  # * Topomap (left side of bottom row)
    ax_et = fig.add_subplot(gs[2, 1])  # * Eye tracking plot (right side of bottom row)

    # win_len_seconds = 2
    # win_len_samples = int(win_len_seconds * eeg_sfreq)
    win_len_samples = steps.cumsum()[20]

    # samples_ticks = np.arange(0, win_len_samples + 101, samples_per_100ms)
    # time_ticks = [int(i / eeg_sfreq * 1000) for i in samples_ticks]
    # TODO: comment

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, eeg_data.shape[0]))

    x_ticks = np.arange(0, win_len_samples + 1, step_size)
    # x_ticks_labels = [None] * len(x_ticks)
    x_ticks_labels = [str(int(i / eeg_sfreq * 1000)) for i in x_ticks]

    ax_eeg1_line = ax_eeg1.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)
    ax_eeg2_line = ax_eeg2.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)

    # TODO: comment
    et_scatter = None
    et_line = None
    last_eeg_x = 0
    last_plot_x = 0
    flash_N = 0

    ax_et_fix_cross = ax_et.scatter(
        screen_resolution[0] / 2,
        screen_resolution[1] / 2,
        s=80,
        marker="+",
        linewidths=1,
        color="black",
    )

    ax_et_plotted_icons = []
    for icon_name, icon_pos in stim_pos:
        left, right, bottom, top = icon_pos

        this_icon = ax_et.imshow(
            icon_images[icon_name],
            extent=[left, right, bottom, top],
            origin="lower",
        )

        ax_et_plotted_icons.append(this_icon)
        this_icon.set_visible(False)

    topo_plot_params = dict(
        ch_type="eeg",
        sensors=True,
        names=None,
        mask=None,
        mask_params=None,
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

    mne.viz.plot_topomap(
        data=np.zeros_like(eeg_data[:, 0]), pos=chans_pos_xy, **topo_plot_params
    )
    ax_topo.set_axis_off()

    eeg_group_data = {
        group: eeg_data[ch_group_inds[group]].mean(axis=0) for group in ch_group_inds
    }
    # min_eeg_by_group = [eeg_data[group_inds].mean().min() for group_inds in ch_group_inds.values()]
    # max_eeg_by_group = [
    #     eeg_data[group_inds].mean().max() for group_inds in ch_group_inds.values()
    # ]
    min_eeg_by_group = [group_data.min() for group_data in eeg_group_data.values()]
    max_eeg_by_group = [group_data.max() for group_data in eeg_group_data.values()]

    def reset_eeg_plot():
        ax_eeg1.clear()

        ax_eeg1.set_xticks(x_ticks)
        ax_eeg1.set_xticklabels([])

        ax_eeg1.set_ylim(y_eeg_min, y_eeg_max)
        ax_eeg1.set_xlim(0, win_len_samples)

        ax_eeg2.clear()
        ax_eeg2.set_xticks(x_ticks, x_ticks_labels)
        ax_eeg2.set_ylim(min(min_eeg_by_group), max(max_eeg_by_group))
        ax_eeg2.set_xlim(0, win_len_samples)
        ax_eeg2.hlines(0, 0, win_len_samples, color="black", linestyle="--")

    def reset_et_plot(show_icons: Union[List, bool] = False, show_fix_cross=False):
        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)

        ax_et.set_xticklabels([])
        ax_et.set_yticklabels([])

        ax_et.set_xticks([])
        ax_et.set_yticks([])

        ax_et.set_aspect("equal", adjustable="box")

        if isinstance(show_icons, bool):
            [
                ax_et_plotted_icons[i].set_visible(show_icons)
                for i in range(len(stim_order))
            ]

        elif isinstance(show_icons, list):
            [ax_et_plotted_icons[i].set_visible(False) for i in range(len(stim_order))]
            [ax_et_plotted_icons[i].set_visible(True) for i in show_icons]

        ax_et_fix_cross.set_visible(show_fix_cross)

    dpi = 150
    reset_eeg_plot()
    reset_et_plot(show_icons=False, show_fix_cross=True)

    for idx_step, step in enumerate(tqdm(steps, desc="Generating frames")):
        ax_eeg1_line.remove()
        ax_eeg2_line.remove()

        bounds = (last_eeg_x, last_eeg_x + step)

        detected_event_inds = [i for i in epoch_evts.index if i in range(*bounds)]

        if detected_event_inds:
            if ax_et_fix_cross.get_visible():
                ax_et_fix_cross.set_visible(False)

            # * Get name, detected events and their descriptions
            detected_events = epoch_evts[detected_event_inds]
            event_desc = detected_events.values
            # event_inds = detected_events.index
            event_inds = detected_events.index - bounds[0] + last_plot_x

            # * Vertical lines to mark events
            ax_eeg1.vlines(
                event_inds, ymin=y_eeg_min, ymax=y_eeg_max, color="red", linestyle="--"
            )
            ax_eeg2.vlines(
                event_inds, ymin=y_eeg_min, ymax=y_eeg_max, color="red", linestyle="--"
            )

            for i, (ind, desc) in enumerate(zip(event_inds, event_desc)):
                ax_eeg1.text(
                    ind,
                    y_eeg_max,
                    desc,
                    rotation=45,
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontsize=8,
                    color="red",
                )

                if "stim-flash" in desc:
                    icon_ind = stim_order[flash_N]
                    reset_et_plot(show_icons=[icon_ind], show_fix_cross=False)
                    # ax_et_plotted_icons[icon_ind].set_visible(True)
                    flash_N += 1

                elif desc == "stim-all_stim":
                    reset_et_plot(show_icons=True, show_fix_cross=False)

                elif desc in ["a", "x", "m", "l", "timeout", "trial_end"]:
                    reset_et_plot(show_icons=False, show_fix_cross=True)

        # * Get EEG data slice
        eeg_slice = eeg_data[:, bounds[0] : bounds[1]]

        # * Get gaze data slice
        x_gaze_slice = x_gaze_resampled[bounds[0] : bounds[1]]
        y_gaze_slice = y_gaze_resampled[bounds[0] : bounds[1]]

        # * Remove previous gaze data
        if et_scatter:
            et_scatter.remove()
            [el.remove() for el in et_line]

        # * Plot gaze data
        cmap = plt.get_cmap("Reds")
        norm = plt.Normalize(0, x_gaze_slice.shape[0])
        et_colors = cmap(
            norm(
                np.linspace(0, x_gaze_slice.shape[0], x_gaze_slice.shape[0]) * 0.5 + 10
            )
        )

        et_scatter = ax_et.scatter(
            x_gaze_slice, y_gaze_slice, c=et_colors, s=2, alpha=0.5
        )
        # segments = np.stack((x_gaze_slice, y_gaze_slice), axis=1)
        et_line = ax_et.plot(
            x_gaze_slice, y_gaze_slice, c="r", ls="-", linewidth=1, alpha=0.3
        )

        # * Plot EEG data
        if last_plot_x == win_len_samples:
            last_plot_x = 0
            reset_eeg_plot()

        x = np.arange(last_plot_x, last_plot_x + step)

        last_eeg_x += step
        last_plot_x += step

        for i in range(eeg_slice.shape[0]):
            ax_eeg1.plot(x, eeg_slice[i], color=colors[i])

        # ax_eeg2.plot(x, avg_eeg_data[bounds[0] : bounds[1]], color="black")
        # for group_name, group_inds in ch_group_inds.items():
        for group_name, group_data in eeg_group_data.items():
            ax_eeg2.plot(
                x,
                group_data[bounds[0] : bounds[1]],
                label=group_name,
                color=group_colors[group_name],
            )
        if idx_step == 0:
            ax_eeg2_legend = ax_eeg2.get_legend_handles_labels()
        # ax_eeg2.legend(
        #     bbox_to_anchor=(1.005, 1),
        #     loc="upper left",
        #     borderaxespad=0,
        # )

        # * Plot vertical lines to mark the end of the current slice
        ax_eeg1_line = ax_eeg1.vlines(
            x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
        )

        ax_eeg2_line = ax_eeg2.vlines(
            x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
        )

        # * Plot EEG Topomap
        ax_topo.clear()
        mne.viz.plot_topomap(
            data=eeg_slice.mean(axis=1), pos=chans_pos_xy, **topo_plot_params
        )

        ax_eeg2.legend().remove()
        ax_eeg2.legend(
            *ax_eeg2_legend,
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        # * Save the figure
        plt.tight_layout()

        plt.savefig(
            eeg_frames_dir / f"frame_{str(idx_step+1).zfill(zfill_len)}.png", dpi=dpi
        )

    reset_eeg_plot()
    reset_et_plot(show_icons=False, show_fix_cross=True)

    ax_eeg2.legend(
        *ax_eeg2_legend,
        bbox_to_anchor=(1.005, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(eeg_frames_dir / f"frame_{str(0).zfill(zfill_len)}.png", dpi=dpi)

    fps = 3
    create_video_from_frames(eeg_frames_dir, "eeg_video.mp4", fps, zfill_len)


# * ####################################################################################
# * CREATE RDMs
# * ####################################################################################


def normalize(data: np.ndarray, method: str = "min-max"):
    avail_methods = {
        "max": lambda data: data / np.max(data),
        "min-max": lambda data: (data - np.min(data)) / (np.max(data) - np.min(data)),
        "z-score": lambda data: (data - np.mean(data)) / np.std(data),
    }
    method_names = list(avail_methods.keys())

    if method not in method_names:
        raise ValueError(f"Invalid normalization method. Choose from: {method_names}")

    return avail_methods[method](data)


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


def rsa_with_rsatoolbox():
    import numpy as np
    import rsatoolbox
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm
    from scipy.signal import find_peaks

    # Example data: patterns x features x participants
    # Replace with your EEG or other data
    data = np.random.rand(10, 64, 20)  # 10 patterns, 64 features, 20 participants

    # Create an rsatoolbox Dataset
    labels = {"pattern_type": ["type1", "type2", "type3"]}  # Add your conditions
    dataset = Dataset(measurements=data, descriptors=labels)
    data.shape

    calc_rdm
    # Compute RDMs for each participant
    rdms = calc_rdm(dataset, method="correlation")  # Correlation-based dissimilarity
    # ##################################################################################

    p_t = list(all_subj_pattern_erps.keys())
    chan_names = all_subj_pattern_erps[p_t[0]].ch_names
    excl = chan_names[27]
    incl = [c for c in chan_names if c != excl]

    occipital_chans = [c for c in incl if c in eeg_chan_groups["occipital"]]

    evoked_times = all_subj_pattern_erps[p_t[0]].times
    time_window = (0.025, 0.2)
    time_mask = (evoked_times >= time_window[0]) & (evoked_times <= time_window[1])

    features = []

    for pattern, pattern_data in all_subj_pattern_erps.items():
        _data = pattern_data.get_data(picks=occipital_chans).copy()

        # plt.plot(_data.T, color="black")
        # plt.plot(_data.mean(axis=0), color="red")
        # vlines = np.where(time_mask)[0]
        # vlines = [vlines[0], vlines[-1]]
        # plt.vlines(vlines, ymin=_data.min(), ymax=_data.max(), color='blue')
        # plt.xticks(np.arange(0, len(evoked_times), 150), evoked_times[::150].round(3)*1000)

        _data = _data.mean(axis=0)
        neg_peak = _data[time_mask].argmin()
        peak_latency = evoked_times[time_mask][neg_peak]
        peak_amp = _data[time_mask][neg_peak]

        # plt.hlines(peak_amp, xmin=0, xmax=len(_data), color='r')
        # plt.vlines(np.where(evoked_times == peak_latency)[0], ymin=_data.min(), ymax=_data.max(), color='r')
        # plt.tight_layout()
        # plt.show()
        features.append([peak_latency, peak_amp])
        # features.append(peak_latency)

        print(f"peak latency = {peak_latency:.5f}  | peak amplitude = {peak_amp:.9f}")

    # Step 5: Create rsatoolbox Dataset
    dataset = Dataset(
        # measurements=np.array(features)[:, None]
        measurements=np.array(features),
        # descriptors={'pattern_type': p_t},
        # descriptors={'participants': participants},
        # obs_descriptors={'pattern_type': [pattern_type] * len(participants)},
        channel_descriptors={"feature": ["latency", "amplitude"]},
    )

    # np.array(features)[:, None].shape
    # rdms = calc_rdm(dataset, method='correlation')
    rdms = calc_rdm(dataset, method="correlation")

    this_rdm = rdms.get_matrices()[0]

    plt.imshow(this_rdm)
    plt.xticks(np.arange(0, len(p_t)), p_t, rotation=90)
    plt.yticks(
        np.arange(0, len(p_t)),
        p_t,
    )
    plt.title("RDM by pattern type")

    normalized_rdm = this_rdm / this_rdm.max()

    for (j, i), label in np.ndenumerate(normalized_rdm):
        plt.text(i, j, int(round(label, 2) * 100), ha="center", va="center")
    plt.colorbar()


def rsa_with_rsatoolbox2():
    import numpy as np
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm

    # Define the time window for analysis
    time_window = (0.025, 0.2)

    # Function to extract latency and amplitude of the first negative peak
    def extract_first_negative_peak(evoked, time_window):
        time_idx = (evoked.times >= time_window[0]) & (evoked.times <= time_window[1])
        data = evoked.get_data().mean(axis=0)  # Average across channels
        time_values = evoked.times[time_idx]
        data_values = data[time_idx]

        # Find first negative peak
        neg_peak_idx = np.argmin(data_values)  # Index of the most negative point
        neg_peak_latency = time_values[neg_peak_idx]
        neg_peak_amplitude = data_values[neg_peak_idx]

        return neg_peak_latency, neg_peak_amplitude

    # Dictionary to store RDMs
    rdms_by_participant = {}

    # Iterate over participants
    for participant, patterns in subj_pattern_erps.items():
        feature_matrix = []
        pattern_labels = []

        # Extract features for each pattern
        for pattern_name, evoked in patterns.items():
            latency, amplitude = extract_first_negative_peak(evoked, time_window)
            feature_matrix.append([latency, amplitude])
            pattern_labels.append(pattern_name)

        feature_matrix = np.array(feature_matrix)

        # Create rsatoolbox Dataset
        dataset = Dataset(
            measurements=feature_matrix,
            obs_descriptors={"pattern": pattern_labels},
            channel_descriptors={"features": ["latency", "amplitude"]},
        )

        # Calculate RDM
        rdm = calc_rdm(dataset, method="euclidean")
        rdms_by_participant[participant] = rdm

    # Display the RDMs for each participant
    for participant, rdm in rdms_by_participant.items():
        print(f"RDM for Participant {participant}:")
        # print(rdm)

    [i for i in dir(rdms_by_participant[1]) if not i.startswith("_")]
    rdms_by_participant[1].to_df()
    rdms_by_participant[1].n_rdm
    rdms_by_participant[1].n_cond

    rdms_by_participant[1].get_matrices()


def plot_matrix(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_values: bool = False,
    as_pct: bool = False,
    norm: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    text_color: Optional[str] = None,
    cbar: bool = False,
    ax: Optional[plt.Axes] = None,
):
    """
    Plots a correlation matrix with optional labels and value annotations.

    Args:
        - corr_matrix (np.ndarray): A square matrix representing correlations.
        - labels (Optional[List[str]]): List of labels for the rows/columns of the matrix.
        - show_values (bool): Whether to display values on the plot. Default is False.
        - ax (Optional[plt.Axes]): An existing matplotlib Axes object. If None, a new one is created.

    Returns:
        - None
    """
    m = matrix.copy()

    if norm is not None:
        m = normalize(m, norm)

    if ax is None:
        fig, ax = plt.subplots()

    if mask is not None:
        m = np.ma.masked_array(m, mask=mask)
        # m_masked = np.where(mask, np.nan, m)

    ax.set_title(title) if title is not None else None

    # * Create the heatmap
    # cax = ax.matshow(m, cmap=cmap)
    cax = ax.imshow(m, cmap=cmap)

    # * Add the colorbar
    if cbar:
        plt.colorbar(cax, ax=ax)

    # Add labels if provided
    if labels is not None:
        ax.set_xticks(range(m.shape[0]))
        ax.set_yticks(range(m.shape[0]))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

    # * Optionally display the values
    if show_values:
        if text_color is None:
            text_colors = np.array(
                ["black" if abs(i) < 0.5 else "white" for i in m.flatten()]
            ).reshape(m.shape)
        else:
            text_colors = np.array([text_color for i in m.flatten()]).reshape(m.shape)

        if as_pct:
            m = (np.round(m, 2) * 100).astype(int)

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}" if not as_pct else f"{m[i, j]}",
                    ha="center",
                    va="center",
                    color=text_colors[i, j],
                )


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

    gaze_features = {}
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

    pupil_diam_features = {}
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

            # plot_rdm(
            #     rdm,
            #     labels=patterns,
            #     title=f"Pupil Diam RDM for Participant {participant}",
            #     axis=ax,
            #     show_values=True,
            # )
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
# * METHOD 1
# * ####################################################################################


def analyze_flash_period_draft1():
    # for epoch_N in tqdm(range(len(manual_et_epochs))):
    for epoch_N in tqdm(range(2, 6)):
        # epoch_N = 6 # ! TEMP

        # * Extract epoch data
        et_epoch = manual_et_epochs[epoch_N]
        eeg_epoch = manual_eeg_epochs[epoch_N]

        # * GET SACCADE AND FIXATION EVENTS
        et_trial_evts, et_trial_evt_ids = mne.events_from_annotations(et_epoch)

        # et_trial_evts = pd.DataFrame(
        #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
        # )
        # et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

        # diff_indices = np.where(pd.Series(et_trial_evts[:, -1]).diff() != 0)[0]
        # et_trial_evts = et_trial_evts[diff_indices]
        # et_trial_evts = pd.DataFrame(
        #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
        # )
        # et_trial_evts["event_id"] = et_trial_evts["event_id"].replace({v:k for k,v in et_trial_evt_ids.items()})
        # # list(zip(et_trial_evts["sample_nb"][:-1], et_trial_evts["sample_nb"][1:]))
        # # fixation_evts = et_trial_evts[et_trial_evts["event_id"] == "fixation"]
        # # et_trial_evts['event_id'].tolist()

        # * Get channel positions for topomap
        info = eeg_epoch.info

        chans_pos_xy = np.array(
            list(info.get_montage().get_positions()["ch_pos"].values())
        )[:, :2]

        trial_info = get_trial_info(epoch_N)
        stim_pos, stim_order = trial_info[:2]

        # * Resample eye-tracking data for the current trial
        x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
            et_epoch, tracked_eye, et_sfreq, eeg_sfreq
        )

        # * Synchronize data lengths
        eeg_data = eeg_epoch.get_data()

        # eeg_trial_evts, epoch_evt_ids = mne.events_from_annotations(eeg_epoch)
        eeg_trial_evts = pd.Series(eeg_data[-1, :])

        # * Find indices where consecutive events are different
        diff_indices = np.where(eeg_trial_evts.diff() != 0)[0]
        eeg_trial_evts = eeg_trial_evts[diff_indices]
        eeg_trial_evts = eeg_trial_evts[eeg_trial_evts != 0]
        eeg_trial_evts = eeg_trial_evts.replace(valid_events_inv)

        # * Drop EOG & Status channels
        eeg_data = eeg_data[:-5, :]

        # * Ensure that the data arrays have the same length
        min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
        eeg_data = eeg_data[:, :min_length]
        avg_eeg_data = eeg_data.mean(axis=0)

        x_gaze_resampled = x_gaze_resampled[:min_length]
        y_gaze_resampled = y_gaze_resampled[:min_length]

        # * Heatmap of gaze data
        all_stim_onset = eeg_trial_evts[eeg_trial_evts == "stim-all_stim"].index[0]
        trial_end = eeg_trial_evts[eeg_trial_evts == "trial_end"].index[0]

        heatmap, _, _ = get_gaze_heatmap(
            x_gaze_resampled[all_stim_onset:trial_end],
            y_gaze_resampled[all_stim_onset:trial_end],
            screen_resolution,
            bin_size=100,
            show=True,
        )

        stim_flash_evts = eeg_trial_evts[eeg_trial_evts.str.contains("stim-")]
        event_bounds = stim_flash_evts.index
        event_bounds = list(zip(event_bounds[:-1], event_bounds[1:]))
        event_bounds = list(zip(stim_flash_evts.values, event_bounds))

        et_data = np.array([x_gaze_resampled, y_gaze_resampled]).T

        targets_fixation = {}

        for i, (event_id, ev_bounds) in enumerate(event_bounds):
            event_et_data = et_data[ev_bounds[0] : ev_bounds[1]]
            target_grid_loc = stim_order[i]
            target_id, target_coords = stim_pos[target_grid_loc]
            targ_left, targ_right, targ_bottom, targ_top = target_coords

            on_target_inds = [[]]
            for j, (eye_x, eye_y) in enumerate(event_et_data):
                on_target = (
                    targ_left <= eye_x <= targ_right
                    and targ_bottom <= eye_y <= targ_top
                )
                if on_target:
                    on_target_inds[-1].append(ev_bounds[0] + j)
                else:
                    if len(on_target_inds[-1]) > 0:
                        on_target_inds.append([])

            for inds_list in on_target_inds:
                if len(inds_list) < 5:
                    on_target_inds.remove(inds_list)

            targets_fixation[target_grid_loc] = on_target_inds

        # * TESTING
        for targ_ind in targets_fixation:
            target_grid_loc = targ_ind
            target_id, target_coords = stim_pos[target_grid_loc]
            targ_left, targ_right, targ_bottom, targ_top = target_coords

            for fixation in targets_fixation[targ_ind]:
                # print(f"{et_data[eye_pos_inds].shape[0] / eeg_sfreq:.3f} seconds")87

                fig, ax = plt.subplots(3, 1, figsize=(10, 6))
                ax_et, ax_eeg, ax_eeg_avg = ax
                ax_et.set_xlim(0, screen_resolution[0])
                ax_et.set_ylim(screen_resolution[1], 0)
                ax_et.imshow(
                    icon_images[target_id],
                    extent=[targ_left, targ_right, targ_bottom, targ_top],
                    origin="lower",
                )

                eye_pos_inds = fixation
                x, y = et_data[eye_pos_inds].T
                ax_et.scatter(x, y, c="red", s=2)

                duration_ms = len(eye_pos_inds) / eeg_sfreq * 1000

                ax_eeg.plot(eeg_data[:-5, eye_pos_inds].T)
                ax_eeg_avg.plot(eeg_data[:-5, eye_pos_inds].T.mean(axis=1))

                xticks = np.arange(
                    0, eeg_data[:-5, eye_pos_inds].T.mean(axis=1).shape[0], 100
                )
                ax_eeg_avg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

                plt.show()


# * ####################################################################################
# * ANALYZE FLASH PERIOD
# * METHOD 2
# * Problem: if fixation on same location as target before first flash,
# * fixation data before first flash will be excluded
# * ####################################################################################


def analyze_flash_period_draft2(epoch_N):
    # ! IMPORTANT
    # TODO
    # * Right Now et_epoch_evts are extracted from raw ET data, but EEG data is sampled
    # * at a different rate. So EEG activity might not be aligned with ET data
    # * Possible solutions:
    # *     1. Use resampled ET data and identify events from this data (might not be possible)
    # *     2. Use time as index instead of sample number, and identify closest sample number
    # *        in EEG data for each event
    # TODO

    # * Extract epoch data
    et_epoch = manual_et_epochs[epoch_N]
    eeg_epoch = manual_eeg_epochs[epoch_N]

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target

        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        first_flash = annotations.query("description == 'stim-flash_sequence'").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]
        last_fixation = annotations.query("description == 'fixation'").iloc[-1]

        # * Convert to seconds
        first_flash_onset = first_flash["onset"]
        all_stim_pres_onset = all_stim_pres["onset"]
        end_time = all_stim_pres_onset + last_fixation["duration"]

        # * Crop the data
        epoch = epoch.copy().crop(first_flash_onset, end_time)

        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # if annotations["description"].iloc[0] != "stim-flash_sequence":
        #     annotations = annotations.iloc[
        #         first_flash.name : all_stim_pres.name + 1
        #     ].copy()

        return epoch, annotations

    et_epoch, et_annotations = crop_et_epoch(et_epoch)

    eeg_epoch = eeg_epoch.copy().crop(
        et_annotations["onset"].iloc[0], et_annotations["onset"].iloc[-1]
    )

    # eeg_epoch.times[-1]
    # eeg_epoch.times[-1]

    # * ########
    # eeg_events, _ = mne.events_from_annotations(eeg_epoch, event_id=valid_events)
    # eeg_events[:, 0] -= eeg_events[0, 0]

    # first_flash_eeg = eeg_events[eeg_events[:, 2] == valid_events["stim-flash_sequence"]][0, 0]
    # all_stim_pres_eeg = eeg_events[eeg_events[:, 2] == valid_events["stim-all_stim"]][0, 0]

    # eeg_epoch = eeg_epoch.get_data()[:, first_flash_eeg : all_stim_pres_eeg + 1]
    # * ########

    # et_trial_evts = pd.DataFrame(
    #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
    # )
    # et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

    # diff_indices = np.where(pd.Series(et_trial_evts[:, -1]).diff() != 0)[0]
    # et_trial_evts = et_trial_evts[diff_indices]
    # et_trial_evts = pd.DataFrame(
    #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
    # )
    # et_trial_evts["event_id"] = et_trial_evts["event_id"].replace({v:k for k,v in et_trial_evt_ids.items()})
    # # list(zip(et_trial_evts["sample_nb"][:-1], et_trial_evts["sample_nb"][1:]))
    # # fixation_evts = et_trial_evts[et_trial_evts["event_id"] == "fixation"]
    # # et_trial_evts['event_id'].tolist()

    # * Get channel positions for topomap
    info = eeg_epoch.info

    chans_pos_xy = np.array(
        list(info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    trial_info = get_trial_info(epoch_N)
    stim_pos, stim_order = trial_info[:2]

    # * Resample eye-tracking data for the current trial
    # x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
    #     et_epoch, tracked_eye, et_sfreq, eeg_sfreq
    # )

    # * Synchronize data lengths
    # eeg_data = eeg_epoch.get_data()

    # eeg_trial_evts, epoch_evt_ids = mne.events_from_annotations(eeg_epoch)
    # eeg_trial_evts = pd.Series(eeg_data[-1, :])

    # * Find indices where consecutive events are different
    # diff_indices = np.where(eeg_trial_evts.diff() != 0)[0]
    # eeg_trial_evts = eeg_trial_evts[diff_indices]
    # eeg_trial_evts = eeg_trial_evts[eeg_trial_evts != 0]
    # eeg_trial_evts = eeg_trial_evts.replace(valid_events_inv)

    # * Drop EOG & Status channels
    # eeg_data = eeg_data[:-5, :] # ! OLD VERSION
    eeg_data = eeg_epoch.copy().pick(["eeg"])
    eog_data = eeg_epoch.copy().pick(["eog"])

    # * Ensure that the data arrays have the same length
    # min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
    # eeg_data = eeg_data[:, :min_length]
    # avg_eeg_data = eeg_data.mean(axis=0)

    # x_gaze_resampled = x_gaze_resampled[:min_length]
    # y_gaze_resampled = y_gaze_resampled[:min_length]

    # * Heatmap of gaze data
    # all_stim_onset = eeg_trial_evts[eeg_trial_evts == "stim-all_stim"].index[0]
    # trial_end = eeg_trial_evts[eeg_trial_evts == "trial_end"].index[0]

    # heatmap, _, _ = get_gaze_heatmap(
    #     x_gaze_resampled[all_stim_onset:trial_end],
    #     y_gaze_resampled[all_stim_onset:trial_end],
    #     screen_resolution,
    #     bin_size=100,
    #     show=True,
    # )

    # stim_flash_evts = eeg_trial_evts[eeg_trial_evts.str.contains("stim-")]
    # event_bounds = stim_flash_evts.index
    # event_bounds = list(zip(event_bounds[:-1], event_bounds[1:]))
    # event_bounds = list(zip(stim_flash_evts.values, event_bounds))

    # * ################################################################################
    ch_group_inds = {
        region: [i for i, ch in enumerate(eeg_epoch.ch_names) if ch in ch_group]
        for region, ch_group in eeg_chan_groups.items()
    }

    # * GET SACCADE AND FIXATION EVENTS
    # et_epoch_evts, _ = mne.events_from_annotations(et_epoch, event_id=et_events_dict)
    # et_epoch_evts = pd.DataFrame(et_epoch_evts, columns=["sample_nb", "prev", "description"])
    # et_epoch_evts['description'] = et_epoch_evts['description'].replace(et_events_dict_inv)

    # et_evt_df = et_epoch.annotations.to_data_frame()
    # # et_trial_start = et_evt_df[et_evt_df["description"] == "trial_start"].index[0]
    # # et_trial_end = et_evt_df[et_evt_df["description"] == "trial_end"].index[0]
    # et_first_flash = et_evt_df[et_evt_df["description"] == "stim-flash_sequence"].index[
    #     0
    # ]
    # et_all_stim_pres = et_evt_df[et_evt_df["description"] == "stim-all_stim"].index[0]

    # et_evt_df = et_evt_df.iloc[et_first_flash : et_all_stim_pres + 1]
    # et_evt_df.reset_index(drop=True, inplace=True)
    # et_evt_df["onset"] = (
    #     (et_evt_df["onset"] - et_evt_df["onset"].iloc[0]).dt.total_seconds().round(3)
    # )

    # eeg_evt_df = eeg_epoch.annotations.to_data_frame()
    # # eeg_trial_start = eeg_evt_df[eeg_evt_df["description"] == "trial_start"].index[0]
    # # eeg_trial_end = eeg_evt_df[eeg_evt_df["description"] == "trial_end"].index[0]

    # eeg_first_flash = eeg_evt_df[
    #     eeg_evt_df["description"] == "stim-flash_sequence"
    # ].index[0]

    # eeg_all_stim_pres = eeg_evt_df[eeg_evt_df["description"] == "stim-all_stim"].index[
    #     0
    # ]
    # eeg_evt_df = eeg_evt_df.iloc[eeg_first_flash : eeg_all_stim_pres + 1]
    # eeg_evt_df.reset_index(drop=True, inplace=True)
    # eeg_evt_df["onset"] = (
    #     (eeg_evt_df["onset"] - eeg_evt_df["onset"].iloc[0]).dt.total_seconds().round(3)
    # )
    # # eeg_evt_df["onset"] = eeg_evt_df["onset"].round(3)

    # # ! TEMP
    # et_evt_df.query("not description.str.contains('fixation|saccade')")
    # eeg_evt_df

    # et_evt_df.tail(20)
    # # ! TEMP

    # ! OLD VERSION
    # et_epoch_evts[:, 0] -= et_epoch_evts[0, 0]

    # et_epoch_evts = pd.DataFrame(
    #     et_epoch_evts, columns=["sample_nb", "prev", "event_id"]
    # )

    # et_epoch_evts["event_id"] = et_epoch_evts["event_id"].replace(et_events_dict_inv)
    # et_epoch_evts["time"] = et_epoch_evts["sample_nb"] / eeg_sfreq

    # trial_start_ind = et_epoch_evts[et_epoch_evts["event_id"] == "trial_start"].index[0]

    # trial_end_ind = et_epoch_evts[et_epoch_evts["event_id"] == "trial_end"].index[0]

    # et_epoch_evts = et_epoch_evts.iloc[trial_start_ind : trial_end_ind + 1]
    # et_epoch_evts = et_epoch_evts.reset_index(drop=True)

    # first_flash = et_epoch_evts.query("event_id == 'stim-flash_sequence'").index[0]
    # all_stim_pres = et_epoch_evts.query("event_id == 'stim-all_stim'").index[0]
    # last_sacc = (
    #     et_epoch_evts.iloc[all_stim_pres:].query("event_id == 'saccade'").index[0]
    # )

    # et_epoch_evts = et_epoch_evts.iloc[first_flash : last_sacc + 1]
    # et_epoch_evts = et_epoch_evts.reset_index(drop=True)

    # flash_event_ids = ["stim-flash_sequence", "stim-flash_choices", "stim-all_stim"]
    # # flash_events = et_epoch_evts[et_epoch_evts["event_id"].isin(flash_event_ids)]

    # fix_and_sac = et_epoch_evts.query("event_id.isin(['fixation', 'saccade'])").copy()

    # fixation_inds = fix_and_sac.query("event_id == 'fixation'").index
    # ! OLD VERSION

    flash_event_ids = ["stim-flash_sequence", "stim-flash_choices", "stim-all_stim"]

    fix_and_sac = et_annotations.query("description.isin(['fixation', 'saccade'])")
    fixation_inds = et_annotations.query("description == 'fixation'").index

    fixation_data = {i: [] for i in stim_order}

    for fixation_ind in fixation_inds:
        # ! OLD VERSION
        # stim_flash_ind = (
        #     et_epoch_evts.iloc[:fixation_ind]
        #     .query(f"event_id.isin({flash_event_ids[:-1]})")
        #     .shape[0]
        #     - 1
        # )
        # ! OLD VERSION

        stim_flash_ind = (
            et_annotations.iloc[:fixation_ind]
            .query(f"description.isin({flash_event_ids[:-1]})")
            .shape[0]
            - 1
        )

        target_grid_loc = stim_order[stim_flash_ind]
        target_id, target_coords = stim_pos[target_grid_loc]
        targ_left, targ_right, targ_bottom, targ_top = target_coords

        # fixation = fix_and_sac.loc[fixation_ind]
        fixation = et_annotations.loc[fixation_ind]

        # ! OLD VERSION
        # fix_sample_ind = fixation["sample_nb"]
        # fix_sample_time = fixation["time"]

        # next_sacc = (
        #     et_epoch_evts.iloc[fixation_ind:].query("event_id == 'saccade'").iloc[0]
        # )
        # next_sacc_sample_ind = next_sacc["sample_nb"]
        # next_sacc_sample_time = next_sacc["time"]
        # ! OLD VERSION

        fixation_start = fixation["onset"]
        fixation_duration = fixation["duration"]
        fixation_stop = fixation_start + fixation_duration

        # start_sample_ind_et = int(fixation_start * et_sfreq)
        # stop_sample_ind_et = int(fixation_stop * et_sfreq)

        # gaze_x, gaze_y, pupil_diam = et_epoch.get_data()[
        #     :, start_sample_ind_et:stop_sample_ind_et
        # ]
        gaze_x, gaze_y, pupil_diam = (
            et_epoch.copy().crop(fixation_start, fixation_stop).get_data()
        )

        # ! OLD VERSION
        # gaze_x, gaze_y = manual_et_epochs[epoch_N].get_data()[
        #     :2, fix_sample_ind:next_sacc_sample_ind
        # ]
        #
        # # gaze_x = x_gaze_resampled[fix_sample_ind:next_sacc_sample_ind]
        # # gaze_y = y_gaze_resampled[fix_sample_ind:next_sacc_sample_ind]

        # fixation_duration = gaze_x.shape[0] / eeg_sfreq * 1000
        # ! OLD VERSION

        mean_x, mean_y = gaze_x.mean(), gaze_y.mean()

        on_target = (targ_left <= mean_x <= targ_right) and (
            targ_bottom <= mean_y <= targ_top
        )

        if fixation_duration >= 0.2 and on_target:
            discarded = False
            fixation_data[stim_flash_ind].append(np.array([gaze_x, gaze_y]))
        else:
            discarded = True

        title = f"stim-{stim_flash_ind + 1} ({fixation_duration * 1000:.0f} ms)"
        title += " " + ("DISCARDED" if discarded else "SAVED")

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax_et = fig.add_subplot(gs[0, :])
        ax_eeg = fig.add_subplot(gs[1, :])
        ax_eeg_avg = fig.add_subplot(gs[2, 0], sharex=ax_eeg)

        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_title(title)

        ax_et.imshow(
            icon_images[target_id],
            extent=[targ_left, targ_right, targ_bottom, targ_top],
            origin="lower",
        )

        rectangle = mpatches.Rectangle(
            (targ_left, targ_bottom),
            targ_right - targ_left,
            targ_top - targ_bottom,
            linewidth=1,
            linestyle="--",
            edgecolor="black",
            facecolor="none",
        )
        ax_et.add_patch(rectangle)

        ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
        ax_et.scatter(mean_x, mean_y, c="yellow", s=3)

        # eeg_slice = eeg_data[:, fix_sample_ind:next_sacc_sample_ind] # ! OLD VERSION
        # eeg_slice = eeg_data[:, start_sample_ind_et:stop_sample_ind_et]
        eeg_slice = eeg_data.copy().crop(fixation_start, fixation_stop).get_data()

        ax_eeg.plot(eeg_slice.T)

        # ax_eeg_avg.plot(eeg_slice.mean(axis=0))

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["occipital"]].mean(axis=0),
            color="red",
            label="occipital",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["parietal"]].mean(axis=0),
            color="green",
            label="parietal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["centro-parietal"]].mean(axis=0),
            color="purple",
            label="centro-parietal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["temporal"]].mean(axis=0),
            color="orange",
            label="temporal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["frontal"]].mean(axis=0),
            color="blue",
            label="frontal",
        )

        ax_eeg_avg.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg.set_xlim(0, eeg_slice.shape[1])

        xticks = np.arange(0, eeg_slice.shape[1], 100)
        ax_eeg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    # set([len(v) for v in fixation_data.values()])
    # fixation_data[11][0]


def analyze_flash_period(epoch_N):
    # ! IMPORTANT
    # TODO
    # * eeg_baseline
    # TODO

    # * Extract epoch data
    et_epoch = manual_et_epochs[epoch_N]
    eeg_epoch = manual_eeg_epochs[epoch_N]

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]
        # last_flash = annotations.query("description.str.contains('flash')").iloc[-1]
        last_fixation = (
            annotations.iloc[: all_stim_pres.name]
            .query("description == 'fixation'")
            .iloc[-1]
        )

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
    trial_info = get_trial_info(epoch_N)
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


def analyze_decision_period_1(
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
    trial_info = get_trial_info(trial_N, raw_behav)
    stim_pos, stim_order, sequence, choices, response_ind, solution, rt = trial_info
    response = choices.get(response_ind, "timeout")
    solution_ind = {v: k for k, v in choices.items()}[solution]
    correct = response == solution

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
    wrong_choice_icon_inds = [i for i in choice_icon_inds if i != solution_ind]

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
    fixation_data = {i: [] for i in range(len(stim_order))}
    eeg_fixation_data = {i: [] for i in range(len(stim_order))}

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
        times = eeg_slice.times
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
            fig = plot_eeg_and_gaze_fixations(
                # * Convert to microvolts
                # eeg_data=eeg_slice * 1e6,
                eeg_data=eeg_slice.get_data(picks="eeg") * 1e6,
                et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                eeg_baseline=eeg_baseline,
                response_onset=response_onset,
                eeg_start_time=eeg_start_time,
                eeg_end_time=eeg_end_time,
                stim_pos=stim_pos,
                chans_pos_xy=chans_pos_xy,
                ch_group_inds=ch_group_inds,
                group_colors=group_colors,
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


def plot_sequence(type):
    subj_N = 1
    sess_N = 1
    epoch_N = 0

    sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals = load_raw_data(
        subj_N, sess_N, data_dir
    )

    trial_info = get_trial_info(epoch_N, raw_behav)
    stim_pos, stim_order, sequence, choices, response_ind, solution, rt = trial_info

    if type == "image":
        fig, ax_et = plt.subplots(frameon=False)
        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_xticks([])
        ax_et.set_yticks([])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # * Plot target icon
        for icon_name, pos in stim_pos:
            targ_left, targ_right, targ_bottom, targ_top = pos
            ax_et.imshow(
                icon_images[icon_name],
                extent=[targ_left, targ_right, targ_bottom, targ_top],
                origin="lower",
            )

            # # * Plot rectangle around target, with dimensions == img_size
            # rectangle = mpatches.Rectangle(
            #     (targ_left, targ_bottom),
            #     img_size[0],
            #     img_size[1],
            #     linewidth=0.8,
            #     linestyle="--",
            #     edgecolor="black",
            #     facecolor="none",
            # )
            # ax_et.add_patch(rectangle)
        ax_et.set_facecolor("lightgrey")
        # ax_et.axis("off")
        ax_et.spines[["left", "right", "top", "bottom"]].set_visible(False)
        # ax_et.splines = []
        fig.set_facecolor((0.5, 0.5, 0.5))  # "lightgrey")
        fig.tight_layout()
        fig.savefig(wd / "sequence.png", dpi=300)

    elif type == "video":
        # * ################################################################################
        # * VIDEO
        # * ################################################################################
        fps = 30
        zfill_len = 3
        frame_count = 0

        save_dir = wd / "sequence_video"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(exist_ok=True)

        fig, ax_et = plt.subplots(frameon=False)
        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_xticks([])
        ax_et.set_yticks([])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax_et.set_facecolor("lightgrey")
        ax_et.spines[["left", "right", "top", "bottom"]].set_visible(False)
        fig.set_facecolor((0.5, 0.5, 0.5))  # "lightgrey")
        fig.tight_layout()

        ax_et_fix_cross = ax_et.scatter(
            screen_resolution[0] / 2,
            screen_resolution[1] / 2,
            s=80,
            marker="+",
            linewidths=1,
            color="black",
        )
        ax_et_fix_cross.set_visible(False)

        ax_et_plotted_icons = []
        for icon_name, icon_pos in stim_pos:
            left, right, bottom, top = icon_pos

            this_icon = ax_et.imshow(
                icon_images[icon_name],
                extent=[left, right, bottom, top],
                origin="lower",
            )

            ax_et_plotted_icons.append(this_icon)
            this_icon.set_visible(True)

        plt.savefig(
            save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
        )  # , bbox_inches="tight")
        frame_count += 1

        [icon.set_visible(False) for icon in ax_et_plotted_icons]

        for frame in range(0, fps):
            ax_et_fix_cross.set_visible(True)
            plt.savefig(
                save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
            )  # , bbox_inches="tight")
            frame_count += 1

        ax_et_fix_cross.set_visible(False)

        for icon_idx in stim_order:
            for frame in range(0, int(0.6 * fps)):
                ax_et_plotted_icons[icon_idx].set_visible(True)
                plt.savefig(
                    save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
                )  # , bbox_inches="tight")
                frame_count += 1
            ax_et_plotted_icons[icon_idx].set_visible(False)

        [icon.set_visible(True) for icon in ax_et_plotted_icons]

        for frame in range(0, int(fps * 2)):
            plt.savefig(
                save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
            )  # , bbox_inches="tight")
            frame_count += 1

        [f.unlink() for f in save_dir.glob("*.png")]
        create_video_from_frames(save_dir, "sequence_video.mp4", fps, zfill_len)


def plot_eeg_and_gaze_fixations(
    eeg_data,
    et_data,
    eeg_baseline,
    response_onset,
    eeg_start_time,
    eeg_end_time,
    stim_pos,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    title: str = None,
    vlines=None,
):
    """
    eeg_data: np.array, shape=(n_channels, n_samples)
    """

    gaze_x, gaze_y = et_data
    mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

    # * Set up the figure
    fig = plt.figure(figsize=(10, 6), dpi=200)
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 3, 3], width_ratios=[1, 1])
    ax_et = fig.add_subplot(gs[0, 0])
    ax_topo = fig.add_subplot(gs[0, 1])
    ax_eeg = fig.add_subplot(gs[1, :])
    ax_eeg_group = fig.add_subplot(gs[2, :])  # , sharex=ax_eeg)

    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticks([])
    ax_et.set_yticks([])

    ax_eeg.grid(axis="x", ls="--")
    ax_eeg_group.grid(axis="x", ls="--")
    ax_topo.set_axis_off()

    fig.suptitle(title)
    # ax_et.set_title(title)

    # * Plot target icon
    for icon_name, pos in stim_pos:
        targ_left, targ_right, targ_bottom, targ_top = pos
        ax_et.imshow(
            icon_images[icon_name],
            extent=[targ_left, targ_right, targ_bottom, targ_top],
            origin="lower",
        )

        # # * Plot rectangle around target, with dimensions == img_size
        rectangle = mpatches.Rectangle(
            (targ_left, targ_bottom),
            img_size[0],
            img_size[1],
            linewidth=0.8,
            linestyle="--",
            edgecolor="black",
            facecolor="none",
        )
        ax_et.add_patch(rectangle)

    # * Plot the topomap
    # TODO: Should we slice the topo data from after the baseline correction period?
    mne.viz.plot_topomap(
        eeg_data.mean(axis=1),
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

    # * Plot gaze data
    ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
    ax_et.scatter(mean_gaze_x, mean_gaze_y, c="yellow", s=3)

    # * Plot EEG data
    ax_eeg.plot(eeg_data.T)

    for group_name, group_inds in ch_group_inds.items():
        ax_eeg_group.plot(
            eeg_data[group_inds].mean(axis=0),
            label=group_name,
            color=group_colors[group_name],
        )

    ax_eeg_group.legend(
        bbox_to_anchor=(1.005, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax_eeg.set_xlim(0, eeg_data.shape[1])
    ax_eeg_group.set_xlim(0, eeg_data.shape[1])

    tick_step_time = 0.05
    tick_step_sample = tick_step_time * eeg_sfreq
    x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)

    x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)
    # x_labels = ((x_ticks / eeg_sfreq - eeg_baseline)).round(3)  # .astype(int)
    # ax_eeg.set_xticks(x_ticks, x_labels)
    ax_eeg.set_xticks(x_ticks, [])
    ax_eeg_group.set_xticks(x_ticks, x_labels)
    ax_eeg_group.set_xlabel("Time (ms) relative to gaze fixation onset")

    eeg_sec_xaxis = ax_eeg.secondary_xaxis(location="top")
    t1 = -(response_onset - eeg_start_time)
    t2 = -(response_onset - eeg_end_time - tick_step_time * 0.9)
    x_labels2 = np.arange(t1, t2, tick_step_time).round(2)
    x_labels2 = [f"+{x}" if x >= 0 else f"{x}" for x in x_labels2]
    eeg_sec_xaxis.set_xticks(x_ticks, x_labels2)
    eeg_sec_xaxis.set_xlabel("Time (s) relative to response")

    if vlines is not None:
        for ax in [ax_eeg, ax_eeg_group]:
            ax.vlines(
                vlines,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color="black",
                ls="--",
                lw=1,
            )

    plt.tight_layout()
    # plt.show()

    return fig


def plot_eeg_and_gaze_fixations_plotly(
    eeg_data,
    et_data,
    ch_names,
    eeg_baseline,
    response_onset,
    eeg_start_time,
    eeg_end_time,
    stim_pos,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    title: str = None,
    vlines=None,
    screen_resolution=(1920, 1080),
):
    """
    eeg_data: np.array, shape=(n_channels, n_samples)
    """

    def numpy_to_base64(img_array):
        """Converts a NumPy array image to a base64-encoded string after scaling to 0-255."""
        img_array = (img_array * 255).astype(np.uint8)  # Ensure image is uint8
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    # Check if icon_images is provided
    if icon_images is None:
        raise ValueError("icon_images must be provided and contain the required icons.")

    # Extract gaze data
    gaze_x, gaze_y = et_data
    mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

    # # Create main figure layout
    fig = go.Figure()

    # # * Subplot 1: Gaze data with target icons
    # gaze_scatter = go.Scatter(
    #     x=gaze_x,
    #     y=gaze_y,
    #     mode="markers",
    #     marker=dict(color="red", size=2),
    #     name="Gaze Points",
    # )

    # mean_gaze_scatter = go.Scatter(
    #     x=[mean_gaze_x],
    #     y=[mean_gaze_y],
    #     mode="markers",
    #     marker=dict(color="yellow", size=8),
    #     name="Mean Gaze Point",
    # )
    # fig.add_trace(gaze_scatter)
    # fig.add_trace(mean_gaze_scatter)

    # # Define the layout for the gaze subplot
    # fig.update_xaxes(range=[0, screen_resolution[0]], title="X Position")
    # fig.update_yaxes(
    #     range=[screen_resolution[1], 0], title="Y Position", scaleanchor="x"
    # )

    # # Add icons to plot
    # for icon_name, pos in stim_pos:
    #     targ_left, targ_right, targ_bottom, targ_top = pos
    #     fig.add_shape(
    #         type="rect",
    #         x0=targ_left,
    #         x1=targ_right,
    #         y0=targ_bottom,
    #         y1=targ_top,
    #         line=dict(color="black", dash="dash"),
    #     )

    #     # Convert icon image to base64 and add to layout
    #     if icon_name in icon_images:
    #         base64_image = numpy_to_base64(icon_images[icon_name])
    #         fig.add_layout_image(
    #             dict(
    #                 source=base64_image,
    #                 xref="x",
    #                 yref="y",
    #                 x=targ_left,
    #                 y=targ_top,
    #                 sizex=(targ_right - targ_left),
    #                 sizey=(targ_top - targ_bottom),
    #                 opacity=0.8,
    #                 layer="below",
    #             )
    #         )
    #     else:
    #         print(f"Warning: {icon_name} not found in icon_images.")

    # * Subplot 2: EEG data (line plot for each channel)
    times = np.arange(0, eeg_data.shape[1])  # Sample indices
    for i, ch_data in enumerate(eeg_data):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ch_data,
                mode="lines",
                name=ch_names[i],
                line=dict(width=1),
                opacity=0.5,
            )
        )

    # # Add gridlines and titles
    # fig.update_layout(
    #     title=title or "EEG and Gaze Fixations",
    #     xaxis=dict(title="Time (samples)", gridcolor="lightgray"),
    #     yaxis=dict(title="Amplitude (µV)", gridcolor="lightgray"),
    #     template="plotly_white",
    # )

    # # * Subplot 3: Grouped EEG channels
    # for group_name, group_inds in ch_group_inds.items():
    #     mean_group_data = eeg_data[group_inds].mean(axis=0)
    #     fig.add_trace(
    #         go.Scatter(
    #             x=times,
    #             y=mean_group_data,
    #             mode="lines",
    #             name=f"{group_name} Group",
    #             line=dict(color=group_colors[group_name], width=2),
    #         )
    #     )

    # * Add Vertical Lines for Specific Events
    if vlines is not None:
        for line_pos in vlines:
            fig.add_vline(
                x=line_pos, line=dict(color="black", dash="dash"), name="Event Line"
            )

    # Customize the legend
    fig.update_layout(
        showlegend=True, legend=dict(x=1.05, y=1), title=dict(x=0.5, y=0.95)
    )

    return fig


def plot_eeg(
    eeg_data,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    eeg_sfreq,
    eeg_baseline=0,
    chan_names=None,
    vlines=None,
    title=None,
    plot_topo=True,
    plot_eeg=True,
    plot_eeg_group=True,
    dpi=100,
    figsize=(10, 6),
):
    # ! Temp
    # figsize=(10, 6)
    # plot_topo = True
    # plot_eeg = True
    # plot_eeg_group = False
    # vlines=[100, 230]
    # chan_names = None
    # dpi=100
    # ! Temp

    if chan_names is None:
        chan_names = [f"Ch {i+1}" for i in range(eeg_data.shape[0])]

    # tick_step_time = 0.05
    # tick_step_sample = tick_step_time * eeg_sfreq
    # x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)
    # x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)
    x = (np.arange(0, eeg_data.shape[1]) / eeg_sfreq - eeg_baseline) * 1000
    x_ticks = np.arange(round(x[0]), round(x[-1]) + 1, 50)

    # * Determine height ratios and rows based on flags
    height_ratios = []
    rows = 0
    if plot_topo:
        height_ratios.append(4)
        rows += 1
    if plot_eeg:
        height_ratios.append(3)
        rows += 1
    if plot_eeg_group:
        height_ratios.append(3)
        rows += 1

    # * Set up the figure and GridSpec based on the updated layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(rows, 3, height_ratios=height_ratios, width_ratios=[1, 1, 1])
    fig.suptitle(title)

    ax_topo, ax_eeg, ax_eeg_group = None, None, None

    # * Conditionally add subplots
    current_row = 0
    if plot_topo:
        ax_topo = fig.add_subplot(gs[current_row, 1])
        ax_topo.set_axis_off()

        # * Plot the topomap
        # TODO: Should we slice the topo data from after the baseline correction period?
        mne.viz.plot_topomap(
            eeg_data.mean(axis=1),
            chans_pos_xy,
            # names=chan_names,
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

        current_row += 1

    if plot_eeg:
        # * Plot EEG data, channel by channel
        ax_eeg = fig.add_subplot(gs[current_row, :])
        ax_eeg.grid(axis="x", ls="--")

        for i in range(eeg_data.shape[0]):
            # [i * eeg_sfreq * 1000 for i in eeg_data[i]]
            ax_eeg.plot(x, eeg_data[i], label=chan_names[i])

        ax_eeg.set_xlim(x[0], x[-1])

        if plot_eeg_group:
            ax_eeg.set_xticks(x_ticks)  # , [])
        else:
            ax_eeg.set_xticks(x_ticks)  # , x_labels)
            ax_eeg.set_xlabel("Time (ms) relative to gaze fixation onset")

        current_row += 1

    if plot_eeg_group:
        # * Plot EEG data, grouped by channel group
        ax_eeg_group = fig.add_subplot(gs[current_row, :])
        ax_eeg_group.grid(axis="x", ls="--")

        for group_name, group_inds in ch_group_inds.items():
            ax_eeg_group.plot(
                x,
                eeg_data[group_inds].mean(axis=0),
                label=group_name,
                color=group_colors[group_name],
            )

        ax_eeg_group.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg_group.set_xlim(x[0], x[-1])
        ax_eeg_group.set_xticks(x_ticks)  # , x_labels)

        ax_eeg_group.set_xlabel("Time (ms) relative to gaze fixation onset")

    if vlines is not None:
        for ax in [ax_eeg, ax_eeg_group]:
            if ax is not None:
                ax.vlines(
                    vlines,
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    color="black",
                    ls="--",
                    lw=1,
                )

    plt.tight_layout()
    mplcursors.cursor(hover=True)

    return fig


def plot_eeg_plotly_static(
    eeg_data,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    eeg_sfreq,
    eeg_baseline,
    vlines=None,
    title=None,
):
    import numpy as np

    # Set up figure with subplots
    fig = ps.make_subplots(
        rows=3,
        cols=3,
        subplot_titles=("Topomap", "EEG Time-Series", "Average EEG by Group"),
        row_heights=[0.5, 0.35, 0.35],
        shared_xaxes=True,
        specs=[
            [None, {"type": "scatter"}, None],
            [{"colspan": 3}, None, None],
            [{"colspan": 3}, None, None],
        ],
    )

    # * Topomap - placeholder, Plotly doesn't have built-in topomap functionality like matplotlib.
    fig.add_trace(
        go.Scatter(
            x=chans_pos_xy[:, 0],
            y=chans_pos_xy[:, 1],
            mode="markers+text",
            marker=dict(size=10),
            # text=[f"Ch {i+1}" for i in range(len(chans_pos_xy))],
            textposition="top center",
        ),
        row=1,
        col=2,
    )

    # * Add EEG time-series for each channel
    for i in range(eeg_data.shape[0]):
        fig.add_trace(
            go.Scatter(y=eeg_data[i], mode="lines", name=f"Channel {i+1}"),
            row=2,
            col=1,
        )

    # * Plot average EEG for each group
    for group_name, group_inds in ch_group_inds.items():
        group_data_mean = eeg_data[group_inds].mean(axis=0)
        fig.add_trace(
            go.Scatter(
                y=group_data_mean,
                mode="lines",
                name=group_name,
                line=dict(color=group_colors[group_name]),
            ),
            row=3,
            col=1,
        )

    # Set x-axis properties
    tick_step_time = 0.05
    tick_step_sample = tick_step_time * eeg_sfreq
    x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)
    x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)

    fig.update_xaxes(
        tickvals=x_ticks,
        ticktext=x_labels,
        title_text="Time (ms) relative to gaze fixation onset",
        row=3,
        col=1,
    )

    # Set title
    if title:
        fig.update_layout(title=title)

    # Add vertical lines if provided
    if vlines is not None:
        for vline in vlines:
            fig.add_vline(
                x=vline,
                line=dict(color="black", dash="dash"),
                layer="below",
            )

    # * Save figure as a static image
    # fig.write_image("eeg_plotly_plot.png")

    # * Save figure as an interactive HTML file
    # fig.write_html("eeg_plotly_plot.html")

    # Show figure
    fig.update_layout(height=800, width=1000)
    fig.show()
    fig.update_layout(showlegend=False)

    return fig


def analyze_phase_coupling():
    # see: https://etiennecmb.github.io/tensorpac/auto_examples/erpac/plot_erpac.html#sphx-glr-auto-examples-erpac-plot-erpac-py
    # Example phase-amplitude coupling analysis using tensorpac
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

        # Extract data and sampling frequency
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]

        # Create phase and amplitude frequency vectors
        f_pha = np.linspace(low_freqs[0], low_freqs[1], low_freqs[2])
        f_amp = np.linspace(high_freqs[0], high_freqs[1], high_freqs[2])

        # Initialize PAC object
        p = tensorpac.Pac(idpac=(6, 2, 3), f_pha=f_pha, f_amp=f_amp, dcomplex="wavelet")

        # Compute PAC
        pac = p.filterfit(data, sfreq, n_jobs=-1)

        return pac, p, f_pha, f_amp

    pass


# * ####################################################################################
# * MAIN FUNCTIONS
# * ####################################################################################


def main():
    res_dir = wd / "results/analyzed"
    res_dir.mkdir(exist_ok=True, parents=True)

    # temp =raw_behav.query("rt != 'timeout'")["rt"].astype("float")
    # temp[temp < 3].sort_values()

    # for subj_N in tqdm(range(7, 8), desc="Analyzing data of every subjects"):

    for subj_N in tqdm(
        range(1, len(list(data_dir.glob("subj_*"))) + 1),
        desc="Analyzing data of every subjects",
    ):
        # ! TEMP
        # subj_N = 6
        # sess_N = 1
        # trial_N = 47
        # ! TEMP

        subj_dir = data_dir / f"subj_{subj_N:02}"

        for sess_N in range(1, len(list(subj_dir.glob("sess_*"))) + 1):
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
                    # if trial_N % 2 == 0:
                    # generate_trial_video(manual_et_trials, manual_eeg_trials, raw_behav, trial_N)

                    # fixation_data, eeg_fixation_data = analyze_flash_period(trial_N)

                    (
                        fixation_data,
                        eeg_fixation_data,
                        gaze_target_fixation_sequence,
                        gaze_info,
                        fixations_sequence_erp,
                        fixations_choices_erp,
                    ) = analyze_decision_period_1(
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

    fig_titles = ["ERP - Sequence Icons", "ERP - Choice Icons"]

    for eeg_data, title in zip(
        [fixations_sequence_erp, fixations_choices_erp], fig_titles
    ):
        fig = plot_eeg(
            eeg_data * 1e6,
            chans_pos_xy,
            ch_group_inds,
            group_colors,
            eeg_sfreq,
            eeg_baseline,
            vlines=None,
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


def prepare_eeg_data_for_plot(
    sess_bad_chans: List[str], group_names: List[str], group_colors
):
    selected_chans = [
        i
        for i, ch in enumerate(eeg_montage.ch_names)
        if ch not in non_eeg_chans + sess_bad_chans
    ]
    selected_chans_names = [eeg_montage.ch_names[i] for i in selected_chans]

    chans_pos_xy = np.array(
        [
            v
            for k, v in eeg_montage.get_positions()["ch_pos"].items()
            if k in selected_chans_names
        ]
    )[:, :2]

    # * Select EEG channel groups to plot
    selected_chan_groups = {
        k: v for k, v in eeg_chan_groups.items() if k in group_names
    }

    group_colors = dict(zip(selected_chan_groups.keys(), group_colors))

    # * Get channel indices for each channel group
    ch_group_inds = {
        group_name: [
            i for i, ch in enumerate(selected_chans_names) if ch in group_chans
        ]
        for group_name, group_chans in selected_chan_groups.items()
    }

    return selected_chans_names, ch_group_inds, group_colors, chans_pos_xy


def inspect_results():
    # plt.close("all")
    # plt.get_backend()
    # plt.switch_backend("webagg")
    # plt.switch_backend(mpl_backend)

    res_dir = wd / "results/analyzed-banpdass[0.1-100]"  # /Oct27-Seq_and_choices"

    # for subj_dir in res_dir.glob("sub*"):
    # print(subj_dir.stem)
    # for sess_dir in subj_dir.glob("sess*"):
    # print("\t", sess_dir.stem)

    # * ################################################################################
    # * Load all the data
    # * ################################################################################

    erps = {}
    erp_files = sorted(res_dir.glob("sub*/sess*/*erps.pkl"))
    behav_files = sorted(data_dir.glob("sub*/sess*/*behav.csv"))

    # * ################################################################################
    # * Inspect ERP data
    # * ################################################################################

    ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
    ch_group_colors = ["red", "green", "blue", "pink", "orange"]

    eeg_baseline = 0.1

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

    all_subj_seq_erps = []
    all_subj_choices_erps = []
    all_subj_overall_erps = []

    subj_pattern_erps = {}
    all_subj_pattern_erps = {p: [] for p in patterns}

    for erp_file in sorted(erp_files):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])

        # erps[subj_N] = []
        subj_pattern_erps[subj_N] = {p: [] for p in patterns}

    for erp_file in tqdm(sorted(erp_files)):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        sess_N = int(erp_file.parents[0].stem.split("_")[-1])

        behav_file = list(
            data_dir.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )[0]
        raw_behav = pd.read_csv(behav_file, index_col=0)

        # subj_N = 1
        # sess_N = 4
        # print(subj_N, sess_N)

        # sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals = (
        #     load_raw_data(subj_N, sess_N, data_dir)
        # )

        sess_bad_chans = all_bad_chans[f"subj_{subj_N}"][f"sess_{sess_N}"]

        selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            prepare_eeg_data_for_plot(
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

        fig = plot_eeg(
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
    # * Inspect Gaze and Behavioral data
    # * ################################################################################
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

    all_subj_rdms = {k: None for k in rdms.keys()}

    for rdm_type, subjects_rdm in rdms.items():
        all_subj_rdms[rdm_type] = np.concat(
            [rdm.get_matrices()[0][None, :, :] for rdm in subjects_rdm.values()]
        ).mean(axis=0)

        plot_matrix(
            all_subj_rdms[rdm_type],
            title=rdm_type.replace("_", " ")
            .title()
            .replace("Rdms", "RDM")
            .replace("Erp", "ERP"),
            show_values=True,
            norm="max",
            as_pct=True,
        )

    # * Get the list of RDM types
    rdm_types = list(all_subj_rdms.keys())

    # * Initialize square DataFrames for Pearson and Spearman correlations
    df_pearson_corr_all_subj = pd.DataFrame(np.nan, index=rdm_types, columns=rdm_types)
    df_spearman_corr_all_subj = pd.DataFrame(np.nan, index=rdm_types, columns=rdm_types)

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

    dfs = [df_pearson_corr_all_subj, df_spearman_corr_all_subj]
    titles = [f"RSA - {m}\nGroup Level" for m in ["Pearson", "Spearman"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for df, title, ax in zip(dfs, titles, axes):
        # print(df.values)
        plot_matrix(
            df.values,
            labels=[c.replace("rdms_", "").replace("_", " ") for c in df.columns],
            show_values=True,
            title=title,
            cmap="RdBu_r",
            ax=ax,
        )
    plt.tight_layout()
    plt.show()

    return all_subj_pattern_erps


def draft():
    montage = mne.channels.make_standard_montage("biosemi64")
    chans_pos_xy = np.array(list(montage.get_positions()["ch_pos"].values()))[:, :2]

    # * Select EEG channel groups to plot
    selected_chan_groups = {
        k: v
        for k, v in eeg_chan_groups.items()
        if k
        in [
            "frontal",
            "parietal",
            "temporal",
            "occipital",
        ]
    }

    group_colors = dict(
        zip(selected_chan_groups.keys(), ["red", "green", "purple", "orange"])
    )

    ch_group_inds = {
        region: [i for i, ch in enumerate(montage.ch_names) if ch in ch_group]
        for region, ch_group in selected_chan_groups.items()
    }

    eeg_baseline = 0.1

    analyzed_res_dir = wd / "results/analyzed/with_ica_eye_removal"
    analyzed_res = {}
    for subj_dir in analyzed_res_dir.glob("*"):
        if subj_dir.is_dir():
            subj = subj_dir.stem
            for sess_dir in subj_dir.glob("*"):
                if sess_dir.is_dir():
                    sess = sess_dir.stem
                    for res_file in sess_dir.glob("*.pkl"):
                        res_name = res_file.stem
                        with open(res_file, "rb") as f:
                            res = pickle.load(f)
                        # print(res)
                        analyzed_res[(subj, sess, res_name)] = res

    analyzed_res.keys()
    analyzed_res.keys()

    for i in range(0, len(analyzed_res), 3):
        subj_sess1_erps = list(analyzed_res.values())[i]

        sequence_erps = subj_sess1_erps["sequence"]
        sequence_erps = [arr for arr in sequence_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in sequence_erps])
        sequence_erps = [arr[:, :min_len] for arr in sequence_erps]

        choice_erps = subj_sess1_erps["choices"]
        choice_erps = [arr for arr in choice_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in choice_erps])
        choice_erps = [arr[:, :min_len] for arr in choice_erps]

        set([arr.shape for arr in sequence_erps])
        np.stack(sequence_erps, axis=0).shape

        mean_sequence_erp = np.mean(np.stack(sequence_erps, axis=0), axis=0)
        mean_choices_erp = np.mean(np.stack(choice_erps, axis=0), axis=0)

        xticks = np.arange(
            0, mean_sequence_erp.shape[1] + 0.1 * eeg_sfreq, 0.1 * eeg_sfreq
        )
        xlabels = np.arange(
            -0.1, (mean_sequence_erp.shape[1] / eeg_sfreq) - 0.1 + 0.1, 0.1
        ).round(2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(mean_sequence_erp.T)
        ax1.set_xlim(0, mean_sequence_erp.shape[1])
        ax1.set_xticks(xticks, xlabels)

        ax2.plot(mean_choices_erp.T)
        ax2.set_xlim(0, mean_choices_erp.shape[1])
        ax2.set_xticks(xticks, xlabels)

        # for trial_erps in subj_sess1_erps:
        #     # i = 0 # ! TEMP
        #     trial_erps = [subj_sess1_erps["sequence"][i], subj_sess1_erps["choices"][i]]
        #     trial_erps = [
        #         np.mean(v, axis=0) if not np.isnan(v).all() else np.array([]) for v in trial_erps
        #     ]
        # trial_erps[0].shape

        # fig_titles = ["ERP - Sequence Icons", "ERP - Choice Icons"]

        # for eeg_data, title in zip(trial_erps, fig_titles):
        #     fig = plot_eeg(
        #         eeg_data * 1e6,
        #         chans_pos_xy,
        #         ch_group_inds,
        #         group_colors,
        #         eeg_sfreq,
        #         eeg_baseline,
        #         vlines=None,
        #         title=title,
        #     )


def draft2_behav():
    # from box import Box
    behav_files = list(data_dir.rglob("*behav*.csv"))
    behav_dfs = []

    for f in behav_files:
        df = pd.read_csv(f, index_col=0)
        df.insert(1, "sess", int(f.parent.stem.split("_")[-1]))
        df.reset_index(drop=False, inplace=True, names=["trial_N"])
        behav_dfs.append(df)

    group_behav_df = pd.concat(behav_dfs)
    group_behav_df.reset_index(drop=True, inplace=True)
    group_behav_df["correct"] = group_behav_df["correct"].astype(str)

    timeout_trials = group_behav_df.query("rt=='timeout'").copy()
    group_behav_df_clean_rt = group_behav_df.copy()

    group_behav_df_clean_rt["rt"] = (
        group_behav_df_clean_rt["rt"].replace({"timeout": np.nan}).astype(float)
    )

    group_behav_df_clean_correct = group_behav_df.copy()

    group_behav_df_clean_correct["correct"] = (
        group_behav_df_clean_correct["correct"]
        .replace({"invalid": False, "True": True, "False": False})
        .astype(bool)
    )
    group_behav_df_clean_correct["correct"].value_counts()

    group_behav_df_clean_rt.groupby("pattern")["rt"].mean().plot(kind="bar")
    group_behav_df_clean_correct.groupby("pattern")["correct"].mean().plot(kind="bar")

    def group_plot(rt_data, correct_data, rt_lim):
        fig, ax = plt.subplots(2, 1)
        rt_data.plot(kind="bar", ax=ax[0])
        correct_data.plot(kind="bar", ax=ax[1])

        ax[0].set_title(f"Mean RT per pattern (s)")
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)
        ax[0].grid(axis="y", ls="--")
        ax[0].set_ylim(0, rt_lim)
        ax[0].legend(
            title=ax[0].get_legend().get_title().get_text(),
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
        )

        ax[1].set_title(f"Mean accuracy per pattern")
        ax[1].grid(axis="y", ls="--")
        ax[1].set_ylim(0, 1)
        ax[1].legend(
            title=ax[1].get_legend().get_title().get_text(),
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
        )
        plt.tight_layout()
        # plt.show()
        return fig

    # * Goup Figure 1
    group_fig1 = group_plot(
        rt_data=(
            group_behav_df_clean_rt.groupby(["pattern", "sess"])["rt"].mean().unstack()
        ),
        correct_data=(
            group_behav_df_clean_correct.groupby(["pattern", "sess"])["correct"]
            .mean()
            .unstack()
        ),
        rt_lim=group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50,
    )

    # * Goup Figure 2
    group_fig2 = group_plot(
        rt_data=(
            group_behav_df_clean_rt.groupby(["pattern", "subj_id"])["rt"]
            .mean()
            .unstack()
        ),
        correct_data=(
            group_behav_df_clean_correct.groupby(["pattern", "subj_id"])["correct"]
            .mean()
            .unstack()
        ),
        rt_lim=group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50,
    )

    for participant in sorted(group_behav_df["subj_id"].unique()):
        fig, ax = plt.subplots(2, 1)
        # group_plot()

        group_behav_df_clean_rt.query("subj_id==@participant").groupby("pattern")[
            "rt"
        ].mean().sort_index().plot(kind="bar", ax=ax[0])
        # plt.show()
        group_behav_df_clean_correct.query("subj_id==@participant").groupby("pattern")[
            "correct"
        ].mean().sort_index().plot(kind="bar", ax=ax[1])

        ax[0].set_title(f"Participant {participant}\nMean RT per pattern (s)")
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)

        ax[0].grid(axis="y", ls="--")
        ax[1].grid(axis="y", ls="--")

        ax[0].set_ylim(
            0, group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50
        )
        ax[1].set_ylim(0, 1)
        ax[1].set_title(f"Mean accuracy per pattern")
        plt.tight_layout()
        plt.show()

        group_behav_df_clean_correct.query("subj_id==@participant").groupby("sess")[
            "correct"
        ].mean().plot(kind="bar")
        plt.show()

        group_behav_df_clean_correct.query("subj_id==@participant")


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
