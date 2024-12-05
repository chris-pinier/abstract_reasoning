import mne
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration
from mne.viz.eyetracking import plot_gaze
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from typing import List, Tuple, Dict
from pathlib import Path
import os
import json
from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap

import subprocess
import shutil


pd.set_option("future.no_silent_downcasting", True)
mne.viz.set_browser_backend("qt")

# * ####################################################################################
# * LOADING FILES
# * ####################################################################################
wd = Path(__file__).parent
os.chdir(wd)

# subj = "subj_92"
subj = "subj_02"
sess_N = 1

config_file = wd.parent / "config/experiment_config.json"
# sess_dir = wd.parent / f"experiment-Lab/results/raw/{subj}/sess_0{sess_N}"
sess_dir = Path(
    f"/Users/chris/Documents/PhD-Local/PhD Data/Experiment 1/data & results/Lab/{subj}/sess_0{sess_N}"
)

# * Load experiment config
with open(config_file) as f:
    exp_config = json.load(f)

# * File paths
et_fpath = [f for f in sess_dir.glob("*.asc")][0]
eeg_fpath = [f for f in sess_dir.glob("*.bdf")][0]
behav_fpath = [f for f in sess_dir.glob("*behav*.csv")][0]
sess_info_file = [f for f in sess_dir.glob("*sess_info.json")][0]
sequences_file = wd.parent / f"experiment-Lab/sequences/session_{sess_N}.csv"


# * Load data
sess_info = json.load(open(sess_info_file))
sequences = pd.read_csv(sequences_file)

raw_behav = pd.read_csv(behav_fpath).merge(sequences, on="item_id")

raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=False)

raw_et = mne.io.read_raw_eyelink(et_fpath)
et_cals = read_eyelink_calibration(et_fpath)


# * Drop unnecessary columns
raw_behav.drop(columns=["Unnamed: 0"], inplace=True)
raw_behav.drop(columns=["pattern_x", "solution_x", "trial_type_x"], inplace=True)
raw_behav.rename(
    columns={
        "pattern_y": "pattern",
        "solution_y": "solution",
        "trial_type_y": "trial_type",
    },
    inplace=True,
)


# * ####################################################################################
# * GLOBAL VARS
# * ####################################################################################
screen_resolution = sess_info["window_size"]
img_size = sess_info["img_size"]
et_sfreq = raw_et.info["sfreq"]
eeg_sfreq = raw_eeg.info["sfreq"]
# valid_events =
tracked_eye = sess_info["eye"]
# vision_correction = sess_info["vision_correction"] # TODO: IMPLEMENT IN EXPERIMENT SCRIPT
# screen_distance = sess_info["screen_distance"]

icon_images_dir = wd.parent / "experiment-Lab/images"
icon_images = {img.stem: mpimg.imread(img) for img in icon_images_dir.glob("*.png")}

x_pos_stim = {
    "items_set": [
        -1042.222,
        -744.444,
        -446.667,
        -148.889,
        148.889,
        446.667,
        744.444,
        1042.222,
    ],
    "avail_choice": [
        -446.66700000000003,
        -148.88900000000012,
        148.88900000000012,
        446.6669999999999,
    ],
}

y_pos_choices, y_pos_sequence = [-img_size[1], img_size[1]]


valid_events = exp_config["lab"]["event_IDs"]
valid_events_inv = {v: k for k, v in valid_events.items()}


# * ####################################################################################
# * BEHAVIORAL ANALYSIS
# * ####################################################################################


def get_score_and_rt(
    raw_behav: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:

    tmp = raw_behav[["choice_key", "choice", "solution", "correct", "rt", "pattern"]]

    # * Remove trials with invalid key presses
    invalid_press_trials = tmp.query("correct == 'invalid' & rt != 'timeout'")
    tmp = tmp.drop(invalid_press_trials.index)

    # * Convert `correct` column to boolean
    tmp["correct"] = tmp["correct"].replace(
        {"True": True, "False": False, "invalid": False}
    )
    correct_trials = tmp.query("correct == True")
    score_global = np.float64(correct_trials.shape[0] / tmp.shape[0])

    print(f"Overall accuracy: {score_global}")

    # * Remove trials with timeout
    timeout_trials = tmp.query("rt == 'timeout'")
    tmp = tmp.drop(timeout_trials.index)
    score_valid_rt = np.float64(correct_trials.shape[0] / tmp.shape[0])

    print(f"Accuracy on valid RT trials (without timeout): {score_valid_rt}")

    # * Convert RTs to float
    tmp["rt"] = tmp["rt"].astype("float")

    # * Calculate mean RTs
    rt_global = tmp["rt"].mean()
    rt_by_pattern_and_correct = tmp.groupby(["pattern", "correct"])["rt"].mean()
    rt_by_pattern = tmp.groupby(["pattern"])["rt"].mean()
    rt_by_correct = tmp.groupby(["correct"])["rt"].mean()

    score = {k: eval("score_" + k).round(4) for k in ["global", "valid_rt"]}

    rt = {
        k: eval("rt_" + k).round(4)
        for k in ["global", "by_pattern_and_correct", "by_pattern", "by_correct"]
    }

    cleaned_df = tmp
    return cleaned_df, score, rt


cleaned_df, score, rt = get_score_and_rt(raw_behav)

# * ####################################################################################
# * GROUP ANALYSIS
# * ####################################################################################
data_dir = Path(
    f"/Users/chris/Documents/PhD-Local/PhD Data/Experiment 1/data & results/Lab"
)

behav_files = list(data_dir.rglob("*-behav.csv"))
behav_files = {f"{f.parents[1].name}{f.parents[0].name[4:]}": f for f in behav_files}

behav_group = pd.concat([pd.read_csv(f) for f in behav_files.values()])
behav_group.drop(columns=["Unnamed: 0"], inplace=True)

timeout_trials = behav_group.query("rt == 'timeout'")
correct_df = behav_group.copy()
correct_df.loc[timeout_trials.index, "correct"] = "False"
correct_df["correct"] = correct_df["correct"].replace({"True": True, "False": False})
correct_df["correct"] = correct_df["correct"].astype(bool)

score_per_subj = correct_df.groupby("subj_id")["correct"].mean()

score_per_subj_and_pattern = (
    correct_df.groupby(["pattern", "subj_id"])["correct"].mean().unstack()
)
score_per_subj_and_pattern = score_per_subj_and_pattern.round(2)

# * #### SCORE ####
# * Step 1: Add a 'Mean' row at the end for each subject (column)
score_per_subj_and_pattern.loc["Mean"] = score_per_subj_and_pattern.mean()

# * Step 2: Add a 'Mean' column at the end for each pattern (row)
score_per_subj_and_pattern["Mean"] = score_per_subj_and_pattern.mean(axis=1)

# * Step 3 (Optional): Add a grand mean at the bottom-right corner
score_per_subj_and_pattern.at["Mean", "Mean"] = score_per_subj_and_pattern[
    "Mean"
].mean()

# * #### RESPONSE TIME ####
rt_df = correct_df.drop(timeout_trials.index)
rt_df["rt"] = rt_df["rt"].astype(float)

rt_per_subj = rt_df.groupby("subj_id")["rt"].mean()
rt_per_subj_and_pattern = rt_df.groupby(["pattern", "subj_id"])["rt"].mean().unstack()

# * Step 1: Add a 'Mean' row at the end for each subject (column)
rt_per_subj_and_pattern.loc["Mean"] = rt_per_subj_and_pattern.mean()

# * Step 2: Add a 'Mean' column at the end for each pattern (row)
rt_per_subj_and_pattern["Mean"] = rt_per_subj_and_pattern.mean(axis=1)

# * Step 3 (Optional): Add a grand mean at the bottom-right corner
rt_per_subj_and_pattern.at["Mean", "Mean"] = rt_per_subj_and_pattern["Mean"].mean()

# * Combine score and RT per subject
score_and_rt_per_subj = pd.concat([score_per_subj, rt_per_subj], axis=1)
score_and_rt_per_subj.loc["Mean"] = score_and_rt_per_subj.mean()

# * Save to CSV / Excel
score_and_rt_per_subj.to_csv(wd / "score_and_rt_per_subj.csv")
score_per_subj_and_pattern.to_csv(wd / "score_per_subj_and_pattern.csv")
rt_per_subj_and_pattern.to_csv(wd / "rt_per_subj_and_pattern.csv")

for df in [
    "score_and_rt_per_subj",
    "score_per_subj_and_pattern",
    "rt_per_subj_and_pattern",
]:
    eval(df).style.background_gradient(cmap="viridis", axis=1).to_excel(
        wd / f"{df}.xlsx"
    )

figsize = (8, 8)
dpi = 150

plt.figure(figsize=figsize, dpi=dpi)
score_and_rt_per_subj["correct"].plot(kind="bar")
plt.title("Accuracy per subject")
plt.grid(axis="y", linestyle="--", alpha=0.65)
plt.tight_layout()
plt.savefig(wd / "score_per_subj.png")


plt.figure(figsize=figsize, dpi=dpi)
score_and_rt_per_subj["rt"].plot(kind="bar")
plt.title("RT per subject")
plt.grid(axis="y", linestyle="--", alpha=0.65)
plt.tight_layout()
plt.savefig(wd / "rt_per_subj.png")

plt_type = "barh"

plt.figure(figsize=figsize, dpi=dpi)
score_per_subj_and_pattern.plot(kind=plt_type).legend(
    loc="center left", bbox_to_anchor=(1, 0.5)
)
plt.title("Accuracy per subject and pattern")
if plt_type == "barh":
    plt.grid(axis="x", linestyle="--", alpha=0.65)
elif plt_type == "bar":
    plt.grid(axis="y", linestyle="--", alpha=0.65)
plt.tight_layout()
plt.savefig(wd / "score_per_subj_and_pattern.png")

plt.figure(figsize=figsize, dpi=dpi)
rt_per_subj_and_pattern.plot(kind=plt_type).legend(
    loc="center left", bbox_to_anchor=(1, 0.5)
)
plt.title("RT per subject and pattern")
if plt_type == "barh":
    plt.grid(axis="x", linestyle="--", alpha=0.65)
elif plt_type == "bar":
    plt.grid(axis="y", linestyle="--", alpha=0.65)
plt.tight_layout()
plt.grid(axis="x", linestyle="--", alpha=0.65)
plt.savefig(wd / "rt_per_subj_and_pattern.png")
