# *
# TODO: Add a notch filter at 100 Hz (resonance of the power supply (50 Hz))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import shutil
import subprocess
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
import json
import matplotlib.image as mpimg
import os
from mne.preprocessing.eyetracking import read_eyelink_calibration
import matplotlib.patches as mpatches
import tomllib
from IPython.display import display
from typing import Union, List, Tuple, Dict, Any, Optional
from box import Box
import pickle
import pendulum
from loguru import logger
from box import Box
import plotly.graph_objs as go
from PIL import Image
import io
import base64
import re
import mplcursors
from scipy import signal as signal
from pyprep.find_noisy_channels import NoisyChannels

# * ####################################################################################
# * LOADING FILES
# * ####################################################################################

wd = Path(__file__).parent
os.chdir(wd)

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


exp_config_file = wd.parent / "config/experiment_config.toml"
anaysis_config_file = wd.parent / "config/analysis_config.toml"
data_dir = Path(
    f"/Users/chris/Documents/PhD-Local/PhD Data/Experiment 1/data & results/Lab/"
)

# * Load experiment config
with open(exp_config_file, "rb") as f:
    # with open(exp_config_file) as f:
    # exp_config = json.load(f)
    exp_config = Box(tomllib.load(f))

# * Load analysis config
with open(anaysis_config_file, "rb") as f:
    analysis_config = Box(tomllib.load(f))


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
    sequences = pd.read_csv(
        sequences_file, dtype={"choice_order": str, "seq_order": str}
    )

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

    return sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals


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

all_bad_chans = analysis_config.eeg.bad_channels
eog_chans = analysis_config.eeg.chans.eog
stim_chan = analysis_config.eeg.chans.stim
non_eeg_chans = eog_chans + [stim_chan]

eeg_chan_groups = analysis_config.eeg.ch_groups
eeg_sfreq = 2048


icon_images_dir = wd.parent / "experiment-Lab/images"
icon_images = {img.stem: mpimg.imread(img) for img in icon_images_dir.glob("*.png")}

x_pos_stim = analysis_config["stim"]["x_pos_stim"]

img_size = (256, 256)
screen_resolution = (2560, 1440)


# * Loading images
icon_images = (wd.parent / "experiment-Lab/images").glob("*.png")
icon_images = {img.stem: mpimg.imread(img) for img in icon_images}

y_pos_choices, y_pos_sequence = [-img_size[1], img_size[1]]

# * Getting Valid Event IDs
valid_events = exp_config["lab"]["event_IDs"]
valid_events_inv = {v: k for k, v in valid_events.items()}

# * Time bounds (seconds) for separating trials
# * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
pre_trial_time = 1
post_trial_time = 1


# * ####################################################################################
# * BEHAVIORAL ANALYSIS
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
        # epoch_data = raw_et.copy().crop(tmin=start_time, tmax=end_time)
        
        start_sample = et_events[start, 0]
        event = np.array([[start_sample, 0, valid_events['trial_start']]])
        
        epoch_data = mne.Epochs(
            raw_et, events=event, tmin=-pre_trial_time, tmax=end_time - (start_time + pre_trial_time), 
            baseline=None, verbose=False,
            )

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
    import numpy as np
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.offline as pyo
    from mne.channels import DigMontage
    from typing import Dict, List

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

    save_dir = wd / f"results/preprocessed_data/"
    save_dir.mkdir(exist_ok=True)

    preprocessed_raw_fpath = (
        save_dir / f"subj_{subj_N:02}{sess_N:02}_preprocessed-raw.fif"
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

        # * Bandpass Filter: 0.1 - 100 Hz
        raw_eeg.filter(l_freq=0.1, h_freq=100)
        raw_eeg.notch_filter(freqs=50)
        raw_eeg.notch_filter(freqs=100)

        # * drop bad channels
        # bad_chans = mne.preprocessing.find_bad_channels_lof(raw_eeg)
        # raw_eeg.plot()

        # raw_eeg.info["bads"] = bad_chans
        # raw_eeg.drop_channels(bad_chans)
        # logger.info(f"subj_{subj_N} - sess_{sess_N} - Bad channels: {bad_chans}")

        # * EOG artifact rejection using ICA *
        ica_fpath = save_dir / f"subj_{subj_N:02}{sess_N:02}_fitted-ica.fif"

        if ica_fpath.is_file():
            ica = mne.preprocessing.read_ica(ica_fpath)

        else:
            ica = mne.preprocessing.ICA(
                n_components=0.999999,
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

        # # * Bandpass Filter: 1 - 10 Hz
        # raw_eeg.filter(l_freq=1, h_freq=10)

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
    # events_ = mne.events_from_annotations(raw_eeg)
    # [raw_eeg.annotations.delete(idx) for idx in range(len(raw_eeg.annotations))]

    for start, end in tqdm(eeg_trial_bounds):
        # * Get start and end times in seconds
        start_time = (eeg_events[start, 0] / raw_eeg.info["sfreq"]) - pre_trial_time
        end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + post_trial_time
        
        # * Crop the raw data to this time window
        # epoch_data = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)
        
        start_sample = eeg_events[start, 0]
        event = np.array([[start_sample, 0, valid_events['trial_start']]])
        
        epoch_data = mne.Epochs(
            raw_eeg, events=event, tmin=-pre_trial_time, tmax=end_time - (start_time + pre_trial_time), 
            baseline=(None, 0), verbose=False,
            )
        
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
    et_sfreq = et_epoch.info["sfreq"]
    eeg_sfreq = eeg_epoch.info["sfreq"]

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
def generate_rdms_draft():
    # ! TEMP
    # * NOTE: Put following code into a function for better organization
    # * remove code block from function if it is not working
    # ! TEMP

    from scipy.spatial.distance import pdist, squareform

    pattern_group_inds = raw_behav.groupby("pattern").groups

    # * Dictionary to store RDMs for each pattern
    pattern_rdms = {}

    # * Initialize lists to collect data from all trials
    all_trial_vectors = []
    all_trial_labels = []  # * To keep track of patterns
    trial_indices = []  # * To keep track of trial indices

    for pattern, inds in tqdm(pattern_group_inds.items()):
        pattern_vectors = []
        for ind in inds:
            eeg_epoch = manual_eeg_epochs[ind]
            ep_annotations = eeg_epoch.annotations.to_data_frame()
            stim_pres_event = ep_annotations.query("description == 'stim-all_stim'")
            response_event = ep_annotations.query(
                "description.isin(['a', 'x', 'm', 'l'])"
            )
            trial_end_event = ep_annotations.query("description == 'trial_end'")

            ep_events, ep_events_dict = mne.events_from_annotations(
                eeg_epoch, verbose=False
            )

            stim_pres_event = ep_events[
                ep_events[:, 2] == ep_events_dict["stim-all_stim"]
            ]
            stim_pres_event = stim_pres_event[0, 0] - ep_events[0, 0]

            pressed_key = [
                k for k in ep_events_dict.keys() if k in ["a", "x", "m", "l"]
            ][0]
            response_event = ep_events[ep_events[:, 2] == ep_events_dict[pressed_key]]
            response_event = response_event[0, 0] - ep_events[0, 0]

            trial_end_event = ep_events[ep_events[:, 2] == ep_events_dict["trial_end"]]
            trial_end_event = trial_end_event[0, 0] - ep_events[0, 0]

            ep_eeg_data = eeg_epoch.get_data()
            cropped_eeg = ep_eeg_data[:, stim_pres_event : response_event + 1]

            mean_eeg_by_chans = cropped_eeg[:-5, :].mean(axis=1)

            pattern_vectors.append(mean_eeg_by_chans)

            all_trial_vectors.append(mean_eeg_by_chans)
            all_trial_labels.append(pattern)
            trial_indices.append(ind)

            # x_gaze, y_gaze, pupil_diameter = epoch.get_data()

        distances = pdist(pattern_vectors, metric="correlation")

        # * Convert to a square matrix (RDM)
        rdm = squareform(distances)

        # * Store the RDM for this pattern
        pattern_rdms[pattern] = rdm

    # * Convert list of trial vectors to a 2D array
    all_trial_vectors = np.array(all_trial_vectors)

    # * Compute pairwise dissimilarities between all trials
    distances = pdist(all_trial_vectors, metric="correlation")

    # * Convert to a square matrix (RDM)
    all_rdm = squareform(distances)

    # * Convert labels to a numpy array for easier indexing
    all_trial_labels = np.array(all_trial_labels)
    trial_indices = np.array(trial_indices)

    # * Get the unique patterns and assign colors
    unique_patterns = np.unique(all_trial_labels)
    pattern_colors = {
        pattern: plt.cm.tab10(i) for i, pattern in enumerate(unique_patterns)
    }

    # * Create a colorbar legend
    legend_handles = [
        mpatches.Patch(color=pattern_colors[pattern], label=pattern)
        for pattern in unique_patterns
    ]

    # * Create a figure to display the RDM
    plt.figure(figsize=(10, 8))
    plt.imshow(all_rdm, interpolation="nearest", cmap="viridis")
    plt.title("Combined RDM for All Patterns")
    plt.colorbar(label="Correlation Distance")
    plt.xlabel("Trial")
    plt.ylabel("Trial")

    # Add pattern color coding on the axes
    num_trials = len(all_trial_labels)
    ax = plt.gca()

    # Set ticks
    # ax.set_xticks(range(num_trials))
    # ax.set_yticks(range(num_trials))
    ax.set_xticks(range(0, num_trials, 10))
    ax.set_xticks(range(0, num_trials, 10))

    # * Optionally limit the number of ticks for readability
    # max_ticks = 20  # Adjust as needed
    # if num_trials > max_ticks:
    #     tick_locs = np.linspace(0, num_trials - 1, max_ticks, dtype=int)
    #     ax.set_xticks(tick_locs)
    #     ax.set_yticks(tick_locs)

    # * Assign labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ax.set_xticklabels(trial_indices[ax.get_xticks()])
    # ax.set_yticklabels(trial_indices[ax.get_yticks()])

    # * Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=90)

    # * Add color coding to ticks based on pattern
    # for tick_label in ax.get_xticklabels():
    #     idx = int(tick_label.get_text())
    #     pattern = all_trial_labels[trial_indices == idx][0]
    #     tick_label.set_color(pattern_colors[pattern])

    # for tick_label in ax.get_yticklabels():
    #     idx = int(tick_label.get_text())
    #     pattern = all_trial_labels[trial_indices == idx][0]
    #     tick_label.set_color(pattern_colors[pattern])

    # * Add legend
    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.15, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors

    # Sample data
    x1 = np.linspace(0, 10, 500)
    y1 = np.repeat([1], 500)

    x2 = np.linspace(10, 0, 500)
    y2 = np.repeat([2], 500)

    # Create line segments
    points = np.array([x1, y1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-4], points[4:]], axis=1)

    # Create a continuous norm to map data to colors
    cmap = plt.get_cmap("PuRd")
    norm = plt.Normalize(x1.min(), x1.max())

    # Create the LineCollection object
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x1)
    lc.set_linewidth(2)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(lc)
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(y1.min() - 0.1, y1.max() + 0.1)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Sine Wave with Gradient Color")

    # Add a colorbar
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("X-value")

    plt.show()

    x = np.concatenate([np.arange(0, 10, 1), np.arange(9, -1, -1), np.arange(0, 10, 1)])
    y = [0] * (len(x) // 3) + list(np.linspace(0, 1, len(x) // 3)) + [1] * (len(x) // 3)

    # Create a continuous norm to map data to colors
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(0, len(x))
    colors = cmap(norm(np.linspace(0, len(x), len(x)) * 0.5 + 10))

    plt.scatter(x, y, c=colors)
    plt.yticks(np.arange(-2, 4, 1))
    plt.show()


# * ####################################################################################
# * ERP TO ALL FLASHES
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


def analyze_decision_period_1(
    manual_eeg_trials: List[mne.io.Raw],
    manual_et_trials: List[mne.io.eyelink.eyelink.RawEyelink],
    raw_behav: pd.DataFrame,
    trial_N: int,
    eeg_baseline: float = 0.100,
    eeg_window: float = 0.600,
    show_plots: bool = True,
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

    et_trial.load_data()
    eeg_trial.load_data()
    
    eeg_sfreq = eeg_trial.info["sfreq"]

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
    duration = (time_bounds[1] + eeg_window + eeg_baseline) - time_bounds[0]
    sample_bounds = [None, None]
    sample_bounds[0] = int(time_bounds[0] * eeg_sfreq)
    sample_bounds[1] = sample_bounds[0] + int(np.ceil(duration * eeg_sfreq))

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

    for idx_fix, fixation_ind in enumerate(tqdm(fixation_inds, leave=False)):
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
        # slice_annotations = eeg_slice.annotations
        # mne.events_from_annotations(eeg_slice)
        
        dir(eeg_slice.annotations)
        
        times = eeg_slice.times
        eeg_slice = eeg_slice.get_data()
        
        # events_ = np.array(
        #     [[eeg_start_sample, 0, 61]]
        # )
        # ep_ = mne.Epochs(eeg_slice, events=events_, tmin=-0.1, tmax=0.6, baseline=(None, 0))
        # ep_.get_data()
        
        # TODO: check which is better: comment or uncomment below
        eeg_slice = mne.EpochsArray(
            [eeg_slice], eeg_trial.info, baseline=(None, 0), tmin=-0.1, verbose=False
        )

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
                eeg_data=eeg_slice * 1e6,
                et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                ch_names=[ch for ch in eeg_trial.ch_names if ch not in non_eeg_chans],
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

    # * GETTING ERPs

    eeg_fixation_data = {
        fixation_ind: mne.concatenate_epochs(data, verbose=False)
        for fixation_ind, data in eeg_fixation_data.items()
        if len(data) > 0
    }

    # TODO: check which is better: uncomment below, comment above, or vice versa
    # eeg_fixation_data = {
    #     fixation_ind: mne.EpochsArray(
    #         np.stack(data), eeg_trial.info, baseline=(None, 0), tmin=-0.1, verbose=False
    #     )
    #     for fixation_ind, data in eeg_fixation_data.items()
    #     if len(data) > 0
    #     # else np.array([])
    # }

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
        fixations_sequence_erp = np.array([])

    # * Calculate ERPs to fixations on choice icons
    if len(eeg_fixations_choices) > 0:
        fixations_choices_erp = mne.concatenate_epochs(
            list(eeg_fixations_choices.values()), verbose=False
        ).average()
    else:
        fixations_choices_erp = np.array([])

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
    import plotly.graph_objects as go
    import plotly.subplots as ps
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
    pass


# * ####################################################################################


def main():

    res_dir = wd / "results" / "analyzed"
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

        # for sess_N in range(1, len(list(subj_dir.glob("sess_*"))) + 1):
        for sess_N in [4]:
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
                et_sfreq = raw_et.info["sfreq"]
                eeg_sfreq = raw_eeg.info["sfreq"]
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
                mne.combine_evoked(sess_erps["sequence"], "equal").plot()

                    np.mean([erp.get_data() for erp in sess_erps["sequence"]], axis=0).T
                )

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


def inspect_results():

    res_dir = wd / "results/analyzed"  # /Oct27-Seq_and_choices"

    # for subj_dir in res_dir.glob("sub*"):
    # print(subj_dir.stem)
    # for sess_dir in subj_dir.glob("sess*"):
    # print("\t", sess_dir.stem)
    erp_files = sorted(res_dir.glob("sub*/sess*/*erps.pkl"))
    erps = {}

    for erp_file in sorted(erp_files):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        sess_N = int(erp_file.parents[0].stem.split("_")[-1])

        # subj_N = 1
        # sess_N = 4
        # print(subj_N, sess_N)

        # sess_info, sequences, raw_behav, raw_eeg, raw_et, et_cals = (
        #     load_raw_data(subj_N, sess_N, data_dir)
        # )

        montage = mne.channels.make_standard_montage("biosemi64")

        sess_bad_chans = all_bad_chans[f"subj_{subj_N}"][f"sess_{sess_N}"]

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
            zip(selected_chan_groups.keys(), ["red", "green", "blue", "pink", "orange"])
        )

        # * Get channel indices for each channel group
        ch_group_inds = {
            group_name: [
                i for i, ch in enumerate(selected_chans_names) if ch in group_chans
            ]
            for group_name, group_chans in selected_chan_groups.items()
        }

        with open(erp_file, "rb") as f:
            erps[subj_N] = {sess_N: pickle.load(f)}

        sequence_erps = erps[subj_N][sess_N]["sequence"]
        sequence_erps = [arr for arr in sequence_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in sequence_erps])
        sequence_erps = [arr[:, :min_len] for arr in sequence_erps]

        choice_erps = erps[subj_N][sess_N]["choices"]
        choice_erps = [arr for arr in choice_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in choice_erps])
        choice_erps = [arr[:, :min_len] for arr in choice_erps]

        mean_sequence_erp = np.mean(np.stack(sequence_erps, axis=0), axis=0)
        mean_choices_erp = np.mean(np.stack(choice_erps, axis=0), axis=0)
        mean_overall_erp = np.mean(
            np.stack(sequence_erps + choice_erps, axis=0), axis=0
        )

        eeg_baseline = 0.1

        for eeg_data, title in zip(
            [mean_sequence_erp, mean_choices_erp, mean_overall_erp],
            ["Sequence ERP", "Choices ERP", "Overall ERP"],
        ):
            fig = plot_eeg(
                eeg_data,
                chans_pos_xy,
                {
                    k: v for k, v in ch_group_inds.items() if k == "occipital"
                },  # ch_group_inds,
                group_colors,
                eeg_sfreq,
                eeg_baseline,
                vlines=None,
                title=f"Subj{subj_N}-Sess{sess_N}\n" + title,
            )

        for trial_N in range(len(erps[subj_N][sess_N]["sequence"])):
            # trial_N=19
            eeg_data = erps[subj_N][sess_N]["sequence"][trial_N]

            if eeg_data.shape[0] == 0:
                continue

            # plt.get_backend()
            # plt.switch_backend("webagg")
            # plt.switch_backend(mpl_backend)
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
                # plot_eeg=False,
                plot_eeg_group=False,
            )

            plt.show()
            plt.close("all")

            # plot_eeg_plotly_static(
            #     eeg_data,
            #     chans_pos_xy,
            #     ch_group_inds,
            #     group_colors,
            #     eeg_sfreq,
            #     eeg_baseline,
            #     vlines=None,
            #     title=f"Subj{subj_N}-Sess{sess_N}-Trial{trial_N + 1}",
            # )

    # * Inspect Eye data
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    # behav_files = list(data_dir.rglob("*behav.csv"))

    for i, file in enumerate(gaze_info_files):

        subj_N = int(file.parents[1].stem.split("_")[-1])
        sess_N = int(file.parents[0].stem.split("_")[-1])
        # behav_file = [f for f in behav_files if f"subj_{subj_N:02}/sess_{sess_N:02}" in str(f)][0]
        # behav_df = pd.read_csv(behav_file, index_col=0)
        # behav_df = behav_df[['rt', 'choice', 'correct', 'pattern', 'item_id']]

        gaze_df = pd.read_pickle(file)
        gaze_df["subj_N"] = subj_N
        gaze_df["sess_N"] = sess_N

        if i == 0:
            # print(df.columns)
            gaze_info = pd.DataFrame(columns=gaze_df.columns)

        gaze_info = pd.concat([gaze_info, gaze_df], axis=0)


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


# gaze_info.query("trial_N==0")
# gaze_info.query("trial_N==0").groupby("stim_ind")["mean_duration"].mean().plot(
#     kind="bar"
# )

# gaze_info.groupby("stim_ind")["mean_duration"].mean().plot(kind="bar")


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
