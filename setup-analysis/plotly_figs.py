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
import plotly.graph_objects as go
import numpy as np

pd.set_option("future.no_silent_downcasting", True)
mne.viz.set_browser_backend("qt")

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
# * ####################################################################################


def inspect_results():

    res_dir = wd / "results/analyzed/Oct27-Seq_and_choices"

    # for subj_dir in res_dir.glob("sub*"):
    # print(subj_dir.stem)
    # for sess_dir in subj_dir.glob("sess*"):
    # print("\t", sess_dir.stem)
    erp_files = list(res_dir.glob("sub*/sess*/*erps.pkl"))
    erps = {}

    for erp_file in sorted(erp_files):
        subj_N = int(erp_file.parents[1].stem.split("_")[-1])
        sess_N = int(erp_file.parents[0].stem.split("_")[-1])

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
                "temporal",
                "occipital",
            ]
        }

        group_colors = dict(
            zip(selected_chan_groups.keys(), ["red", "green", "purple", "orange"])
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
                ch_group_inds,
                group_colors,
                eeg_sfreq,
                eeg_baseline,
                vlines=None,
                title=f"Subj{subj_N}-Sess{sess_N}\n" + title,
            )

    # * Inspect Eye data
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))

    for i, file in enumerate(gaze_info_files):

        subj_N = int(file.parents[1].stem.split("_")[-1])
        sess_N = int(file.parents[0].stem.split("_")[-1])

        df = pd.read_pickle(file)
        df["subj_N"] = subj_N
        df["sess_N"] = sess_N

        if i == 0:
            # print(df.columns)
            gaze_info = pd.DataFrame(columns=df.columns)

        gaze_info = pd.concat([gaze_info, df], axis=0)


# * ####################################################################################
# * PLOTLY STATIC
# * ####################################################################################


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
    fig.write_image("eeg_plotly_plot.png")

    # * Save figure as an interactive HTML file
    fig.write_html("eeg_plotly_plot.html")

    # Show figure
    fig.update_layout(height=800, width=1000)
    fig.show()
    fig.update_layout(showlegend=False)


# * ####################################################################################
# * PLOTLY DASH
# * ####################################################################################
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# * Set up Dash application
app = dash.Dash(__name__)

# * Example data (placeholder)
eeg_data = np.random.randn(32, 500)  # 32 channels, 500 samples
eeg_sfreq = 500  # Sample frequency
eeg_baseline = 0.2  # Baseline period in seconds

# * Layout for Dash app
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="eeg-plot",
                    config={"displayModeBar": False},
                    style={
                        "width": "75%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                )
            ],
            style={"width": "75%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="channel-selector",
                    options=[
                        {"label": f"Channel {i+1}", "value": f"Channel {i+1}"}
                        for i in range(eeg_data.shape[0])
                    ],
                    value=[],
                    multi=True,
                    placeholder="Select channels...",
                    searchable=True,
                    style={"height": "50px", "width": "100%", "overflowY": "scroll"},
                )
            ],
            style={
                "width": "20%",
                "display": "inline-block",
                "vertical-align": "top",
                "padding-left": "20px",
            },
        ),
    ]
)


# * Initial figure setup
def create_figure(selected_channels=[]):
    # * Set up figure for EEG data
    fig = go.Figure()

    # * Add EEG time-series for each channel
    for i in range(eeg_data.shape[0]):
        color = "red" if f"Channel {i+1}" in selected_channels else "black"
        fig.add_trace(
            go.Scatter(
                y=eeg_data[i],
                mode="lines",
                name=f"Channel {i+1}",
                hoverinfo="y",
                line=dict(color=color),
            )
        )

    # * Set x-axis properties
    tick_step_time = 0.05
    tick_step_sample = tick_step_time * eeg_sfreq
    x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)
    x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)

    fig.update_xaxes(
        tickvals=x_ticks,
        ticktext=x_labels,
        title_text="Time (ms) relative to gaze fixation onset",
    )

    # * Set title and layout properties
    fig.update_layout(
        title="EEG Time-Series Plot",
        height=600,
        width=1000,
        showlegend=False,
        hovermode="x unified",
    )

    return fig


# * Callback to update figure based on selected channels
@app.callback(Output("eeg-plot", "figure"), [Input("channel-selector", "value")])
def update_figure(selected_channels):
    return create_figure(selected_channels=selected_channels)


# * Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=True)
