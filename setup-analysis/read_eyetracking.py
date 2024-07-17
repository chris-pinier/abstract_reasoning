import mne
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration
from mne.viz.eyetracking import plot_gaze
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

mne.viz.set_browser_backend("qt")
from pathlib import Path
import os
import json

wd = Path(__file__).parent
os.chdir(wd)
config_dir = wd.parent / "config"

# * Load experiment config
with open(config_dir / "experiment_config.json") as f:
    exp_config = json.load(f)

valid_events = exp_config["local"]["event_IDs"]
valid_events_inv = {v: k for k, v in valid_events.items()}

# res_dir = r"C:\Users\topuser\Documents\ChrisPinier\experiment1-new\experiment-Lab\results\raw"
# res_dir = Path(res_dir)
sess_dir = wd.parent / "experiment-Lab/results/raw/subj_00/sess_03/"
et_fpath = sess_dir / "cp0003.asc"
eeg_fpath = sess_dir / "cp0003.bdf"

# * #################### EYE TRACKING ####################
screen_resolution = exp_config["local"]["monitor"]["resolution"]

raw_et = mne.io.read_raw_eyelink(et_fpath)  # , create_annotations=["blinks"])

cals = read_eyelink_calibration(et_fpath)
print(f"number of calibrations: {len(cals)}")
first_cal = cals[0]  # let's access the first (and only in this case) calibration
print(first_cal)

# raw_et.plot(duration=0.5, scalings=dict(eyegaze=1e3))
# raw_et.annotations

print(f"{raw_et.ch_names = }")
chan_xpos, chan_ypos, chan_pupil = raw_et.ch_names
x_pos = raw_et[chan_xpos][0][0]
y_pos = raw_et[chan_ypos][0][0]

# Read events from annotations
et_events, et_events_dict = mne.events_from_annotations(raw_et)
print("Unique event IDs before update:", np.unique(et_events[:, 2]))

# Convert keys to strings (if they aren't already)
et_events_dict = {str(k): v for k, v in et_events_dict.items()}

# Create a mapping from old event IDs to new event IDs
id_mapping = {}
eye_events_idx = 60

for event_name, event_id in et_events_dict.items():
    if event_name in valid_events:
        new_id = valid_events[event_name]
    else:
        eye_events_idx += 1
        new_id = eye_events_idx
    id_mapping[event_id] = new_id

# Update event IDs in et_events
for i in range(et_events.shape[0]):
    old_id = et_events[i, 2]
    if old_id in id_mapping:
        et_events[i, 2] = id_mapping[old_id]

print("Unique event IDs after update:", np.unique(et_events[:, 2]))


# Update et_events_dict with new IDs
et_events_dict = {k: id_mapping[v] for k, v in et_events_dict.items()}
et_events_dict_inv = {v: k for k, v in et_events_dict.items()}

inds_responses = np.where(np.isin(et_events[:, 2], [11, 12, 13, 14, 61, 64]))
responses = et_events[inds_responses]
n_responses = int(np.sum(np.diff(inds_responses) > 2))
print(f"{n_responses = }")

# print("ID Mapping:", id_mapping)
# print("Updated et_events_dict:", et_events_dict)
# print("Unique event IDs after update:", np.unique(et_events[:, 2]))


# Find trial start and end events
trial_start_inds = np.where(et_events[:, 2] == valid_events["trial_start"])[0]
choice_onset_inds = np.where(et_events[:, 2] == valid_events["stim-all_stim"])[0]
trial_end_inds = np.where(et_events[:, 2] == valid_events["trial_aborted"])[0]

choice_onset_inds = choice_onset_inds[:-1]  # ! TEMP

trial_start_inds = trial_start_inds[:-1]  # ! TEMP

# Ensure we have matching pairs of start and end events
assert len(trial_start_inds) == len(
    trial_end_inds
), "Mismatch in number of start and end events"


# Create a list to store our manual epochs
manual_epochs = []

# Loop through each trial
for start, end in zip(trial_start_inds, trial_end_inds):
    # for start, end in zip(choice_onset_inds, trial_end_inds):
    # for start, end in zip(trial_start_inds, choice_onset_inds):
    # Get start and end times in seconds
    start_time = et_events[start, 0] / raw_et.info["sfreq"] - 0.5
    end_time = et_events[end, 0] / raw_et.info["sfreq"]  # + 0.5

    # Crop the raw data to this time window
    epoch_data = raw_et.copy().crop(tmin=start_time, tmax=end_time)

    # Add this epoch to our list
    manual_epochs.append(epoch_data)

# Print some information about our epochs
print(f"Number of epochs created: {len(manual_epochs)}")
for i, epoch in enumerate(manual_epochs):
    print(f"Epoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

# manual_epochs[0].plot()


# If you want to combine these into a single Epochs object:
combined_epochs = mne.concatenate_raws(manual_epochs)

screen_resolution = (2560, 1440)
img_size = [213, 213]

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

from matplotlib.patches import Rectangle


def get_stim_screen(
    screen_resolution, img_size, x_pos_stim, y_pos_choices, y_pos_sequence
):
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the limits of the plot to match the screen resolution
    ax.set_xlim(0, screen_resolution[0])
    ax.set_ylim(0, screen_resolution[1])

    # Invert y-axis because screen coordinates typically have (0,0) at top-left
    ax.invert_yaxis()

    # Function to add image rectangle
    def add_image_rect(x, y, color):
        rect = Rectangle(
            xy=(
                x - img_size[0] / 2 + screen_resolution[0] / 2,
                -y - img_size[1] / 2 + screen_resolution[1] / 2,
            ),
            width=img_size[0],
            height=img_size[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

    # Plot stimulus positions
    # Top row (sequence)
    for x in x_pos_stim["items_set"]:
        add_image_rect(x, y_pos_sequence, "blue")

    # Bottom row (choices)
    for x in x_pos_stim["avail_choice"]:
        add_image_rect(x, y_pos_choices, "red")

    return fig


fig = get_stim_screen(
    screen_resolution, img_size, x_pos_stim, y_pos_choices, y_pos_sequence
)


def plot_eye_movements(
    epoch,
    tracked_eye,
    x_pos_stim,
    y_pos_choices,
    y_pos_sequence,
    screen_resolution,
    img_size,
):
    # Extract x and y gaze positions
    x_gaze = epoch[f"xpos_{tracked_eye}"][0][0]
    y_gaze = epoch[f"ypos_{tracked_eye}"][0][0]

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the limits of the plot to match the screen resolution
    ax.set_xlim(0, screen_resolution[0])
    ax.set_ylim(0, screen_resolution[1])

    # Invert y-axis because screen coordinates typically have (0,0) at top-left
    ax.invert_yaxis()

    # Function to add image rectangle
    def add_image_rect(x, y, color):
        rect = Rectangle(
            xy=(
                x - img_size[0] / 2 + screen_resolution[0] / 2,
                y - img_size[1] / 2 + screen_resolution[1] / 2,
            ),
            width=img_size[0],
            height=img_size[1],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(rect)

    # Plot stimulus positions
    # Top row (sequence)
    for x in x_pos_stim["items_set"]:
        # - y_pos_sequence because we inverted the y-axis
        add_image_rect(x, -y_pos_sequence, "blue")

    # Bottom row (choices)
    for x in x_pos_stim["avail_choice"]:
        # - y_pos_choices because we inverted the y-axis
        add_image_rect(x, -y_pos_choices, "red")

    # Plot eye movements
    # ax.plot(x_gaze + screen_resolution[0]/2, y_gaze + screen_resolution[1]/2, 'k-', alpha=0.5)

    # scatter = ax.scatter(x_gaze + screen_resolution[0]/2, y_gaze + screen_resolution[1]/2,)
    #  c=np.arange(len(x_gaze)), cmap='viridis', s=5)
    scatter = ax.scatter(x_gaze, y_gaze, c=np.arange(len(x_gaze)), cmap="viridis")

    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time (samples)", rotation=270, labelpad=15)

    # # Set labels and title
    ax.set_xlabel("X position (pixels)")
    ax.set_ylabel("Y position (pixels)")
    ax.set_title(
        f'Eye Movements (Trial duration: {len(x_gaze)/epoch.info["sfreq"]:.2f} s)'
    )

    plt.tight_layout()

    return fig


tracked_eye = "left"

epoch = manual_epochs[1]
epoch.times.shape

np.where(epoch.annotations.description == "stim-all_stim")[0]
fig = plot_eye_movements(
    epoch,
    tracked_eye,
    x_pos_stim,
    y_pos_choices,
    y_pos_sequence,
    screen_resolution,
    img_size,
)

# Now let's plot for each epoch
eye_mvmts_figs = []
for i, epoch in enumerate(manual_epochs):
    fig = plot_eye_movements(
        epoch,
        tracked_eye,
        x_pos_stim,
        y_pos_choices,
        y_pos_sequence,
        screen_resolution,
        img_size,
    )
    plt.savefig(f"eye_movements_trial_{i+1}.png", dpi=300)
    plt.close(fig)
    eye_mvmts_figs.append(fig)
# * #################### EEG  ####################

raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=True)

eeg_events = mne.find_events(
    raw_eeg, min_duration=0.01, initial_event=False, shortest_event=1, uint_cast=True
)

# eeg_events -= 4096

raw_eeg.plot(events=eeg_events)
valid_events_inv = {v: k for k, v in valid_events.items()}
[valid_events_inv.get(i) for i in eeg_events[:, 2]]
[valid_events_inv[int(i)] for i in np.unique(eeg_events[:, 2])]

# pd.DataFrame(eeg_events[np.isin(eeg_events[:, 2], [10, 11,12,13,14])])


# np.where(eeg_events[:, 2] == valid_events["m"])[0].shape
# np.where(et_events[:, 2] == valid_events["m"])[0].shape

np.unique(et_events[:, 2], return_counts=True)


def align_eeg_et(eeg_events, et_events):
    start_eeg = np.where(eeg_events[:, 2] == valid_events["exp_start"])[0]
    start_et = np.where(et_events[:, 2] == valid_events["exp_start"])[0]

    end_eeg = np.where(eeg_events[:, 2] == valid_events["experiment_end"])[0]
    end_et = np.where(et_events[:, 2] == valid_events["experiment_end"])[0]
    np.unique(et_events[:, 2], return_counts=True)


########################################################################################
import mne

# Path to your EEG data
raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=True)
raw = raw_eeg.copy()
fsaverage_path = mne.datasets.fetch_fsaverage(verbose=True)
print("Fsaverage is located at:", fsaverage_path)

# Drop non-EEG channels
non_eeg_channels = ["EMG1", "EMG2", "EMG3", "EMG4", "EOGL", "EOGR", "EOGT", "EOGB"]
raw.drop_channels(non_eeg_channels)

# Load the standard head model from FreeSurfer's 'fsaverage'
# subjects_dir = mne.datasets.sample.data_path() / "subjects"
subjects_dir = Path(fsaverage_path).parent
subject = "fsaverage"

# Set montage for EEG
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# Create or load forward model using fsaverage as surrogate
src = mne.setup_source_space(
    subject, spacing="oct6", subjects_dir=subjects_dir, add_dist=False
)

model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
fwd = mne.make_forward_solution(
    raw.info, trans="fsaverage", src=src, bem=bem, eeg=True, mindist=5.0
)

# Compute inverse operator
inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, forward=fwd, noise_cov=None
)

# Compute inverse solution
method = "dSPM"  # dynamic Statistical Parametric Mapping
snr = 3.0
lambda2 = 1.0 / snr**2
stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2, method=method)

# Visualize
brain = stc.plot(
    subjects_dir=subjects_dir,
    subject=subject,
    initial_time=0.1,
    hemi="both",
    views="lateral",
    size=(800, 400),
    time_viewer=True,
)
