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
valid_events_inv = {v:k for k,v in valid_events.items()}

# res_dir = r"C:\Users\topuser\Documents\ChrisPinier\experiment1-new\experiment-Lab\results\raw"
# res_dir = Path(res_dir)
sess_dir = wd.parent / "experiment-Lab/results/raw/subj_00/sess_03/"
et_fpath = sess_dir / "cp0003.asc"
eeg_fpath = sess_dir / "cp0003.bdf"


# * #################### EYE TRACKING ####################

raw_et = mne.io.read_raw_eyelink(et_fpath)  # , create_annotations=["blinks"])

cals = read_eyelink_calibration(et_fpath)
print(f"number of calibrations: {len(cals)}")
first_cal = cals[0]  # let's access the first (and only in this case) calibration
print(first_cal)

raw_et.plot(duration=0.5, scalings=dict(eyegaze=1e3))
raw_et.annotations


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

# Update et_events_dict with new IDs
et_events_dict = {k: id_mapping[v] for k, v in et_events_dict.items()}
et_events_dict_inv = {v:k for k,v in et_events_dict.items()}

# print("ID Mapping:", id_mapping)
# print("Updated et_events_dict:", et_events_dict)
# print("Unique event IDs after update:", np.unique(et_events[:, 2]))
np.where(et_events[:, 2] == valid_events['trial_start'])[0].shape
np.where(et_events[:, 2] == valid_events['trial_end'])[0].shape

# [valid_events_inv[i] for i in et_events[:, 2] if i in valid_events_inv]

epochs = mne.Epochs(
    raw_et, et_events, event_repeated="merge", event_id=et_events_dict
)  # event_id=dict(GOOD=1), tmin=0, tmax=10, preload=True)

epochs["stim-all_stim"].plot()

print(f"{raw_et.ch_names = }")
chan_xpos, chan_ypos, chan_pupil = raw_et.ch_names
x_pos = raw_et[chan_xpos][0]
y_pos = raw_et[chan_ypos][0]

np.nanmin(x_pos[0])
np.nanmax(y_pos[0])


import matplotlib.pyplot as plt

t = 30000
plt.plot(raw_et["xpos_right"][0][0][:t], raw_et["ypos_right"][0][0][:t])
t1, t2 = 30000, 34000
plt.plot(raw_et["xpos_right"][0][0][t1:t2], raw_et["ypos_right"][0][0][t1:t2])


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
