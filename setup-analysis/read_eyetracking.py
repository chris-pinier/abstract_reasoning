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

# res_dir = r"C:\Users\topuser\Documents\ChrisPinier\experiment1-new\experiment-Lab\results\raw"
# res_dir = Path(res_dir)
et_fpath = r"C:\Users\topuser\Documents\ChrisPinier\experiment1-new\experiment-Lab\results\raw\subj_000\sess_01\cp000.asc"
eeg_fpath = r"C:\Users\topuser\Documents\ChrisPinier\experiment1-new\experiment-Lab\results\raw\subj_000\sess_01\cp-testing2.bdf"

raw_et = mne.io.read_raw_eyelink(
    et_fpath,
)  # , create_annotations=["blinks"])
# raw_eeg = mne.io.read_raw_egi(eeg_fpath, preload=True, verbose="warning")


cals = read_eyelink_calibration(et_fpath)
print(f"number of calibrations: {len(cals)}")
first_cal = cals[0]  # let's access the first (and only in this case) calibration
print(first_cal)

raw_et.plot(duration=0.5, scalings=dict(eyegaze=1e3))
raw_et.annotations

et_events = mne.find_events(raw_et, min_duration=0.01, shortest_event=1, uint_cast=True)

mne.events_from_annotations()


events, events_dict = mne.events_from_annotations(raw_et)

epochs = mne.Epochs(
    raw_et, events, event_repeated="merge", event_id=events_dict
)  # event_id=dict(GOOD=1), tmin=0, tmax=10, preload=True)

epochs["stim-all_stim"].plot()

dir(raw_et)

raw_et.ch_names

np.nanmin(raw_et["xpos_right"][0][0])
np.nanmax(raw_et["xpos_right"][0][0])

import matplotlib.pyplot as plt

t = 30000
plt.plot(raw_et["xpos_right"][0][0][:t], raw_et["ypos_right"][0][0][:t])
t1, t2 = 30000, 34000
plt.plot(raw_et["xpos_right"][0][0][t1:t2], raw_et["ypos_right"][0][0][t1:t2])


# * ################ EEG
raw_eeg = mne.io.read_raw_bdf(eeg_fpath)
# raw_eeg.plot()
eeg_events = mne.find_events(
    raw_eeg, min_duration=0.01, initial_event=False, shortest_event=1, uint_cast=True
)

eeg_events -= 4096

valid_events_inv = {v: k for k, v in valid_events.items()}
[valid_events_inv.get(i) for i in eeg_events[:, 2]]
sorted([int(i) for i in np.unique(eeg_events[:, 2])])
