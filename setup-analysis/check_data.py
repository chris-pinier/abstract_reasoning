import os
from pathlib import Path
import pendulum.tz

WD = Path(__file__).parent
os.chdir(WD)
from analysis_main import (
    load_raw_data,
    preprocess_eeg_data,
    preprocess_et_data,
    crop_et_trial,
    get_trial_info,
    is_fixation_on_target,
    load_and_clean_behav_data,
)
from analysis_utils import check_et_calibrations, check_notes
from directory_tree import DisplayTree
from tqdm.auto import tqdm
import pendulum
import tomllib
from box import Box
import pandas as pd
import mne
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import contextlib
import numpy as np
from mne.preprocessing.eyetracking import read_eyelink_calibration
import io

# * ################################################################################
# * GLOBAL VARIABLES
EXP_CONFIG_FILE = WD.parent / "config/experiment_config.toml"
ANAYSIS_CONFIG_FILE = WD.parent / "config/analysis_config.toml"

# DATA_DIR = Path("/Users/chris/Documents/PhD-Local/PhD Data/experiment1/data/Lab")
DATA_DIR = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab")

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

# * Time bounds (seconds) for separating trials
# * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
PRE_TRIAL_TIME = 1
POST_TRIAL_TIME = 1

# * ################################################################################

# # * Get positions and presentation order of stimuli
# trial_info = trial_info = get_trial_info(
#     epoch_N,
#     raw_behav,
#     x_pos_stim,
#     y_pos_choices,
#     y_pos_sequence,
#     screen_resolution,
#     img_size,
# )


# stim_pos, stim_order = trial_info[:2]


def get_all_sess_info():
    sessions_info = pd.DataFrame()
    for subj_N_dir in sorted(DATA_DIR.glob("*subj*")):
        for sess_N_dir in sorted(subj_N_dir.glob("*sess*")):
            try:
                info_file = next(sess_N_dir.glob("*-sess_info.json"))

                with open(info_file, "rb") as f:
                    sess_info = json.load(f)

                sess_info["window_size"] = [sess_info["window_size"]]
                sess_info["img_size"] = [sess_info["img_size"]]
                sess_info["date"] = pendulum.from_format(
                    sess_info["date"], "YYYYMMDD_HHmmss", tz=TIMEZONE
                )

                sessions_info = pd.concat(
                    [sessions_info, pd.DataFrame.from_dict(sess_info)]
                )
            except Exception as e:
                subj_N = subj_N_dir.name
                sess_N = sess_N_dir.name
                print(f"{subj_N} - {sess_N}: an error occured, details below")
                raise e

    sessions_info["days_since_last"] = sessions_info.groupby("subj_id")["date"].agg(
        "diff"
    )

    return sessions_info


def analyze_ET_decision_period(
    et_trial: mne.io.eyelink.eyelink.RawEyelink,
    raw_behav: pd.DataFrame,
    trial_N: int,
    # show_plots: bool = True,
    pbar_off=True,
    min_fix_duration=0.1,
):
    """_summary_

    Args:
        et_trial (mne.io.eyelink.eyelink.RawEyelink): _description_
        trial_N (int): _description_
    """

    et_trial, et_annotations, time_bounds = crop_et_trial(et_trial)

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

    # * Loop through each fixation event
    pbar = tqdm(fixation_inds, leave=False, disable=pbar_off)

    for fixation_ind in pbar:
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

        # * Check if fixation is on target and longer than 100 ms
        if fixation_duration >= min_fix_duration and on_target:
            # discarded = False

            fixation_data[target_ind].append(np.array([gaze_x, gaze_y]))

            gaze_target_fixation_sequence.append(
                [target_ind, fixation_start, fixation_duration, pupil_diam.mean()]
            )

        # else:
        #     # * Only for visualization purposes
        #     discarded = True

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
        gaze_target_fixation_sequence_df,
        gaze_info,
    )


tests: dict = {
    t: []
    for t in [
        "incorrect_trial_N",
        "montage",
        "eeg_sfreq",
        "et_sfreq",
        "et_cal",
        "sequences",
    ]
}


def log_test(subj_N, sess_N, test_name, desc):
    print(f"CHECK WARNING: subj_{subj_N}_sess_{sess_N} {desc}")
    tests[test_name].append((f"subj_{subj_N:02}", f"sess_{sess_N:02}", desc))


if __name__ == "__main__":
    # * Create a directory for data check results
    data_check_dir = WD / "results/data_check_dir"
    data_check_dir.mkdir(exist_ok=True, parents=True)

    # * Check the structure of the Data Directory
    data_dir_tree = DisplayTree(
        DATA_DIR, stringRep=True, showHidden=True, ignoreList=[".DS_Store"], maxDepth=3
    )

    # data_dir_tree = DisplayTree(
    #     "/Volumes/psychology$/Projects/2024_Pinier_FMG-7163_reasoning_EEG/data/Lab",
    #     stringRep=True,
    #     showHidden=True,
    #     ignoreList=[".DS_Store"],
    #     maxDepth=4,
    # )

    print(data_dir_tree)

    # * Check for notes in session info files
    notes = check_notes(DATA_DIR)

    # * Get DataFrame of all session info
    sessions_info = get_all_sess_info()

    # * Check that all sessions have the same window size and image size
    window_sizes = np.unique(np.array([ws for ws in sessions_info.window_size]))
    assert sorted(window_sizes) == sorted(SCREEN_RESOLUTION)

    img_sizes = np.unique(np.array([ws for ws in sessions_info.img_size]), axis=0)[0]
    assert sorted(img_sizes) == sorted(IMG_SIZE)

    eye_per_subj = sessions_info.groupby("subj_id")["eye"].unique()
    assert [len(i) == i for i in eye_per_subj], (
        "Some subjects have different dominant eye in different sessions"
    )

    subj_sess = sessions_info[["subj_id", "sess"]].to_numpy().astype(int)

    # * Check ET calibrations
    # et_cals_summary, et_cal_stats = check_et_calibrations(DATA_DIR)

    # * Check eye tracker data
    for subj_N, sess_N in tqdm(subj_sess):
        fpath = data_check_dir / f"{subj_N:02}{sess_N:02}-fixated_icons_per_trial.csv"
        if fpath.exists():
            continue

        # * Load raw data
        sess_dir = DATA_DIR / f"subj_{subj_N:02}/sess_{sess_N:02}"

        # * File paths
        et_fpath = [f for f in sess_dir.glob("*.asc")][0]
        # sess_info_file = [f for f in sess_dir.glob("*sess_info.json")][0]
        sequences_file = WD.parent / f"experiment-Lab/sequences/session_{sess_N}.csv"

        # * Load data
        # sess_info = json.load(open(sess_info_file))

        sequences = pd.read_csv(
            sequences_file, dtype={"choice_order": str, "seq_order": str}
        )

        # raw_behav = pd.read_csv(behav_fpath).merge(sequences, on="item_id")
        raw_behav = load_and_clean_behav_data(DATA_DIR, subj_N, sess_N).merge(
            sequences, on="item_id"
        )

        raw_behav.drop(columns=["pattern_x", "solution_x"], inplace=True)

        raw_behav.rename(
            columns={
                "pattern_y": "pattern",
                "solution_y": "solution",
            },
            inplace=True,
        )

        raw_et = mne.io.read_raw_eyelink(et_fpath, verbose="WARNING")

        with contextlib.redirect_stdout(io.StringIO()):
            et_cals = read_eyelink_calibration(et_fpath)

        (
            manual_et_trials,
            *_,
            # et_events_dict,
            # et_events_dict_inv,
            # et_trial_bounds,
            # et_trial_events_df,
        ) = preprocess_et_data(raw_et, et_cals)

        fixated_icons_per_trial = []

        for trial_N in tqdm(raw_behav.index, desc="Analyzing every trial", leave=False):
            # * Get the EEG and ET data for the current trial
            et_trial = next(manual_et_trials)

            (fixation_data, gaze_target_fixation_sequence_df, gaze_info) = (
                analyze_ET_decision_period(
                    et_trial,
                    raw_behav,
                    trial_N,
                    pbar_off=True,
                    min_fix_duration=0.01,
                )
            )

            fixated_icons_per_trial.append(
                {k: len(v) for k, v in fixation_data.items()}
            )

        df = pd.DataFrame(fixated_icons_per_trial)
        df.columns = [f"icon_{int(i)}" for i in df.columns]
        df.insert(0, "subj_N", subj_N)
        df.insert(1, "sess_N", sess_N)
        df.insert(2, "trial_N", range(len(df)))
        df["rt"] = raw_behav["rt"]
        # df["correct"] = raw_behav["correct"].astype(int)
        df.to_csv(fpath, index=False)

        # #! TEMP
        # folder = Path("results/data_check_dir")
        # files = list(folder.glob("*.csv"))
        # df = pd.concat([pd.read_csv(f) for f in files])

        # df.sum()

        # df.groupby(["subj_N", "sess_N"]).mean()
