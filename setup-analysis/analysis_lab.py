from pathlib import Path
import json
import re
from itertools import combinations
from typing import Dict, Final, List, Optional, Tuple, Union
import contextlib
from matplotlib import pyplot as plt, patches as mpatches, ticker
import mne
import mne.baseline
from mne_icalabel import label_components
from mne.preprocessing.eyetracking import read_eyelink_calibration
import numpy as np
import pandas as pd
import pendulum
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorpac
import io
from IPython.display import display
from loguru import logger
from rsatoolbox.data import Dataset, TemporalDataset
from rsatoolbox.rdm import calc_rdm, RDMs, compare as compare_rdms, calc_rdm_movie
import rsatoolbox
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm
import logging
import seaborn as sns
import dataframe_image as dfi
from analysis_plotting import (
    get_gaze_heatmap,
    plot_eeg,
    plot_eeg_and_gaze_fixations,
    plot_matrix,
    prepare_eeg_data_for_plot,
    show_ch_groups,
    plot_sequence_img,
    plot_rdm,
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
    apply_df_style,
    read_file,
    reorder_item_ids,
    get_reference_rdms,
    # email_sender as EmailSender,
)
from analysis_lab_conf import Config as c
import copy
# from tensorpac.methods import
# from scipy import signal
# import hmp
# from pprint import pprint
# from string import ascii_letters
# from tensorpac import EventRelatedPac, Pac
# from tensorpac.signals import pac_signals_wavelet
# from autoreject import AutoReject

WD = Path(__file__).parent
# os.chdir(WD)
assert WD == Path.cwd()

# * Get the tensorpac logger
tensorpac_logger = logging.getLogger("tensorpac")

# * Set the logging level to WARNING or ERROR to suppress INFO messages
tensorpac_logger.setLevel(logging.WARNING)  # Or logging.ERROR

# TODO: WARNING: unknown channels detected. Dropping:  ['EMG1', 'EMG2', 'EMG3', 'EMG4']: modify appropriate code to adjust when first slot was used instead of 7
# * Packages to look into:
# import mplcursors
# import polars as pl
# import pylustrator
"""
Abbreviations:
- ET: Eye Tracking
- EEG: Electroencephalography
- ERP: Event-Related Potential
- FRP: Fixation-Related Potential
- RDM: Representational Dissimilarity Matrix
- RSA: Representational Similarity Analysis
- PAC: Phase-Amplitude Coupling
- EOG: Electrooculography
- EMG: Electromyography
- MNE: MNE-Python
"""

# * ####################################################################################
# * GLOBAL VARS
# * ####################################################################################

c.LOG_DIR.mkdir(exist_ok=True, parents=True)
c.EXPORT_DIR.mkdir(exist_ok=True, parents=True)

mne.viz.set_browser_backend(c.MNE_BROWSER_BACKEND)
plt.switch_backend(c.MPL_BACKEND)
pd.set_option("future.no_silent_downcasting", True)

log_files = list(c.LOG_DIR.glob("*.log"))

if len(log_files) > 0:
    last_log_file = sorted(log_files)[-1]
    last_log_file_N = int(last_log_file.stem.split("-")[1])
else:
    last_log_file_N = -1

LOG_FILE = c.LOG_DIR / f"anlysis_log-{last_log_file_N + 1:03}-{c.TIMESTAMP}.log"
logger.add(LOG_FILE)

# * ####################################################################################
# * DATA PREPROCESSING AND LOADING
# * ####################################################################################


def load_and_clean_behav_data(data_dir: Path, subj_N: int, sess_N: int):
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

    # * ----------------------------------------
    sequences_file = WD.parent / f"config/sequences/session_{sess_N}.csv"

    sequences = pd.read_csv(
        sequences_file, dtype={"choice_order": str, "seq_order": str}
    )

    behav_data = behav_data.merge(sequences, on="item_id")

    # * Drop unnecessary columns
    behav_data.drop(columns=["pattern_x", "solution_x"], inplace=True)

    behav_data.rename(
        columns={
            "pattern_y": "pattern",
            "solution_y": "solution",
        },
        inplace=True,
    )

    return behav_data


def load_and_clean_behav_data_all(data_dir: Path):
    for subj_dir in data_dir.glob("subj_*"):
        subj_N = int(subj_dir.stem.split("_")[1])
        for sess_dir in subj_dir.glob("sess_*"):
            sess_N = int(sess_dir.stem.split("_")[1])
            behav_data = load_and_clean_behav_data(data_dir, subj_N, sess_N)
            yield behav_data


def load_raw_data(
    subj_N: int,
    sess_N: int,
    data_dir: Path,
    eeg_montage,
    bad_chans=None,
    logger=None,
):
    """_summary_

    Args:
        subj_N (int): _description_
        sess_N (int): _description_
        data_dir (Path): _description_
        eeg_montage (_type_): _description_
        bad_chans (_type_, optional): _description_. Defaults to None.
        logger (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple[Any, DataFrame, DataFrame, RawEDF, RawEyelink, list]: sess_info,
        sequences, raw_behav, raw_eeg, raw_et, et_cals

    """
    sess_dir = data_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"

    # * File paths
    et_fpath = [f for f in sess_dir.glob("*.asc")][0]
    eeg_fpath = [f for f in sess_dir.glob("*.bdf")][0]
    # behav_fpath = [f for f in sess_dir.glob("*behav*.csv")][0]
    sess_info_file = [f for f in sess_dir.glob("*sess_info.json")][0]

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

    raw_behav = load_and_clean_behav_data(data_dir, subj_N, sess_N)
    raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=False, verbose="WARNING")

    # set_eeg_montage(subj_N, sess_N, raw_eeg, eeg_montage, eog_chans, bad_chans)

    set_eeg_montage(
        raw_eeg,
        eeg_montage,
        c.EOG_CHANS,
        c.NON_EEG_CHANS,
        verbose=True,
    )

    raw_eeg.info["bads"] = bad_chans

    raw_et = mne.io.read_raw_eyelink(et_fpath, verbose="WARNING")

    with contextlib.redirect_stdout(io.StringIO()):
        et_cals = read_eyelink_calibration(et_fpath)

    return sess_info, raw_behav, raw_eeg, raw_et, et_cals


# * ####################################################################################
# * BEHAVIORAL ANALYSIS
# * ####################################################################################
def analyze_perf(
    cleaned_behav_data: pd.DataFrame,
    return_raw: Optional[bool] = False,
    patterns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    cleaned_behav_data["correct"] = cleaned_behav_data["correct"].astype(int)

    if patterns is None:
        patterns = sorted(cleaned_behav_data["pattern"].unique().tolist())

    # * --- Overall Results ---
    overall_acc = cleaned_behav_data["correct"].describe()
    overall_acc = pd.DataFrame(overall_acc).T
    overall_acc.reset_index(drop=True, inplace=True)

    overall_rt = cleaned_behav_data["rt"].describe()
    overall_rt = pd.DataFrame(overall_rt).T
    overall_rt.reset_index(drop=True, inplace=True)

    # * --- Detailed Results ---
    acc_by_patt = cleaned_behav_data.groupby("pattern")["correct"].describe()
    acc_by_patt = acc_by_patt.reindex(patterns, fill_value=np.nan)

    rt_by_patt = cleaned_behav_data.groupby("pattern")["rt"].describe()
    rt_by_patt = rt_by_patt.reindex(patterns, fill_value=np.nan)

    rt_by_crct = cleaned_behav_data.groupby("correct")["rt"].describe()
    rt_by_crct = rt_by_crct.reindex([0, 1], fill_value=np.nan)

    rt_by_crct_and_patt = cleaned_behav_data.groupby(["pattern", "correct"])[
        "rt"
    ].describe()
    index = pd.MultiIndex.from_product([patterns, [0, 1]], names=["pattern", "correct"])
    rt_by_crct_and_patt = rt_by_crct_and_patt.reindex(index, fill_value=np.nan)

    for df in [acc_by_patt, rt_by_patt, rt_by_crct, rt_by_crct_and_patt]:
        df.reset_index(drop=False, inplace=True)

    res = dict(
        overall_acc=overall_acc,
        overall_rt=overall_rt,
        acc_by_patt=acc_by_patt,
        rt_by_patt=rt_by_patt,
        rt_by_crct=rt_by_crct,
        rt_by_crct_and_patt=rt_by_crct_and_patt,
    )

    if return_raw:
        res["raw_cleaned"] = cleaned_behav_data
    return res


def perf_analysis_sess(
    data_dir: Path,
    subj_N: int,
    sess_N: int,
    return_raw: Optional[bool] = False,
    patterns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    # * --- Data Cleaning ---
    behav_data = load_and_clean_behav_data(data_dir, subj_N, sess_N)

    res = analyze_perf(behav_data, return_raw=return_raw, patterns=patterns)

    for df in res.values():
        df["subj_N"] = subj_N
        df["sess_N"] = sess_N

    return res


def perf_analysis_subj(
    data_dir: Path,
    subj_N: int,
    return_raw: Optional[bool] = False,
    patterns: Optional[List[str]] = None,
    # TODO: implement -> by_session: Optional[bool] = False,
) -> Dict[str, pd.DataFrame]:
    """Analyzes the performance of a single subject for all completed trials.
    Warning: This is not a session-level analysis, but a subject-level analysis,
    meaning that the results will be aggregated over all sessions.

    Args:
        data_dir (Path): _description_
        subj_N (int): _description_
        return_raw (bool, optional): _description_. Defaults to False.
        patterns (_type_, optional): _description_. Defaults to None.

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """
    behav_files = sorted(data_dir.rglob(f"subj_{subj_N:02}/sess_*/*behav.csv"))

    _behav_data = []
    for behav_file in behav_files:
        sess_N = int(behav_file.parents[0].stem.split("_")[1])
        _behav_data.append(load_and_clean_behav_data(data_dir, subj_N, sess_N))

    cleaned_behav_data = pd.concat(_behav_data).reset_index(drop=True)
    del _behav_data

    res = analyze_perf(cleaned_behav_data, return_raw=return_raw, patterns=patterns)

    return res


def perf_analysis_all_subj(
    data_dir: Path,
    patterns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    # TODO: implement -> by_session: Optional[bool] = False,
) -> Dict[str, pd.DataFrame]:
    # # ! TEMP
    # data_dir = c.DATA_DIR
    # patterns = c.PATTERNS
    # save_dir = c.EXPORT_DIR / "analyzed/group_lvl"
    # # ! TEMP

    subj_dirs = sorted([f for f in data_dir.glob("subj_*")])
    subjects_N = [int(d.stem.split("_")[1]) for d in subj_dirs]

    def get_subj_res(subj_N):
        subj_res = perf_analysis_subj(
            data_dir, subj_N=subj_N, return_raw=True, patterns=patterns
        )

        for df in subj_res.values():
            df["subj_N"] = subj_N

        return subj_res

    subj_res = get_subj_res(subjects_N.pop(0))

    # * Initialize results dictionary from the first file
    res: Dict[str, pd.DataFrame] = {k: [] for k in subj_res.keys()}

    for k, v in subj_res.items():
        res[k].append(v)

    # * Iterate over the rest of the subjects and append the results to the dictionary
    for subj_N in subjects_N:
        subj_res = get_subj_res(subj_N)

        for k, v in subj_res.items():
            res[k].append(v)

    # * Concatenate the results
    for k in res.keys():
        res[k] = pd.concat(res[k]).reset_index(drop=True)

    # * Save the results to CSV files
    if save_dir is not None:
        prefix = "perf-"
        suffix = ""

        save_data_dir = Path(save_dir)
        save_data_dir.mkdir(exist_ok=True, parents=True)

        for file_name, data in res.items():
            file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
            data.to_csv(save_dir / f"{file_name}.csv", index=False)

    return res


def plot_perf_analysis(
    data_dir: Path,
    show_figs: bool = False,
    figs_params: Optional[Dict] = None,
    save_fig_params: Optional[Dict] = None,
    save_dir: Optional[Path] = None,
):
    # # ! TEMP
    # data_dir = c.DATA_DIR
    # show_figs = False
    # figs_params = None
    # save_figs_params = None
    # save_data_dir = None
    # save_figs_dir = c.EXPORT_DIR / "analyzed/group_lvl"
    # # ! TEMP

    if figs_params is None:
        fig_params = dict(figsize=(10, 6), dpi=300)
    if save_fig_params is None:
        save_fig_params = dict(dpi=300, bbox_inches="tight")

    ax_params = dict(
        grid={"ls": "--", "c": "k", "alpha": 0.5},
    )

    (
        overall_acc,
        overall_rt,
        acc_by_patt,
        rt_by_patt,
        rt_by_crct,
        rt_by_crct_and_patt,
        raw_cleaned,
    ) = perf_analysis_all_subj(data_dir=data_dir, patterns=c.PATTERNS).values()

    raw_cleaned.sort_values(["subj_N", "pattern", "item_id"], inplace=True)

    # *  --------- Accuracies by Subject and Pattern ---------
    acc_by_patt_styled = acc_by_patt.pivot(
        index="subj_N", columns="pattern", values="mean"
    )
    acc_by_patt_styled.loc[:, "mean"] = acc_by_patt_styled.mean(axis=1)
    acc_by_patt_styled.loc["mean", :] = acc_by_patt_styled.mean(axis=0)

    # * --------- RT by Subject and Pattern ---------
    rt_by_patt_styled = rt_by_patt.pivot(
        index="subj_N", columns="pattern", values="mean"
    )
    rt_by_patt_styled.loc[:, "mean"] = rt_by_patt_styled.mean(axis=1)
    rt_by_patt_styled.loc["mean", :] = rt_by_patt_styled.mean(axis=0)

    tables = dict(
        acc_by_patt=apply_df_style(acc_by_patt_styled),
        rt_by_patt=apply_df_style(rt_by_patt_styled),
    )

    # * --------- Plotting ---------
    yticks_pct = np.round(np.arange(0, 1.1, 0.1), 2)
    ytick_labels_pct = [f"{i * 100:.0f}" for i in yticks_pct]

    figs = {}

    # * --------- FIGURE: Accuracy by Subject ---------
    title = "Accuracy by Subject"
    fig, ax = plt.subplots(**fig_params)
    # ax.bar(
    #     overall_acc["subj_N"],
    #     overall_acc["mean"],
    # )  # yerr=overall_acc["std"])
    # sns.barplot(x="subj_N", y="correct", data=raw_cleaned, hue='sess_N', ax=ax)
    sns.barplot(x="subj_N", y="correct", errorbar="ci", data=raw_cleaned, ax=ax)
    ax.hlines(
        overall_acc["mean"].mean(),
        xmin=0,
        xmax=len(overall_acc["subj_N"].unique()),
        ls="--",
        color="red",
        alpha=0.7,
        label="Mean",
    )
    # ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.set_xlabel("Subject")
    # ax.set_xticks(overall_acc["subj_N"])
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: Accuracy by Subject and Pattern ---------
    title = "Accuracy by Subject and Pattern"
    fig, ax = plt.subplots(**fig_params)
    # sns.barplot(x="subj_N", y="correct", errorbar="ci", data=raw_cleaned, ax=ax)
    sns.lineplot(
        x="pattern",
        y="correct",
        hue="subj_N",
        errorbar=None,
        data=raw_cleaned.astype({"subj_N": str}),
        legend=False,
        marker="o",
        ax=ax,
    )
    # ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: Accuracy by Session ---------
    title = "Accuracy by Session"
    fig, ax = plt.subplots(**fig_params)
    sns.barplot(x="sess_N", y="correct", data=raw_cleaned, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by Session")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Session")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: Accuracy by Session and Subject - 1 ---------
    title = "Accuracy by Session and Subject - 1"
    fig, ax = plt.subplots(**fig_params)
    sns.lineplot(
        x="subj_N",
        y="correct",
        hue="sess_N",
        data=raw_cleaned.astype({"subj_N": str}),
        errorbar=None,
        marker="o",
        # legend=False,
        ax=ax,
    )
    ax.legend(title="Session", bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Subject")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: Accuracy by Session and Subject - 2 ---------
    title = "Accuracy by Session and Subject - 2"
    fig, ax = plt.subplots(**fig_params)
    sns.barplot(
        x="sess_N",
        y="correct",
        hue="subj_N",
        data=raw_cleaned.astype({"subj_N": str}),
        errorbar=None,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Session")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: RT by Subject ---------
    title = "RT by Subject"
    fig, ax = plt.subplots(**fig_params)
    # sns.barplot(x="subj_N", y="mean", data=raw_cleaned, errorbar="sd", ax=ax)
    ax.bar(
        overall_rt["subj_N"],
        overall_rt["mean"],
    )  # yerr=overall_rt["std"])
    ax.hlines(
        overall_rt["mean"].mean(),
        xmin=0,
        xmax=len(overall_acc["subj_N"].unique()),
        ls="--",
        color="red",
        alpha=0.7,
        label="Mean",
    )
    ax.set_xticks(overall_rt["subj_N"])
    ax.set_title(title)
    ax.set_ylabel("RT (s)")
    ax.set_xlabel("Subject")
    ax.grid(axis="y", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # *  --------- FIGURE PARAMS  ---------
    patterns_legend = "Patterns:\n  "
    patterns_legend += "\n  ".join(
        [f"{i + 1}: {' '.join(list(patt))}" for i, patt in enumerate(c.PATTERNS)]
    )
    # * these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)

    patterns_legend_params = dict(
        x=1.015,
        y=0.015,
        s=patterns_legend,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=props,
    )

    xticks = [i for i in range(len(c.PATTERNS))]
    xticks_labels = [str(i) for i in range(1, len(c.PATTERNS) + 1)]

    # * --------- FIGURE: Accuracy by Pattern ---------
    title = "Accuracy by Pattern"
    fig, ax = plt.subplots(**fig_params)
    data = acc_by_patt.groupby("pattern")[["mean", "std"]].mean()
    sns.lineplot(
        x="pattern", y="correct", data=raw_cleaned, errorbar="ci", marker="o", ax=ax
    )
    # ax.plot(data["mean"], marker="o")
    # ax.errorbar(x=data.index, y=data["mean"], yerr=data["std"], capsize=5, fmt="o")
    ax.hlines(
        acc_by_patt["mean"].mean(),
        xmin=0,
        xmax=len(c.PATTERNS) - 1,
        ls="--",
        color="red",
        alpha=0.7,
        label="Mean",
    )
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    # ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels)
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    # ax.text(**patterns_legend_params, transform=ax.transAxes)
    ax.grid(axis="both", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: RT by Pattern ---------
    title = "RT by Pattern"
    # data = rt_by_patt.groupby("pattern")[["mean", "std"]].mean()
    fig, ax = plt.subplots(**fig_params)
    # ax.plot(data["mean"], marker="o")
    # ax.errorbar(x=data.index, y=data["mean"], yerr=data["std"], capsize=5, fmt="o")
    sns.lineplot(
        x="pattern", y="rt", data=raw_cleaned, errorbar="ci", marker="o", ax=ax
    )
    ax.hlines(
        # data["mean"].mean(),
        rt_by_patt["mean"].mean(),
        xmin=0,
        xmax=len(c.PATTERNS) - 1,
        ls="--",
        color="red",
        alpha=0.7,
        label="Mean",
    )
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_xlabel("Pattern")
    ax.set_ylabel("RT (s)")
    ax.set_title(title)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels)
    # ax.text(**patterns_legend_params, transform=ax.transAxes)
    ax.grid(axis="both", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: RT by Pattern and Subject ---------
    title = "RT by Pattern and Subject"
    fig, ax = plt.subplots(**fig_params)
    sns.lineplot(
        x="pattern",
        y="rt",
        hue="subj_N",
        data=raw_cleaned.astype({"subj_N": str}),
        errorbar=None,
        marker="o",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Pattern")
    ax.set_ylabel("RT (s)")
    ax.set_title(title)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels)
    # ax.text(**patterns_legend_params, transform=ax.transAxes)
    ax.grid(axis="both", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    # * --------- FIGURE: RT by Pattern and Correctness ---------
    title = "RT by Pattern and Correctness"
    fig, ax = plt.subplots(**fig_params)
    sns.lineplot(
        x="pattern",
        y="rt",
        data=raw_cleaned.replace({"correct": {0: "Incorrect", 1: "Correct"}}),
        hue="correct",
        marker="o",
        errorbar="ci",
        ax=ax,
    )
    ax.legend(title="Correctness", bbox_to_anchor=(1, 1))
    ax.set_ylabel("RT (s)")
    ax.set_title(title)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks_labels)
    # ax.text(**patterns_legend_params, transform=ax.transAxes)
    ax.grid(axis="both", **ax_params.get("grid", {}))
    plt.tight_layout()
    plt.show() if show_figs else None
    figs[title] = fig

    #  * --------- Export figures and tables ---------
    if save_dir is not None:
        prefix = "perf-"
        suffix = ""
        # suffix = "-(human_data)-(group_lvl)"

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # * Export figures to PNG
        for file_name, fig in figs.items():
            file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
            fig.savefig(save_dir / f"{file_name}.png", **save_fig_params)

        # * Export tables to PNG
        prefix += "table-"
        for file_name, table in tables.items():
            file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
            dfi.export(
                table,
                save_dir / f"{file_name}.png",
                table_conversion="matplotlib",
            )

    plt.close("all")


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
        if event_name in c.VALID_EVENTS:
            new_id = c.VALID_EVENTS[event_name]
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
    choice_key_et = [c.VALID_EVENTS_INV[i] for i in et_events[inds_responses, 2][0]]

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
        start_time = (et_events[start, 0] / raw_et.info["sfreq"]) - c.PRE_TRIAL_TIME
        end_time = et_events[end, 0] / raw_et.info["sfreq"] + c.POST_TRIAL_TIME

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


def crop_et_trial(epoch: mne.Epochs):
    # ! WARNING: We may not be capturing the first fixation if it is already on target
    # epoch = et_trial.copy()

    # * Get annotations, convert to DataFrame, and adjust onset times
    annotations = epoch.annotations.to_data_frame(time_format="ms")
    annotations["onset"] -= annotations["onset"].iloc[0]
    annotations["onset"] /= 1000

    # first_flash = annotations.query("description.str.contains('flash')").iloc[0]
    all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

    response_ids = c.EXP_CONFIG.lab.allowed_keys + ["timeout", "invalid"]
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


# * ####################################################################################
# * EEG ANALYSIS
# * ####################################################################################


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


def split_eeg_data_into_trials(
    raw_eeg: mne.io.Raw,
    raw_behav: pd.DataFrame,
    remove_practice: bool = True,
    practice_ind: int = 3,
    n_trials: int = 80,
    incomplete: str = "error",
    pb_on=False,
):
    """_summary_ #TODO

    Args:
        raw_eeg (mne.io.Raw): _description_
        raw_behav (pd.DataFrame): _description_
        remove_practice (bool, optional): _description_. Defaults to True.
        practice_ind (int, optional): _description_. Defaults to 3.
        n_trials (int, optional): _description_. Defaults to 80.
        incomplete (str, optional): _description_. Defaults to "error".

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if incomplete not in ["allow", "error", "skip"]:
        raise ValueError(
            "incomplete must be either of the following: 'allow', 'error', 'skip'"
        )

    eeg_events, _ = mne.events_from_annotations(
        raw_eeg, c.VALID_EVENTS, verbose="WARNING"
    )

    choice_key_eeg = [
        c.VALID_EVENTS_INV[i]
        for i in eeg_events[:, 2]
        if i in [10, 11, 12, 13, 14, 15, 16]
    ]

    eeg_trial_bounds, eeg_events_df = locate_trials(eeg_events, c.VALID_EVENTS)

    if remove_practice is True:
        if len(choice_key_eeg) > n_trials and len(eeg_trial_bounds) > n_trials:
            choice_key_eeg = choice_key_eeg[practice_ind:]
            eeg_trial_bounds = eeg_trial_bounds[practice_ind:]

    if not len(choice_key_eeg) == len(eeg_trial_bounds) == n_trials:
        warning_msg = (
            "Error with EEG events: incorrect number of trials.\n"
            f"{len(choice_key_eeg) = }\n{len(eeg_trial_bounds) = }"
        )
        if incomplete == "error":
            raise ValueError(warning_msg)
        elif incomplete == "skip":
            print(warning_msg)
            # * Return a list of None matching the size of the expect output
            return [None] * 4
        elif incomplete == "allow":
            print(warning_msg)
            pass

    raw_behav["choice_key_eeg"] = choice_key_eeg
    raw_behav["same"] = raw_behav["choice_key"] == raw_behav["choice_key_eeg"]

    manual_eeg_trials = []

    # * Loop through each trial
    if pb_on is True:
        eeg_trial_bounds = tqdm(eeg_trial_bounds, "Creating EEG epochs")

    for start, end in eeg_trial_bounds:
        # * Get start and end times in seconds
        start_time = (eeg_events[start, 0] / raw_eeg.info["sfreq"]) - c.PRE_TRIAL_TIME
        end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + c.POST_TRIAL_TIME

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


def preprocess_eeg_data(
    raw_eeg: mne.io.Raw,
    eeg_chan_groups: Dict[str, str],
    raw_behav: pd.DataFrame,
    preprocessed_dir: Path,
    force: Optional[bool] = False,
    reuse_ica: Optional[bool] = True,
    # # TODO: bad_chs_method="interpolate",
):
    # ! TEMP
    # eeg_chan_groups = c.EEG_CHAN_GROUPS
    # subj_N = 1
    # sess_N = 4
    # bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
    # sess_info, raw_behav, raw_eeg, raw_et, et_cals = load_raw_data(subj_N, sess_N, c.DATA_DIR, c.EEG_MONTAGE, bad_chans)
    # ! TEMP

    fpath = Path(raw_eeg.filenames[0])
    subj_N = int(fpath.parents[1].name.split("_")[1])
    sess_dir = fpath.parents[0]
    sess_N = int(sess_dir.name.split("_")[1])

    # * Create preprocessed_dir and ica_dir if they don't exist
    ica_dir = preprocessed_dir / "ICA"
    ica_dir.mkdir(exist_ok=True, parents=True)

    preprocessed_raw_fpath = (
        preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_preprocessed-raw.fif"
    )

    if not preprocessed_raw_fpath.exists() or force is True:
        print("Preprocessing raw data...")
        raw_eeg.load_data(verbose="WARNING")
        prepro_eeg = raw_eeg.copy()
        del raw_eeg

        # * Detecting events
        eeg_events = mne.find_events(
            prepro_eeg,
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
            prepro_eeg.info["sfreq"],
            event_desc=c.VALID_EVENTS_INV,
            verbose="WARNING",
        )

        prepro_eeg.set_annotations(annotations, verbose="WARNING")

        bad_chans = prepro_eeg.info["bads"]

        # TODO: automatically remove bad channels (e.g., amplitude cutoff)

        manually_set_bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(
            f"sess_{sess_N}"
        )

        if not bad_chans == manually_set_bad_chans:
            print(
                "WARNING: raw EEG bad channels do not match expected bad channels, combining them"
            )

        bad_chans = list(set(bad_chans) | set(manually_set_bad_chans))

        prepro_eeg.info["bads"] = bad_chans

        # prepro_eeg.drop_channels(bad_chans)

        # * Check if channel groups include all channels present in the montage
        # * i.e., that there are no "orphan" channels
        check_ch_groups(prepro_eeg.get_montage(), eeg_chan_groups)

        # * Average Reference
        prepro_eeg = prepro_eeg.set_eeg_reference(
            ref_channels="average", verbose="WARNING"
        )

        # * Filter to remove power line noise
        prepro_eeg.notch_filter(freqs=np.arange(50, 251, 50), verbose="WARNING")

        # * Bandpass Filter: 1-100 Hz
        prepro_eeg.filter(l_freq=1, h_freq=100, verbose="WARNING")

        # * ############################################################################
        # * EOG artifact rejection using ICA
        # * ############################################################################

        ica_fpath = ica_dir / f"subj_{subj_N:02}{sess_N:02}_fitted-ica.fif"

        if ica_fpath.exists() and reuse_ica is True:
            # * if ICA file exists, load it
            ica = mne.preprocessing.read_ica(ica_fpath)

        else:
            # * Create a copy of the raw data to hihg-pass filter at 1Hz before ICA
            # * as recommended by MNE: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
            # prepro_eeg_copy_for_ica = prepro_eeg.copy()
            # prepro_eeg.filter(l_freq=1, h_freq=100, verbose="WARNING")

            ica = mne.preprocessing.ICA(
                n_components=None,
                noise_cov=None,
                random_state=c.RAND_SEED,
                # method="fastica",
                method="infomax",
                fit_params=dict(extended=True),
                max_iter="auto",
                verbose="WARNING",
            )

            ica.fit(prepro_eeg, verbose="WARNING")

            ica.save(ica_fpath, verbose="WARNING")

        # eog_inds, eog_scores = ica.find_bads_eog(prepro_eeg)
        # ica.exclude = eog_inds

        # # * Label components using IClabel
        ic_labels = label_components(prepro_eeg, ica, method="iclabel")

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
        prepro_eeg = ica.apply(prepro_eeg, verbose="WARNING")

        # * Interpolate bad channels and remove them from "bads" list in eeg info
        prepro_eeg = prepro_eeg.interpolate_bads(reset_bads=True)

        # * Set average reference again, including interpolated bad channels
        prepro_eeg = prepro_eeg.set_eeg_reference(
            ref_channels="average", verbose="WARNING"
        )

        # * Add bad channels to the "bads" list in eeg info again
        prepro_eeg.info["bads"] = bad_chans

        # * Save preprocessed raw data
        prepro_eeg.save(preprocessed_raw_fpath, overwrite=True, verbose="WARNING")

    else:
        prepro_eeg = mne.io.read_raw_fif(
            preprocessed_raw_fpath, preload=False, verbose="WARNING"
        )

    # split_eeg_data_into_trials(prepro_eeg, raw_behav)

    return prepro_eeg


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
        include=c.EEG_CHAN_GROUPS.frontal,
        exclude=eeg_trial.info["bads"],
    )

    # * ################################################################################

    assert c.EEG_SFREQ == eeg_trial.info["sfreq"], (
        "EEG data has incorrect sampling rate"
    )

    # * Crop the data
    et_trial, et_annotations, time_bounds = crop_et_trial(et_trial)

    # * Adjust time bounds for EEG baseline and window
    # * Cropping with sample bounds
    trial_duration = (time_bounds[1] + eeg_window + eeg_baseline) - time_bounds[0]
    sample_bounds = [0, 0]
    sample_bounds[0] = int(time_bounds[0] * c.EEG_SFREQ)
    sample_bounds[1] = sample_bounds[0] + int(np.ceil(trial_duration * c.EEG_SFREQ))

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
        c.X_POS_STIM,
        c.Y_POS_CHOICES,
        c.Y_POS_SEQUENCE,
        c.SCREEN_RESOLUTION,
        c.IMG_SIZE,
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
        eeg_start_sample = int(fixation_start * c.EEG_SFREQ)
        eeg_end_sample = eeg_start_sample + int(
            np.ceil((eeg_window + eeg_baseline) * c.EEG_SFREQ)
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

        # * Check if fixation is on target and duration is above minimum
        if fixation_duration >= c.MIN_FIXATION_DURATION and on_target:
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
                    c.EEG_CHAN_GROUPS,
                    c.EEG_MONTAGE,
                    c.NON_EEG_CHANS,
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
                eeg_sfreq=c.EEG_SFREQ,
                et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                eeg_baseline=eeg_baseline,
                response_onset=response_onset,
                eeg_start_time=eeg_start_time,
                eeg_end_time=eeg_end_time,
                icon_images=c.ICON_IMAGES,
                img_size=c.IMG_SIZE,
                stim_pos=stim_pos,
                chans_pos_xy=chans_pos_xy,
                ch_group_inds=ch_group_inds,
                group_colors=group_colors,
                screen_resolution=c.SCREEN_RESOLUTION,
                title=title,
                vlines=[
                    eeg_baseline * c.EEG_SFREQ,
                    eeg_baseline * c.EEG_SFREQ + fixation_duration * c.EEG_SFREQ,
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
            sfreq=c.EEG_SFREQ,
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
        for group_name, group_chans in c.EEG_CHAN_GROUPS.items()
    }

    # * Get positions and presentation order of stimuli
    trial_info = get_trial_info(
        epoch_N,
        raw_behav,
        c.X_POS_STIM,
        c.Y_POS_CHOICES,
        c.Y_POS_SEQUENCE,
        c.SCREEN_RESOLUTION,
        c.IMG_SIZE,
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

        if fixation_duration >= c.MIN_FIXATION_DURATION and on_target:
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

        ax_et.set_xlim(0, c.SCREEN_RESOLUTION[0])
        ax_et.set_ylim(c.SCREEN_RESOLUTION[1], 0)
        ax_et.set_title(title)

        # * Plot target icon
        ax_et.imshow(
            c.ICON_IMAGES[target_name],
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
            c.IMG_SIZE[0],
            c.IMG_SIZE[1],
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
        ax_eeg.set_xticks(xticks, ((xticks / c.EEG_SFREQ) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    return fixation_data, eeg_fixation_data


def get_neg_erp_peak(
    evoked: mne.Evoked,
    time_window: tuple,
    selected_chans: Optional[List[str]] = None,
    unit: str = "uV",
    plot: bool = False,
) -> Tuple[float, float]:
    """TODO:_summary_

    Args:
        evoked (mne.Evoked): _description_
        time_window (tuple): _description_
        selected_chans (Optional[List[str]], optional): _description_. Defaults to None.
        unit (str, optional): _description_. Defaults to "uV".
        plot (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[float, float]: (peak latency, peak amplitude)
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


# * ####################################################################################
# * MAIN FUNCTIONS
# * ####################################################################################
def analyze_session(subj_N: int, sess_N: int, save_dir: Path, preprocessed_dir: Path):
    """ """
    # # ! TEMP
    # preprocessed_dir = c.EXPORT_DIR / "preprocessed_data"
    # subj_N = 4
    # sess_N = 1
    # save_dir = c.EXPORT_DIR / f"subj_{subj_N:02}-sess_{sess_N:02}"
    # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    if not preprocessed_dir.exists():
        raise FileNotFoundError("Preprocessed data directory not found")

    bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])

    # * Load the data
    sess_info, raw_behav, raw_eeg, raw_et, et_cals = load_raw_data(
        subj_N, sess_N, c.DATA_DIR, c.EEG_MONTAGE, bad_chans
    )

    if notes := sess_info["Notes"]:
        print(f"SESSION NOTES:\n{notes}")

    # sess_screen_resolution = sess_info["window_size"]
    # sess_img_size = sess_info["img_size"]
    # et_sfreq = raw_et.info["sfreq"]
    # tracked_eye = sess_info["eye"]
    # vision_correction = sess_info["vision_correction"]
    # eye_screen_distance = sess_info["eye_screen_dist"]

    if not c.ET_SFREQ == raw_et.info["sfreq"]:
        raise ValueError("Eye-tracking data has incorrect sampling rate")

    if not c.EEG_SFREQ == raw_eeg.info["sfreq"]:
        raise ValueError("EEG data has incorrect sampling rate")

    (
        manual_et_trials,
        *_,
        # et_events_dict,
        # et_events_dict_inv,
        # et_trial_bounds,
        # et_trial_events_df,
    ) = preprocess_et_data(raw_et, et_cals)

    prepro_eeg = preprocess_eeg_data(
        raw_eeg=raw_eeg,
        eeg_chan_groups=c.EEG_CHAN_GROUPS,
        raw_behav=raw_behav,
        preprocessed_dir=preprocessed_dir,
        force=False,
    )
    (
        manual_eeg_trials,
        *_,
        # eeg_trial_bounds,
        # eeg_events,
        # eeg_events_df,
    ) = split_eeg_data_into_trials(prepro_eeg, raw_behav)

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
            eeg_baseline=c.EEG_BASELINE_FRP,
            eeg_window=c.FRP_WINDOW,
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


def main(data_dir: Path, preprocessed_dir: Path, save_dir: Path):
    # ! TEMP
    # data_dir = c.DATA_DIR
    # save_dir = EXPORT_DIR / "analyzed/subj_lvl"
    # preprocessed_dir = EXPORT_DIR / "preprocessed_data"
    # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    n_subjs = len(list(data_dir.glob("subj_*")))

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

        subj_dir = data_dir / f"subj_{subj_N:02}"

        n_sessions = len(list(subj_dir.glob("sess_*")))

        for sess_N in range(1, n_sessions + 1):
            sess_save_dir = save_dir / f"subj_{subj_N:02}/sess_{sess_N:02}"

            if sess_save_dir.exists() and len(list(sess_save_dir.glob("*.pkl"))) > 0:
                print(
                    f"subj_{subj_N:02}-sess_{sess_N:02}: data already analyzed, skipping..."
                )
                continue
            else:
                try:
                    sess_save_dir.mkdir(exist_ok=True, parents=True)
                    (
                        sess_erps,
                        fixation_data,
                        eeg_fixation_data,
                        gaze_info,
                        gaze_target_fixation_sequence,
                    ) = analyze_session(subj_N, sess_N, sess_save_dir, preprocessed_dir)

                except Exception as e:
                    print(f"Error in subj_{subj_N:02}-sess_{sess_N:02}: {e}")
                    errors.append((subj_N, sess_N, e))
                    continue
        # print([" ".join([str(i) for i in j]) for j in errors])
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


def analyze_processed_gaze_data_all_subj(data_dir: Path, res_dir: Path):
    # # ! TEMP
    # data_dir = c.DATA_DIR
    # res_dir = Path(
    #     "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/Lab/analyzed"
    # )
    # # ! TEMP

    # * load cleaned behavioral data
    behav_data = pd.concat(load_and_clean_behav_data_all(data_dir))

    gaze_info_files = sorted(res_dir.rglob("gaze_info.pkl"))

    _gaze_info = []
    for f in gaze_info_files:
        subj_N = int(f.parents[1].name.split("_")[-1])
        sess_N = int(f.parents[0].name.split("_")[-1])

        sess_gaze_info = read_file(f)
        sess_behav_data = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")

        # * Add subj and sess info to gaze data
        sess_gaze_info["subj_N"] = subj_N
        sess_gaze_info["sess_N"] = sess_N

        sess_gaze_info = pd.merge(
            sess_gaze_info,
            sess_behav_data,
            on=["subj_N", "sess_N", "trial_N"],
            how="inner",
        )

        _gaze_info.append(sess_gaze_info)

    gaze_info = pd.concat(_gaze_info)
    gaze_info.reset_index(drop=True, inplace=True)
    del _gaze_info

    res = dict(
        gaze_data=gaze_info,
        # * Stats on number of fixations per pattern type
        fix_count_per_patt_type=gaze_info.groupby("pattern")["count"].describe(),
        # * Stats on duration of fixations per pattern type
        fix_dur_per_patt_type=gaze_info.groupby("pattern")["total_duration"].describe(),
        # * Stats on duration of fixations on each icon per pattern type
        icon_fix_per_patt_type=gaze_info.groupby("pattern")["mean_duration"].describe(),
        # * Stats on pupil diameter during fixations per pattern type
        pupil_diam_per_patt_type=gaze_info.groupby("pattern")[
            "mean_pupil_diam"
        ].describe(),
    )

    return res


def analyze_processed_frp_data_all_subj(
    data_dir: Path,
    res_dir: Path,
    selected_chans: List[str],
    neg_peak_time_window: tuple = (0, 0.2),
    pbar=True,
):
    # # ! TEMP
    # data_dir = c.DATA_DIR
    # res_dir = Path(
    #     "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/Lab/analyzed"
    # )
    # selected_chans = c.EEG_CHAN_GROUPS["occipital"]
    # neg_peak_time_window = (0, 0.2)
    # # ! TEMP

    # * load cleaned behavioral data
    behav_data = pd.concat(load_and_clean_behav_data_all(data_dir))

    frp_files = sorted(res_dir.rglob("sess_frps.pkl"))

    _neg_peak_data_seq = []
    # _neg_peak_data_choices = []

    for f in tqdm(frp_files, disable=not pbar, desc="Analyzing FRP data"):
        subj_N = int(f.parents[1].name.split("_")[-1])
        sess_N = int(f.parents[0].name.split("_")[-1])

        sess_frps = read_file(f)

        sess_behav_data = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")
        item_ids = sess_behav_data["item_id"]
        patterns = sess_behav_data["pattern"]

        for i, evoked in enumerate(sess_frps["sequence"]):
            item_id, pattern = item_ids[i], patterns[i]
            if evoked is None:
                # print(f"subj_{subj_N:02}-sess_{sess_N:02}-trial{i:03}: No data for sequence")
                peak_latency, peak_amplitude = np.nan, np.nan
            else:
                peak_latency, peak_amplitude = get_neg_erp_peak(
                    evoked,
                    neg_peak_time_window,
                    selected_chans,
                )
            _neg_peak_data_seq.append(
                [
                    subj_N,
                    sess_N,
                    i,
                    item_id,
                    pattern,
                    peak_latency,
                    peak_amplitude,
                    "sequence",
                ]
            )

        # for i, evoked in enumerate(sess_frps["choices"]):
        #     item_id, pattern = item_ids[i], patterns[i]
        #     if evoked is None:
        #         # print(f"subj_{subj_N:02}-sess_{sess_N:02}-trial_{i:03}: No data for choices")
        #         peak_latency, peak_amplitude = np.nan, np.nan
        #     else:
        #         peak_latency, peak_amplitude = get_neg_erp_peak(
        #         evoked,
        #         neg_peak_time_window,
        #         selected_chans,
        #     )
        #     _neg_peak_data_choices.append(
        #         [
        #             subj_N,
        #             sess_N,
        #             i,
        #             item_id,
        #             pattern,
        #             peak_latency,
        #             peak_amplitude,
        #             'sequence',
        #         ]
        #     )

    df_cols = [
        "subj_N",
        "sess_N",
        "trial_N",
        "item_id",
        "pattern",
        "peak_latency",
        "peak_amplitude",
        "stim_type",
    ]

    neg_peak_data_seq = pd.DataFrame(_neg_peak_data_seq, columns=df_cols)
    # neg_peak_data_choices = pd.DataFrame(_neg_peak_data_choices, columns=df_cols)

    amplitude_per_item = (
        neg_peak_data_seq.groupby("item_id")["peak_amplitude"].describe().reset_index()
    )
    latency_per_item = (
        neg_peak_data_seq.groupby("item_id")["peak_latency"].describe().reset_index()
    )

    amplitude_per_patt = (
        neg_peak_data_seq.groupby("pattern")["peak_amplitude"].describe().reset_index()
    )
    latency_per_patt = (
        neg_peak_data_seq.groupby("pattern")["peak_latency"].describe().reset_index()
    )

    res = dict(
        neg_peak_data_seq=neg_peak_data_seq,
        # neg_peak_data_choices=neg_peak_data_choices,
        amplitude_per_patt_type=amplitude_per_patt,
        latency_per_patt_type=latency_per_patt,
        amplitude_per_item=amplitude_per_item,
        latency_per_item=latency_per_item,
    )

    return res


# * ####################################################################################
# * REPRESENTATIONAL SIMILARITY ANALYSIS
# * ####################################################################################
def get_response_locked_eeg_rdm_all_subj(save_dir: Path, preprocessed_dir: Path):
    # # ! TEMP
    # preprocessed_dir = c.EXPORT_DIR / "preprocessed_data-SAVE"
    # dissimilarity_metric = "correlation"
    # save_dir = (
    #     c.EXPORT_DIR
    #     / f"analyzed/RSA-Response_ERP-last_chan_removed-{dissimilarity_metric}"
    # )
    # similarity_metric = "corr"
    # # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory not found, check path: {preprocessed_dir}"
        )

    prepro_eeg_files = [
        f
        for f in preprocessed_dir.glob("*-raw.fif")
        if f.is_file() and not f.name.startswith(".")
    ]
    subjects = [int(re.search(r"subj_(\d{2})", f.name)[1]) for f in prepro_eeg_files]
    subjects = sorted(set(subjects))

    # * ----------------------------------------
    # * ----------------------------------------
    resp_events = ["a", "x", "m", "l", "invalid", "timeout"]
    # subjects = [4, 5]

    eeg_data = dict()
    behav_data = []

    for subj_N in tqdm(subjects):
        eeg_files = sorted(
            [f for f in prepro_eeg_files if f"subj_{subj_N:02}" in f.name]
        )

        subj_eeg_data = []
        subj_behav_data = []

        for eeg_file in eeg_files:
            sess_N = re.search(r"subj_\d{2}(\d{2})", eeg_file.name)

            if sess_N is not None:
                sess_N = int(sess_N[1])
            else:
                raise ValueError(f"session number not found in file name: {eeg_file}")

            sess_behav = load_and_clean_behav_data(c.DATA_DIR, subj_N, sess_N)

            sess_prepro_eeg = mne.io.read_raw(eeg_file, verbose=False)
            sess_prepro_eeg.info["bads"] = []  # ! TEMP

            eeg_trials, eeg_trial_bounds, eeg_events, eeg_events_df = (
                split_eeg_data_into_trials(
                    sess_prepro_eeg, sess_behav, incomplete="skip"
                )
            )

            if eeg_trials is None:
                continue
            else:
                eeg_trials = list(eeg_trials)
                subj_eeg_data.extend(eeg_trials)
                subj_behav_data.append(sess_behav)

                # # ! TEMP
                sess_eeg = copy.deepcopy(eeg_trials)
                sess_eeg = mne.concatenate_raws(sess_eeg)
                sess_annotations = sess_eeg.annotations.to_data_frame()["description"]

                sess_filtered_events = [a for a in sess_annotations if a in resp_events]
                sess_filtered_events = list(set(sess_filtered_events))

                sess_epochs = mne.Epochs(
                    sess_eeg,
                    # event_id=["trial_end"],
                    event_id=sess_filtered_events,
                    tmin=-1.0,
                    tmax=0,
                    baseline=None,
                    verbose=False,
                )
                evoked = sess_epochs.average()
                fig = evoked.plot(show=False)
                fig.savefig(save_dir / f"ERP-subj{subj_N:02}{sess_N:02}.png")
                plt.close()
                # # ! TEMP

        subj_behav_data = pd.concat(subj_behav_data).reset_index()

        reordered_inds = reorder_item_ids(
            original_order_df=subj_behav_data,
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )

        subj_behav_data = subj_behav_data.iloc[reordered_inds].reset_index()

        subj_eeg_data = mne.concatenate_raws([subj_eeg_data[i] for i in reordered_inds])

        annotations = subj_eeg_data.annotations.to_data_frame()["description"]
        filtered_events = [a for a in annotations if a in resp_events]

        # * Check if data is complete (participants have finished all their sessions)
        n_trials = len(filtered_events)

        if not n_trials == 400:
            print(f"WARNING: missing data for subj {subj_N}")
            print(f"\t {n_trials} trials found")
            continue

        filtered_events = list(set(filtered_events))

        epochs = mne.Epochs(
            subj_eeg_data,
            # event_id=["trial_end"],
            event_id=filtered_events,
            tmin=-1.0,
            tmax=0,
            baseline=None,
            verbose=False,
        )

        eeg_data[subj_N] = epochs  # .get_data(picks=c.EEG_CHAN_GROUPS.all)
        behav_data.append(subj_behav_data)

    included_subjects = list(eeg_data.keys())
    eeg_data = list(eeg_data.values())

    behav_data = pd.concat(behav_data)

    # ! TEMP
    eeg_data_arr = np.array([d.get_data(picks=c.EEG_CHAN_GROUPS.all) for d in eeg_data])

    eeg_data_group_avg = eeg_data_arr[1:].mean(axis=0)
    info = mne.create_info(c.EEG_CHAN_GROUPS.all, 2048, ch_types="eeg", verbose=None)
    eeg_data_group_avg = mne.EpochsArray(eeg_data_group_avg, info)
    montage = c.EEG_MONTAGE
    eeg_data_group_avg.set_montage(montage)
    eeg_data_group_avg.average().plot()
    plt.plot(eeg_data_group_avg[0].T)
    # ! TEMP

    evokeds = [d.average() for d in eeg_data]
    for evoked in evokeds:
        evoked.plot()

    # eeg_data[0][:1].average().plot()

    # * ----------------------------------------
    # * ----------------------------------------

    group_eeg_data_seq_lvl = []
    group_eeg_data_patt_lvl = []

    for subj_N in tqdm(subjects):
        eeg_fpaths = sorted(
            [f for f in preprocessed_eeg_data if f"subj_{subj_N:02}" in f.name]
        )

        sess_Ns = [int(re.search(r"subj_\d{2}(\d{2})", f.name)[1]) for f in eeg_fpaths]

        raw_behav = pd.concat(
            [
                load_and_clean_behav_data(c.DATA_DIR, subj_N, sess_N)
                for sess_N in sess_Ns
            ]
        )
        raw_behav.reset_index(inplace=True)

        subj_eeg_data = [
            mne.io.read_raw(f, preload=False, verbose="WARNING") for f in eeg_fpaths
        ]

        epochs = [
            mne.Epochs(
                d,
                # event_id=["trial_end"],
                event_id=["a", "x", "m", "l", "timeout"],
                tmin=-1.0,
                tmax=0,
                baseline=None,
                verbose=False,
            )
            for d in subj_eeg_data
        ]

        # epochs = mne.concatenate_epochs(epochs)
        for i, eps in enumerate(epochs, start=1):
            evoked = eps.average()

            fig = evoked.plot()
            fig.savefig(save_dir / f"response_lock_ERP-subj_{subj_N}-{i}", dpi=300)

        # del subj_eeg_data

        eeg_data_seq_lvl = [ep.get_data(picks=c.EEG_CHAN_GROUPS.all) for ep in epochs]

        # * Remove practice trials
        if eeg_data_seq_lvl[0].shape[0] > 80:
            eeg_data_seq_lvl[0] = eeg_data_seq_lvl[0][3:]

        eeg_data_seq_lvl = np.concatenate(eeg_data_seq_lvl)

        if eeg_data_seq_lvl.shape[0] != raw_behav.shape[0]:
            print(
                f"WARNING: different number of trials between EEG and behavioral data for subj {subj_N}. Skipping..."
            )
            continue

        # * Reorder the data
        reordered_inds = reorder_item_ids(
            original_order_df=raw_behav,
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )
        raw_behav = raw_behav.iloc[reordered_inds]
        raw_behav.reset_index(inplace=True)
        eeg_data_seq_lvl = eeg_data_seq_lvl[reordered_inds]

        #! TEMP
        # eeg_data_seq_lvl = eeg_data_seq_lvl[:, :63, :]
        #! TEMP

        group_eeg_data_seq_lvl.append(eeg_data_seq_lvl)

        # * Get reference RDMs
        ref_rdm_seq_lvl, ref_rdm_patt_lvl = get_reference_rdms(
            eeg_data_seq_lvl.shape[0], int(eeg_data_seq_lvl.shape[0] / 8)
        )

        ref_rdm_seq_lvl = RDMs(
            dissimilarities=ref_rdm_seq_lvl[None, :],
            pattern_descriptors={"patterns": list(raw_behav["pattern"])},
        )

        ref_rdm_patt_lvl = RDMs(
            dissimilarities=ref_rdm_patt_lvl[None, :],
            pattern_descriptors={"patterns": c.PATTERNS},
        )
        # * ----------------------------------------
        # * Sequence level analysis
        # * ----------------------------------------

        dataset = Dataset(
            measurements=eeg_data_seq_lvl[:, :, :].mean(axis=1),
            obs_descriptors={
                "item_ids": list(raw_behav["item_id"]),
                "patterns": list(raw_behav["pattern"]),
            },
        )

        rdm_sequence_lvl = calc_rdm(dataset, method=dissimilarity_metric)

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-sequence_lvl-all_chans.hdf5"
        fpath = save_dir / fname
        rdm_sequence_lvl.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm_sequence_lvl, "patterns", True)
        ax.set_title(f"RDM - subj {subj_N:02} - sequence level \n all chans")
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close()

        temporal_ds = TemporalDataset(
            measurements=eeg_data_seq_lvl,
            obs_descriptors={
                "item_ids": list(raw_behav["item_id"]),
                "patterns": list(raw_behav["pattern"]),
            },
        )

        rdms_per_timestep = calc_rdm_movie(temporal_ds, method=dissimilarity_metric)

        rdm_comparison = np.array(
            [
                compare_rdms(ref_rdm_seq_lvl, ts_rdm, method=similarity_metric).item()
                for ts_rdm in rdms_per_timestep
            ]
        )

        fig, ax = plt.subplots()
        ax.plot(rdm_comparison)  # , marker="o")
        title = f"Timestep Similarity with Reference RDM\nSubj {subj_N}"
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Correlation")
        ax.grid(True, ls="--")
        fname = f"{title.lower().replace(' ', '_').replace('\n', '_')}.png"
        fpath = save_dir / fname
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()

        # for i in np.linspace(0, rdms_per_timestep.n_rdm - 1).astype(int):
        #     plot_rdm(rdms_per_timestep[i], cluster_name="patterns")
        #     plt.show()

        # * Select the timepoint RDM that is most similar to the reference RDM
        best_timestep_rdm_ind = np.argmax(rdm_comparison)
        rdm_sequence_lvl_tp = rdms_per_timestep[best_timestep_rdm_ind]

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-sequence_lvl-all_chans-ts_{best_timestep_rdm_ind}.hdf5"
        fpath = save_dir / fname
        rdm_sequence_lvl_tp.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm_sequence_lvl_tp, "patterns", True)
        ax.set_title(
            f"RDM - subj {subj_N:02} - sequence level \n all chans - Timestep {best_timestep_rdm_ind}"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close()

        # * ----------------------------------------
        # * Pattern level analysis
        # * ----------------------------------------
        eeg_data_patt_lvl = {p: [] for p in c.PATTERNS}

        for i, patt in raw_behav["pattern"].items():
            eeg_data_patt_lvl[patt].append(eeg_data_seq_lvl[i])

        eeg_data_patt_lvl = np.array(
            [np.array(v) for v in eeg_data_patt_lvl.values()]
        ).mean(axis=1)

        group_eeg_data_patt_lvl.append(eeg_data_patt_lvl)

        ds = Dataset(
            eeg_data_patt_lvl.mean(axis=1), obs_descriptors={"patterns": c.PATTERNS}
        )
        rdm_patt_lvl = calc_rdm(ds, dissimilarity_metric)

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-pattern_lvl-all_chans.hdf5"
        fpath = save_dir / fname
        rdm_sequence_lvl_tp.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm_patt_lvl, "patterns")
        ax.set_title(f"RDM - subj {subj_N:02} - pattern level \n all chans")
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close()

    filtered_data = [arr for arr in group_eeg_data_seq_lvl if arr.shape[0] == 400]

    group_eeg_data_seq_lvl = np.array(filtered_data)

    fname = "all_subj-response_erp_data.npy"
    fpath = save_dir / fname
    np.save(fpath, group_eeg_data_seq_lvl, allow_pickle=True)

    group_eeg_data_patt_lvl = np.array(group_eeg_data_patt_lvl)

    ds = Dataset(
        measurements=group_eeg_data_patt_lvl[:, :, :].mean(axis=0).mean(axis=1),
        obs_descriptors={
            # "item_ids": c.ITEM_ID_SORT["item_id"],
            "patterns": c.PATTERNS,
        },
    )
    rdm = calc_rdm(ds, dissimilarity_metric)

    plot_rdm(rdm, "patterns")

    # group_eeg_data = np.load(fpath)
    # group_eeg_data.shape
    # group_eeg_data[0][:, :30].mean(axis=1).shape

    # for subj_N in range(group_eeg_data.shape[0]):
    #     ds = Dataset(
    #         measurements=group_eeg_data[subj_N][:, :30].mean(axis=1),
    #         obs_descriptors={
    #             "item_ids": c.ITEM_ID_SORT["item_id"],
    #             "patterns": c.ITEM_ID_SORT["pattern"],
    #         },
    #     )
    #     rdm = calc_rdm(ds, method=dissimilarity_metric)
    #     fig, ax = plot_rdm(rdm, "patterns")
    #     plt.show()

    # plt.close("all")

    group_eeg_data_avg = group_eeg_data_seq_lvl.mean(axis=0)

    ds = Dataset(
        measurements=group_eeg_data_avg[:, :, :].mean(axis=1),
        obs_descriptors={
            "item_ids": c.ITEM_ID_SORT["item_id"],
            "patterns": c.ITEM_ID_SORT["pattern"],
        },
    )

    rdm = calc_rdm(ds, method=dissimilarity_metric)
    plot_rdm(rdm, cluster_name="patterns", separate_clusters=True)

    temporal_ds = TemporalDataset(
        measurements=group_eeg_data_avg,
        obs_descriptors={
            "item_ids": c.ITEM_ID_SORT["item_id"],
            "patterns": c.ITEM_ID_SORT["pattern"],
        },
    )
    rdms_per_timestep = calc_rdm_movie(temporal_ds, method=dissimilarity_metric)

    rdm_comparison = np.array(
        [
            compare_rdms(ref_rdm_seq_lvl, ts_rdm, method=similarity_metric).item()
            for ts_rdm in rdms_per_timestep
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(rdm_comparison)  # , marker="o")
    title = f"Timestep Similarity with Reference RDM\nAll Subj"
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Correlation")
    ax.grid(True, ls="--")
    fname = f"{title.lower().replace(' ', '_').replace('\n', '_')}.png"
    fpath = save_dir / fname
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()

    # * Select the timepoint RDM that is most similar to the reference RDM
    ind_most_similar = np.argmax(rdm_comparison)
    rdm_sequence_lvl = rdms_per_timestep[ind_most_similar]
    plot_rdm(rdm_sequence_lvl, cluster_name="patterns")


def get_frp_rdms_all_subj(
    data_dir: Path,
    processed_data_dir: Path,
    save_dir: Path,
    dissimilarity_metric: str,
    selected_chan_group: str,
    time_window: Optional[Tuple[float, float]] = None,
):
    """TODO: summary

    Args:
        data_dir (Path): _description_
        processed_data_dir (Path): _description_
        save_dir (Path): _description_
        dissimilarity_metric (str): _description_
        selected_chan_group (str): _description_
        time_window (Optional[Tuple[float]], optional): _description_. Defaults to None.
    """

    # # ! TEMP
    # data_dir = c.DATA_DIR
    # processed_data_dir = c.EXPORT_DIR / "analyzed/subj_lvl"
    # save_dir = c.EXPORT_DIR / "analyzed/group_lvl/RSA"
    # dissimilarity_metric = "correlation"
    # similarity_metric = "cosine"
    # selected_chan_group = "frontal"
    # chan_group = selected_chan_group
    # save_dir = (
    #     c.EXPORT_DIR / f"analyzed/RSA-FRP-{chan_group}-metric_{dissimilarity_metric}"
    # )
    # time_window = (0, 0.6)
    # # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    selected_chans = c.EEG_CHAN_GROUPS[selected_chan_group]

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    frp_files = sorted(processed_data_dir.glob("sub*/sess*/sess_frps.pkl"))

    behav_data_list = []
    for subj_dir in processed_data_dir.glob("subj_*"):
        subj_N = int(subj_dir.name.split("_")[-1])
        for sess_dir in subj_dir.glob("sess_*"):
            sess_N = int(sess_dir.name.split("_")[-1])
            behav_data_list.append(load_and_clean_behav_data(data_dir, subj_N, sess_N))

    behav_data = pd.concat(behav_data_list)
    behav_data.reset_index(drop=True, inplace=True)
    del behav_data_list

    # * Replace the above with line below, but won't work if not all participants have been analyzed
    # behav_data = pd.concat(list(load_and_clean_behav_data_all(data_dir)))

    subj_data: dict = {int(subj): {} for subj in behav_data["subj_N"].unique()}

    missing_frps = []

    for frp_file in sorted(frp_files):
        subj_N = int(frp_file.parents[1].name.split("_")[-1])
        sess_N = int(frp_file.parents[0].name.split("_")[-1])

        subj_data[subj_N][sess_N] = []

        sess_df = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")
        sess_item_ids = sess_df["item_id"].to_list()
        sess_patterns = sess_df["pattern"].to_list()

        frp_data = read_file(frp_file)

        sequence_frps = frp_data["sequence"]
        # choices_frps = frp_data["choices"]

        if sess_missing_frps := [i for i, s in enumerate(sequence_frps) if s is None]:
            missing_frps.append((subj_N, sess_N, sess_missing_frps))

        if time_window:
            data_shape = (
                [s for s in sequence_frps if s is not None][0]
                .copy()
                .crop(*time_window)
                .get_data(picks=selected_chans)
                .shape
            )

            empty_data = np.zeros(data_shape)
            empty_data[:] = np.nan

            sequence_frps = [
                frp.copy().crop(*time_window).get_data(picks=selected_chans)
                if frp is not None
                else empty_data
                for frp in sequence_frps
            ]

        else:
            data_shape = (
                [s for s in sequence_frps if s is not None][0]
                .copy()
                .get_data(picks=selected_chans)
                .shape
            )

            empty_data = np.zeros(data_shape)
            empty_data[:] = np.nan

            sequence_frps = [
                frp.copy().get_data(picks=selected_chans)
                if frp is not None
                else empty_data
                for frp in sequence_frps
            ]

        subj_data[subj_N][sess_N] = [
            sess_item_ids,
            sess_patterns,
            sequence_frps,
        ]

    # * Show missing FRPs
    if missing_frps:
        print("Missing FRPs:")
        df = pd.DataFrame([[*i[:2], len(i[2])] for i in missing_frps])
        df.columns = ["subj_N", "sess_N", "n_missing"]
        df.sort_values(["n_missing"], ascending=False, inplace=True)
        display(df)

    # * Removing subjects if analyzed data is missing
    subj_data = {k: v for k, v in subj_data.items() if len(v) > 0}

    subj_rdms = []

    _group_average_rdm = []
    _group_patterns = []
    _group_item_ids = []
    _pattern_frps_avg = {p: [] for p in c.PATTERNS}

    for subj_N in sorted(subj_data.keys()):
        # * Making sure data is correctly sorted
        sessions_data = {
            k: v for k, v in sorted(subj_data[subj_N].items(), key=lambda x: x[0])
        }

        subj_item_ids = np.concatenate([i for i, _, _ in sessions_data.values()])
        subj_patterns = np.concatenate([p for _, p, _ in sessions_data.values()])
        subj_frps = np.concatenate([f for _, _, f in sessions_data.values()])
        pattern_frps: dict = {p: [] for p in c.PATTERNS}

        # * Group FRPs by pattern
        for i, frp in enumerate(subj_frps):
            pattern = subj_patterns[i]
            pattern_frps[pattern].append(frp)

        # * Compute average activity for each channel of each pattern FRP
        pattern_frps_avg_by_chan: dict = {p: [] for p in c.PATTERNS}
        for pattern, trials in pattern_frps.items():
            avg_frp_by_chan = np.nanmean(np.stack(trials), axis=0)
            pattern_frps_avg_by_chan[pattern] = avg_frp_by_chan

        # * Compute overall average FRP for each pattern
        pattern_frps_avg = {
            p: np.nanmean(v, axis=0) for p, v in pattern_frps_avg_by_chan.items()
        }

        assert list(pattern_frps.keys()) == c.PATTERNS
        assert list(pattern_frps_avg_by_chan.keys()) == c.PATTERNS
        assert list(pattern_frps_avg.keys()) == c.PATTERNS

        [_pattern_frps_avg[patt].append(frp) for patt, frp in pattern_frps_avg.items()]

        # * -------- Deal with missing data --------
        # * Get the indices of the missing data
        inds_missing_data = [
            i for i, subj_frp in enumerate(subj_frps) if np.all(np.isnan(subj_frp))
        ]
        # np.all(np.isnan(subj_frps[inds_missing_data]))

        # * Replace missing data
        overall_avg_frp = np.nanmean(subj_frps, axis=0)

        for ind in inds_missing_data:
            subj_frps[ind] = overall_avg_frp
        #     # missing_data_pattern = subj_patterns[ind]
        #     # subj_frps[ind] = pattern_frps_avg_by_chan[missing_data_pattern]

        # * -------- Reorder the FRPs --------
        reordered_inds = reorder_item_ids(
            original_order_df=pd.DataFrame(
                {"item_id": subj_item_ids, "pattern": subj_patterns}
            ),
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )

        subj_frps = subj_frps[reordered_inds]
        subj_patterns = subj_patterns[reordered_inds]
        subj_item_ids = subj_item_ids[reordered_inds]

        # * Get reference RDMs
        ref_rdm_seq_lvl, ref_rdm_patt_lvl = get_reference_rdms(
            subj_frps.shape[0], int(subj_frps.shape[0] / 8)
        )

        ref_rdm_seq_lvl = RDMs(
            dissimilarities=ref_rdm_seq_lvl[None, :],
            pattern_descriptors={"patterns": subj_patterns},
        )

        ref_rdm_patt_lvl = RDMs(
            dissimilarities=ref_rdm_patt_lvl[None, :],
            pattern_descriptors={"patterns": c.PATTERNS},
        )

        # * -------- Compute Subject-Sequence Level RDM at every time step  --------
        temporal_ds = TemporalDataset(
            measurements=subj_frps,
            descriptors={"subj_N": subj_N},
            obs_descriptors={"patterns": subj_patterns, "item_ids": subj_item_ids},
            time_descriptors={"time": np.arange(subj_frps.shape[-1])},
        )

        rdms_per_timestep = calc_rdm_movie(temporal_ds, method=dissimilarity_metric)

        rdm_comparison = np.array(
            [
                compare_rdms(ref_rdm_seq_lvl, ts_rdm).item()
                for ts_rdm in rdms_per_timestep
            ]
        )

        fig, ax = plt.subplots()
        ax.plot(rdm_comparison, marker="o")
        title = f"Timestep Similarity with Reference RDM\nSubj {subj_N:02}"
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Correlation")
        ax.grid(True, ls="--")
        fname = f"{title.lower().replace(' ', '_').replace('\n', '_')}.png"
        fpath = save_dir / fname
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close()

        best_timestep_rdm_ind = np.argmax(rdm_comparison)
        rdm = rdms_per_timestep[best_timestep_rdm_ind]

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-sequence_lvl-{selected_chan_group}-ts_{best_timestep_rdm_ind}.hdf5"
        fpath = save_dir / fname
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm, "patterns")
        ax.set_title(
            f"RDM - subj {subj_N:02} - sequence level \n {selected_chan_group} chans - Timestep {best_timestep_rdm_ind}"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

        # * -------- Compute Subject-Sequence Level RDM  --------

        # * Flatten FRPs across channels into a new array of shape (n_items, n_channels * n_timepoints)
        # flattened_frps = subj_frps.reshape(subj_frps.shape[0], -1)

        # * Compute the average FRP across all channels
        subj_frps = subj_frps.mean(axis=1)

        dataset = Dataset(
            measurements=subj_frps,
            descriptors={"subj_N": subj_N},
            obs_descriptors={"patterns": subj_patterns, "item_ids": subj_item_ids},
        )
        fname = (
            f"dataset-human-subj_{subj_N:02}-sequence_lvl-{selected_chan_group}.hdf5"
        )
        fpath = save_dir / fname
        dataset.save(fpath, file_type="hdf5", overwrite=True)

        # * Compute RDM
        rdm = calc_rdm(dataset, method=dissimilarity_metric)

        subj_rdms.append(rdm)

        # * If number of items == 400, then subject completed the whole experiment
        # * Append the RDM to the group average RDM
        rdm_array = rdm.get_matrices()[0]
        if rdm_array.shape[0] == 400:
            _group_average_rdm.append(rdm_array)
            _group_patterns.append(subj_patterns)
            _group_item_ids.append(subj_item_ids)

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-sequence_lvl-{selected_chan_group}.hdf5"
        fpath = save_dir / fname
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=True)
        ax.set_title(
            f"RDM - subj {subj_N:02} - sequence level \n {selected_chan_group} chans"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

        # * -------- Compute Subject-Pattern Level RDM  --------
        # * Group all the average FRPs into a single array (n_patterns, n_timepoints)
        # pattern_frps_avg = np.array([v for v in pattern_frps_avg.values()])

        dataset = Dataset(
            measurements=np.array([v for v in pattern_frps_avg.values()]),
            descriptors={"subj_N": subj_N},
            # channel_descriptors={"names": [f"average_of_{selected_chan_group}_chans"]},
            obs_descriptors={"patterns": list(pattern_frps_avg.keys())},
        )
        fname = f"dataset-human-subj_{subj_N:02}-pattern_lvl-{selected_chan_group}.hdf5"
        fpath = save_dir / fname
        dataset.save(fpath, file_type="hdf5", overwrite=True)

        # * Compute RDM
        rdm = calc_rdm(dataset, method=dissimilarity_metric)

        fname = f"rdm-human-subj_{subj_N:02}-pattern_lvl-{selected_chan_group}.hdf5"
        fpath = save_dir / fname
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=False)
        ax.set_title(
            f"RDM - subj {subj_N:02} - pattern level \n {selected_chan_group} chans"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    # * -------- Compute Group-Sequence Level RDM  --------
    group_average_rdm = np.nanmean(np.array(_group_average_rdm), axis=0)
    group_average_rdm = group_average_rdm[None, :]
    del _group_average_rdm

    group_patterns = _group_patterns[0]
    group_item_ids = _group_item_ids[0]
    del _group_patterns, _group_item_ids

    rdm = rsatoolbox.rdm.rdms.RDMs(
        dissimilarities=group_average_rdm,
        dissimilarity_measure=dissimilarity_metric,
        descriptors={},
        rdm_descriptors={"group_average_rdm": [selected_chan_group]},
        pattern_descriptors={"patterns": group_patterns, "item_ids": group_item_ids},
        # pattern_descriptors=subj_rdms[0].pattern_descriptors,
    )

    fname = f"rdm-human-group_avg-sequence_lvl-{selected_chan_group}.hdf5"
    fpath = save_dir / fname
    rdm.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=True)
    ax.set_title(f"RDM - Group Average - sequence level \n {selected_chan_group} chans")
    fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    # * -------- Compute Group-Pattern Level RDM  --------
    pattern_frps_avg = {k: np.nanmean(v, axis=0) for k, v in _pattern_frps_avg.items()}
    del _pattern_frps_avg

    assert list(pattern_frps_avg.keys()) == c.PATTERNS

    pattern_frps_avg = np.array([v for v in pattern_frps_avg.values()])

    # * -------- Plot the group average FRPs --------
    x_ticks = np.arange(0, pattern_frps_avg.shape[1], 200).astype(float)
    x_ticklabels = np.round(x_ticks / c.EEG_SFREQ, 2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=300)
    ax1.plot(pattern_frps_avg.T)  # , alpha=0.8)
    ax1.sharex(ax2)
    # ax1.sharey(ax2)
    ax1.set_title(f"Group Average FRPs - {selected_chan_group} chans")
    ax1.legend(c.PATTERNS, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(axis="both", ls="--")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticklabels)
    ax1.set_ylabel("Amplitude (uV)")
    ax1.set_xlabel("Time (s)")
    line_colors = [l.get_color() for l in ax1.get_lines()]
    for i, patt_frp in enumerate(pattern_frps_avg):
        lat = patt_frp.argmin()
        amp = patt_frp[lat]
        ax2.scatter(lat, amp, marker="+", color=line_colors[i], label=c.PATTERNS[i])
    ax2.set_title("Negative Peak Latency and Amplitude")
    ax2.set_ylabel("Amplitude (uV)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(axis="both", ls="--")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticklabels)
    fig.tight_layout()

    fig.savefig(
        save_dir / f"group_avg_FRPs-{selected_chan_group}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    # * ---------------------------------------------------

    dataset = Dataset(
        measurements=pattern_frps_avg,
        descriptors={},
        obs_descriptors={"patterns": c.PATTERNS},
    )
    fname = f"dataset-human-group_avg-pattern_lvl-{selected_chan_group}.hdf5"
    fpath = save_dir / fname
    dataset.save(fpath, file_type="hdf5", overwrite=True)

    rdm = calc_rdm(dataset, method=dissimilarity_metric)

    fname = f"rdm-human-group_avg-pattern_lvl-{selected_chan_group}.hdf5"
    fpath = save_dir / fname
    rdm.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=False)
    ax.set_title(f"RDM - Group Average - pattern level \n {selected_chan_group} chans")
    fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def get_frp_amp_rdms_all_subj(
    data_dir: Path,
    processed_data_dir: Path,
    save_dir: Path,
    dissimilarity_metric: str,
    selected_chan_group: str,
    # TODO: time_window: Optional[Tuple[float]] = None,
):
    """TODO:_summary_

    Args:
        data_dir (Path): _description_
        processed_data_dir (Path): _description_
        save_dir (Path): _description_
        dissimilarity_metric (str): _description_
        selected_chan_group (str): _description_
    """

    # # # ! TEMP
    # data_dir = c.DATA_DIR
    # processed_data_dir = c.EXPORT_DIR / "analyzed/subj_lvl"
    # dissimilarity_metric = "euclidean"
    # selected_chan_group = "occipital"
    # chan_group = selected_chan_group
    # save_dir = (
    #     c.EXPORT_DIR
    #     / f"analyzed/RSA-FRP_AMP-{chan_group}-metric_{dissimilarity_metric}"
    # )
    # # # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    selected_chans = c.EEG_CHAN_GROUPS[selected_chan_group]

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    frp_files = sorted(processed_data_dir.glob("sub*/sess*/sess_frps.pkl"))

    behav_data_list = []
    for subj_dir in processed_data_dir.glob("subj_*"):
        subj_N = int(subj_dir.name.split("_")[-1])
        for sess_dir in subj_dir.glob("sess_*"):
            sess_N = int(sess_dir.name.split("_")[-1])
            behav_data_list.append(load_and_clean_behav_data(data_dir, subj_N, sess_N))

    behav_data = pd.concat(behav_data_list)
    behav_data.reset_index(drop=True, inplace=True)
    del behav_data_list

    # * Replace the above with line below, but won't work if not all participants have been analyzed
    # behav_data = pd.concat(list(load_and_clean_behav_data_all(data_dir)))

    subj_data: dict = {int(subj): {} for subj in behav_data["subj_N"].unique()}

    missing_frps = []
    for frp_file in sorted(frp_files):
        subj_N = int(frp_file.parents[1].name.split("_")[-1])
        sess_N = int(frp_file.parents[0].name.split("_")[-1])

        subj_data[subj_N][sess_N] = []

        sess_df = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")
        sess_item_ids = sess_df["item_id"].to_list()
        sess_patterns = sess_df["pattern"].to_list()

        frp_data = read_file(frp_file)

        sequence_frps = frp_data["sequence"]
        # choices_frps = frp_data["choices"]

        if sess_missing_frps := [i for i, s in enumerate(sequence_frps) if s is None]:
            missing_frps.append((subj_N, sess_N, sess_missing_frps))

        neg_peaks_lat_and_amp = [
            get_neg_erp_peak(frp, (0.02, 0.2), selected_chans)
            if frp is not None
            else (np.nan, np.nan)
            for frp in sequence_frps
        ]

        neg_peaks_amp = [amp for _, amp in neg_peaks_lat_and_amp]

        subj_data[subj_N][sess_N] = [
            sess_item_ids,
            sess_patterns,
            neg_peaks_amp,
            # neg_peaks_lat_and_amp,  # neg_peaks_amp,  # sequence_frps,
            # timepoints,
        ]

    # * Show missing FRPs
    if missing_frps:
        print("Missing FRPs:")
        df = pd.DataFrame([[*i[:2], len(i[2])] for i in missing_frps])
        df.columns = ["subj_N", "sess_N", "n_missing"]
        df.sort_values(["n_missing"], ascending=False, inplace=True)
        display(df)

    # * Removing subjects if analyzed data is missing
    subj_data = {k: v for k, v in subj_data.items() if len(v) > 0}

    subj_rdms = []
    _group_average_rdm = []
    _group_patterns = []
    _pattern_neg_amp_avg = {p: [] for p in c.PATTERNS}

    for subj_N in sorted(subj_data.keys()):
        # * Making sure data is correctly sorted
        sessions_data = {
            k: v for k, v in sorted(subj_data[subj_N].items(), key=lambda x: x[0])
        }

        subj_item_ids = np.concatenate([i for i, _, _ in sessions_data.values()])
        subj_patterns = np.concatenate([p for _, p, _ in sessions_data.values()])
        subj_peak_amps = np.concatenate([f for _, _, f in sessions_data.values()])
        pattern_peak_amps: dict = {p: [] for p in c.PATTERNS}

        # * Group FRP negative peak amplitudes by pattern
        for i, frp in enumerate(subj_peak_amps):
            pattern = subj_patterns[i]
            pattern_peak_amps[pattern].append(frp)

        # * Compute average negative peak amplitude
        pattern_neg_amp_avg: dict = {p: [] for p in c.PATTERNS}
        for pattern, trials in pattern_peak_amps.items():
            avg_neg_peak_amp = np.nanmean(np.stack(trials))
            pattern_neg_amp_avg[pattern] = avg_neg_peak_amp

        assert list(pattern_neg_amp_avg.keys()) == c.PATTERNS

        for patt, neg_amp in pattern_neg_amp_avg.items():
            _pattern_neg_amp_avg[patt].append(neg_amp)

        # * -------- Deal with missing data --------
        # * Get the indices of the missing data
        inds_missing_data = [
            i for i, subj_frp in enumerate(subj_peak_amps) if np.all(np.isnan(subj_frp))
        ]
        # np.all(np.isnan(subj_peak_amps[inds_missing_data]))

        # * Replace missing data
        overall_avg_neg_amp = np.nanmean(subj_peak_amps)

        for ind in inds_missing_data:
            subj_peak_amps[ind] = overall_avg_neg_amp
        #     # missing_data_pattern = subj_patterns[ind]
        #     # subj_frps[ind] = pattern_frps_avg_by_chan[missing_data_pattern]
        # assert np.all(subj_peak_amps[inds_missing_data] == overall_avg_neg_amp)

        # * -------- Reorder the FRPs --------
        reordered_inds = reorder_item_ids(
            original_order_df=pd.DataFrame(
                {"item_id": subj_item_ids, "pattern": subj_patterns}
            ),
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )

        subj_peak_amps = subj_peak_amps[reordered_inds]
        subj_patterns = subj_patterns[reordered_inds]
        subj_item_ids = subj_item_ids[reordered_inds]

        # * -------- Compute Subject-Item Level RDM  --------
        subj_peak_amps = subj_peak_amps[:, None]

        dataset = Dataset(
            measurements=subj_peak_amps,
            descriptors={"subj_N": subj_N},
            obs_descriptors={"patterns": subj_patterns, "item_ids": subj_item_ids},
        )

        # * Compute RDM
        rdm = calc_rdm(dataset, method=dissimilarity_metric)

        subj_rdms.append(rdm)

        # * If number of items == 400, then subject completed the whole experiment
        # * Append the RDM to the group average RDM
        rdm_array = rdm.get_matrices()[0]
        if rdm_array.shape[0] == 400:
            _group_average_rdm.append(rdm_array)
            _group_patterns.append(subj_patterns)

        # * Save the RDM
        fname = f"rdm-human-subj_{subj_N:02}-item_lvl-{selected_chan_group}.hdf5"
        fpath = save_dir / fname
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=True)
        ax.set_title(
            f"RDM - subj {subj_N:02} - item level \n {selected_chan_group} chans"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

        # * -------- Compute Subject-Pattern Level RDM  --------
        # * Group all the average FRPs into a single array (n_patterns, n_timepoints)
        # pattern_frps_avg = np.array([v for v in pattern_frps_avg.values()])

        dataset = Dataset(
            measurements=np.array([v for v in pattern_neg_amp_avg.values()])[:, None],
            descriptors={"subj_N": subj_N},
            # channel_descriptors={"names": [f"average_of_{selected_chan_group}_chans"]},
            obs_descriptors={"patterns": list(pattern_neg_amp_avg.keys())},
        )
        # ! HERE

        # * Compute RDM
        rdm = calc_rdm(dataset, method=dissimilarity_metric)

        fname = f"rdm-human-subj_{subj_N:02}-pattern_lvl-{selected_chan_group}.hdf5"
        fpath = save_dir / fname
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=False)
        ax.set_title(
            f"RDM - subj {subj_N:02} - pattern level \n {selected_chan_group} chans"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    # * -------- Compute Group-Item Level RDM  --------
    group_average_rdm = np.nanmean(np.array(_group_average_rdm), axis=0)
    group_average_rdm = group_average_rdm[None, :]
    del _group_average_rdm

    group_patterns = _group_patterns[0]
    del _group_patterns

    rdm = rsatoolbox.rdm.rdms.RDMs(
        dissimilarities=group_average_rdm,
        dissimilarity_measure=dissimilarity_metric,
        descriptors={},
        rdm_descriptors={"group_average_rdm": [selected_chan_group]},
        pattern_descriptors={"patterns": group_patterns},
        # pattern_descriptors=subj_rdms[0].pattern_descriptors,
    )

    fname = f"rdm-human-group_avg-item_lvl-{selected_chan_group}.hdf5"
    fpath = save_dir / fname
    rdm.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=True)
    ax.set_title(f"RDM - Group Average - item level \n {selected_chan_group} chans")
    fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    # * -------- Compute Group-Pattern Level RDM  --------
    pattern_neg_amp_avg = {
        k: np.nanmean(v, axis=0) for k, v in _pattern_neg_amp_avg.items()
    }
    del _pattern_neg_amp_avg

    patterns = list(pattern_neg_amp_avg.keys())
    assert patterns == c.PATTERNS

    pattern_neg_amp_avg = np.array([v for v in pattern_neg_amp_avg.values()])[:, None]

    dataset = Dataset(
        measurements=pattern_neg_amp_avg,
        descriptors={},
        obs_descriptors={"patterns": patterns},
    )

    rdm = calc_rdm(dataset, method=dissimilarity_metric)

    fname = f"rdm-human-group_avg-pattern_lvl-{selected_chan_group}.hdf5"
    fpath = save_dir / fname
    rdm.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    fig, ax = plot_rdm(rdm, cluster_name="patterns", separate_clusters=False)
    ax.set_title(f"RDM - Group Average - pattern level \n {selected_chan_group} chans")
    fig.savefig(fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def rsa_between_subjects(rdm_files: List[Path], save_dir: Path, similarity_metric: str):
    # ! TEMP
    # similarity_metric = "cosine"
    # ! TEMP

    subj_rdms = [rsatoolbox.rdm.rdms.load_rdm(str(f)) for f in rdm_files]

    similarity_matrix = np.zeros((len(subj_rdms), len(subj_rdms)))

    for i, rdm1 in enumerate(subj_rdms):
        for j, rdm2 in enumerate(subj_rdms):
            similarity_matrix[i, j] = compare_rdms(
                rdm1, rdm2, method=similarity_metric
            ).item()

    ticks = np.arange(1, len(subj_rdms), 2)
    ticklabels = [f"{i + 1}" for i in ticks]
    fig, ax = plt.subplots()
    ax.imshow(similarity_matrix)
    ax.set_title(f"Similarity Matrix - {chan_group}")
    colorbar = ax.figure.colorbar(ax.imshow(similarity_matrix))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    plt.show()


# * ----------------------------------------


def get_accuracy_rdms(
    data_dir: Path, save_dir: Path, dissimilarity_metric="euclidean", show_plots=True
):
    save_dir.mkdir(exist_ok=True, parents=True)

    behav_data = pd.concat(list(load_and_clean_behav_data_all(data_dir=c.DATA_DIR)))
    behav_data.reset_index(drop=True, inplace=True)

    subjects = [int(i) for i in behav_data["subj_N"].unique()]
    assert sorted(behav_data["pattern"].unique()) == c.PATTERNS

    accuracy_by_pattern = (
        behav_data.groupby(["subj_N", "pattern"])["correct"].mean().reset_index()
    )
    accuracy_by_pattern.rename(columns={"correct": "accuracy"}, inplace=True)

    # * -- Compute Individual RDM --
    subj_rdms = []

    for subj_N in subjects:
        # * ---- Item Level ----
        # # * Convert to Dataset for RDM calculation
        # subj_behav_data = behav_data.query("subj_N == @subj_N")
        # subj_behav_data.reset_index(drop=True, inplace=True)

        # item_lvl_acc = subj_behav_data["correct"].astype(int).values
        # item_lvl_acc = item_lvl_acc[:, None]

        # reordered_inds = reorder_item_ids(
        #     subj_behav_data[["item_id", "pattern"]],
        #     ITEM_ID_SORT[["item_id", "pattern"]],
        # )
        # subj_patterns = subj_behav_data["pattern"].values[reordered_inds]
        # item_lvl_acc = item_lvl_acc[reordered_inds]

        # behav_dataset = Dataset(
        #     measurements=item_lvl_acc,
        #     obs_descriptors={"patterns": subj_patterns},
        # )

        # rdm = calc_rdm(behav_dataset, method="euclidean")
        # subj_rdms.append(rdm)

        # subj_N = f"{subj_N:02}"
        # fpath = save_dir / f"rdm-human-subj_{subj_N}-by_pattern-accuracy.hdf5"
        # # rdm.save(fpath, file_type="hdf5", overwrite=True)

        # # * Plot the RDM and save the figure
        # tick_marks = np.arange(0, len(c.PATTERNS), 1)
        # tick_labels = c.PATTERNS

        # fig, ax = plt.subplots()
        # im = ax.imshow(rdm.get_matrices()[0])
        # ax.set_title(f"subj {subj_N} RDM")
        # fig.colorbar(im, ax=ax)
        # # ax.set_yticks(tick_marks)
        # # ax.set_yticklabels(tick_labels)
        # # ax.set_xticks(tick_marks)
        # # ax.set_xticklabels(tick_labels, rotation=90)
        # # fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
        # plt.show()
        # plt.close()

        # * ---- Pattern Level ----
        accuracy_by_pattern_subj = accuracy_by_pattern.query("subj_N == @subj_N").copy()
        accuracy_by_pattern_subj.sort_values("pattern", inplace=True)

        # assert accuracy_by_pattern_subj["pattern"].to_list() == c.PATTERNS

        # * Convert to Dataset for RDM calculation
        behav_dataset = Dataset(
            measurements=accuracy_by_pattern_subj["accuracy"].values[:, None],
            obs_descriptors={"patterns": c.PATTERNS},
        )
        rdm = calc_rdm(behav_dataset, method=dissimilarity_metric)
        subj_rdms.append(rdm)

        fpath = save_dir / f"rdm-human-subj_{subj_N:02}-by_pattern-accuracy.hdf5"
        rdm.save(fpath, file_type="hdf5", overwrite=True)

        # ----------------------------------------
        tick_marks = np.arange(0, len(c.PATTERNS), 1)
        tick_labels = c.PATTERNS

        fig, ax = plt.subplots()
        im = ax.imshow(rdm.get_matrices()[0])
        ax.set_title(f"subj {subj_N:02} RDM")
        # plot_matrix(
        #     rdm,
        #     labels=c.PATTERNS,
        #     title="ERP RDMs",
        #     show_values=True,
        #     norm="max",
        #     as_pct=True,
        #     ax=ax,
        # )
        fig.colorbar(im, ax=ax)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(tick_labels)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(tick_labels, rotation=90)
        fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()

    # * -- Compute Group Average RDM --
    group_average_rdm: list = []

    pattern_descriptors = []
    for rdm in subj_rdms:
        group_average_rdm.append(rdm.get_matrices()[0])
        pattern_descriptors.append(rdm.pattern_descriptors["patterns"])

    assert all([p == c.PATTERNS for p in pattern_descriptors])
    del pattern_descriptors

    group_average_rdm: np.ndarray = np.nanmean(np.array(group_average_rdm), axis=0)
    group_average_rdm = group_average_rdm[None, :]

    group_average_rdm: rsatoolbox.rdm.rdms.RDMs = rsatoolbox.rdm.rdms.RDMs(
        dissimilarities=group_average_rdm,
        dissimilarity_measure=dissimilarity_metric,
        descriptors={},
        rdm_descriptors={"group_average_rdm": ["accuracy"]},
        pattern_descriptors={"patterns": c.PATTERNS},
    )

    fpath = save_dir / f"rdm-human-group_avg-by_pattern-accuracy.hdf5"

    group_average_rdm.save(fpath, file_type="hdf5", overwrite=True)

    fig, ax = plt.subplots()
    im = ax.imshow(group_average_rdm.get_matrices()[0])
    ax.set_title("Group Average RDM")
    fig.colorbar(im, ax=ax)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_labels, rotation=90)
    fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.show()


def get_rdms_eye_gaze(gaze_data: pd.DataFrame, method="euclidean", show_plots=True):
    gaze_duration_for_sequences = (
        gaze_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern"])["mean_fix_duration"]
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


def get_rdms_negative_peak_eeg(
    erp_data,
    selected_chans,
    time_window=(0, 0.2),
    unit="uV",
    method="euclidean",
    show_plots=True,
):
    # ! TEMP
    # method = 'euclidean'
    # erp_data = subj_pattern_erps.copy()
    # ! TEMP

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


def get_rdms_negative_peak_eeg_old(
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


def get_rdms_negative_peak_frp_all_subj(
    data_dir: Path,
    processed_data_dir: Path,
    save_dir: Path,
    dissimilarity_metric: str,
    selected_chan_group: str,
):
    # erp_data,
    # selected_chans,
    # method,
    # time_window=(0, 0.2),
    # unit="uV",
    # show_plots=True,

    # ! TEMP
    # method = 'corr'
    # erp_data = subj_pattern_erps.copy()
    # ! TEMP

    return


# * ####################################################################################
# * INSPECT RESULTS
# * ####################################################################################


def inspect_results():
    # plt.close("all")
    # plt.get_backend()
    # plt.switch_backend("webagg")
    # plt.switch_backend(mpl_backend)

    # res_dir = WD / "results/analyzed"  # /Oct27-Seq_and_choices"
    res_dir = Path(
        "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/Lab/analyzed/subj_lvl"
    )

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    erps = {}
    erp_files = sorted(res_dir.glob("sub*/sess*/sess_frps.pkl"))
    behav_files = sorted(c.DATA_DIR.glob("sub*/sess*/*behav.csv"))
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    behav_files = list(c.DATA_DIR.rglob("*behav.csv"))

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

    # * --------------- behavioral data ---------------
    behav_data = pd.concat(behav_data, axis=0)

    # behav_data = behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
    behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
    behav_data.reset_index(drop=True, inplace=True)

    # * --------------- gaze data ---------------
    gaze_data = pd.concat(gaze_data, axis=0)

    gaze_data = gaze_data.merge(
        behav_data[["subj_N", "sess_N", "trial_N", "pattern", "item_id"]],
        on=["subj_N", "sess_N", "trial_N"],
    )

    gaze_data = gaze_data.sort_values(
        ["subj_N", "sess_N", "trial_N", "target_ind"]
    ).reset_index(drop=True)
    gaze_data.rename(columns={"mean_duration": "mean_fix_duration"}, inplace=True)

    gaze_data["mean_fix_duration"].mean()
    gaze_data.groupby(["subj_N", "sess_N"])["mean_pupil_diam"].mean()
    gaze_data.groupby(["subj_N"])["mean_pupil_diam"].mean()

    pupil_diam_by_subj_pattern = (
        gaze_data.groupby(["subj_N", "pattern"])["mean_pupil_diam"].mean().reset_index()
    )

    fix_duration_by_subj_pattern = (
        gaze_data.groupby(["subj_N", "pattern"])["mean_fix_duration"]
        .mean()
        .reset_index()
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
            c.DATA_DIR.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
        )[0]
        raw_behav = pd.read_csv(behav_file, index_col=0)

        sess_bad_chans = c.ALL_BAD_CHANS[f"subj_{subj_N}"][f"sess_{sess_N}"]

        selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
            prepare_eeg_data_for_plot(
                c.EEG_CHAN_GROUPS,
                c.EEG_MONTAGE,
                c.NON_EEG_CHANS,
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

        # * check if any erp type is an empty list:
        erp_check = {
            erp_type: eval(f"len({erp_type})")
            for erp_type in ["sequence_erps", "choice_erps"]
        }
        if not all(erp_check.values()):
            empty_lists = [erp_type for erp_type, l in erp_check.items() if l == 0]
            print(f"WARNING: subj {subj_N} - sess {sess_N}: {empty_lists} are empty")
            continue

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
                    c.EEG_SFREQ,
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
                    c.EEG_SFREQ,
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
            c.EEG_CHAN_GROUPS,
            c.EEG_MONTAGE,
            c.NON_EEG_CHANS,
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
            c.EEG_SFREQ,
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
            c.EEG_SFREQ,
            eeg_baseline,
            vlines=None,
            title="All Subjects\n" + f"Pattern: {pattern}",
        )
        plt.show()

    # * ################################################################################
    # * Get RDMs
    # * ################################################################################

    # * --------------- Get the negative FRP peaks ---------------
    subj_neg_peaks = pd.DataFrame()

    for subj_N in subj_pattern_frps.keys():
        for pattern, erp in subj_pattern_frps[subj_N].items():
            latency, amplitude = get_neg_erp_peak(
                erp, time_window=(0, 0.2), selected_chans=c.EEG_CHAN_GROUPS["occipital"]
            )

            temp_data = {
                "subj_N": subj_N,
                "pattern": pattern,
                "latency": latency,
                "amplitude": amplitude,
            }

            subj_neg_peaks = pd.concat([subj_neg_peaks, pd.DataFrame([temp_data])])

    subj_neg_peaks.reset_index(drop=True, inplace=True)
    subj_neg_peaks.to_csv(WD / "results/subj_neg_peaks.csv", index=False)
    # TODO: change above to relative path

    # * --------------- Get the behavioral data summary ---------------
    behav_data["rt"] = behav_data["rt"].replace("timeout", np.nan)
    behav_data["rt"] = behav_data["rt"].astype(float)
    behav_data["correct"] = (
        behav_data["correct"]
        .replace({"invalid": False, "False": False, "True": True})
        .astype(bool)
    )
    behav_rt_correct_per_pattern = (
        behav_data.groupby(["subj_N", "pattern"])[["rt", "correct"]]
        .mean()
        .reset_index()
    )

    data_summary = behav_rt_correct_per_pattern.merge(
        subj_neg_peaks, on=["subj_N", "pattern"]
    )
    data_summary = data_summary.merge(
        pupil_diam_by_subj_pattern, on=["subj_N", "pattern"]
    )
    data_summary = data_summary.merge(
        fix_duration_by_subj_pattern, on=["subj_N", "pattern"]
    )
    data_summary.rename(columns={"correct": "accuracy"}, inplace=True)
    data_summary.iloc[:, 2:].corr()

    data_summary.to_csv(res_dir / "lab_data_summary.csv", index=False)

    data_summary_by_pattern = data_summary.groupby("pattern").mean()
    data_summary_by_pattern = data_summary_by_pattern.loc[
        :, [c for c in data_summary_by_pattern.columns if c != "subj_N"]
    ]
    data_summary_by_pattern.sort_index(inplace=True)

    # * --------------- Check correlations and plot the data ---------------
    data_summary_corr = data_summary.iloc[:, 2:].corr()
    data_summary_corr_styled = apply_df_style(
        data_summary_corr, 2, vmin=-1, vmax=1
    ).set_caption("Correlation Matrix")

    data_summary_by_pattern_corr = data_summary_by_pattern.corr()
    data_summary_by_pattern_corr_styled = apply_df_style(
        data_summary_by_pattern_corr, 2, vmin=-1, vmax=1
    ).set_caption("Correlation Matrix - Data Groupped by Patterns")

    display(data_summary_corr_styled)
    display(data_summary_by_pattern_corr_styled)

    dfi.export(
        data_summary_corr_styled,
        res_dir / "lab_data_summary.png",
        table_conversion="matplotlib",
    )

    dfi.export(
        data_summary_by_pattern_corr,
        res_dir / "lab_data_summary_by_pattern_corr.png",
        table_conversion="matplotlib",
    )

    for col in data_summary_by_pattern.columns:
        fig, ax = plt.subplots()
        sns.barplot(data=data_summary_by_pattern, x=col, y="pattern", ax=ax)
        ax.set_title(col)
        fig.savefig(
            res_dir / f"lab_data_summary_{col}.png", dpi=200, bbox_inches="tight"
        )
        plt.show()

    col_combinations = list(combinations(data_summary_by_pattern_corr.columns, 2))

    for col1, col2 in col_combinations:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data_summary_by_pattern, x=col1, y=col2, ax=ax)
        ax.set_title(col)
        # fig.savefig(
        #     res_dir / f"lab_data_summary_{col}.png", dpi=200, bbox_inches="tight"
        # )
        ax.set_title(
            f"{col1} vs {col2} (corr: {data_summary_by_pattern_corr.loc[col1, col2]:.2f})"
        )
        plt.show()

    # * --------------- Statistical Tests ---------------
    from statsmodels.stats.anova import AnovaRM
    import statsmodels.stats.multicomp as mc
    import pingouin as pg
    import statsmodels.formula.api as smf

    for depvar in data_summary_corr.columns:
        print("Testing for:", depvar)

        anova = AnovaRM(
            data=data_summary,
            depvar=depvar,
            subject="subj_N",
            within=["pattern"],
        )
        results = anova.fit()
        print(results)

        # Create a MultiComparison object for post-hoc comparisons
        comp = mc.MultiComparison(data_summary["rt"], data_summary["pattern"])

        # Perform Tukey's HSD test
        post_hoc_res = comp.tukeyhsd(alpha=0.05)
        print(post_hoc_res.summary())

    # * --- Partial Correlation ---
    partial_corr = pg.partial_corr(
        data=data_summary, x="rt", y="amplitude", covar="mean_fix_duration"
    )
    print(partial_corr)

    # * --- Multiple Regression ---
    # Example: Predicting RT from latency, amplitude, and mean_fix_duration
    model = smf.ols(
        # "rt ~ latency + amplitude + mean_fix_duration", data=data_summary
        "accuracy ~ amplitude",
        data=data_summary,
    ).fit()
    print(model.summary())

    # * --- Mixed-Effects Models ---
    # Example: Predicting RT with random intercepts for subject and pattern
    model = smf.mixedlm(
        "rt ~ latency + amplitude + mean_fix_duration",
        data=data_summary,
        groups=data_summary["subj_N"],
        re_formula="~pattern",
    )
    # re_formula is used to specify random slopes, in this case it allows the slopes
    # of the relationship between "rt" and the predictors to vary across patterns.
    # without re_formula, the model assumes a common slope for all subjects, only
    # allowing the intercept to vary.
    results = model.fit()
    print(results.summary())

    # * --------------- Build the RDMs ---------------
    rdm_method = "euclidean"

    (
        rdms_erp_latency,
        rdms_erp_amplitude,
        rdms_erp_combined,
    ) = get_rdms_negative_peak_eeg(
        subj_pattern_frps,
        c.EEG_CHAN_GROUPS["occipital"],
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

            all_subj_rdms[rdm_type] = np.concatenate(
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
            pearson_corr = compare_rdmss(rdm1, rdm2, method="pearson")
            df_pearson_corr_all_subj.loc[comb[0], comb[1]] = pearson_corr
            df_pearson_corr_all_subj.loc[comb[1], comb[0]] = pearson_corr

            spearman_corr = compare_rdmss(rdm1, rdm2, method="spearman")
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

    for name, group_rdm in group_rdms.items():
        rdm_type = name.replace("rdms_", "")
        save_pickle(
            group_rdm,
            c.EXPORT_DIR / f"lab/analyzed/RDMs/[participants]-[{rdm_type}].pkl",
        )

    def get_group_rdms2(rdm_dict):
        """
        Compute the group-averaged RDM from a dictionary of RDMs.

        Args:
            rdm_dict (dict): A dictionary where keys are model names and values are RDMs.

        Returns:
            rsatoolbox.rdm.RDMs: The group-averaged RDM.
        """
        # Extract dissimilarity matrices from all RDMs
        dissimilarities = [rdm.get_matrices()[0] for rdm in rdm_dict.values()]

        # Compute the mean dissimilarity matrix
        mean_dissimilarity = np.mean(dissimilarities, axis=0)

        # Create a new RDM object for the group-averaged RDM
        group_rdm = RDMs(
            dissimilarities=mean_dissimilarity[np.newaxis, :],  # Add a batch dimension
            dissimilarity_measure=rdm_dict[
                next(iter(rdm_dict))
            ].dissimilarity_measure,  # Copy measure
            descriptors=rdm_dict[next(iter(rdm_dict))].descriptors,  # Copy descriptors
            rdm_descriptors={"index": [0]},  # Add a dummy index
            pattern_descriptors=rdm_dict[
                next(iter(rdm_dict))
            ].pattern_descriptors,  # Copy pattern descriptors
        )

        return group_rdm

    for rdm_type, participant_rdms in rdms.items():
        group_rdm = get_group_rdms2(participant_rdms)
        rdm_type = rdm_type.replace("rdms_", "")
        save_pickle(
            group_rdm,
            c.EXPORT_DIR / f"lab/analyzed/RDMs/[participants]-[{rdm_type}].pkl",
        )

    return group_pattern_frps


def inspect_results2():
    # res_dir = WD / "results/analyzed"  # /Oct27-Seq_and_choices"
    res_dir = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis")

    # * ################################################################################
    # * Load all the data
    # * ################################################################################
    frps = {}
    frp_files = sorted(res_dir.glob("sub*/sess*/*frps.pkl"))
    behav_files = sorted(c.DATA_DIR.glob("sub*/sess*/*behav.csv"))
    gaze_info_files = list(res_dir.glob("sub*/sess*/gaze_info.pkl"))
    behav_files = list(c.DATA_DIR.rglob("*behav.csv"))

    subjs = sorted([int(f.stem.split("_")[1]) for f in res_dir.glob("subj*")])

    # * baseline correction period in seconds, used for plotting here
    eeg_baseline = 0.1
    ch_group_names = ["frontal", "parietal", "central", "temporal", "occipital"]
    ch_group_colors = ["red", "green", "blue", "pink", "orange"]

    def get_sess_frps(res_dir, subj_N, sess_N):
        frp_file = res_dir / f"subj_{subj_N:02}/sess_{sess_N:02}/sess_frps.pkl"

        behav_file = list(
            c.DATA_DIR.glob(f"subj_{subj_N:02}/sess_{sess_N:02}/*behav.csv")
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

        pattern_seq_frps = {p: [] for p in c.PATTERNS}
        pattern_choices_frps = {p: [] for p in c.PATTERNS}

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

    group_avg_pattern_seq_frps = {p: [] for p in c.PATTERNS}
    group_avg_pattern_choices_frps = {p: [] for p in c.PATTERNS}

    # * Load FRPs
    for subj_N in tqdm(subjs):
        (
            pattern_seq_frps,
            pattern_choices_frps,
            avg_pattern_seq_frps,
            avg_pattern_choices_frps,
        ) = get_subj_frps(res_dir, subj_N)

        for p in c.PATTERNS:
            group_avg_pattern_seq_frps[p].append(avg_pattern_seq_frps[p])
            group_avg_pattern_choices_frps[p].append(avg_pattern_choices_frps[p])

            selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
                prepare_eeg_data_for_plot(
                    c.EEG_CHAN_GROUPS,
                    c.EEG_MONTAGE,
                    c.NON_EEG_CHANS,
                    avg_pattern_seq_frps[p].info["bads"],
                    ch_group_names,
                    ch_group_colors,
                )
            )

            peak_latency, peak_amplitude = get_neg_erp_peak(
                avg_pattern_seq_frps[p], (0, 0.2), c.EEG_CHAN_GROUPS["occipital"]
            )

            fig = plot_eeg(
                avg_pattern_seq_frps[p].get_data(
                    units="uV", picks=selected_chans_names
                ),
                chans_pos_xy,
                ch_group_inds,
                group_colors,
                c.EEG_SFREQ,
                eeg_baseline,
                vlines=[peak_latency * 1000],
            )
            plt.show()
            plt.close()

    # * Combine the FRPs of all subjects
    for p in c.PATTERNS:
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
                c.EEG_CHAN_GROUPS,
                c.EEG_MONTAGE,
                c.NON_EEG_CHANS,
                frp.info["bads"],
                ch_group_names,
                ch_group_colors,
            )
        )

        peak_latency, peak_amplitude = get_neg_erp_peak(
            frp, (0, 0.2), c.EEG_CHAN_GROUPS["occipital"]
        )

        fig = plot_eeg(
            frp.get_data(units="uV", picks=selected_chans_names),
            chans_pos_xy,
            ch_group_inds,
            group_colors,
            c.EEG_SFREQ,
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


def inspect_results3(data_dir: Path, res_dir: Path, selected_chans: List[str]):
    # # # ! TEMP
    # data_dir = c.DATA_DIR
    # res_dir = c.EXPORT_DIR / "analyzed/subj_lvl"
    # selected_chans = c.EEG_CHAN_GROUPS["occipital"]
    # # ! TEMP

    from functools import reduce

    behav_data = perf_analysis_all_subj(data_dir, c.PATTERNS)["raw_cleaned"]
    gaze_data = analyze_processed_gaze_data_all_subj(data_dir, res_dir)["gaze_data"]

    eeg_data = analyze_processed_frp_data_all_subj(
        data_dir, res_dir, selected_chans, neg_peak_time_window=(0, 0.2), pbar=False
    )

    eeg_data = eeg_data["neg_peak_data_seq"]

    included_subjs = behav_data.groupby(["subj_N"])["sess_N"].last()
    included_subjs = included_subjs[included_subjs == 5].index

    # behav_data = behav_data.query(f"subj_N in @included_subjs").reset_index(drop=True)
    # gaze_data = gaze_data.query(f"subj_N in @included_subjs").reset_index(drop=True)
    # eeg_data = eeg_data.query(f"subj_N in @included_subjs").reset_index(drop=True)

    # eeg_data.groupby("pattern").mean(numeric_only=True)
    # eeg_data.groupby("pattern").count()

    # # * --------------- behavioral data ---------------
    pattern_acc = behav_data.groupby("pattern")["correct"].mean()
    pattern_difficulty = pattern_acc.rank(ascending=False).sort_values()

    pattern_acc.name = "accuracy"
    pattern_acc = pattern_acc.reset_index()

    pattern_difficulty.name = "difficulty"
    pattern_difficulty = pattern_difficulty.reset_index()

    pattern_rt = behav_data.groupby("pattern")["rt"].mean()
    pattern_rt.name = "rt"
    pattern_rt = pattern_rt.reset_index()

    # * --------------- gaze data ---------------
    fix_duration_on_seq_icons = (
        gaze_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern", "item_id"], as_index=False)["total_duration"]
        .sum()
    )

    nb_fix_on_seq_icons = (
        gaze_data.query("stim_type=='sequence'").groupby(
            ["subj_N", "pattern", "item_id"], as_index=False
        )["count"]
    ).sum()

    pupil_diam_on_seq_icons = (
        gaze_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern", "item_id"], as_index=False)["mean_pupil_diam"]
        .mean()
    )

    pupil_diam_per_patt = pupil_diam_on_seq_icons.groupby(["pattern"], as_index=False)[
        "mean_pupil_diam"
    ].mean()

    fix_duration_per_patt = fix_duration_on_seq_icons.groupby(
        ["pattern"], as_index=False
    )["total_duration"].mean()

    fix_nb_per_patt = nb_fix_on_seq_icons.groupby(["pattern"], as_index=False)[
        "count"
    ].mean()

    # * --------------- EEG data ---------------
    eeg_data_on_seq_icons = (
        eeg_data.query("stim_type=='sequence'")
        .groupby(["subj_N", "pattern", "item_id"], as_index=False)[
            ["peak_latency", "peak_amplitude"]
        ]
        .mean()
    )

    eeg_data_per_patt = (
        eeg_data.query("stim_type=='sequence'")
        .groupby("pattern", as_index=False)[["peak_latency", "peak_amplitude"]]
        .mean()
    )

    # * --------------- merge data ---------------
    summary_df_items = reduce(
        lambda left, right: pd.merge(
            left, right, on=["subj_N", "pattern", "item_id"], how="inner"
        ),
        [
            nb_fix_on_seq_icons,
            fix_duration_on_seq_icons,
            pupil_diam_on_seq_icons,
            eeg_data_on_seq_icons,
            behav_data[["subj_N", "pattern", "item_id", "correct", "rt"]],
        ],
    )
    summary_df_items = summary_df_items.merge(
        pattern_difficulty, on="pattern", how="inner"
    )

    summary_df_items.rename(
        columns={
            "count": "nb_fix",
            "total_duration": "total_fix_duration",
            "peak_latency": "frp_lat",
            "peak_amplitude": "frp_amp",
        },
        inplace=True,
    )

    summary_df_patterns = reduce(
        lambda left, right: pd.merge(left, right, on="pattern", how="inner"),
        [
            pattern_acc,
            pattern_difficulty,
            pupil_diam_per_patt,
            fix_duration_per_patt,
            fix_nb_per_patt,
            pattern_rt,
            eeg_data_per_patt,
        ],
    )

    summary_df_patterns.rename(
        columns={
            "total_duration": "mean_fix_duration",
            "count": "mean_fix_count",
            "peak_latency": "frp_lat",
            "peak_amplitude": "frp_amp",
        },
        inplace=True,
    )

    print("number of values in behav data:")
    display(behav_data.value_counts("subj_N").reset_index())

    print("number of values in gaze data:")
    display(summary_df_items.value_counts("subj_N").reset_index())

    display(summary_df_items)
    display(summary_df_patterns)

    # * Adding pattern features
    summary_df_items["patt_unique_el_count"] = 0

    patt_info = {}
    for patt in list(summary_df_items.pattern.unique()):
        unique_els = list(set(patt))
        count_unique = len(unique_els)
        count_by_el = {el: patt.count(el) for el in unique_els}
        patt_info[patt] = {"count_unique": count_unique, "count_by_el": count_by_el}
        patt_inds = summary_df_items.query("pattern == @patt").index
        summary_df_items.loc[patt_inds, "patt_unique_el_count"] = count_unique

    # * --------------- Correlations ---------------
    # for subj_N in summary_df_items["subj_N"].unique():
    #     subj_data = summary_df_items.query(f"subj_N=={subj_N}")
    #     display(apply_df_style(subj_data.iloc[:, 2:].corr(), style=3, vmin=-1, vmax=1))

    difficulty_data = Dataset(
        summary_df_items.groupby("pattern")["difficulty"]
        .mean()
        # .sort_values()
        .to_numpy()[:, None]
    )

    fig, ax = plt.subplots()
    im = ax.imshow(calc_rdm(difficulty_data, "euclidean").get_matrices()[0])
    fig.colorbar(im, ax=ax)

    summary_df_items_corr = summary_df_items.iloc[:, 2:].corr()

    summary_df_patterns_corr = summary_df_patterns.iloc[:, 2:].corr()

    summary_df_items_corr_styled = apply_df_style(
        summary_df_items_corr, style=3, vmin=-1, vmax=1
    ).set_caption("Correlation Matrix - Data Groupped by Items / Individual Sequences")

    summary_df_patterns_corr_styled = apply_df_style(
        summary_df_patterns_corr, style=3, vmin=-1, vmax=1
    ).set_caption("Correlation Matrix - Data Groupped by Patterns")

    dfi.export(
        summary_df_items_corr_styled,
        c.EXPORT_DIR / "analyzed/group_lvl/summary_df_items_corr.png",
        table_conversion="matplotlib",
    )

    dfi.export(
        summary_df_patterns_corr_styled,
        c.EXPORT_DIR / "analyzed/group_lvl/summary_df_patterns_corr.png",
        table_conversion="matplotlib",
    )

    # * ----------------------------------------
    # *  Encode pattern_type as a categorical variable
    summary_df_items["pattern"] = summary_df_items["pattern"].astype("category")
    # * ----------------------------------------
    # * ----------------------------------------
    # * mixed-effects multinomial logistic regression model. This model predicts the
    # * categorical outcome (pattern type) from FRP amplitude while controlling for the
    # * number of fixations and including a random intercept for each subject.
    # * ----------------------------------------
    import bambi as bmb
    import arviz as az
    import pandas as pd

    # Define the model formula:
    # "pattern" is the categorical outcome.
    # "frp_amplitude" and "num_fixations" are fixed effects.
    # "(1|subject)" adds a random intercept for each subject.
    model = bmb.Model(
        "pattern ~ frp_amp + nb_fix + (1|subj_N)",
        data=summary_df_items,
        family="categorical",
    )

    # Fit the model. The 'draws' parameter controls the number of posterior samples.
    results = model.fit(draws=500)

    # Summarize the posterior estimates.
    print(az.summary(results))
    df_res = az.summary(results)
    df_res.head(60)

    # * ----------------------------------------

    data = (
        summary_df_items.groupby(["subj_N", "pattern"])["frp_amp"].mean().reset_index()
    )

    # * ANOVA with statsmodels
    from statsmodels.stats.anova import AnovaRM

    aov = AnovaRM(data=data, depvar="frp_amp", subject="subj_N", within=["pattern"])
    anova_results = aov.fit()
    print(anova_results)

    # * ANOVA with pingouin
    import pingouin as pg

    # Example: using the same DataFrame 'df'
    rm_anova = pg.rm_anova(
        dv="frp_amp", within="pattern", subject="subj_N", data=data, detailed=True
    )
    display(rm_anova)

    # * Linear mixed effects model
    # Fit a Linear mixed-effects model with a random intercept for each participant.
    model = smf.mixedlm(
        "frp_amp ~ C(pattern)",
        data=summary_df_items,
        groups=summary_df_items["subj_N"],
    )

    mixedlm_results = model.fit()
    print(mixedlm_results.summary())

    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.barplot(x="pattern", y="frp_amp", data=summary_df_items, errorbar="ci", ax=ax)
    # ax.set_title("Mean FRP Amplitude by Pattern")
    # plt.tight_layout()
    # plt.show()

    # * ----------------------------------------
    # * ----------------------------------------

    # * Reset index and ensure subj_N is a categorical variable
    summary_df_items = summary_df_items.dropna(subset=["rt", "frp_amp", "subj_N"])
    summary_df_items = summary_df_items.reset_index(drop=True)
    summary_df_items["subj_N"] = summary_df_items["subj_N"].astype("category")

    # * Use the categorical codes for groups
    groups = summary_df_items["subj_N"].cat.codes

    # * COMMENT THIS LINE
    mixed_model = smf.mixedlm(
        "rt ~ frp_amp", data=summary_df_items, groups=groups
    ).fit()

    print(mixed_model.summary())

    # * COMMENT THIS LINE
    mixed_model_rs = smf.mixedlm(
        "rt ~ frp_amp", data=summary_df_items, groups=groups, re_formula="~frp_amp"
    ).fit()
    print(mixed_model_rs.summary())

    from scipy.stats import chi2

    # -------------------------------
    # 1. Fit two mixed-effects models using ML (reml=False)
    # -------------------------------

    # Model 1: Random intercept only
    model_intercept_ml = smf.mixedlm(
        "rt ~ frp_amp", data=summary_df_items, groups=summary_df_items["subj_N"]
    ).fit(reml=False)

    # Model 2: Random intercept and random slope for frp_amp
    model_slope_ml = smf.mixedlm(
        "rt ~ frp_amp",
        data=summary_df_items,
        groups=summary_df_items["subj_N"],
        re_formula="~frp_amp",
    ).fit(reml=False)

    # -------------------------------
    # 2. Print AIC and BIC for both models
    # -------------------------------
    print("Random Intercept Model AIC (ML):", model_intercept_ml.aic)
    print("Random Slope Model AIC (ML):", model_slope_ml.aic)
    print("Random Intercept Model BIC (ML):", model_intercept_ml.bic)
    print("Random Slope Model BIC (ML):", model_slope_ml.bic)

    # -------------------------------
    # 3. Likelihood Ratio Test between models
    # -------------------------------
    # Compute the LR statistic using ML log-likelihoods:
    lr_stat = 2 * (model_slope_ml.llf - model_intercept_ml.llf)
    # Calculate the difference in the number of parameters
    df_diff = model_slope_ml.df_modelwc - model_intercept_ml.df_modelwc
    # Compute the p-value from the chi-square distribution
    p_value = chi2.sf(lr_stat, df_diff)

    print("\nLikelihood Ratio Test")
    print("---------------------")
    print("LR statistic:", lr_stat)
    print("Degrees of freedom difference:", df_diff)
    print("p-value:", p_value)

    # -------------------------------
    # 4. Visualize the random effects from the random slopes model
    # -------------------------------
    # Extract the random effects dictionary (keyed by group/participant)
    random_effects = model_slope_ml.random_effects

    # Convert the dictionary into a DataFrame where each row corresponds to a participant
    re_df = pd.DataFrame(random_effects).T
    print("\nRandom Effects DataFrame (first 5 rows):")
    print(re_df.head())

    # Rename the random intercept column for clarity (it may be named "Group" by default)
    re_df = re_df.rename(columns={"Group": "Intercept"})

    # Create a scatter plot: Random intercept vs. random slope for frp_amp
    plt.figure(figsize=(8, 6))
    plt.scatter(re_df["Intercept"], re_df["frp_amp"], s=50, alpha=0.7)
    plt.xlabel("Random Intercept")
    plt.ylabel("Random Slope (frp_amp)")
    plt.title("Random Effects: Intercept vs. Slope by Participant")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.show()

    # * ----------------------------------------
    # Convert the categorical 'pattern' into numeric codes (0, 1, 2, ...)
    summary_df_items["pattern"] = summary_df_items["pattern"].astype("category")
    summary_df_items["pattern_numeric"] = summary_df_items["pattern"].cat.codes

    # Fit the multinomial logistic regression model using the numeric variable
    mnlogit_model = smf.mnlogit(
        "pattern_numeric ~ frp_amp", data=summary_df_items
    ).fit()
    print(mnlogit_model.summary())

    # ----------------------------------------
    # Get the odds ratios for the frp_amp predictor
    # Example for pattern_numeric=1
    odds_ratio = np.exp(0.1224)
    print("Odds Ratio for frp_amp in category 1:", odds_ratio)

    # ----------------------------------------
    # Get predicted probabilities for each pattern category
    predicted_probabilities = pd.DataFrame(mnlogit_model.predict())

    # Get the predicted class (category with the highest probability)
    summary_df_items["predicted_pattern"] = predicted_probabilities.idxmax(axis=1)

    # Compare predicted vs. actual pattern
    print(summary_df_items[["pattern_numeric", "predicted_pattern"]].head(10))

    # Now, compare the predicted pattern with the actual pattern (the numeric codes)
    accuracy = np.mean(
        summary_df_items["predicted_pattern"] == summary_df_items["pattern_numeric"]
    )
    print("Prediction accuracy:", accuracy)

    # ----------------------------------------
    # Include participant as a fixed effect
    mnlogit_fe_model = smf.mnlogit(
        "pattern_numeric ~ frp_amp + C(subj_N)", data=summary_df_items
    ).fit()
    print(mnlogit_fe_model.summary())

    # * ----------------------------------------
    # * Bayesian Hierarchical Model
    # Ensure that the response variable is categorical with defined levels
    summary_df_items["pattern_numeric"] = summary_df_items["pattern_numeric"].astype(
        "category"
    )

    import bambi as bmb
    import arviz as az

    # Fit a hierarchical (mixed-effects) multinomial logistic regression
    # (1|subj_N) adds a random intercept for each participant.
    model = bmb.Model(
        "pattern_numeric ~ frp_amp + (1|subj_N)",
        data=summary_df_items,
        family="categorical",
    )

    results = model.fit(draws=2000)  # adjust draws for convergence
    az.summary(results)

    # * ----------------------------------------
    # * ----------------------------------------
    eeg_fixation_pac_files = sorted(res_dir.glob("sub*/sess*/eeg_fixation_pac*.pkl"))
    eeg_fixation_pac_data = [read_file(f) for f in eeg_fixation_pac_files]
    len(eeg_fixation_pac_data[0])
    eeg_fixation_pac_data[0][0].keys()


def inspect_behav():
    # *

    res = perf_analysis_all_subj(data_dir=c.DATA_DIR, patterns=c.PATTERNS)

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


def save_sequences_as_images():
    seq_images_dir = WD / "sequence_images"
    seq_images_dir.mkdir(exist_ok=True)

    pos_seq = c.ANALYSIS_CONFIG.stim.x_pos_stim.items_set
    pos_choices = c.ANALYSIS_CONFIG.stim.x_pos_stim.avail_choice
    center_y_pos_choices = c.SCREEN_RESOLUTION[1] - (
        c.SCREEN_RESOLUTION[1] / 2 + c.Y_POS_CHOICES
    )
    center_y_pos_seq = c.SCREEN_RESOLUTION[1] - (
        c.SCREEN_RESOLUTION[1] / 2 + c.Y_POS_SEQUENCE
    )

    stim_pos = []
    for i, center_pos in enumerate(pos_seq + pos_choices):
        left = center_pos - (c.IMG_SIZE[0] / 2)
        right = center_pos + (c.IMG_SIZE[0] / 2)
        if i < 8:
            bottom = center_y_pos_seq - c.IMG_SIZE[1] / 2
            top = center_y_pos_seq + c.IMG_SIZE[1] / 2
        else:
            bottom = center_y_pos_choices - c.IMG_SIZE[1] / 2
            top = center_y_pos_choices + c.IMG_SIZE[1] / 2

        left, right = [i + c.SCREEN_RESOLUTION[0] / 2 for i in (left, right)]

        stim_pos.append((left, right, bottom, top))

    df_sequences = pd.concat(
        [
            pd.read_csv(f)
            for f in (WD.parent / "experiment-ANNs/sequences").iterdir()
            if re.search(r"session_\d.csv", f.name)
        ]
    )
    df_sequences.reset_index(drop=True, inplace=True)
    choice_cols = [c for c in df_sequences.columns if re.search(r"choice\d", c)]
    seq_cols = [c for c in df_sequences.columns if re.search(r"figure\d", c)]

    sequences_pos = []
    for i, row in df_sequences.iterrows():
        stim = row.loc[seq_cols + choice_cols].values
        stim[row["masked_idx"]] = "question-mark"
        sequences_pos.append([row["item_id"]] + list(zip(stim, stim_pos)))

    for seq in tqdm(sequences_pos):
        item_id, seq_pos = seq[0], seq[1:]

        plot_sequence_img(
            seq_pos, c.ICON_IMAGES, c.SCREEN_RESOLUTION, seq_images_dir, seq_id=item_id
        )
        plt.close()

    for i, row in df_sequences.sample(5).iterrows():
        item_id = row["item_id"]
        seq_image = seq_images_dir / f"sequence_{item_id}.png"

        seq_str = row[seq_cols].values
        seq_str[row["masked_idx"]] = "?"
        seq_str = " ".join(seq_str)
        choice_str = " ".join(row[choice_cols].values)

        fig, ax = plt.subplots()
        ax.imshow(plt.imread(seq_image))
        ax.set_title(f"Item ID: {item_id}\n{row['pattern']}\n{seq_str}\n{choice_str}")
        ax.set_axis_off()
        plt.show()


# clear_jupyter_artifacts()
# memory_usage = get_memory_usage()
# memory_usage.head(50)


def test_lazy_predict(data_dir, processed_data_dir, selected_chan_group):
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split
    # # ! TEMP
    # data_dir = DATA_DIR
    # processed_data_dir = EXPORT_DIR / "analyzed/subj_lvl"
    # save_dir = EXPORT_DIR / "analyzed/group_lvl/RSA"
    # dissimilarity_metric = "correlation"
    # similarity_metric = "cosine"
    # selected_chan_group = "frontal"
    # chan_group = selected_chan_group
    # save_dir = (
    #     EXPORT_DIR / f"analyzed/RSA-FRP-{chan_group}-metric_{dissimilarity_metric}"
    # )
    # # ! TEMP

    selected_chans = c.EEG_CHAN_GROUPS[chan_group]

    # * ################################################################################
    # * Load all the data
    # * ################################################################################

    frp_files = sorted(processed_data_dir.glob("sub*/sess*/sess_frps.pkl"))

    behav_data_list = []
    for subj_dir in processed_data_dir.glob("subj_*"):
        subj_N = int(subj_dir.name.split("_")[-1])
        for sess_dir in subj_dir.glob("sess_*"):
            sess_N = int(sess_dir.name.split("_")[-1])
            behav_data_list.append(load_and_clean_behav_data(data_dir, subj_N, sess_N))

    behav_data = pd.concat(behav_data_list)
    behav_data.reset_index(drop=True, inplace=True)
    del behav_data_list

    # * Replace the above with line below, but won't work if not all participants have been analyzed
    # behav_data = pd.concat(list(load_and_clean_behav_data_all(data_dir)))

    # subj_data: dict = {int(subj): {} for subj in behav_data["subj_N"].unique()}

    # missing_frps = []
    frps = []
    patterns = []
    item_ids = []

    for frp_file in tqdm(sorted(frp_files)):
        subj_N = int(frp_file.parents[1].name.split("_")[-1])
        sess_N = int(frp_file.parents[0].name.split("_")[-1])

        # subj_data[subj_N][sess_N] = []

        sess_df = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")
        sess_item_ids = sess_df["item_id"].to_list()
        sess_patterns = sess_df["pattern"].to_list()

        frp_data = read_file(frp_file)

        sequence_frps = frp_data["sequence"]

        if sess_missing_frps := [i for i, s in enumerate(sequence_frps) if s is None]:
            missing_frps.append((subj_N, sess_N, sess_missing_frps))

        data_shape = (
            [s for s in sequence_frps if s is not None][0]
            .get_data(picks=selected_chans)
            .shape
        )

        # empty_data = np.zeros(data_shape)
        # empty_data[:] = np.nan

        sequence_frps = [
            frp.get_data(picks=selected_chans) if frp is not None else None
            for frp in sequence_frps
        ]

        valid_data_inds = [i for i, s in enumerate(sequence_frps) if s is not None]

        sequence_frps = np.array([frp for frp in sequence_frps if frp is not None])
        sess_item_ids = np.array(sess_item_ids)[valid_data_inds]
        sess_patterns = np.array(sess_patterns)[valid_data_inds]

        assert (
            sess_item_ids.shape[0] == sequence_frps.shape[0] == sess_patterns.shape[0]
        )

        frps.append(sequence_frps)
        patterns.append(sess_patterns)
        item_ids.append(sess_item_ids)

        # subj_data[subj_N][sess_N] = [
        #     sess_item_ids,
        #     sess_patterns,
        #     sequence_frps,
        #     # timepoints,
        # ]

    frps = np.concatenate(frps)
    patterns = np.concatenate(patterns)
    item_ids = np.concatenate(item_ids)

    assert frps.shape[0] == patterns.shape[0] == item_ids.shape[0]
    frps_avg_over_chans = np.mean(frps, axis=1)

    X = frps_avg_over_chans
    y = patterns
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=123
    )

    clf = LazyClassifier(verbose=0)  # ,ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    predictions


def get_frp_pca(frp_data):
    from sklearn.decomposition import PCA

    def get_pca_rdms(frps):
        pca = PCA(n_components=2)

        pca_data = []
        for frp in tqdm(frps):
            splits = np.array_split(frp, 3, axis=1)
            frp_pca_data = []

            for split in splits:
                princ_comps = pca.fit_transform(split)

                frp_pca_data.append(princ_comps[:, 0])
                # principalDf = pd.DataFrame(
                #     data=principalComponents, columns=["PC1", "PC2"]
                # )
                # plt.plot(principalDf["PC1"], "o")
                # plt.plot(principalDf["PC2"], "o")

            frp_pca_data = np.array(frp_pca_data)
            # frp_pca_data = np.concatenate(frp_pca_data)
            pca_data.append(frp_pca_data)

        pca_data = np.array(pca_data)

        pca_data = pca_data.reshape(pca_data.shape[0], -1)

        pca_dataset = Dataset(
            pca_data,
            # obs_descriptors={"item_id": subj_item_ids, "pattern": subj_patterns},
        )

        for method in [
            "euclidean",
            "correlation",
            # "mahalanobis",
            # "crossnobis",
            # "poisson",
            # "poisson_cv",
        ]:
            rdm = calc_rdm(pca_dataset, method)

            fig, ax = plt.subplots()
            rdm_array = rdm.get_matrices()[0]
            im = ax.imshow(rdm_array)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"RDM - PCA - {method}")
            plt.show()
            plt.close()

    subj_data = {k: v for k, v in subj_data.items() if len(v) > 4}
    # [v.keys() for v in subj_data.values()]

    subjs_frps = []

    for subj_N in subj_data.keys():
        subj_item_ids = np.concatenate([i[0] for i in subj_data[subj_N].values()])
        subj_patterns = np.concatenate([i[1] for i in subj_data[subj_N].values()])
        subj_frps = np.concatenate([i[2] for i in subj_data[subj_N].values()])

        inds_missing_data = [
            i for i, subj_frp in enumerate(subj_frps) if np.all(np.isnan(subj_frp))
        ]
        # print("missing_inds:", inds_missing_data)
        # np.all(np.isnan(subj_frps[inds_missing_data]))

        # * Replace missing data
        overall_avg_frp = np.nanmean(subj_frps, axis=0)

        for ind in inds_missing_data:
            subj_frps[ind] = overall_avg_frp

        # * -------- Reorder the FRPs --------
        reordered_inds = reorder_item_ids(
            original_order_df=pd.DataFrame(
                {"item_id": subj_item_ids, "pattern": subj_patterns}
            ),
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )
        subj_frps = subj_frps[reordered_inds]
        subj_patterns = np.array(subj_patterns)[reordered_inds]
        subj_item_ids = np.array(subj_item_ids)[reordered_inds]

        subjs_frps.append(subj_frps)
        get_pca_rdms(subj_frps)

    group_avg_frps = np.array(subjs_frps).mean(axis=0)
    group_avg_frps.shape
    get_pca_rdms(group_avg_frps)


def locate_unique_icons(sequences_file: Path, subj_behav: pd.DataFrame) -> Tuple:
    # # ! TEMP
    # sequences_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
    # behav_res = perf_analysis_all_subj(data_dir=c.DATA_DIR, patterns=c.PATTERNS)
    # subj_behav = behav_res["raw_cleaned"].copy()
    # subj_behav = subj_behav.query("subj_N == 1")
    # # ! TEMP

    # * Load file with sequences
    sequences = pd.read_csv(sequences_file)

    icon_cols = [f"figure{i}" for i in range(1, 9)] + [
        f"choice{i}" for i in range(1, 5)
    ]

    cols_filter = ["item_id", "pattern"] + icon_cols

    subj_behav = subj_behav.merge(
        sequences[cols_filter], on=["item_id", "pattern"], how="outer"
    )
    unique_icons = tuple(sorted(set(subj_behav[icon_cols].values.flatten())))

    icons_masks = {}
    icons_locs = {}
    for icon in unique_icons:
        # * Getting mask of icon position
        mask = np.zeros_like(subj_behav[icon_cols])
        mask[np.where(subj_behav[icon_cols] == icon)] = 1
        icons_masks[icon] = mask

        # * Locate trial indices and column indices where icon is present
        trial_inds = tuple(set(np.where(mask == 1)[0]))
        positions_inds = [tuple(np.where(row == 1)[0]) for row in mask[trial_inds, :]]
        icons_locs[icon] = tuple(zip(trial_inds, positions_inds))

    # * filter trials where selected icon is present
    # sel_icon = "hammer"
    # subj_behav[icon_cols].iloc[[i[0] for i in icons_locs[sel_icon]]]
    # subj_behav.iloc[[i[0] for i in icons_locs[sel_icon]]]

    # * Show masked dataframe
    # sel_icon = "hammer"
    # arr = subj_behav[icon_cols].values
    # arr[~icons_masks[sel_icon].astype(bool)] = "0"
    # masked_df = pd.DataFrame(arr, columns=icon_cols)
    # masked_df['item_id'] = subj_behav['item_id']
    # masked_df['pattern'] = subj_behav['pattern']
    # masked_df

    # * Check total number of occurences per unique icon
    # pd.Series(
    #     {k: sum([len(v[1]) for v in values]) for k, values in icons_locs.items()}
    # ).sort_values(ascending=False)

    return icons_masks, icons_locs


if __name__ == "__main__":
    # * ####################################################################################
    # * Preprocessing data, extracting features, and saving them
    # * ####################################################################################
    # main()

    # * ####################################################################################
    # * Behavioral analysis
    # * ####################################################################################
    save_dir = c.EXPORT_DIR / "analyzed/group_lvl"

    behav_res = perf_analysis_all_subj(data_dir=c.DATA_DIR, patterns=c.PATTERNS)

    for name, df in behav_res.items():
        print(name)
        display(df)

    accuracy_by_item = (
        behav_res["raw_cleaned"]
        .groupby("item_id")["correct"]
        .mean()
        .sort_values(ascending=False)
    )
    rank_acc_by_item = accuracy_by_item.rank()
    rank_acc_by_item.name = "rank_acc"

    sequences = pd.read_csv(
        WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
    )
    top_5_hardest = rank_acc_by_item.nsmallest(5)

    # sequence_images_dir = WD.parent / "config/sequence_images"`
    # for item_id in top_5_hardest.index:
    #     fig, ax = plt.subplots()
    #     image = plt.imread(sequence_images_dir / f"sequence_{item_id}.png")
    #     ax.imshow(image)
    #     plt.show()

    # sequences.query("item_id in @top_5_hardest.index")
    # behav_res["raw_cleaned"].merge(rank_acc_by_item, on="item_id").sort_values("rank_acc")['correct']

    # * ####################################################################################
    # * Representational Similarity Analysis
    # * ####################################################################################

    # * ------- FRP RDMs -------
    # save_dir = EXPORT_DIR / "analyzed/group_lvl/RSA"

    for chan_group in tqdm(["frontal", "occipital"]):
        # * ------- RDMs for FRP time series -------
        dissimilarity_metric = "correlation"

        save_dir = (
            c.EXPORT_DIR
            / f"analyzed/RSA-FRP-{chan_group}-metric_{dissimilarity_metric}"
        )
        # # ! TEMP
        # save_dir = save_dir.parent / f"{str(save_dir.name)}-300ms"
        # # ! TEMP

        get_frp_rdms_all_subj(
            data_dir=c.DATA_DIR,
            processed_data_dir=c.EXPORT_DIR / "analyzed/subj_lvl",
            save_dir=save_dir,
            dissimilarity_metric=dissimilarity_metric,
            selected_chan_group=chan_group,
            time_window=(0.0, 0.6),
        )

        # * ------- RDMs for FRP peak amplitude -------
        dissimilarity_metric = "euclidean"

        save_dir = (
            c.EXPORT_DIR
            / f"analyzed/RSA-FRP_AMP-{chan_group}-metric_{dissimilarity_metric}"
        )

        get_frp_amp_rdms_all_subj(
            data_dir=c.DATA_DIR,
            processed_data_dir=c.EXPORT_DIR / "analyzed/subj_lvl",
            save_dir=save_dir,
            dissimilarity_metric=dissimilarity_metric,
            selected_chan_group=chan_group,
        )

    # * ------- RSA -------
    similarity_metric = "cosine"
    dissimilarity_metric = "correlation"

    for chan_group in ["occipital", "frontal"]:
        rdms_dir = (
            c.EXPORT_DIR
            / f"analyzed/RSA-FRP-{chan_group}-metric_{dissimilarity_metric}"
        )

        subj_rdm_files = sorted(
            [f for f in rdms_dir.glob("*subj*.hdf5") if not f.name.startswith(".")]
        )
        subj_rdms = [rsatoolbox.rdm.rdms.load_rdm(f, "hdf5") for f in subj_rdm_files]
        full_rdm_inds = [i for i, rdm in enumerate(subj_rdms) if rdm.n_cond == 400]
        subj_rdm_files = [subj_rdm_files[i] for i in full_rdm_inds]

        rsa_between_subjects(
            rdm_files=subj_rdm_files,
            save_dir=rdms_dir,
            similarity_metric=similarity_metric,
        )
        plt.imshow(subj_rdms[1].get_matrices()[0])

        pd.DataFrame([rdm.pattern_descriptors["patterns"] for rdm in subj_rdms])

    # * ------- Accuracy RDMs -------
    save_dir = c.EXPORT_DIR / "analyzed/RSA-Accuracy"

    get_accuracy_rdms(
        data_dir=c.DATA_DIR,
        save_dir=save_dir,
        dissimilarity_metric=dissimilarity_metric,
    )
