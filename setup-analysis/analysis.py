import mne
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from database import Database
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm.auto import tqdm
from pprint import pprint
from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components
from mne.preprocessing import ICA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from IPython.display import display
from scipy.stats import chi2_contingency
from utils import (
    pickle_save,
    pickle_load,
    invert_dict,
    to_named_tuple,
    create_report_doc,
    custom_plot,
)
import os
# from typing import List, Tuple, Union, Callable


# * #### CONFIGURATION #### * #
wd = Path(__file__).parent
os.chdir(wd)

with open(wd.parent/"config/experiment_config.json") as f:
    exp_config = json.load(f)

directories = dict(results=wd / "local/results/pilot results")
directories = to_named_tuple(directories, "Directories")

default_fig_params = dict(size=(12, 8), dpi=300, tight_layout=True)
default_fig_params = to_named_tuple(default_fig_params, "FigParams")

file_exts = {
    "mne_epochs": "-epo.fif",
    "mne_raw": "-raw.fif",
    "mne_ica": "-ica.fif",
    "plot": ".png",
}

file_exts = to_named_tuple(file_exts, "FileExts")

db = Database("./global_config/database.db")

eeg_files = directories.results.glob("*.bdf")


# * #### ANALYSIS FUNCTIONS #### * #
def locate_stim_in_trials(sequences: pd.DataFrame):
    # sequences = pd.read_csv("../global_config/sequences-format[names].csv") # ! TEMP
    # sequences = pd.read_csv("../global_config/old_combs/selected_combinations-format[names].csv")
    # sequences = pd.read_excel("../global_config/new_combs/selected_combinations-format[names].xlsx")

    seq_cols = [f"figure{i+1}" for i in range(8)]
    choice_cols = [f"choice{i+1}" for i in range(4)]

    unique_figs = [set(sequences[col]) for col in seq_cols + choice_cols]
    unique_figs = set.union(*unique_figs)

    fig_locs = {fig: [] for fig in unique_figs}

    for fig in unique_figs:
        for i, row in sequences.iterrows():
            if fig in row.loc[seq_cols + choice_cols].values:
                seq_locs = np.where(row.loc[seq_cols].values == fig)[0].tolist()
                choice_locs = np.where(row.loc[choice_cols].values == fig)[0].tolist()

                fig_locs[fig].append(
                    {
                        "trial_id": i,
                        "combinationID": row["combinationID"],
                        "pattern": row["pattern"],
                        "seq_locs": seq_locs,
                        "choice_locs": choice_locs,
                        # "locs": np.where(row.values == fig)[0].tolist(), # * Indiscriminate of seq or options
                    }
                )

    pprint({k: f"{len(v)} trials" for k, v in fig_locs.items()})
    # pprint(fig_locs["eye"], sort_dicts=False)
    # pprint(fig_locs, sort_dicts=False)
    return fig_locs


def preprocess_first_pilot_eeg(fpath, stim_channel="Status"):

    old_event_IDs = {
        "a": 1,
        "x": 2,
        "m": 3,
        "l": 4,
        "invalid": 50,
        "trial_start": 60,
        "stim_flash": 61,
        "all_stim": 62,
        "exp_start": 70,
    }

    new_event_IDs = {
        "a": 1,
        "x": 2,
        "m": 3,
        "l": 4,
        "invalid": 5,
        "exp_start": 21,
        "block_start": 22,
        "block_end": 23,
        "trial_start": 24,
        "trial_end": 25,
        "stim_flash": 31,
        "all_stim": 32,
    }

    new_event_IDs2 = {
        "a": 1,
        "x": 2,
        "m": 3,
        "l": 4,
        "invalid": 5,
        "exp_start": 21,
        "block_start": 22,
        "block_end": 23,
        "trial_start": 24,
        "trial_end": 25,
        "stim-flash_sequence": 31,
        "stim-flash_choices": 32,
        "stim-all_stim": 33,
    }

    raw = mne.io.read_raw_bdf(
        fpath,
        eog=None,
        misc=None,
        stim_channel=stim_channel,  # * also works with 'auto'
        exclude=(),
        infer_types=False,
        include=None,
        preload=False,
        units=None,
        encoding="utf8",
        verbose=None,
    )

    info = raw.info
    fpath = Path(raw.filenames[0])
    stim_channel = "Status"

    stim_chan_idx = raw.ch_names.index(stim_channel)
    raw_array = raw.get_data()
    stim_chan_data = raw_array[stim_chan_idx, :]

    # test = []
    for name, id in old_event_IDs.items():
        new_id = new_event_IDs[name]
        # test.append((name, id, new_id))
        stim_chan_data[stim_chan_data == id] = new_id

    raw_array[stim_chan_idx, :] = stim_chan_data

    new_raw = mne.io.RawArray(raw_array, info)

    events = mne.find_events(
        new_raw,
        stim_channel=stim_channel,
        output="onset",
        consecutive="increasing",
        min_duration=0,
        shortest_event=1,
        mask=None,
        uint_cast=False,
        mask_type="and",
        initial_event=False,
        verbose=None,
    )

    # annotations = mne.annotations_from_events(
    #     events=events,
    #     event_desc={v:k for k,v in new_event_IDs.items()},
    #     sfreq=info["sfreq"],
    #     orig_time=info["meas_date"],
    # )

    # new_raw.set_annotations(annotations)

    # new_raw.plot()
    # raw.plot(events=events, event_id=new_event_IDs, scalings={"eeg": 100e-6})

    t_start = 550

    trimmed_raw = new_raw.copy().crop(tmin=t_start, tmax=None)
    # trimmed_raw.plot(events=events, event_id=new_event_IDs, scalings={"eeg": 100e-6})
    trimmed_raw = trimmed_raw.get_data()
    # np.where(trimmed_raw[stim_chan_idx,:]==5)
    trimmed_raw[stim_chan_idx, 33809:33830] = new_event_IDs["exp_start"]
    # np.unique(trimmed_raw[stim_chan_idx, :], return_counts=True)

    # ! < Event IDs Version 2
    all_stim_inds = np.where(trimmed_raw[stim_chan_idx, :] == new_event_IDs["all_stim"])
    trimmed_raw[stim_chan_idx, all_stim_inds] = new_event_IDs2["stim-all_stim"]

    stim_flash_inds = np.where(
        trimmed_raw[stim_chan_idx, :] == new_event_IDs["stim_flash"]
    )[0]

    stim_flash_inds = stim_flash_inds[
        (trimmed_raw[stim_chan_idx, stim_flash_inds - 1] == 0)
        & (
            trimmed_raw[stim_chan_idx, stim_flash_inds + 1]
            == new_event_IDs["stim_flash"]
        )
    ]

    n_seq_items = 7
    n_choice_items = 4

    stim_flash_inds = np.split(
        stim_flash_inds, stim_flash_inds.shape[0] / (n_seq_items + n_choice_items)
    )

    for inds in stim_flash_inds:
        trimmed_raw[stim_chan_idx, inds[:n_seq_items]] = new_event_IDs2[
            "stim-flash_sequence"
        ]
        trimmed_raw[stim_chan_idx, inds[n_seq_items:]] = new_event_IDs2[
            "stim-flash_choices"
        ]

    print(
        "new_events:\n",
        [
            k
            for k, v in new_event_IDs2.items()
            if v in np.unique(trimmed_raw[stim_chan_idx, :])
        ],
    )

    #  ! Event IDs Version 2 >
    trimmed_raw = mne.io.RawArray(trimmed_raw, info)

    # mne.find_events(trimmed_raw)
    # trimmed_raw.plot(events=events, event_id=new_event_IDs, scalings={"eeg": 100e-6})

    new_fpath = fpath.parent / f"{fpath.stem}-trimmed_raw.fif"
    trimmed_raw.save(new_fpath, overwrite=True)

    trimmed_raw = mne.io.read_raw(new_fpath, preload=True)
    trimmed_raw.info["subject_info"] = {"name": "pilot_subj1"}

    return trimmed_raw


def preprocess_first_pilot_behav(sequences_file="TBD", res_file="TBD"):
    # ! TEMP
    sequences_file = "./global_config/old_combs/selected_combinations-format[names].csv"
    res_file = "./local/results/pilot results/2024-03-21-Pilot_human.xlsx"
    # ! TEMP

    sequences = pd.read_csv(sequences_file)
    sequences.insert(8, "figure8", sequences["solution"])

    df_behav = pd.read_excel(res_file, index_col=0)
    df_behav["trial_type"] = ["practice"] * 2 + ["experiment"] * (len(df_behav) - 2)

    # df_behav["choice_key"].value_counts()

    sequences.rename(columns={"itemid": "item_id"}, inplace=True)

    temp1 = df_behav.loc[:, ["item_id", "solution", "pattern"]]
    temp2 = sequences.loc[:, ["item_id", "solution", "pattern"]]
    df_check = temp1.merge(temp2, on="item_id", suffixes=("_behav", "_seq"))

    assert all(df_check["solution_behav"] == df_check["solution_seq"])
    assert all(df_check["pattern_behav"] == df_check["pattern_seq"])

    sequences.drop(
        columns=["solution", "pattern", "seq_order", "choice_order"], inplace=True
    )

    df_behav = df_behav.merge(sequences, how="left", on="item_id")
    df_behav.rename(columns={"item_id": "combinationID"}, inplace=True)

    return df_behav


# * ####### BEHAVIOR ANALYSIS ####### * #
def subj_analysis_behav(
    subj: str = None,
    sequences: pd.DataFrame = None,
    fig_params: dict = None,
    export_dir=None,
):

    if not fig_params:
        fig_params = default_fig_params

    # df_behav = pd.read_csv("...")
    df_behav = preprocess_first_pilot_behav()
    stim_locs = locate_stim_in_trials(df_behav)
    # stim_locs = pd.concat([pd.DataFrame(v, index=[k]*len(v)) for k,v in stim_locs.items()])
    # stim_locs.reset_index(names='stim',inplace=True)

    # ! TEMP
    subj = "pilot_subj1"
    seq_cols = [f"figure{i+1}" for i in range(8)]
    choice_cols = [f"choice{i+1}" for i in range(4)]
    selected_cols = ["combinationID"] + seq_cols + choice_cols
    export_dir = f"./local/results/analysis/subj_lvl/{subj}"
    # ! TEMP

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # sequence_IDs = ", ".join(df_behav["item_id"].astype(str))
    # query = f"SELECT * FROM combinations WHERE itemID IN ({sequence_IDs})"
    # sequences = db.execute(query)

    # * DATAFRAME "CHOICE"
    df_choice = (
        df_behav.groupby("pattern")["choice_key"].value_counts().unstack().fillna(0)
    )
    df_choice = df_choice[["a", "x", "m", "l", "timeout"]]
    df_choice["total"] = df_choice.sum(axis=1)

    # * DATAFRAME "CORRECT"
    df_crct = df_behav.groupby("pattern")["correct"].value_counts().unstack().fillna(0)
    df_crct.columns = ["incorrect", "correct", "invalid"]
    df_crct["invalid"] = df_crct["invalid"] - df_choice["timeout"]
    df_crct["timeout"] = df_choice["timeout"]
    df_crct = df_crct.astype("int")
    # df_crct["inval_timeout"] =
    df_crct["total"] = df_crct.sum(axis=1)

    for col in df_crct.columns[:-1]:
        df_crct[col + "_pct"] = round(df_crct[col] / df_crct["total"], 3)

    df_crct["inval_timeout_pct"] = (df_crct["invalid"] + df_crct["timeout"]) / df_crct[
        "total"
    ]

    df_crct.sort_values("correct_pct", ascending=False)
    df_incrct = df_crct[df_crct["inval_timeout_pct"] > 0]

    # * STATS
    # rt_stats = df_behav[df_behav["rt"] != "timeout"]["rt"].astype(float).describe()
    rt_clean = df_behav[df_behav["rt"] != "timeout"].copy()
    rt_clean["rt"] = rt_clean["rt"].astype(float)

    rt_by_pattern = rt_clean.groupby("pattern")["rt"].describe()
    rt_stats_group = rt_clean.groupby(["pattern", "correct"])["rt"].describe()
    rt_stas_global = rt_clean["rt"].describe()

    rt_global_mean = rt_stas_global["mean"]
    rt_global_sd = rt_stas_global["std"]

    df_rt_stats_group_mean = rt_stats_group["mean"].reset_index(name="mean")

    # * PATTERN GROUPING
    patt_inds = {
        pat: {"inds": [], "max_rt": None} for pat in df_behav["pattern"].unique()
    }

    for idx, row in df_behav.iterrows():
        patt_inds[row["pattern"]]["inds"].append(idx)

    for pat, vals in patt_inds.items():
        vals["max_rt"] = rt_by_pattern.loc[pat]["max"]

    # ! TEMP
    # df_behav.iloc[patt_inds['AABBAABB']['inds'], :]['pattern']

    # * PLOTS
    # data = df_behav.copy()
    # data['correct'] = data['correct'].replace({"invalid":False}).astype(int)
    # sns.barplot(data=data, x="correct", y="pattern", errorbar='sedi')
    plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    df_crct["correct_pct"].sort_values(ascending=True).plot(kind="barh")
    title = "Percent of Correct Trials"
    plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Percent")
    plt.grid(axis="x", linestyle="--")
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    df_incrct["inval_timeout_pct"].sort_values(ascending=True).plot(kind="barh")
    title = "Percent of Incorrect and Timeout Trials"
    plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Percent")
    plt.grid(axis="x", linestyle="--")
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    # * Create figure with specified size
    fig = plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    # * Create GridSpec layout within the figure
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[3, 1], figure=fig)
    # * Add subplots according to the GridSpec layout
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    # * The boxplot for correct/incorrect responses
    sns.boxplot(x="correct", y="rt", data=rt_clean, ax=ax0)
    ax0.set_title("Response Time by Correctness")
    ax0.set_ylabel("Time (s)")
    ax0.set_xlabel("")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["Incorrect", "Correct"])
    ax0.grid(axis="y", linestyle="--")
    # * The global RT boxplot on the right
    sns.boxplot(y="rt", data=rt_clean, ax=ax1, color="lightblue")
    ax1.set_title("Global Response Time")
    ax1.set_ylabel("")  # * Remove y-label to avoid duplication
    ax1.grid(axis="y", linestyle="--")
    # * Hide y-ticks for the second plot to avoid clutter
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklines(), visible=False)
    plt.setp(ax1.get_yticklines(), visible=False)
    # * Add a suptitle and adjust layout
    title = "Response Time Boxplots"
    fig.suptitle(title)
    plt.tight_layout()
    # * Save the figure
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    sns.boxplot(
        x="rt",
        y="pattern",
        data=rt_clean,
        order=rt_clean.groupby("pattern")["rt"].median().sort_values().index,
    )
    title = "Response Time Boxplot by Pattern"
    plt.title(title)
    plt.xlabel("Response Time (s)")
    plt.grid(axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    # df_behav[df_behav["rt"] != "timeout"]["rt"].astype(float).plot(kind="kde")
    sns.histplot(
        df_behav[df_behav["rt"] != "timeout"]["rt"].astype(float),
        kde=True,
        stat="percent",
    )
    title = "Response Time Density Plot"
    plt.title(title)
    plt.xlabel("Response Time (s)")
    plt.grid(linestyle="--")
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    sns.barplot(
        data=df_rt_stats_group_mean, x="mean", y="pattern", hue="correct", orient="h"
    )
    title = "Mean Response Time by Pattern and Correctness"
    plt.title(title)
    plt.grid(axis="x", linestyle="--")
    plt.xlabel("Mean Response Time (s)")
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=fig_params.dpi
    )
    plt.show()

    # plt.figure(figsize=fig_params.size, dpi=fig_params.dpi)
    # df_behav[df_behav["rt"] != "timeout"]["rt"].astype(float).hist()
    # plt.title("Response Time Histogram")
    # plt.xlabel("Response Time (s)")
    # plt.ylabel("Count")
    # plt.vlines(rt_global_mean, 0, 12, color="red", label="mean")
    # plt.vlines(rt_global_mean - rt_global_sd, 0, 12, color="yellow", label="std")
    # plt.vlines(rt_global_mean + rt_global_sd, 0, 12, color="yellow")
    # plt.legend()
    # plt.show()
    plt.close("all")

    df_behav["correct"].value_counts() / df_behav.shape[0]

    results = dict(
        df_behav=df_behav,
        df_choice=df_choice,
        df_crct=df_crct,
        df_incrct=df_incrct,
        df_rt_stats_group_mean=df_rt_stats_group_mean,
        df_rt_by_pattern=rt_by_pattern,
        rt_stas_global=rt_stas_global,
        patt_inds=patt_inds,
        stim_locs=stim_locs,
    )

    for name, res in results.items():
        if isinstance(res, pd.DataFrame) or isinstance(res, pd.Series):
            res.to_csv(export_dir / f"{name}.csv")

    return results


def custom_evoked_plot(evoked, selected_chans):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    # fig.suptitle(f"ERP - {stim}")
    evoked.plot(axes=ax, show=False)
    lines = ax.get_lines()
    for idx, line in enumerate(lines):
        if idx not in selected_chans:
            line.remove()
    return fig


# * ####### EEG ANALYSIS ####### * #
def get_events_df(events, sampling_freq, event_IDs_inv):
    # * Initialize 'events_df'
    events_df = pd.DataFrame(events, columns=["sampleN", "last_val", "event_id"])
    events_df["event_name"] = events_df["event_id"].replace(event_IDs_inv)
    events_df["event_type"] = ""
    events_df.insert(0, "time", events_df["sampleN"].div(sampling_freq))
    events_df["trial"] = events_df["event_name"].eq("trial_start").cumsum() - 1
    # events_df["event_name"].replace({"all_stim":"stim_flash"}, inplace=True)

    # * TODO: Write comment
    # seq_cols = [c for c in df_behav.columns if re.search(r"figure\d{1,2}", c)]
    # seq_cols.remove("figure8")  # ! TEMP # TODO: Remove
    # seq_len = len(seq_cols)
    # choice_cols = [c for c in df_behav.columns if re.search(r"choice\d{1,2}", c)]

    # * TODO: Write comment
    # n_trials = df_behav.index.max() + 1
    # for trial in range(n_trials):
    #     trial_idx = trial_inds[trial]
    #     figs = df_behav.loc[trial, seq_cols]
    #     choices = df_behav.loc[trial, choice_cols]
    #     stim_flashes = events_df.query(
    #         "trial == @trial & event_name.str.contains('stim-flash')", engine="python"
    #     )
    #     stim_flashes_inds = stim_flashes.index
    #     events_df.loc[stim_flashes_inds[:seq_len], "event_name"] = figs.values
    #     events_df.loc[stim_flashes_inds[:seq_len], "event_type"] = "stim-flash_sequence"
    #     events_df.loc[stim_flashes_inds[seq_len:], "event_name"] = choices.values
    #     events_df.loc[stim_flashes_inds[seq_len:], "event_type"] = "stim-flash_choice"
    stim_seq_inds = events_df.query("event_name.str.contains('flash_seq')").index
    stim_choice_inds = events_df.query("event_name.str.contains('flash_choice')").index
    stim_all_inds = events_df.query("event_name.str.contains('all_stim')").index

    events_df.loc[stim_seq_inds, "event_type"] = "stim-seq"
    events_df.loc[stim_choice_inds, "event_type"] = "stim-choice"
    events_df.loc[stim_all_inds, "event_type"] = "stim-all"

    events_df.loc[
        events_df.query("event_name.str.contains('all_stim')").index, "event_type"
    ] = "stim-all_stim"

    possible_resp_events = ["a", "x", "m", "l", "invalid", "timeout"]
    possible_exp_events = [
        "exp_start",
        "block_start",
        "block_end",
        "trial_start",
        "trial_end",
    ]

    events_df.loc[
        events_df.query(f"event_name in {possible_resp_events}").index, "event_type"
    ] = "response"

    events_df.loc[
        events_df.query(f"event_name in {possible_exp_events}").index, "event_type"
    ] = "exp_stage"

    return events_df


def get_trial_lvl_data(raw, event_name="trial_start"):

    trial_inds = [
        i for i, a in enumerate(raw.annotations) if a["description"] == event_name
    ]

    trial_onsets = [raw.annotations[i]["onset"] for i in trial_inds]

    first_flash_inds = [i + 1 for i in trial_inds]
    first_flash_onsets = [raw.annotations[i]["onset"] for i in first_flash_inds]

    pre_first_flash_duration = 0.9  # seconds

    trial_windows = [
        (onset - pre_first_flash_duration, trial_onsets[i + 1])
        for i, onset in enumerate(first_flash_onsets[:-1])
    ]

    # * Add last trial
    trial_windows.append((first_flash_onsets[-1], raw.times[-1]))

    trial_data = (
        raw.copy().crop(tmin=start, tmax=stop) for start, stop in trial_windows
    )

    return trial_data, trial_windows


def apply_ica(raw, save_fname=None):
    # ! TEMP
    save_dir = "./local/results/analysis/ICA"
    # ! TEMP

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ica_file = save_dir / f"{save_fname}-ica.fif"

    if ica_file.exists():
        ica = mne.preprocessing.read_ica(ica_file)
    else:
        ica = ICA(
            n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )

        ica.fit(raw)
        ica.save(ica_file)

    # * PLOTTING ICA RESULTS
    # raw.load_data()
    # ica.plot_sources(raw, show_scrollbars=False, show=True)

    # ica.plot_components()
    # * blinks
    # ica.plot_overlay(raw, exclude=[0], picks="eeg")

    # ica.plot_properties(raw, picks=[0])

    # * SELECTING ICA COMPONENTS AUTOMATICALLY
    ic_labels = label_components(raw, ica, method="iclabel")

    print(ic_labels["labels"])
    # ica.plot_properties(raw, picks=[0, 12], verbose=False)

    labels = ic_labels["labels"]

    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"Excluding these ICA components: {exclude_idx}")

    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    # raw.plot()
    # reconst_raw.plot()
    return reconst_raw


def get_spatial_colors_manual(raw):
    # * NOTE: keep for later
    import numpy as np
    import matplotlib.pyplot as plt
    from mne import find_layout

    # Assuming 'raw' is your already loaded Raw object
    info = raw.info

    # Find the layout for EEG
    layout = find_layout(
        info, ch_type="eeg", exclude=[]
    )  # Exclude bad channels if necessary
    pos = layout.pos.copy()

    # Normalize the positions
    x, y = pos[:, 0], pos[:, 1]
    x -= np.mean(x)
    y -= np.mean(y)
    norms = np.sqrt(x**2 + y**2)
    norms /= norms.max()

    # Get colors from a colormap
    colors = plt.cm.viridis(norms)

    # * Create a colorbar with the colormap
    # fig, ax = plt.subplots(figsize=(8, 5))
    # cbar = plt.colorbar(
    #     plt.cm.ScalarMappable(cmap="viridis"), ax=ax, orientation="horizontal"
    # )
    # cbar.set_label("Normalized Value")

    # # Display each color as a horizontal line
    # for i, color in enumerate(colors):
    #     ax.hlines(i, 0, 1, color=color, linewidth=5)

    # # Set the limits and labels
    # ax.set_ylim(0, len(colors))
    # ax.set_xlim(0, 1)
    # ax.set_xlabel("Value")
    # ax.set_yticks([])  # Hide y-axis ticks

    # plt.show()

    # * Set custom channel colors on the evoked plot
    # fig = evoked.plot(spatial_colors=False, show=False)

    # # Get the lines from the plot and set custom colors
    # lines = fig.axes[0].lines  # This accesses the lines from the main data axes
    # for line, color in zip(lines, custom_colors):
    #     line.set_color(color)


def get_spatial_colors(evoked):
    fig, ax = plt.subplots()
    evoked.plot(axes=ax, show=False)
    lines = ax.get_lines()
    channel_colors = {
        evoked.ch_names[i]: line.get_color() for i, line in enumerate(lines)
    }
    plt.close()

    return channel_colors


def group_analysis_eeg():
    epochs_files = list(results_dir.glob("*-epo.fif"))
    patterns = ["AABBAABB", "ABABABAB", "AABBCCDD", "ABCDCDBA"]

    def load_by_pattern(patt):

        for file in epochs_files:
            epochs = mne.read_epochs(file)[f"pattern == '{patt}'"]

    # epochs = mne.read_epochs("...")

    pass


def subj_analysis_eeg(
    fpath=None, subj_name=None, stim_channel="Status", fig_params: dict = None
):

    # ! TEMP
    stim_channel = "Status"
    fig_params = default_fig_params
    fpath = directories.results / "2024-03-21-Pilot_human.bdf"
    subj_name = "pilot_subj1"
    # ! TEMP

    export_dir = wd / f"local/results/analysis/subj_lvl/{subj_name}"
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if fig_params is None:
        fig_params = default_fig_params

    behav_results = subj_analysis_behav()
    df_rt_by_pattern = behav_results["df_rt_by_pattern"]
    patt_inds = behav_results["patt_inds"]
    stim_locs = behav_results["stim_locs"]
    df_behav = behav_results["df_behav"]
    # behav_results.keys()
    # stim_locs['baby-carriage'][0]

    # event_IDs = config["event_IDs"]
    event_IDs = exp_config["local"]["event_IDs"]
    event_IDs_inv = invert_dict(event_IDs)

    # raw = mne.io.read_raw_bdf(
    #     fpath,
    #     eog=None,
    #     misc=None,
    #     stim_channel=stim_channel,  # * also works with 'auto'
    #     exclude=(),
    #     infer_types=False,
    #     include=None,
    #     preload=False,

    #     units=None,
    #     encoding="utf8",
    #     verbose=None,
    # )

    raw = preprocess_first_pilot_eeg(fpath)
    subj_name = raw.info["subject_info"]["name"]  # ! TEMP

    raw.info["subject_info"]["name"] = subj_name

    # * Drop EOG and EMG channels
    eog_emg_chans = re.compile(r"(EOG|EMG)", re.IGNORECASE)
    chans_to_drop = [ch for ch in raw.ch_names if eog_emg_chans.search(ch)]
    raw.drop_channels(chans_to_drop)

    # * Montage
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage)

    # * Channel Groups
    ch_names = raw.ch_names.copy()
    ch_names.remove(stim_channel)
    ch_groups = exp_config["local"]["EEG"]["ch_groups"]
    assert all([ch in ch_names for group in ch_groups.values() for ch in group])

    for ch_group, chans in ch_groups.items():
        fig, ax = plt.subplots()
        ax.set_title(ch_group)
        chan_fig = montage.plot(kind="topomap", show_names=chans, show=False, axes=ax)
        plt.close()
        ch_groups[ch_group] = {
            "channels": chans,
            "fig": chan_fig,
            "pos": montage.get_positions()["ch_pos"],
        }

    # * Info
    info = raw.info
    sampling_freq = info["sfreq"]

    # * Preprocessing
    # raw.filter(l_freq=0.1, h_freq=100, verbose=True)
    raw.set_eeg_reference(ref_channels="average")
    raw.filter(l_freq=0.1, h_freq=100, verbose=True)
    raw = apply_ica(raw, save_fname=fpath.stem)
    raw.filter(l_freq=1, h_freq=10, verbose=True)

    events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        output="onset",
        consecutive="increasing",
        min_duration=0,
        shortest_event=1,
        mask=None,
        uint_cast=False,
        mask_type="and",
        initial_event=False,
        verbose=None,
    )

    trial_inds = np.where(events[:, 2] == event_IDs["trial_start"])[0]

    events_df = get_events_df(
        events=events,
        sampling_freq=sampling_freq,
        event_IDs_inv=event_IDs_inv,
    )

    seq_cols = [c for c in df_behav.columns if re.search(r"figure\d{1,2}", c)]
    choice_cols = [c for c in df_behav.columns if re.search(r"choice\d{1,2}", c)]

    for trial_idx in range(len(trial_inds)):
        seq_inds = events_df.query(f"trial=={trial_idx} & event_type=='stim-seq'").index
        stim_names = df_behav.query(f"idx == {trial_idx}")[seq_cols]
        stim_names = stim_names.values.flatten()[:-1]
        events_df.loc[seq_inds, "event_name"] = stim_names

        choice_inds = events_df.query(
            "trial==@trial_idx & event_type=='stim-choice'"
        ).index
        choice_names = df_behav.query("idx == @trial_idx")[choice_cols]
        choice_names = choice_names.values.flatten()
        events_df.loc[choice_inds, "event_name"] = choice_names

    annotations = mne.annotations_from_events(
        events=events,
        event_desc=event_IDs_inv,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )

    raw.set_annotations(annotations)
    fname = f"preprocessed_raw-subj[{subj_name}]-raw.fif"

    raw.save(
        export_dir / fname,
        picks=None,
        tmin=0,
        tmax=None,
        buffer_size_sec=None,
        drop_small_buffer=False,
        proj=False,
        fmt="single",
        overwrite=False,
        split_size="2GB",
        split_naming="neuromag",
        verbose=None,
    )

    trial_data, trial_windows = get_trial_lvl_data(raw, event_name="trial_start")

    # trial_data = [next(trial_data) for _ in range(4)]
    # trial_data[1].plot()

    patt_trial_inds = {}
    for pat, patt_info in patt_inds.items():
        patt_trial_inds[pat] = [trial_inds[i] for i in patt_info["inds"]]

    pres_flash_duration = 0.6 * 11  # seconds
    min_rt = 1.07  # seconds
    min_iti = 1  # seconds
    max_resp_duration = 8  # seconds
    epoch_start_time = -(pres_flash_duration + 1)
    epoch_baseline = (-(pres_flash_duration + 0.2), -pres_flash_duration)

    epoch_event = "stim-all_stim"

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_IDs[epoch_event],
        # event_id = [1,2,3,4,5, 33],
        tmin=epoch_start_time,
        # tmax=min_rt + min_iti,
        tmax=max_resp_duration,
        baseline=epoch_baseline,
        # picks=,
        detrend=1,
        preload=True,
    )

    metadata = df_behav[["rt", "choice", "correct", "pattern"]].copy()
    metadata["rt"] = metadata["rt"].replace("timeout", np.nan)
    metadata = metadata.astype({"rt": float, "correct": bool})
    epochs.metadata = metadata

    # epochs.plot_topo_image()

    fname = f"epochs-event[{epoch_event}]-subj[{subj_name}]-epo.fif"
    epochs.save(export_dir / fname, overwrite=True)

    fname = f"events-subj[{subj_name}]-eve.fif"
    mne.write_events(export_dir / fname, events, overwrite=True)
    # epochs = mne.read_epochs(export_dir / fname)
    # epochs.compute_psd().plot_topo()

    evoked = epochs.average()
    evoked.plot_image()

    fig, ax = plt.subplots(figsize=(16, 6), dpi=fig_params.dpi)
    evoked.plot(axes=ax, show=False)
    # min, max = [i * 1e6 for i in [evoked.data.min(), evoked.data.max()]]
    min, max = 0, 10
    min_y, max_y = [-10, 25]
    # min, max = [-40, 85]
    ax.set_ylim(min_y, max_y)
    text_yloc = max * 1.02
    # fig.suptitle(f"ERP (locked on all stim pres) - {patt}")
    for idx, line in enumerate(np.arange(-0.6 * 11, 0.1, 0.6)):
        ax.axvline(line, ls="--", color="black", lw=0.5)
        if idx <= 6:
            line_text = "stim."  # f"stim. {idx+1}"
        elif 7 <= idx <= 10:
            line_text = "choice"  # f"choice {idx-6}"
        elif idx == 11:
            line_text = "all stim"
        ax.text(
            line,
            text_yloc,
            line_text,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=15,
        )
    ax.set_xlim(None, 2)
    plt.savefig(export_dir / "ERP.png", dpi=fig_params.dpi, bbox_inches="tight")

    evoked_plot_funcs = [
        "plot",
        "plot_image",
        "plot_joint",
        "plot_psd",
        "plot_psd_topo",
        "plot_psd_topomap",
        "plot_sensors",
        "plot_topo",
        "plot_topomap",
    ]

    # evoked.plot_topomap(
    #     times=np.arange(-6.6, 0.1, 0.62), ch_type="eeg", time_unit="s"
    # )
    # plt.show()

    # evoked.plot_topomap(
    #     times=np.arange(0, 4, 0.5), ch_type="eeg", time_unit="s"
    # )
    # plt.show()

    epochs["pattern=='AAAAAAAA'"].average().crop(tmin=-2, tmax=2).plot()

    epochs.plot_image(picks="Pz")

    # epochs.plot()

    evokeds = dict()
    for patt in epochs.metadata["pattern"].unique():
        evokeds[str(patt)] = epochs[f"pattern == '{patt}'"].average()

    selected_times = times = [-0.6, 0, 0.6]
    selected_chans = ch_groups["frontal"]["channels"]
    selected_patt = "AABBAABB"
    data = evokeds[selected_patt]

    fig = data.plot_joint(selected_times, title=selected_patt, show=False)
    lines = [a for a in fig.get_axes() if a.get_ylabel() == "µV"][0].get_lines()
    chan_lines = enumerate(lines[: -len(selected_times)])
    for idx, line in chan_lines:
        name = ch_names[idx]
        line.set_label(name)
        if name not in selected_chans:
            line.set_visible(False)
    plt.show()

    selected_patts = ["AAAAAAAA", "AABBAABB", "ABBBABBB", "ABABCDCD"]
    selected_evokeds = {k: v for k, v in evokeds.items() if k in selected_patts}

    mne.viz.plot_compare_evokeds(selected_evokeds, cmap=("pattern", "viridis"))
    plt.show()

    for patt, evoked in selected_evokeds.items():
        fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_params.dpi)
        evoked.plot(axes=ax)
        # plt.savefig(f"results/ERP-{patt}.png", dpi=fig_params.dpi)
        plt.show()

    def plot_epochs_together(selected_epochs, ch_group=None):
        # selected_epochs = epochs["pattern=='AAAAAAAA'"]
        # ch_group = ch_groups["frontal"]['channels']

        max_rt = selected_epochs.metadata["rt"].max()

        if ch_group is not None:
            selected_chans = enumerate(epochs.ch_names)
            selected_chans = {ch: idx for idx, ch in selected_chans if ch in ch_group}
        else:
            selected_chans = {ch: idx for idx, ch in enumerate(epochs.ch_names)}

        n_epochs = len(selected_epochs)

        fig, ax = plt.subplots(nrows=n_epochs, figsize=(12, 18), dpi=fig_params.dpi)

        for idx in range(n_epochs):
            trial_rt = selected_epochs[idx].metadata["rt"].values[0]

            epoch_data = selected_epochs[idx].average().crop(tmax=trial_rt * 1.25)
            selected_times = np.linspace(epoch_data.times[0], trial_rt, 4)
            epoch_data.plot(axes=ax[idx], spatial_colors=True, show=False)
            # epoch_data.plot_joint(times=selected_times)

            lines = ax[idx].get_lines()
            for idx_line, line in enumerate(lines):
                if idx_line not in selected_chans.values():
                    line.remove()

            ax[idx].vlines(
                trial_rt, *ax[idx].get_ylim(), color="black", ls="--", lw=0.85
            )
            ax[idx].set_title("")
            ax[idx].texts[1].set_visible(False)

            if idx < n_epochs - 1:
                ax[idx].set_xlabel("")
                # ax[idx].set_xticks(np.arange(-6, 2, 2))
                # ax[idx].set_xticks([])
                ax[idx].set_xticklabels([])

            ax[idx].set_xlim(left=None, right=max_rt * 1.5)
            ax[idx].grid(
                axis="x",
                linestyle="--",
            )

        selected_epochs.metadata
        epochs_data = selected_epochs.get_data()
        epoch1 = epochs_data[0]

        # for i in range(epoch1.shape[0]):
        selected_epochs[0].average().plot()
        plt.figure(figsize=(12, 5), dpi=fig_params.dpi)
        for i in range(epoch1.shape[0] - 1):
            plt.plot(selected_epochs.times, epoch1[i] * 1e6)
        plt.xlim(selected_epochs.times[0], selected_epochs.times[-1])
        plt.show()

    evoked.plot(spatial_colors=True, axes=ax, show=False)

    for patt, evoked in selected_evokeds.items():
        mean_rt = df_rt_by_pattern.loc[patt, "mean"]
        std_rt = df_rt_by_pattern.loc[patt, "std"]
        rt_range = (mean_rt - std_rt, mean_rt + std_rt)

        if rt_range[1] > max_resp_duration:
            rt_range = (mean_rt - std_rt, max_resp_duration)

        time_range = np.logical_and(
            evoked.times > rt_range[0], evoked.times < rt_range[1]
        )
        times = evoked.times[time_range]

        # fig, ax = plt.subplots(figsize=(12, 5), dpi=200)  # fig_params.dpi
        fig, axes = plt.subplots(nrows=2, figsize=(16, 13), dpi=200)  # fig_params.dpi
        min, max = [i * 1e6 for i in [evoked.data.min(), evoked.data.max()]]
        # min, max = [-40, 85]
        axes[0].set_ylim(min * 1.3, max * 1.3)
        axes[1].set_ylim(min * 1.3, max * 1.3)
        text_yloc = max / 2.8
        fig.suptitle(f"ERP (locked on all stim pres) - {patt}")
        for ax in axes:
            for idx, line in enumerate(np.arange(-0.6 * 11, 0.1, 0.6)):
                ax.axvline(line, ls="--", color="black", lw=0.5)
                if idx <= 6:
                    line_text = "stim."  # f"stim. {idx+1}"
                elif 7 <= idx <= 10:
                    line_text = "choice"  # f"choice {idx-6}"
                elif idx == 11:
                    line_text = "all stim"
                ax.text(
                    line, text_yloc, line_text, rotation=90, va="bottom", ha="right"
                )

        evoked.plot(spatial_colors=True, gfp=True, axes=axes[0], show=False)
        evoked.copy().crop(tmax=2).plot(axes=axes[1], show=False)
        plt.show()

        # evoked.get_peak(tmin=rt_range[0], tmax=rt_range[1])
        # evoked.plot_topomap(times=mean_rt, ch_type="eeg", time_unit="s")
        # evoked.plot_joint()
        # evoked.compute_psd().plot_topomap()

    def hmp_analysis():
        import hmp

        results_dir = wd / "local/results/analysis/subj_lvl"
        subj = "pilot_subj1"

        file = (
            results_dir
            / f"{subj}/epochs-event[stim-all_stim]-subj[pilot_subj1]-epo.fif"
        )
        file.exists()

        eeg_data = hmp.utils.read_mne_data(
            str(file),
            epoched=True,
            lower_limit_RT=0.05,
            upper_limit_RT=8,
            # event_id=event_IDs["stim-all_stim"],
            event_id=[
                event_IDs[event] for event in ["stim-all_stim", "stim-flash_sequence"]
            ],
            resp_id=[event_IDs[event] for event in ["a", "x", "m", "l", "invalid"]],
            events_provided=events,
            rt_col="rt",
            # verbose=False,
        )

        hmp_data = hmp.utils.transform_data(eeg_data, n_comp=4)

        positions = epochs.get_montage().get_positions()["ch_pos"]

        # * Extracts x, y coordinates
        positions = np.array([pos for pos in positions.values()])[:, :2]

        hmp.visu.plot_components_sensor(hmp_data, positions)

        init = hmp.models.hmp(
            data=hmp_data,
            epoch_data=eeg_data,
            sfreq=eeg_data.sfreq,
            event_width=50,
            distribution="gamma",
            shape=2,
            cpus=6,
        )
        # dir(init)

        # * Fitting
        n_events = 5
        # * function to fit an instance of a 10 events model
        # selected = init.fit_single(n_events)

        selected = init.fit_single(
            n_events, method="random", starting_points=1, return_max=False, cpus=6
        )

        # * Visualizing
        hmp.visu.plot_topo_timecourse(
            eeg_data,
            # selected.sel(iteration=1),
            selected,
            positions,
            init,
            magnify=1,
            sensors=False,
            # times_to_display=np.mean(np.cumsum(sim_source_times, axis=1), axis=0),
            as_time=True,
            # dpi=300,
            # times_to_display=np.arange(0, 3501, 500),
            # figsize=(18, 6),
        )

        plt.plot(init.template, "x")
        plt.ylabel("Normalized value")
        plt.xlabel("Samples NOT time")
        plt.show()

        plt.plot(init.data_matrix[:, 0, :])
        plt.show()

        # * ###############
        cpus = 6
        correct_trials = hmp.utils.condition_selection(
            hmp_data, eeg_data, True, variable="correct"
        )
        init_correct = hmp.models.hmp(
            correct_trials, eeg_data, sfreq=eeg_data.sfreq, cpus=cpus
        )
        dir(init_correct)
        init_correct.n_trials
        init_correct.n_samples
        init_correct.n_dims

        # * apply fit function to build maximal model
        estimates_correct = init_correct.fit()
        hmp.visu.plot_topo_timecourse(
            eeg_data, estimates_correct, positions, init_correct, as_time=True
        )

    def get_erp_per_stim():
        stim_IDs = [
            id
            for id in events_df["event_name"].unique().tolist()
            if id not in exp_config["local"]["event_IDs"]
        ]

        base_id = 900
        assert base_id not in np.unique(events[:, 2])
        stim_IDs = {stim: base_id + idx for idx, stim in enumerate(stim_IDs)}

        events_df2 = events_df.copy()
        event_names = events_df2["event_name"].to_list()
        stim_inds = [idx for idx, name in enumerate(event_names) if name in stim_IDs]
        stim_names = [name for name in event_names if name in stim_IDs.keys()]

        events_df2.loc[stim_inds, "event_id"] = [stim_IDs[name] for name in stim_names]
        new_events = events_df2[["sampleN", "last_val", "event_id"]].values
        # * Get sample numbers of each stimulus (e.g., all times 'camera' was presented)
        # sample_numbers = pd.Series(events[:, 0])

        selected_stims = events_df2.query("event_name in @stim_names")["event_name"]
        selected_stims = selected_stims.value_counts().index.values
        selected_stims = [stim_IDs[stim] for stim in selected_stims]

        selected_chans = []
        for g in ["occipital", "parietal"]:
            selected_chans += ch_groups[g]["channels"]

        raw.plot_sensors(show_names=selected_chans)
        selected_chans = [ch_names.index(ch) for ch in selected_chans]

        combined_epochs = []
        evokeds = {}

        for stim in selected_stims:
            epochs = mne.Epochs(
                raw,
                new_events,
                event_id=stim,
                tmin=-0.100,
                tmax=0.600,
                # baseline=,
                # picks=,
                detrend=1,
                preload=True,
            )
            combined_epochs.append(epochs)
            evoked = epochs.average()
            evokeds[stim] = evoked

            fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_params.dpi)
            title = f"ERP - {invert_dict(stim_IDs)[stim]}"
            fig.suptitle(title)
            evoked.plot(axes=ax, show=False)
            lines = ax.get_lines()
            for idx, line in enumerate(lines):
                if idx not in selected_chans:
                    line.remove()
            fname = f"ERP-{invert_dict(stim_IDs)[stim].replace('-', '_')}"
            plt.savefig(
                export_dir / f"{fname}.png", dpi=fig_params.dpi, bbox_inches="tight"
            )
            # plt.show()
            plt.close()

        main_evoked = mne.concatenate_epochs(combined_epochs).average()
        custom_evoked_plot(main_evoked, selected_chans)
        main_evoked.plot()

        fig, ax = plt.subplots()
        for chan in selected_chans:
            ax.plot(
                main_evoked.data[chan, :] * 1e6, label=ch_names[chan], color="black"
            )
        ax.plot(main_evoked.data[selected_chans].mean(axis=0) * 1e6, color="red")

        custom_evoked_plot(
            main_evoked, [ch_names.index(c) for c in ch_groups["occipital"]["channels"]]
        )

        return evokeds

    # * ######### PLOTS #########
    def plots():
        for event in [event_IDs["all_stim"]]:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_params.dpi)
            evoked = epochs[str(event)].average()
            # min, max = [i * 1E6 for i in [evoked.data.min(), evoked.data.max()]]
            min, max = [-40, 85]
            ax.set_ylim(min * 1.3, max * 1.3)
            text_yloc = max / 2.8
            fig.suptitle(f"ERP (locked on all stim pres)")  # - {event_IDs_inv[event]}")
            for idx, line in enumerate(np.arange(-0.6 * 11, 0.1, 0.6)):
                ax.axvline(line, ls="--", color="black", lw=0.5)
                if idx <= 6:
                    line_text = "stim."  # f"stim. {idx+1}"
                elif 7 <= idx <= 10:
                    line_text = "choice"  # f"choice {idx-6}"
                elif idx == 11:
                    line_text = "all stim"
                ax.text(
                    line, text_yloc, line_text, rotation=90, va="bottom", ha="right"
                )
            evoked.plot(spatial_colors=True, axes=ax, show=False)
            plt.savefig(f"results/ERP-[All_Patterns].png", dpi=fig_params.dpi)
            # plt.show()
        plt.close("all")

        for patt, info in patt_inds.items():
            fig, ax = plt.subplots(figsize=(12, 5), dpi=fig_params.dpi)
            evoked = epochs[info["inds"]].average()
            # min, max = [i * 1E6 for i in [evoked.data.min(), evoked.data.max()]]
            min, max = [-40, 85]
            ax.set_ylim(min * 1.3, max * 1.3)
            text_yloc = max / 2.8
            fig.suptitle(f"ERP (locked on all stim pres) - {patt}")
            for idx, line in enumerate(np.arange(-0.6 * 11, 0.1, 0.6)):
                ax.axvline(line, ls="--", color="black", lw=0.5)
                if idx <= 6:
                    line_text = "stim."  # f"stim. {idx+1}"
                elif 7 <= idx <= 10:
                    line_text = "choice"  # f"choice {idx-6}"
                elif idx == 11:
                    line_text = "all stim"
                ax.text(
                    line, text_yloc, line_text, rotation=90, va="bottom", ha="right"
                )
            evoked.plot(spatial_colors=True, axes=ax, show=False)
            plt.savefig(f"results/ERP-[{patt}].png", dpi=fig_params.dpi)
            # plt.show()
        plt.close("all")

        evoked = epochs.average()
        # evoked_arr = evoked.data #* 1e6

        dpi = 300
        fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)  # dpi=fig_params.dpi)
        fig.suptitle("ERP (locked on button press)")
        for line in np.arange(-pres_flash_duration, 0, 0.6):
            ax.axvline(line, ls="--", color="black", lw=0.5)
        # highlight_range = np.arange(-pres_flash_duration, 0, 0.6)
        # highlight_range = [(i, i +0.1) for i in highlight_range]
        highlight_range = None
        evoked.plot(axes=ax, gfp=False, highlight=highlight_range, show=False)
        # fig.tight_layout()
        # plt.savefig("results/erp1.png", dpi=fig_params.dpi)
        plt.savefig("results/erp1_2.png", dpi=dpi)
        # plt.show()

        xlims = [(None, None), (-3, None), (-2, None)]
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle("ERP (locked on button press)")
        for i, ax in enumerate(axes.flatten()):
            for line in np.arange(-pres_flash_duration, 0, 0.6):
                ax.axvline(line, ls="--", color="black", lw=0.5)
            # highlight_range = np.arange(-pres_flash_duration, 0, 0.6)
            # highlight_range = [(i, i +0.1) for i in highlight_range]
            highlight_range = None
            evoked.plot(axes=ax, gfp=False, highlight=highlight_range, show=False)
            ax.set_xlim(*xlims[i])
        fig.tight_layout()
        plt.savefig("results/erp2.png", dpi=fig_params.dpi)
        # plt.show()


def website_plots(df_behav, df_crct, df_incrct, rt_stats_group_mean, rt_clean):
    import plotly.express as px
    import plotly.figure_factory as ff

    # save_dir = Path("website")

    def save_plot(fig, type: str, fname: str, save_dir: str = None):
        if not save_dir:
            save_dir = Path("website/figs")
            save_dir.mkdir(exist_ok=True, parents=True)

        if type == "html":
            fig.write_html(save_dir / f"{fname}.html", full_html=False)
        elif type == "json":
            fig.write_json(save_dir / f"{fname}.json")

        # Save to HTML
        # figname = "example_plot"
        # fig.write_html(save_dir/f"{figname}.html", full_html=False)
        # fig.write_json(save_dir/f"{figname}.json")

    # Create a plot
    # df = px.data.iris()  # Example dataset
    # fig = px.scatter(df, x="sepal_width", y="sepal_length")

    title = "Percent of Correct Trials"
    fig1 = px.bar(
        df_crct["correct_pct"].sort_values(ascending=True), orientation="h", title=title
    )
    fig1.update_layout(showlegend=False)

    title = "Percent of Incorrect and Timeout Trials"
    fig2 = px.bar(
        df_incrct["inval_timeout_pct"].sort_values(ascending=True),
        orientation="h",
        title=title,
    )
    fig2.update_layout(showlegend=False)

    title = "Response Time Boxplot"
    fig3 = px.box(df_behav[df_behav["rt"] != "timeout"], y="rt", title=title)
    # fig3.update_layout(showlegend=False)

    title = "Response Time Density Plot"
    fig4 = px.histogram(
        df_behav[df_behav["rt"] != "timeout"],
        x="rt",
        title=title,
        histnorm="probability density",
    )

    title = "Response Time Density Plot (bis)"
    data = pd.to_numeric(df_behav["rt"], errors="coerce")  # Converts non-numeric to NaN
    data = data.dropna()  # Drops rows where 'rt' is NaN
    fig4_bis = ff.create_distplot(
        [data.tolist()], group_labels=["test"], show_hist=False, show_rug=False
    )
    fig4_bis.update_layout(title=title, showlegend=False)

    title = "Mean Response Time by Pattern and Correctness"
    fig5 = px.bar(
        rt_stats_group_mean,
        x="mean",
        y="pattern",
        color="correct",
        title=title,
        barmode="group",
    )

    title = "Mean Response Time by Correctness"
    fig6 = px.bar(rt_clean.groupby("correct")["rt"].mean(), title=title)
    fig6.update_layout(showlegend=False)

    for fig in [fig1, fig2, fig3, fig4, fig4_bis, fig5, fig6]:
        save_plot(fig, "html", fig.layout.title.text.lower().replace(" ", "_"))
        save_plot(fig, "json", fig.layout.title.text.lower().replace(" ", "_"))
        fig.show()


def compare_with_ANNs():

    subj = "pilot_subj1"
    subj_results = {
        d.name: d.glob("*.csv")
        for d in Path("./results/analysis").glob("*")
        if d.is_dir()
    }
    subj_results = {f.stem: pd.read_csv(f) for f in subj_results[subj]}
    subj_results.keys()

    df_behav = subj_results["df_behav"].copy()
    n_by_pattern = df_behav["pattern"].value_counts()
    df_behav["correct"].replace({"True": 1, "False": 0, "invalid": 0}, inplace=True)
    df_behav.groupby("pattern")["correct"].mean()

    ann_results = [
        f for f in (Path.cwd().parent / "ANNs/results/analysis").glob("*.csv")
    ]
    ann_results = {f.stem: pd.read_csv(f) for f in ann_results}
    ann_patterns = ann_results["results_by_pattern"]

    selected_models = ["chatgpt", "claude"]
    ann_patterns_selected = ann_patterns.query("model.isin(@selected_models)")

    subj_patterns = subj_results["df_crct"][["pattern", "correct"]]
    subj_patterns.insert(0, "subj", "subj_pilot01")

    subj_patterns = subj_patterns.rename(columns={"correct": "score", "subj": "type"})
    ann_patterns_selected = ann_patterns_selected.rename(columns={"model": "type"})

    # *
    data = pd.concat([subj_patterns, ann_patterns_selected], ignore_index=True)
    data = data.sort_values("pattern")
    fig, ax = plt.subplots(figsize=(12, 8))  # , dpi=300)
    sns.barplot(data=data, x="score", y="pattern", hue="type", axes=ax)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    ann_patterns_selected = pd.DataFrame(
        ann_patterns.groupby("pattern")["score"].mean()
    )
    ann_patterns_selected["type"] = "ANNs"
    ann_patterns_selected.reset_index(inplace=True)
    data = pd.concat([subj_patterns, ann_patterns_selected], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 8))  # , dpi=300)
    sns.barplot(data=data, x="score", y="pattern", hue="type", axes=ax)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # * Top performing models
    export_dir = Path("./results/analysis/group_lvl")
    export_dir.mkdir(parents=True, exist_ok=True)

    seq_per_pattern = 6
    accuracy = ann_patterns.groupby(["model"])["score"].mean() / seq_per_pattern

    top_models = accuracy[accuracy >= 0.4].sort_values(ascending=False).index

    top_models_res = pd.DataFrame(
        ann_patterns.query("model in @top_models")
        .groupby(["model", "pattern"])["score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_models_res["type"] = "ANNs"
    top_models_res.reset_index(inplace=True)

    data = pd.concat([subj_patterns, top_models_res], ignore_index=True)
    data["score"] = (data["score"] / seq_per_pattern) * 100
    # overall = data.copy().groupby("pattern")['score'].mean()
    # overall = overall.reset_index()
    # overall['type'] = "overall"
    # data = pd.concat([data, overall], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 8))  # , dpi=300)
    sns.barplot(data=data, x="score", y="pattern", hue="type", axes=ax)
    title = "Top Performing Models vs. Humans"
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.grid(axis="x", linestyle="--")
    ax.set_xlabel("Accuracy (%)")
    plt.savefig(export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)


def group_analysis_online(export_dir=None):
    export_dir = wd / "online/results/analysis/group_lvl"
    export_dir.mkdir(parents=True, exist_ok=True)

    color_palette = sns.color_palette()
    data_dir = wd / "online/results/2024-04-18"
    files = [f for f in data_dir.glob("*csv")]
    demographics = pd.read_csv(files[2])
    demographics.columns = demographics.columns.str.lower()
    demographics.age.describe()
    demographics.sex.value_counts()
    demographics["age"].plot(
        kind="hist",
        title="Age",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(demographics["age"], bins=15, kde=True, ax=axes[0])
    sns.barplot(
        data=demographics["sex"].value_counts().reset_index(),
        y="count",
        hue="sex",
        ax=axes[1],
    )
    axes[1].set_xlabel("Sex")
    axes[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()
    # sns.barplot(data=demographics, y='age', hue='sex')

    pie_cols = ["language", "nationality", "country of residence", "student status"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for idx, col in enumerate(pie_cols):
        demographics[col].value_counts().plot(kind="pie", title=col, ax=axes.flat[idx])
        # sns.set_style("whitegrid")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

    sequences = pd.read_csv(files[1])

    # for idx, row in sequences.iterrows():
    #     seq = row.iloc[1:9].values
    #     choices = row.iloc[9:13].values
    #     solution = row['solution']
    #     if solution not in choices:
    #         print("ERROR")

    df = pd.read_csv(files[0])

    # grouped = df.groupby('participant_fk')
    # dfs_subjs = {p:grouped.get_group(p) for idx,p in enumerate(df['participant_fk'].unique())}

    # accuracy_info = {}

    # for subj, df_subj in dfs_subjs.items():
    #     temp = pd.merge(df_subj, sequences, on='itemid')
    #     temp['correct'] = temp['solution'] == temp['response_img']
    #     accuracy_info[subj] = temp['correct'].mean()
    #     assert temp['correct'].mean() == temp['correct'].sum() / len(temp)
    #     # df_subj.column

    df = pd.merge(df, sequences, on="itemid")
    df["rt"] /= 1000
    df["correct"] = df["solution"] == df["response_img"]

    # * STAT ANALYSIS
    # * Convert 'correct' from boolean to int (True as 1 and False as 0)
    df["correct_numerical"] = df["correct"].astype(int)

    # * Calculate the correlation matrix for the numerical variables
    correlation_matrix = df[["correct_numerical", "rt"]].corr()
    display(correlation_matrix)

    # * Perform ANOVA to test differences in 'rt' among different 'pattern' categories
    model = ols("rt ~ C(pattern)", data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results)

    # * Create a contingency table of 'pattern' and 'correct'
    contingency_table = pd.crosstab(df["pattern"], df["correct"])

    # * Perform the chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2, p, dof, expected

    # * STAT ANALYSIS
    df_accuracy = df[["participant_fk", "correct", "pattern"]]
    df_accuracy = df_accuracy.groupby(["participant_fk", "pattern"]).mean()
    df_accuracy.reset_index(inplace=True)
    df_accuracy["correct"] *= 100

    df.groupby("participant_fk")["correct_numerical"].describe()
    df.groupby("pattern")["correct_numerical"].describe()

    df.groupby(["participant_fk", "pattern"])["correct"].mean().reset_index()

    df_subjs = df[["participant_fk", "correct", "pattern", "rt"]].copy()
    df_subjs_pattern = df_subjs.groupby(["participant_fk", "pattern"])["correct"]
    df_subjs_pattern = df_subjs_pattern.mean().reset_index()
    df_subjs_pattern["type"] = "Humans"
    df_subjs_pattern.rename(columns={"correct": "score"}, inplace=True)
    df_subjs_pattern

    cols = [
        "participant_fk",
        "correct",
        "rt",
        "solution",
        "pattern",
        "itemnr",
        "itemid",
    ]
    final_df = df[cols]
    seq_cols = [c for c in df.columns if re.search(r"figure\d{1,2}", c)]
    final_df["sequence"] = df[seq_cols].apply(lambda x: " ".join(x), axis=1)
    final_df.to_csv(export_dir / "raw_results.csv", index=False)
    # df[['participant_fk', 'correct', 'pattern', 'rt', 'itemid']].to_csv(wd/"corr_analysis.csv", index=False)

    # * PLOTS
    temp_kwargs = dict(
        show=False,
        to_pickle=True,
    )

    labelsize = 14
    titlesize = 20
    tick_labelsize = 12

    with plt.rc_context(
        {
            "axes.labelsize": labelsize,
            "axes.titlesize": titlesize,
            "ytick.labelsize": tick_labelsize,
            "xtick.labelsize": tick_labelsize,
        }
    ):
        fig1 = custom_plot(
            [sns.boxplot],
            plots_kwargs=[dict(data=df, y="rt", x="correct")],
            aes_kwargs=[
                dict(
                    set_title=dict(label="Response Time by Correctness"),
                    set_xlabel=dict(xlabel="Correctness"),
                    set_ylabel=dict(ylabel="Response Time (s)"),
                )
            ],
            save_path=export_dir / "rt_by_correctness-type[boxplot]",
            **temp_kwargs,
        )

        fig1_2 = custom_plot(
            [sns.histplot],
            plots_kwargs=[dict(data=df, x="rt", stat="percent", kde=True)],
            aes_kwargs=[
                dict(
                    set_title=dict(label="Response Time Distribution"),
                    set_xlabel=dict(xlabel="Response Time (s)"),
                    set_ylabel=dict(ylabel="Percent"),
                )
            ],
            save_path=export_dir / "rt_distribution-type[histplot]",
            **temp_kwargs,
        )

        fig1_3 = custom_plot(
            [sns.histplot],
            plots_kwargs=[
                dict(data=df, x="rt", stat="percent", kde=True, hue="correct")
            ],
            aes_kwargs=[
                dict(
                    set_title=dict(label="Response Time Distribution"),
                    set_xlabel=dict(xlabel="Response Time (s)"),
                    set_ylabel=dict(ylabel="Percent"),
                    legend=dict(labels=["True", "False"], title="Correct"),
                )
            ],
            save_path=export_dir / "rt_distribution_by_correctness-type[histplot]",
            **temp_kwargs,
        )

        fig2 = custom_plot(
            [sns.barplot],
            plots_kwargs=[
                dict(data=df.sort_values("pattern"), x="correct", y="rt", hue="correct")
            ],
            aes_kwargs=[
                dict(
                    set_title=dict(label="Response Time by Correctness"),
                    legend=dict(labels=[], frameon=False),
                    set_ylabel="Response Time (s)",
                    set_xlabel="Correct",
                )
            ],
            save_path=export_dir / "rt_by_correctness-type[barplot]",
            **temp_kwargs,
        )

        fig3 = custom_plot(
            [sns.barplot],
            plots_kwargs=[
                dict(data=df.sort_values("pattern"), y="pattern", x="rt", hue="correct")
            ],
            aes_kwargs=[
                dict(
                    set_title=dict(label=" Response Time by Pattern & Correctness"),
                    legend=dict(
                        title="Correct", bbox_to_anchor=(1, 1), loc="upper left"
                    ),
                    set_xlabel=dict(xlabel="Response Time (s)"),
                    set_ylabel="",
                )
            ],
            save_path=export_dir / "rt_by_pattern_and_correctness-type[barplot]",
            **temp_kwargs,
        )

        fig4 = custom_plot(
            [sns.barplot],
            plots_kwargs=[dict(data=df_accuracy, y="pattern", x="correct")],
            aes_kwargs=[
                dict(
                    set_title=dict(label="Accuracy by Pattern"),
                    set_xlim=(0, 100),
                    set_xlabel="Accuracy (%)",
                    set_ylabel="",
                )
            ],
            save_path=export_dir / "accuracy_by_pattern-type[barplot]",
            **temp_kwargs,
        )

    figures = {i: eval(f"fig{i}") for i in range(1, 5)}
    for fig_id, fig in figures.items():
        display(fig)


# ! PLAYGROUND
def analysis_online():
    color_palette = sns.color_palette()
    wd = Path(__file__).parent

    export_dir = wd / "local/results/analysis/group_lvl"

    files = [f for f in (wd / "online/results/2024-04-18").glob("*csv")]
    demographics = pd.read_csv(files[2])
    demographics.columns = demographics.columns.str.lower()
    demographics.age.describe()
    demographics.sex.value_counts()
    sequences = pd.read_csv(files[1])

    # for idx, row in sequences.iterrows():
    #     seq = row.iloc[1:9].values
    #     choices = row.iloc[9:13].values
    #     solution = row['solution']
    #     if solution not in choices:
    #         print("ERROR")

    df = pd.read_csv(files[0])
    # grouped = df.groupby('participant_fk')
    # dfs_subjs = {p:grouped.get_group(p) for idx,p in enumerate(df['participant_fk'].unique())}

    # accuracy_info = {}

    # for subj, df_subj in dfs_subjs.items():
    #     temp = pd.merge(df_subj, sequences, on='itemid')
    #     temp['correct'] = temp['solution'] == temp['response_img']
    #     accuracy_info[subj] = temp['correct'].mean()
    #     assert temp['correct'].mean() == temp['correct'].sum() / len(temp)
    #     # df_subj.column

    df = pd.merge(df, sequences, on="itemid")
    df["correct"] = df["solution"] == df["response_img"]

    df_accuracy = pd.DataFrame()

    df.groupby("participant_fk")["correct"].mean()
    df.groupby("pattern")["correct"].mean()
    df.groupby(["participant_fk", "pattern"])["correct"].mean().reset_index()

    ann_results = [f for f in (wd / "ANNs/results/analysis").glob("*.csv")]

    ann_results = {f.stem: pd.read_csv(f) for f in ann_results}
    ann_results.keys()

    ann_patterns = ann_results["results_by_pattern"]
    seq_per_pattern = 6
    accuracy = ann_patterns.groupby(["model"])["score"].mean() / seq_per_pattern
    top_models = accuracy[accuracy >= 0.25].sort_values(ascending=False).index
    top_models_res = pd.DataFrame(accuracy[top_models])
    top_models_res["type"] = "LLMs"
    top_models_res.reset_index(inplace=True)

    accuracy_by_pattern = (
        ann_patterns.groupby(["model", "pattern"])["score"].mean() / seq_per_pattern
    ).reset_index()
    top_models_res_pattern = accuracy_by_pattern.query("model in @top_models")
    top_models_res_pattern["type"] = "LLMs"

    df_subjs = df[["participant_fk", "correct", "pattern", "rt"]].copy()
    df_subjs_pattern = df_subjs.groupby(["participant_fk", "pattern"])["correct"]
    df_subjs_pattern = df_subjs_pattern.mean().reset_index()
    df_subjs_pattern["type"] = "Humans"
    df_subjs_pattern.rename(columns={"correct": "score"}, inplace=True)
    df_subjs_pattern
    # df_subjs_pattern = df_subjs.groupby(["pattern"])["correct"].mean().reset_index()
    # df_subjs_pattern.rename(columns={"correct": "score"}, inplace=True)
    # df_subjs_pattern['type'] = "Humans"

    # ! DATA COMBINED
    data = pd.concat([df_subjs_pattern, top_models_res_pattern], ignore_index=True)
    data = data.query("pattern not in ['ABCDEFAB','ABCDEFGH']")
    data["score"] *= 100
    data = data.sort_values("pattern", ascending=True)
    data = data[["type", "pattern", "score", "participant_fk", "model"]]
    data.query("not model.isna()").groupby("model")["score"].mean().sort_values(
        ascending=False
    )
    data.query("type=='Humans'")["score"].mean()

    data.query("model.isna() | model in @top_models").groupby(["pattern", "type"])[
        "score"
    ].mean()
    # ! DATA COMBINED

    # data.to_csv(f"results_analyzed.csv", index=False)
    # pd.read_csv("results_analyzed.csv")
    data.to_excel(f"results_analyzed.xlsx", index=False)
    # pd.read_excel("results_analyzed.xlsx")

    global_accuracy1 = (
        data.groupby(["type", "participant_fk"])["score"].mean().reset_index()
    )
    global_accuracy2 = data.groupby(["type", "model"])["score"].mean().reset_index()
    global_accuracy = pd.concat([global_accuracy1, global_accuracy2], ignore_index=True)

    data.groupby("participant_fk")["score"].describe()

    # * TOP AND BOTTOM PATTERNS
    overall_pattern_performance = data.groupby("pattern")["score"].mean()
    top_three_patterns = overall_pattern_performance.nlargest(3).index.tolist()
    bottom_three_patterns = overall_pattern_performance.nsmallest(3).index.tolist()

    llm_pattern_performance = (
        data.query("type=='LLMs'").groupby("pattern")["score"].mean()
    )
    top_three_patterns_llm = llm_pattern_performance.nlargest(3)
    bottom_three_patterns_llm = llm_pattern_performance.nsmallest(3)

    human_pattern_performance = (
        data.query("type=='Humans'").groupby("pattern")["score"].mean()
    )
    top_three_patterns_human = human_pattern_performance.nlargest(3).index.tolist()
    bottom_three_patterns_human = human_pattern_performance.nsmallest(3).index.tolist()

    top3_pat = overall_pattern_performance.head(3).index.tolist()
    bottom3_pat = overall_pattern_performance.tail(3).index.tolist()

    param = [
        ["top3", "Humans"],
        ["bottom3", "Humans"],
        ["top3", "LLMs"],
        ["bottom3", "LLMs"],
    ]
    df_top_bot = pd.DataFrame()
    for p in param:
        temp = data.query(f"pattern in @{p[0]}_pat & type=='{p[1]}'").copy()
        temp["seq_type"] = p[0]
        df_top_bot = pd.concat([df_top_bot, temp])
    # df_top_bot.describe()
    df_top_bot.groupby(["seq_type", "type"])["score"].mean()

    # * ############ FIG 0 ############
    d = data.copy().query(
        "type=='LLMs'"
    )  # .groupby("model")['score'].mean().reset_index()
    d.sort_values("score", ascending=False, inplace=True)
    d["model"] = [m.split("_")[1] if "_" in m else m for m in d["model"]]
    d["model"] = d["model"].str.capitalize()
    d["model"] = d["model"].replace({"Claude": "Claude 3 Sonnet", "Chatgpt": "GPT-4"})
    human_data = data.query("type=='Humans'")
    human_data["model"] = "Humans"
    d = pd.concat([d, human_data], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.barplot(
        data=d,
        x="score",
        y="model",
        ax=ax,
        order=d.groupby("model")["score"].mean().sort_values(ascending=False).index,
    )
    title = "Average Humans and LLMs Performance"
    ax.set_title(title)
    bars = ax.patches
    bars[0].set_facecolor(color_palette[1])
    [bar.set_facecolor(color_palette[-1]) for bar in bars[1:]]
    ax.set_ylabel("")
    ax.set_xticks(np.arange(0, 101, 10), minor=True)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="x", linestyle="--")
    # ax.axvline(d["score"].mean(), color=color_palette[0], linestyle="--")
    ax.axvline(25, color=color_palette[3], linestyle="--")
    ax.set_xlabel("Accuracy (%)")
    plt.savefig(
        export_dir / f"{title.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # * ############ FIG 1 ############
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[16, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    sns.barplot(data=data, x="score", y="pattern", hue="type", ax=ax0)  # errorbar=None)
    title = "Top Performing Models vs. Humans"
    ax0.set_title(title)
    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax0.set_xticks(np.arange(0, 100, 20))
    ax0.set_xticks(np.arange(10, 100, 20), minor=True)
    ax0.grid(axis="x", linestyle="--")
    ax0.set_xticklabels([])
    # ax0.set_xlabel("Accuracy (%)")
    ax0.set_xlabel("")
    ax0.set_xlim(0, 100)
    ax1 = sns.barplot(data=global_accuracy, x="score", hue="type", ax=ax1)
    # ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax1.get_legend().remove()
    # ax1.set_ylabel("Overall")
    ax1.set_yticklabels(["Overall"])
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)

    data.groupby("type")["score"].describe()
    data.groupby(["pattern", "type"])["score"].describe()
    data.groupby(["type", "pattern"])["score"].describe()

    data_claude_gpt = data.query("model.isna() | model in ['claude', 'chatgpt']")
    global_accuracy1 = (
        data_claude_gpt.groupby(["type", "participant_fk"])["score"]
        .mean()
        .reset_index()
    )
    global_accuracy2 = (
        data_claude_gpt.groupby(["type", "model"])["score"].mean().reset_index()
    )
    global_accuracy = pd.concat([global_accuracy1, global_accuracy2], ignore_index=True)

    # * ############ FIG 2 ############
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[16, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    sns.barplot(data=data_claude_gpt, x="score", y="pattern", hue="type", ax=ax0)
    title = "Claude & ChatGPT vs. Humans"
    ax0.set_title(title)
    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax0.set_xticks(np.arange(0, 100, 20))
    ax0.set_xticks(np.arange(10, 100, 20), minor=True)
    ax0.grid(axis="x", linestyle="--")
    ax0.set_xticklabels([])
    # ax0.set_xlabel("Accuracy (%)")
    ax0.set_xlabel("")
    ax0.set_xlim(0, 100)
    ax1 = sns.barplot(data=global_accuracy, x="score", hue="type", ax=ax1)
    # ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax1.get_legend().remove()
    # ax1.set_ylabel("Overall")
    ax1.set_yticklabels(["Overall"])
    ax1.set_xticks(np.arange(0, 100, 20))
    ax1.set_xticks(np.arange(10, 100, 20), minor=True)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)

    # * ############ FIG 3 ############
    df_subjs["rt"] /= 1000
    sns.boxplot(df_subjs, x="rt", y="pattern")

    # * ############ FIG 4 ############
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[16, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    sns.barplot(data=data, x="score", y="pattern", hue="type", ax=ax0)  # errorbar=None)
    title = "Top Performing Models vs. Humans"
    ax0.set_title(title)
    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax0.set_xticks(np.arange(0, 100, 20))
    ax0.set_xticks(np.arange(10, 100, 20), minor=True)
    ax0.grid(axis="x", linestyle="--")
    ax0.set_xticklabels([])
    # ax0.set_xlabel("Accuracy (%)")
    ax0.set_xlabel("")
    ax0.set_xlim(0, 100)
    ax1 = sns.barplot(data=global_accuracy, x="score", hue="type", ax=ax1)
    # ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax1.get_legend().remove()
    # ax1.set_ylabel("Overall")
    ax1.set_yticklabels(["Overall"])
    ax1.set_xticks(np.arange(0, 100, 20))
    ax1.set_xticks(np.arange(10, 100, 20), minor=True)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)

    data.groupby("type")["score"].describe()
    data.groupby(["pattern", "type"])["score"].describe()
    data.groupby(["type", "pattern"])["score"].describe()

    data_claude_gpt = data.query("model.isna() | model in ['claude', 'chatgpt']")
    global_accuracy1 = (
        data_claude_gpt.groupby(["type", "participant_fk"])["score"]
        .mean()
        .reset_index()
    )
    global_accuracy2 = (
        data_claude_gpt.groupby(["type", "model"])["score"].mean().reset_index()
    )
    global_accuracy = pd.concat([global_accuracy1, global_accuracy2], ignore_index=True)

    # * ############ FIG 5 ############
    # data =
    data = data.query("pattern not in ['ABCDEFAB','ABCDEFGH']")
    data["score"] *= 100
    data = data.sort_values("pattern", ascending=True)
    global_accuracy = data.groupby(["type", "model"])["score"].mean().reset_index()

    sns.barplot(data=data, x="score", y="pattern")
    data = data.groupby("model")["score"].mean().reset_index()
    sns.barplot()

    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[16, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    sns.barplot(data=data, x="score", y="pattern", hue="type", ax=ax0)  # errorbar=None)
    title = "Top Performing Models vs. Humans"
    ax0.set_title(title)
    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax0.set_xticks(np.arange(0, 100, 20))
    ax0.set_xticks(np.arange(10, 100, 20), minor=True)
    ax0.grid(axis="x", linestyle="--")
    ax0.set_xticklabels([])
    # ax0.set_xlabel("Accuracy (%)")
    ax0.set_xlabel("")
    ax0.set_xlim(0, 100)
    ax1 = sns.barplot(data=global_accuracy, x="score", hue="type", ax=ax1)
    # ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax1.get_legend().remove()
    # ax1.set_ylabel("Overall")
    ax1.set_yticklabels(["Overall"])
    ax1.set_xticks(np.arange(0, 100, 20))
    ax1.set_xticks(np.arange(10, 100, 20), minor=True)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--")
    plt.tight_layout()
    plt.savefig(export_dir / f"{title.lower().replace(' ', '_')}.png", dpi=300)

def read_eyetracking_data():
    import mne
    from mne.datasets.eyelink import data_path
    from mne.preprocessing.eyetracking import read_eyelink_calibration
    from mne.viz.eyetracking import plot_gaze
    import numpy as np

    et_fpath = r"C:\Users\topuser\Documents\ChrisPinier\experiment1\local\cp200.asc"
    # eeg_fpath = data_path() / "eeg-et" / "sub-01_task-plr_eeg.mff"

    raw_et = mne.io.read_raw_eyelink(et_fpath)  # , create_annotations=["blinks"])
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
