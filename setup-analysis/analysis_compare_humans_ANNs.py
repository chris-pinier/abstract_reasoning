from pathlib import Path
import os
import pandas as pd
import pickle
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import (
    calc_rdm,
    compare_cosine,
    compare as compare_rdms,
    combine as combine_rdms,
)
from analysis_plotting import plot_matrix
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import numpy.typing as npt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from itertools import permutations
import seaborn as sns
import dataframe_image as dfi
from analysis_utils import save_pickle, load_pickle, apply_df_style, read_file
from transformers import AutoTokenizer
from analysis_ANNs import (
    clean_and_eval_model_responses,
    load_model_responses,
    performance_analysis_all,
)
from analysis_lab import perf_analysis_all_subj

WD = Path(__file__).parent
# os.chdir(WD)
assert WD == Path.cwd()

SSD_PATH = Path("/Volumes/Realtek 1Tb")
DATA_DIR = SSD_PATH / "PhD Data/experiment1/data/"

ANN_DIR = WD.parent / "experiment-ANNs"

SEQ_DIR = WD.parent / "config/sequences"

EXPORT_DIR = SSD_PATH / "PhD Data/experiment1-analysis/"

if not SSD_PATH.exists():
    print("WARNING: SSD not connected")
else:
    RDM_DIR = EXPORT_DIR / "RDMs"
    RDM_DIR.mkdir(parents=True, exist_ok=True)

with open(ANN_DIR / "config/instructions.txt", "r") as f:
    instructions = f.read()


PATTERNS: Final = sorted(
    [
        "AAABAAAB",
        "ABABCDCD",
        "ABBAABBA",
        "ABBACDDC",
        "ABBCABBC",
        "ABCAABCA",
        "ABCDDCBA",
        "ABCDEEDC",
    ]
)


def rsa_between_humans_and_ANNs(humans_rdms, ANNs_rdms, similarity_metric: str):
    # ! TEMP
    # similarity_metric = "cosine"
    similarity_metric = "corr"
    # ! TEMP

    # *  -------- Load ANNs RDMs --------
    level = "pattern"
    
    rdms_dir = EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation"
    ANNs_rdm_files = sorted(rdms_dir.glob(f"*{level}*hdf5"))
    ANNs_rdm_files = [f for f in ANNs_rdm_files if not f.name.startswith(".")]

    ANNs_ids = [
        re.findall(r"\((.+?)\)", f.stem)[0].replace("--", "/") for f in ANNs_rdm_files
    ]

    ANNs_rdms = {
        ann_id: rsatoolbox.rdm.rdms.load_rdm(str(f))
        for ann_id, f in zip(ANNs_ids, ANNs_rdm_files)
    }

    # * -------- Load Humans RDMs --------
    rdms_dir = EXPORT_DIR / "Lab/analyzed/RSA-FRP-frontal-metric_correlation"

    humans_rdm_files = sorted(
        [f for f in rdms_dir.glob(f"*subj*{level}*.hdf5") if not f.name.startswith(".")]
    )
    subj_ids = [re.findall(r"subj_\d{2}", f.stem)[0] for f in humans_rdm_files]

    humans_rdms = {
        subj_id: rsatoolbox.rdm.rdms.load_rdm(str(f))
        for subj_id, f in zip(subj_ids, humans_rdm_files)
    }

    humans_group_avg_rdm_file = next(rdms_dir.glob(f"*group_avg*{level}*.hdf5"))
    humans_group_avg_rdm = rsatoolbox.rdm.rdms.load_rdm(str(humans_group_avg_rdm_file))
    humans_rdms["group_avg"] = humans_group_avg_rdm

    if level == "item_lvl":
        humans_rdms = {subj: rdm for subj, rdm in humans_rdms.items() if rdm.n_cond == 400}
    # ! TEMP

    #  * -------- Compare RDMs --------
    results = {}
    for model_id, model_rdm in ANNs_rdms.items():
        results[model_id] = {}
        for subj_id, subj_rdm in humans_rdms.items():
            results[model_id][subj_id] = compare_rdms(
                model_rdm, subj_rdm, similarity_metric
            ).item()

    df_comparison = pd.DataFrame(results)
    df_comparison.loc["mean", :] = df_comparison.mean(axis=0)

    df_comparison.loc["mean"].sort_values(ascending=False)

    # * -------- Compare ANN RDMs with each other --------
    results = {}
    for model_id_1, model_rdm_1 in ANNs_rdms.items():
        results[model_id_1] = {}
        for model_id_2, model_rdm_2 in ANNs_rdms.items():
            results[model_id_1][model_id_2] = compare_rdms(
                model_rdm_1, model_rdm_2, similarity_metric
            ).item()

    df_comparison = pd.DataFrame(results)
    df_comparison.loc["mean", :] = df_comparison.mean(axis=0)

    df_comparison.loc["mean"].sort_values(ascending=False).plot(kind="barh")

    temp = (
        df_comparison.loc["mean"]
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "model_id", "mean": "similarity"})
    )
    
    dfi.export(
        temp,
        EXPORT_DIR / "RSA-ANN-Human_Avg.png",
        table_conversion="matplotlib",
    )


    # * -------- Compare Subject RDMs with each other --------
    results = {}
    for subj_id_1, subj_rdm_1 in humans_rdms.items():
        results[subj_id_1] = {}
        for subj_id_2, subj_rdm_2 in humans_rdms.items():
            results[subj_id_1][subj_id_2] = compare_rdms(
                subj_rdm_1, subj_rdm_2, similarity_metric
            ).item()
    df_comparison = pd.DataFrame(results)
    df_comparison.loc["mean", :] = df_comparison.mean(axis=0)

    temp = df_comparison.loc["mean"].sort_values(ascending=False).reset_index().rename(
        columns={"index": "subj_id", "mean": "similarity"}
    )
    dfi.export(
        temp,
        EXPORT_DIR / "RSA-Human-Human.png",
        table_conversion="matplotlib",
    )   

    # * -------- Get RDMs of Patterns -------

    pattern_inds = np.arange(0, 401, 50)
    patterns_inds = np.array(list(zip(pattern_inds, pattern_inds[1:])))

    rdm = humans_group_avg_rdm.get_matrices()[0]

    _pattern_rdm_data = []
    for i, j in patterns_inds:
        sub_rdm = rdm[i:j, i:j]
        triu_inds = np.triu_indices_from(sub_rdm)
        sub_rdm_mean = sub_rdm[triu_inds[0]].mean()
        _pattern_rdm_data.append(sub_rdm_mean)

    # _pattern_rdm_data = [pattern_rdm_data[i:j, i:j].mean() for i, j in patterns_inds]
    pattern_rdm_data = np.array(_pattern_rdm_data)[:, None]
    del _pattern_rdm_data

    dataset = Dataset(
        measurements=pattern_rdm_data,
        # descriptors={"subj_N": subj_N},
        # obs_descriptors={"patterns": subj_patterns},
    )

    rdm_pattern_lvl = calc_rdm(dataset, method="euclidean")
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(rdm_pattern_lvl.get_matrices()[0])
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(0, 8))
    ax.set_xticklabels(PATTERNS, rotation=70)
    ax.set_yticks(np.arange(0, 8))
    ax.set_yticklabels(PATTERNS)
    plt.show()


def compare_performance(ANN_data_dir: Path, human_data_dir: Path):
    # # ! TEMP
    # ANN_data_dir = DATA_DIR / "ANNs/local_run/sessions-1_to_5-masked_idx(7)"
    # human_data_dir = DATA_DIR / "Lab"
    # # ! TEMP

    # * Human data
    human_data = behav_analysis_all(data_dir=human_data_dir)
    human_data = human_data["raw_cleaned"]
    human_data["type"] = "human"
    human_data = human_data.rename(columns={"subj_N": "id"})

    # * ANN data
    ann_data = performance_analysis_all(
        data_dir=ANN_data_dir, answer_regex=r"Answer:\s?\n?(\w+)"
    )
    ann_data["type"] = "ANN"
    ann_data = ann_data.rename(columns={"model_id": "id"})

    # * Combined data
    combined_data = pd.concat([human_data, ann_data], axis=0).reset_index(drop=True)
    combined_data["id"] = combined_data["id"].astype(str)

    # * Human performance analysis
    average_human_acc = (
        human_data.groupby("id")["correct"].mean().sort_values(ascending=False)
    )
    best_humans = [str(i) for i in average_human_acc[average_human_acc > 0.7].index]
    worst_humans = [str(i) for i in average_human_acc[average_human_acc < 0.7].index]

    # * ANN performance analysis
    average_ann_acc = (
        ann_data.groupby("id")["correct"].mean().sort_values(ascending=False)
    )
    best_anns = average_ann_acc[average_ann_acc > 0.7].index.to_list()
    worst_anns = average_ann_acc[average_ann_acc < 0.7].index.to_list()

    # * Compare performance
    human_acc_by_pattern = pd.DataFrame(
        human_data.query(f"id in {best_humans}").groupby("pattern")["correct"].mean()
    ).reset_index()

    human_acc_by_pattern["type"] = "human"

    ann_acc_by_pattern = pd.DataFrame(
        ann_data.query(f"id in {best_anns}").groupby("pattern")["correct"].mean()
    ).reset_index()

    ann_acc_by_pattern["type"] = "ANN"

    # df_acc = pd.concat([human_acc_by_pattern, ann_acc_by_pattern], axis=0).reset_index(
    #     drop=True
    # )
    # df_acc = df_acc.rename(columns={"correct": "accuracy"})
    # df_acc["accuracy"] = round(df_acc["accuracy"] * 100, 2)
    # df_acc.pivot_table(values="accuracy", index="pattern", columns="type").corr()

    corr_by_item_id = combined_data.pivot_table(
        values="correct", index="item_id", columns="type"
    ).corr()

    corr_by_item_id_best_performers = (
        combined_data.query(f"id in {best_humans + best_anns}")
        .pivot_table(values="correct", index="item_id", columns="type")
        .corr()
    )

    corr_by_pattern = combined_data.pivot_table(
        values="correct", index="pattern", columns="type"
    ).corr()

    corr_by_pattern_best_performers = (
        combined_data.query(f"id in {best_humans + best_anns}")
        .pivot_table(values="correct", index="pattern", columns="type")
        .corr()
    )

    # * ------ Figures -------
    hue_order = ["human", "ANN"]
    fig, ax = plt.subplots(dpi=200)
    sns.lineplot(
        data=combined_data.query(f"id in {best_humans + worst_humans + best_anns}"),
        # data=combined_data.query(f"id in {best_humans + best_anns}"),
        # data=combined_data,
        x="pattern",
        y="correct",
        hue="type",
        ax=ax,
        hue_order=hue_order,
        marker="o",
        errorbar="ci",
    )

    line_colors = [ax.lines[i].get_c() for i in range(len(hue_order))]
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    ax.set_title("Performance by pattern")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Pattern")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

    acc_combined = pd.merge(
        combined_data.groupby(["id", "pattern"]).correct.mean().reset_index(),
        combined_data.groupby(["id"])["type"].first().reset_index(),
        on="id",
        how="left",
    ).rename(columns={"correct": "accuracy"})

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.violinplot(
        acc_combined,
        hue="type",
        x="pattern",
        y="accuracy",
        ax=ax,
        split=True,
        inner="quartile",
        linewidth=0.5,  # palette=line_colors
    )
    ax.legend(bbox_to_anchor=(1, 1))
    ax.tick_params(axis="x", rotation=45)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.stripplot(
        acc_combined,
        hue="type",
        x="pattern",
        y="accuracy",
        jitter=0.1,
        dodge=True,
        ax=ax,
    )
    ax.legend(bbox_to_anchor=(1, 1))
    ax.tick_params(axis="x", rotation=45)


def filter_rdms(rdms: list):
    pass


ann_rdms_dir = EXPORT_DIR / "ANNs/RDMs"
ann_rdm_files = [
    f for f in ann_rdms_dir.glob("*layers*item_lvl*.hdf5") if not f.name.startswith(".")
]
ann_rdms = [rsatoolbox.rdm.rdms.load_rdm(str(f)) for f in ann_rdm_files]

human_rdms_dir = EXPORT_DIR / "Lab/analyzed/RSA-FRP-occipital-metric_correlation"
human_rdm_files = [
    f for f in human_rdms_dir.glob("*subj*item_lvl.hdf5") if not f.name.startswith(".")
]
human_rdms = [rsatoolbox.rdm.rdms.load_rdm(str(f)) for f in human_rdm_files]

complete_human_rdms = [rdm for rdm in human_rdms if rdm.n_cond == 400]
human_grp_lvl_rdm = np.array([rdm.get_matrices()[0] for rdm in complete_human_rdms])
human_grp_lvl_rdm = human_grp_lvl_rdm.mean(axis=0)[None, :, :]


human_grp_lvl_rdm = rsatoolbox.rdm.rdms.RDMs(
    dissimilarities=human_grp_lvl_rdm,
    dissimilarity_measure=complete_human_rdms[0].dissimilarity_measure,
    # pattern_descriptors={"patterns": list(range(400))},
    pattern_descriptors=complete_human_rdms[0].pattern_descriptors,
    # rdm_descriptors=[rdm.rdm_descriptors for rdm in human_rdms if rdm.n_cond == 400],
)

human_grp_lvl_rdm.get_matrices()[0].max()

fig, ax = plt.subplots(dpi=300)
im = plt.imshow(human_grp_lvl_rdm.get_matrices()[0])
fig.colorbar(im, ax=ax)

[
    compare_rdms(ann_rdm, complete_human_rdms[0], method="kendall").item()
    for ann_rdm in ann_rdms
]

[compare_rdms(ann_rdms[0], rdm, "cosine") for rdm in ann_rdms]
[compare_rdms(complete_human_rdms[0], rdm, "cosine") for rdm in complete_human_rdms]
