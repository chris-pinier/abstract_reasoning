from box import Box
from pathlib import Path
import os
import pandas as pd
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare as compare_rdms
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import seaborn as sns
import dataframe_image as dfi

WD = Path(__file__).parent
os.chdir(WD)
assert WD == Path.cwd()
# *
from draft.analysis_lab_OLD import perf_analysis_all_subj
from analysis_plotting import plot_rdm, plot_matrix
from utils.analysis_utils import (
    save_pickle,
    load_pickle,
    apply_df_style,
    get_timestamp,
    clean_ann_id,
    clean_filename,
)
from analysis_ANNs import perf_analysis_all_anns
from analysis_rsa import (
    get_reference_rdms,
    get_ds_and_rdm,
    match_datasets_on_nan,
    match_datasets_on_descriptor,
    simple_rsa,
    rsa_bootstrap,
    rsa_permutation,
)

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

ANN_ID_ORDER: Final = [
    "Phi-4",
    "Gemma-2-2B",
    "Gemma-2-9B",
    "Gemma-2-27B",
    "Llama-3.2-3B",
    "Llama-3.3-70B",
    "Qwen2.5-72B",
    "Deepseek-R1-Distill-Llama-70B",
]


# * ####################################################################################
# * PERFORMANCE ANALYSIS
# * ####################################################################################
def prepare_ann_perf_data(
    ANN_data_dir: Path, seq_file: Path, ann_id_order: Optional[List[str]] = None
):
    sequences = pd.read_csv(seq_file)

    ann_data = perf_analysis_all_anns(
        data_dir=ANN_data_dir, answer_regex=r"Answer:\s?\n?(\w+)"
    )
    ann_data: pd.DataFrame = ann_data["raw_cleaned"]
    ann_data["type"] = "ANN"
    ann_data.rename(columns={"model_id": "id"}, inplace=True)

    # * Add sequence info to ANN data
    cols_to_use = list(sequences.columns.difference(ann_data.columns)) + ["item_id"]
    ann_data = ann_data.merge(sequences[cols_to_use], on="item_id", how="left")

    ann_data["choice"] = pd.NA

    for i, row in ann_data.iterrows():
        choices = row[["choice1", "choice2", "choice3", "choice4"]].values
        if row["cleaned_response"] in choices:
            ann_data.loc[i, "choice"] = row["cleaned_response"]
        else:
            ann_data.loc[i, "choice"] = "invalid"

    if ann_id_order is None:
        ann_id_order = sorted(ann_data["id"].unique())

    # * Reorder columns
    ann_data = ann_data[
        [
            "id",
            "type",
            "item_id",
            "pattern",
            "masked_idx",
            "prompt",
            "solution",
            "response",
            "cleaned_response",
            "choice",
            "correct",
            "choice1",
            "choice2",
            "choice3",
            "choice4",
            "choice_order",
            "figure1",
            "figure2",
            "figure3",
            "figure4",
            "figure5",
            "figure6",
            "figure7",
            "figure8",
            "seq_order",
            "sess_N",
            "trial_type",
        ]
    ]

    return ann_data


def prepare_human_perf_data(human_data_dir: Path):
    human_data = perf_analysis_all_subj(data_dir=human_data_dir)

    human_data: pd.DataFrame = human_data["raw_cleaned"]
    human_data["type"] = "human"
    human_data.rename(columns={"subj_N": "id"}, inplace=True)
    human_data["id"] = human_data["id"].astype(str)

    return human_data


def error_analysis(combined_data: pd.DataFrame, sequences: pd.DataFrame):
    # * ----------------------------------------
    # * ----------------------------------------
    temp = (
        combined_data.query('correct == 0 & not choice=="invalid"')
        .groupby("item_id")["choice"]
        .value_counts()
        .reset_index()
        .sort_values("count", ascending=False)
        .merge(
            combined_data[["item_id", "pattern"]].drop_duplicates(),
            on="item_id",
            how="left",
        )
    )

    temp[temp["count"] > 1].head(20)
    temp["pattern"].value_counts() / len(temp)

    ids = combined_data["id"].unique()
    cols = ["id", "item_id", "choice", "correct"]
    # df_res = pd.DataFrame(columns='id1')
    res = np.zeros((len(ids), len(ids)))
    res[:] = np.nan

    for i, id1 in enumerate(ids):
        # id1, id2 = '1', '1'
        df1 = combined_data.query("id == @id1")[cols]
        for j, id2 in enumerate(ids):
            df2 = combined_data.query("id == @id2")[cols]
            # common = df1.merge(df2, on="item_id", how="inner")
            # common["same_err"] = common["choice_x"] == common["choice_y"]
            # res[i, j] = common["same_err"].mean()
            common = df1.merge(df2, on="item_id", how="inner")
            if len(common) == 0 or id1 == id2:
                continue
            common["same_resp"] = common["choice_x"] == common["choice_y"]
            common_err = len(
                common.query("correct_x==0 & correct_y == 0 & same_resp")
            ) / len(common)
            res[i, j] = common_err

    res_df = pd.DataFrame(res, columns=ids, index=ids)
    res_df.describe().T.sort_values("mean")

    fig, ax = plt.subplots()
    # sns.heatmap(res, annot=True, fmt=".2f", ax=ax, cmap="coolwarm", xticklabels=ids, yticklabels=ids)
    heatmap = ax.imshow(res_df)  # , aspect="auto")
    ax.set_xticks(range(len(ids)), ids, rotation=90)
    ax.set_yticks(range(len(ids)), ids)
    plt.colorbar(heatmap, ax=ax, label="Same Error Rate")
    plt.show()

    # grouped = combined_data.query("id in ['1', '20']").groupby("id")
    # df1, df2 = [grouped.get_group(id) for id in grouped.groups.keys()]
    # common = df1.merge(df2, on="item_id", how="inner")
    # common["same_resp"] = common["choice_x"] == common["choice_y"]
    # common["same_resp"].mean()
    # common_err = len(common.query("correct_x==0 & correct_y == 0 & same_resp")) / len(
    #     common
    # )
    # * ----------------------------------------
    # * ----------------------------------------

    # ann_data.groupby(["item_id", "id"])["correct"].mean().sort_values()
    # human_data.groupby(["item_id"])["correct"].mean().sort_values()

    # d1 = human_data.groupby(["item_id"])["correct"].mean().sort_index()
    # accuracy_corr_item_lvl = []

    # for ann in ann_data["id"].unique():
    #     d2 = (
    #         ann_data.query("id==@ann")
    #         .groupby(["item_id"])["correct"]
    #         .mean()
    #         .sort_index()
    #     )
    #     # print(f"{ann:<50}: {d1.corr(d2)}")
    #     accuracy_corr_item_lvl.append([ann, d1.corr(d2)])

    # accuracy_corr_item_lvl = pd.DataFrame(
    #     accuracy_corr_item_lvl, columns=["id", "corr"]
    # )
    # accuracy_corr_item_lvl.sort_values("corr")

    # * ----------------------------------------
    # d1 = human_data.groupby(["item_id"])["correct"].mean().nsmallest(20)
    # d2 = ann_data.groupby(["item_id"])["correct"].mean().nsmallest(20)

    # common_errors = list(set(d1.index).intersection(set(d2.index)))
    # sequences.query("item_id in @common_errors")

    # combined_data.query("correct == 0").groupby(["item_id", "type"])[
    #     "choice"
    # ].value_counts()

    def _compare_errors(df_query: str):
        vc1 = (
            combined_data.query("type == 'human' & correct == 0 & choice != 'invalid'")
            .groupby("item_id")["choice"]
            .value_counts()
        )

        vc2 = combined_data.query(df_query).groupby("item_id")["choice"].value_counts()

        # * For each trial, select the incorrect choice with highest number of occurences
        vc1 = vc1.groupby(level=0).head(1).reset_index()[["item_id", "choice"]]
        vc2 = vc2.groupby(level=0).head(1).reset_index()[["item_id", "choice"]]

        # * Select common trials
        common_item_ids = list(set(vc1.item_id).intersection(set(vc2.item_id)))
        vc1 = vc1.query(f"item_id in {common_item_ids}").sort_values("item_id")
        vc2 = vc2.query(f"item_id in {common_item_ids}").sort_values("item_id")

        # * Merge the ANN and Human dataframe
        errors = vc1.merge(vc2, on="item_id", suffixes=("_human", "_ANN"))
        errors["same"] = errors["choice_human"] == errors["choice_ANN"]

        # * filter out dissimilar choices
        common_errors = errors.query("same==True")

        # * Merge with Sequence Dataframe
        errors = errors.merge(sequences, on="item_id", how="left")
        common_errors = common_errors.merge(sequences, on="item_id", how="left")

        # * Select irrelevant choices (icons not present in the serquence)
        unmasked_seq_cols = [f"figure{i}" for i in range(1, 8)]

        irrelevant_choices_mask = ~common_errors[unmasked_seq_cols].eq(
            common_errors["choice_human"], axis=0
        ).any(axis=1)

        common_irrelevant_errors = common_errors[irrelevant_choices_mask]

        # * Get some stats on common errors:
        # ! line below only considers valid errors, not all incorrect errors, and also only
        # ! considers the erroneous choices with the highest occurences per trial
        pct_common_errors = errors["same"].mean()

        pct_common_errors_by_pat = (
            common_errors["pattern"].value_counts() / common_errors["pattern"].size
        )

        return errors, common_errors, pct_common_errors, pct_common_errors_by_pat

    template_idx = pd.Index(combined_data["pattern"].unique(), name="pattern")
    df_res_by_pat = pd.DataFrame(index=template_idx)

    df_query = "type == 'ANN' & correct == 0 & choice != 'invalid'"
    _, _, _, pct_common_errors_by_pat = _compare_errors(df_query)
    df_res_by_pat["all_ANNs"] = pct_common_errors_by_pat

    for ann_id in combined_data.query("type=='ANN'")["id"].unique():
        df_query = f"id == '{ann_id}' & correct == 0 & choice != 'invalid'"
        _, _, _, pct_common_errors_by_pat = _compare_errors(df_query)
        df_res_by_pat[ann_id] = pct_common_errors_by_pat.reindex(template_idx)

    # * reorder index
    df_res_by_pat = df_res_by_pat.sort_index()
    df_res_by_pat = df_res_by_pat.round(2)

    # sns.lineplot(data=df_res_by_pat)

    return df_res_by_pat

    # for i, row in common_errors[~mask].iterrows():
    #     print(f"choice = {row['choice_human']}")
    #     show_seq_img(row["item_id"])


def compare_performance(
    ANN_data_dir: Path,
    human_data_dir: Path,
    fig_params: Optional[Dict] = None,
    save_fig_params: Optional[Dict] = None,
    save_dir: Optional[Path] = None,
    ann_id_order: Optional[List[str]] = None,
):
    # # ! TEMP
    # ANN_data_dir = (
    #     DATA_DIR / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-all_tokens_acts"
    # )
    # ANN_data_dir = (
    #     DATA_DIR / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
    # )
    # human_data_dir = DATA_DIR / "Lab"
    # save_dir = EXPORT_DIR / "comparison/Performance"
    # fig_params = None
    # save_fig_params = None
    # ann_id_order=ANN_ID_ORDER
    # # ! TEMP

    # * ----------------------------------------
    # * Data Preparation
    # * ----------------------------------------

    seq_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
    sequences = pd.read_csv(seq_file)

    # * Human data
    human_data = prepare_human_perf_data(human_data_dir)

    # * ANN data
    ann_data = prepare_ann_perf_data(ANN_data_dir, seq_file, ann_id_order=None)

    # * Cleaning ANN IDs
    ann_data["id"] = ann_data["id"].apply(clean_ann_id)

    ann_data.query("choice=='invalid'")["id"].value_counts()
    ann_data.query("choice=='invalid' & id.str.contains('Deepseek')").iloc[-1].response

    # * Get unique IDs
    unique_human_ids = human_data["id"].unique()
    unique_ann_ids = ann_data["id"].unique()

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
    best_perf_thresh_ann = 0.6
    best_anns = average_ann_acc[average_ann_acc >= best_perf_thresh_ann].index.to_list()
    worst_anns = average_ann_acc[average_ann_acc < best_perf_thresh_ann].index.to_list()

    # * ------ Error Analysis ------
    error_analysis(combined_data, sequences)

    # * ------ Accuracy By Pattern ------
    # * Overall Human accuracy by pattern
    human_acc_pattern_group = (
        pd.DataFrame(human_data.groupby("pattern")["correct"].mean())
        .reset_index()
        .rename(columns={"correct": "accuracy"})
        .sort_values("pattern")
    )

    human_acc_pattern_group["type"] = "human"

    # * Individual Human accuracy by pattern
    human_acc_pattern_indiv = (
        pd.DataFrame(human_data.groupby(["id", "pattern"])["correct"].mean())
        .reset_index()
        .rename(columns={"correct": "accuracy"})
        .sort_values("pattern")
    )

    human_acc_pattern_indiv["type"] = "human"

    # * Overall ANN accuracy by pattern
    ann_acc_pattern_group = (
        pd.DataFrame(ann_data.groupby("pattern")["correct"].mean())
        .reset_index()
        .rename(columns={"correct": "accuracy"})
        .sort_values("pattern")
    )

    ann_acc_pattern_group["type"] = "ANN"

    # * individual ANN accuracy by pattern
    ann_acc_pattern_indiv = (
        pd.DataFrame(ann_data.groupby(["id", "pattern"])["correct"].mean())
        .reset_index()
        .rename(columns={"correct": "accuracy"})
        .sort_values("pattern")
    )
    ann_acc_pattern_indiv["type"] = "ANN"

    # * ------ Accuracy By item / sequence ------
    # * Overall ANN accuracy by item / sequence
    ann_acc_by_item_group = (
        combined_data.query("type=='ANN'")
        .groupby("item_id")["correct"]
        .mean()
        .rename("accuracy")
    )

    # * Overall Human accuracy by item / sequence
    human_acc_by_item_group = (
        combined_data.query("type=='human'")
        .groupby("item_id")["correct"]
        .mean()
        .rename("accuracy")
    )

    # * ----------------------------------------
    # * Analysis
    # * ----------------------------------------
    results = {}

    df_compare_patt = pd.DataFrame()

    for ann_id in unique_ann_ids:
        data_patt_ann = ann_acc_pattern_indiv.query("id == @ann_id").reset_index()

        # * Group-level comparison
        data_patt_human_group = human_acc_pattern_group

        res = data_patt_ann["accuracy"].corr(data_patt_human_group["accuracy"])

        df_temp = pd.DataFrame(
            [[ann_id, "group", res]], columns=["ann_id", "human_id", "corr"]
        )
        df_compare_patt = pd.concat([df_compare_patt, df_temp])

        # * Subjet-level comparison
        for human_id in unique_human_ids:
            # * ------ by Pattern------
            data_patt_human_indiv = human_acc_pattern_indiv.query(
                "id == @human_id"
            ).reset_index()

            res = data_patt_ann["accuracy"].corr(data_patt_human_indiv["accuracy"])

            df_temp = pd.DataFrame(
                [[ann_id, human_id, res]], columns=["ann_id", "human_id", "corr"]
            )
            df_compare_patt = pd.concat([df_compare_patt, df_temp])

    df_compare_patt = df_compare_patt.sort_values(["human_id", "ann_id"])

    df_compare_patt.groupby(["ann_id", "human_id"])[
        "corr"
    ].mean().reset_index().sort_values("corr", ascending=False)

    # * Saving results
    results["ANNs_vs_human_group_corr_pattern"] = {
        "res": df_compare_patt.query("human_id=='group'")
        .sort_values("corr", ascending=False)
        .reset_index(drop=True),
        "notes": "correlation in accuracy by pattern type between each ANN and the entire human group",
    }

    results["ANNs_vs_humans_corr_pattern"] = {
        "res": df_compare_patt.query("human_id != 'group'").reset_index(drop=True),
        "notes": "correlation in accuracy by pattern type between each ANN and each human subject",
    }

    results["ANN_overall_accuracy_and_human_similarity"] = {
        "res": (
            results["ANNs_vs_human_group_corr_pattern"]["res"]
            .merge(
                average_ann_acc.reset_index().rename(
                    columns={"id": "ann_id", "correct": "accuracy"}
                ),
                on="ann_id",
            )
            .sort_values("accuracy")
        ),
        "notes": "Overall ANN accuracy and correlation with humans",
    }

    # * ----------------------------------------
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

    # combined_data.pivot_table(values="correct", index="item_id", columns=["type", 'pattern'])
    corr_by_pattern_2 = pd.DataFrame()

    for patt in combined_data["pattern"].unique():
        _data = combined_data.query("pattern==@patt").pivot_table(
            values="correct", index="item_id", columns=["type"]
        )
        res = pd.DataFrame(
            [[patt, _data.corr().iloc[0, 1]]], columns=["pattern", "corr"]
        )
        corr_by_pattern_2 = pd.concat([corr_by_pattern_2, res])

    display(corr_by_pattern_2)

    corr_by_ANN = pd.DataFrame()
    for ann_id in unique_ann_ids:
        _data = combined_data.query("type=='human' | id ==@ann_id")
        _data = (
            _data.groupby(["type", "pattern"])["correct"]
            .mean()
            .reset_index()
            .pivot_table(values="correct", index="pattern", columns=["type"])
        )
        res = pd.DataFrame(
            [[ann_id, _data.corr().iloc[0, 1]]], columns=["ann_id", "corr"]
        )
        corr_by_ANN = pd.concat([corr_by_ANN, res])

    display(
        apply_df_style(
            corr_by_ANN.set_index("ann_id"),
            1,
        )
    )

    # * Adding pattern features
    combined_data["patt_unique_el_count"] = 0
    patt_info = {}
    for patt in list(combined_data.pattern.unique()):
        unique_els = list(set(patt))
        count_unique = len(unique_els)
        count_by_el = {el: patt.count(el) for el in unique_els}
        patt_info[patt] = {"count_unique": count_unique, "count_by_el": count_by_el}

    # * ------ Figures -------
    figs = {}

    if fig_params is None:
        fig_params = dict(figsize=(10, 6), dpi=300)

    if save_fig_params is None:
        save_fig_params = dict(dpi=300, bbox_inches="tight")

    # * ----- FIGURE: Average ANN Accuracy  ------
    # * ----- FIGURE:ANN Accuracy Correlation with Humans vs. ANN Overall Accuracy ------
    data = average_ann_acc.reset_index().rename(columns={"id": "ann_id"})
    data = data.merge(results["ANNs_vs_human_group_corr_pattern"]["res"], on="ann_id")
    data = data.rename(columns={"correct": "accuracy"})

    title = "ANN Accuracy Correlation with Humans vs. ANN Overall Accuracy"
    fig, ax = plt.subplots(**fig_params)
    sns.scatterplot(
        data=data.sort_values("ann_id"),
        x="accuracy",
        y="corr",
        hue="ann_id",
        hue_order=ann_id_order,
        marker="o",
        s=100,
        ax=ax,
    )
    ax.set_xlim(0, 1)
    if data["corr"].min() < 0:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(0, 1)
    # ax.scatter(data["accuracy"], data["corr"])
    ax.set_xlabel("Overall Accuracy")
    ax.set_ylabel("Correlation with Human Accuracy")
    # ax.set_title(title)
    ax.grid(axis="both", ls="--")
    ax.hlines(0.5, 0, 1, colors="black", ls="--", lw=0.9)
    ax.vlines(0.5, 0, 1, colors="black", ls="--", lw=0.9)
    # ax.legend(bbox_to_anchor=(1, 1), title="ANN ID")
    ax.legend(title="")
    plt.tight_layout()
    plt.show()
    figs[title + " scatter"] = fig
    # data.to_csv(clean_filename(save_dir / f"{title}.csv", "lower"), index=False)

    # * ----------- FIGURES: Indiviudal ANN Accuracy Correlation with Humans -----------
    for ann_id in unique_ann_ids:
        title = f"Accuracy Correlation with Humans \n{ann_id}"
        data = df_compare_patt.query("ann_id==@ann_id").reset_index()
        d1 = data.query("human_id != 'group'")["corr"]
        d2 = data.query("human_id == 'group'")["corr"]

        fig, ax = plt.subplots()
        ax.bar(d1.index, d1, color="tab:blue", label="individual subjects")
        ax.bar(d2.index, d2, color="tab:orange", label="group")
        ax.set_title(title)
        ax.set_ylim(-1, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(axis="y", ls="--")
        figs[title + " bar"] = fig
        plt.show()
        plt.close()

    data = df_compare_patt.query("human_id == 'group'")
    title = "Accuracy Correlation with Humans"
    fig, ax = plt.subplots()
    sns.barplot(
        data.sort_values("corr", ascending=False),
        x="corr",
        y="ann_id",
        hue="ann_id",
        hue_order=ann_id_order,
        ax=ax,
        # orient="h"
    )
    ax.grid(axis="x", ls="--")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params("x", rotation=90)
    plt.show()
    plt.close()
    figs[title + " bar"] = fig

    # * ----------- FIGURES: Accuracy RDM - All humans and ANNs -----------
    accuracy_arr = []
    ids = []
    types = []
    for id in combined_data["id"].unique():
        # for id in [str(i) for i in unique_human_ids] + ann_id_order:
        # for id in ann_id_order + list(unique_human_ids):
        _data = combined_data.query("id==@id")  # .sort_values(['pattern', 'item_id'])

        if _data.shape[0] < 400:
            continue

        types.append(_data["type"].iloc[0])
        ids.append(id)
        accuracy_arr.append(
            _data.groupby("pattern")["correct"].mean().sort_index().values
        )

    accuracy_arr = np.array(accuracy_arr)

    ds = Dataset(
        measurements=accuracy_arr, obs_descriptors={"ids": ids, "types": types}
    )

    # * RDM 1
    dissimilarity_metric = "euclidean"
    rdm = calc_rdm(ds, dissimilarity_metric)

    title = f"Accuracy RDM"  # - {dissimilarity_metric.capitalize()} Distance"
    fig, ax, sep_mask = plot_rdm(rdm, "types", True, title=title, get_sep_mask=True)

    labels = sep_mask.astype("str")[:, 1]
    labels[np.where(labels == "1.0")] = ds.obs_descriptors["ids"]
    labels[np.where(labels == "0.0")] = ""

    ax.set_xticks(np.arange(labels.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(labels.shape[0]))
    ax.set_yticklabels(labels)
    figs[f"{title}_{dissimilarity_metric}"] = fig
    plt.show()

    # * RDM 2
    dissimilarity_metric = "correlation"
    rdm = calc_rdm(ds, dissimilarity_metric)
    fig, ax = plot_rdm(rdm, "types", True, title=title)
    ax.set_xticks(np.arange(labels.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(labels.shape[0]))
    ax.set_yticklabels(labels)

    figs[f"{title}_{dissimilarity_metric}"] = fig
    plt.show()

    # * ----------------------------------------
    # * ----------------------------------------
    # * --------- FIGURE: Accuracy by Pattern - Line ---------
    title = "Accuracy by Pattern"
    hue_order = ["human", "ANN"]
    fig, ax = plt.subplots(**fig_params)
    sns.lineplot(
        # data=combined_data.query(f"id in {best_humans + worst_humans + best_anns}"),
        # data=combined_data.query(f"id in {best_humans + best_anns}"),
        data=combined_data,
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
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Pattern")
    # ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    figs[title + " line"] = fig
    plt.close()

    # * --------- FIGURE: Accuracy by Pattern - All ANNs Groups - Line ---------
    # title = "Accuracy by Pattern"
    hue_order = ["human", "ANN"]
    fig, ax = plt.subplots(**fig_params)
    _plot_params = dict(
        x="pattern",
        y="correct",
        ax=ax,
        hue_order=hue_order,
        marker="o",
        errorbar="ci",
    )
    sns.lineplot(
        data=combined_data.query(f"id in {best_humans + worst_humans}"),
        label="Humans",
        **_plot_params,
    )
    sns.lineplot(
        data=combined_data.query(f"id in {best_anns + worst_anns}"),
        label="All ANNs",
        **_plot_params,
    )
    sns.lineplot(
        data=combined_data.query(f"id in {best_anns}"),
        label="Best ANNs",
        **_plot_params,
    )
    sns.lineplot(
        data=combined_data.query(f"id in {worst_anns}"),
        label="Worst ANNs",
        **_plot_params,
    )

    # * Plot overall mean accuracy of each group
    line_colors = [line.get_c() for line in ax.get_lines()]
    group_means = [
        combined_data.query(f"id in {group}")["correct"].mean()
        for group in [
            best_humans + worst_humans,
            best_anns + worst_anns,
            best_anns,
            worst_anns,
        ]
    ]
    ax.hlines(group_means, 0, len(PATTERNS) - 1, colors=line_colors, linestyles="--")

    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    # ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels((ax.get_yticks() * 100).astype(int))
    # ax.set_xlabel("Pattern")
    ax.set_xlabel("")
    # ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    figs[title + " line - ANN groups"] = fig
    plt.close()

    # * --------- FIGURE: Accuracy by Pattern - All ANNs - Line ---------
    # hue_order = ["human", "ANN"]

    fig, ax = plt.subplots(**fig_params)
    _plot_params = dict(
        x="pattern",
        y="correct",
        ax=ax,
        # hue_order=hue_order,
        marker="o",
        errorbar="ci",
    )
    sns.lineplot(
        data=combined_data.query(f"id in {best_humans + worst_humans}"),
        label="humans",
        color="black",
        **_plot_params,
    )
    sns.lineplot(
        data=combined_data.query(f"id in {best_anns + worst_anns}"),
        hue="id",
        **_plot_params,
    )

    # line_colors = [ax.lines[i].get_c() for i in range(len(hue_order))]
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Pattern")
    # ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    figs[title + " line - All ANNs"] = fig

    # * --------- FIGURE: Performance by Pattern - Violin ---------
    acc_combined = pd.merge(
        combined_data.groupby(["id", "pattern"]).correct.mean().reset_index(),
        combined_data.groupby(["id"])["type"].first().reset_index(),
        on="id",
        how="left",
    ).rename(columns={"correct": "accuracy"})

    fig, ax = plt.subplots(**fig_params)
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
    # ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    # ax.tick_params(axis="x", rotation=45)
    figs[title + " violin"] = fig
    # fig.savefig()

    # # * --------- FIGURE: Performance by Pattern - Violin ---------
    # fig, ax = plt.subplots(**fig_params)
    # sns.violinplot(
    #     combined_data.query("type=='human' | id.str.contains('Qwen2.5-72B-')"),
    #     hue="type",
    #     x="pattern",
    #     y="correct",
    #     ax=ax,
    #     split=True,
    #     inner="quartile",
    #     linewidth=0.5,  # palette=line_colors
    # )
    # # ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # ax.legend(bbox_to_anchor=(1, 1))
    # ax.set_title(title)
    # ax.set_ylabel("Accuracy (%)")
    # # ax.tick_params(axis="x", rotation=45)
    # figs[title + " violin-qwen2.5-72B"] = fig
    # # fig.savefig()

    # # * --------- FIGURE: Performance by Pattern - Violin ---------
    # fig, ax = plt.subplots(**fig_params)
    # sns.violinplot(
    #     combined_data.query(
    #         "type=='human' | id.str.contains('Qwen2.5-72B-|Llama-3.3-70B')"
    #     ),
    #     hue="type",
    #     x="pattern",
    #     y="correct",
    #     ax=ax,
    #     split=True,
    #     inner="quartile",
    #     linewidth=0.5,  # palette=line_colors
    # )
    # # ax.grid(axis="y", linestyle="--", linewidth=0.5)
    # ax.legend(bbox_to_anchor=(1, 1))
    # ax.set_title(title)
    # ax.set_ylabel("Accuracy (%)")
    # # ax.tick_params(axis="x", rotation=45)
    # figs[title + " violin-qwen_and_llama"] = fig
    # # fig.savefig()

    # * --------- FIGURE: Performance by Pattern - Scatter ---------
    fig, ax = plt.subplots(**fig_params)
    sns.stripplot(
        acc_combined,
        hue="type",
        x="pattern",
        y="accuracy",
        jitter=0.1,
        dodge=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1, 1))
    # ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Accuracy (%)")
    figs[title + " scatter"] = fig
    # fig.savefig()

    #  * --------- Export figures and tables ---------
    if save_dir is not None:
        prefix = "perf-"
        suffix = ""
        # suffix = "-(human_data)-(group_lvl)"

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # * Export figures to PNG
        for file_name, fig in figs.items():
            file_name = clean_filename(f"{prefix}{file_name}{suffix}", "lower")
            fig.savefig(save_dir / f"{file_name}.png", **save_fig_params)

        # # * Export tables to PNG
        # prefix += "table-"
        # for file_name, table in tables.items():
        #     file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
        #     dfi.export(
        #         table,
        #         save_dir / f"{file_name}.png",
        #         table_conversion="matplotlib",
        #     )

        # * Save the 'figs' dictionnary to a pickle file
        fpath = save_dir / "saved_figs.pickle"
        save_pickle(figs, fpath)

    # fig = figs["ANN Accuracy Correlation with Humans vs. ANN Overall Accuracy scatter"]
    # ax = fig.get_axes()[0]

    # ax.set_xticks(np.arange(0, 1.01, 0.15))
    # fig.set_dpi(500)


# * ####################################################################################
# * REPRESENTATIONS ANALYSIS
# * ####################################################################################
def load_representation_datasets(human_data_dir: Path, ANN_data_dir: Path, level: str):
    assert level in ["pattern", "sequence"], "level must be 'pattern' or 'sequence'"

    # * 1. Load Humans Datasets
    humans_dataset_files = sorted(
        [
            f
            for f in human_data_dir.glob(f"datas*{level}*.hdf5")
            if not f.name.startswith(".")
        ]
    )

    subj_ids = [
        re.findall(r"dataset-human-(.+?)-", f.stem)[0] for f in humans_dataset_files
    ]

    humans_datasets = {
        subj_id: rsatoolbox.data.dataset.load_dataset(str(f))
        for subj_id, f in zip(subj_ids, humans_dataset_files)
    }

    # # * filter for complete dataset
    # if level == "sequence":
    #     humans_datasets = {
    #         subj_id: ds for subj_id, ds in humans_datasets.items() if ds.n_obs == 400
    #     }

    # * 2. Load ANNs Datasets
    ANNs_dataset_files = sorted(
        [
            f
            for f in ANN_data_dir.glob(f"datas*ANN*{level}*.hdf5")
            if not f.name.startswith(".")
        ]
    )

    ANNs_ids = [
        re.findall(r"dataset-ANN-(.+?)-model", f.stem)[0].replace("--", "/")
        for f in ANNs_dataset_files
    ]

    ANNs_datasets = {
        ann_id: rsatoolbox.data.dataset.load_dataset(str(f))
        for ann_id, f in zip(ANNs_ids, ANNs_dataset_files)
    }

    return humans_datasets, ANNs_datasets


def load_accuracy_datasets(human_data_dir: Path, ANN_data_dir: Path, level: str):
    assert level in ["pattern", "sequence"], "level must be 'pattern' or 'sequence'"

    # * 1. Load Humans Datasets
    humans_dataset_files = sorted(
        [
            f
            for f in human_data_dir.glob(f"datas*{level}*.hdf5")
            if not f.name.startswith(".")
        ]
    )

    subj_ids = [
        re.findall(r"dataset-human-(.+?)-", f.stem)[0] for f in humans_dataset_files
    ]

    humans_datasets = {
        subj_id: rsatoolbox.data.dataset.load_dataset(str(f))
        for subj_id, f in zip(subj_ids, humans_dataset_files)
    }

    # # * filter for complete dataset
    # if level == "sequence":
    #     humans_datasets = {
    #         subj_id: ds for subj_id, ds in humans_datasets.items() if ds.n_obs == 400
    #     }

    # * 2. Load ANNs Datasets
    ANNs_dataset_files = sorted(
        [
            f
            for f in ANN_data_dir.glob(f"datas*ANN*{level}*.hdf5")
            if not f.name.startswith(".")
        ]
    )

    ANNs_ids = [
        re.findall(r"dataset-ANN-(.+:?)-.+_lvl", f.stem)[0].replace("--", "/")
        for f in ANNs_dataset_files
    ]

    ANNs_datasets = {
        ann_id: rsatoolbox.data.dataset.load_dataset(str(f))
        for ann_id, f in zip(ANNs_ids, ANNs_dataset_files)
    }

    return humans_datasets, ANNs_datasets


def compare_representations(
    datasets1: Dict[str, Dataset],
    datasets2: Dict[str, Dataset],
    similarity_metric: str,
    dissimilarity_metric: str,
    tail: str,
    save_dir: Path,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    boot_conf_int: Tuple[float, float] = (2.5, 97.5),
    random_state: Optional[int] = None,
    pbar: bool = False,
    pbar_perm: bool = False,
    pbar_boot: bool = False,
    descriptor_match: Optional[str] = None,
):
    # # ! TEMP
    # level = "pattern"
    # eeg_chan_group = "frontal"
    # human_data_dir = (
    #     EXPORT_DIR / f"Lab/analyzed/RSA-FRP-{eeg_chan_group}-metric_correlation"
    # )
    # # human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Response_ERP-Frontal"

    # ANN_data_dir = (
    #     EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
    # )
    # humans_datasets, ANNs_datasets = load_representation_datasets(
    #     human_data_dir, ANN_data_dir, level
    # )
    # datasets1, datasets2 = ANNs_datasets, humans_datasets
    # similarity_metric = "corr"
    # dissimilarity_metric = "correlation"
    # tail = "greater"
    # save_dir = EXPORT_DIR / f"comparison/RSA-{eeg_chan_group}-{get_timestamp()}"
    # n = 10
    # n_perm, n_boot = n, n
    # boot_conf_int = (2.5, 97.5)
    # random_state = None
    # pbar = True
    # # ----------------------------------------
    # conf = dict(
    #     eeg_chan_group=eeg_chan_group,
    #     level=level,
    #     similarity_metric=similarity_metric,
    #     dissimilarity_metric=dissimilarity_metric,
    #     tail=tail,
    #     n_perm=n_perm,
    #     n_boot=n_boot,
    #     boot_conf_int=boot_conf_int,
    # )
    # conf = Box(conf)
    # save_dir.mkdir(exist_ok=True, parents=True)

    # with open(save_dir / "conf.json", "w") as f:
    #     json.dump(conf, f, indent=4)
    # # ! TEMP

    save_dir.mkdir(exist_ok=True, parents=True)

    observed_corrs = []
    permuted_corrs = []
    bootstrap_corrs = []

    res = []

    iterator1 = tqdm(
        datasets1.items(), disable=not (pbar), desc="Compare Representations"
    )
    iterator2 = tqdm(datasets2.items(), disable=True)

    for id1, ds1 in iterator1:
        for id2, ds2 in iterator2:
            _obs_corr1, _permuted_corrs, _p_val = None, None, None
            _obs_corr2, _bootstrap_corrs, _conf_int = None, None, (None, None)

            ds1_new, ds2_new = match_datasets_on_descriptor(ds1, ds2, descriptor_match)
            ds1_new, ds2_new = match_datasets_on_nan(ds1_new, ds2_new)

            # * Permutation test
            if n_perm > 0:
                _obs_corr1, _permuted_corrs, _p_val = rsa_permutation(
                    dataset1=ds1_new,
                    dataset2=ds2_new,
                    dissimilarity_metric=dissimilarity_metric,
                    similarity_metric=similarity_metric,
                    tail=tail,
                    n_perm=n_perm,
                    random_state=random_state,
                    pbar=pbar_perm,
                )

            # * Bootstrapping test
            if n_boot > 0:
                _obs_corr2, _bootstrap_corrs, _conf_int = rsa_bootstrap(
                    dataset1=ds1_new,
                    dataset2=ds2_new,
                    dissimilarity_metric=dissimilarity_metric,
                    similarity_metric=similarity_metric,
                    n_boot=n_boot,
                    ci_percentiles=boot_conf_int,
                    random_state=random_state,
                    pbar=pbar_boot,
                )

            if n_perm == 0 and n_boot == 0:
                ann_rdm = calc_rdm(ds1_new, method=dissimilarity_metric)
                subj_rdm = calc_rdm(ds2_new, method=dissimilarity_metric)
                _obs_corr1 = compare_rdms(
                    ann_rdm, subj_rdm, method=similarity_metric
                ).item()

            row = [id1, id2, _obs_corr1, _p_val, *_conf_int]
            res.append(row)

            observed_corrs.append([id1, id2, _obs_corr1])
            permuted_corrs.append([id1, id2, _permuted_corrs])
            bootstrap_corrs.append([id1, id2, _bootstrap_corrs])

    df_cols = ["id1", "id2", "corr", "p_val"] + [f"ci{i}" for i in boot_conf_int]
    df_res = pd.DataFrame(res, columns=df_cols)

    return observed_corrs, permuted_corrs, bootstrap_corrs, df_res


# * ####################################################################################
# * MAIN
# * ####################################################################################
def temp_fig_with_grouped_ax():
    data = np.random.randint(0, 100, (40, 40))

    labels = ["human"] * 25 + ["ANN"] * 10 + ["other"] * 5
    labels = np.array(labels)
    # first = (0, labels[0])
    # last = (len(labels) -1 , labels[len(labels) -1])
    boundaries = [i for i, v in enumerate(labels[:-1]) if v != labels[i + 1]]
    boundaries = [0, *boundaries, len(labels) - 1]
    inds = list(zip(boundaries[:-1], boundaries[1:]))
    inds = [np.array(range(*ind)) for ind in inds]

    fig, ax = plt.subplots()
    ax.imshow(data)
    ax.set_xticks([])
    ax.set_yticks([])

    # * ------ X axis labels ------
    label_locations = np.array([_inds[0] + len(_inds) / 2 for _inds in inds])
    x_sec = ax.secondary_xaxis(location=0)
    x_sec.set_xticks(label_locations, labels=[f"\n\n{l}" for l in np.unique(labels)])
    x_sec.tick_params("x", length=0)

    # * lines between the classes:
    x_sec2 = ax.secondary_xaxis(location=0)
    x_sec2.set_xticks(boundaries, labels=[])
    x_sec2.tick_params("x", length=40, width=1.5)

    # * ------ Y axis labels ------
    y_sec = ax.secondary_yaxis(location=0)
    y_sec.set_yticks(
        label_locations - 2, labels=[f"\n\n{l}" for l in np.unique(labels)]
    )
    y_sec.tick_params("y", length=0)

    # * lines between the classes:
    y_sec2 = ax.secondary_yaxis(location=0)
    y_sec2.set_yticks(boundaries, labels=[])
    y_sec2.tick_params("y", length=40, width=1.5)

    plt.show()


def main(eeg_chan_group: str, n_perm, n_boot):
    # # # ! TEMP
    # eeg_chan_group = "frontal"
    # n = 10_000
    # n = 1
    # n_perm, n_boot = n, 0
    # # # ! TEMP

    save_fig_params = dict(dpi=300, bbox_inches="tight")

    timestamp = get_timestamp()
    save_dir = EXPORT_DIR / f"comparison/{timestamp}"
    # save_dir = EXPORT_DIR / "20250515_102950"

    # * ================================================================================
    # * Compare Performance
    # * ================================================================================
    save_dir_perf = save_dir / "performance"
    save_dir_perf.mkdir(exist_ok=True, parents=True)

    def accuracy_rsa():
        # conf = dict()
        # conf = Box(conf)

        # with open(save_dir_perf / "conf.json", "w") as f:
        #     json.dump(conf, f, indent=4)

        human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Accuracy"
        ANN_data_dir = EXPORT_DIR / "ANNs/analyzed/RSA-Accuracy"

        for level in ["pattern", "sequence"]:
            human_ds, ann_ds = load_accuracy_datasets(
                human_data_dir=human_data_dir, ANN_data_dir=ANN_data_dir, level=level
            )
            ann_ids = [clean_ann_id(k) if k != "ann_avg" else k for k in ann_ds.keys()]
            ann_ds = dict(zip(ann_ids, ann_ds.values()))

            cstm_order = sorted(human_ds.keys()) + ["ann_avg"] + ANN_ID_ORDER

            rsm, fig = simple_rsa(
                datasets={**human_ds, **ann_ds},
                dissimilarity_metric="euclidean",
                similarity_metric="corr",
                order=cstm_order,
            )
            rsm_fpath = save_dir_perf / f"rsm-accuracy-{level}_lvl.feather"
            rsm.to_feather(rsm_fpath)
            fig.savefig(rsm_fpath.with_suffix(".png"), **save_fig_params)

    # * ================================================================================
    # * Compare Representations
    # * ================================================================================

    save_dir_repr = save_dir / "representation"
    save_dir_repr.mkdir(exist_ok=True, parents=True)

    def repr_rsa():
        # human_data_dir = (
        #     EXPORT_DIR / f"Lab/analyzed/RSA-FRP-{eeg_chan_group}-metric_correlation"
        # )

        human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Rest_ERP-frontal"
        # human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Response_ERP-frontal"

        ANN_data_dir = (
            EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
        )

        conf = dict(
            human_data_dir=str(human_data_dir),
            ANN_data_dir=str(ANN_data_dir),
            eeg_chan_group=eeg_chan_group,
            similarity_metric="corr",
            dissimilarity_metric="correlation",
            tail="greater",
            n_perm=n_perm,
            n_boot=n_boot,
            boot_conf_int=(2.5, 97.5),
        )
        conf = Box(conf)

        with open(save_dir_repr / "conf.json", "w") as f:
            json.dump(conf, f, indent=4)

        # for lvl in ["sequence", "pattern"]:
        # for lvl in ["pattern"]:
        for lvl in ["sequence"]:
            descriptor_match = "item_ids" if lvl == "sequence" else None

            humans_datasets, ANNs_datasets = load_representation_datasets(
                human_data_dir, ANN_data_dir, lvl
            )
            humans_datasets = {
                k.replace("group_avg", "human_avg"): v
                for k, v in humans_datasets.items()
            }

            ANNs_datasets = {clean_ann_id(k): v for k, v in ANNs_datasets.items()}
            all_datasets = {**ANNs_datasets, **humans_datasets}

            # cstm_order = sorted(humans_datasets.keys()) + ["ann_avg"] + ANN_ID_ORDER
            cstm_order = sorted(humans_datasets.keys()) + ANN_ID_ORDER

            rsm, fig = simple_rsa(
                datasets=all_datasets,
                dissimilarity_metric=conf.dissimilarity_metric,
                similarity_metric=conf.similarity_metric,
                order=cstm_order,
                descriptor_match=descriptor_match,
            )

            rsm_fpath = save_dir_repr / f"rsm-repr-{lvl}_lvl.feather"
            rsm.to_feather(rsm_fpath)
            fig.savefig(rsm_fpath.with_suffix(".png"), **save_fig_params)
            save_pickle(fig, rsm_fpath.parent / f"{rsm_fpath.name}-fig.pkl")

            observed_corrs, permuted_corrs, bootstrap_corrs, df_res = (
                compare_representations(
                    datasets1=ANNs_datasets,
                    datasets2=humans_datasets,
                    similarity_metric=conf.similarity_metric,
                    dissimilarity_metric=conf.dissimilarity_metric,
                    tail=conf.tail,
                    save_dir=save_dir_repr,
                    n_perm=conf.n_perm,
                    n_boot=conf.n_boot,
                    boot_conf_int=conf.boot_conf_int,
                    random_state=0,
                    pbar=True,
                    pbar_perm=True,
                    pbar_boot=False,
                    descriptor_match=descriptor_match,
                )
            )

            observed_corrs = pd.DataFrame(
                observed_corrs, columns=["ANN_id", "subj_id", "obs_corr"]
            )

            perm_corrs_dir = save_dir_repr / "perm_corrs"
            perm_corrs_dir.mkdir(exist_ok=True, parents=True)

            for ann_id, subj_id, corrs in permuted_corrs:
                _ann_id = clean_ann_id(ann_id)
                np.save(perm_corrs_dir / f"perm_corrs-{_ann_id}-{subj_id}.npy", corrs)

            boot_corrs_dir = save_dir_repr / "boot_corrs"
            boot_corrs_dir.mkdir(exist_ok=True, parents=True)

            for ann_id, subj_id, corrs in bootstrap_corrs:
                _ann_id = clean_ann_id(ann_id)
                np.save(boot_corrs_dir / f"boot_corrs-{_ann_id}-{subj_id}.npy", corrs)

            # * ----------------------------------------

            # * Clean ANN IDs
            df_res["id1"] = df_res["id1"].apply(clean_ann_id)

            # * Sort by custom order of ANN IDs
            df_res = df_res.sort_values(
                by=["id1", "id2"],
                key=lambda x: x.map({k: i for i, k in enumerate(ANN_ID_ORDER)}),
            ).reset_index(drop=True)

            df_res["sig"] = df_res["p_val"] < 0.05
            sig_res = df_res[df_res["sig"]]

            hue_order = sorted(sig_res["id2"].unique())
            fig, ax = plt.subplots()
            sns.scatterplot(
                sig_res, x="corr", y="id1", hue="id2", hue_order=hue_order, ax=ax
            )
            ax.grid(axis="x", ls="--")
            ax.set_xlim(0.2, 0.8)
            ax.set_ylabel("")
            ax.set_xlabel("Pearson Correlation")
            ax.legend(title="")
            fig.savefig(save_dir_repr / "sig_corr.png", **save_fig_params)
            plt.show()

            df_res.to_csv(save_dir_repr / f"RSA_results.csv", index=False)

            df_res_group = (
                df_res.query("id2=='human_avg'")
                .sort_values(
                    by=["id1", "id2"],
                    key=lambda x: x.map({k: i for i, k in enumerate(ANN_ID_ORDER)}),
                )
                .reset_index(drop=True)
            )

            fig, ax = plt.subplots()
            # scatter = sns.scatterplot(
            #     df_res_group, x="corr", y="ANN_id", ax=ax, hue="ANN_id", hue_order=ANN_ID_ORDER, legend=False,
            # )
            ax.scatter(df_res_group["corr"], range(len(df_res_group)))
            ax.hlines(
                range(len(df_res_group)),
                df_res_group[f"ci{conf.boot_conf_int[0]}"],
                df_res_group[f"ci{conf.boot_conf_int[1]}"],
            )
            ax.set_yticks(
                range(len(df_res_group)), df_res_group["id1"]
            )  # , rotation=90)
            ax.grid(axis="x", ls="--")
            # ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_xlabel("correlation")
            fig_path = save_dir_repr / f"RSA_results-group_lvl.png"
            fig.savefig(fig_path, **save_fig_params)
            plt.show()
            save_pickle(fig, fig_path.with_suffix(".pickle"))

        # ! TEMP {1} ! #
        # repr_corr = df_res_group.rename(
        #     columns={"id1": "ann_id", "id2": "human_id", "corr": "repr_corr"}
        # )

        # acc_corr = pd.read_csv(
        #     "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/Performance/ann_accuracy_correlation_with_humans_vs._ann_overall_accuracy.csv"
        # ).rename(columns={"corr": "acc_corr"})

        # acc_corr["human_id"] = acc_corr["human_id"].str.replace("group", "human_avg")
        # corr_df = repr_corr.merge(acc_corr, on=["ann_id", "human_id"])

        # fig, ax = plt.subplots()
        # sns.scatterplot(data=corr_df, x="acc_corr", y="repr_corr", ax=ax)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.grid(which="both", ls="--")

        # fig, ax = plt.subplots()
        # sns.scatterplot(data=corr_df, x="accuracy", y="repr_corr", ax=ax)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.grid(which="both", ls="--")

        # fig, ax = plt.subplots()
        # sns.scatterplot(data=corr_df, x="accuracy", y="acc_corr", ax=ax)
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.grid(which="both", ls="--")

        # # * ----------------------------------------

        # group_res = df_res.query("subj_id == 'group_avg'").set_index("ANN_id")
        # group_res = group_res.loc[ANN_ID_ORDER]
        # group_res.to_csv(save_dir_repr / "RSA_results-group.csv")
        # ! TEMP {1} ! #

        return df_res

    def multimodal_rsa():
        # conf = Box(dict(
        #     human_data_dir=str(human_data_dir),
        #     ANN_data_dir=str(ANN_data_dir),
        #     eeg_chan_group=eeg_chan_group,
        #     similarity_metric="corr",
        #     dissimilarity_metric="correlation",
        #     tail="greater",
        #     n_perm=10_000,  # n_perm,
        #     n_boot=0,  # n_boot,
        #     boot_conf_int=(2.5, 97.5),
        # ))

        # with open(save_dir_repr / "conf.json", "w") as f:
        #     json.dump(conf, f, indent=4)

        human_data_dir_repr = (
            EXPORT_DIR / f"Lab/analyzed/RSA-FRP-{eeg_chan_group}-metric_correlation"
        )
        # human_data_dir_repr = EXPORT_DIR / "Lab/analyzed/RSA-Response_ERP-Frontal"

        ANN_data_dir_repr = (
            EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
        )

        human_data_dir_acc = EXPORT_DIR / "Lab/analyzed/RSA-Accuracy"
        ANN_data_dir_acc = EXPORT_DIR / "ANNs/analyzed/RSA-Accuracy"

        lvl = "pattern"

        human_ds_repr, ann_ds_repr = load_representation_datasets(
            human_data_dir_repr, ANN_data_dir_repr, lvl
        )

        human_ds_acc, ann_ds_acc = load_accuracy_datasets(
            human_data_dir_acc, ANN_data_dir_acc, lvl
        )

        human_ds_repr = {
            k.replace("group_avg", "human_avg"): v for k, v in human_ds_repr.items()
        }

        human_ds_acc = {
            k.replace("group_avg", "human_avg"): v for k, v in human_ds_acc.items()
        }
        # human_ds = {**human_ds_repr, **human_ds_acc}
        # ann_ds = {**ann_ds_repr, **ann_ds_acc}

        common_human_ids = sorted(set(human_ds_repr).intersection(set(human_ds_acc)))
        common_ann_ids = sorted(set(ann_ds_repr).intersection(set(ann_ds_acc)))

        human_ds_repr = {k: human_ds_repr[k] for k in common_human_ids}
        human_ds_acc = {k: human_ds_acc[k] for k in common_human_ids}
        ann_ds_repr = {k: ann_ds_repr[k] for k in common_ann_ids}
        ann_ds_acc = {k: ann_ds_acc[k] for k in common_ann_ids}

        assert list(human_ds_repr) == list(human_ds_acc)
        assert list(ann_ds_repr) == list(ann_ds_acc)

        for datasets in [(human_ds_repr, human_ds_acc), (ann_ds_repr, ann_ds_acc)]:
            datasets1, datasets2 = datasets

            similarities = np.zeros((len(datasets1), len(datasets2)))
            similarities[:] = np.nan
            similarities.shape

            for i, (id1, ds1) in enumerate(datasets1.items()):
                for j, (id2, ds2) in enumerate(datasets2.items()):
                    assert ds1 != ds2
                    ds1_new, ds2_new = match_datasets_on_nan(ds1, ds2)
                    rdm1 = calc_rdm(ds1_new, "euclidean")
                    rdm2 = calc_rdm(ds2_new, "euclidean")
                    similarities[i, j] = compare_rdms(rdm1, rdm2, "cosine").item()

            df_similarities = pd.DataFrame(
                similarities, columns=datasets1.keys(), index=datasets1.keys()
            )

            res = pd.DataFrame(
                similarities[np.diag_indices_from(similarities)],
                index=datasets1.keys(),
                columns=["similarity"],
            )

            display(res.style)

            fig, ax = plt.subplots()
            ax.imshow(similarities)
            ax.set_xticks(range(len(datasets1)), datasets1.keys(), rotation=90)
            ax.set_yticks(range(len(datasets1)), datasets1.keys())
            plt.show()
            plt.close()

            # calc_rdm(human_ds_repr['human_avg'], 'correlation')
            # calc_rdm(human_ds_acc['human_avg'], 'euclidean')


def temp_perf_and_repr_corr():
    save_fig_params = dict(dpi=300, bbox_inches="tight")

    res_dir = Path(
        "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/0-MAIN/representation"
    )
    lvl = "pattern"
    res_files = list(res_dir.rglob(f"RSA_results-{lvl}.csv"))

    res = {
        f.parent.name.split("-")[0].replace("ERP_", "").lower(): pd.read_csv(f)
        for f in res_files
    }
    rename_map = {
        "frp": "FRP",
        "response": "ERP Response",
        "rest": "EEG Rest",
    }

    for res_type, df in res.items():
        res_type = rename_map[res_type]
        df["eeg_type"] = res_type

    res_df1 = pd.concat(res.values())
    res_df1["id1"] = pd.Categorical(
        res_df1["id1"], categories=ANN_ID_ORDER, ordered=True
    )
    res_df1["eeg_type"] = pd.Categorical(
        res_df1["eeg_type"], categories=list(rename_map.values()), ordered=True
    )
    res_df1.sort_values(["id1", "eeg_type"], ascending=[True, False])  # , inplace=True)
    res_df2 = res_df1.query("id2=='human_avg'")
    res_df3 = res_df1.query("id2!='human_avg'")

    res_df2.groupby(["eeg_type", "id1"])[["corr", "p_val"]].mean()
    print(
        res_df2.groupby(["eeg_type", "id1"])["corr"]
        .mean()["FRP"]
        .sort_values()
        .round(2)
        .to_latex()
    )

    dfs = dict(zip(["all", "human_avg", "human_subjects"], [res_df1, res_df2, res_df3]))

    # min_corr = round(min([min(df["corr"]) for df in res.values()]) * 1.025, 1)
    # max_corr = round(max([max(df['corr']) for df in res.values()]) * 1.025, 1)

    # mean_res_df = pd.DataFrame(columns=["type", "corr", "p_val"])

    for eeg_dtype, df in dfs.items():
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="corr", y="id1", ax=ax, errorbar=None, hue="eeg_type")
        ax.grid(axis="x", ls="--")
        ax.legend(title="", bbox_to_anchor=(1, 1))
        ax.set_ylabel("")
        ax.set_xlabel("Pearson Correlation")
        # ax.set_xlim(min_corr, max_corr)
        fig.savefig(res_dir / f"comparison-{lvl}-{eeg_dtype}.png", **save_fig_params)
        # print((res_dir / f"comparison-{eeg_dtype}.png").name)
        plt.show()

        # df_mean = df.groupby("eeg_type")[["corr", "p_val"]].mean().reset_index()
        # df_mean.insert(1, "agg_type", eeg_dtype)

        # df.groupby("type")["sig"].value_counts()
    # ! TEMP {2} ! #

    # ! TEMP {3} ! #
    gaze_ds = [
        f for f in (EXPORT_DIR / "lab/analyzed/RSA-Gaze_Heatmap").glob("*dataset*")
    ]
    gaze_ds = [f for f in gaze_ds if not f.name.startswith(".")]
    gaze_ds = {
        f.name.split("-")[2]: rsatoolbox.data.dataset.load_dataset(str(f))
        for f in gaze_ds
    }

    human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Rest_ERP-frontal"
    # human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Response_ERP-frontal"
    # human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-FRP-frontal-metric_correlation"

    ANN_data_dir = (
        EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
    )

    for lvl in ["pattern"]:
        eeg_ds, ANNs_ds = load_representation_datasets(
            human_data_dir, ANN_data_dir, lvl
        )
        eeg_ds = {k.replace("group_avg", "human_avg"): v for k, v in eeg_ds.items()}

        common = set(eeg_ds.keys()).intersection(set(gaze_ds.keys()))

        eeg_ds = {k + "_eeg": v for k, v in eeg_ds.items() if k in common}
        gaze_ds = {k + "_gaze": v for k, v in gaze_ds.items() if k in common}

        datasets = {**gaze_ds, **eeg_ds}
        rsm, fig = simple_rsa(datasets, "correlation", "corr")
        fig.get_axes()[0].tick_params("both", labelsize=8)
        fig

    # ! TEMP {3} ! #

    # * ----------------------------------------
    # * Behavior
    # * ----------------------------------------
    human_data_dir = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab")
    ANN_data_dir = Path(
        "/Volumes/Realtek 1Tb/PhD Data/experiment1/data/ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
    )
    seq_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"

    sequences = pd.read_csv(seq_file)

    # * Human data
    human_data = prepare_human_perf_data(human_data_dir)

    # * ANN data
    ann_data = prepare_ann_perf_data(ANN_data_dir, seq_file, ann_id_order=None)

    # * Cleaning ANN IDs
    ann_data["id"] = ann_data["id"].apply(clean_ann_id)

    ann_data.query("choice=='invalid'")["id"].value_counts()
    ann_data.query("choice=='invalid' & id.str.contains('Deepseek')").iloc[-1].response

    # * Get unique IDs
    unique_human_ids = human_data["id"].unique()
    unique_ann_ids = ann_data["id"].unique()

    # * Combined data
    combined_data = pd.concat([human_data, ann_data], axis=0).reset_index(drop=True)
    combined_data["id"] = combined_data["id"].astype(str)

    human_avg_perf = (
        combined_data.query('type=="human"')
        .groupby(["id", "pattern"])["correct"]
        .mean()
        .groupby("pattern")
        .mean()
    )
    human_avg_perf = human_avg_perf.reset_index()
    human_avg_perf["type"] = "human"
    human_avg_perf["id"] = "human_group"

    ann_avg_perf = (
        combined_data.query("type == 'ANN'")
        .groupby(["id", "pattern"])["correct"]
        .mean()
    )
    ann_avg_perf = ann_avg_perf.reset_index()
    ann_avg_perf["type"] = "ANN"

    res = []
    for model_id in combined_data.query('type=="ANN"')["id"].unique():
        model_perf = ann_avg_perf.query(f"id=='{model_id}'")
        corr_res = (
            human_avg_perf.merge(model_perf, on="pattern")[["correct_x", "correct_y"]]
            .corr()
            .iloc[0, 1]
        )
        res.append((model_id, corr_res))
    df_res = pd.DataFrame(res, columns=["model_id", "corr"])
    df_res = df_res.sort_values("corr")

    perf_by_patt = pd.concat([human_avg_perf, ann_avg_perf], axis=0)

    sns.lineplot(
        data=perf_by_patt,
        x="pattern",
        y="correct",
        hue="id",
        hue_order=[*ANN_ID_ORDER, "human_group"],
    )
    plt.xticks(rotation=90)
    plt.legend(title="", bbox_to_anchor=(1, 1))
    plt.show()

    # * ----------------------------------------
    # * Representations
    # * ----------------------------------------
    dfs = {
        "frp": pd.read_csv(
            "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/0-MAIN/representation/FRP-Frontal/RSA_results-pattern.csv"
        ),
        "resp": pd.read_csv(
            "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/0-MAIN/representation/ERP_Response-Frontal/RSA_results-pattern.csv"
        ),
        "rest": pd.read_csv(
            "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/20250527_184012/representation/RSA_results.csv"
        ),
    }
    dfs = Box(dfs)
    dfs.rest["id2"] = dfs.rest["id2"].replace({"group_avg": "human_avg"})
    dfs.frp["type"] = "FRP"
    dfs.resp["type"] = "ERP Response"
    dfs.rest["type"] = "EEG Rest"

    df_repr_corr = pd.concat(dfs.values(), axis=0).reset_index(drop=True)

    data_grouped = (
        df_repr_corr.query("id2=='human_avg'")
        .groupby(["type", "id1"])[["corr", "p_val"]]
        .first()
    )
    data_tidy = data_grouped.reset_index()
    data_tidy["id1"] = pd.Categorical(
        data_tidy["id1"], categories=ANN_ID_ORDER, ordered=True
    )
    data_tidy = data_tidy.sort_values(["type", "id1"]).reset_index(drop=True)

    hue_order = ["FRP", "ERP Response", "EEG Rest"]
    fig, ax = plt.subplots()
    bars = sns.barplot(
        data=data_tidy,
        x="corr",
        y="id1",
        hue="type",
        hue_order=hue_order,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Pearson Correlation")
    ax.set_ylabel("")
    ax.grid(axis="x", ls="--")
    ax.legend(title="", bbox_to_anchor=(1, 1))

    bars_xy = [
        (rect.get_x(), rect.get_y())
        for rect in ax.get_children()
        if isinstance(rect, patches.Rectangle)
    ]
    for i, (x, y) in enumerate(bars_xy):
        ax.text(
            x,
            y,
            f"{data_tidy.iloc[i]['p_val']:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ref_rdm_trial, ref_rdm_patt = get_reference_rdms(dissimilarity_metric="correlation")
    patterns = np.array([[p] * 50 for p in PATTERNS]).flatten()
    ref_rdm_trial.pattern_descriptors["patterns"] = patterns
    fig, _ = plot_rdm(ref_rdm_trial, "patterns")
    fig.savefig(
        EXPORT_DIR / "comparison/0-MAIN/representation/ref_rdm.png",
        dpi=300,
        bbox_inches="tight",
    )


def temp_behav_vs_repr_rdms():
    # * ----------------------------------------
    human_data_dir = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab")
    ANN_data_dir = Path(
        "/Volumes/Realtek 1Tb/PhD Data/experiment1/data/ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
    )
    seq_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"

    sequences = pd.read_csv(seq_file)

    # * Human data
    human_data = prepare_human_perf_data(human_data_dir)

    # * ANN data
    ann_data = prepare_ann_perf_data(ANN_data_dir, seq_file, ann_id_order=None)

    # * Cleaning ANN IDs
    ann_data["id"] = ann_data["id"].apply(clean_ann_id)

    ann_data.query("choice=='invalid'")["id"].value_counts()
    ann_data.query("choice=='invalid' & id.str.contains('Deepseek')").iloc[-1].response

    # * Get unique IDs
    unique_human_ids = human_data["id"].unique()
    unique_ann_ids = ann_data["id"].unique()

    # * Combined data
    combined_data = pd.concat([human_data, ann_data], axis=0).reset_index(drop=True)
    combined_data["id"] = combined_data["id"].astype(str)

    human_avg_perf = (
        combined_data.query('type=="human"')
        .groupby(["id", "pattern"])["correct"]
        .mean()
        .groupby("pattern")
        .mean()
    )
    human_avg_perf = human_avg_perf.reset_index()
    human_avg_perf["type"] = "human"
    human_avg_perf["id"] = "human_group"

    ann_avg_perf = (
        combined_data.query("type == 'ANN'")
        .groupby(["id", "pattern"])["correct"]
        .mean()
    )
    ann_avg_perf = ann_avg_perf.reset_index()
    ann_avg_perf["type"] = "ANN"

    perf_by_id = (
        combined_data.groupby(["id", "pattern"])["correct"].mean().reset_index()
    )

    behav_datasets, behav_rdms = {}, {}

    for id, data in perf_by_id.groupby("id"):
        data = data.set_index("pattern").sort_index()["correct"]
        ds, rdm = get_ds_and_rdm(
            measurements=data.values[:, None],
            dissimilarity_metric="correlation",
            descriptors={"id": id},
            obs_descriptors={"patterns": data.index},
        )
        behav_datasets[id] = ds
        behav_rdms[id] = rdm

    # or_hist = pd.read_csv("/Users/chris/Downloads/openrouter_activity_2025-06-12.csv")
    # or_hist.groupby("model_permaslug")["cost_total"].aggregate(['sum', 'count', 'mean']).sort_values("mean")


def recover_rsm_val(rsm_fpath: Path):
    # ! TEMP
    # rsm_fpath = (
    #     EXPORT_DIR
    #     / "comparison/0-MAIN/representation/ERP-Frontal/RSM-repr-ERP_Resp-pattern_lvl.npy"
    # )
    # ! TEMP
    rsm_fpath = Path(rsm_fpath)

    rsm = np.load(rsm_fpath, allow_pickle=True)

    fig_path = rsm_fpath.parent / f"{rsm_fpath.stem}-fig.pkl"

    fig = load_pickle(fig_path)
    ax = fig.get_axes()[0]
    labels = [l.get_text() for l in ax.get_xticklabels()]

    rsm = pd.DataFrame(rsm, columns=labels, index=labels)

    return rsm


def temp_merge_rsm():
    rsm_frp_f = "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/0-MAIN/representation/FRP-Frontal/RSM-repr-FRP-pattern_lvl.npy"
    rsm_erp_f = "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/comparison/0-MAIN/representation/ERP-Frontal/RSM-repr-ERP_Resp-pattern_lvl.npy"

    rsm_frp = recover_rsm_val(rsm_frp_f)
    rsm_erp = recover_rsm_val(rsm_erp_f)

    # label_filter = [i for i, c in enumerate(rsm_frp.columns) if not "subj" in c]
    label_filter = [c for c in rsm_frp.columns if not re.search(r"subj|human", c)]
    filt_rsm_frp = rsm_frp.loc[label_filter, "human_avg"].rename("FRP")

    # label_filter = [i for i, c in enumerate(rsm_erp.columns) if not "subj" in c]
    label_filter = [c for c in rsm_erp.columns if not re.search(r"subj|human", c)]
    filt_rsm_erp = rsm_erp.loc[label_filter, "human_avg"].rename("ERP")

    df = pd.concat([filt_rsm_erp, filt_rsm_frp], axis=1).reset_index(names="ANN")

    df.plot(kind="bar")
    df.mean()

    save_dir = Path(rsm_erp_f).parents[2]
    fpath = save_dir / "rsa-ANN_vs_Human_avg-ERP_vs_FRP.png"

    dfi.export(
        df.style.format(precision=2), fpath, table_conversion="matplotlib", dpi=300
    )
    # * ################################################################################
    rsa_res_erp_f = (
        EXPORT_DIR
        / "comparison/0-MAIN/representation/ERP-Frontal/RSA_results-pattern.csv"
    )
    rsa_res_frp_f = (
        EXPORT_DIR
        / "comparison/0-MAIN/representation/FRP-Frontal/RSA_results-pattern.csv"
    )

    rsa_res_erp = pd.read_csv(rsa_res_erp_f)
    rsa_res_frp = pd.read_csv(rsa_res_frp_f)

    rsa_res_frp["type"] = "FRP"
    rsa_res_erp["type"] = "ERP"

    rsa_res = pd.concat([rsa_res_frp, rsa_res_erp])
    rsa_res.query("id2=='human_avg'")


if __name__ == "__main__":
    pass
    # for eeg_chan_group in ["all", "occipital", "frontal"]:
    #     main(eeg_chan_group, "pattern", 100, 100)
