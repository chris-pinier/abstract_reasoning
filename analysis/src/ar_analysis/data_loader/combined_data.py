from box import Box
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import rsatoolbox
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm import calc_rdm, compare as compare_rdms
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import seaborn as sns
import dataframe_image as dfi
from IPython.display import display

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from ar_analysis.data_loader.human_data import HumanGroupData
from ar_analysis.data_loader.ann_data import ANNGroupData
from ar_analysis.analysis_plotting import plot_rdm
from ar_analysis.utils.analysis_utils import (
    save_pickle,
    apply_df_style,
    clean_ann_id,
    clean_filename,
)
from ar_analysis.analysis_rsa import (
    match_datasets_on_nan,
    match_datasets_on_descriptor,
    simple_rsa,
    rsa_bootstrap,
    rsa_permutation,
)
from ar_analysis.analysis_config import Config as c

# * ########################################
# * GLOBAL VARIABLES
# * ########################################

PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER
SEQ_FILE = c.SEQ_FILE


MAIN_SAVE_DIR: Final = Path("/Volumes/Realtek 1Tb")
DATA_DIR: Final = MAIN_SAVE_DIR / "PhD Data/experiment1/data/"
EXPORT_DIR: Final = MAIN_SAVE_DIR / "PhD Data/experiment1-analysis/"

DIRECTORIES = Box(
    {
        "ann": {
            "data": DATA_DIR
            / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
            "prepro": None,
            "analyzed": EXPORT_DIR / "ANNs/analyzed",
            "export": EXPORT_DIR / "ANNs/analyzed",
        },
        "human": {
            "data": DATA_DIR / "Lab",
            "prepro": EXPORT_DIR / "Lab/preprocessed_data",
            "analyzed": EXPORT_DIR / "Lab/analyzed",
            "export": EXPORT_DIR / "Lab/analyzed",
        },
    }
)


# * ########################################
# * DATA LOADERS
# * ########################################
class CombinedData:
    def __init__(
        self,
        ann_directories: Dict[str, Path],
        human_directories: Dict[str, Path],
        seq_file: Path,
    ):
        self.directories = Box({"ann": ann_directories, "human": human_directories})
        self.seq_file = seq_file

    def prepare_ann_perf_data(self, ann_id_order: List[str] | None = None):
        ann_data = ANNGroupData(
            data_dir=DIRECTORIES.ann.data,
            export_dir=DIRECTORIES.ann.export,
        ).get_behav_data()

        # ann_data = ann_data["raw_cleaned"]
        ann_data["type"] = "ANN"
        ann_data.rename(columns={"ann_id": "id"}, inplace=True)
        ann_data["id"] = ann_data["id"].apply(clean_ann_id)

        # * Add sequence info to ANN data
        sequences = pd.read_csv(self.seq_file)
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

    def prepare_human_perf_data(self):
        # human_data = perf_analysis_all_subj(data_dir=human_data_dir)
        human_data = HumanGroupData(
            data_dir=DIRECTORIES.human.data,
            preprocessed_dir=DIRECTORIES.human.prepro,
            export_dir=DIRECTORIES.human.export,
        ).get_behav_data()

        # human_data: pd.DataFrame = human_data["raw_cleaned"]
        human_data["type"] = "human"
        human_data.rename(columns={"subj_N": "id"}, inplace=True)
        human_data["id"] = human_data["id"].astype(str)

        return human_data

    def get_perf_data(self, ann_id_order: List[str] | None = None):
        ann_data = self.prepare_ann_perf_data(ann_id_order=ann_id_order)
        human_data = self.prepare_human_perf_data()

        # * Combined data
        combined_data = pd.concat([human_data, ann_data], axis=0).reset_index(drop=True)
        combined_data["id"] = combined_data["id"].astype(str)

        return combined_data

    def compare_performance(
        self,
        ann_data_dir: Path,
        human_data_dir: Path,
        fig_params: Optional[Dict] = None,
        save_fig_params: Optional[Dict] = None,
        save_dir: Optional[Path] = None,
        ann_id_order: Optional[List[str]] = None,
    ):
        # # ! TEMP
        # ann_data_dir = (
        #     DATA_DIR / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-all_tokens_acts"
        # )
        # ann_data_dir = (
        #     DATA_DIR / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
        # )
        # human_data_dir = DIRECTORIES.human.data
        # save_dir = EXPORT_DIR / "comparison/Performance"
        # fig_params = None
        # save_fig_params = None
        # ann_id_order=ANN_ID_ORDER
        # # ! TEMP

        # * ----------------------------------------
        # * Data Preparation
        # * ----------------------------------------
        # sequences = pd.read_csv(self.seq_file)

        combined_data = self.get_perf_data(ann_id_order=ann_id_order)

        human_data = combined_data.query("type=='human")
        ann_data = combined_data.query("type=='ann")

        # * Get unique IDs
        unique_human_ids = human_data["id"].unique()
        unique_ann_ids = ann_data["id"].unique()

        # * Human performance analysis
        average_human_acc = (
            human_data.groupby("id")["correct"].mean().sort_values(ascending=False)
        )
        best_humans = [str(i) for i in average_human_acc[average_human_acc > 0.7].index]
        worst_humans = [
            str(i) for i in average_human_acc[average_human_acc < 0.7].index
        ]

        # * ANN performance analysis
        average_ann_acc = (
            ann_data.groupby("id")["correct"].mean().sort_values(ascending=False)
        )
        best_perf_thresh_ann = 0.6
        best_anns = average_ann_acc[
            average_ann_acc >= best_perf_thresh_ann
        ].index.to_list()
        worst_anns = average_ann_acc[
            average_ann_acc < best_perf_thresh_ann
        ].index.to_list()

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

        # * Overall accuracy by item / sequence
        acc_by_item = (
            combined_data.astype({"correct": int})
            .groupby(["item_id", "type"])["correct"]
            .mean()
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

        display(apply_df_style(corr_by_ANN.set_index("ann_id"), 1))

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
        data = data.merge(
            results["ANNs_vs_human_group_corr_pattern"]["res"], on="ann_id"
        )
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
            _data = combined_data.query(
                "id==@id"
            )  # .sort_values(['pattern', 'item_id'])

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
        ax.hlines(
            group_means, 0, len(PATTERNS) - 1, colors=line_colors, linestyles="--"
        )

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

    def get_rsa_datasets(self, human_data_dir: Path, ann_data_dir: Path, level: str):
        assert level in ["pattern", "sequence"], "level must be 'pattern' or 'sequence'"

        # * 1. Load Humans Datasets
        humans_dataset_files = sorted(
            [
                f
                for f in human_data_dir.glob(f"datas*{level}*.hdf5")
                if not f.name.startswith(".")
            ]
        )

        # subj_ids = [
        #     re.findall(r"dataset-human-(.+?)-", f.stem)[0] for f in humans_dataset_files
        # ]

        # humans_datasets = {
        #     subj_id: rsatoolbox.data.dataset.load_dataset(str(f))
        #     for subj_id, f in zip(subj_ids, humans_dataset_files)
        # }
        # ! URGENT
        # TODO: code block below is brittle, fix it
        _humans_datasets = [
            rsatoolbox.data.dataset.load_dataset(str(f)) for f in humans_dataset_files
        ]

        # TODO: change "subj_N" to "id"
        # humans_datasets = {d.descriptors["id"]: d for d in humans_datasets}
        humans_datasets = {}

        for ds in _humans_datasets:
            # descriptor = d.descriptors.get("subj_N")
            # descriptor = d.descriptors.get("id") if descriptor is None else descriptor
            try:
                descriptor_key = [
                    d for d in ds.descriptors if d in ("subj_N", "id", "group")
                ][0]
            except IndexError:
                print(ds.descriptors)
                raise ValueError(
                    "Human dataset is missing 'subj_N', 'id' or 'group' descriptor"
                )

            descriptor = ds.descriptors[descriptor_key]

            if isinstance(descriptor, np.ndarray):
                descriptor = descriptor.item()
            humans_datasets[descriptor] = ds

        # # * filter for complete dataset
        # if level == "sequence":
        #     humans_datasets = {
        #         subj_id: ds for subj_id, ds in humans_datasets.items() if ds.n_obs == 400
        #     }

        # * 2. Load ANNs Datasets
        ANNs_dataset_files = sorted(
            [
                f
                for f in ann_data_dir.glob(f"datas*ANN*{level}_lvl*.hdf5")
                if not f.name.startswith(".")
            ]
        )
        _ANNs_datasets: list = [
            rsatoolbox.data.dataset.load_dataset(str(f)) for f in ANNs_dataset_files
        ]

        # TODO: change "model" to "id"
        # ANNs_datasets = {
        #     d.descriptors["id"].replace("--", "/"): d for d in ANNs_datasets
        # }

        # # ! TEMP
        # print([d.descriptors for d in _ANNs_datasets])
        # print([d.descriptors for d in _humans_datasets])
        # # ! TEMP

        ANNs_datasets = {}
        for ds in _ANNs_datasets:
            try:
                descriptor_key = [
                    d for d in ds.descriptors if d in ("model", "id", "group")
                ][0]
            except IndexError:
                print(ds.descriptors)
                raise ValueError(
                    "Human dataset is missing 'model', 'id' or 'group' descriptor"
                )

            descriptor = ds.descriptors[descriptor_key]
            # descriptor = d.descriptors.get("model")
            # descriptor = d.descriptors.get("id") if descriptor is None else descriptor

            if isinstance(descriptor, np.ndarray):
                descriptor = descriptor.item()

            descriptor = descriptor.replace("--", "/")

            ANNs_datasets[descriptor] = ds

        # ANNs_ids = [
        #     re.findall(r"dataset-ANN-(.+:?).+_lvl", f.stem)[0].replace("--", "/")
        #     for f in ANNs_dataset_files
        # ]

        # ANNs_datasets = {
        #     ann_id: rsatoolbox.data.dataset.load_dataset(str(f))
        #     for ann_id, f in zip(ANNs_ids, ANNs_dataset_files)
        # }

        return humans_datasets, ANNs_datasets

    def rsa_accuracy(
        self,
        human_data_dir: Path,
        ann_data_dir: Path,
        save_dir: Path,
        save_fig_params: dict | None = None,
        level: Literal["both", "pattern", "sequence"] = "both",
    ):
        # conf = dict()
        # conf = Box(conf)

        # with open(save_dir_perf / "conf.json", "w") as f:
        #     json.dump(conf, f, indent=4)
        save_dir.mkdir(exist_ok=True, parents=True)

        if save_fig_params is None:
            save_fig_params = dict(dpi=300, bbox_inches="tight")

        level = ["pattern", "sequence"] if level == "both" else [level]

        for lvl in level:
            human_ds, ann_ds = self.get_rsa_datasets(
                human_data_dir=human_data_dir, ann_data_dir=ann_data_dir, level=lvl
            )
            ann_ids = [clean_ann_id(k) if k != "ann_avg" else k for k in ann_ds.keys()]
            ann_ds = dict(zip(ann_ids, ann_ds.values()))

            cstm_order = sorted(human_ds.keys()) + ["ann_avg"] + ANN_ID_ORDER

            # # ! TEMP 1
            # print(f"{human_ds.keys() = }")
            # print(f"{ann_ds.keys() = }")
            # print(f"{cstm_order = }")
            # # ! TEMP

            rsm, fig = simple_rsa(
                datasets={**human_ds, **ann_ds},
                dissimilarity_metric="euclidean",
                similarity_metric="corr",
                order=cstm_order,
            )
            rsm_fpath = save_dir / f"rsm-accuracy-{lvl}_lvl.feather"
            rsm.to_feather(rsm_fpath)
            fig.savefig(rsm_fpath.with_suffix(".png"), **save_fig_params)

    def compare_representations(
        self,
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

        # ann_data_dir = (
        #     EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
        # )
        # humans_datasets, ANNs_datasets = get_rsa_representation_datasets(
        #     human_data_dir, ann_data_dir, level
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

                ds1_new, ds2_new = match_datasets_on_descriptor(
                    ds1, ds2, descriptor_match
                )
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

    def rsa_representations(
        self,
        human_data_dir: Path,
        ann_data_dir: Path,
        save_dir: Path,
        save_fig_params: dict | None = None,
        level: Literal["both", "pattern", "sequence"] = "both",
    ):
        save_dir.mkdir(exist_ok=True, parents=True)

        conf = dict(
            human_data_dir=str(human_data_dir),
            ann_data_dir=str(ann_data_dir),
            # eeg_chan_group=eeg_chan_group,
            similarity_metric="corr",
            dissimilarity_metric="correlation",
            tail="greater",
            # n_perm=n_perm,
            # n_boot=n_boot,
            boot_conf_int=(2.5, 97.5),
        )
        conf = Box(conf)

        with open(save_dir / "conf.json", "w") as f:
            json.dump(conf, f, indent=4)

        if save_fig_params is None:
            save_fig_params = dict(dpi=300, bbox_inches="tight")

        level = ["pattern", "sequence"] if level == "both" else [level]

        for lvl in level:
            descriptor_match = "item_ids" if lvl == "sequence" else None

            humans_datasets, ANNs_datasets = self.get_rsa_datasets(
                human_data_dir, ann_data_dir, lvl
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

            rsm_fpath = save_dir / f"rsm-repr-{lvl}_lvl.feather"
            rsm.to_feather(rsm_fpath)
            fig.savefig(rsm_fpath.with_suffix(".png"), **save_fig_params)
            save_pickle(fig, rsm_fpath.parent / f"{rsm_fpath.name}-fig.pkl")

            observed_corrs, permuted_corrs, bootstrap_corrs, df_res = (
                self.compare_representations(
                    datasets1=ANNs_datasets,
                    datasets2=humans_datasets,
                    similarity_metric=conf.similarity_metric,
                    dissimilarity_metric=conf.dissimilarity_metric,
                    tail=conf.tail,
                    save_dir=save_dir,
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

            perm_corrs_dir = save_dir / "perm_corrs"
            perm_corrs_dir.mkdir(exist_ok=True, parents=True)

            for ann_id, subj_id, corrs in permuted_corrs:
                _ann_id = clean_ann_id(ann_id)
                np.save(perm_corrs_dir / f"perm_corrs-{_ann_id}-{subj_id}.npy", corrs)

            boot_corrs_dir = save_dir / "boot_corrs"
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
            fig.savefig(save_dir / "sig_corr.png", **save_fig_params)
            plt.show()

            df_res.to_csv(save_dir / f"RSA_results.csv", index=False)

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
            fig_path = save_dir / f"RSA_results-group_lvl.png"
            fig.savefig(fig_path, **save_fig_params)
            plt.show()
            save_pickle(fig, fig_path.with_suffix(".pickle"))

        return df_res
