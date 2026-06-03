# * ########################################
# * IMPORTS
# * ########################################
from pathlib import Path
import pandas as pd
import pickle
import json
import re
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches
import torch
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare_cosine, compare as compare_rdm
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import numpy.typing as npt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from itertools import permutations, combinations
import seaborn as sns
from transformers import AutoTokenizer
from dataclasses import dataclass, field

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from ar_analysis.analysis_config import Config as c
from ar_analysis.analysis_plotting import plot_rdm
from ar_analysis.analysis_rsa import get_reference_rdms, get_ds_and_rdm
from ar_analysis.paths import SCRIPTS_DIR
from ar_analysis.utils.analysis_utils import read_file, reorder_item_ids


# * ########################################
# * GLOBAL VARIABLES
# * ########################################
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER
ANSWER_REGEX = r"Answer:\s?\n?(\w+)"
ITEM_ID_SORT = pd.read_csv(SCRIPTS_DIR / "item_ids_sort_for_rdm.csv")
PATTERNS = c.PATTERNS


# * ########################################
# * DATA LOADERS
# * ########################################
@dataclass
class ANNDataClass:
    data_dir: Path
    export_dir: Path

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)

    def load_data(self):
        raise NotImplementedError

    @staticmethod
    def list_contents(
        directory: Path,
        pattern: str = "*",
        sort: bool = True,
        recurs: bool = False,
        excl_dotunder=True,
    ):
        """List all files in the data directory matching the given pattern."""
        search_func = getattr(directory, "rglob" if recurs else "glob")

        if excl_dotunder:
            res = [f for f in search_func(pattern) if not f.name.startswith("._")]
        else:
            res = [f for f in search_func(pattern)]
        if sort:
            res.sort()
        return res


@dataclass
class ANNSubjData(ANNDataClass):
    ann_id: str
    behav_data: pd.DataFrame | None = None
    answer_regex: str = ANSWER_REGEX

    def __post_init__(self):
        super().__post_init__()
        ann_dir: Path = self.data_dir / self.ann_id

        if not ann_dir.exists():
            raise FileNotFoundError(f"ANN directory not found: {ann_dir}")
        self.ann_dir = ann_dir

    @staticmethod
    def clean_ann_id(ann_id):
        ann_id = ann_id.replace("/", "--")
        ann_id = ann_id.replace(":", "-")
        return ann_id

    @staticmethod
    def clean_layer_name(layer_name: str):
        layer_name = (
            layer_name.lower()
            .replace("model", "")
            .replace(".", " ")
            .replace("layers", "layer")
            .strip()
        )
        return layer_name

    def get_tokenizer():
        raise NotImplementedError

    # * ################################################################################
    # * Loading General info
    # * ################################################################################
    def get_run_info(self, res_dir: Path, model_id: str) -> Dict[str, Any]:
        run_info_file = self.ann_dir / "run_info.json"
        return read_file(run_info_file)

    # * ################################################################################
    # * "Behavioral" Data Preprocessing
    # * ################################################################################
    def load_behav(self):
        self.behav_data = self.get_behav_data()

    def preprocess_behav(
        self,
        behav_data: pd.DataFrame,
    ) -> pd.DataFrame:
        behav_data["cleaned_response"] = behav_data["response"].str.extract(
            self.answer_regex
        )
        behav_data["correct"] = behav_data["cleaned_response"] == behav_data["solution"]
        # behav_data["correct"].astype(int)

        return behav_data

    def get_behav_data(self, preprocess: bool = True) -> pd.DataFrame:
        if self.behav_data is None:
            behav_data = pd.read_csv(self.ann_dir / "responses.csv")
            if preprocess:
                behav_data = self.preprocess_behav(behav_data)
            return behav_data
        else:
            return self.behav_data

    def analyze_perf(
        self,
        return_raw: Optional[bool] = False,
        patterns: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        cleaned_behav_data: pd.DataFrame = self.get_behav_data()

        cleaned_behav_data["correct"] = cleaned_behav_data["correct"].astype(int)

        if patterns is None:
            patterns = sorted(cleaned_behav_data["pattern"].unique().tolist())

        # * --- Overall Results ---
        overall_acc = cleaned_behav_data["correct"].describe()
        overall_acc = pd.DataFrame(overall_acc).T
        overall_acc.reset_index(drop=True, inplace=True)

        # * --- Detailed Results ---
        acc_by_patt = cleaned_behav_data.groupby("pattern")["correct"].describe()
        acc_by_patt = acc_by_patt.reindex(patterns, fill_value=np.nan)
        acc_by_patt.reset_index(drop=False, inplace=True)

        res = dict(
            overall_acc=overall_acc,
            acc_by_patt=acc_by_patt,
        )

        if return_raw:
            res["raw_cleaned"] = cleaned_behav_data

        return res

    # * ################################################################################
    # * Loading Tokens and Activations
    # * ################################################################################
    def get_layer_acts(self, layer: str) -> List[np.ndarray]:
        layer_acts_file = self.ann_dir / f"acts_by_layer-{layer}.pkl"
        return read_file(layer_acts_file)

    def get_tokens(self) -> pd.DataFrame:
        tokens_file = self.ann_dir / "tokens.jsonl"
        return pd.read_json(path_or_buf=tokens_file, lines=True)

    # * ################################################################################
    # * Represetational Similarity Analysis
    # * ################################################################################
    def get_rdms_on_all_layers(
        self,
        res_dir: Path,
        dissimilarity_metric: str,
        similarity_metric: str,
        save_dir: Path,
        show_figs: bool = False,
    ):
        # # # ! TEMP
        # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
        # dissimilarity_metric = "correlation"
        # similarity_metric = "corr"
        # save_dir = WD/"temp_results"
        # show_figs = False
        # # # ! TEMP

        ann_id = self.ann_id

        save_dir.mkdir(exist_ok=True, parents=True)

        save_dir_all_layers = save_dir / "all_layers"
        save_dir_all_layers.mkdir(exist_ok=True, parents=True)

        save_dir_best_layer = save_dir / "best_layer"
        save_dir_best_layer.mkdir(exist_ok=True, parents=True)

        res_dir_model = res_dir / ann_id

        layers = [
            re.search(r"acts_by_layer-(.+)", f.stem)[1]
            for f in res_dir_model.glob("acts*.pkl")
        ]

        run_info = self.get_run_info(res_dir, ann_id)
        responses = self.get_behav_data()

        assert all(responses["item_id"] == run_info["item_ids"])

        tokens = self.get_tokens()

        # responses["cleaned_responses"] = [
        #     responses["response"][i].replace(responses["prompt"][i], "")
        #     for i in range(len(responses))
        # ]

        # # * Reorder the items
        reordered_inds = reorder_item_ids(
            original_order_df=responses[["pattern", "item_id"]],
            new_order_df=ITEM_ID_SORT[["pattern", "item_id"]],
        )
        responses = responses.iloc[reordered_inds]

        layers = sorted(layers, key=lambda x: int(x.split(".")[2]))

        layers_acts = {}

        for layer in layers:
            layer_acts = self.get_layer_acts(layer)
            layer_acts = np.concatenate(layer_acts)[reordered_inds]

            # * Flatten layers activations across tokens into a new array
            # * of shape (n_items, n_tokens * n_units)
            flattened_layers_acts = layer_acts.reshape(len(layer_acts), -1)

            # * ----- Sequence level RDM -----
            # * Convert to Dataset for RDM calculation
            layers_act_dataset = Dataset(
                measurements=flattened_layers_acts,
                descriptors={"model": ann_id, "layer": layer},
                obs_descriptors={
                    "patterns": responses["pattern"].tolist(),
                    "item_ids": responses["item_id"].tolist(),
                },
            )

            layers_acts[layer] = layers_act_dataset

        # * Generate RDM for every layer
        rdms_sequence_lvl = []
        for layer, layer_acts in tqdm(layers_acts.items()):
            rdm = calc_rdm(layer_acts, method=dissimilarity_metric)
            rdms_sequence_lvl.append(rdm)

            # * Plot the RDM and save the figure
            fig, ax = plot_rdm(rdm, "patterns", True)
            ax.set_title(f"RDM Layer Activations per Pattern\n{ann_id}-{layer}")

            # * Save the RDM data and figure
            fpath = save_dir_all_layers / f"rdm-ANN-({ann_id})-({layer})-item_lvl.hdf5"
            rdm.save(fpath, file_type="hdf5", overwrite=True)
            fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")

            plt.show() if show_figs else None
            plt.close()

        ref_rdm_trial, ref_rdm_patt = get_reference_rdms(
            dissimilarity_metric="correlation"
        )

        # * Get "Reference" RDM
        ref_item_lvl_rdm = rsatoolbox.rdm.rdms.RDMs(
            # dissimilarities=get_reference_rdms()[0][None, :, :],
            dissimilarities=ref_rdm_trial.get_matrices()[0][None, :, :],
            dissimilarity_measure="correlation",
            # descriptors="Reference RDM",
            # rdm_descriptors="item_lvl",
            pattern_descriptors={"item_ids": run_info["item_ids"]},
        )

        # * Compare every layer RDM to Reference RDM
        rdm_comparison = [
            compare_rdm(rdm, ref_item_lvl_rdm, method=similarity_metric).item()
            for rdm in rdms_sequence_lvl
        ]
        plt.plot(rdm_comparison)

        fig, ax = plt.subplots()
        ax.plot(rdm_comparison, marker="o")
        title = f"Layer-wise Similarity with Reference RDM\n{ann_id}"
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Correlation")
        ax.grid(True, ls="--")
        fname = f"{re.sub(r'\n|\s', '_', title.lower())}.png"
        fpath = save_dir / fname
        fig.savefig(fpath, dpi=200, bbox_inches="tight")

        # # * Select the layer RDM that is most similar to the reference RDM
        ind_most_similar = np.argmax(rdm_comparison)
        most_similar_layer = layers[ind_most_similar]
        rdm_sequence_lvl = rdms_sequence_lvl[ind_most_similar]

        # * Save the sequence-level dataset of this layer RDM
        file_suffix = f"ANN-{ann_id}-{most_similar_layer}-sequence_lvl.hdf5"

        dataset = layers_acts[most_similar_layer]
        fname = f"dataset-{file_suffix}"
        fpath = save_dir_best_layer / fname
        dataset.save(fpath, file_type="hdf5", overwrite=True)

        # * Save the RDM
        fname = f"rdm-{file_suffix}"
        fpath = save_dir_best_layer / fname
        rdm_sequence_lvl.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        model_name_clean = ANN_ID_MAPPING.get(ann_id, ann_id)
        layer_name_clean = self.clean_layer_name(most_similar_layer)

        fig, ax = plot_rdm(rdm_sequence_lvl, "patterns", separate_clusters=True)
        ax.set_title(
            f"RDM Layer Activations - Sequence Level\n{model_name_clean} - {layer_name_clean}"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.show() if show_figs else None
        plt.close()

        # * --------------------------------------------------------------------------------
        # * ----- Pattern level RDM -----
        # * Get the average activations per pattern
        avg_acts_per_pattern: Dict = {pat: [] for pat in PATTERNS}

        for trial, sequence_acts in enumerate(dataset.get_measurements()):
            pattern = responses.iloc[trial]["pattern"]
            avg_acts_per_pattern[pattern].append(sequence_acts)

        avg_acts_per_pattern = {
            k: np.array(v).mean(axis=0) for k, v in avg_acts_per_pattern.items()
        }

        # * Reorder the patterns, just in case
        avg_acts_per_pattern = {k: avg_acts_per_pattern[k] for k in PATTERNS}

        # * Convert to Dataset for RDM calculation
        dataset = Dataset(
            measurements=np.array([v for v in avg_acts_per_pattern.values()]),
            descriptors={"model": ann_id, "layer": layer_name_clean},
            obs_descriptors={"patterns": list(avg_acts_per_pattern.keys())},
        )

        # * Save the pattern-level dataset of this layer RDM
        file_suffix = f"ANN-{ann_id}-{most_similar_layer}-pattern_lvl.hdf5"

        fname = f"dataset-{file_suffix}"
        fpath = save_dir_best_layer / fname
        dataset.save(fpath, file_type="hdf5", overwrite=True)

        # * Compute the RDM and save the RDM
        rdm_pattern_lvl = calc_rdm(dataset, method=dissimilarity_metric)
        fname = f"rdm-{file_suffix}"
        fpath = save_dir_best_layer / fname
        rdm_pattern_lvl.save(fpath, file_type="hdf5", overwrite=True)

        # * Plot the RDM and save the figure
        fig, ax = plot_rdm(rdm_pattern_lvl, "patterns", separate_clusters=False)
        ax.set_title(
            f"RDM Layer Activations - Pattenr Level\n{model_name_clean} - {layer_name_clean}"
        )
        fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.show() if show_figs else None
        plt.close()

        return (rdm_sequence_lvl, rdm_pattern_lvl, rdms_sequence_lvl, rdm_comparison)

    def get_behav_rdms(self):
        raise NotImplementedError


@dataclass
class ANNGroupData(ANNDataClass):
    # subj_Ns: List[int] | None = None
    # subjects: Dict[int, ANNSubjData] = field(default_factory=dict)
    ann_ids: List[str] | None = None
    anns: Dict[str, ANNSubjData] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.behav_data = None

        if self.ann_ids is None:
            try:
                self.ann_ids = [f.name for f in self.list_contents(self.data_dir)]
            except Exception as e:
                raise ValueError(
                    "Couldn't find subject directories, make sure they exist at: "
                    f"{self.data_dir}.\nError Details: {e}"
                )

        for ann_id in self.ann_ids:
            self.anns[ann_id] = ANNSubjData(
                data_dir=self.data_dir,
                export_dir=self.export_dir,
                ann_id=ann_id,
            )

    def get_behav_data(self):
        if self.behav_data is None:
            behav_data = {}
            for ann_id, ann_obj in self.anns.items():
                behav_data[ann_id] = ann_obj.get_behav_data()
                behav_data[ann_id].insert(0, "ann_id", ann_id)
            behav_data = pd.concat(behav_data.values())
            # behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
            behav_data.reset_index(drop=True, inplace=True)
            return behav_data
        else:
            return self.behav_data

    def analyze_perf(self):
        # ! TEMP
        # data_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
        # answer_regex = r"Answer:\s?\n?(\w+)"
        # ! TEMP

        _acc_by_pattern, _overall_acc, _raw_cleaned = [], [], []

        for ann_id, ann_obj in self.anns.items():
            res = ann_obj.analyze_perf(return_raw=True)
            res["raw_cleaned"].insert(0, "ann_id", ann_id)

            _acc_by_pattern.append(res["acc_by_patt"])
            _overall_acc.append(res["overall_acc"])
            _raw_cleaned.append(res["raw_cleaned"])

        acc_by_pattern = pd.concat(_acc_by_pattern).reset_index(drop=True)
        overall_acc = pd.concat(_overall_acc).reset_index(drop=True)
        raw_cleaned = pd.concat(_raw_cleaned).reset_index(drop=True)

        del _acc_by_pattern, _overall_acc, _raw_cleaned

        overall_acc_stats = overall_acc.describe()

        acc_by_pattern_stats = acc_by_pattern.groupby("pattern")["mean"].describe()

        res = dict(
            acc_by_pattern=acc_by_pattern,
            overall_acc=overall_acc,
            raw_cleaned=raw_cleaned,
            acc_by_pattern_stats=acc_by_pattern_stats,
            overall_acc_stats=overall_acc_stats,
        )

        return res

    def get_accuracy_rdms(self, data_dir: Path, save_dir: Path):
        # # ! TEMP
        # # data_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
        # # save_dir = EXPORT_DIR / "analyzed/RSA-Accuracy"
        # # ! TEMP
        save_fig_params = c.SAVE_FIG_PARAMS

        save_dir.mkdir(exist_ok=True, parents=True)

        perf_res = self.analyze_perf()

        data = perf_res["raw_cleaned"]
        data.sort_values(["ann_id", "pattern"], inplace=True)

        data["type"] = "ANN"
        data.rename(columns={"ann_id": "id"}, inplace=True)
        # data["id"] = data["id"].apply(ANNSubjData.clean_ann_id)
        data["id"] = data["id"].replace(ANN_ID_MAPPING)

        dissimilarity_metric = "euclidean"

        # seq_lvl_acc_ds = {}
        # seq_lvl_acc_rdm = {}
        # patt_lvl_acc_ds = {}
        # patt_lvl_acc_rdm = {}
        # * --------------------- Sequence(trial)-level Analysis  ----------------------

        df_seq_lvl = data.copy()

        df_seq_lvl_group = data.groupby("item_id")["correct"].mean().reset_index()
        df_seq_lvl_group["id"] = "ann_avg"

        df_seq_lvl = (
            pd.concat([df_seq_lvl, df_seq_lvl_group])
            .sort_values(["pattern", "item_id"])
            .reset_index(drop=True)
        )

        idx = pd.Index(
            df_seq_lvl.sort_values(["pattern", "item_id"])["item_id"].drop_duplicates()
        )

        for id in sorted(df_seq_lvl["id"].unique()):
            d = df_seq_lvl.query(f"id=='{id}'")
            d = d.set_index(["item_id"]).reindex(idx).reset_index()

            base_fname = f"ANN-{id}-sequence_lvl"

            ds, rdm = get_ds_and_rdm(
                measurements=d["correct"].to_numpy()[:, None],
                dissimilarity_metric=dissimilarity_metric,
                ds_fpath=save_dir / f"dataset-{base_fname}.hdf5",
                rdm_fpath=save_dir / f"rdm-{base_fname}.hdf5",
                descriptors={
                    "id": id,
                    "type": "ANN",
                    "level": "sequence",
                },
                obs_descriptors={
                    "item_ids": d["item_id"].to_numpy(),
                    "patterns": d["pattern"].astype(str).to_numpy(),
                },
            )

            fig, ax = plot_rdm(rdm, "patterns", False)
            fig.savefig(save_dir / f"rdm-{base_fname}.png", **save_fig_params)
            plt.close()

            # seq_lvl_acc_ds[id] = ds
            # seq_lvl_acc_rdm[id] = rdm

        # # * -------------------------- Pattern-level Analysis --------------------------

        df_patt_lvl = data.groupby(["pattern", "id"])["correct"].mean().reset_index()
        df_patt_lvl_group = data.groupby("pattern")["correct"].mean().reset_index()
        df_patt_lvl_group["id"] = "ann_avg"

        df_patt_lvl = (
            pd.concat([df_patt_lvl, df_patt_lvl_group])
            .sort_values(["pattern", "id"])
            .reset_index(drop=True)
        )

        idx = pd.Index(df_patt_lvl["pattern"].sort_values().drop_duplicates())

        for id in sorted(df_patt_lvl["id"].unique()):
            # for id in sorted(data["id"].unique()):
            d = df_patt_lvl.query(f"id=='{id}'").sort_values("pattern")
            d = d.set_index(["pattern"]).reindex(idx).reset_index()
            # d = data.query(f"id == '{id}'")
            # sess_groups = d.groupby("sess_N").groups

            # d_per_sess = []
            # for sess_N, inds in sess_groups.items():
            #     sess_d = d.loc[inds].groupby("pattern")["correct"].mean().rename(sess_N)
            #     d_per_sess.append(sess_d)
            # d = pd.concat(d_per_sess, axis=1)

            base_fname = f"ANN-{id}-pattern_lvl"

            ds, rdm = get_ds_and_rdm(
                measurements=d["correct"].to_numpy()[:, None],
                # measurements=d.to_numpy(),
                dissimilarity_metric=dissimilarity_metric,
                ds_fpath=save_dir / f"dataset-{base_fname}.hdf5",
                rdm_fpath=save_dir / f"rdm-{base_fname}.hdf5",
                descriptors={
                    "id": id,
                    "type": "ANN",
                    "level": "pattern",
                },
                obs_descriptors={
                    "patterns": d["pattern"].astype(str).to_numpy(),
                    # "patterns": d.index.astype(str).to_numpy(),
                },
            )
            fig, ax = plot_rdm(rdm, "patterns", False)
            fig.savefig(save_dir / f"rdm-{base_fname}.png", **save_fig_params)
            plt.close()

            # patt_lvl_acc_ds[id] = ds
            # patt_lvl_acc_rdm[id] = rdm

    def rsa_between_anns(self, rdm_files: List[Path], similarity_metric: str):
        # # # ! TEMP
        # # rdm_files = [
        # #     f
        # #     for f in (EXPORT_DIR / "RDMs").glob("*layers*.hdf5")
        # #     if not f.name.startswith(".")
        # # ]
        # # similarity_metric = "cosine"
        # # # ! TEMP

        # rdm_files = sorted(rdm_files)
        # model_ids = [
        #     re.findall(r"\((.+?)\)", f.stem)[1].replace("--", "/") for f in rdm_files
        # ]

        # models_rdms = {
        #     model_id: rsatoolbox.rdm.rdms.load_rdm(str(f))
        #     for model_id, f in zip(model_ids, rdm_files)
        # }

        # similarity_matrix = np.zeros((len(models_rdms), len(models_rdms)))

        # for i, rdm1 in enumerate(models_rdms.values()):
        #     for j, rdm2 in enumerate(models_rdms.values()):
        #         similarity_matrix[i, j] = compare_rdm(
        #             rdm1, rdm2, method=similarity_metric
        #         ).item()

        # # * Plot the similarity matrix
        # models_legend = "Models:\n  "
        # models_legend += "\n  ".join(
        #     [f"{i + 1}: {model_id}" for i, model_id in enumerate(model_ids)]
        # )

        # # * these are matplotlib.patch.Patch properties
        # props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)

        # models_legend_params = dict(
        #     x=1.32,
        #     y=1.0,
        #     s=models_legend,
        #     fontsize=11,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        #     bbox=props,
        # )
        # ticks = np.arange(0, len(models_rdms), 1)
        # ticklabels = [f"{i + 1}" for i in ticks]
        # fig, ax = plt.subplots()
        # ax.imshow(similarity_matrix)
        # ax.set_title(f"Similarity Matrix - Activations on Sequence Tokens")
        # colorbar = ax.figure.colorbar(ax.imshow(similarity_matrix))
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(ticklabels)
        # ax.set_yticks(ticks)
        # ax.set_yticklabels(ticklabels)
        # ax.text(**models_legend_params, transform=ax.transAxes)
        # plt.show()
        raise NotImplementedError

    def plot_perf_analysis(
        self,
        data_dir: Path,
        show_figs: bool = False,
        figs_params: Optional[Dict] = None,
        save_fig_params: Optional[Dict] = None,
        save_dir: Optional[Path] = None,
    ):
        # # # ! TEMP
        # # data_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
        # # show_figs = True
        # # figs_params = None
        # # save_fig_params = None
        # # save_dir = EXPORT_DIR / "analyzed/group_lvl"
        # # # ! TEMP

        # if figs_params is None:
        #     fig_params = dict(figsize=(10, 6), dpi=300)
        # if save_fig_params is None:
        #     save_fig_params = dict(dpi=300, bbox_inches="tight")

        # ax_params = dict(
        #     grid={"ls": "--", "c": "k", "alpha": 0.5},
        # )

        # # ann_data = performance_analysis(
        # #     data_dir,
        # #     "meta-llama--Llama-3.3-70B-Instruct",
        # #     r"Answer:\s?\n?(\w+)",
        # #     return_raw=True,
        # # )

        # # if fig_params is None:
        # #     fig_params = {
        # #         "dpi": 300,
        # #     }

        # # ann_data["raw_cleaned"].groupby("pattern")["correct"].describe()

        # perf_res = perf_analysis_all_anns(data_dir, ANSWER_REGEX)
        # models_perf_df = perf_res["raw_cleaned"]
        # models_perf_df.sort_values(["model_id", "pattern"], inplace=True)

        # best_models = (
        #     models_perf_df.groupby("model_id")["correct"]
        #     .mean()
        #     .sort_values(ascending=False)
        # )
        # best_models = best_models[best_models > 0.7].index.tolist()

        # # models_perf_df = pd.concat(models_perf_list)

        # figs = {}

        # # * --------- FIGURE PARAMS ---------
        # patterns_legend = "Patterns:\n  "
        # patterns_legend += "\n  ".join(
        #     [f"{i + 1}: {' '.join(list(patt))}" for i, patt in enumerate(PATTERNS)]
        # )
        # # * these are matplotlib.patch.Patch properties
        # props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)

        # patterns_legend_params = dict(
        #     x=1.02,
        #     y=0.015,
        #     s=patterns_legend,
        #     fontsize=11,
        #     verticalalignment="bottom",
        #     horizontalalignment="left",
        #     bbox=props,
        # )

        # xticks = np.arange(0, len(PATTERNS), 1)
        # xticks_labels = [f"{i + 1}" for i in xticks]
        # yticks_pct = np.round(np.arange(0, 1.1, 0.1), 2)
        # ytick_labels_pct = (yticks_pct * 100).astype(int)

        # # * --------- FIGURE: Accuracy by Pattern and Model ---------
        # title = "Accuracy by Pattern and Model"
        # fig, ax = plt.subplots(**fig_params)
        # sns.lineplot(
        #     data=models_perf_df,
        #     x="pattern",
        #     y="correct",
        #     hue="model_id",
        #     errorbar=None,
        #     marker="o",
        #     ax=ax,
        # )
        # ax.legend(title="Models:", bbox_to_anchor=(1, 1))
        # ax.set_title(title)
        # ax.grid(axis="y", **ax_params.get("grid", {}))
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks_labels)
        # ax.set_yticks(yticks_pct)
        # ax.set_yticklabels(ytick_labels_pct)
        # ax.set_ylabel("Accuracy (%)")
        # ax.text(**patterns_legend_params, transform=ax.transAxes)
        # plt.tight_layout()
        # figs[title] = fig
        # plt.show() if show_figs else None
        # # plt.close()

        # # * --------- FIGURE: Accuracy by Pattern and Model Group ---------
        # title = "Accuracy by Pattern and Model Group"
        # fig, ax = plt.subplots(**fig_params)
        # sns.lineplot(
        #     data=models_perf_df,
        #     x="pattern",
        #     y="correct",
        #     errorbar="ci",
        #     marker="o",
        #     ax=ax,
        # )
        # sns.lineplot(
        #     data=models_perf_df.query("model_id in @best_models"),
        #     x="pattern",
        #     y="correct",
        #     errorbar="ci",
        #     marker="o",
        #     ax=ax,
        # )
        # lines = ax.get_lines()
        # line_colors = [l.get_color() for l in lines]
        # ax.legend(
        #     handles=lines,
        #     # labels=["All Models", "Best Models", "Mean"],
        #     labels=["All Models", "Best Models"],
        #     title="Models Groups:",
        #     bbox_to_anchor=(1, 1),
        # )
        # ax.set_title(title)
        # ax.grid(axis="y", **ax_params.get("grid", {}))
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks_labels)
        # ax.set_yticks(yticks_pct)
        # ax.set_yticklabels(ytick_labels_pct)
        # ax.set_ylabel("Accuracy (%)")
        # ax.text(**patterns_legend_params, transform=ax.transAxes)
        # plt.tight_layout()
        # figs[title] = fig
        # # plt.show()
        # # plt.close()

        # # * --------- FIGURE: Accuracy by Model ---------
        # title = "Accuracy by Model"
        # fig, ax = plt.subplots(**fig_params)
        # sns.barplot(
        #     data=models_perf_df,
        #     x="model_id",
        #     y="correct",
        #     # hue="model_id",
        #     errorbar="ci",
        #     ax=ax,
        # )
        # model_names = [
        #     f"{i + 1}: {m.get_text()}" for i, m in enumerate(ax.get_xticklabels())
        # ]
        # ax.set_xticklabels(range(1, len(model_names) + 1))
        # # properties for the text box
        # props = dict(boxstyle="round", facecolor="white", alpha=0.8, ec="black")
        # # Place the text box to the right of the Axes (x>1 means right of Axes)
        # ax.text(
        #     x=1.02,
        #     y=1,
        #     s="\n".join(model_names),
        #     transform=ax.transAxes,
        #     fontsize=11,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        #     bbox=props,
        # )
        # ax.set_title("Accuracy by Model")
        # ax.set_ylabel("Accuracy (%)")
        # ax.set_yticks(yticks_pct)
        # ax.set_yticklabels(ytick_labels_pct)
        # ax.set_ylabel("Accuracy (%)")
        # ax.grid(axis="y", **ax_params.get("grid", {}))
        # # plt.close()
        # figs[title] = fig

        # #  * --------- Export figures and tables ---------
        # if save_dir is not None:
        #     prefix = "perf-"
        #     suffix = ""

        #     save_dir = Path(save_dir)
        #     save_dir.mkdir(exist_ok=True, parents=True)

        #     # * Export figures to PNG
        #     for file_name, fig in figs.items():
        #         file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
        #         fig.savefig(save_dir / f"{file_name}.png", **save_fig_params)

        #     # # * Export tables to PNG
        #     # prefix += "table-"
        #     # for file_name, table in tables.items():
        #     #     file_name = f"{prefix}{file_name.replace(' ', '_').lower()}{suffix}"
        #     #     dfi.export(
        #     #         table,
        #     #         save_dir / f"{file_name}.png",
        #     #         table_conversion="matplotlib",
        #     #     )

        # return figs
        raise NotImplementedError


if __name__ == "__main__":
    # * -----------------------------
    # * -----------------------------
    # SSD_PATH = Path("/Volumes/Realtek 1Tb")
    # DATA_DIR = SSD_PATH / "PhD Data/experiment1/data/ANNs"

    # ANN_DIR = WD.parent / "experiment-ANNs"
    # SEQ_DIR = WD.parent / "config/sequences"

    # EXPORT_DIR = SSD_PATH / "PhD Data/experiment1-analysis/ANNs"

    # if not SSD_PATH.exists():
    #     print("WARNING: SSD not connected")
    # else:
    #     RDM_DIR = EXPORT_DIR / "RDMs"
    #     RDM_DIR.mkdir(parents=True, exist_ok=True)

    # with open(ANN_DIR / "config/instructions.txt", "r") as f:
    #     instructions = f.read()
    # * -----------------------------
    # * -----------------------------

    pass
