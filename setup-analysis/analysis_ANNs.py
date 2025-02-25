from pathlib import Path
import os
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
from analysis_plotting import plot_matrix
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import numpy.typing as npt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from itertools import permutations
import seaborn as sns
from transformers import AutoTokenizer
from analysis_utils import (
    save_pickle,
    load_pickle,
    apply_df_style,
    read_file,
    reorder_item_ids,
)
from analysis_lab_conf import Config as c

WD = Path(__file__).parent
# os.chdir(WD)
assert WD == Path.cwd()

SSD_PATH = Path("/Volumes/Realtek 1Tb")
DATA_DIR = SSD_PATH / "PhD Data/experiment1/data/ANNs"

ANN_DIR = WD.parent / "experiment-ANNs"

SEQ_DIR = WD.parent / "config/sequences"

EXPORT_DIR = SSD_PATH / "PhD Data/experiment1-analysis/ANNs"

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
# seq_prompts = pd.read_csv(ANN_DIR / "sequence_prompts.csv")


# def load_seq_files(seq_dir):
#     seq_files = [f for f in seq_dir.glob("sess*.csv")]
#     df_sequences = pd.concat([pd.read_csv(f) for f in seq_files])
#     df_sequences.reset_index(names=["trial_N"], inplace=True)

#     return df_sequences
ANSWER_REGEX = r"Answer:\s?\n?(\w+)"
ITEM_ID_SORT = pd.read_csv("item_ids_sort_for_rdm.csv")


# * -----------------------------
def compare_humans_and_anns():
    rdm_files = [f for f in RDM_DIR.glob("*.pkl") if not f.stem.startswith(".")]

    ann_rdm_files = [f for f in rdm_files if "participant" not in f.stem]
    ann_accuracy_rdm_files = [f for f in ann_rdm_files if "[accuracy]" in f.stem]
    ann_activations_rdm_files = [f for f in ann_rdm_files if "[accuracy]" not in f.stem]

    participants_rdm_files = [f for f in rdm_files if "participants" in f.stem]
    participants_accuracy_rdm = load_pickle(
        [f for f in participants_rdm_files if "[accuracy]" in f.stem][0]
    )
    participants_erp_combined_rdm = load_pickle(
        [f for f in participants_rdm_files if "[erp_combined]" in f.stem][0]
    )

    # * Load RDMs
    # ann_rdms = {f.stem: pickle.load(open(f, "rb")) for f in ann_rdms}
    ann_accuracy_rdms = {f.stem: load_pickle(f) for f in ann_accuracy_rdm_files}
    ann_activations_rdms = {f.stem: load_pickle(f) for f in ann_activations_rdm_files}

    accuracy_comparison = {}
    for name, ann_accuracy_rdm in ann_accuracy_rdms.items():
        model_id, layer = name.split("]-")
        model_id = model_id.replace("[", "").replace("]", "")
        res = compare_cosine(ann_accuracy_rdm, participants_accuracy_rdm)
        accuracy_comparison[model_id] = res.item()

    df_accuracy_comparison = pd.DataFrame([accuracy_comparison]).T
    df_accuracy_comparison.columns = ["cosine_similarity"]
    df_accuracy_comparison.sort_values(
        "cosine_similarity", ascending=False, inplace=True
    )

    activations_eeg_comparison = {}
    for name, ann_acts_rdm in ann_activations_rdms.items():
        model_id, layer = name.split("]-")
        model_id = model_id.replace("[", "").replace("]", "")
        res = compare_cosine(ann_acts_rdm, participants_erp_combined_rdm)
        activations_eeg_comparison[model_id] = res.item()

    df_activations_eeg_comparison = pd.DataFrame([activations_eeg_comparison]).T
    df_activations_eeg_comparison.columns = ["cosine_similarity"]
    df_activations_eeg_comparison.sort_values(
        "cosine_similarity", ascending=False, inplace=True
    )

    for name, rdm in ann_activations_rdms.items():
        model_id, layer = name.split("]-")
        model_id = model_id.replace("[", "").replace("]", "")
        model_id = model_id.split("--")[1]

        layer = (
            layer.replace("[", "").replace("]", "").replace("model.layers.", "layer")
        )

        plot_matrix(
            rdm.get_matrices()[0],
            title=f"{model_id}",
            labels=rdm.pattern_descriptors["pattern"],
            norm="max",
            show_values=True,
        )


def clean_model_id(model_id):
    model_id = model_id.replace("/", "--")
    model_id = model_id.replace(":", "-")
    return model_id


# * -----------------------------
def eval_model(
    responses: pd.DataFrame,
    df_prompts: pd.DataFrame,
    answer_regex=r"Answer:\s?\n?(.+)",
):
    # answer_regex = (r"Answer:\s?\n?([A-Za-z]+)",)

    resp_copy = responses.copy()

    resp_copy["cleaned_response"] = resp_copy["response"].str.extract(answer_regex)

    eval_df = resp_copy.merge(df_prompts, on=["item_id", "masked_idx"], how="left")

    eval_df["correct"] = eval_df["solution"] == eval_df["cleaned_response"]

    overall_perf = eval_df["correct"].mean()

    # model_id_cleaned = clean_model_id(responses["model_id"][0])

    # pattern_acc = eval_df.groupby("pattern")["correct"].mean().sort_index()

    # ax = plt.axes()
    # pattern_acc.plot(kind="barh", ax=ax, title=f"{model_id_cleaned}\nPattern Accuracy")
    # ax.grid(axis="x", alpha=0.4, ls="--", c="black")
    # plt.show()

    # print(f"{model_id_cleaned}: {overall_perf:.2%} accuracy")

    return (eval_df, overall_perf)


def eval_all_online_models(resp_files):
    # ! TEMP
    # resp_dir = WD.parent / "experiment-ANNs/results/online_queries/saved/masked_idx(7)"
    # resp_dir = WD.parent / "experiment-ANNs/results/local_run"
    # resp_files = (f for f in resp_dir.glob("*resp*.csv"))
    # resp_files = list(resp_files)
    # ! TEMP

    import seaborn as sns

    df_prompts = pd.read_csv(WD.parent / "experiment-ANNs/sequence_prompts.csv")
    df_prompts.reset_index(drop=True, inplace=True)

    eval_df = pd.DataFrame()
    for resp_file in resp_files:
        model_responses = pd.read_csv(resp_file)
        model_eval_df, _ = eval_model(model_responses, df_prompts)
        eval_df = pd.concat([eval_df, model_eval_df])

    display(eval_df.groupby("model_id")["item_id"].count().sort_values())

    # * Accuracy by model
    overall_perf = (
        eval_df.groupby("model_id")["correct"].mean().sort_values(ascending=False)
    )

    # * Accuracy by model and pattern
    perf_by_pattern = (
        eval_df.groupby(["model_id", "pattern"])["correct"].mean().unstack()
    )

    # * Drop rows with any NaNs (i.e., incomplete results)
    perf_by_pattern.dropna(how="any", inplace=True)

    perf_by_pattern_styled = perf_by_pattern.style.background_gradient(
        cmap="YlGn", axis=1, vmin=0, vmax=1
    ).format("{:.2f}")

    display(perf_by_pattern_styled)

    fig, ax = plt.subplots()
    ax.plot(perf_by_pattern.T, alpha=0.5, label=perf_by_pattern.index)
    ax.plot(perf_by_pattern.mean(), c="black", marker="o", markersize=4, label="Mean")
    ax.set_xticks(range(len(perf_by_pattern.columns)))
    ax.set_xticklabels(perf_by_pattern.columns, rotation=90)
    ax.set_title("Model Accuracy by Pattern")
    ax.grid(axis="both", alpha=0.4, ls="--", c="black")
    plt.legend(title="Model", bbox_to_anchor=(1, 1))
    plt.show()

    eval_df.query("model_id=='qwen/qwen-2.5-7b-instruct'").groupby("pattern")[
        "correct"
    ].mean().sort_values(ascending=False)

    # eval_df.query("model_id=='qwen/qwq-32b-preview'")
    # print(eval_df.query("model_id=='qwen/qwq-32b-preview'")['prompt'].iloc[0])
    # print(eval_df.query("model_id=='qwen/qwq-32b-preview'")["response"].iloc[0])


# * -----------------------------
def eval_mixed_masked_idx():
    df_prompts = pd.read_csv(
        WD.parent
        # / "experiment-ANNs/sequence_prompts/sequence_prompts-masked_idx(mixed)-scrambled.csv"
        / "experiment-ANNs/sequence_prompts/sequence_prompts-masked_idx(mixed)-ascii_symbols.csv"
    )

    resp_files = [
        f
        for f in (
            WD.parent
            # / "experiment-ANNs/results/online_queries/saved/masked_idx(mixed)-scrambled"
            / "experiment-ANNs/results/online_queries/saved/masked_idx(mixed)-ascii_symbols"
        ).glob("*.csv")
    ]

    eval_df = pd.DataFrame()
    for resp_file in resp_files:
        model_resps = pd.read_csv(resp_file)
        model_resps = model_resps.merge(
            df_prompts, on=["item_id", "masked_idx"], how="left"
        )
        cleaned_resp = model_resps["response"].str.extract(
            r"Answer: (.+)", flags=re.IGNORECASE
        )[0]
        model_resps["cleaned_resp"] = cleaned_resp
        model_resps["correct"] = model_resps["cleaned_resp"] == model_resps["solution"]
        eval_df = pd.concat([eval_df, model_resps])

    overall_perf = (
        eval_df.groupby("model_id")["correct"].mean().sort_values(ascending=False)
    )

    perf_per_pattern = (
        eval_df.groupby(["model_id", "pattern"])["correct"].mean().unstack().T
    ).T

    per_per_index = (
        eval_df.groupby(["model_id", "masked_idx"])["correct"].mean().unstack().T
    ).T

    perf_per_pattern_styled = perf_per_pattern.style.background_gradient(
        cmap="YlOrRd", axis=1, vmin=0, vmax=1
    ).format("{:.2f}")

    per_per_index_styled = per_per_index.style.background_gradient(
        cmap="YlOrRd", axis=1, vmin=0, vmax=1
    ).format("{:.2f}")

    perf_idx7 = (
        eval_df.query("masked_idx == 7")
        .groupby(["model_id", "pattern"])["correct"]
        .mean()
        .unstack()
    )
    perf_idx7_styled = perf_idx7.style.background_gradient(
        cmap="YlOrRd", axis=1, vmin=0, vmax=1
    ).format("{:.2f}")

    fig, ax = plt.subplots()
    sns.barplot(
        data=eval_df.groupby(["masked_idx", "model_id"])["correct"]
        .mean()
        .reset_index(),
        x="masked_idx",
        y="correct",
        hue="model_id",
        ax=ax,
        alpha=0.85,
    )
    ax.plot(
        eval_df.groupby(["masked_idx"])["correct"].mean(),
        c="red",
        marker="o",
        markersize=4,
        label="Mean",
        ls="--",
    )
    ax.grid(axis="y", alpha=0.4, ls="--", c="black")
    ax.legend(bbox_to_anchor=(1, 1))

    eval_df.groupby(["masked_idx", "model_id"])["correct"].mean().unstack()

    perf_per_pattern_styled
    per_per_index_styled
    perf_idx7_styled


# * -----------------------------
def clean_and_eval_model_responses(
    responses: pd.DataFrame,
    answer_regex: str,
) -> pd.DataFrame:
    responses["cleaned_response"] = responses["response"].str.extract(answer_regex)
    responses["correct"] = responses["cleaned_response"] == responses["solution"]

    return responses


def load_model_layer_acts(res_dir: Path, model_id: str, layer: str) -> List[np.ndarray]:
    layer_acts_file = res_dir / f"{model_id}/acts_by_layer-{layer}.pkl"
    return read_file(layer_acts_file)

    return layer_acts


def load_model_responses(res_dir: Path, model_id: str) -> pd.DataFrame:
    responses_file = res_dir / f"{model_id}/responses.csv"
    return read_file(responses_file)


def load_model_tokens(res_dir: Path, model_id: str) -> Dict[str, List[List[str]]]:
    tokens_file = res_dir / f"{model_id}/tokens.pkl"
    return read_file(tokens_file)


def load_model_run_info(res_dir: Path, model_id: str) -> Dict[str, Any]:
    run_info_file = res_dir / f"{model_id}/run_info.json"
    return read_file(run_info_file)


def load_model_files(res_dir, model_id):
    raise NotImplementedError


def clean_tokens(text_tokens: List[str], tokenizer: AutoTokenizer) -> List[str]:
    cleaned_tokens = text_tokens.copy()

    whitespace_text_token = tokenizer.tokenize(" ")
    newline_text_token = tokenizer.tokenize("\n")

    if len(whitespace_text_token) > 1:
        raise ValueError("More than one whitespace token found")
    else:
        whitespace_text_token = whitespace_text_token[0]

    if len(newline_text_token) > 1:
        raise ValueError("More than one newline token found")
    else:
        newline_text_token = newline_text_token[0]

    for i in range(len(cleaned_tokens)):
        cleaned_tokens[i] = (
            cleaned_tokens[i]
            .replace(whitespace_text_token, " ")
            .replace(newline_text_token, "\n")
        )
    return cleaned_tokens


def locate_target_tokens(
    prefix: str,
    suffix: str,
    tokens: List[str],
    error_val: Any = None,
) -> Tuple[int | None, int | None]:
    """_summary_
    Locate the indices of the target tokens in a list of tokens, using prefix and suffix strings as reference.
    Assuming that words are tokenized with their leading space (if any).

    Args:
        prefix (str): string to match at the beginning of the target tokens.
        suffix (str): string to match at the end of the target tokens.
        tokens (List[str]): list of tokens to search in.
        error_val (Any, optional): _description_. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    def tokens_list_to_string(tokens: List[str], sep: str = ""):
        return sep.join([t.strip() for t in tokens]).lower().strip().replace("\n", " ")

    prefix = prefix.lower().strip().replace("\n", " ")
    suffix = suffix.lower().strip().replace("\n", " ")

    idx_start, idx_stop = None, None

    # * Find the start index
    for idx_token in range(len(tokens)):
        reconstructed_txt = tokens_list_to_string(tokens[:idx_token])
        if reconstructed_txt == prefix.replace(" ", ""):
            idx_start = idx_token
            break

    # * Find the stop index
    for idx_token in range(1, len(tokens)):
        reconstructed_txt = tokens_list_to_string(tokens[-idx_token:])
        if reconstructed_txt == suffix.replace(" ", ""):
            idx_stop = len(tokens) - idx_token
            break

    indices = (idx_start, idx_stop)

    # * if both indices are found return them, otherwise return the error value or
    # * raise an exception
    if all([i is not None for i in indices]):
        return indices
    else:
        if error_val is not None:
            return error_val
        else:
            txt = "Unknown error, the tokens could not be matched with the target.\n"
            txt += f"{prefix} [...] {suffix}\n"
            txt += f"{tokens = }\n"

            raise Exception(txt)


def get_sequence_tokens(
    model_id: str, df_prompts: pd.DataFrame, df_sequences: pd.DataFrame
) -> List[tuple]:
    """_summary_

    Args:
        model_name (str): the model ID on https://huggingface.co/models
        df_prompts (pd.DataFrame): a pandas DataFrame containing the prompts.
            Columns must include: 'item_id', 'masked_idx', 'prompt'.
        df_sequences (pd.DataFrame): a pandas DataFrame of the csv file containing the
            sequences on which the prompts are based. Columns must include:
            'item_id', 'masked_idx'.

    Raises:
        ValueError: _description_

    Returns:
        List[tuple]: _description_
    """
    # * Load the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # * Get the columns of the individual sequence items from the sequences dataframe
    seq_cols = [c for c in df_sequences.columns if re.search(r"figure\d", c)]
    # choice_cols = [c for c in df_sequences.columns if re.search(r"choice\d", c)]

    seq_token_inds = []
    for idx in range(len(df_prompts)):
        # * Get the item_id and masked_idx of the current sequence
        item_id, masked_idx = df_prompts.iloc[idx][["item_id", "masked_idx"]].values
        item_id, masked_idx = int(item_id), int(masked_idx)

        this_prompt = df_prompts["prompt"][idx]

        # * Get a list of the individual sequence items from the sequences dataframe
        sequence = df_sequences.query(
            f"item_id == {item_id} and masked_idx == {masked_idx}"
        )[seq_cols].values[0]

        # * Replace the masked item with a question mark
        sequence[masked_idx] = "?"

        # * Join the sequence items into a single string & search for it in the prompt
        sequence_str = " ".join(sequence)
        search_res = re.search(sequence_str.replace("?", r"\?"), this_prompt)

        if search_res is None:
            raise ValueError(f"Sequence not found in prompt: {sequence_str}")

        start_idx, stop_idx = search_res.span()
        prefix, suffix = this_prompt[:start_idx], this_prompt[stop_idx + 1 :]

        # * Tokenize the prompt & clean the tokens (remove whitespace & newline tokens)
        text_tokens = tokenizer.tokenize(this_prompt)
        text_tokens = clean_tokens(text_tokens, tokenizer)

        # * Locate the start & stop index of the target tokens in the tokenized prompt
        idx_start, idx_stop = locate_target_tokens(prefix, suffix, text_tokens)

        # * Check if the sequence string matches the target tokens
        assert sequence_str == "".join(text_tokens[idx_start:idx_stop]).strip()

        seq_token_inds.append((item_id, masked_idx, idx_start, idx_stop))

    return seq_token_inds


# * -----------------------------


def get_rdms_activations_on_seq_tokens(
    res_dir: Path,
    model_id: str,
    layer: str,
    sequences_file: Path,
    dissimilarity_metric: str,
    save_dir: Optional[Path] = None,
    answer_regex: str = r"Answer:\s?\n?(\w+)",
    show_figs: bool = False,
):
    # # ! TEMP
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    # model_id = "meta-llama--Llama-3.3-70B-Instruct"
    # selected_layers = {
    #     "google--gemma-2-9b-it": "model.layers.41.post_attention_layernorm",
    #     "google--gemma-2-2b-it": "model.layers.25.post_attention_layernorm",
    #     "Qwen--Qwen2.5-7B-Instruct": "model.layers.27.post_attention_layernorm",
    #     "Qwen--Qwen2.5-72B-Instruct": "model.layers.79.post_attention_layernorm",
    #     "meta-llama--Llama-3.2-3B-Instruct": "model.layers.27.post_attention_layernorm",
    #     "meta-llama--Meta-Llama-3-8B-Instruct": "model.layers.31.post_attention_layernorm",
    #     "meta-llama--Llama-3.3-70B-Instruct": "model.layers.79.post_attention_layernorm",
    # }

    # layer = selected_layers[model_id]
    # answer_regex: str = r"Answer:\s?\n?(\w+)"
    # targ_tokens_file = (
    #     WD.parent
    #     / "experiment-ANNs/results/target_tokens/sequence_prompts-masked_idx(7)-sequence_tokens.csv"
    # )
    # sequences_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
    # dissimilarity_metric = "correlation"
    # #! TEMP

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    layer_acts = load_model_layer_acts(res_dir, model_id, layer)

    responses = load_model_responses(res_dir, model_id)

    tokens = load_model_tokens(res_dir, model_id)

    run_info = load_model_run_info(res_dir, model_id)

    sequences = pd.read_csv(sequences_file)

    # df_eval = responses.merge(
    #     sequences, on=["item_id", "solution", "masked_idx", "pattern"]
    # )

    # * Get the columns of the individual sequence items from the sequences dataframe
    seq_cols = [c for c in sequences.columns if re.search(r"figure\d", c)]
    # choice_cols = [c for c in df_eval.columns if re.search(r"figure\d", c)]

    responses = clean_and_eval_model_responses(responses, answer_regex)

    target_token_inds = pd.DataFrame(
        get_sequence_tokens(model_id.replace("--", "/"), responses, sequences),
        columns=["item_id", "masked_idx", "idx_start", "idx_stop"],
    )

    # * Check if the item IDs match between the different files
    if not all(run_info["item_ids"] == target_token_inds["item_id"]):
        raise ValueError("Item IDs do not match between run_info and target_token_inds")
    if not all(run_info["item_ids"] == responses["item_id"]):
        raise ValueError("Item IDs do not match between run_info and responses")

    tokenizer = AutoTokenizer.from_pretrained(model_id.replace("--", "/"))

    sequences_acts = []
    for trial in range(len(responses)):
        # * Extract tokens related to the sequence items
        start, stop = target_token_inds.iloc[trial][["idx_start", "idx_stop"]]

        sequence_tokens = tokens["prompt"][trial][start:stop]
        sequence_tokens = clean_tokens(sequence_tokens, tokenizer)

        # * Sanity check
        sequence_text_from_tokens = "".join(sequence_tokens).strip()
        sequence_text_from_csv = sequences.iloc[trial][seq_cols].values
        sequence_text_from_csv[sequences.iloc[trial]["masked_idx"]] = "?"
        sequence_text_from_csv = " ".join(sequence_text_from_csv)
        assert sequence_text_from_tokens == sequence_text_from_csv

        # * Extract activations for the sequence tokens
        prompt_acts = layer_acts[trial][0]
        sequence_acts = prompt_acts[:, start:stop]
        sequences_acts.append(sequence_acts)

        # response_tokens = tokens["response"][trial]
        # response_tokens = clean_tokens(response_tokens, tokenizer)
        # response_text = responses.iloc[trial]["response"]
        # response_text_from_tokens = " ".join(response_tokens)
        # assert len(response_tokens) == len(layer_acts[trial][1:])
        # assert response_text == response_text_from_tokens

        # response_acts = layer_acts[trial][1:]
        # response_acts = np.concat(response_acts, axis=1)

    assert sorted(set(responses["pattern"])) == PATTERNS, (
        "Unexpected or missing patterns"
    )

    # * Flatten layers activations across tokens into a new array of shape (n_items, n_tokens * n_units)
    flattened_layers_acts = np.concat(sequences_acts).reshape(len(sequences_acts), -1)

    # * Reorder the items
    reordered_inds = reorder_item_ids(
        original_order_df=responses[["pattern", "item_id"]],
        new_order_df=ITEM_ID_SORT[["pattern", "item_id"]],
    )
    flattened_layers_acts = flattened_layers_acts[reordered_inds, :]

    # * ----- Item level RDM -----
    # * Convert to Dataset for RDM calculation
    layers_act_dataset = Dataset(
        measurements=flattened_layers_acts,
        descriptors={"model": model_id, "layer": layer},
        obs_descriptors={"patterns": responses["pattern"].tolist()},
    )
    rdm = calc_rdm(layers_act_dataset, method=dissimilarity_metric)

    # * Plot the RDM and save the figure
    # tick_marks = np.arange(0, len(PATTERNS), 1)
    # tick_labels = PATTERNS
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(rdm.get_matrices()[0])
    ax.set_title(f"RDM Layer Activations per Pattern\n{model_id}")
    fig.colorbar(im, ax=ax)
    # ax.set_yticks(tick_marks)
    # ax.set_yticklabels(tick_labels)
    # ax.set_xticks(tick_marks)
    # ax.set_xticklabels(tick_labels, rotation=90)

    # * Save the RDM data and figure
    if save_dir is not None:
        fpath = save_dir / f"rdm-ANN-({model_id})-({layer})-item_lvl.hdf5"
        rdm.save(fpath, file_type="hdf5", overwrite=True)
        fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.show() if show_figs else None
    plt.close()

    # * ----- Pattern level RDM -----
    # * Get the average activations per pattern
    avg_acts_per_pattern: Dict = {pat: [] for pat in PATTERNS}

    for trial, sequence_acts in enumerate(sequences_acts):
        pattern = responses.iloc[trial]["pattern"]
        avg_acts_per_pattern[pattern].append(sequence_acts.mean(axis=1))

    avg_acts_per_pattern = {
        k: np.concat(v).mean(axis=0) for k, v in avg_acts_per_pattern.items()
    }

    del layer_acts, sequences_acts

    # * Reorder the patterns, just in cases
    avg_acts_per_pattern = {k: avg_acts_per_pattern[k] for k in PATTERNS}

    # * Convert to Dataset for RDM calculation
    layers_act_dataset = Dataset(
        measurements=np.array([v for v in avg_acts_per_pattern.values()]),
        descriptors={"model": model_id, "layer": layer},
        obs_descriptors={"patterns": list(avg_acts_per_pattern.keys())},
    )

    # * Compute the RDM
    rdm = calc_rdm(layers_act_dataset, method=dissimilarity_metric)

    # * Plot the RDM and save the figure
    tick_marks = np.arange(0, len(PATTERNS), 1)
    tick_labels = PATTERNS

    fig, ax = plt.subplots()
    im = ax.imshow(rdm.get_matrices()[0])
    ax.set_title(f"RDM Layer Activations per Pattern\n{model_id}")
    fig.colorbar(im, ax=ax)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_labels, rotation=90)

    # * Save the RDM data and figure
    if save_dir is not None:
        fpath = save_dir / f"rdm-ANN-({model_id})-({layer})-pattern_lvl.hdf5"
        rdm.save(fpath, file_type="hdf5", overwrite=True)
        fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")

    plt.show() if show_figs else None
    plt.close()

    return avg_acts_per_pattern, rdm


def get_rdms_activations_on_response_tokens(
    res_dir: Path,
    model_id: str,
    layer: str,
    targ_tokens_file: Path,
    sequences_file: Path,
    dissimilarity_metric: str,
    answer_regex: str = r"Answer:\s?\n?(\w+)",
    show_figs: bool = False,
):
    raise NotImplementedError


def get_rdms_activations_on_accuracy(
    res_dir: Path,
    model_id: str,
    sequences_file: Path,
    dissimilarity_metric: str = "euclidean",
    answer_regex: str = r"Answer:\s?\n?(\w+)",
    show_figs: bool = False,
):
    # # ! TEMP
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    # model_id = "google--gemma-2-9b-it"
    # selected_layers = {
    #     "google--gemma-2-9b-it": "model.layers.41.post_feedforward_layernorm"
    # }
    # layer = selected_layers[model_id]
    # answer_regex: str = r"Answer:\s?\n?(\w+)"
    # targ_tokens_file = (
    #     WD.parent
    #     / "experiment-ANNs/results/target_tokens/sequence_prompts-masked_idx(7)-sequence_tokens.csv"
    # )
    # sequences_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
    # dissimilarity_metric = "euclidean"
    # show_figs = True
    # #! TEMP

    responses = load_model_responses(res_dir, model_id)
    responses = clean_and_eval_model_responses(responses, answer_regex)

    order_by_pattern = responses.groupby("pattern").groups
    order_by_pattern = np.concat([order_by_pattern[k] for k in PATTERNS])
    responses = responses.iloc[order_by_pattern]

    # * Convert to Dataset for RDM calculation
    accuracy_dataset = Dataset(
        measurements=responses["correct"].astype(int).values[:, None],
        # measurements=acc_by_pattern.values.reshape(-1, 1),
        descriptors={"model": model_id},
        obs_descriptors={
            "patterns": responses["pattern"].tolist(),
        },
    )

    # * Compute the RDM
    rdm_acc_by_item = calc_rdm(accuracy_dataset, method=dissimilarity_metric)
    # * Save the RDM
    fpath = RDM_DIR / f"(ANN)-({model_id})-(accuracy_by_item).hdf5"
    rdm_acc_by_item.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    tick_marks = np.arange(0, len(responses), len(responses) / len(PATTERNS))
    tick_marks += len(responses) / len(PATTERNS) / 2
    tick_labels = PATTERNS

    fig, ax = plt.subplots()
    im = ax.imshow(rdm_acc_by_item.get_matrices()[0])
    ax.set_title(f"RDM - Accuracy by Sequence\n{model_id}")
    fig.colorbar(im, ax=ax)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_labels, rotation=90)
    fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.show() if show_figs else None
    plt.close()

    # * ------------------------------------------------------------

    acc_by_pattern = responses.groupby("pattern")["correct"].mean().sort_index()

    assert acc_by_pattern.index.tolist() == PATTERNS, (
        "Unexpected or missing patterns, or wrong order"
    )

    # * Convert to Dataset for RDM calculation
    accuracy_by_pattern_dataset = Dataset(
        # measurements=acc_by_pattern.values[:, None],
        measurements=acc_by_pattern.values.reshape(-1, 1),
        descriptors={"model": model_id},
        obs_descriptors={"patterns": PATTERNS},
    )

    # * Compute the RDM
    rdm_acc_by_pattern = calc_rdm(
        accuracy_by_pattern_dataset, method=dissimilarity_metric
    )
    # * Save the RDM
    fpath = RDM_DIR / f"(ANN)-({model_id})-(accuracy_by_pattern).hdf5"
    rdm_acc_by_pattern.save(fpath, file_type="hdf5", overwrite=True)

    # * Plot the RDM and save the figure
    tick_marks = np.arange(0, len(PATTERNS), 1)
    tick_labels = PATTERNS

    fig, ax = plt.subplots()
    im = ax.imshow(rdm_acc_by_pattern.get_matrices()[0])
    ax.set_title(f"RDM - Accuracy by Pattern\n{model_id}")
    fig.colorbar(im, ax=ax)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_labels, rotation=90)
    fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.show() if show_figs else None
    plt.close()

    # * ------------------------------------------------------------
    # mean_acc = acc_by_pattern.mean()
    # yticks = np.round(np.arange(0, 1.1, 0.2), 2)
    # fig, ax = plt.subplots()
    # acc_by_pattern.plot(kind="line", ax=ax)
    # ax.hlines(
    #     mean_acc,
    #     0,
    #     len(acc_by_pattern) - 1,
    #     color="red",
    #     ls="--",
    #     label="Mean",
    #     alpha=0.4,
    # )
    # ax.set_title(f"Accuracy by Pattern\n{model_id}")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.set_ylim(yticks[0], yticks[-1])
    # ax.set_yticks(yticks)
    # ax.set_yticklabels((yticks * 100).astype(int))
    # ax.grid(axis="y", alpha=0.4, ls="--", c="black")
    # # fpath = None
    # fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    # plt.show() if show_figs else None
    # plt.close()
    mean_acc = acc_by_pattern.mean()
    yticks = np.round(np.arange(0, 1.1, 0.2), 2)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=responses,
        x="pattern",
        y="correct",
        estimator="mean",
        errorbar=("ci", 95),
        n_boot=1000,
        ax=ax,
    )
    acc_by_pattern.plot(kind="line", ax=ax)
    ax.hlines(
        mean_acc,
        0,
        len(acc_by_pattern) - 1,
        color="red",
        ls="--",
        label="Mean",
        alpha=0.4,
    )
    ax.set_title(f"Accuracy by Pattern\n{model_id}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticks * 100).astype(int))
    ax.grid(axis="y", alpha=0.4, ls="--", c="black")

    # fpath = None
    fig.savefig(fpath.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.show() if show_figs else None
    plt.close()

    return rdm_acc_by_item, rdm_acc_by_pattern, acc_by_pattern, responses


def temp():
    pass
    # res_dir = DATA_DIR / "local_run/old"
    # act_files = [f for f in res_dir.glob("*layers_acts.pkl") if not f.stem.startswith(".")]

    # for act_file in act_files:

    #     acts = read_file(act_file)
    #     for trial_idx
    #     len(acts)


def rsa_between_models(rdm_files: List[Path], similarity_metric: str):
    # # ! TEMP
    # rdm_files = [
    #     f
    #     for f in (EXPORT_DIR / "RDMs").glob("*layers*.hdf5")
    #     if not f.name.startswith(".")
    # ]
    # similarity_metric = "cosine"
    # # ! TEMP
    rdm_files = sorted(rdm_files)
    model_ids = [
        re.findall(r"\((.+?)\)", f.stem)[1].replace("--", "/") for f in rdm_files
    ]

    models_rdms = {
        model_id: rsatoolbox.rdm.rdms.load_rdm(str(f))
        for model_id, f in zip(model_ids, rdm_files)
    }

    similarity_matrix = np.zeros((len(models_rdms), len(models_rdms)))

    for i, rdm1 in enumerate(models_rdms.values()):
        for j, rdm2 in enumerate(models_rdms.values()):
            similarity_matrix[i, j] = compare_rdm(
                rdm1, rdm2, method=similarity_metric
            ).item()

    # * Plot the similarity matrix
    models_legend = "Models:\n  "
    models_legend += "\n  ".join(
        [f"{i + 1}: {model_id}" for i, model_id in enumerate(model_ids)]
    )

    # * these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)

    models_legend_params = dict(
        x=1.32,
        y=1.0,
        s=models_legend,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
    )
    ticks = np.arange(0, len(models_rdms), 1)
    ticklabels = [f"{i + 1}" for i in ticks]
    fig, ax = plt.subplots()
    ax.imshow(similarity_matrix)
    ax.set_title(f"Similarity Matrix - Activations on Sequence Tokens")
    colorbar = ax.figure.colorbar(ax.imshow(similarity_matrix))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.text(**models_legend_params, transform=ax.transAxes)
    plt.show()


def performance_analysis(
    res_dir: Path, model_id: str, answer_regex: str, return_raw: bool = False
) -> Dict[str, Any]:
    # # ! TEMP
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    # model_id = "google--gemma-2-9b-it"
    # # ! TEMP

    responses = load_model_responses(res_dir, model_id)
    responses = clean_and_eval_model_responses(responses, answer_regex)

    model_id = model_id.replace("--", "/")
    responses["model_id"] = model_id
    responses["correct"] = responses["correct"].astype(int)

    overall_acc = responses["correct"].describe()
    overall_acc = pd.DataFrame(overall_acc).T
    overall_acc["model_id"] = responses["model_id"].iloc[0]
    overall_acc.reset_index(drop=True, inplace=True)

    acc_by_pattern = (
        responses.groupby("pattern")["correct"].describe().sort_index().reset_index()
    )
    acc_by_pattern["model_id"] = model_id

    assert all(acc_by_pattern["pattern"] == PATTERNS)

    res = dict(
        acc_by_pattern=acc_by_pattern,
        overall_acc=overall_acc,
    )

    if return_raw:
        res["raw_cleaned"] = responses

    return res


def performance_analysis_all(data_dir: Path, answer_regex: str):
    # ! TEMP
    data_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    answer_regex = r"Answer:\s?\n?(\w+)"
    # ! TEMP

    model_ids = [d.name for d in data_dir.iterdir() if d.is_dir()]

    _acc_by_pattern, _overall_acc, _raw_cleaned = [], [], []

    for model_id in model_ids:
        res = performance_analysis(data_dir, model_id, answer_regex, return_raw=True)
        _acc_by_pattern.append(res["acc_by_pattern"])
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


def plot_perf_analysis(fig_params: Optional[Dict[str, Any]] = None):
    # # ! TEMP
    data_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"

    ann_data = performance_analysis(
        data_dir,
        "meta-llama--Llama-3.3-70B-Instruct",
        r"Answer:\s?\n?(\w+)",
        return_raw=True,
    )

    model_id = "meta-llama--Llama-3.3-70B-Instruct"
    layer = "model.layers.79.post_attention_layernorm"

    # layer_acts = load_model_layer_acts(
    #     data_dir,
    #     model_id,
    #     layer,
    # )
    # ! TEMP

    if fig_params is None:
        fig_params = {
            "dpi": 300,
        }

    ann_data["raw_cleaned"].groupby("pattern")["correct"].describe()

    perf_res = performance_analysis_all(data_dir, ANSWER_REGEX)
    models_perf_df = perf_res["raw_cleaned"]
    models_perf_df.sort_values(["model_id", "pattern"], inplace=True)

    best_models = (
        models_perf_df.groupby("model_id")["correct"]
        .mean()
        .sort_values(ascending=False)
    )
    best_models = best_models[best_models > 0.7].index.tolist()

    # models_perf_df = pd.concat(models_perf_list)

    figs = []
    # * --------- FIGURE PARAMS ---------
    patterns_legend = "Patterns:\n  "
    patterns_legend += "\n  ".join(
        [f"{i + 1}: {' '.join(list(patt))}" for i, patt in enumerate(PATTERNS)]
    )
    # * these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)

    patterns_legend_params = dict(
        x=1.02,
        y=0.015,
        s=patterns_legend,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=props,
    )

    # *-----------------------
    # * ------- FIGURE -------
    xticks = np.arange(0, len(PATTERNS), 1)
    xticks_labels = [f"{i + 1}" for i in xticks]
    yticks_pct = np.round(np.arange(0, 1.1, 0.1), 2)
    ytick_labels_pct = (yticks_pct * 100).astype(int)

    fig, ax = plt.subplots(dpi=fig_params["dpi"])
    sns.lineplot(
        data=models_perf_df,
        x="pattern",
        y="correct",
        hue="model_id",
        errorbar=None,
        marker="o",
        ax=ax,
    )
    ax.legend(title="Models:", bbox_to_anchor=(1, 1))
    ax.set_title("Accuracy by Pattern")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.text(**patterns_legend_params, transform=ax.transAxes)
    # plt.tight_layout()
    # plt.show()
    figs.append(fig)
    # plt.close()

    # *-----------------------
    # * ------- FIGURE -------
    fig, ax = plt.subplots(dpi=fig_params["dpi"])
    sns.lineplot(
        data=models_perf_df, x="pattern", y="correct", errorbar=None, marker="o", ax=ax
    )
    sns.lineplot(
        data=models_perf_df.query("model_id in @best_models"),
        x="pattern",
        y="correct",
        errorbar=None,
        marker="o",
        ax=ax,
    )
    # ax.plot(
    #     range(len(c.PATTERNS)),
    #     [models_perf_df["correct"].mean()] * len(c.PATTERNS),
    #     ls="--",
    #     color="red",
    #     alpha=0.7,
    #     label="Mean",
    # )
    lines = ax.get_lines()
    line_colors = [l.get_color() for l in lines]
    ax.legend(
        handles=lines,
        # labels=["All Models", "Best Models", "Mean"],
        labels=["All Models", "Best Models"],
        title="Models:",
        bbox_to_anchor=(1, 1),
    )
    ax.set_title("Overall Accuracy by Pattern")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.text(**patterns_legend_params, transform=ax.transAxes)
    # plt.tight_layout()
    # plt.show()
    figs.append(fig)
    # plt.close()

    # *-----------------------
    # * ------- FIGURE -------
    fig, ax = plt.subplots(dpi=fig_params["dpi"])
    sns.barplot(
        data=models_perf_df,
        x="model_id",
        y="correct",
        hue="model_id",
        errorbar=None,
        ax=ax,
    )
    model_names = [
        f"{i + 1}: {m.get_text()}" for i, m in enumerate(ax.get_xticklabels())
    ]
    ax.set_xticklabels(range(1, len(model_names) + 1))
    # properties for the text box
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, ec="black")
    # Place the text box to the right of the Axes (x>1 means right of Axes)
    ax.text(
        x=1.02,
        y=1,
        s="\n".join(model_names),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=props,
    )
    ax.set_title("Accuracy by Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_yticks(yticks_pct)
    ax.set_yticklabels(ytick_labels_pct)
    ax.grid(axis="y", ls="--", alpha=0.4)
    figs.append(fig)
    # plt.close()

    return figs

    # *-----------------------
    # * ------- FIGURE -------

    pass


def stat_analysis_on_ann_acts(ann_responses, ann_activations):
    # # ! TEMP
    # model_id = "meta-llama--Llama-3.3-70B-Instruct"
    # layer = "model.layers.79.post_attention_layernorm"
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    # ann_responses = performance_analysis(res_dir, model_id, ANSWER_REGEX, True)
    # # ! TEMP

    avg_acts_per_pattern, rdm_acts = get_rdms_activations_on_seq_tokens(
        res_dir=res_dir,
        model_id=model_id,
        layer=layer,
        sequences_file=WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv",
        dissimilarity_metric="correlation",
        save_dir=None,
        answer_regex=r"Answer:\s?\n?(\w+)",
        show_figs=False,
    )


# * -----------------------------

if __name__ == "__main__":
    selected_layers = {
        "google--gemma-2-9b-it": "model.layers.41.post_attention_layernorm",
        "google--gemma-2-2b-it": "model.layers.25.post_attention_layernorm",
        "Qwen--Qwen2.5-7B-Instruct": "model.layers.27.post_attention_layernorm",
        "Qwen--Qwen2.5-72B-Instruct": "model.layers.79.post_attention_layernorm",
        "meta-llama--Llama-3.2-3B-Instruct": "model.layers.27.post_attention_layernorm",
        "meta-llama--Meta-Llama-3-8B-Instruct": "model.layers.31.post_attention_layernorm",
        "meta-llama--Llama-3.3-70B-Instruct": "model.layers.79.post_attention_layernorm",
    }

    res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)"
    model_ids = [d for d in res_dir.iterdir() if d.is_dir()]
    model_ids = [d.name for d in model_ids if d.name not in ["old", "new"]]

    sequences_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(7).csv"

    targ_tokens_file = (
        WD.parent
        / "experiment-ANNs/results/target_tokens/sequence_prompts-masked_idx(7)-sequence_tokens.csv"
    )

    prog_bar = tqdm(model_ids)
    responses_all = []

    for model_id in prog_bar:
        prog_bar.set_description(f"Processing {model_id}")

        # *
        rdm_acc_by_item, rdm_acc_by_pattern, acc_by_pattern, responses = (
            get_rdms_activations_on_accuracy(
                res_dir,
                model_id,
                sequences_file,
                dissimilarity_metric="euclidean",
                answer_regex=ANSWER_REGEX,
                show_figs=False,
            )
        )
        responses["model_id"] = model_id.replace("--", "/")
        responses_all.append(responses)

        # *
        dissimilarity_metric = "correlation"
        save_dir = EXPORT_DIR / f"analyzed/RSA-seq_tokens-metric_{dissimilarity_metric}"

        avg_acts_per_pattern, rdm_acts = get_rdms_activations_on_seq_tokens(
            res_dir,
            model_id,
            layer=selected_layers[model_id],
            sequences_file=sequences_file,
            dissimilarity_metric=dissimilarity_metric,
            save_dir=save_dir,
            answer_regex=ANSWER_REGEX,
            show_figs=False,
        )

    responses_all = pd.concat(responses_all)

    acc_by_pattern = (
        responses_all.groupby(["model_id", "pattern"])["correct"].mean().unstack()
    )
    acc_by_pattern["mean"] = acc_by_pattern.mean(axis=1)
    acc_by_pattern.sort_values("mean", ascending=False, inplace=True)

    acc_by_pattern_styled = apply_df_style(acc_by_pattern, 1, vmin=0, vmax=1)
    display(acc_by_pattern_styled)

    # for model_id in responses_all.model_id.unique():
    #     model_responses = responses_all.query(f"model_id == '{model_id}'")
    #     sns.lineplot(data=model_responses, x="pattern", y="correct")

    # * Plot Models' Accuracy by Pattern
    fig, ax = plt.subplots()
    responses_all.groupby(["model_id", "pattern"])["correct"].mean().unstack().T.plot(
        ax=ax
    )
    responses_all.groupby(["pattern"])["correct"].mean().plot(
        ax=ax, c="black", label="Mean", ls="--"
    )
    ax.legend(title="Models:", bbox_to_anchor=(1, 1))
    textstr = "Patterns:\n  "
    textstr += "\n  ".join([f"{i + 1}: {patt}" for i, patt in enumerate(PATTERNS)])
    # * these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white")  # , alpha=0.5)
    # * place a text box in upper left in axes coords
    ax.text(
        1.03,
        0.4,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xticklabels([i for i in range(len(ax.get_xticklabels()))])
    ax.grid(axis="y", ls="--")
