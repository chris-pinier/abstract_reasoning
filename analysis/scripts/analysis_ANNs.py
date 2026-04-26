# * ########################################
# * IMPORTS
# * ########################################
from pathlib import Path
import os
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm, compare_cosine, compare as compare_rdm
from typing import Dict, Final, List, Tuple, Any, Literal, Optional
from tqdm.auto import tqdm
import seaborn as sns
from transformers import AutoTokenizer
import sys

WD = Path(__file__).parent
sys.path.append(WD)
os.chdir(WD)
assert WD == Path.cwd()

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from utils.analysis_utils import (
    apply_df_style,
    reorder_item_ids,
    get_timestamp,
)
from analysis_conf import Config as c
from analysis_plotting import plot_rdm
from data_loader.ann_data import ANNDataClass, ANNSubjData, ANNGroupData


# * ########################################
# * GLOBAL VARIABLES
# * ########################################
PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER
ANSWER_REGEX = c.ANSWER_REGEX
ITEM_ID_SORT = c.ITEM_ID_SORT


# * -----------------------------
# * -----------------------------
def main_new():
    timestamp = get_timestamp()

    res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"

    model_ids = [
        d for d in res_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    model_ids = [d.name for d in model_ids]

    dissimilarity_metric = "correlation"
    similarity_metric = "corr"

    save_dir = (
        EXPORT_DIR
        / f"analyzed/{timestamp}/RSA-seq_tokens-metric_{dissimilarity_metric}"
    )

    all_rdms_sequence_lvl = {}
    all_rdms_pattern_lvl = {}
    all_rdm_comparisons = {}


def main():
    res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"

    model_ids = [
        d for d in res_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    model_ids = [d.name for d in model_ids]

    dissimilarity_metric = "correlation"
    similarity_metric = "corr"

    save_dir = EXPORT_DIR / f"analyzed/RSA-seq_tokens-metric_{dissimilarity_metric}"

    all_rdms_sequence_lvl = {}
    all_rdms_pattern_lvl = {}
    all_rdm_comparisons = {}

    for model_id in tqdm(model_ids):
        rdm_sequence_lvl, rdm_pattern_lvl, rdms_sequence_lvl, rdm_comparison = (
            get_rdms_on_all_layers(
                res_dir,
                model_id,
                dissimilarity_metric,
                similarity_metric,
                save_dir,
                # show_figs: bool = False,
            )
        )
        # dissimilarity_metric = "correlation"
        # dissimilarity_metric = "corr"
        # save_dir = EXPORT_DIR / f"analyzed/RSA-seq_tokens-metric_{dissimilarity_metric}"
        # show_figs = False

        all_rdms_sequence_lvl[model_id] = rdm_sequence_lvl
        all_rdms_pattern_lvl[model_id] = rdm_pattern_lvl
        all_rdm_comparisons[model_id] = rdm_comparison

    all_rdm_comparisons = {
        k: all_rdm_comparisons[k] for k in sorted(all_rdm_comparisons.keys())
    }

    # save_dir / all_rdm_comparisons

    all_rdm_comparison_df = pd.DataFrame(columns=["id", "layer", "similarity"])
    # all_rdm_comparison_df["id"] = []
    for id in all_rdm_comparisons.keys():
        for i, layer_sim in enumerate(all_rdm_comparisons[id]):
            all_rdm_comparison_df = pd.concat(
                [
                    all_rdm_comparison_df,
                    pd.DataFrame(
                        [[id, i, layer_sim]], columns=["id", "layer", "similarity"]
                    ),
                ]
            )

    all_rdm_comparison_df["id"] = all_rdm_comparison_df["id"].str.replace(
        "--", "/", regex=False
    )
    all_rdm_comparison_df.reset_index(drop=True, inplace=True)
    all_rdm_comparison_df.to_csv(save_dir / "layer-wise-similarity.csv", index=False)

    # TODO: clean up the mess below and incorporate it into the above code
    # * ----------------------------------------
    # * Save best layer information
    df = all_rdm_comparison_df.copy()
    df.reset_index(drop=True, inplace=True)
    n_layers = df.groupby("id")["layer"].last().rename("n_layers") + 1
    df = df.merge(n_layers, on="id")
    df["layer_depth"] = round((df["layer"] + 1) / df["n_layers"], 3)
    best_layer_df = (
        df.sort_values(["id", "similarity"], ascending=[True, False])
        .groupby("id")
        .first()
    )
    best_layer_df = best_layer_df[
        ["similarity", "n_layers", "layer", "layer_depth"]
    ].reset_index()
    best_layer_df.to_csv(save_dir / "layer-wise-similarity-best.csv")

    best_layer_df["id"] = (
        best_layer_df["id"]
        .str.replace(r".+\/|-(?:it|instruct)", "", regex=True, flags=re.IGNORECASE)
        .str.title()
    )

    fig, ax = plt.subplots(dpi=500)
    # for model_id, comp_data in all_rdm_comparisons.items():
    #     ax.plot(comp_data, marker="o", label=model_id)

    sns.barplot(
        data=best_layer_df,
        x="layer_depth",
        y="id",
        hue="id",
        hue_order=ANN_ID_ORDER,
        order=ANN_ID_ORDER,
        ax=ax,
    )
    ax.set_xlabel("Layer Depth (normalized)")
    ax.set_ylabel("")
    ax.grid(axis="x", ls="--")
    ax.vlines(
        best_layer_df["layer_depth"].mean(), *ax.get_ylim(), color="k", ls="--", lw=1
    )
    fig.savefig(save_dir / "best_layer_depth.png", dpi=500, bbox_inches="tight")
    plt.show()
    # * ----------------------------------------

    all_rdm_comparison_df["id"] = (
        all_rdm_comparison_df["id"]
        .str.replace(r".+\/|-(?:it|instruct)", "", regex=True, flags=re.IGNORECASE)
        .str.title()
    )

    # TODO: remove the following and use the df above instead
    # * Normalize number of layers
    all_rdm_comparison_df["layer"] = all_rdm_comparison_df["layer"].astype("float")
    for ann_id in all_rdm_comparison_df["id"].unique():
        _df = all_rdm_comparison_df.query("id == @ann_id")
        ind = _df.index
        layer_vals = _df["layer"] / _df["layer"].max()
        all_rdm_comparison_df.loc[ind, "layer"] = layer_vals

    fig, ax = plt.subplots(dpi=500)
    # for model_id, comp_data in all_rdm_comparisons.items():
    #     ax.plot(comp_data, marker="o", label=model_id)

    sns.lineplot(
        data=all_rdm_comparison_df,
        x="layer",
        y="similarity",
        hue="id",
        hue_order=ANN_ID_ORDER,
        # errorbar=("ci", 95),
        # n_boot=1000,
        # marker="o",
        ax=ax,
    )
    # ax.set_title("Layer-wise Similarity with Reference RDM")
    ax.set_xlabel("Layer Depth (normalized)")
    ax.set_ylabel("Correlation")
    ax.legend(title="")
    ax.grid(True, ls="--")
    ax.legend(bbox_to_anchor=(1, 1), title="")
    fig.savefig(save_dir / "layer-wise-similarity.png", dpi=500, bbox_inches="tight")
    plt.show()

    # rsa_res = []
    # similarity_metric = "corr"
    # for model_1, model_2 in combinations(rdms_sequence_lvl.keys(), r=2):
    #     res_sequence_lvl = compare_rdm(
    #         rdms_sequence_lvl[model_1], rdms_sequence_lvl[model_2], similarity_metric
    #     ).item()
    #     res_pattern_lvl = compare_rdm(
    #         rdms_pattern_lvl[model_1], rdms_pattern_lvl[model_2], similarity_metric
    #     ).item()

    #     rsa_res.append([model_1, model_2, "sequence", res_sequence_lvl])
    #     rsa_res.append([model_1, model_2, "pattern", res_pattern_lvl])

    # df_rsa = pd.DataFrame(rsa_res, columns=["model_1", "model_2", "level", "res"])

    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
    # # model_id = "deepseek-ai--DeepSeek-R1-Distill-Llama-70B"
    # model_id = "meta-llama--Llama-3.2-3B-Instruct"
    # dissimilarity_metric = "correlation"
    # # save_dir = EXPORT_DIR / f"analyzed/RSA-seq_tokens-metric_{dissimilarity_metric}"
    # save_dir = None
    # show_figs = False


# ! ###########################################################
# ! ########################### OLD ###########################
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

    prefix = prefix.lower().strip().replace("\n", " ").replace(" ", "")
    suffix = suffix.lower().strip().replace("\n", " ").replace(" ", "")

    # * Find start and stop indices in the reconstructed text
    idx_start, idx_stop = None, None
    reconstructed_txt = tokens_list_to_string(tokens, sep="")

    # * Find the start index in the reconstructed text
    found_start = reconstructed_txt.find(prefix)
    if found_start != -1:
        idx_start = found_start + len(prefix)

    # # * Find the stop index in the reconstructed text
    found_stop = reconstructed_txt.find(suffix)
    if found_stop != -1:
        idx_stop = found_stop

    # * Find start and stop indices in the tokens list
    tok_idx_start = None
    tok_idx_stop = None

    # * Find start index in the token list
    for i in range(len(tokens)):
        if tokens_list_to_string(tokens[:i]) == reconstructed_txt[:idx_start]:
            tok_idx_start = i
            break

    # * Find stop index in the token list
    # * when i is 0, tokens[-0:] is equivalent to the entire list, so start at 1
    for i in range(1, len(tokens) + 1):
        if tokens_list_to_string(tokens[-i:]) == reconstructed_txt[idx_stop:]:
            tok_idx_stop = len(tokens) - i
            break

    indices = (tok_idx_start, tok_idx_stop)

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
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-all_tokens_acts"
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
        # response_acts = np.concatenate(response_acts, axis=1)

    assert sorted(set(responses["pattern"])) == PATTERNS, (
        "Unexpected or missing patterns"
    )

    # * Flatten layers activations across tokens into a new array of shape (n_items, n_tokens * n_units)
    flattened_layers_acts = np.concatenate(sequences_acts).reshape(
        len(sequences_acts), -1
    )

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
        k: np.concatenate(v).mean(axis=0) for k, v in avg_acts_per_pattern.items()
    }

    del layer_acts, sequences_acts

    # * Reorder the patterns, just in case
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


def get_rdms_activations_on_accuracy(
    res_dir: Path,
    model_id: str,
    sequences_file: Path,
    dissimilarity_metric: str = "euclidean",
    answer_regex: str = r"Answer:\s?\n?(\w+)",
    show_figs: bool = False,
):
    # # ! TEMP
    # res_dir = DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"
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
    order_by_pattern = np.concatenate([order_by_pattern[k] for k in PATTERNS])
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
    df = pd.read_csv(
        "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/ANNs/analyzed/RSA-seq_tokens-metric_correlation/layer-wise-similarity.csv"
    )

    df.reset_index(drop=True, inplace=True)
    n_layers = df.groupby("id")["layer"].last().rename("n_layers") + 1
    df = df.merge(n_layers, on="id")
    df["layer_depth"] = round((df["layer"] + 1) / df["n_layers"], 3)
    best_layer_df = (
        df.sort_values(["id", "similarity"], ascending=[True, False])
        .groupby("id")
        .first()
    )
    best_layer_df = best_layer_df[["similarity", "n_layers", "layer", "layer_depth"]]
    best_layer_df.reset_index(inplace=True)

    perf_res = perf_analysis_all_anns(
        DATA_DIR / "local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
        ANSWER_REGEX,
    )
    overall_acc = perf_res["overall_acc"]

    overall_acc = overall_acc[["model_id", "mean", "std"]].rename(
        columns={"mean": "acc_mean", "std": "acc_std", "model_id": "id"}
    )
    overall_acc["id"] = (
        overall_acc["id"]
        .str.replace(r".+\/|-(?:it|instruct)", "", regex=True, flags=re.IGNORECASE)
        .str.title()
    )
    best_layer_df["id"] = (
        best_layer_df["id"]
        .str.replace(r".+\/|-(?:it|instruct)", "", regex=True, flags=re.IGNORECASE)
        .str.title()
    )
    overall_acc = overall_acc.merge(
        best_layer_df[["id", "layer_depth", "similarity"]], on="id"
    )
    overall_acc["acc_mean"].corr(overall_acc["similarity"])

    sns.scatterplot(
        data=overall_acc, x="acc_mean", y="similarity", hue="id", hue_order=ANN_ID_ORDER
    )

    print(overall_acc.to_string(index=False))

    # * ----------------------------------------
    # * ----------------------------------------
    from box import Box

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

    print("All data")
    for name, df in dfs.items():
        print(f"\t{name:<5}: {df['corr'].mean():.2f} | {df['p_val'].mean():.2f}")

    print(f"\n{'-' * 30}\n")

    print("Human Average")
    for name, df in dfs.items():
        _df = df.query("id2 == 'human_avg'")
        print(f"\t{name:<5}: {_df['corr'].mean():.2f} | {_df['p_val'].mean():.2f}")

    print(f"\n{'-' * 30}\n")

    print("Subjects")
    for name, df in dfs.items():
        _df = df.query("id2 != 'human_avg'")
        print(f"\t{name:<5}: {_df['corr'].mean():.2f} | {_df['p_val'].mean():.2f}")

    for name, df in dfs.items():
        df["type"] = name

    df = pd.concat([df for df in dfs.values()])
    df_corr = df.query("id2 == 'human_avg'").groupby("type")["corr"].describe()
    df_p_val = df.query("id2 == 'human_avg'").groupby("type")["p_val"].describe()

    filtered_cols = ["type", "mean", "std", "min", "max"]
    df_corr = df_corr.round(2).reset_index(drop=False)[filtered_cols]
    df_p_val = df_p_val.round(2).reset_index(drop=False)[filtered_cols]

    df_corr = df_corr.astype("str")
    df_p_val = df_p_val.astype("str")
    df_summary = pd.DataFrame(columns=filtered_cols)
    for col in filtered_cols:
        if col == "type":
            df_summary[col] = df_corr[col]
        else:
            df_summary[col] = df_corr[col] + " (" + df_p_val[col] + ")"

    # print(df_summmary.to_string(index=False))
    print(df_summary.to_latex(index=False))


def main_1():
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

        # ! TEMP
        save_dir = Path(str(save_dir) + "-TEST")
        # ! TEMP

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


if __name__ == "__main__":
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
    pass
