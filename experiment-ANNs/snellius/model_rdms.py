import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from transformers import AutoModel, AutoTokenizer
import re
from .. import (
    load_model_layer_acts,
    load_model_responses,
    load_model_tokens,
    load_model_run_info,
    clean_and_eval_model_responses,
    clean_tokens,
)

PATTERNS = None
ITEM_ID_SORT = None


def reorder_item_ids(
    original_order_df: pd.DataFrame, new_order_df: pd.DataFrame
) -> np.ndarray:
    """TODO:_summary_

    Args:
        original_df (pd.DataFrame): DataFrame containing the columns: item_id, pattern
        new_df (pd.DataFrame): DataFrame containing the columns: item_id, pattern

    Returns:
        np.ndarray: reordered indices
    """

    original_order_df = original_order_df[["pattern", "item_id"]].reset_index(
        names=["original_order"]
    )

    new_order_df = new_order_df[["pattern", "item_id"]].reset_index(names=["new_order"])

    new_order_df = original_order_df.merge(
        new_order_df, on=["pattern", "item_id"], how="left"
    )

    new_order_df = new_order_df[["original_order", "new_order"]].sort_values(
        "new_order"
    )

    reordered_inds = new_order_df["original_order"].values

    return reordered_inds


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
