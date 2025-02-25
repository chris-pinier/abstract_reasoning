from pathlib import Path
import os
from transformers import AutoTokenizer
import pandas as pd
import re
from tqdm.auto import tqdm
from typing import Any, List

WD = Path(__file__).parent
# os.chdir(WD)
assert WD == Path.cwd()


def clean_tokens(text_tokens: List[str], tokenizer: AutoTokenizer):
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
    prefix: str, suffix: str, tokens: List[str], error_val: Any = None
):
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
    model_name: str, df_prompts: pd.DataFrame, df_sequences: pd.DataFrame
) -> List[tuple]:
    # * Load the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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


if __name__ == "__main__":
    # # * ################################################################################
    # # * Get token indices for the sequence elements/words in the prompts
    # # * ################################################################################
    # data_dir = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/ANNs/local_run")

    # model_names = [
    #     d.name.replace("--", "/").replace("-responses", "")
    #     for d in data_dir.glob("*") if d.is_dir()
    # ]


    # sequence_filenames = [
    #     "sessions-1_to_5-masked_idx(7).csv",
    #     "sessions-1_to_5-masked_idx(7)-unsorted.csv",
    #     # "sessions-1_to_5-masked_idx(mixed)-ascii_symbols.csv",
    #     "sessions-1_to_5-masked_idx(mixed)-scrambled.csv",
    #     "sessions-1_to_5-masked_idx(mixed).csv",
    # ]
    # prompts_filenames = [
    #     "sequence_prompts-masked_idx(7).csv",
    #     # "sequence_prompts-masked_idx(7)-unsorted.csv",
    #     # "sequence_prompts-masked_idx(mixed)-ascii_symbols.csv",
    #     "sequence_prompts-masked_idx(mixed)-scrambled.csv",
    #     "sequence_prompts-masked_idx(mixed).csv",
    # ]

    # files = zip(sequence_filenames, prompts_filenames)

    # for sequence_filename, prompts_filename in files:
    #     prompts_file = WD / f"sequence_prompts/{prompts_filename}"
    #     sequences_file = WD.parent / f"config/sequences/{sequence_filename}"

    #     df_prompts = pd.read_csv(prompts_file)
    #     df_sequences = pd.read_csv(sequences_file)

    #     seq_token_inds = pd.DataFrame()

    #     for model_name in tqdm(model_names):
    #         model_seq_token_inds = get_sequence_tokens(
    #             model_name, df_prompts, df_sequences
    #         )
    #         model_seq_token_inds = pd.DataFrame(
    #             model_seq_token_inds,
    #             columns=["item_id", "masked_idx", "idx_start", "idx_stop"],
    #         )
    #         model_seq_token_inds.insert(0, "model_id", model_name)
    #         seq_token_inds = pd.concat([seq_token_inds, model_seq_token_inds])

    #     seq_token_inds.reset_index(drop=True, inplace=True)

    #     fname = f"results/target_tokens/{prompts_file.stem}-sequence_tokens.csv"
    #     fpath = WD / fname
    #     seq_token_inds.to_csv(fpath, index=False)

    # * ################################################################################
    # *
    # * ################################################################################
