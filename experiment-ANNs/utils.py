import pandas as pd
from pathlib import Path
import os
from typing import Optional
import re
import numpy as np
from typing import Literal, Optional
import pickle
from tqdm.auto import tqdm


# * ------ General Purpose Functions ------
def prepare_prompt(
    row,
    base_prompt,
    seq_cols,
    choice_cols,
    unique_icons=None,
    prompt_format: Optional[str] = None,
):
    sequence = row.loc[seq_cols].tolist()
    # mapped_sequence = [str(unique_icons[icon]) for icon in sequence]

    choices = row.loc[choice_cols].tolist()
    # solution = sequence[row.loc["maskedImgIdx"]]
    masked_idx = row.loc["masked_idx"]
    sequence[masked_idx] = "?"
    # mapped_sequence[row.loc["maskedImgIdx"]] = "?"

    sequence_string = "Sequence: " + " ".join(sequence)
    # choices_string = "Choices: " + " ".join(choices)
    choices_string = "Options: " + " ".join(choices)
    sequence_prompt = f"\n{sequence_string}\n{choices_string}"
    sequence_prompt = base_prompt + sequence_prompt

    if prompt_format is not None:
        sequence_prompt = prompt_format.replace("{prompt}", sequence_prompt)

    return sequence_prompt  # , solution


def generate_sequence_prompts(
    sequences: pd.DataFrame,
    base_prompt: str,
):
    seq_cols = [c for c in sequences.columns if re.search(r"figure\d", c)]
    choice_cols = [c for c in sequences.columns if re.search(r"choice\d", c)]

    prompts = []
    seq_copy = sequences.copy()

    for idx, row in sequences.iterrows():
        prompt = prepare_prompt(
            row,
            base_prompt,
            seq_cols,
            choice_cols,
            prompt_format=None,
        )
        prompts.append(prompt)

    sequences_info = seq_copy[["item_id", "pattern", "masked_idx", "solution"]]

    prompts_df = pd.DataFrame(prompts, columns=["prompt"])
    prompts_df = pd.concat([sequences_info, prompts_df], axis=1)
    prompts_df.reset_index(drop=True, inplace=True)

    return prompts_df


def reformat_act_files(res_dir: Path, verbose=True, delete_original=True):
    act_files = sorted(res_dir.glob("layers_acts*.pkl"))

    acts = []

    for act_f in act_files:
        with open(act_f, "rb") as f:
            acts.append(pickle.load(f))

    acts_by_layer: dict = {l: [] for l in acts[0].keys()}

    # print("Loading models' activations...", end=" ") if verbose else None
    for act in acts:
        for l, v in act.items():
            acts_by_layer[l].append(v)
    # print("Done!") if verbose else None

    # print("Saving activations by layer...", end=" ") if verbose else None
    for layer, acts in tqdm(acts_by_layer.items()):
        with open(res_dir / f"acts_by_layer-{layer}.pkl", "wb") as f:
            pickle.dump(acts, f)
    # print("Done !") if verbose else None

    if delete_original:
        [f.unlink() for f in act_files]

# # * ------ Other Functions ------


def generate_scrambled_and_symbols_sequences(rand_seed=42):
    # * ------ Generate Sequences (and Prompts) with Mixed Masked Indicies  ------
    seq_fpath = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(mixed).csv"
    sequences = pd.read_csv(seq_fpath)

    sequences_sample_fpath = (
        WD / "config/sequences/sessions-1_to_5-masked_idx(mixed)-sample.csv"
    )
    sequences_sample = pd.read_csv(sequences_sample_fpath)

    selected_items = []

    rng = np.random.default_rng(seed=rand_seed)

    for pattern in sorted(sequences["pattern"].unique()):
        temp = sequences.query("pattern == @pattern")
        selected = rng.choice(temp["item_id"].unique(), 4, replace=False)
        selected_items.extend(selected)

    sequences_sample = sequences.query("item_id in @selected_items")
    sequences_sample.reset_index(drop=True, inplace=True)

    # sequences_sample.value_counts("pattern").unique()
    # sequences_sample.value_counts("item_id").unique()
    # sequences_sample.value_counts(["item_id", "pattern"]).unique()
    # sequences_sample.groupby(["masked_idx"])["item_id"].count().unique()

    with open(WD / "config/instructions.txt", "r") as f:
        instructions = f.read()

    seq_prompts = generate_sequence_prompts(sequences_sample, instructions)
    seq_prompt_fpath = WD / "sequence_prompts/sequence_prompts-masked_idx(mixed).csv"
    seq_prompts.to_csv(seq_prompt_fpath, index=False)

    # * ------ Generate Scramble Sequences and Prompts ------
    # seq_file = WD.parent / "config/sequences/sessions-1_to_5-masked_idx(mixed).csv"
    # sequences = pd.read_csv(seq_file)

    # * generate scrambled sequences
    def shuffle(i):
        # * Create a new shuffled string
        new = "".join(np.random.choice(list(i), len(i), replace=False))

        # *  If the new string is the same as the original, try again
        if new == i:
            return shuffle(i)
        return new

    seq_cols = [c for c in sequences_sample.columns if re.search(r"figure\d", c)]
    choice_cols = [c for c in sequences_sample.columns if re.search(r"choice\d", c)]

    original_icons = np.unique(
        sequences_sample.loc[:, seq_cols + choice_cols].values.flatten()
    )

    icon_mapping = {i: shuffle(i) for i in original_icons}
    scrambled_sequences = sequences_sample.copy()
    scrambled_sequences.replace(icon_mapping, inplace=True)

    scrambled_sequences = scrambled_sequences.query("item_id in @selected_items")
    scrambled_sequences.reset_index(drop=True, inplace=True)

    scrambled_sequences_fpath = seq_fpath.parent / f"{seq_fpath.stem}-scrambled.csv"

    scrambled_sequences.to_csv(scrambled_sequences_fpath, index=False)

    icon_mapping = pd.DataFrame(icon_mapping.items(), columns=["original", "scrambled"])

    icon_mapping_fpath = (
        seq_fpath.parent / f"{seq_fpath.stem}-scrambled-icon_mapping.csv"
    )
    icon_mapping.to_csv(icon_mapping_fpath, index=False)

    # * generate prompts for scrambled sequences
    # seq_file = (
    #     WD.parent / "config/sequences/sessions-1_to_5-masked_idx(mixed)-scrambled.csv"
    # )
    # sequences = pd.read_csv(seq_file)

    # with open(WD / "config/instructions.txt", "r") as f:
    #     instructions = f.read()

    scrambled_seq_prompts = generate_sequence_prompts(scrambled_sequences, instructions)
    # # print(seq_prompts.loc[0, "prompt"])
    scrambled_sequences_fpath.name
    scrambled_seq_prompts_fpath = (
        seq_prompt_fpath.parent / f"{seq_prompt_fpath.stem}-scrambled.csv"
    )
    scrambled_seq_prompts.to_csv(scrambled_seq_prompts_fpath, index=False)

    # * ------ Generate uncommon ASCII symbols sequences and Prompts ------

    # Extended ASCII symbols and their descriptions
    symbols = [
        (950, "ζ"),
        (1141, "ѵ"),
        (535, "ȗ"),
        (1386, "ժ"),
        (221, "Ý"),
        (1333, "Ե"),
        (1010, "ϲ"),
        (326, "ņ"),
        (663, "ʗ"),
        (1398, "ն"),
        (1407, "տ"),
        (1992, "߈"),
        (282, "Ě"),
        (547, "ȣ"),
        (350, "Ş"),
        (1138, "Ѳ"),
        (1498, "ך"),
        (330, "Ŋ"),
        (1097, "щ"),
        (949, "ε"),
        (1355, "Ջ"),
        (1869, "ݍ"),
        (316, "ļ"),
        (1402, "պ"),
        (1818, "ܚ"),
        (355, "ţ"),
        (468, "ǔ"),
        (1603, "ك"),
    ]

    scrambled_sequences_fpath = (
        WD.parent / "config/sequences/sessions-1_to_5-masked_idx(mixed)-scrambled.csv"
    )
    scrambled_sequences = pd.read_csv(scrambled_sequences_fpath)

    seq_cols = [c for c in scrambled_sequences.columns if re.search(r"figure\d", c)]
    choice_cols = [c for c in scrambled_sequences.columns if re.search(r"choice\d", c)]

    original_icons = np.unique(
        scrambled_sequences.loc[:, seq_cols + choice_cols].values.flatten()
    )

    chars = np.array([symb for idx, symb in symbols])

    np.random.seed(1)
    chars = np.random.choice(chars, len(original_icons), replace=False)

    icon_mapping = dict(zip(original_icons, [str(c) for c in chars]))

    symbol_sqs = scrambled_sequences.copy().replace(icon_mapping)

    icon_mapping = pd.DataFrame(icon_mapping.items(), columns=["original", "new"])
    icon_mapping_fpath = (
        seq_fpath.parent / f"{seq_fpath.stem}-ascii_symbols-icon_mapping.csv"
    )
    icon_mapping.to_csv(icon_mapping_fpath, index=False)

    with open(WD / "config/instructions.txt", "r") as f:
        instructions = f.read()

    unused_symbols = [
        symbol for _, symbol in symbols if symbol not in icon_mapping["new"]
    ]

    for word, symbol in list(zip(["smile", "eye", "camera", "bone"], unused_symbols)):
        instructions = instructions.replace(word, symbol)

    instructions = instructions.replace("word", "symbol")

    symbol_sqs_prompts = generate_sequence_prompts(symbol_sqs, instructions)

    symbol_sqs_prompts_fpath = (
        WD / "sequence_prompts/sequence_prompts-masked_idx(mixed)-ascii_symbols.csv"
    )
    symbol_sqs_prompts.to_csv(symbol_sqs_prompts_fpath, index=False)


def read_file_content(
    filepath: Path, mode: Literal["rt", "rb"] = "rt", encoding: Optional[str] = None
):
    """Reads the content of a file.

    Args:
        filepath: The path to the file.
        mode: The mode to open the file in ("r" for text, "rb" for binary). Defaults to "r".
        encoding: The encoding to use if mode is "r". Ignored if mode is "rb".

    Returns:
        The file content as a string (if mode is "r") or bytes (if mode is "rb").
        Returns None if file not found or an error occurs.
    """
    if mode == "rt" and encoding is None:
        encoding = "utf-8"

    if mode == "rb" and encoding is not None:
        raise ValueError("encoding must be None when mode is 'rb'")

    try:
        with open(
            filepath, mode, encoding=encoding if mode == "r" else None
        ) as f:  # Conditional encoding
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        return None


if __name__ == "__main__":
    import re

    RAND_SEED = 0

    WD = Path(__file__).parent
    # os.chdir(WD)
    assert WD == Path.cwd()

    sequences_dir = WD.parent / "config/sequences"

    seq_fpaths = sorted(
        [f for f in sequences_dir.glob("*") if re.search(r"session_\d\.csv", f.name)]
    )

    sequences = pd.concat([pd.read_csv(fpath) for fpath in seq_fpaths])
    sequences.reset_index(drop=True, inplace=True)

    instructions = read_file_content(WD / "config/instructions.txt")

    sequence_prompts = generate_sequence_prompts(
        sequences,
        base_prompt=instructions,
    )

    masked_idx = sequence_prompts["masked_idx"].unique().item()
    fname = f"sequence_prompts-masked_idx({masked_idx}).csv"
    fpath = WD / f"sequence_prompts/{fname}"

    sequence_prompts.to_csv(fpath, index=False)

    # print(sequence_prompts.loc[0, "prompt"])

