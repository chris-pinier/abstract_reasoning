from pathlib import Path
from PIL import Image, ImageDraw
from screeninfo import get_monitors
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from icecream import ic
import sqlite3
from typing import Union
import shutil
import re
from functools import wraps
import time

# from typing import Any, Callable


def invert_dict(d: dict):
    return {v: k for k, v in d.items()}


def get_timestamp(fmt="%Y_%m_%d-%H_%M_%S"):
    return time.strftime(fmt, time.localtime())


def disable_decorator(disable=False, message=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if disable:
                # internal_message = (
                #     f"{func.__name__} execution has been disabled."
                #     if message is None
                #     else message
                # )
                # print(internal_message)
                if message is not None:
                    print(message)
                return None
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def invert_dict(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def autoconfing_exp():
    config = {
        "feedback": None,
        "stim_folder": None,
        "resolution": None,
        "monitor_dims_mm": None,
        "img_dims_px": None,
        "intertrial_interval": None,
    }

    monitors = get_monitors_info()
    if len(monitors) == 1:
        monitor = monitors[0]


def get_pixel_counts(images: Dict[str, Image.Image]):
    # images = {k: Image.open(v) for k, v in imgs_dict.items()}

    pixel_counts = {}
    for img_name, img in images.items():
        arr = np.asarray(img)
        converted = np.where(arr > 0, 255, 0)
        unique, counts = np.unique(converted, return_counts=True)
        pixel_counts[img_name] = dict(zip(unique, counts))

    df = pd.DataFrame(pixel_counts).T
    df.rename(columns={0: "white", 255: "black"}, inplace=True)
    df.drop(columns=["white"], inplace=True)

    df["diff_ratio"] = round(df["black"] / df["black"].max(), 3)

    df.sort_values("diff_ratio", ascending=False, inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "img_name"}, inplace=True)

    df["rank"] = df["diff_ratio"].rank(method="dense", ascending=False)

    df.groupby("rank")["black"].mean()

    # converted[:, :, 0][np.where(converted[:, :, 3] == 255)] = 255
    # converted = converted.astype(np.uint8)
    # converted = Image.fromarray(converted)

    return df


def get_monitors_info():
    current_monitors = [
        {
            "name": monitor.name if monitor.name else f"Monitor{i+1}",
            "res": (monitor.width, monitor.height),
            "width": monitor.width,
            "height": monitor.height,
            "width_mm": monitor.width_mm,
            "height_mm": monitor.height_mm,
            "primary": monitor.is_primary,
        }
        for i, monitor in enumerate(get_monitors())
    ]

    return current_monitors


def convert_to_black(img, img_name: str = None, save_path: str = None):
    # img = imgs_list[10]
    arr = np.array(img)

    # * Convert to purely black and white
    converted = np.where(arr > 0, 255, 0)

    # convert all black pixels to red
    # converted[:, :, 0][np.where(converted[:, :, 3] == 255)] = 255

    converted = converted.astype(np.uint8)
    converted = Image.fromarray(converted)

    np.unique(np.asarray(converted), return_counts=True)

    if save_path:
        save_path = Path(save_path)
        if img_name is None:
            img_name = np.random.randint(0, 1_000_000)
            img_name = f"img_{str(img_name).zfill(7)}.png"
            if img_name not in save_path.iterdir():
                plt.imshow(converted)
                plt.axis("off")
                plt.savefig(save_path / img_name, bbox_inches="tight", pad_inches=0)
            else:
                raise ValueError(f"{img_name} already exists in {save_path}")
    # np.unique(converted, return_counts=True)


def sess_prep(
    images: Dict[str, str],
    icons: List[str],
    sequences: pd.DataFrame,
    allowed_keys: List[str],
    window_size: List[int],
    block_size: int,
) -> Tuple[Dict, Dict]:

    assert (
        len(set([Image.open(im).size for im in images.values()])) == 1
    ), "Images must have same size"

    # solution_mask = "question-mark"

    img_size = Image.open(images[icons[0]]).size
    x_positions = {}
    resp_mapping = {}

    match_cols_choice = lambda x: re.compile("choice\d{1,2}", re.IGNORECASE).search(x)
    match_cols_seq = lambda x: re.compile("figure\d{1,2}", re.IGNORECASE).search(x)

    seq_cols = [c for c in sequences.columns if match_cols_seq(c)]
    choice_cols = [c for c in sequences.columns if match_cols_choice(c)]

    x_pos_seq = {}
    # new_sequences = sequences.copy()

    for seq_length in range(1, len(seq_cols) + 1):  # * +1 for 0 index
        # * Calculate the total width of images & blank spaces for the item set
        total_width = img_size[0] * seq_length
        empty_space_width = window_size[0] - total_width
        sep_space_width = empty_space_width / (seq_length + 1)
        x_shift = img_size[0] + sep_space_width
        start_pos = sep_space_width + img_size[0] / 2

        positions = [
            start_pos + (i * x_shift) - window_size[0] / 2 for i in range(seq_length)
        ]
        x_pos_seq[seq_length] = {}
        x_pos_seq[seq_length]["pos"] = [round(pos, 3) for pos in positions]
        x_pos_seq[seq_length]["sep_space_width"] = round(sep_space_width, 3)

    # for idx_row, row in tqdm(new_sequences.iterrows()):
    for idx_row, row in tqdm(sequences.iterrows()):
        x_positions[idx_row] = {}

        avail_choices = row.loc[choice_cols].dropna().tolist()
        sequence = row.loc[seq_cols].dropna().tolist()

        # * replace solution with the mask
        # new_sequences.loc[idx_row, seq_cols[row["masked_idx"]]] = solution_mask

        # * Calculate the total width of images & blank spaces for the item set
        pos_info = x_pos_seq[len(sequence)]
        x_positions[idx_row]["items_set"] = pos_info["pos"]
        sep_space_width = x_pos_seq[len(sequence)]["sep_space_width"]

        # * Calculate the total width of images & blank spaces for the available choices
        # * keep same sep_space_width as for the item set => number of choices must be
        # * <= number of items
        x_shift = img_size[0] + sep_space_width
        total_width = x_shift * len(avail_choices)
        empty_space_width = window_size[0] - total_width
        start_pos = empty_space_width / 2 + x_shift / 2

        x_positions[idx_row]["avail_choice"] = [
            start_pos + (i * x_shift) - window_size[0] / 2
            for i in range(len(avail_choices))
        ]

        resp_mapping[idx_row] = {k: v for k, v in (zip(allowed_keys, avail_choices))}

    if (remainder := len(sequences) % block_size) != 0:
        n_blocks = (len(sequences) - remainder) / block_size
        blocks = np.array_split(sequences[:-remainder], n_blocks)
        block_trials = sequences.iloc[-remainder:]
        blocks.append(block_trials)
        print(f"WARNING: uneven block sizes: {[len(b) for b in blocks]}")

    trial_blocks = []

    for block in blocks:
        trials = []
        for idx_row, row in block.iterrows():
            trial = row.to_dict()
            trial.update(
                {
                    "item_id": row["itemid"],
                    "x_pos": x_positions[idx_row],
                    "resp_map": resp_mapping[idx_row],
                    "trial_type": "",
                }
            )
            trial["seq_order"] = [int(i) for i in trial["seq_order"] if i.isdigit()]
            trial["choice_order"] = [
                int(i) for i in trial["choice_order"] if i.isdigit()
            ]

            trials.append(trial)
        trial_blocks.append(trials)

    trial_blocks = (block for block in trial_blocks)

    return trial_blocks


def load_stim_combs(n: int = None, inds: np.array = None, pb=False) -> pd.DataFrame:
    if not ((n is None) ^ (inds is None)):
        raise ValueError("Either `n` or `inds` must be provided (not both).")

    # TODO: load from sqlite

    ranges = pd.read_csv("config/stimuli_indices_ranges.csv", index_col=0)
    n_combs = ranges["end"].max()

    if inds is None:
        inds = np.sort(np.random.choice(n_combs, n, replace=False))
    else:
        inds = np.sort(inds)

    start_end = zip(ranges["start"], ranges["end"])

    stim_combs_csv = [f for f in Path("config").glob("stimuli_combinations_*.csv")]
    stim_combs_csv = {int(f.stem.split("_")[-1]): f for f in stim_combs_csv}

    df_combs = pd.DataFrame()

    for idx in tqdm(range(len(ranges)), disable=not pb):
        start, end = next(start_end)
        batch_inds = inds[np.where(np.logical_and(inds >= start, inds <= end))]

        if batch_inds.size > 0:
            tmp = pd.read_csv(stim_combs_csv[idx + 1], index_col=0).loc[batch_inds]
            df_combs = pd.concat([df_combs, tmp], axis=0)

    df_combs = df_combs.sample(frac=1, random_state=0, replace=False)

    return df_combs


def load_stim_combs2(
    n: int = None,
    inds: np.array = None,
    pb: bool = False,
) -> pd.DataFrame:

    if not ((n is None) ^ (inds is None)):
        raise ValueError("Either `n` or `inds` must be provided (and not both).")

    db_file = "./config/database.db"
    tb_name = "combinations"

    conf_file = "./config/experiment_config.json"
    with open(conf_file, "r") as f:
        config = json.load(f)

    rule_mapping, stim_mapping = config["rules_mapping"], config["stim_mapping"]
    rule_mapping = {v: k for k, v in rule_mapping.items()}
    stim_mapping = {v: k for k, v in stim_mapping.items()}

    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    col_names = [i[1] for i in c.execute(f"PRAGMA table_info({tb_name})").fetchall()]

    # n_per_rule = c.execute(
    #     f"SELECT pattern, COUNT(pattern) FROM {tb_name} GROUP BY pattern"
    # ).fetchall()

    max_n = c.execute(f"SELECT MAX(itemid) FROM {tb_name}").fetchall()[0][0]

    selected_ids = [
        str(i) for i in np.random.choice(np.arange(max_n), n, replace=False)
    ]

    combs = c.execute(
        f"SELECT * FROM {tb_name} WHERE itemid in ({','.join(selected_ids)})"
    ).fetchall()

    combs = pd.DataFrame(combs, columns=col_names)
    # combs.groupby("pattern").size()

    selected_cols = [c for c in col_names if c not in ["id", "rule"]]
    combs[selected_cols] = combs[selected_cols].replace(stim_mapping)
    combs["pattern"].replace(rule_mapping, inplace=True)
    combs["pattern"].unique()

    return combs


def update_conf_file():
    # TODO
    pass


def reset_dir(directory: Union[str, Path], keep_stuct: bool = True):
    directory = Path(directory)

    if directory.exists():
        subdirs = [d for d in directory.iterdir() if d.is_dir()]

        for d in subdirs:
            shutil.rmtree(d)

        for subdir in subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
    else:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    wd = Path.cwd()

    img_dir = wd / "images"
    images = [im for im in img_dir.iterdir()]

    imgs_dict = {im.stem: Image.open(im) for im in images if im.is_file()}
    imgs_list = list(imgs_dict.values())

    shapes = list(imgs_dict.keys())
