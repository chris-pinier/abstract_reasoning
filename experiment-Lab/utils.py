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


def prepare_sess(
    images: Dict[str, str],
    icons: List[str],
    sequences: pd.DataFrame,
    allowed_keys: List[str],
    window_size: List[int],
    block_size: int,
) -> Tuple[Dict, Dict]:

    assert (
        len(set([Image.open(im).size for im in images.values()])) == 1
    ), "Images must have the same size"

    img_size = Image.open(images[icons[0]]).size
    x_positions = {}
    resp_mapping = {}

    match_cols_choice = lambda x: re.compile(r"choice\d{1,2}", re.IGNORECASE).search(x)
    match_cols_seq = lambda x: re.compile(r"figure\d{1,2}", re.IGNORECASE).search(x)

    seq_cols = [c for c in sequences.columns if match_cols_seq(c)]
    choice_cols = [c for c in sequences.columns if match_cols_choice(c)]

    x_pos_seq = {}

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

    for idx_row, row in tqdm(sequences.iterrows()):
        x_positions[idx_row] = {}

        avail_choices = row.loc[choice_cols].dropna().tolist()
        sequence = row.loc[seq_cols].dropna().tolist()

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
    else:
        blocks = np.array_split(sequences, len(sequences) / block_size)

    trial_blocks = []

    for block in blocks:
        trials = []
        for idx_row, row in block.iterrows():
            row.index
            trial = row.to_dict()
            trial.update(
                {
                    "item_id": row["item_id"],
                    "x_pos": x_positions[idx_row],
                    "resp_map": resp_mapping[idx_row],
                }
            )
            trial["seq_order"] = [
                int(i) for i in str(trial["seq_order"]) if i.isdigit()
            ]
            trial["choice_order"] = [
                int(i) for i in str(trial["choice_order"]) if i.isdigit()
            ]

            trials.append(trial)
        trial_blocks.append(trials)

    # trial_blocks = (block for block in trial_blocks)

    return trial_blocks


if __name__ == "__main__":
    wd = Path.cwd()

    img_dir = wd / "images"
    images = [im for im in img_dir.iterdir()]

    imgs_dict = {im.stem: Image.open(im) for im in images if im.is_file()}
    imgs_list = list(imgs_dict.values())

    shapes = list(imgs_dict.keys())
