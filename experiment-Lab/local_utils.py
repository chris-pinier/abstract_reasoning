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


# from typing import Any, Callable


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


def exp_prep():
    # img_dir = wd / "images/original"
    # images = {im.stem: Image.open(im) for im in img_dir.glob("*.png")}

    # df_imgs = standardize_images(images)
    # df_imgs.sort_values("new_ratio", ascending=False)
    # df_imgs["new_ratio"].hist()
    # df_imgs["new_ratio"][1:].describe()
    # df_imgs.head(60)
    # df_imgs.to_excel(wd / "config/images_info.xlsx")  # , float_format="%.3f")

    # std_img_dir = wd / "images/pixel_standardized"
    # resized_dir = wd / "images/standardized-resized"

    # if resized_dir.exists():
    #     shutil.rmtree(resized_dir)

    # images = [im for im in std_img_dir.glob("*.png")]
    # scale_images(images, scale=0.08, res=res[0], save_dir=resized_dir)

    # images = [im for im in resized_dir.glob("*.png")]
    # len(images)

    # imgs_dict = {im.stem: im for im in images}
    # question_mark = imgs_dict.pop("question-mark")

    # imgs_list = list(imgs_dict.values())
    # shapes = list(imgs_dict.keys())

    # # set(np.unique(shapes, return_counts=True)[1])

    # selected_shapes = np.random.choice(shapes, replace=False, size=20)

    # # all_combs = generate_all_combinations(selected_shapes, rules_1)

    # all_combs = generate_all_combinations2(shapes[:20], rules_1)
    # f"{len(all_combs):,}"

    # # all_combs.to_csv(wd / "config/stimuli_combinations.csv")
    # # all_combs = pd.read_csv(wd / "config/stimuli_combinations_0.csv")
    # # all_combs.shape[0]
    # n_splits = np.ceil(all_combs.shape[0] / 150_000)
    # split_inds = np.array_split(np.arange(all_combs.shape[0]), n_splits)
    # split_inds_ranges = [(r[0], r[-1]) for r in split_inds]

    # pd.DataFrame(split_inds_ranges, columns=["start", "end"]).to_csv(
    #     "config/stimuli_combinations_ranges.csv",
    # )

    # for i, inds in enumerate(split_inds):
    #     all_combs.iloc[inds].to_csv(
    #         wd / f"config/stimuli_combinations_{i}.csv", index_label="index"
    #     )
    #     # all_combs.iloc[inds].to_feather(wd / f"config/stimuli_combinations_{i}.feather")
    # # pd.read_csv(wd / f"config/stimuli_combinations_{i}.csv", index_col="index")

    # # for i in tqdm(range(n_splits)):
    # #     df = pd.read_csv(wd / f"config/stimuli_combinations_{i}.csv")
    # #     df.to_csv(
    # #         wd / f"config/stimuli_combinations_{i}.csv", index=True, index_label="index"
    # #     )
    # #     # df.reset_index(drop=True, inplace=True)

    # all_combs = [list(vals) for idx, vals in all_combs.iterrows()]
    raise NotImplementedError


def sess_prep(
    images: Dict[str, str],
    shapes: List[str],
    stim_combinations: pd.DataFrame,
    allowed_keys: List[str],
    window_size: List[int],
) -> Tuple[Dict, Dict]:

    assert (
        len(set([Image.open(im).size for im in images.values()])) == 1
    ), "Images must have same size"

    img_size = Image.open(images[shapes[0]]).size
    x_positions = {}
    resp_mapping = {}

    match_cols_choice = lambda x: re.compile("choice\d{1,2}", re.IGNORECASE).search(x)
    match_cols_seq = lambda x: re.compile("figure\d{1,2}", re.IGNORECASE).search(x)

    seq_cols = [c for c in stim_combinations.columns if match_cols_seq(c)]
    choice_cols = [c for c in stim_combinations.columns if match_cols_choice(c)]

    x_pos_seq = {}

    for seq_length in range(1, len(seq_cols) + 2):  # +1 for 0 index, +1 for solution
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
        x_pos_seq[seq_length]["pos"] = [round(pos, 3) for pos in positions]  # positions
        x_pos_seq[seq_length]["sep_space_width"] = round(sep_space_width, 3)

    for idx_row, row in tqdm(stim_combinations.iterrows()):
        x_positions[idx_row] = {}

        solution = [row.loc["solution"]]
        avail_choices = row.loc[choice_cols].dropna().tolist()
        sequence = row.loc[seq_cols].dropna().tolist()
        # row = row.to_dict()
        # solution = [v for k, v in row.items() if "solution" in k]
        # avail_choices = [v for k, v in row.items() if match_cols_choice(k)]
        # sequence = [v for k, v in row.items() if match_cols_seq(k)]

        sequence += ["question-mark"]

        if len(solution) != 1:
            raise NotImplementedError("Modify code to handle multiple solutions")
        else:
            solution = solution[0]

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

    return x_positions, resp_mapping


def sess_prep2(
    images: Dict[str, str],
    shapes: List[str],
    sequences: pd.DataFrame,
    allowed_keys: List[str],
    window_size: List[int],
) -> Tuple[Dict, Dict]:

    assert (
        len(set([Image.open(im).size for im in images.values()])) == 1
    ), "Images must have same size"

    solution_mask = "question-mark"

    img_size = Image.open(images[shapes[0]]).size
    x_positions = {}
    resp_mapping = {}

    match_cols_choice = lambda x: re.compile("choice\d{1,2}", re.IGNORECASE).search(x)
    match_cols_seq = lambda x: re.compile("figure\d{1,2}", re.IGNORECASE).search(x)

    seq_cols = [c for c in sequences.columns if match_cols_seq(c)]
    choice_cols = [c for c in sequences.columns if match_cols_choice(c)]

    x_pos_seq = {}
    new_sequences = sequences.copy()

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
        x_pos_seq[seq_length]["pos"] = [round(pos, 3) for pos in positions]  # positions
        x_pos_seq[seq_length]["sep_space_width"] = round(sep_space_width, 3)

    for idx_row, row in tqdm(new_sequences.iterrows()):
        x_positions[idx_row] = {}

        avail_choices = row.loc[choice_cols].dropna().tolist()
        sequence = row.loc[seq_cols].dropna().tolist()

        # * replace solution with the mask
        new_sequences.loc[idx_row, seq_cols[row["maskedImageIdx"]]] = solution_mask

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

    return new_sequences, x_positions, resp_mapping


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


def check_images_pixels():
    # import matplotlib.pyplot as plt

    img_dir = wd / "images/standardized-resized"
    imgs_dict = {im.stem: Image.open(im) for im in img_dir.iterdir()}
    img_sizes = tuple(set([im.size for im in imgs_dict.values()]))

    pixel_counts = {}
    converted_images = {}
    for img_name, img in imgs_dict.items():
        arr = np.asarray(img).copy()

        np.unique(arr, return_counts=True)
        np.unique(arr[:, :, 3])

        # plt.imshow(arr)

        arr[:, :, 3]
        converted = np.where(arr > 0, 255, 0)
        # converted = np.where(arr < 255, 0, 255)
        plt.imshow(converted)
        # Image.fromarray(converted, mode="RGBA").show()
        # plt.imshow(converted)

        # converted_images[img_name] = Image.fromarray(converted)
        unique, counts = np.unique(converted, return_counts=True)
        pixel_counts[img_name] = dict(zip(unique, counts))

    df = pd.DataFrame(pixel_counts).T
    df
    df.sort_values(255, ascending=False)


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
