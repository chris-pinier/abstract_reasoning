import os
import platform
from pathlib import Path
from typing import Union, List, Dict
from icecream import ic
from PIL import Image
import json
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from math import factorial
from itertools import combinations, permutations
import random
import numpy as np
from tqdm.auto import tqdm
import concurrent.futures
import multiprocessing

wd = Path(__file__).parent
os.chdir(wd)
from utils import get_monitors_info

# from database import Database
def get_mapping(items, sort_items=True, both=True) -> List[Dict]:
    if sort_items:
        items = sorted(items)

    mapping = [{item: i for i, item in enumerate(items)}]

    if both:
        mapping.append({v: k for k, v in mapping[0].items()})

    return mapping


def get_sequences(save_dir: Union[str, Path], icons=None, patterns=None):

    #! TEMP
    save_dir = wd

    # if icons is not None or patterns is not None:
    #     raise NotImplementedError("Custom icons and patterns are not yet supported.")

    icons = [
        "wheat",
        "key",
        "smile",
        "heart",
        "eye",
        "bulb",
        "carrot",
        "helicopter",
        "truck",
        "guitar",
        "cube",
        "star",
        "hammer",
        "bell",
        "sun",
        "pyramid",
    ]

    patterns = [
        ["AAAB AAA", "B"],
        ["ABAB CDC", "D"],
        ["ABBA ABB", "A"],
        ["ABBA CDD", "C"],
        ["ABBC ABB", "C"],
        ["ABCA ABC", "A"],
        ["ABCD DCB", "A"],
        ["ABCDE ED", "C"],
    ]

    save_dir = Path(save_dir / "sequences")
    save_dir.mkdir(exist_ok=True, parents=True)

    # * Determine the maximum number of unique characters in any pattern
    _patterns = ["".join(p).replace(" ", "") for p in patterns]
    all_letters = set(sorted("".join(_patterns)))

    stim_IDs, stim_IDs_inv = get_mapping(icons)
    pattern_IDs, pattern_IDs_inv = get_mapping(_patterns)

    with open(save_dir / "items_mapping.json", "w") as f:
        json.dump({"stim_IDs": stim_IDs, "pattern_IDs": pattern_IDs}, f, indent=4)

    # * Generate all unique permutations of icons for the maximum number of characters
    n_icons = len(icons)
    n_perms = int(factorial(n_icons) / factorial(n_icons - len(all_letters)))
    icons_int = stim_IDs.values()
    icon_permutations = permutations(icons, len(all_letters))

    all_combs = set()

    for perm in tqdm(icon_permutations, total=n_perms):
        icon_dict = dict(zip(sorted(all_letters), perm))
        for pattern in _patterns:
            comb = [icon_dict[char] for char in pattern]
            comb = [stim_IDs[name] for name in comb]
            comb += [pattern_IDs[pattern]]
            all_combs.add(tuple(comb))

    n_combs = len(all_combs)
    print(f"{n_combs:,} unique combinations generated.")

    cols = [f"figure{i}" for i in range(1, 9)] + ["pattern"]
    all_combs = pd.DataFrame(all_combs, columns=cols)

    all_combs.to_csv(save_dir / "all_combinations.csv", index=False)

    # all_combs["pattern"] = all_combs["pattern"].map(pattern_IDs)

    min_n_patterns = all_combs["pattern"].value_counts().min()
    selected_combs = all_combs.groupby("pattern").sample(min_n_patterns)
    # selected_combs = selected_combs.astype("object")

    n_per_pattern = 10
    n_sessions = 5
    # solution_inds = [-1] * (len(patterns) * n_per_pattern)
    solution_idx = 7

    sequences = []

    for i in range(n_sessions):
        sequences_i = selected_combs.groupby("pattern").sample(n_per_pattern)
        selected_combs.drop(sequences_i.index, inplace=True)

        patterns = sequences_i.pop("pattern")
        choice_cols = [f"choice{i+1}" for i in range(4)] + [
            "solution",
            "masked_idx",
            "seq_order",
            "choice_order",
        ]
        choices = pd.DataFrame(columns=choice_cols)

        for idx, seq in sequences_i.iterrows():
            seq = list(seq)
            unique_items = list(set(seq))
            solution = seq[solution_idx]

            n = len(unique_items)

            if n > 4:
                seq_choices = [solution] + random.sample(
                    [i for i in unique_items if i != solution], k=3
                )
            elif n <= 4:
                seq_choices = unique_items + random.sample(
                    [i for i in icons_int if i not in unique_items], k=4 - n
                )

            seq_choices = random.sample(seq_choices, len(seq_choices))

            seq_order = "".join([str(i) for i in random.sample(range(8), 8)])
            choice_order = "".join([str(i) for i in random.sample(range(4), 4)])

            seq_choices.extend([solution, solution_idx, seq_order, choice_order])
            seq_choices = pd.DataFrame([seq_choices], columns=choice_cols, index=[idx])

            choices = pd.concat([choices, seq_choices], axis=0)
            # choices.reset_index(drop=True, inplace=True)

        sequences_i = pd.concat([sequences_i, choices], axis=1)

        sequences_i = sequences_i.astype("object")
        sequences_i.iloc[:, :13] = sequences_i.iloc[:, :13].replace(stim_IDs_inv)
        sequences_i["masked_idx"] = sequences_i["masked_idx"].astype(int)
        sequences_i["pattern"] = patterns.replace(pattern_IDs_inv)
        sequences_i.reset_index(drop=False, inplace=True, names=["item_id"])
        sequences_i.to_csv(save_dir / f"session_{i+1}.csv", index=False)

        sequences.append(sequences_i)

    return sequences


def process_permutation(args):
    perm, _patterns, all_letters, stim_IDs, pattern_IDs = args
    icon_dict = dict(zip(sorted(all_letters), perm))
    combs = []
    for pattern in _patterns:
        comb = [icon_dict[char] for char in pattern]
        comb = [stim_IDs[name] for name in comb]
        comb += [pattern_IDs[pattern]]
        combs.append(tuple(comb))
    return combs


def get_practice_sequences(patterns, n_practice=4):
    raise NotImplementedError("This function is not yet implemented.")


def process_permutation(args):
    perm, _patterns, all_letters, stim_IDs, pattern_IDs = args
    icon_dict = dict(zip(sorted(all_letters), perm))
    combs = []
    for pattern in _patterns:
        comb = [icon_dict[char] for char in pattern]
        comb = [stim_IDs[name] for name in comb]
        comb += [pattern_IDs[pattern]]
        combs.append(tuple(comb))
    return combs


def get_sequences_mpcsing(save_dir: Union[str, Path], icons=None, patterns=None):
    save_dir = Path(save_dir) / "sequences2"
    save_dir.mkdir(exist_ok=True, parents=True)

    icons = [
        "wheat",
        "key",
        "smile",
        "heart",
        "eye",
        "bulb",
        "carrot",
        "helicopter",
        "truck",
        "guitar",
        "cube",
        "star",
        "hammer",
        "bell",
        "sun",
        "pyramid",
    ]

    # patterns = [
    #     ["AAAB AAA", "B"],
    #     ["ABAB CDC", "D"],
    #     ["ABBA ABB", "A"],
    #     ["ABBA CDC", "D"],
    #     ["ABBC ABB", "C"],
    #     ["ABCA ABC", "A"],
    #     ["ABCD DCB", "A"],
    #     ["ABCDE ED", "C"],
    # ]
    patterns = [
        ["AAAA AAA", "A"],
        ["AAAB AAA", "B"],
        ["AABB AAB", "B"],
        ["ABBB ABB", "B"],
        ["AABC AAB", "C"],
        ["ABCC ABC", "C"],
        ["ABBC ABB", "C"],
        ["ABAB ABA", "B"],
        ["ABBA ABB", "A"],
        ["ABCA ABC", "A"],
        ["ABAC ABA", "C"],
        ["ABAB CDC", "D"],
        ["AABB CCD", "D"],
        ["ABCD ABC", "D"],
        ["ABBA CDD", "C"],
        ["ABCD DCB", "A"],
        ["ABCD EAB", "C"],
        ["ABCD EFA", "B"],
        ["ABCD EFG", "H"],
        ["ABCD EED", "C"],
        ["ABC CBA A", "B"],
    ]
    _patterns = ["".join(p).replace(" ", "") for p in patterns]
    all_letters = set(sorted("".join(_patterns)))

    stim_IDs, stim_IDs_inv = get_mapping(icons)
    pattern_IDs, pattern_IDs_inv = get_mapping(_patterns)

    with open(save_dir / "items_mapping.json", "w") as f:
        json.dump({"stim_IDs": stim_IDs, "pattern_IDs": pattern_IDs}, f, indent=4)

    n_icons = len(icons)
    n_perms = int(factorial(n_icons) / factorial(n_icons - len(all_letters)))
    icons_int = stim_IDs.values()
    icon_permutations = list(permutations(icons, len(all_letters)))

    args = [
        (perm, _patterns, all_letters, stim_IDs, pattern_IDs)
        for perm in icon_permutations
    ]

    all_combs = set()
    with concurrent.futures.ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("fork")
    ) as executor:
        futures = [executor.submit(process_permutation, arg) for arg in args]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=n_perms,
            desc="Processing permutations",
        ):
            all_combs.update(future.result())

    n_combs = len(all_combs)
    print(f"{n_combs:,} unique combinations generated.")

    cols = [f"figure{i}" for i in range(1, 9)] + ["pattern"]
    all_combs = pd.DataFrame(all_combs, columns=cols)

    all_combs.to_csv(save_dir / "all_combinations.csv", index=False)

    min_n_patterns = all_combs["pattern"].value_counts().min()
    selected_combs = all_combs.groupby("pattern").sample(min_n_patterns)

    n_per_pattern = 10
    n_sessions = 5
    solution_idx = 7

    sequences = []

    for i in range(n_sessions):
        sequences_i = selected_combs.groupby("pattern").sample(n_per_pattern)
        selected_combs.drop(sequences_i.index, inplace=True)

        patterns = sequences_i.pop("pattern")
        choice_cols = [f"choice{i+1}" for i in range(4)] + [
            "solution",
            "masked_idx",
            "seq_order",
            "choice_order",
        ]
        choices = pd.DataFrame(columns=choice_cols)

        for idx, seq in sequences_i.iterrows():
            seq = list(seq)
            unique_items = list(set(seq))
            solution = seq[solution_idx]

            n = len(unique_items)

            if n > 4:
                seq_choices = [solution] + random.sample(
                    [i for i in unique_items if i != solution], k=3
                )
            elif n <= 4:
                seq_choices = unique_items + random.sample(
                    [i for i in icons_int if i not in unique_items], k=4 - n
                )

            seq_choices = random.sample(seq_choices, len(seq_choices))

            seq_order = "".join([str(i) for i in random.sample(range(8), 8)])
            choice_order = "".join([str(i) for i in random.sample(range(4), 4)])

            seq_choices.extend([solution, solution_idx, seq_order, choice_order])
            seq_choices = pd.DataFrame([seq_choices], columns=choice_cols, index=[idx])

            choices = pd.concat([choices, seq_choices], axis=0)

        sequences_i = pd.concat([sequences_i, choices], axis=1)

        sequences_i = sequences_i.astype("object")
        sequences_i.iloc[:, :13] = sequences_i.iloc[:, :13].replace(stim_IDs_inv)
        sequences_i["masked_idx"] = sequences_i["masked_idx"].astype(int)
        sequences_i["pattern"] = patterns.replace(pattern_IDs_inv)
        sequences_i.reset_index(drop=False, inplace=True, names=["item_id"])
        sequences_i.to_csv(save_dir / f"session_{i+1}.csv", index=False)

        sequences.append(sequences_i)

    return sequences


# if __name__ == "__main__":
    # multiprocessing.set_start_method("fork")
    # sequences = get_sequences_mpcsing(save_dir=Path.cwd())

# def manage_directories(directories: list, action: str):
#     for directory in directories:
#         if action == "create":
#             directory.mkdir(parents=True, exist_ok=True)
#         elif action == "remove":
#             shutil.rmtree(directory)
#         elif action == "clean":
#             for f in directory.glob("*"):
#                 if f.is_file():
#                     f.unlink()
#                 else:
#                     shutil.rmtree(f)
#         else:
#             raise ValueError(f"Invalid action: {action}")


# def setup_stimuli(root_dir: Union[str, Path]):
#     ic.enable()

#     # * Experiment directory (either lab, online or ANNs)
#     root_dir = Path(root_dir)

#     config_dir = root_dir.parent / "config"
#     config_file = config_dir / "experiment_config.json"
#     img_dir = config_dir / "images/original"

#     std_img_dir = root_dir / "images/pixel_standardized"
#     resized_dir = root_dir / "images/resized"
#     std_img_dir.mkdir(exist_ok=True, parents=True)
#     resized_dir.mkdir(exist_ok=True, parents=True)

#     monitors = get_monitors_info()
#     monitor = [m for m in monitors if m["primary"] == True][0]
#     res = monitor["res"]

#     with open(config_dir / "rules.json", "r") as f:
#         rules = json.load(f)

#     imgs_dict = {im.stem: im for im in img_dir.glob("*.png")}

#     im_size = Image.open(imgs_dict[list(imgs_dict.keys())[0]]).size

#     # * Remove directory containing scaled images if it exists
#     if resized_dir.exists():
#         shutil.rmtree(resized_dir)

#     # * Standardize images to have ~ same number of black pixels
#     df_imgs = standardize_images(imgs_dict, dest_dir=std_img_dir)
#     standardize_imgs = [im for im in std_img_dir.glob("*.png")]

#     df_imgs.sort_values("new_ratio", ascending=False, inplace=True)
#     df_imgs["new_ratio"].hist()
#     plt.title("black pixels distribution (standardized)")
#     plt.show()

#     df_imgs.to_excel(root_dir / "images/imag|es_info.xlsx")  # , float_format="%.3f")

#     # * Resize images based on screen resolution
#     max_items = 12
#     max_img_height = res[1] / 3
#     resize_factor_x = (res[0] / max_items) / im_size[0]
#     resize_factor_y = max_img_height / im_size[1]
#     resize_factor = round(min(resize_factor_x, resize_factor_y), 3)
#     new_size = [int(im_size[0] * resize_factor)] * 2
#     ic(new_size)

#     scale_images(standardize_imgs, absolute=new_size, save_dir=resized_dir)
#     scaled_images = {im.stem: Image.open(im) for im in resized_dir.glob("*.png")}

#     pix_counts = get_pixel_counts(scaled_images)
#     pix_counts["black"].hist()
#     plt.title("black pixels distribution (scaled)")
#     plt.show()
#     shutil.rmtree(resized_dir)

#     bckgrd_color = (255, 255, 255)  # RGB color code for white
#     img = Image.new("RGB", new_size, bckgrd_color)

#     # Save the image if you want
#     img.save(root_dir / "images/blank_image.png")

#     # * Generate all possible combinations
#     question_mark = imgs_dict.pop("question-mark")

#     imgs_list = list(imgs_dict.values())
#     shapes = list(imgs_dict.keys())

#     n_letters = len(set("".join([(r[0] + r[1]).replace(" ", "") for r in rules])))
#     # selected_shapes = np.random.choice(shapes, replace=False, size=n_letters + 12)
#     # selected_icons = [
#     #     "bone",
#     #     "bulb",
#     #     "camera",
#     #     "eye",
#     #     "gamepad",
#     #     "headphones",
#     #     "heart",
#     #     "home",
#     #     "lock",
#     #     "megaphone",
#     #     "smile",
#     #     "wheat",
#     # ]
#     selected_icons = [
#         "wheat",
#         "key",
#         "smile",
#         "heart",
#         "eye",
#         "bulb",
#         "carrot",
#         "helicopter",
#         "truck",
#         "guitar",
#         "cube",
#         "star",
#         "hammer",
#         "bell",
#         "sun",
#         "pyramid",
#     ]

#     assert len(selected_icons) >= n_letters

#     stim_IDs, pattern_IDs = get_mappings(shapes, rules)
#     # stim_IDs_inv = {v: k for k, v in stim_IDs.items()}
#     # pattern_IDs_inv = {v: k for k, v in pattern_IDs.items()}

#     with open(config_file, "r") as f:
#         exp_config = json.load(f)

#     # config["pattern_IDs"] = pattern_IDs
#     # config["stim_IDs"] = stim_IDs

#     # with open(config_file, "w") as f:
#     #     json.dump(config, f, indent=4)

#     # n_combs, all_combs = generate_all_combinations4(selected_icons, rules, stim_IDs)

#     # # * Insert combinations into the database
#     # db_file = Path(config_dir / "database.db")
#     # db = Database(db_file)

#     # db.insert_combinations(all_combs)
#     # # db.manage_table("combinations", "count")

#     # # db.manage_table("combinations", "empty")

#     # stim_IDs = {k: v for k, v in stim_IDs.items() if k in selected_icons}
#     # sequences = get_experiment_sequences(db, stim_IDs, n_combs)


# def create_symlinks(source, targets):
#     """
#     Create symbolic links for a source (file or directory) in each of the target directories,
#     generalizable over operating systems.

#     :param source: The path to the source file or directory.
#     :param targets: A list of paths to the target directories where the symlink will be created.
#     """
#     source_name = os.path.basename(source)
#     is_directory = os.path.isdir(source)

#     for target in targets:
#         target_path = os.path.join(target, source_name)
#         if not os.path.exists(target_path):
#             try:
#                 if platform.system() == "Windows":
#                     os.symlink(source, target_path, target_is_directory=is_directory)
#                 else:
#                     os.symlink(source, target_path)
#                 print(f"Created symlink: {target_path} -> {source}")
#             except OSError as e:
#                 print(f"Error creating symlink: {e}")
#         else:
#             print(f"Symlink already exists: {target_path}")


# def setup_structure(root_dir: Union[str, Path]):

#     config_dir = root_dir.parent / "config"
#     config_file = config_dir / "experiment_config.json"

#     # * ### Create symlinks for images ###
#     source_dirs = [config_dir / "images/resized"]

#     target_directories = [
#         root_dir / "local/images/",
#         root_dir / "online/images/",
#     ]

#     for target_dir in target_directories:
#         target_dir.mkdir(parents=True, exist_ok=True)

#     for source_dir in source_dirs:
#         create_symlinks(source_dir, target_directories)

#     # * ### Create symlinks for configuration files ###
#     source_files = [
#         config_dir / "rules.json",
#         config_dir / "database.db",
#         config_dir / "selected_combinations-format[names].csv",
#         config_dir / "selected_combinations-format[IDs].csv",
#         config_dir / "experiment_config.json",
#     ]

#     target_directories = [
#         root_dir / "local/config/",
#         root_dir / "online/config/",
#         root_dir / "ANNs/config/",
#     ]

#     for source_f in source_files:
#         create_symlinks(source_f, target_directories)

#     # * ### Create symlink for blank image ###
#     create_symlinks(
#         source=config_dir / "images/blank_image.png",
#         targets=[root_dir / "local/images/", root_dir / "online/images/"],
#     )


# def create_database(root_dir: Union[str, Path], fname: str = "database.db"):
#     root_dir = Path(root_dir)

#     db_fpath = root_dir / f"global_config/{fname}"

#     if db_fpath.exists():
#         print(f"Database already exists: {db_fpath}")
#     else:
#         print(f"Creating database: {db_fpath}")
#         db = Database(db_fpath)

#     for table in db.tables:
#         print(f"TABLE: `{table}`:")
#         info = db.manage_table(table, "info")
#         print("\n\n")


# if __name__ == "__main__":
#     try:
#         root_dir = Path(__file__).resolve().parent
#     except NameError as e:
#         print(
#             "Something went wrong. Make sure to run this script from the command line.\n"
#             "Instructions:"
#             "\n\t1. Open a terminal."
#             "\n\t2. Navigate to the directory containing this script."
#             "\n\t3. Run the script using the command `python setup.py`."
#             "\nYou may need to lauch the terminal as an administrator to create symlinks."
#         )
#         raise e

#     setup_stimuli(root_dir)

#     # setup_structure(root_dir)

#     print("Setup complete.")
