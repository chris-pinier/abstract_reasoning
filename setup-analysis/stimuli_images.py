from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from itertools import combinations, permutations
import pandas as pd
from tqdm.auto import tqdm
import shutil
import random
from typing import List, Dict, Union
from icecream import ic
import json
import sqlite3
from math import factorial
import shutil
import pyperclip
import time
import pendulum
from loguru import logger
from string import ascii_uppercase, ascii_lowercase
from functools import wraps
from rich import print as rprint
from tabulate import tabulate
from local.database import Database
from local.local_utils import get_pixel_counts, reset_dir
import re

def timing_filter(record):
    return record["extra"].get("timing", False)


# * Remove any existing handlers
logger.remove()
# * Add general log files (example)
logger.add("./logs/file_X.log", rotation="00:01", retention="5 days", level="INFO")
logger.add("./logs/timer.log", filter=timing_filter, level="INFO")

timer_on = True


def timer(enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            start = time.perf_counter()
            results = func(*args, **kwargs)
            end = time.perf_counter()

            elapsed = pendulum.duration(seconds=end - start)
            elapsed_str = f"{elapsed.total_seconds()} seconds"
            elapsed_str = f"Elapsed time for `{func.__name__}`: {elapsed_str}"

            rprint(f"[green]{elapsed_str}[/green]")
            logger.info(elapsed_str)

            return results

        return wrapper

    return decorator


def create_collage(
    image_paths: List[str],
    max_images: int,
    idx_solution: int = None,
    question_mark: str = None,
):
    rect_stroke_width = 6

    # ! inneficient to open images twice:
    # ! first get a set of images then open / copy them according to the set
    imgs = [Image.open(image_path) for image_path in image_paths]
    img_size = [img.size for img in imgs][0][0]

    if question_mark:
        imgs.append(Image.open(question_mark))

    n_images = len(imgs)
    blank_space = int(img_size * 0.2)
    width = img_size * max_images + blank_space * (max_images - 1)
    center = width // 2
    collage_length = n_images * img_size + (n_images - 1) * blank_space
    first_position = center - collage_length // 2

    # * Adjust collage height to account for the rectangle stroke width
    collage_height = img_size + 2 * rect_stroke_width
    collage = Image.new("RGBA", (width, collage_height), (255, 255, 255, 0))

    # * Vertical offset to center images in the new collage height
    vertical_offset = rect_stroke_width

    positions = []

    for i, img in enumerate(imgs):
        position = (first_position + i * img_size + i * blank_space, vertical_offset)
        collage.paste(img, position)
        positions.append((position, img_size))

    # * Drawing a red square around the solution image
    if idx_solution is not None:
        draw = ImageDraw.Draw(collage)
        solution_pos = (
            first_position + idx_solution * img_size + idx_solution * blank_space,
            vertical_offset,
        )
        rect_coords = (
            solution_pos[0] - rect_stroke_width,
            solution_pos[1] - rect_stroke_width,
            solution_pos[0] + rect_stroke_width + img_size,
            solution_pos[1] + rect_stroke_width + img_size,
        )
        draw.rectangle(rect_coords, outline="red", width=rect_stroke_width)

    return collage, positions


def create_stimuli(
    shapes,
    imgs_dict: Dict[str, str],
    combs: List[List[str]],
    question_mark: str,
    n: int = None,
    zerofill: int = None,
):
    # ! TEMP
    # n = 150
    # zerofill=None
    # combs = all_combs
    # ! TEMP

    img_types = ("problems", "choices", "solutions")
    for type in img_types:
        folder = wd / f"stimuli/{type}"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()

    # * Create dataframe to store stimuli info
    unique_n_items = set([len(i) - 1 for i in combs])
    ic(unique_n_items)

    max_items = max(unique_n_items)

    df_cols = [f"item_{i+1}" for i in range(max_items)]
    df_cols += [f"choice_{i+1}" for i in range(4)]
    df_cols += ["solution"]

    # df_cols = [(f"item_{i+1}", pl.String) for i in range(max_items)]
    # df_cols += [(f"choice_{i+1}", pl.String) for i in range(4)]
    # df_cols += [("solution", pl.String)]

    df = pd.DataFrame(columns=df_cols)
    # df = pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in df_cols})

    if n is None:
        selected = combs
    else:
        selected = np.random.choice(len(combs), n, replace=False)
        selected = [combs[i] for i in selected]

    if zerofill is None:
        n_combs = len(combs)
        n_digits = len(str(n_combs))
    else:
        n_digits = zerofill

    positions = {}

    for idx, problem_set in enumerate(tqdm(selected)):
        n_distract = random.randint(1, 3)
        distractors = [el for el in shapes if el not in problem_set]
        distractors = random.sample(distractors, k=n_distract)

        solution = problem_set.pop(-1)

        choices = [el for el in set(problem_set) if el != solution]
        # ic(f"{3 - n_distract = }")

        choices = random.sample(choices, k=3 - n_distract) + [solution] + distractors
        np.random.shuffle(choices)

        df_row = problem_set + [pd.NA] * (max_items - len(problem_set))
        df_row += choices + [solution]
        df_row = pd.DataFrame([df_row], columns=df_cols)
        # df_row = pl.DataFrame({col[0]: val for col, val in zip(df_cols, df_row)})

        df = pd.concat([df, df_row])
        df.reset_index(drop=True, inplace=True)

        idx_solution = choices.index(solution)
        max_images = max([len(i) for i in combs])

        # * Problem set
        problem_set = [imgs_dict[item] for item in problem_set]

        pb_set_positions = {}

        img_problems, img_problems_pos = create_collage(
            problem_set,
            max_images=max_images,
            question_mark=question_mark,
        )
        pb_set_positions["problems"] = img_problems_pos

        # * Choices
        choices = [imgs_dict[item] for item in choices]

        img_choices, img_choices_pos = create_collage(choices, max_images=max_images)
        pb_set_positions["choices"] = img_choices_pos

        # * Solution
        img_solution, img_solutions_pos = create_collage(
            choices,
            max_images=max_images,
            idx_solution=idx_solution,
        )
        pb_set_positions["solutions"] = img_problems_pos

        positions[idx + 1] = pb_set_positions

        for img, type in zip(
            [img_problems, img_choices, img_solution],
            img_types,
        ):
            if type == "solutions":
                img_name = f"{str(idx + 1).zfill(n_digits)}_{type}-{idx_solution + 1}"
            else:
                img_name = f"{str(idx + 1).zfill(n_digits)}_{type}"

            img.save(f"stimuli/{type}/{img_name}.png")

    return df, positions


def create_stimuli2(
    shapes,
    imgs_dict: Dict[str, str],
    combs: List[List[str]],
    question_mark: str,
    n: int = None,
    zerofill: int = None,
):
    # ! TEMP
    # n = 15
    # zerofill=None
    # combs = all_combs
    # ! TEMP

    df = pd.DataFrame(columns=combs.columns)
    # df = pl.DataFrame({name: pl.Series([], dtype=dtype) for name, dtype in df_cols})

    if n is None:
        selected = combs
    else:
        selected = np.random.choice(len(combs), n, replace=False)
        selected = combs.iloc[selected]

    if zerofill is None:
        n_combs = len(combs)
        n_digits = len(str(n_combs))
    else:
        n_digits = zerofill

    for idx, problem_set in tqdm(selected.iterrows()):
        unique_items = problem_set.unique()
        distractors = [el for el in shapes if el not in unique_items]
        distractors = random.sample(distractors, k=2)

        solution = [item for name, item in problem_set.items() if "solution" in name]

        choices = [el for el in unique_items if el not in solution]
        choices = random.sample(choices, k=1) + solution + distractors
        np.random.shuffle(choices)
        pd.Series(choices, name=[f"choice_{i+1}" for i in range(len(choices))])

        # df_row = problem_set + [pd.NA] * (max_items - len(problem_set))
        df_row += choices + [solution]
        df_row = pd.DataFrame([df_row], columns=df_cols)
        # df_row = pl.DataFrame({col[0]: val for col, val in zip(df_cols, df_row)})

        df = pd.concat([df, df_row])
        df.reset_index(drop=True, inplace=True)

        idx_solution = choices.index(solution)
        max_images = max([len(i) for i in combs])

        # * Problem set
        problem_set = [imgs_dict[item] for item in problem_set]

        img_problems = create_collage(
            problem_set,
            max_images=max_images,
            question_mark=question_mark,
        )

        # * Choices
        choices = [imgs_dict[item] for item in choices]

        img_choices = create_collage(choices, max_images=max_images)

        # * Solution
        img_solution = create_collage(
            choices,
            max_images=max_images,
            idx_solution=idx_solution,
        )

        for img, type in zip(
            [img_problems, img_choices, img_solution],
            ["problems", "choices", "solutions"],
        ):
            if type == "solutions":
                img_name = f"{str(idx + 1).zfill(n_digits)}_{type}-{idx_solution + 1}"
            else:
                img_name = f"{str(idx + 1).zfill(n_digits)}_{type}"

            img.save(f"stimuli/{type}/{img_name}.png")

    return df


def generate_all_combinations(shapes, rules):
    # ! TEMP
    # rules = rules
    # shapes = selected_shapes
    # ! TEMP

    # problems = [r[:-1] for r in rules]
    # solutions = [r[-1] for r in rules]

    letters = sorted(set("".join(rules)))

    # items = np.random.choice(shapes, len(letters), replace=False)
    # items = dict(zip(letters, items))

    shape_combinations = list(combinations(shapes, len(letters)))

    all_combs = []
    for comb in tqdm(shape_combinations):
        inner_combs = []

        # comb = list(comb)

        for i in range(len(comb)):
            comb.insert(0, comb.pop())
            inner_combs.append(dict(zip(letters, comb)))

        for inner_comb in inner_combs:
            for rule in rules:
                problem_set = [inner_comb[l] for l in rule]
                # solution = problem_set.pop()
                # all_combs.append((problem_set, solution))
                all_combs.append(problem_set)

    return all_combs


def generate_all_combinations2(shapes: List[str], patterns: List[List[str]]):
    # * Determine the maximum number of unique characters in any rule
    rules = [rule for rule, solution in patterns]
    solutions = [solution for rule, solution in patterns]

    solution_length = set([len(sol) for sol in solutions])
    assert len(solution_length) == 1, "All solutions must be of equal length"
    solution_length = solution_length.pop()

    all_letters = set(("".join(rules) + "".join(solutions)).replace(" ", ""))
    # max_letters = max(len(set(rule.replace(" ", ""))) for rule in rules)
    max_letters = len(all_letters)

    # * Generate all unique permutations of shapes for the maximum number of characters
    ic("Generating all unique permutations")
    shape_permutations = list(permutations(shapes, max_letters))
    ic("DONE")
    ic(len(shape_permutations))

    all_combs = set()

    for perm in tqdm(shape_permutations):
        shape_dict = dict(zip(sorted(set("".join(rules).replace(" ", ""))), perm))

        for rule, solution_items in zip(rules, solutions):
            full_pattern, partial_pattern = rule.split()

            # * Apply the full and partial patterns to the current permutation
            full_pattern = [shape_dict[char] for char in full_pattern]
            partial_pattern = [shape_dict[char] for char in partial_pattern]

            solution = [shape_dict[i] for i in solution_items]

            comb = full_pattern + partial_pattern + solution

            temp = []
            for item in comb:
                if item not in temp:
                    temp.append(item)

            items_mapping = {item: ascii_uppercase[i] for i, item in enumerate(temp)}
            comb.append("".join([items_mapping[item] for item in comb]))

            all_combs.add(tuple(comb))

    return tuple(all_combs)


def generate_all_combinations3(shapes: List[str], patterns: List[List[str]]):
    from string import ascii_uppercase

    patterns = [
        # ["AAAAA", "A"],
        ["ABC AB", "C"],
        ["AA BB C", "C"],
        ["ABAB ABA", "B"],
        ["ABBC ABB", "C"],
    ]

    patterns = [[p[0].replace(" ", ""), p[1]] for p in patterns]

    # * Determine the maximum number of unique characters in any rule
    # rules = [rule for rule, solution in patterns]
    # solutions = [solution for rule, solution in patterns]

    solution_length = set([len(sol) for sol in solutions])
    assert len(solution_length) == 1, "All solutions must be of equal length"
    solution_length = solution_length.pop()

    rules = patterns

    rules_lengths = set([len(rule[0]) for rule in rules])
    rules_by_len = {
        r_len: [r for r in rules if len(r[0]) == r_len] for r_len in rules_lengths
    }
    rules_by_len = {
        k: v for k, v in sorted(rules_by_len.items(), key=lambda item: item[0])
    }

    all_combs = set()

    for r_len, r_list in rules_by_len.items():
        max_letters = max(len(set(rule[0].replace(" ", ""))) for rule in r_list)

        # * Generate all unique permutations of shapes for the maximum number of characters
        shape_permutations = list(permutations(shapes, max_letters))

        rules = [rule for rule, solution in r_list]
        solutions = [solution for rule, solution in r_list]
        solutions = [solutions] if len(solutions) == 1 else solutions

        for perm in tqdm(shape_permutations):
            shape_dict = dict(zip(sorted(set("".join(rules).replace(" ", ""))), perm))

            for rule, solution in zip(rules, solutions):
                # * Apply the full and partial patterns to the current permutation
                full_pattern = [shape_dict[char] for char in rule]

                # * Deduce the solution shape based on the logic of the full pattern
                solution = [shape_dict[i] for i in solution]

                comb = tuple(full_pattern + solution)
                all_combs.add(comb)

    rules = []
    for comb in all_combs:
        temp = []
        for item in comb:
            if item not in temp:
                temp.append(item)

        items_mapping = {item: ascii_uppercase[i] for i, item in enumerate(temp)}
        pattern = "".join([items_mapping[item] for item in comb])
        rules.append(pattern)

    df = pd.DataFrame(all_combs)
    df = df.sort_values(list(df.columns)).reset_index(drop=True)
    df_cols = [f"item_{i+1}" for i in range(df.shape[1])]
    df.columns = df_cols
    df["rule"] = rules
    df.sort_values(["rule"] + df_cols, inplace=True)

    # df.groupby("rule").size().sort_values(ascending=False)
    combs_sizes = df.groupby("rule").size().sort_values(ascending=False)
    min_number_combs = df.groupby("rule").size().min()
    min_number_combs

    return df, combs_sizes


def generate_all_combinations4(
    shapes: List[str],
    patterns: List[List[str]],
    stim_IDs: Dict[str, int],
    online_format=False,
):
    # * Determine the maximum number of unique characters in any pattern
    patterns = [(p[0] + p[1]).replace(" ", "") for p in patterns]
    all_letters = set(sorted("".join(patterns)))

    # * Generate all unique permutations of shapes for the maximum number of characters
    n_shapes = len(shapes)
    n_perms = int(factorial(n_shapes) / factorial(n_shapes - len(all_letters)))
    shape_permutations = permutations(shapes, len(all_letters))

    all_combs = set()

    for perm in tqdm(shape_permutations, total=n_perms):
        shape_dict = dict(zip(sorted(all_letters), perm))
        for pattern in patterns:
            comb = [shape_dict[char] for char in pattern]
            comb = [stim_IDs[name] for name in comb]
            comb += [pattern]
            all_combs.add(tuple(comb))

    n_combs = len(all_combs)
    print(f"{n_combs} unique combinations generated.")

    return n_combs, (c for c in all_combs)


def get_choices_and_pres_order(
    combs, stim_IDs, n_combs: int, online_format: str = True
):
    stim_IDs = list(stim_IDs.values())

    if online_format:
        # seq_order = list(range(1, 8))
        # choices_order = list(range(1, 5))
        seq_order = list(range(7))
        choices_order = list(range(4))

    if online_format:
        combs_arr = np.zeros((n_combs, 15), dtype="object")
    else:
        combs_arr = np.zeros((n_combs, 13), dtype="object")

    for idx, comb in enumerate(tqdm(combs, total=n_combs)):
        comb = list(comb)
        pattern = comb.pop()
        unique_shapes = list(set(comb))
        n_unique = len(unique_shapes)

        solution = comb[-1]

        if n_unique < 4:
            choices = [i for i in stim_IDs if i not in unique_shapes]
            choices = np.random.choice(choices, 4 - n_unique, replace=False).tolist()
            choices += unique_shapes
        elif n_unique > 4:
            choices = [i for i in unique_shapes if i != solution]
            choices = np.random.choice(choices, 3, replace=False).tolist()
            choices += [solution]
        else:
            choices = unique_shapes

        np.random.shuffle(choices)

        if online_format:
            np.random.shuffle(seq_order)
            np.random.shuffle(choices_order)

            seq_order_str = "".join([str(i) for i in seq_order])
            choices_order_str = "".join([str(i) for i in choices_order])
            comb = (
                comb
                + choices
                + [seq_order_str]  # [int(seq_order_str)]
                + [choices_order_str]  # [int(choices_order_str)]
                + [pattern]
            )
            combs_arr[idx] = comb
        else:
            comb = comb + choices + [pattern]
            combs_arr[idx] = comb

    return (i for i in combs_arr)


def get_experiment_sequences(db, stim_IDs, n_by_pattern=6, masked_image_inds=None, save_dir=None):
    # TODO: implement masked_img_idx if index of list of indices is given

    # stim_IDs = list(stim_IDs.values())
    stim_IDs_inv = {v: k for k, v in stim_IDs.items()}
    stim_IDs_list = list(stim_IDs.values())

    command = "SELECT pattern \nFROM combinations \nGROUP BY pattern;"
    results = db.execute(command)
    unique_patterns = [i[0] for i in results]

    cols = [i[1] for i in db.execute("PRAGMA table_info(combinations);")]
    n_cols = len(cols)

    combinations = np.zeros((n_by_pattern * len(unique_patterns), n_cols), dtype="object")

    for i, pattern in enumerate(tqdm(unique_patterns)):
        command = f"SELECT * FROM combinations WHERE pattern='{pattern}' "
        command += f"ORDER BY RANDOM() LIMIT {n_by_pattern};"
        samples = db.execute(command)
        combinations[i * n_by_pattern : (i + 1) * n_by_pattern] = samples

    icon_col_inds = [i for i, c in enumerate(cols) if "figure" in c]

    seq_order = list(range(8))
    choices_order = list(range(4))

    sequences = []

    for row in combinations:
        combination_ID = row[0]
        pattern = row[-1]
        seq_icon_IDs = list(row[icon_col_inds])
        unique_icons = list(set(seq_icon_IDs))
        n_unique = len(unique_icons)

        if masked_image_inds is None:
            masked_image_idx = np.random.choice([0, 3, 4, 7])
        else:
            masked_image_idx = np.random.choice(masked_image_inds)

        solution = seq_icon_IDs[masked_image_idx]

        if n_unique < 4:
            choices = [i for i in stim_IDs.values() if i not in unique_icons]
            choices = np.random.choice(choices, 4 - n_unique, replace=False).tolist()
            choices += unique_icons
        elif n_unique > 4:
            choices = [i for i in unique_icons if i != solution]
            choices = np.random.choice(choices, 3, replace=False).tolist()
            choices += [solution]
        else:
            choices = unique_icons

        np.random.shuffle(choices)
        np.random.shuffle(seq_order)
        np.random.shuffle(choices_order)

        # * Move the image to be masked to the end of the presentation order
        seq_order.append(seq_order.pop(seq_order.index(masked_image_idx)))

        seq_order_str = "".join([str(i) for i in seq_order])
        choices_order_str = "".join([str(i) for i in choices_order])

        formatted_row = [combination_ID] + seq_icon_IDs + choices
        formatted_row += [masked_image_idx, seq_order_str, choices_order_str, pattern]
        sequences.append(formatted_row)
   
    cols = [
        "combinationID",
        "figure1",
        "figure2",
        "figure3",
        "figure4",
        "figure5",
        "figure6",
        "figure7",
        "figure8",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "maskedImgIdx",
        "seq_order",
        "choice_order",
        "pattern",
    ]
    
    sequences = pd.DataFrame(sequences)
    sequences.columns = cols

    fname1 = "sequences-format[IDs].csv"
    fname2 = "sequences-format[names].csv"

    sequences.to_csv(fname1, index=False)

    mapped_cols = [c for c in cols if re.search(r"(figure\d{1,2})|(choice\d{1,2})", c)]

    transformed = sequences.loc[:, mapped_cols].replace(stim_IDs_inv)
    sequences.loc[:, mapped_cols] = transformed
    sequences.to_csv(fname2, index=False)

    # db.insert_sequences(#TODO)
    
    # formatted_sequences = formatted_sequences.sample(frac=1)

    # pd.read_csv(fname1)
    # pd.read_csv(fname2)

    # return (i for i in combs_arr)


def get_mappings(stims, patterns):
    stim_IDs = {stim_name: i for i, stim_name in enumerate(sorted(stims))}

    pattern_IDs = sorted(["".join(i[0].replace(" ", "") + i[1]) for i in patterns])
    pattern_IDs = {pattern: i for i, pattern in enumerate(pattern_IDs)}

    return stim_IDs, pattern_IDs


def scale_images(
    images: List[str],
    scale: int = None,
    res: int = None,
    absolute: List[int] = None,
    min_dims: List[int] = None,
    max_dims: List[int] = None,
    save_dir: str = None,
    return_imgs: bool = False,
):
    """_summary_

    Args:
        images (List[str]): _description_
        scale (int, optional): _description_. Defaults to None.
        res (int, optional): _description_. Defaults to None.
        absolute (List[int], optional): _description_. Defaults to None.
        min_dims (List[int], optional): _description_. Defaults to None.
        max_dims (List[int], optional): _description_. Defaults to None.
        save_dir (str, optional): _description_. Defaults to None.
        return_imgs (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    images = [Path(img) for img in images]

    if not bool(scale) ^ bool(absolute):
        raise ValueError(
            "Either a scaling factor (`scale`) or absolute dimensions (`absolute`) must be provided, not both."
        )

    if scale is not None:
        if res is not None:
            new_dims = [int(res * scale)] * 2
        else:
            raise ValueError(
                "If scale is provided, resolution must be provided as well."
            )

    if absolute is not None:
        new_dims = absolute

    if min_dims is not None:
        min_check = all([dim1 >= dim2 for dim1, dim2 in zip(new_dims, min_dims)])
    else:
        min_check = True
    if max_dims is not None:
        max_check = all([dim1 <= dim2 for dim1, dim2 in zip(new_dims, max_dims)])
    else:
        max_check = True

    if all([min_check, max_check]):
        new_imgs = [Image.open(img).resize(new_dims) for img in images]

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

            for img, img_path in zip(new_imgs, images):
                img.save(save_dir / f"{img_path.stem}.png")
        if return_imgs:
            return new_imgs

    else:
        raise ValueError(
            f"New dimensions {new_dims} must be between {min_dims} and {max_dims}. "
            "Readjust the scale or min/max dimensions."
        )


def standardize_images(images: Dict[str, str], dest_dir: Union[str, Path]):
    dest_dir = Path(dest_dir)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    images = {k: Image.open(v) for k, v in images.items()}

    pix_counts = get_pixel_counts(images)

    # * Identify the image with the highest count of black pixels
    max_black_pixels = pix_counts["black"].max()

    # * Add a column for the resizing factor
    # * The resizing factor is calculated as the square root of the ratio of the max black pixels to the current black pixels
    pix_counts["resize_factor"] = (max_black_pixels / pix_counts["black"]) ** 0.5
    pix_counts["size"] = [im.size for im in images.values()]

    pix_counts["new_black"] = [None] * len(pix_counts)
    new_sizes = []

    new_images = {}

    for row, values in pix_counts.iterrows():
        size = values["size"]
        new_size = tuple([int(i * values["resize_factor"]) for i in size])
        new_sizes.append(new_size)

        new_img = images[values["img_name"]].resize(new_size)

        arr = np.asarray(new_img)
        converted = np.where(arr > 0, 255, 0)
        unique, counts = np.unique(converted, return_counts=True)
        black_pixels = counts[1]

        pix_counts.loc[row, "new_black"] = black_pixels

        new_images[values["img_name"]] = new_img

    new_max_black_pixels = pix_counts["new_black"].max()
    pix_counts["new_size"] = new_sizes
    pix_counts["new_ratio"] = pix_counts["new_black"] / new_max_black_pixels
    pix_counts["new_ratio"] = pix_counts["new_ratio"].astype(float)
    pix_counts["new_ratio"] = pix_counts["new_ratio"].round(3)

    max_new_size = pix_counts["new_size"].max()[0]

    for name, im in new_images.items():
        background = Image.new("RGBA", (max_new_size, max_new_size), (255, 255, 255, 1))
        size = im.size[0]
        offset = (max_new_size - size) // 2
        background.paste(im, (offset, offset))
        background.save(dest_dir / f"{name}.png")
    return pix_counts


def get_samples_per_pattern(n=6, output_sequences=False):
    img_dir = Path("global_config/images/original")
    imgs_dict = {im.stem: im for im in img_dir.glob("*.png")}
    shapes = list(imgs_dict.keys())
    shapes.pop(shapes.index("question-mark"))

    with open("global_config/rules.json", "r") as f:
        rules = json.load(f)

    stim_IDs, pattern_IDs = get_mappings(shapes, rules)
    stim_IDs_inv = {v: k for k, v in stim_IDs.items()}
    pattern_IDs_inv = {v: k for k, v in pattern_IDs.items()}

    db_file = Path("global_config/database.db")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    command = "SELECT pattern \nFROM combinations \nGROUP BY pattern;"
    unique_patterns = [i[0] for i in c.execute(command).fetchall()]

    n_cols = 16
    selected_combs = np.zeros((n * len(unique_patterns), n_cols), dtype="object")

    for i, pattern in enumerate(tqdm(unique_patterns)):
        # print(i, pattern)
        # print(i * n, (i + 1) * n,'\n')
        command = f'SELECT * FROM combinations WHERE pattern="{pattern}" ORDER BY RANDOM() LIMIT {n};'
        samples = c.execute(command).fetchall()
        selected_combs[i * n : (i + 1) * n] = samples

    transformed = [[stim_IDs_inv[i] for i in row[1:13]] for row in selected_combs]
    transformed = np.array(transformed)
    selected_combs[:, 1:13] = transformed

    selected_combs = pd.DataFrame(selected_combs)
    columns = [c.execute(f"PRAGMA table_info('combinations');").fetchall()][0]
    columns = [i[1] for i in columns]
    selected_combs.columns = columns

    conn.close()

    fname1 = "selected_combinations-format[names].csv"
    fname2 = "selected_combinations-format[IDs].csv"

    selected_combs.to_csv(fname1, index=False)
    selected_combs.replace(stim_IDs).to_csv(fname2, index=False)
    selected_combs = selected_combs.sample(frac=1)

    # pd.read_csv(fname1)
    # pd.read_csv(fname2)
    nzfill = len(str(selected_combs.shape[0]))

    if output_sequences:
        seq_dir = Path("global_config/draft/sequences")

        reset_dir(seq_dir)

        txt = ""

        loaded_images = {im_name: Image.open(im) for im_name, im in imgs_dict.items()}

        sequences = selected_combs.iterrows()
        sequences = tqdm(sequences, total=len(selected_combs))

        for seqN, (idx, seq) in enumerate(sequences):
            fig, axs = plt.subplots(2, 8, figsize=(12, 4))
            for ax in axs.ravel():
                ax.set_axis_off()

            items = seq[1:8]
            for i, item in enumerate(items):
                axs[0][i].imshow(loaded_images[item])

            axs[0][7].imshow(loaded_images["question-mark"])

            unique_shapes = items.unique().tolist()
            n_unique = len(unique_shapes)

            if n_unique < 4:
                choices = [i for i in stim_IDs.keys() if i not in unique_shapes]
                choices = np.random.choice(
                    choices, 4 - n_unique, replace=False
                ).tolist()
                choices += unique_shapes
            elif n_unique > 4:
                choices = [i for i in unique_shapes if i != seq["solution"]]
                choices = np.random.choice(choices, 3, replace=False).tolist()
                choices += [seq["solution"]]
            else:
                choices = unique_shapes

            np.random.shuffle(choices)

            items_str = ", ".join(items)
            choices_str = ", ".join(choices)
            txt += f"ID: {idx}\nSequence Number: {seqN}\nSequence: {items_str}\nChoices: {choices_str}\n\n"

            for i, item in enumerate(choices):
                axs[1][i + 2].imshow(loaded_images[item])

            pattern = seq["pattern"]
            itemid = seq["itemid"]
            img_name = f"stim_{str(seqN).zfill(nzfill)}-{pattern}-{itemid}.png"
            plt.savefig(seq_dir / img_name, bbox_inches="tight")
            # plt.show()
            plt.close("all")

        selected_combs[["itemid", "solution"]].to_csv(
            seq_dir / "solutions.csv", index=False
        )
        selected_combs.iloc[:, :13].to_csv(seq_dir / "sequences.csv", index=False)

        with open(seq_dir / "sequences.txt", "w") as f:
            f.write(txt)

        instructions_img = """I'm going to present you images containing each two rows of black icons. The top row shows a sequence of icons arranged in a logical order, followed by a question mark. Your goal is to continue the sequence with the correct icon out of the four options available on the bottom row. Do not try to use OCR, manually inspect each image. 
        I want you to create a csv file and append each of your choices to it. The csv file must contain two columns, the first one with the image ID, which is the name of the image file without the extension, and the second one with your answer to each image. 
        Here are the names of all the icons you might see:
        shopping_cart, ice-skate, headphones, cube, peach, bell, alarm-clock, apps, rugby, settings, key, island-tropical, lock, wheat, playing-cards, spade, paw, chess, box-open, carrot, skiing, cocktail-alt, home, user, mug-hot-alt, plane-alt, hand-horns, megaphone, bulb, broadcast-tower, music-alt, search, heart, gamepad, social-network, shopping-basket, brightness, rocket, globe, gift, eye, truck-side, diamond, hammer, bone, star, guitar, graduation-cap, pyramid, phone-call, club, camera, baby-carriage, paper-plane, trophy-star, candy-cane, smile, fish, helicopter-side, biking"""
        print(instructions_img)

        instructions_txt = """I'm going to present you sequences of words arranged in a logical order. Your goal is to continue the sequence with the correct icon out of the four options available.
        Do not use python to analyze the sequence, reason by yourself to find the pattern in each sequence. 
        I want you to create a csv file and append each of your choices to it. The csv file must contain two columns, the first one with sequence ID and the second one with your answer to each sequence."""

        print(f"{instructions_txt}\n\n{txt}")
        pyperclip.copy(f"{instructions_txt}\n\n{txt}")

        instructions_grade = "here are the solutions, compare your answers to these and grade yourself accordingly. Additionally, for the answers that you got wrong, give me the ID of the corresponding sequences."
        pyperclip.copy(instructions_grade)

    return selected_combs


def generate_sequences(item_ids: List[int]):

    sequences = pd.read_csv(
        "global_config/new_combs/selected_combinations-format[names].csv"
    )
    item_ids = sequences["itemid"].tolist()
    # img_dir = Path("global_config/images/original")
    # imgs_dict = {im.stem: im for im in img_dir.glob("*.png")}
    # shapes = list(imgs_dict.keys())
    # shapes.pop(shapes.index("question-mark"))

    # with open("global_config/rules.json", "r") as f:
    #     rules = json.load(f)

    # stim_IDs, pattern_IDs = get_mappings(shapes, rules)
    # stim_IDs_inv = {v: k for k, v in stim_IDs.items()}
    # pattern_IDs_inv = {v: k for k, v in pattern_IDs.items()}

    # maskedImageIdx = np.random.choice([0, 2, 3, 7], sequences.shape[0])
    # solution = []
    # for masked_image, (idx_row, row) in zip(
    #     maskedImageIdx, sequences.iloc[:, 1:9].iterrows()
    # ):
    #     solution.append(row.iloc[masked_image])
    # sequences.insert(9, "maskedImageIdx", maskedImageIdx)
    # sequences.insert(10, "solution", solution)
    # sequences.to_csv(config_dir / "selected_combinationsNEW-format[names].csv", index=False)
    # sequences.iloc[:, 1:15] = sequences.iloc[:, 1:15].replace(config["stim_IDs"])
    # sequences.to_csv(config_dir / "selected_combinationysNEW-format[IDs].csv", index=False)
    db_file = Path("global_config/database.db")
    db = Database(db_file)
    item_ids = ",".join([str(i) for i in item_ids])
    column_names = [i[1] for i in db.execute("PRAGMA table_info(combinations);")]
    sequences = db.execute(f"SELECT * FROM combinations WHERE itemid IN ({item_ids})")
    sequences = pd.DataFrame(sequences, columns=column_names)

    return selected_combs


def count_patterns():
    # stim_IDs, pattern_IDs = get_mappings(shapes, rules, export=True)
    db_file = Path("./config/database.db")
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    tb_name = "combinations"
    pattern_IDs_inv = {v: k for k, v in pattern_IDs.items()}
    command = f"SELECT pattern, COUNT(pattern) FROM {tb_name} GROUP BY pattern"

    n_per_rule = c.execute(command).fetchall()
    # n_per_rule = {
    #     pattern_IDs_inv[rule]: f"{count:,}" for rule, count in n_per_rule
    # }
    n_per_rule = {rule: count for rule, count in n_per_rule}
    min_n = min(n_per_rule.values())
    n_trials = 120
    trials_per_rule = n_trials / len(n_per_rule)


def validity_check(n=5):
    stim_dir = wd / "stimuli"

    problems = sorted([img for img in stim_dir.rglob("*problems*.png")])
    solutions = sorted([img for img in stim_dir.rglob("*solutions*.png")])
    problems = np.array(problems)
    solutions = np.array(solutions)

    selected = np.random.choice(len(problems), n, replace=False)
    selected.sort()

    problems = problems[selected]
    solutions = solutions[selected]

    for problem, solution in zip(problems, solutions):
        problem_id = problem.stem.split("_")[0]

        img_problem = Image.open(problem)
        img_solution = Image.open(solution)
        size = img_problem.size

        background = Image.new("RGBA", (size[0], size[1] * 2), (255, 255, 255, 0))
        background.paste(img_problem, (0, 0))
        background.paste(img_solution, (0, size[1]))

        plt.imshow(background)
        plt.title(problem_id)
        plt.axis("off")
        plt.show()
        plt.close()
