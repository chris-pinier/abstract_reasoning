import os
import platform
from pathlib import Path
from local.database import Database
from typing import Union
from stimuli_images import (
    standardize_images,
    scale_images,
    get_pixel_counts,
    get_mappings,
    generate_all_combinations4,
    get_choices_and_pres_order,
    get_experiment_sequences,
)
from icecream import ic
from PIL import Image
import json
import shutil
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from local.local_utils import get_monitors_info


def manage_directories(directories: list, action: str):
    for directory in directories:
        if action == "create":
            directory.mkdir(parents=True, exist_ok=True)
        elif action == "remove":
            shutil.rmtree(directory)
        elif action == "clean":
            for f in directory.glob("*"):
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
        else:
            raise ValueError(f"Invalid action: {action}")


def setup_stimuli(root_dir: Union[str, Path]):
    ic.enable()

    root_dir = Path(root_dir)

    config_dir = root_dir / "global_config"
    config_file = config_dir / "experiment_config.json"
    img_dir = config_dir / "images/original"

    std_img_dir = config_dir / "images/pixel_standardized"
    resized_dir = config_dir / "images/resized"
    std_img_dir.mkdir(exist_ok=True, parents=True)
    resized_dir.mkdir(exist_ok=True, parents=True)

    monitors = get_monitors_info()
    monitor = [m for m in monitors if m["primary"] == True][0]
    res = monitor["res"]

    with open(config_dir / "rules.json", "r") as f:
        rules = json.load(f)

    imgs_dict = {im.stem: im for im in img_dir.glob("*.png")}

    im_size = Image.open(imgs_dict[list(imgs_dict.keys())[0]]).size

    # * Remove directory containing scaled images if it exists
    if resized_dir.exists():
        shutil.rmtree(resized_dir)

    # * Standardize images to have ~ same number of black pixels
    df_imgs = standardize_images(imgs_dict, dest_dir=std_img_dir)
    standardize_imgs = [im for im in std_img_dir.glob("*.png")]

    df_imgs.sort_values("new_ratio", ascending=False, inplace=True)
    df_imgs["new_ratio"].hist()
    plt.title("black pixels distribution (standardized)")
    plt.show()

    df_imgs.to_excel(config_dir / "images_info.xlsx")  # , float_format="%.3f")

    # * Resize images based on screen resolution
    max_items = 12
    max_img_height = res[1] / 3
    resize_factor_x = (res[0] / max_items) / im_size[0]
    resize_factor_y = max_img_height / im_size[1]
    resize_factor = round(min(resize_factor_x, resize_factor_y), 3)
    new_size = [int(im_size[0] * resize_factor)] * 2
    ic(new_size)

    scale_images(standardize_imgs, absolute=new_size, save_dir=resized_dir)
    scaled_images = {im.stem: Image.open(im) for im in resized_dir.glob("*.png")}

    pix_counts = get_pixel_counts(scaled_images)
    pix_counts["black"].hist()
    plt.title("black pixels distribution (scaled)")
    plt.show()

    bckgrd_color = (255, 255, 255)  # RGB color code for white
    img = Image.new("RGB", new_size, bckgrd_color)

    # Save the image if you want
    img.save(config_dir / "images/blank_image.png")

    # * Generate all possible combinations
    question_mark = imgs_dict.pop("question-mark")

    imgs_list = list(imgs_dict.values())
    shapes = list(imgs_dict.keys())

    n_letters = len(set("".join([(r[0] + r[1]).replace(" ", "") for r in rules])))
    # selected_shapes = np.random.choice(shapes, replace=False, size=n_letters + 12)
    selected_icons = [
        "bone",
        "bulb",
        "camera",
        "eye",
        "gamepad",
        "headphones",
        "heart",
        "home",
        "lock",
        "megaphone",
        "smile",
        "wheat",
    ]

    assert len(selected_icons) >= n_letters

    stim_IDs, pattern_IDs = get_mappings(shapes, rules)
    # stim_IDs_inv = {v: k for k, v in stim_IDs.items()}
    # pattern_IDs_inv = {v: k for k, v in pattern_IDs.items()}

    with open(config_file, "r") as f:
        config = json.load(f)

    # config["pattern_IDs"] = pattern_IDs
    # config["stim_IDs"] = stim_IDs

    # with open(config_file, "w") as f:
    #     json.dump(config, f, indent=4)

    # n_combs, all_combs = generate_all_combinations4(selected_icons, rules, stim_IDs)

    # # * Insert combinations into the database
    # db_file = Path(config_dir / "database.db")
    # db = Database(db_file)

    # db.insert_combinations(all_combs)
    # # db.manage_table("combinations", "count")

    # # db.manage_table("combinations", "empty")

    # stim_IDs = {k: v for k, v in stim_IDs.items() if k in selected_icons}
    # sequences = get_experiment_sequences(db, stim_IDs, n_combs)


def create_symlinks(source, targets):
    """
    Create symbolic links for a source (file or directory) in each of the target directories,
    generalizable over operating systems.

    :param source: The path to the source file or directory.
    :param targets: A list of paths to the target directories where the symlink will be created.
    """
    source_name = os.path.basename(source)
    is_directory = os.path.isdir(source)

    for target in targets:
        target_path = os.path.join(target, source_name)
        if not os.path.exists(target_path):
            try:
                if platform.system() == "Windows":
                    os.symlink(source, target_path, target_is_directory=is_directory)
                else:
                    os.symlink(source, target_path)
                print(f"Created symlink: {target_path} -> {source}")
            except OSError as e:
                print(f"Error creating symlink: {e}")
        else:
            print(f"Symlink already exists: {target_path}")


def setup_structure(root_dir: Union[str, Path]):

    gb_conf_dir = root_dir / "global_config"

    # * ### Create symlinks for images ###
    source_dirs = [gb_conf_dir / "images/resized"]

    target_directories = [
        root_dir / "local/images/",
        root_dir / "online/images/",
    ]

    for target_dir in target_directories:
        target_dir.mkdir(parents=True, exist_ok=True)

    for source_dir in source_dirs:
        create_symlinks(source_dir, target_directories)

    # * ### Create symlinks for configuration files ###
    source_files = [
        gb_conf_dir / "rules.json",
        gb_conf_dir / "database.db",
        gb_conf_dir / "selected_combinations-format[names].csv",
        gb_conf_dir / "selected_combinations-format[IDs].csv",
        gb_conf_dir / "experiment_config.json",
    ]

    target_directories = [
        root_dir / "local/config/",
        root_dir / "online/config/",
        root_dir / "ANNs/config/",
    ]

    for source_f in source_files:
        create_symlinks(source_f, target_directories)

    # * ### Create symlink for blank image ###
    create_symlinks(
        source=gb_conf_dir / "images/blank_image.png",
        targets=[root_dir / "local/images/", root_dir / "online/images/"],
    )


def create_database(root_dir: Union[str, Path], fname: str = "database.db"):
    root_dir = Path(root_dir)

    db_fpath = root_dir / f"global_config/{fname}"

    if db_fpath.exists():
        print(f"Database already exists: {db_fpath}")
    else:
        print(f"Creating database: {db_fpath}")
        db = Database(db_fpath)

    for table in db.tables:
        print(f"TABLE: `{table}`:")
        info = db.manage_table(table, "info")
        print("\n\n")


if __name__ == "__main__":
    try:
        root_dir = Path(__file__).resolve().parent
    except NameError as e:
        print(
            "Something went wrong. Make sure to run this script from the command line.\n"
            "Instructions:"
            "\n\t1. Open a terminal."
            "\n\t2. Navigate to the directory containing this script."
            "\n\t3. Run the script using the command `python setup.py`."
            "\nYou may need to lauch the terminal as an administrator to create symlinks."
        )
        raise e

    setup_stimuli(root_dir)

    setup_structure(root_dir)

    print("Setup complete.")
