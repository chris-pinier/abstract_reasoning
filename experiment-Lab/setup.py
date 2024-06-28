import os
import platform
from pathlib import Path
from typing import Union
from icecream import ic
from PIL import Image
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List


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
    dest_dir: str = None,
    return_imgs: bool = True,
):
    """_summary_

    Args:
        images (List[str]): _description_
        scale (int, optional): _description_. Defaults to None.
        res (int, optional): _description_. Defaults to None.
        absolute (List[int], optional): _description_. Defaults to None.
        min_dims (List[int], optional): _description_. Defaults to None.
        max_dims (List[int], optional): _description_. Defaults to None.
        dest_dir (str, optional): _description_. Defaults to None.
        return_imgs (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    images = {img.stem: Path(img) for img in images}

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
        new_imgs = {
            name: Image.open(img).resize(new_dims) for name, img in images.items()
        }

        if dest_dir is not None:
            # dest_dir = Path(dest_dir)
            # if dest_dir.exists():
            #     shutil.rmtree(dest_dir)
            # dest_dir.mkdir(parents=True)

            for name, img in new_imgs.items():
                img.save(dest_dir / f"{name}.png")
        if return_imgs:
            return new_imgs

    else:
        raise ValueError(
            f"New dimensions {new_dims} must be between {min_dims} and {max_dims}. "
            "Re-adjust the scale or min/max dimensions."
        )


def standardize_images(
    images: List[str],
    dest_dir: Union[str, Path] = None,
    return_imgs: bool = True,
):
    imgs_dict = {Path(img).stem: Image.open(img) for img in images}

    pix_counts = get_pixel_counts(imgs_dict)

    # * Determine highest number of black pixels across images
    max_black_pixels = pix_counts["black"].max()

    # * Add a column for the resizing factor
    # * The resizing factor is calculated as the square root of the ratio of the
    # * max_black_pixels to the number of black pixels in each image
    pix_counts["resize_factor"] = (max_black_pixels / pix_counts["black"]) ** 0.5
    pix_counts["size"] = [im.size for im in imgs_dict.values()]

    pix_counts["new_black"] = [None] * len(pix_counts)

    new_sizes = []
    new_images = {}

    for row, values in pix_counts.iterrows():
        size = values["size"]
        new_size = tuple([int(i * values["resize_factor"]) for i in size])
        new_sizes.append(new_size)

        new_img = imgs_dict[values["img_name"]].resize(new_size)

        arr = np.asarray(new_img)
        converted = np.where(arr > 0, 255, 0)
        _, counts = np.unique(converted, return_counts=True)
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
        rescaled_img = Image.new(
            "RGBA", (max_new_size, max_new_size), (255, 255, 255, 1)
        )
        size = im.size[0]
        offset = (max_new_size - size) // 2
        rescaled_img.paste(im, (offset, offset))
        new_images[name] = rescaled_img

    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(exist_ok=True)

        for name, img in new_images.items():
            img.save(dest_dir / f"{name}.png")

    if return_imgs:
        return new_images, pix_counts
    else:
        return pix_counts


def setup_stimuli(
    root_dir: Union[str, Path], res: tuple, max_height: int, max_items: int
):
    """
    root_dir: path to experiment directory (either lab, online or ANNs)
    res: monitor resolution in pixels
    max_height:
    max_items:
    """
    ic.enable()

    root_dir = Path(root_dir)
    dest_dir = root_dir / "images"
    config_dir = root_dir.parent / "config"
    config_file = config_dir / "experiment_config.json"
    source_imgs_dir = config_dir / "images/original"

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    original_imgs = [im for im in source_imgs_dir.glob("*.png")]
    im_size = Image.open(original_imgs[0]).size

    # * Standardize images to have ~ same number of black pixels
    pix_counts = standardize_images(original_imgs, dest_dir=dest_dir, return_imgs=False)

    pix_counts.sort_values("new_ratio", ascending=False, inplace=True)
    # pix_counts["new_ratio"].hist()
    # plt.title("black pixels distribution (standardized)")
    # plt.show()
    # df_imgs.to_excel(root_dir / "images/images_info.xlsx")  # , float_format="%.3f")

    # * Resize images based on screen resolution
    resize_factor_x = (res[0] / max_items) / im_size[0]
    resize_factor_y = max_height / im_size[1]
    resize_factor = min(resize_factor_x, resize_factor_y)

    new_size = [int(im_size[0] * resize_factor)] * 2
    ic(new_size)

    stdz_imgs = list(dest_dir.glob("*.png"))
    scaled_images = scale_images(original_imgs, absolute=new_size, dest_dir=dest_dir)

    pix_counts = get_pixel_counts(scaled_images)
    # pix_counts["black"].hist()
    # plt.title("black pixels distribution (scaled)")
    # plt.show()


# if __name__ == "__main__":
# root_dir =
# setup_stimuli(root_dir)

# setup_structure(root_dir)

# print("Setup complete.")
