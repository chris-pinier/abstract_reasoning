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
from typing import Dict, List, Tuple, Union
import pathlib


def count_black_pixels(img: Image.Image) -> np.ndarray:
    """
    img: PIL image object, mode: RGBA. Assumes black pixels have alpha value of 255 and
    transparent pixels have alpha value of 0.
    """
    assert img.mode == "RGBA", "Image mode must be RGBA."

    img_array = np.array(img)

    return np.sum(img_array[:, :, 3] == 255)


def prepare_images(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    size: Tuple[int, int] = None,
    max_items: int = None,
    blank_space_pct: float = 0.05,
    resolution: Tuple[int, int] = None,
) -> Dict[str, float]:
    """
    Resize images to a standard size while maintaining the aspect ratio. The images are
    padded with transparent pixels to fill the standard size. Images are each scaled
    down by a ratio based on the image with the minimum number of black pixels.
    """

    # wd = Path(__file__).parent  # ! TEMP
    # input_folder = wd.parent / "config" / "images" / "original copy" # ! TEMP
    # output_folder = wd /"test_images" # ! TEMP

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # original_imgs = [im for im in input_folder.glob("*.png")]
    # im_size = Image.open(original_imgs[0]).size

    if size is not None:
        target_size = size
    else:
        # conds = [arg is not None for arg in [max_height, max_width, resolution]]
        # assert all(
        #     conds
        # ), "Either size or all of max_height, max_width, and resolution must be provided."
        # # * Resize images based on screen resolution
        # resize_factor_x = max_width / im_size[0]
        # resize_factor_y = max_height / im_size[1]
        # resize_factor = min(resize_factor_x, resize_factor_y)
        width, height = resolution
        max_img_width = int(np.floor(width / max_items * (1 - blank_space_pct)))
        max_img_height = int(np.floor(height / 3))
        target_size = [min(max_img_width, max_img_height)] * 2

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

    if output_folder.exists():
        imgs = {img.stem: Image.open(img) for img in output_folder.glob("*.png")}
        images_sizes = set([img.size for img in imgs.values()])

        if len(images_sizes) == 1:
            if images_sizes.pop() == target_size:
                black_pix_counts = {img: count_black_pixels(imgs[img]) for img in imgs}
                black_pix_counts_arr = np.array(list(black_pix_counts.values()))
                black_pix_counts_arr = black_pix_counts_arr / black_pix_counts_arr.max()
                black_pix_counts = dict(
                    zip(black_pix_counts.keys(), black_pix_counts_arr)
                )
                black_pix_counts = dict(
                    sorted(black_pix_counts.items(), key=lambda x: x[1])
                )
                return black_pix_counts
            else:
                shutil.rmtree(output_folder)
        else:
            shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    # * First pass: resize all images and find the minimum black pixel count
    min_black_pixels = float("inf")
    resized_images = []

    for filename in input_folder.glob("*.png"):
        img = Image.open(filename)
        img = img.resize(target_size, Image.LANCZOS)
        black_pixels = count_black_pixels(img)
        min_black_pixels = min(min_black_pixels, black_pixels)
        resized_images.append((filename, img, black_pixels))

    # print(f"Target black pixels: {min_black_pixels}")

    # * Second pass: scale images based on the minimum black pixel count
    black_pix_counts = {}
    for filename, img, black_pixels in resized_images:
        # * Calculate scaling factor
        scale_factor = np.sqrt(min_black_pixels / black_pixels)

        # * Scale the image
        new_size = tuple(int(dim * scale_factor) for dim in img.size)
        scaled_img = img.resize(new_size, Image.LANCZOS)

        # * Create a new blank image of target size
        final_img = Image.new("RGBA", target_size, color=(0, 0, 0, 0))

        # * Calculate padding
        left = (target_size[0] - new_size[0]) // 2
        top = (target_size[1] - new_size[1]) // 2

        # * Paste the scaled image onto the blank image
        final_img.paste(scaled_img, (left, top))
        black_pix_counts[filename.stem] = count_black_pixels(final_img)

        # * Save the processed image
        final_img.save(output_folder / f"{filename.stem}.png")

    black_pix_counts_arr = np.array(list(black_pix_counts.values()))
    black_pix_counts_arr = black_pix_counts_arr / black_pix_counts_arr.max()
    black_pix_counts = dict(zip(black_pix_counts.keys(), black_pix_counts_arr))
    black_pix_counts = dict(sorted(black_pix_counts.items(), key=lambda x: x[1]))

    return black_pix_counts


if __name__ == "__main__":
    wd = Path(__file__).parent
    original_imgs = [im for im in (wd.parent / "config/images/original").glob("*.png")]
    img_size = set([Image.open(img).size for img in original_imgs])

    if len(img_size) != 1:
        raise ValueError("All images must have the same size.")
    else:
        img_size = img_size.pop()

    imgs_info = prepare_images(
        wd.parent / "config/images/original",
        wd.parent / "config/images/standardized",
        img_size,
    )

    # fig, axes = plt.subplots(21, 3, figsize=(20, 20))
    # for i, ax in enumerate(axes.flatten()):
    #     if i < len(original_imgs):
    #         img = Image.open(original_imgs[i])
    #         img_arr = np.array(img)
    #         ax.imshow(img_arr)
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()
    # plt.close("all")

    # ratios = list(imgs_info.values())
    # plt.bar(range(len(ratios)), ratios)

    # # img = original_imgs[0]
    # img = [img for img in original_imgs if "question-mark" in img.stem][0]
    # for img in original_imgs:
    #     img = Image.open(img)
    #     img_arr = np.array(img)
    #     img_arr[img_arr[:, :, 3] == 255] = 100

    #     plt.imshow(img_arr)
    #     plt.axis("off")
    #     # plt.tight_layout()
    #     plt.xlim(0, img_arr.shape[0])
    #     plt.ylim(img_arr.shape[1], 0)
    #     plt.show()
    #     plt.close()

##############
