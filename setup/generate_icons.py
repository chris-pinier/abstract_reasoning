import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from tqdm.auto import tqdm

WD = Path.cwd()

shapes = [np.random.randint(0, 2, (4, 4)) for _ in range(10)]

# for shape in shapes:
#     shape[np.where(shape==1)] = 255
# plt.imshow(shape)
# plt.show()
# plt.close()

patterns = ["AAABAAAB", "ABCDABCD", "ABBAABBA", "ABBACDDC"]


def plot_shape(shape: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=500)
    ax.imshow(shape, cmap="gray_r")
    ax.set_axis_off()
    plt.axis("off")
    plt.show()


def plot_shapes(shapes: List[np.ndarray], one_row: bool = False):
    if one_row:
        fig, axes = plt.subplots(1, len(shapes), dpi=500)
        for i, ax in enumerate(axes):
            ax.imshow(shapes[i], cmap="gray_r")
            ax.set_axis_off()
        plt.show()

    else:
        blank_shape = np.zeros_like(shapes[0])
        n_cols = 5
        n_rows = len(shapes) // n_cols

        if n_rows == 0 or len(shapes) % (n_rows * n_cols) != 0:
            n_rows += 1

        fig, axes = plt.subplots(n_rows, n_cols, dpi=500)

        for i, ax in enumerate(axes.flatten()):
            if i < len(shapes):
                ax.imshow(shapes[i], cmap="gray_r")
            else:
                ax.imshow(blank_shape, cmap="gray_r")
            ax.set_axis_off()

        plt.show()


def generate_shapes(
    n: int,
    shape_size: tuple,
    pix_constraints: Optional[Tuple[str, Tuple[float, float]]] = None,
):
    # ! TEMP
    # n, shape_size, pix_constraints = 21, (4, 4), ("sum", [7, 7])
    # n = 41
    # shape_size = (4, 4)
    # pix_constraints = ("sum", [7, 9])
    # ! TEMP

    shapes = []

    if pix_constraints is not None:
        assert pix_constraints[0] in ["sum", "mean"], (
            "pix_constraints must be either 'sum' or 'mean'."
        )

        if pix_constraints[0] == "sum":
            for i in range(n):
                shape = np.random.randint(0, 2, shape_size)
                n_black_pix = np.sum(shape)

                while pix_constraints[1][0] >= n_black_pix >= pix_constraints[1][1]:
                    shape = np.random.randint(0, 2, shape_size)
                    n_black_pix = np.sum(shape)

                shapes.append(shape)

    plot_shapes(shapes, one_row=False)

    return shapes


def generate_sequence(pattern: str, shapes: list):
    sequence = list(pattern)
    unique_els = np.unique(sequence)
    assert len(shapes) >= len(unique_els), "Not enough shapes"

    selected_shape_inds = np.random.choice(range(len(shapes)), len(unique_els))
    selected_shapes = [shapes[i] for i in selected_shape_inds]

    els_map = dict(zip(unique_els, selected_shapes))
    return [els_map[el] for el in sequence]


def plot_sequence(seq: list):
    fig, axes = plt.subplots(1, len(seq), figsize=(12, 12), dpi=500)
    for i, ax in enumerate(axes):
        ax.imshow(seq[i], cmap="gray_r")
        ax.set_axis_off()
    plt.show()


def save_shapes(shapes: List[np.ndarray], save_dir: Path):
    save_dir.mkdir(exist_ok=True, parents=True)

    zfill_len = len(str(len(shapes)))

    for i, shape in enumerate(tqdm(shapes)):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=500)
        ax.imshow(shape, cmap="gray_r")
        ax.set_axis_off()
        plt.axis("off")
        plt.savefig(save_dir / f"fig{i:0{zfill_len}}.png", dpi=500)
        plt.close()

    np.save(save_dir / "shapes.npy", np.array(shapes))


def load_shapes_img(shapes_dir: Path):
    shape_files = [f for f in shapes_dir.glob("*.png") if not f.name.startswith(".")]

    shape_arrs = []

    for shape_file in shape_files:
        img = Image.open(shape_file)
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb)

        shape_arrs.append(img_array)

    return shape_arrs


def load_shapes_arr(shapes_dir):
    pass


def encode_shapes(shape_ids: List[int], n, r, shuffle: bool = True):
    # ! TEMP
    # n = 5
    # r = 3
    # ! TEMP

    from math import factorial
    from itertools import permutations

    assert n >= r, "n must be greater than r"
    assert n > 0, "n must be greater than 0"
    assert r > 0, "r must be greater than 0"

    n_permut = factorial(n) / factorial(n - r)

    encoded_ids = list(permutations(range(n), r))
    if shuffle:
        np.random.shuffle(encoded_ids)

    assert len(encoded_ids) >= len(shape_ids)

    mapping = dict(zip(shape_ids, encoded_ids))

    return mapping


shapes = generate_shapes(13, (4, 4), ("sum", [7, 8]))

# shapes = np.array(shapes)

fig, ax = plt.subplots(figsize=(12, 12), dpi=500)
pattern = patterns[2]

seq = generate_sequence(pattern, shapes)
plot_sequence(seq)

save_shapes(shapes, WD / "test")
# np.random.choice(range(len(shapes)), unique_els)

shape_arrs = load_shapes_img(WD / "test")
a = shape_arrs[0].copy()

# a[np.where(a == [255, 255, 255])] = 120
a[np.where(a == [0, 0, 0])] = 120

a[np.where(a[:, :, 2] == 0)] = 120
plt.imshow(a)

np.unique(np.where(a == [0, 0, 0])[0]).shape

shapes_arr = load_shapes_arr(WD / "test")
shape_imgs = load_shapes_img(WD / "test")

img = shape_imgs[0].copy()
img[np.where(shape_imgs[0][:, :, :] == 0)[:2]] = [250, 250, 0]
plt.imshow(img)

img[0, 0] == [255, 255, 255]

img.shape
