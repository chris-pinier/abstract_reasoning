# * IMPORTS
from pathlib import Path
from typing import Dict, Final
from box import Box
import pandas as pd
import os
from IPython.display import display
from pprint import pprint
import plotly.express as px
from contextlib import redirect_stdout
import mne
import sys

WD = Path(__file__).parent.resolve()
ROOT = Path("/".join(WD.parts[: WD.parts.index("abstract_reasoning") + 1]))

sys.path.append(WD)
os.chdir(WD)
assert WD == Path.cwd()

# * RELATIVE IMPORTS
# from analysis_conf import Config as c
# from data_loader.human_data import HumanSessData, HumanSubjData, HumanGroupData
# from utils.analysis_utils import read_file, list_contents
# from analysis_compare_clean import CombinedData
from ar_analysis.data_loader.human_data import (
    HumanSessData,
    HumanSubjData,
    HumanGroupData,
)
from ar_analysis.utils.custom_type_hints import DATA_FMTS
from ar_analysis.utils.analysis_utils import (
    read_file,
    reorder_item_ids,
    list_contents,
    save_figure as save_analysis_figure,
)
from ar_analysis.analysis_rsa import get_ds_and_rdm
from ar_analysis.analysis_config import Config as c

# ! TEMP: to locate and use ffmpeg
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

# * GLOBAL VARIABLES
PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER

DATASET = c.DATASET
SEQ_FILE = c.SEQ_FILE
# DIRECTORIES = c.DIRECTORIES

# SAVE_DISK: Final = Path("/Volumes/Realtek 1Tb")
SAVE_DISK: Final = Path("/Volumes/SSD-512Go")
assert SAVE_DISK.exists(), "WARNING: SSD not connected"
MAIN_DATA_DIR = SAVE_DISK / "PhD Data/experiment1/data/"
DIRECTORIES: Final = Box(
    {
        "ann": {
            "data": MAIN_DATA_DIR
            / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
            "prepro": None,
            "analyzed": MAIN_DATA_DIR / "ANNs/analyzed",
            "export": MAIN_DATA_DIR / "ANNs/analyzed",
        },
        "human": {
            "data": MAIN_DATA_DIR / "Lab/raw",
            "prepro": MAIN_DATA_DIR / "Lab/preprocessed",
            "analyzed": MAIN_DATA_DIR / "Lab/analyzed",
            "export": MAIN_DATA_DIR / "Lab/analyzed",
        },
    }
)
# * GLOBAL VARIABLES
PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER

DATASET = c.DATASET
SEQ_FILE = c.SEQ_FILE
# DIRECTORIES = c.DIRECTORIES

# SAVE_DISK: Final = Path("/Volumes/Realtek 1Tb")
SAVE_DISK: Final = Path("/Volumes/SSD-512Go")
assert SAVE_DISK.exists(), "WARNING: SSD not connected"
MAIN_DATA_DIR = SAVE_DISK / "PhD Data/experiment1/data/"
DIRECTORIES: Final = Box(
    {
        "ann": {
            "data": MAIN_DATA_DIR
            / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
            "prepro": None,
            "analyzed": MAIN_DATA_DIR / "ANNs/analyzed",
            "export": MAIN_DATA_DIR / "ANNs/analyzed",
        },
        "human": {
            "data": MAIN_DATA_DIR / "Lab/raw",
            "prepro": MAIN_DATA_DIR / "Lab/preprocessed",
            "analyzed": MAIN_DATA_DIR / "Lab/analyzed",
            "export": MAIN_DATA_DIR / "Lab/analyzed",
        },
    }
)

# * ----------------------------------------
# * ----------------------------------------
import itertools
import math
import numpy as np
import rsatoolbox
import h5py

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".temp/matplotlib"))
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from rsatoolbox.rdm import calc_rdm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, ttest_rel
from tqdm.auto import tqdm

FIGURE_DPI = 600
RSA_CMAP = "RdBu_r"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": FIGURE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

def temp_file(filename: str) -> Path:
    """Resolve temp files saved either directly in .temp or in .temp/data."""
    direct_path = ROOT / ".temp" / filename
    data_path = ROOT / ".temp/data" / filename
    return direct_path if direct_path.exists() else data_path


cv_rdms_fpath = temp_file("cv_rdms_mean.npy")
fv_rdms_fpath = temp_file("fv_rdms_mean.npy")
resid_rdms_fpath = temp_file("resid_rdms_mean.npy")

cv_rdms = np.load(cv_rdms_fpath)
fv_rdms = np.load(fv_rdms_fpath)
resid_rdms_mean = np.load(resid_rdms_fpath)

fv_rdms.shape
cv_rdms.shape
resid_rdms_mean.shape


rdm_labels = [
    "AAABAAAB",
    "ABABCDCD",
    "ABBAABBA",
    "ABBACDDC",
    "ABBCABBC",
    "ABCAABCA",
    "ABCDDCBA",
    "ABCDEEDC",
]

assert list(c.PATTERNS) == rdm_labels, "Analysis and CV/FV pattern orders differ."


def rdm_upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Return the non-redundant off-diagonal entries of a square RDM."""
    tri_i, tri_j = np.triu_indices(rdm.shape[0], k=1)
    return rdm[tri_i, tri_j]


def pairwise_rdm_spearman(rdms: np.ndarray) -> np.ndarray:
    """Correlate each RDM with every other RDM using Spearman rho."""
    rdm_vecs = np.array([rdm_upper_triangle(rdm) for rdm in rdms])
    corr = np.eye(len(rdm_vecs))
    for i in range(len(rdm_vecs)):
        for j in range(i + 1, len(rdm_vecs)):
            rho = spearmanr(rdm_vecs[i], rdm_vecs[j]).correlation
            corr[i, j] = rho
            corr[j, i] = rho
    return corr


def rdm_correlation(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """Compare two square RDMs using their upper-triangular entries."""
    a = rdm_upper_triangle(rdm_a)
    b = rdm_upper_triangle(rdm_b)
    if method == "spearman":
        return spearmanr(a, b).correlation
    if method == "pearson":
        return np.corrcoef(a, b)[0, 1]
    raise ValueError(f"Unknown method: {method}")


def assert_square_rdm_labels(
    labels: list[str],
    target_labels: list[str] = rdm_labels,
    name: str = "RDM",
) -> None:
    if len(labels) != len(target_labels):
        raise ValueError(
            f"{name} has {len(labels)} labels, expected {len(target_labels)}."
        )
    if set(labels) != set(target_labels):
        missing = set(target_labels) - set(labels)
        extra = set(labels) - set(target_labels)
        raise ValueError(
            f"{name} labels do not match target labels. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}"
        )


def save_figure(fig: plt.Figure, save_path: Path) -> None:
    """Save a figure to the requested path, a matching PDF, and a pickle sidecar."""
    save_analysis_figure(fig, save_path, dpi=FIGURE_DPI)


def rsa_color_norm(vmin: float, vmax: float) -> TwoSlopeNorm:
    """Use a zero-centered diverging scale: negative blue, positive red."""
    return TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


def plot_cv_fv_rdm_correlations(
    fv_rdms: np.ndarray,
    cv_rdms: np.ndarray,
    resid_rdms: np.ndarray | None = None,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Recreate the model RDM correlation heatmap."""
    rdm_stacks = [fv_rdms, cv_rdms]
    labels = [f"fv_{i + 1}" for i in range(fv_rdms.shape[0])] + [
        f"cv_{i + 1}" for i in range(cv_rdms.shape[0])
    ]
    groups = ["fv"] * fv_rdms.shape[0] + ["cv"] * cv_rdms.shape[0]
    if resid_rdms is not None:
        resid_stack = np.asarray(resid_rdms, dtype=float)
        if resid_stack.ndim == 2:
            resid_stack = resid_stack[np.newaxis, :, :]
        rdm_stacks.append(resid_stack)
        labels += [f"resid_{i + 1}" for i in range(resid_stack.shape[0])]
        groups += ["resid"] * resid_stack.shape[0]

    all_rdms = np.concatenate(rdm_stacks, axis=0)
    corr = pairwise_rdm_spearman(all_rdms)

    fig_size = max(8, min(18, len(labels) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85), constrained_layout=True)
    im = ax.imshow(
        corr,
        cmap=RSA_CMAP,
        norm=rsa_color_norm(vmin=-0.15, vmax=1.0),
        interpolation="none",
    )

    ax.set_xticks(np.arange(len(labels)), labels=labels,)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    label_size = 8 if len(labels) <= 40 else 4
    ax.tick_params(axis="both", length=0, labelsize=label_size)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            text_color = (
                "white" if corr[row, col] < 0.2 or corr[row, col] > 0.65 else "0.25"
            )
            if len(labels) <= 40:
                ax.text(
                    col,
                    row,
                    f"{corr[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    for boundary in [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.2)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSA")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax, corr


def plot_residual_rdm_autocorrelation(
    resid_rdms: np.ndarray,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Plot residual-stream RDM autocorrelation across residual indices."""
    resid_stack = np.asarray(resid_rdms, dtype=float)
    if resid_stack.ndim == 2:
        resid_stack = resid_stack[np.newaxis, :, :]
    labels = [f"resid_{i + 1}" for i in range(resid_stack.shape[0])]
    corr = pairwise_rdm_spearman(resid_stack)

    fig_size = max(8, min(18, len(labels) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(
        corr,
        cmap=RSA_CMAP,
        norm=rsa_color_norm(vmin=-0.15, vmax=1.0),
        interpolation="none",
    )

    ax.set_xticks(np.arange(len(labels)), labels=labels,)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(axis="both", length=0, labelsize=4 if len(labels) > 40 else 8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSA")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax, corr


fig, ax, cv_fv_rsa = plot_cv_fv_rdm_correlations(
    fv_rdms=fv_rdms,
    cv_rdms=cv_rdms,
    save_path=ROOT / ".temp/cv_fv_rdm_correlations.png",
)
fig, ax, cv_fv_resid_rsa = plot_cv_fv_rdm_correlations(
    fv_rdms=fv_rdms,
    cv_rdms=cv_rdms,
    resid_rdms=resid_rdms_mean,
    save_path=ROOT / ".temp/cv_fv_resid_rdm_correlations.png",
)
fig, ax, resid_rsa = plot_residual_rdm_autocorrelation(
    resid_rdms=resid_rdms_mean,
    save_path=ROOT / ".temp/resid_rdm_autocorrelation.png",
)

# * ----------------------------------------
# * ----------------------------------------
# * ----------------------------------------
analyzed_dir = DIRECTORIES.human.analyzed
from rsatoolbox.rdm.rdms import load_rdm as load_rsatoolbox_rdm
from rsatoolbox.data.dataset import load_dataset as load_rsatoolbox_dataset

ds_files = dict(
    frp=dict(
        patt=analyzed_dir / "RSA-FRP-Frontal/dataset-human-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir / "RSA-FRP-Frontal/dataset-human-group_avg-sequence_lvl.hdf5",
    ),
    erp=dict(
        patt=analyzed_dir
        / "RSA-Response_ERP-frontal/dataset-human-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir
        / "RSA-Response_ERP-frontal/dataset-human-group_avg-sequence_lvl.hdf5",
    ),
    rest=dict(
        patt=analyzed_dir
        / "RSA-Rest_ERP-frontal/dataset-human-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir
        / "RSA-Rest_ERP-frontal/dataset-human-group_avg-sequence_lvl.hdf5",
    ),
)

rdm_files = dict(
    frp=dict(
        patt=analyzed_dir / "RSA-FRP-Frontal/rdm-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir / "RSA-FRP-Frontal/rdm-group_avg-sequence_lvl.hdf5",
    ),
    erp=dict(
        patt=analyzed_dir
        / "RSA-Response_ERP-frontal/rdm-human-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir
        / "RSA-Response_ERP-frontal/rdm-human-group_avg-sequence_lvl.hdf5",
    ),
    rest=dict(
        patt=analyzed_dir / "RSA-Rest_ERP-frontal/rdm-human-group_avg-pattern_lvl.hdf5",
        seq=analyzed_dir / "RSA-Rest_ERP-frontal/rdm-human-group_avg-sequence_lvl.hdf5",
    ),
)


rdm_files = Box(rdm_files)
ds_files = Box(ds_files)

rdms = {}
for data_type in rdm_files:
    rdms[data_type] = {}
    for rdm_type, rdm in rdm_files[data_type].items():
        rdms[data_type][rdm_type] = load_rsatoolbox_rdm(str(rdm))


datasets = {}
for data_type in ds_files:
    datasets[data_type] = {}
    for rdm_type, ds in ds_files[data_type].items():
        assert ds.exists(), f"{ds}"
        datasets[data_type][rdm_type] = load_rsatoolbox_dataset(str(ds))

rdms = Box(rdms)
datasets = Box(datasets)


def normalize_labels(labels) -> list[str]:
    return [
        label.decode() if isinstance(label, bytes) else str(label) for label in labels
    ]


def get_human_rdm_matrix(
    rdm: rsatoolbox.rdm.RDMs,
    dataset=None,
    target_labels: list[str] = rdm_labels,
) -> np.ndarray:
    """Extract and reorder an 8x8 human RDM to match the CV/FV pattern order."""
    matrices = np.asarray(rdm.get_matrices())
    if matrices.shape[0] != 1:
        matrices = matrices.mean(axis=0, keepdims=True)
    matrix = matrices[0]

    labels = None
    if "patterns" in rdm.pattern_descriptors:
        labels = normalize_labels(rdm.pattern_descriptors["patterns"])
    elif dataset is not None and "patterns" in dataset.obs_descriptors:
        labels = normalize_labels(dataset.obs_descriptors["patterns"])

    if labels is None:
        if matrix.shape == (len(target_labels), len(target_labels)):
            return matrix
        raise ValueError("Could not find pattern labels for human RDM.")

    assert_square_rdm_labels(labels, target_labels=target_labels, name="Human RDM")

    reorder_idx = [labels.index(label) for label in target_labels]
    return matrix[np.ix_(reorder_idx, reorder_idx)]


def cv_fv_similarity_to_dissimilarity(rdms: np.ndarray) -> np.ndarray:
    """Convert model cosine-similarity matrices to dissimilarity matrices."""
    # return 1 - rdms
    return rdms  # ! Note: data was actually given as RDM, not RSM


def as_rdm_stack(rdms: np.ndarray, name: str) -> np.ndarray:
    """Normalize a single RDM or RDM stack to shape n_rdms x n_cond x n_cond."""
    rdms = np.asarray(rdms, dtype=float)
    if rdms.ndim == 2:
        rdms = rdms[np.newaxis, :, :]
    if rdms.ndim != 3:
        raise ValueError(f"{name} must be a 2D RDM or 3D RDM stack; got {rdms.shape}.")
    if rdms.shape[1:] != (len(rdm_labels), len(rdm_labels)):
        raise ValueError(
            f"{name} RDMs must have shape (*, {len(rdm_labels)}, {len(rdm_labels)}); "
            f"got {rdms.shape}."
        )
    return rdms


def add_model_rdm_stack(
    model_rdms: dict[str, np.ndarray],
    prefix: str,
    rdms: np.ndarray,
    include_mean_rdms: bool = True,
) -> None:
    """Add individual model RDMs and their mean RDM to a named RDM dictionary."""
    rdm_stack = cv_fv_similarity_to_dissimilarity(as_rdm_stack(rdms, prefix))
    for i, rdm in enumerate(rdm_stack):
        model_rdms[f"{prefix}_{i + 1}"] = rdm
    if include_mean_rdms:
        mean_label = (
            f"{prefix}_mean_top{rdm_stack.shape[0]}"
            if prefix in {"fv", "cv"}
            else f"{prefix}_mean"
        )
        model_rdms[mean_label] = rdm_stack.mean(axis=0)


def get_cv_fv_model_rdms(
    fv_rdms: np.ndarray,
    cv_rdms: np.ndarray,
    resid_rdms: np.ndarray | None = None,
    include_mean_rdms: bool = True,
) -> dict[str, np.ndarray]:
    """Return model RDMs keyed by their plot labels."""
    model_rdms = {}
    add_model_rdm_stack(model_rdms, "fv", fv_rdms, include_mean_rdms)
    add_model_rdm_stack(model_rdms, "cv", cv_rdms, include_mean_rdms)
    if resid_rdms is not None:
        add_model_rdm_stack(model_rdms, "resid", resid_rdms, include_mean_rdms)
    return model_rdms


def get_metric_matched_human_matrix(
    dataset,
    metric: str = "cosine",
    target_labels: list[str] = rdm_labels,
) -> np.ndarray:
    """Regenerate a human RDM from the dataset and align it to target_labels."""
    if metric == "cosine":
        measurements = np.asarray(dataset.get_measurements(), dtype=float)
        row_norms = np.linalg.norm(measurements, axis=1, keepdims=True)
        if np.any(row_norms == 0):
            raise ValueError("Cannot compute cosine distance with zero-norm rows.")
        normalized = measurements / row_norms
        matrix = 1 - normalized @ normalized.T
        np.fill_diagonal(matrix, 0)

        labels = normalize_labels(dataset.obs_descriptors["patterns"])
        assert_square_rdm_labels(
            labels, target_labels=target_labels, name="Human dataset"
        )
        reorder_idx = [labels.index(label) for label in target_labels]
        return matrix[np.ix_(reorder_idx, reorder_idx)]

    rdm = calc_rdm(dataset, method=metric)
    return get_human_rdm_matrix(rdm, dataset=dataset, target_labels=target_labels)


def get_metric_matched_human_matrices(
    human_rdms: Box,
    human_datasets: Box,
    level: str = "patt",
    human_dissimilarity_metric: str = "cosine",
) -> dict[str, np.ndarray]:
    """Regenerate and align all human RDMs for a given level."""
    human_matrices = {}
    for human_name in human_rdms:
        dataset = human_datasets[human_name][level]
        human_matrices[human_name] = get_metric_matched_human_matrix(
            dataset,
            metric=human_dissimilarity_metric,
            target_labels=rdm_labels,
        )
    return human_matrices


def compare_human_to_cv_fv(
    human_rdms: Box,
    human_datasets: Box | None,
    fv_rdms: np.ndarray,
    cv_rdms: np.ndarray,
    resid_rdms: np.ndarray | None = None,
    level: str = "patt",
    method: str = "spearman",
    human_dissimilarity_metric: str = "correlation",
) -> pd.DataFrame:
    """Compare human RDMs with model RDMs."""
    if human_datasets is None:
        raise ValueError("human_datasets is required to regenerate human RDMs.")

    human_matrices = get_metric_matched_human_matrices(
        human_rdms=human_rdms,
        human_datasets=human_datasets,
        level=level,
        human_dissimilarity_metric=human_dissimilarity_metric,
    )
    model_rdms = get_cv_fv_model_rdms(
        fv_rdms=fv_rdms,
        cv_rdms=cv_rdms,
        resid_rdms=resid_rdms,
    )

    rows = {}
    for human_name, human_matrix in human_matrices.items():
        rows[human_name] = {
            model_name: rdm_correlation(human_matrix, model_matrix, method=method)
            for model_name, model_matrix in model_rdms.items()
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for a flat p-value array."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def permutation_test_human_cv_fv(
    human_rdms: Box,
    human_datasets: Box,
    fv_rdms: np.ndarray,
    cv_rdms: np.ndarray,
    resid_rdms: np.ndarray | None = None,
    level: str = "patt",
    human_dissimilarity_metric: str = "correlation",
    method: str = "spearman",
    n_perm: int = 10000,
    seed: int = 0,
    exact_permutations: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run condition-label permutation tests for human-vs-model RSA."""
    rng = np.random.default_rng(seed)
    human_matrices = get_metric_matched_human_matrices(
        human_rdms=human_rdms,
        human_datasets=human_datasets,
        level=level,
        human_dissimilarity_metric=human_dissimilarity_metric,
    )
    model_rdms = get_cv_fv_model_rdms(
        fv_rdms=fv_rdms,
        cv_rdms=cv_rdms,
        resid_rdms=resid_rdms,
    )

    n_conditions = len(rdm_labels)
    rows = []
    comparisons = [
        (human_name, model_name, human_matrix, model_matrix)
        for human_name, human_matrix in human_matrices.items()
        for model_name, model_matrix in model_rdms.items()
    ]
    comparison_iter = tqdm(
        comparisons,
        desc="Human x model permutation tests",
        disable=not show_progress,
    )

    for human_name, model_name, human_matrix, model_matrix in comparison_iter:
        if show_progress:
            comparison_iter.set_postfix(human=human_name, model=model_name)
        observed = rdm_correlation(human_matrix, model_matrix, method=method)
        null = []
        if exact_permutations:
            permutations = itertools.permutations(range(n_conditions))
            total_perms = math.factorial(n_conditions)
        else:
            permutations = (rng.permutation(n_conditions) for _ in range(n_perm))
            total_perms = n_perm

        permutations_iter = tqdm(
            permutations,
            total=total_perms,
            desc=f"{human_name} x {model_name}",
            leave=False,
            disable=not show_progress,
        )
        for perm in permutations_iter:
            perm = np.asarray(perm)
            perm_model = model_matrix[np.ix_(perm, perm)]
            null.append(rdm_correlation(human_matrix, perm_model, method=method))

        null = np.asarray(null)
        n_perm_actual = len(null)
        p_greater = (np.sum(null >= observed) + 1) / (n_perm_actual + 1)
        p_less = (np.sum(null <= observed) + 1) / (n_perm_actual + 1)
        p_two_sided = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm_actual + 1)
        rows.append(
            {
                "human_rdm": human_name,
                "model_rdm": model_name,
                "rsa": observed,
                "p_greater": p_greater,
                "p_less": p_less,
                "p_two_sided": p_two_sided,
                "null_mean": float(np.mean(null)),
                "null_sd": float(np.std(null, ddof=1)),
                "n_perm": n_perm_actual,
                "exact_permutations": exact_permutations,
                "human_metric": human_dissimilarity_metric,
                "rsa_method": method,
            }
        )

    results = pd.DataFrame(rows)
    results["q_two_sided_fdr"] = bh_fdr(results["p_two_sided"].to_numpy())
    results["q_greater_fdr"] = bh_fdr(results["p_greater"].to_numpy())
    return results.sort_values("rsa", ascending=False).reset_index(drop=True)


def plot_human_cv_fv_comparison(
    comparison: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig_width = max(8, min(18, comparison.shape[1] * 0.32))
    fig_height = max(3, comparison.shape[0] * 0.75)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    vmax = np.nanmax(np.abs(comparison.values))
    im = ax.imshow(
        comparison.values,
        cmap=RSA_CMAP,
        norm=rsa_color_norm(vmin=-vmax, vmax=vmax),
        interpolation="none",
    )

    ax.set_xticks(np.arange(comparison.shape[1]), comparison.columns,)
    ax.set_yticks(np.arange(comparison.shape[0]), comparison.index)
    x_label_size = 8 if comparison.shape[1] <= 40 else 5
    ax.tick_params(axis="x", length=0, labelsize=x_label_size)
    ax.tick_params(axis="y", length=0)

    column_groups = [model_rdm_group(col) for col in comparison.columns]
    for boundary in [
        i
        for i in range(1, comparison.shape[1])
        if column_groups[i] != column_groups[i - 1]
    ]:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.2)

    for row in range(comparison.shape[0]):
        for col in range(comparison.shape[1]):
            value = comparison.iloc[row, col]
            text_color = "white" if abs(value) > 0.6 * vmax else "0.25"
            if comparison.shape[1] <= 40:
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSA")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def model_rdm_group(label: str) -> str:
    """Return the plot group for a model or human RDM label."""
    label = str(label)
    if label in {"frp", "erp", "rest"}:
        return "human"
    if label.startswith("fv_") or label.startswith("fv_mean"):
        return "fv"
    if label.startswith("cv_") or label.startswith("cv_mean"):
        return "cv"
    if label.startswith("resid_") or label.startswith("resid_mean"):
        return "resid"
    return label


def component_index(label: str, prefix: str) -> int | None:
    """Return the 1-based component index from labels like fv_1 or cv_1."""
    label = str(label)
    label_prefix = f"{prefix}_"
    if not label.startswith(label_prefix):
        return None
    suffix = label.removeprefix(label_prefix)
    return int(suffix) if suffix.isdigit() else None


def cv_fv_component_pairs(comparison: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Return shared component indices with matching FV and CV columns."""
    fv_by_idx = {
        idx: col
        for col in comparison.columns
        if (idx := component_index(col, "fv")) is not None
    }
    cv_by_idx = {
        idx: col
        for col in comparison.columns
        if (idx := component_index(col, "cv")) is not None
    }
    return [
        (idx, fv_by_idx[idx], cv_by_idx[idx])
        for idx in sorted(set(fv_by_idx) & set(cv_by_idx))
    ]


def cv_fv_components_long(comparison: pd.DataFrame) -> pd.DataFrame:
    """Convert the top-k FV/CV RSA columns to long format."""
    rows = []
    for human_rdm in comparison.index:
        for component, fv_col, cv_col in cv_fv_component_pairs(comparison):
            rows.append(
                {
                    "human_rdm": human_rdm,
                    "component": component,
                    "model_family": "FV",
                    "model_rdm": fv_col,
                    "rsa": comparison.loc[human_rdm, fv_col],
                }
            )
            rows.append(
                {
                    "human_rdm": human_rdm,
                    "component": component,
                    "model_family": "CV",
                    "model_rdm": cv_col,
                    "rsa": comparison.loc[human_rdm, cv_col],
                }
            )
        if "fv_mean_top5" in comparison.columns:
            rows.append(
                {
                    "human_rdm": human_rdm,
                    "component": "mean",
                    "model_family": "FV",
                    "model_rdm": "fv_mean_top5",
                    "rsa": comparison.loc[human_rdm, "fv_mean_top5"],
                }
            )
        if "cv_mean_top5" in comparison.columns:
            rows.append(
                {
                    "human_rdm": human_rdm,
                    "component": "mean",
                    "model_family": "CV",
                    "model_rdm": "cv_mean_top5",
                    "rsa": comparison.loc[human_rdm, "cv_mean_top5"],
                }
            )
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(diff: np.ndarray) -> float:
    """Exact two-sided paired sign-flip test for a mean difference."""
    diff = np.asarray(diff, dtype=float)
    observed = abs(float(np.mean(diff)))
    null = []
    for signs in itertools.product([-1, 1], repeat=len(diff)):
        null.append(abs(float(np.mean(diff * np.asarray(signs)))))
    return (np.sum(np.asarray(null) >= observed) + 1) / (len(null) + 1)


def test_cv_vs_fv_difference(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare paired CV and FV RSA values across matched top-k components."""
    rows = []
    pairs = cv_fv_component_pairs(comparison)
    for human_rdm in comparison.index:
        fv_values = np.array(
            [comparison.loc[human_rdm, fv_col] for _, fv_col, _ in pairs],
            dtype=float,
        )
        cv_values = np.array(
            [comparison.loc[human_rdm, cv_col] for _, _, cv_col in pairs],
            dtype=float,
        )
        diff = cv_values - fv_values
        t_res = ttest_rel(cv_values, fv_values)
        mean_column_diff = np.nan
        if "cv_mean_top5" in comparison.columns and "fv_mean_top5" in comparison.columns:
            mean_column_diff = (
                comparison.loc[human_rdm, "cv_mean_top5"]
                - comparison.loc[human_rdm, "fv_mean_top5"]
            )
        rows.append(
            {
                "human_rdm": human_rdm,
                "n_pairs": len(pairs),
                "mean_cv": float(np.mean(cv_values)),
                "mean_fv": float(np.mean(fv_values)),
                "mean_cv_minus_fv": float(np.mean(diff)),
                "rdm_mean_cv_minus_fv": float(mean_column_diff),
                "median_cv_minus_fv": float(np.median(diff)),
                "paired_t": float(t_res.statistic),
                "paired_t_p_two_sided": float(t_res.pvalue),
                "sign_flip_p_two_sided": exact_sign_flip_pvalue(diff),
            }
        )

    results = pd.DataFrame(rows)
    results["paired_t_q_fdr"] = bh_fdr(results["paired_t_p_two_sided"].to_numpy())
    results["sign_flip_q_fdr"] = bh_fdr(results["sign_flip_p_two_sided"].to_numpy())
    return results


def plot_cv_fv_human_barplot(
    comparison: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired FV/CV RSA bars for each top-k component and human RDM."""
    pairs = cv_fv_component_pairs(comparison)
    include_mean = "fv_mean_top5" in comparison.columns and "cv_mean_top5" in comparison.columns
    n_rows_per_group = len(pairs) + int(include_mean)
    group_gap = 1.25
    bar_height = 0.33
    colors = {"FV": "#C44E52", "CV": "#4C72B0"}

    fig_height = max(5.5, len(comparison.index) * (n_rows_per_group + group_gap) * 0.42)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), constrained_layout=True)
    y_ticks = []
    y_labels = []
    group_midpoints = []

    for group_i, human_rdm in enumerate(comparison.index):
        group_start = group_i * (n_rows_per_group + group_gap)
        group_midpoints.append(group_start + (n_rows_per_group - 1) / 2)
        for component_i, (component, fv_col, cv_col) in enumerate(pairs):
            y = group_start + component_i
            y_ticks.append(y)
            y_labels.append(str(component))
            ax.barh(
                y - bar_height / 2,
                comparison.loc[human_rdm, fv_col],
                height=bar_height,
                color=colors["FV"],
                label="FV" if group_i == 0 and component_i == 0 else None,
            )
            ax.barh(
                y + bar_height / 2,
                comparison.loc[human_rdm, cv_col],
                height=bar_height,
                color=colors["CV"],
                label="CV" if group_i == 0 and component_i == 0 else None,
            )
        if include_mean:
            y = group_start + len(pairs)
            y_ticks.append(y)
            y_labels.append("mean")
            ax.barh(
                y - bar_height / 2,
                comparison.loc[human_rdm, "fv_mean_top5"],
                height=bar_height,
                color=colors["FV"],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.88,
            )
            ax.barh(
                y + bar_height / 2,
                comparison.loc[human_rdm, "cv_mean_top5"],
                height=bar_height,
                color=colors["CV"],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.88,
            )

    ax.axvline(0, color="0.2", linewidth=0.9)
    ax.set_yticks(y_ticks, y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("RSA with human RDM", labelpad=10)
    ax.set_ylabel("Component", labelpad=12)
    ax.legend(frameon=False, loc="lower right")

    x_min, x_max = ax.get_xlim()
    label_x = x_min + 0.02 * (x_max - x_min)
    for human_rdm, midpoint in zip(comparison.index, group_midpoints):
        ax.text(
            label_x,
            midpoint,
            str(human_rdm).upper(),
            ha="left",
            va="center",
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def plot_cv_fv_human_paired_dotplot(
    comparison: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paired CV/FV component RSA values plus the mean RDM per human RDM."""
    pairs = cv_fv_component_pairs(comparison)
    colors = {"FV": "#C44E52", "CV": "#4C72B0"}
    row_gap = 1.5

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    y_ticks = []
    y_labels = []

    for row_i, human_rdm in enumerate(comparison.index):
        y = row_i * row_gap
        y_ticks.append(y)
        y_labels.append(str(human_rdm).upper())

        for component, fv_col, cv_col in pairs:
            fv_value = comparison.loc[human_rdm, fv_col]
            cv_value = comparison.loc[human_rdm, cv_col]
            ax.plot(
                [fv_value, cv_value],
                [y, y],
                color="0.75",
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                fv_value,
                y,
                color=colors["FV"],
                s=34,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
                label="FV components" if row_i == 0 and component == 1 else None,
            )
            ax.scatter(
                cv_value,
                y,
                color=colors["CV"],
                s=34,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
                label="CV components" if row_i == 0 and component == 1 else None,
            )

        if "fv_mean_top5" in comparison.columns and "cv_mean_top5" in comparison.columns:
            fv_mean = comparison.loc[human_rdm, "fv_mean_top5"]
            cv_mean = comparison.loc[human_rdm, "cv_mean_top5"]
            ax.plot([fv_mean, cv_mean], [y, y], color="0.25", linewidth=1.8, zorder=3)
            ax.scatter(
                fv_mean,
                y,
                color=colors["FV"],
                marker="D",
                s=78,
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
                label="FV mean RDM" if row_i == 0 else None,
            )
            ax.scatter(
                cv_mean,
                y,
                color=colors["CV"],
                marker="D",
                s=78,
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
                label="CV mean RDM" if row_i == 0 else None,
            )

    ax.axvline(0, color="0.15", linewidth=0.9)
    ax.set_yticks(y_ticks, y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("RSA with human RDM")
    ax.set_ylabel("Human RDM")
    ax.legend(frameon=False, ncols=2, loc="lower right")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def plot_residual_stream_comparison(
    comparison: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot human RSA against each residual-stream RDM."""
    resid_cols = [
        col
        for col in comparison.columns
        if str(col).startswith("resid_") and str(col).split("_", 1)[1].isdigit()
    ]
    if not resid_cols:
        raise ValueError("No residual-stream columns found in comparison table.")

    resid_cols = sorted(resid_cols, key=lambda col: int(str(col).split("_", 1)[1]))
    x = np.array([int(str(col).split("_", 1)[1]) for col in resid_cols])

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for human_name in comparison.index:
        line = ax.plot(
            x,
            comparison.loc[human_name, resid_cols],
            marker="o",
            markersize=2.5,
            linewidth=1.6,
            label=str(human_name).upper(),
        )[0]
        if "resid_mean" in comparison.columns:
            ax.axhline(
                comparison.loc[human_name, "resid_mean"],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.1,
                alpha=0.55,
            )

    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.65)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(np.arange(x.min(), x.max() + 1, 5))
    ax.set_xlabel("Residual-stream RDM index")
    ax.set_ylabel("RSA with human RDM")
    ax.legend(frameon=False, ncols=len(comparison.index), loc="upper center")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def residual_stream_columns(comparison: pd.DataFrame) -> list[str]:
    """Return residual-stream columns sorted by residual index."""
    resid_cols = [
        col
        for col in comparison.columns
        if str(col).startswith("resid_") and str(col).split("_", 1)[1].isdigit()
    ]
    return sorted(resid_cols, key=lambda col: int(str(col).split("_", 1)[1]))


def residual_stream_trajectory(
    comparison: pd.DataFrame,
    smooth_window: int = 7,
    polyorder: int = 2,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract and smooth human-vs-residual RSA trajectories."""
    resid_cols = residual_stream_columns(comparison)
    if not resid_cols:
        raise ValueError("No residual-stream columns found in comparison table.")

    x = np.array([int(str(col).split("_", 1)[1]) for col in resid_cols])
    values = comparison.loc[:, resid_cols].to_numpy(dtype=float)
    if smooth_window > 1 and values.shape[1] >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        values = savgol_filter(values, smooth_window, polyorder, axis=1)

    trajectory = pd.DataFrame(values, index=comparison.index, columns=x)
    return x, trajectory


def segment_linear_sse(x: np.ndarray, y: np.ndarray, start: int, end: int) -> float:
    """Fit a multivariate line to y[:, start:end] and return summed SSE."""
    x_segment = x[start:end]
    design = np.column_stack([np.ones_like(x_segment), x_segment])
    coef, *_ = np.linalg.lstsq(design, y[:, start:end].T, rcond=None)
    fitted = design @ coef
    residual = y[:, start:end].T - fitted
    return float(np.sum(residual**2))


def choose_residual_stream_phases(
    comparison: pd.DataFrame,
    min_phase_len: int = 6,
    max_phases: int = 7,
    smooth_window: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment residual-stream trajectories into piecewise-linear phases."""
    x, trajectory = residual_stream_trajectory(
        comparison,
        smooth_window=smooth_window,
    )
    y = trajectory.to_numpy(dtype=float)
    n = len(x)
    max_phases = min(max_phases, n // min_phase_len)

    sse = np.full((n, n + 1), np.inf)
    for start in range(n):
        for end in range(start + min_phase_len, n + 1):
            sse[start, end] = segment_linear_sse(x, y, start, end)

    dp = np.full((max_phases + 1, n + 1), np.inf)
    prev = np.full((max_phases + 1, n + 1), -1, dtype=int)
    dp[1, min_phase_len:] = sse[0, min_phase_len:]
    for k in range(2, max_phases + 1):
        for end in range(k * min_phase_len, n + 1):
            candidates = range((k - 1) * min_phase_len, end - min_phase_len + 1)
            for split in candidates:
                cost = dp[k - 1, split] + sse[split, end]
                if cost < dp[k, end]:
                    dp[k, end] = cost
                    prev[k, end] = split

    n_observations = y.size
    rows = []
    for k in range(1, max_phases + 1):
        total_sse = dp[k, n]
        n_parameters = k * 2 * y.shape[0] + (k - 1)
        bic = n_observations * np.log(total_sse / n_observations) + (
            n_parameters * np.log(n_observations)
        )
        rows.append({"n_phases": k, "sse": total_sse, "bic": bic})
    model_selection = pd.DataFrame(rows)
    best_k = int(model_selection.loc[model_selection["bic"].idxmin(), "n_phases"])

    boundaries = [n]
    end = n
    for k in range(best_k, 1, -1):
        split = prev[k, end]
        boundaries.append(split)
        end = split
    boundaries.append(0)
    boundaries = sorted(boundaries)

    phase_rows = []
    for phase_i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        segment = trajectory.iloc[:, start:end]
        phase_rows.append(
            {
                "phase": phase_i,
                "start_resid": int(x[start]),
                "end_resid": int(x[end - 1]),
                "n_residuals": int(end - start),
                **{
                    f"{human_name}_mean": float(segment.loc[human_name].mean())
                    for human_name in trajectory.index
                },
                **{
                    f"{human_name}_slope": float(
                        np.polyfit(x[start:end], segment.loc[human_name], deg=1)[0]
                    )
                    for human_name in trajectory.index
                },
            }
        )

    phases = pd.DataFrame(phase_rows)
    return phases, model_selection


def plot_residual_stream_phases(
    comparison: pd.DataFrame,
    phases: pd.DataFrame,
    save_path: Path | None = None,
    smooth_window: int = 7,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot smoothed residual trajectories with automatically detected phases."""
    x, trajectory = residual_stream_trajectory(comparison, smooth_window=smooth_window)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    phase_colors = ["#f3f3f3", "#e8eef7"]
    for row in phases.itertuples(index=False):
        ax.axvspan(
            row.start_resid - 0.5,
            row.end_resid + 0.5,
            color=phase_colors[(row.phase - 1) % len(phase_colors)],
            zorder=0,
        )
        ax.text(
            (row.start_resid + row.end_resid) / 2,
            0.98,
            f"P{row.phase}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )

    for human_name in trajectory.index:
        ax.plot(
            x,
            trajectory.loc[human_name],
            marker="o",
            markersize=2.3,
            linewidth=1.8,
            label=str(human_name).upper(),
        )

    for boundary in phases["end_resid"].iloc[:-1]:
        ax.axvline(boundary + 0.5, color="0.25", linewidth=1.0, linestyle="--")

    ax.axhline(0, color="0.2", linewidth=0.8, alpha=0.65)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(np.arange(x.min(), x.max() + 1, 5))
    ax.set_xlabel("Residual-stream RDM index")
    ax.set_ylabel("Smoothed RSA with human RDM")
    ax.legend(frameon=False, ncols=len(trajectory.index), loc="upper center")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def compute_named_rdm_rsa(
    named_rdms: dict[str, np.ndarray],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute a full RSA matrix across named square RDMs."""
    labels = list(named_rdms.keys())
    rsa = np.eye(len(labels))
    for i, label_i in enumerate(labels):
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            value = rdm_correlation(
                named_rdms[label_i], named_rdms[label_j], method=method
            )
            rsa[i, j] = value
            rsa[j, i] = value
    return pd.DataFrame(rsa, index=labels, columns=labels)


def plot_full_human_cv_fv_rsa(
    rsa: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a square RSA heatmap including human and model RDMs."""
    n = rsa.shape[0]
    fig_size = max(8, min(18, n * 0.32))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(
        rsa.values,
        cmap=RSA_CMAP,
        norm=rsa_color_norm(vmin=-0.5, vmax=1.0),
        interpolation="none",
    )

    ax.set_xticks(np.arange(n), rsa.columns,)
    ax.set_yticks(np.arange(n), rsa.index)
    label_size = 8 if n <= 40 else 4
    ax.tick_params(axis="both", length=0, labelsize=label_size)

    for row in range(n):
        for col in range(n):
            value = rsa.iloc[row, col]
            text_color = "white" if value < -0.15 or value > 0.55 else "0.25"
            if n <= 40:
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    groups = []
    for label in rsa.index:
        groups.append(model_rdm_group(label))

    for boundary in [i for i in range(1, n) if groups[i] != groups[i - 1]]:
        ax.axhline(boundary - 0.5, color="black", linewidth=1.2)
        ax.axvline(boundary - 0.5, color="black", linewidth=1.2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSA")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def rsa_distance_matrix(rsa: pd.DataFrame) -> pd.DataFrame:
    """Convert an RSA correlation matrix to a bounded dissimilarity matrix."""
    distance_values = 1 - rsa.to_numpy(dtype=float, copy=True)
    distance_values = np.clip(distance_values, a_min=0, a_max=None)
    np.fill_diagonal(distance_values, 0)
    return pd.DataFrame(distance_values, index=rsa.index, columns=rsa.columns)


def classical_mds(distance: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Classical MDS from a square distance matrix."""
    d = distance.to_numpy(dtype=float)
    n = d.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(positive)
    return pd.DataFrame(
        coords,
        index=distance.index,
        columns=[f"MDS{i + 1}" for i in range(n_components)],
    )


def plot_rdm_mds(
    rsa: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot a 2D MDS embedding of RDMs based on RSA distance."""
    coords = classical_mds(rsa_distance_matrix(rsa), n_components=2)
    colors = {"human": "#222222", "fv": "#C44E52", "cv": "#4C72B0", "resid": "#55A868"}
    markers = {"human": "s", "fv": "o", "cv": "o", "resid": "."}

    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    for group in ["resid", "fv", "cv", "human"]:
        labels = [label for label in coords.index if model_rdm_group(label) == group]
        if not labels:
            continue
        ax.scatter(
            coords.loc[labels, "MDS1"],
            coords.loc[labels, "MDS2"],
            color=colors[group],
            marker=markers[group],
            s=26 if group == "resid" else 58,
            alpha=0.45 if group == "resid" else 0.9,
            label=group.upper(),
        )

    for label in coords.index:
        group = model_rdm_group(label)
        if group != "resid" or label == "resid_mean":
            ax.text(
                coords.loc[label, "MDS1"],
                coords.loc[label, "MDS2"],
                str(label),
                fontsize=8,
                ha="left",
                va="bottom",
            )

    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.legend(frameon=False)

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax, coords


def plot_rdm_clustering(
    rsa: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot hierarchical clustering of RDMs based on RSA distance."""
    distance = rsa_distance_matrix(rsa)
    condensed = squareform(distance.to_numpy(), checks=False)
    clusters = linkage(condensed, method="average")
    fig_height = max(6, min(22, len(rsa.index) * 0.18))
    fig, ax = plt.subplots(figsize=(8, fig_height), constrained_layout=True)
    dendrogram(
        clusters,
        labels=list(rsa.index),
        orientation="right",
        leaf_font_size=7 if len(rsa.index) <= 40 else 4,
        ax=ax,
    )
    ax.set_xlabel("1 - RSA")

    if save_path is not None:
        save_figure(fig, save_path)

    return fig, ax


def get_full_human_cv_fv_rsa(
    human_dissimilarity_metric: str = "correlation",
    method: str = "spearman",
    resid_rdms: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the full human/model RSA matrix."""
    human_matrices = get_metric_matched_human_matrices(
        human_rdms=rdms,
        human_datasets=datasets,
        level="patt",
        human_dissimilarity_metric=human_dissimilarity_metric,
    )
    model_matrices = get_cv_fv_model_rdms(
        fv_rdms=fv_rdms,
        cv_rdms=cv_rdms,
        resid_rdms=resid_rdms,
    )
    named_rdms = {**human_matrices, **model_matrices}
    return compute_named_rdm_rsa(named_rdms, method=method)


def run_human_cv_fv_analysis(
    human_dissimilarity_metric: str,
    output_prefix: str,
    resid_rdms: np.ndarray | None = None,
    n_perm: int = 10000,
    exact_permutations: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run observed and permutation RSA for one human RDM metric."""
    rsa, _ = write_human_cv_fv_plot_outputs(
        human_dissimilarity_metric=human_dissimilarity_metric,
        output_prefix=output_prefix,
        resid_rdms=resid_rdms,
    )

    perm = permutation_test_human_cv_fv(
        human_rdms=rdms,
        human_datasets=datasets,
        fv_rdms=fv_rdms,
        cv_rdms=cv_rdms,
        resid_rdms=resid_rdms,
        level="patt",
        human_dissimilarity_metric=human_dissimilarity_metric,
        method="spearman",
        n_perm=n_perm,
        seed=0,
        exact_permutations=exact_permutations,
        show_progress=show_progress,
    )
    perm.to_csv(
        ROOT / f".temp/{output_prefix}_permutation_results.csv",
        index=False,
    )

    q = perm.pivot(
        index="human_rdm",
        columns="model_rdm",
        values="q_two_sided_fdr",
    ).loc[rsa.index, rsa.columns]
    q.to_csv(ROOT / f".temp/{output_prefix}_permutation_q_fdr.csv")

    return rsa, perm, q


def write_human_cv_fv_plot_outputs(
    human_dissimilarity_metric: str,
    output_prefix: str,
    resid_rdms: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write observed human/model RSA tables and plots without permutation tests."""
    rsa = compare_human_to_cv_fv(
        human_rdms=rdms,
        human_datasets=datasets,
        fv_rdms=fv_rdms,
        cv_rdms=cv_rdms,
        resid_rdms=resid_rdms,
        level="patt",
        method="spearman",
        human_dissimilarity_metric=human_dissimilarity_metric,
    )
    rsa.to_csv(ROOT / f".temp/{output_prefix}_correlations.csv")

    plot_human_cv_fv_comparison(
        rsa,
        save_path=ROOT / f".temp/{output_prefix}_correlations.png",
    )
    cv_fv_long = cv_fv_components_long(rsa)
    cv_fv_long.to_csv(
        ROOT / f".temp/{output_prefix}_cv_vs_fv_long.csv",
        index=False,
    )
    cv_fv_stats = test_cv_vs_fv_difference(rsa)
    cv_fv_stats.to_csv(
        ROOT / f".temp/{output_prefix}_cv_vs_fv_tests.csv",
        index=False,
    )
    plot_cv_fv_human_barplot(
        rsa,
        save_path=ROOT / f".temp/{output_prefix}_cv_vs_fv_barplot.png",
    )
    plot_cv_fv_human_paired_dotplot(
        rsa,
        save_path=ROOT / f".temp/{output_prefix}_cv_vs_fv_paired_dotplot.png",
    )
    if resid_rdms is not None:
        plot_residual_stream_comparison(
            rsa,
            save_path=ROOT / f".temp/{output_prefix}_residual_stream.png",
        )
        phases, phase_model_selection = choose_residual_stream_phases(rsa)
        phases.to_csv(
            ROOT / f".temp/{output_prefix}_residual_stream_phases.csv",
            index=False,
        )
        phase_model_selection.to_csv(
            ROOT / f".temp/{output_prefix}_residual_stream_phase_model_selection.csv",
            index=False,
        )
        plot_residual_stream_phases(
            rsa,
            phases,
            save_path=ROOT / f".temp/{output_prefix}_residual_stream_phases.png",
        )

    full_rsa = get_full_human_cv_fv_rsa(
        human_dissimilarity_metric=human_dissimilarity_metric,
        method="spearman",
        resid_rdms=resid_rdms,
    )
    full_rsa.to_csv(ROOT / f".temp/{output_prefix}_full_rsa.csv")
    plot_full_human_cv_fv_rsa(
        full_rsa,
        save_path=ROOT / f".temp/{output_prefix}_full_rsa.png",
    )
    _, _, mds_coords = plot_rdm_mds(
        full_rsa,
        save_path=ROOT / f".temp/{output_prefix}_mds.png",
    )
    mds_coords.to_csv(ROOT / f".temp/{output_prefix}_mds_coordinates.csv")
    plot_rdm_clustering(
        full_rsa,
        save_path=ROOT / f".temp/{output_prefix}_clustering.png",
    )

    return rsa, full_rsa


# Primary analysis:
# - human EEG RDMs: correlation distance
# - CV/FV/residual-stream LLM RDMs: precomputed model RDMs
# - cross-domain RSA: Spearman correlation with label-permutation testing
human_cv_fv_only_rsa, human_cv_fv_only_full_rsa = write_human_cv_fv_plot_outputs(
    human_dissimilarity_metric="correlation",
    output_prefix="human_correlation_cv_fv_only_rdm",
    resid_rdms=None,
)
human_cv_fv_rsa, human_cv_fv_perm, human_cv_fv_q = run_human_cv_fv_analysis(
    human_dissimilarity_metric="correlation",
    output_prefix="human_correlation_cv_fv_resid_rdm",
    resid_rdms=resid_rdms_mean,
    n_perm=10000,
    exact_permutations=False,
    show_progress=True,
)

# Backward-compatible copies for downstream notebooks that read the old names.
human_cv_fv_rsa.to_csv(ROOT / ".temp/human_cv_fv_rdm_correlations.csv")
human_cv_fv_perm.to_csv(
    ROOT / ".temp/human_cv_fv_rdm_permutation_results.csv",
    index=False,
)
human_cv_fv_q.to_csv(ROOT / ".temp/human_cv_fv_rdm_permutation_q_fdr.csv")


# Optional sensitivity analysis: metric-matched cosine distance on the human side.
RUN_COSINE_SENSITIVITY = False
if RUN_COSINE_SENSITIVITY:
    (
        human_cosine_cv_fv_rsa,
        human_cosine_cv_fv_perm,
        human_cosine_cv_fv_q,
    ) = run_human_cv_fv_analysis(
        human_dissimilarity_metric="cosine",
        output_prefix="human_cosine_cv_fv_resid_rdm",
        resid_rdms=resid_rdms_mean,
        n_perm=10000,
        exact_permutations=False,
        show_progress=True,
    )

human_cv_fv_only_rsa
human_cv_fv_only_full_rsa
human_cv_fv_rsa
human_cv_fv_perm
human_cv_fv_q

from pathlib import Path
# Assuming read_file is already imported

fig_path = Path(
    "/Users/chris/Documents/PhD-Local/abstract_reasoning/.temp/human_correlation_cv_fv_resid_rdm_residual_stream-fig.pkl"
)

fig = read_file(fig_path)

ax = fig.get_axes()[0]

# Fixed the unpacking order here to (bottom, top)
ymin, ymax = ax.get_ylim()

# Draw both lines
ax.vlines([57, 62], ymin, ymax, ls="--", color="black", lw=0.6)

# Pick a height for the text (95% up the y-axis keeps it near the top)
text_y = ymin + (ymax - ymin) * 0.95

# Add the labels right next to the lines
# 'va' is vertical alignment, 'ha' is horizontal alignment
ax.text(
    57.5,
    text_y,
    "CV",
    color="black",
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="left",
)
ax.text(
    62.5,
    text_y,
    "FV",
    color="black",
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="left",
)

suffix = "".join(fig_path.suffixes)
new_fpath = fig_path.with_name(fig_path.name.replace(suffix, ".png"))
save_figure(fig, new_fpath)

from rsatoolbox.data.dataset import load_dataset
from rsatoolbox.rdm.rdms import load_rdm

rdm = load_rdm(
    "/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/Lab/analyzed/RSA-FRP-frontal/rdm-group_avg-pattern_lvl.hdf5"
)
np.save("mean_human_rdm.npy", rdm.get_matrices().squeeze())
rdm.pattern_descriptors["patterns"]
