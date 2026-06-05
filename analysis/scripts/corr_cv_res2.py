import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

WD = Path(__file__).parents[1]
ROOT = WD.parent
sys.path.append(str(WD))
os.chdir(WD)
assert WD == Path.cwd()

from ar_analysis.utils.analysis_utils import (
    list_contents,
    read_file,
    save_figure as save_analysis_figure,
    xdir,
)
from scripts.analysis_rsa import get_ds_and_rdm


# * --------------------------------------------------------------------------
# * Configuration
# * --------------------------------------------------------------------------
DATA_ROOT_CANDIDATES = [
    Path(os.environ["ABSTRACT_REASONING_DATA_ROOT"])
    if "ABSTRACT_REASONING_DATA_ROOT" in os.environ
    else None,
    Path("/Volumes/SSD-512Go/PhD Data/experiment1/data"),
    Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data"),
]
DATA_ROOT_CANDIDATES = [path for path in DATA_ROOT_CANDIDATES if path is not None]
DATA_ROOT = next((path for path in DATA_ROOT_CANDIDATES if path.exists()), DATA_ROOT_CANDIDATES[-1])

ANN_DIR = Path(
    os.environ.get(
        "ABSTRACT_REASONING_ANN_DIR",
        str(DATA_ROOT / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts"),
    )
)
REALTEK_DATA_ROOT = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data")

HUMAN_DATASET_DIR_CANDIDATES = {
    "frp": [
        Path(os.environ["ABSTRACT_REASONING_HUMANS_DIR"])
        if "ABSTRACT_REASONING_HUMANS_DIR" in os.environ
        else None,
        DATA_ROOT / "Lab/analyzed/RSA-FRP-Frontal",
        DATA_ROOT / "Lab/analyzed/RSA-FRP-frontal",
        REALTEK_DATA_ROOT / "Lab/analyzed/RSA-FRP-Frontal",
        REALTEK_DATA_ROOT / "Lab/analyzed/RSA-FRP-frontal",
    ],
    "rest_erp": [
        Path(os.environ["ABSTRACT_REASONING_REST_ERP_DIR"])
        if "ABSTRACT_REASONING_REST_ERP_DIR" in os.environ
        else None,
        REALTEK_DATA_ROOT / "Lab/analyzed/RSA-Rest_ERP-frontal",
        DATA_ROOT / "Lab/analyzed/RSA-Rest_ERP-frontal",
    ],
    "response_erp": [
        Path(os.environ["ABSTRACT_REASONING_RESPONSE_ERP_DIR"])
        if "ABSTRACT_REASONING_RESPONSE_ERP_DIR" in os.environ
        else None,
        REALTEK_DATA_ROOT / "Lab/analyzed/RSA-Response_ERP-frontal",
        DATA_ROOT / "Lab/analyzed/RSA-Response_ERP-frontal",
    ],
}


def first_existing_path(candidates: list[Path | None]) -> Path:
    candidates = [path for path in candidates if path is not None]
    return next((path for path in candidates if path.exists()), candidates[-1])


HUMAN_DATASET_DIRS = {
    dataset: first_existing_path(candidates)
    for dataset, candidates in HUMAN_DATASET_DIR_CANDIDATES.items()
}
HUMANS_DIR = HUMAN_DATASET_DIRS["frp"]
SEQUENCES_FILE = ROOT / "config/sequences/sessions-1_to_5-masked_idx(7).csv"
EXPORT_DIR = ROOT / ".temp/corr_cv_res2"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

DISSIMILARITY_METRIC = "correlation"
SIMILARITY_METRIC = "corr"
FIGURE_DPI = 300


# * --------------------------------------------------------------------------
# * Small helpers
# * --------------------------------------------------------------------------
def save_figure(fig: plt.Figure, save_path: Path) -> None:
    """Save PNG/PDF plus a Matplotlib figure pickle sidecar."""
    save_analysis_figure(fig, save_path, dpi=FIGURE_DPI)


def rdm_matrix(rdm) -> np.ndarray:
    """Return a single square RDM matrix from an rsatoolbox RDM object."""
    matrices = np.asarray(rdm.get_matrices(), dtype=float)
    if matrices.shape[0] > 1:
        matrices = matrices.mean(axis=0, keepdims=True)
    return matrices[0]


def rdm_vector(rdm_or_matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangular off-diagonal RDM vector."""
    matrix = np.asarray(rdm_or_matrix, dtype=float)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri]


def rdm_similarity(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Pearson correlation between upper-triangular RDM entries."""
    a = rdm_vector(rdm_a)
    b = rdm_vector(rdm_b)
    return float(np.corrcoef(a, b)[0, 1])


def clean_human_label(path: Path) -> str:
    """Extract a readable human label from rdm-...-pattern_lvl.hdf5."""
    match = re.search(r"rdm-(.+?)-pattern", path.stem)
    label = match[1] if match else path.stem
    label = label.replace("human-", "")
    label = label.replace("group_avg", "average_human_file")
    return label


def layer_number(layer_file: Path) -> int:
    return int(re.search(r"layers\.(\d+)\.", str(layer_file))[1])


def output_prefix(dataset: str, stem: str) -> Path:
    return EXPORT_DIR / f"{dataset}_{stem}"


def safe_model_name(model: str) -> str:
    return re.sub(r"[^\w.-]+", "_", model)


# * --------------------------------------------------------------------------
# * Human-human similarity and noise ceiling
# * --------------------------------------------------------------------------
def load_human_pattern_rdms(humans_dir: Path = HUMANS_DIR) -> dict[str, np.ndarray]:
    """Load individual human pattern-level RDMs and append computed average_human."""
    from rsatoolbox.rdm.rdms import load_rdm

    supported_suffixes = {".hdf5", ".h5", ".pkl"}
    rdm_candidates = list_contents(humans_dir, reg=r"rdm-.+pattern.+")
    rdm_files = [
        path
        for path in rdm_candidates
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    skipped_files = sorted(set(rdm_candidates) - set(rdm_files))
    if skipped_files:
        skipped_names = ", ".join(path.name for path in skipped_files[:6])
        more = f", ... +{len(skipped_files) - 6} more" if len(skipped_files) > 6 else ""
        print(f"Skipping unsupported RDM sidecar files in {humans_dir.name}: {skipped_names}{more}")
    if not rdm_files:
        raise ValueError(
            f"No supported RDM files found in {humans_dir}. "
            f"Matched {len(rdm_candidates)} candidates, but none had one of: "
            f"{sorted(supported_suffixes)}"
        )

    loaded = {clean_human_label(f): rdm_matrix(load_rdm(str(f))) for f in rdm_files}

    subject_rdms = {
        label: matrix
        for label, matrix in loaded.items()
        if "avg" not in label.lower() and "average" not in label.lower()
    }
    if not subject_rdms:
        raise ValueError(f"No individual human RDMs found in {humans_dir}")

    average_human = np.mean(np.stack(list(subject_rdms.values())), axis=0)
    subject_rdms["average_human"] = average_human
    return subject_rdms


def human_similarity_matrix(human_rdms: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare every human RDM with every other human RDM."""
    labels = list(human_rdms)
    matrix = np.eye(len(labels))
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            matrix[i, j] = rdm_similarity(human_rdms[label_i], human_rdms[label_j])
    return pd.DataFrame(matrix, index=labels, columns=labels)


def estimate_noise_ceiling(human_rdms: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate RSA noise ceiling from human inter-RDM reliability.

    Lower ceiling: each subject RDM compared with the leave-one-subject-out average.
    Upper ceiling: each subject RDM compared with the all-subject average.
    """
    subject_labels = [label for label in human_rdms if label != "average_human"]
    subject_stack = np.stack([human_rdms[label] for label in subject_labels])
    all_subject_average = subject_stack.mean(axis=0)

    rows = []
    for i, label in enumerate(subject_labels):
        loo_average = np.delete(subject_stack, i, axis=0).mean(axis=0)
        rows.append(
            {
                "human": label,
                "lower_noise_ceiling": rdm_similarity(human_rdms[label], loo_average),
                "upper_noise_ceiling": rdm_similarity(human_rdms[label], all_subject_average),
            }
        )

    per_human = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "lower_noise_ceiling_mean": per_human["lower_noise_ceiling"].mean(),
                "lower_noise_ceiling_sd": per_human["lower_noise_ceiling"].std(ddof=1),
                "upper_noise_ceiling_mean": per_human["upper_noise_ceiling"].mean(),
                "upper_noise_ceiling_sd": per_human["upper_noise_ceiling"].std(ddof=1),
                "n_humans": len(subject_labels),
            }
        ]
    )
    return per_human, summary


def plot_human_similarity_matrix(similarity: pd.DataFrame, save_path: Path) -> None:
    fig_size = max(7, len(similarity) * 0.36)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(similarity.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(similarity)), similarity.columns, rotation=90)
    ax.set_yticks(np.arange(len(similarity)), similarity.index)
    ax.tick_params(axis="both", length=0, labelsize=8 if len(similarity) <= 30 else 5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Human-human RSA")
    save_figure(fig, save_path)
    plt.close(fig)


def plot_noise_ceiling(per_human: pd.DataFrame, summary: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    x = np.array([0, 1])
    means = [
        summary.loc[0, "lower_noise_ceiling_mean"],
        summary.loc[0, "upper_noise_ceiling_mean"],
    ]
    errs = [
        summary.loc[0, "lower_noise_ceiling_sd"],
        summary.loc[0, "upper_noise_ceiling_sd"],
    ]
    ax.bar(x, means, yerr=errs, color=["0.65", "0.35"], width=0.6)
    for _, row in per_human.iterrows():
        ax.plot(
            x,
            [row["lower_noise_ceiling"], row["upper_noise_ceiling"]],
            color="0.25",
            marker="o",
            alpha=0.45,
            linewidth=0.8,
        )
    ax.set_xticks(x, ["lower\nleave-one-out", "upper\nall-subject avg"])
    ax.set_ylabel("Human reliability RSA")
    ax.set_ylim(-0.05, 1.0)
    save_figure(fig, save_path)
    plt.close(fig)


# * --------------------------------------------------------------------------
# * LLM layer trajectories against each human RDM
# * --------------------------------------------------------------------------
def pattern_indices(sequences_file: Path = SEQUENCES_FILE) -> dict[str, np.ndarray]:
    input_ds = pd.read_csv(sequences_file)
    sorted_ds = input_ds.reset_index(drop=True).sort_values("pattern").reset_index(drop=False)
    return {
        pattern: group["index"].to_numpy()
        for pattern, group in sorted_ds.groupby("pattern", sort=True)
    }


def layer_rdm_from_activations(layer_file: Path, patt_inds: dict[str, np.ndarray]) -> np.ndarray:
    """Create an 8-pattern LLM RDM from one layer activation file.

    The activation array is expected to be (trials, 1, 8, hidden) or
    (trials, 8, hidden) after squeezing. We use the final sequence token,
    matching the previous exploratory version of this script.
    """
    act_arr = np.asarray(read_file(layer_file)).squeeze()
    if act_arr.ndim != 3:
        raise ValueError(f"Expected squeezed activation shape (trials, tokens, hidden); got {act_arr.shape}")

    mean_acts_per_patt = np.array(
        [act_arr[inds, -1].mean(axis=0) for inds in patt_inds.values()]
    )
    _, llm_rdm = get_ds_and_rdm(
        mean_acts_per_patt,
        DISSIMILARITY_METRIC,
        obs_descriptors={"patterns": list(patt_inds.keys())},
    )
    return rdm_matrix(llm_rdm)


def compare_llm_layers_to_humans(
    ann_dir: Path,
    human_rdms: dict[str, np.ndarray],
    patt_inds: dict[str, np.ndarray],
    upper_noise_ceiling: float,
    human_dataset: str,
) -> pd.DataFrame:
    """Compare every layer of one LLM with every human RDM."""
    ann_files = sorted(
        list_contents(ann_dir, reg=r"layers.+\.pkl"),
        key=layer_number,
    )
    rows = []
    for layer_file in tqdm(ann_files, desc=ann_dir.name, leave=False):
        layer_idx = layer_number(layer_file)
        llm_rdm = layer_rdm_from_activations(layer_file, patt_inds)
        for human_label, human_rdm in human_rdms.items():
            similarity = rdm_similarity(human_rdm, llm_rdm)
            rows.append(
                {
                    "model": ann_dir.name,
                    "layer": layer_idx,
                    "human_dataset": human_dataset,
                    "human": human_label,
                    "rsa": similarity,
                    "rsa_noise_ceiling_norm": similarity / upper_noise_ceiling,
                }
            )
    return pd.DataFrame(rows)


def compute_all_llm_layer_trajectories(
    ann_root: Path,
    human_rdms: dict[str, np.ndarray],
    noise_summary: pd.DataFrame,
    human_dataset: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    requested_output_file = output_prefix(human_dataset, "llm_layer_human_rsa.csv")
    output_file = requested_output_file
    legacy_frp_file = EXPORT_DIR / "llm_layer_human_rsa.csv"
    if human_dataset == "frp" and not output_file.exists() and legacy_frp_file.exists():
        output_file = legacy_frp_file

    if use_cache and output_file.exists():
        results = pd.read_csv(output_file)
        if "human_dataset" not in results.columns:
            results["human_dataset"] = human_dataset
        results = add_normalized_layer_depth(results)
        results.to_csv(output_file, index=False)
        if output_file != requested_output_file:
            results.to_csv(requested_output_file, index=False)
        return results

    patt_inds = pattern_indices()
    upper_noise_ceiling = noise_summary.loc[0, "upper_noise_ceiling_mean"]
    ann_dirs = list_contents(ann_root, incl="folder", recurs=False)
    all_results = []
    for ann_dir in tqdm(ann_dirs, desc="LLM models"):
        all_results.append(
            compare_llm_layers_to_humans(
                ann_dir=ann_dir,
                human_rdms=human_rdms,
                patt_inds=patt_inds,
                upper_noise_ceiling=upper_noise_ceiling,
                human_dataset=human_dataset,
            )
        )

    results = add_normalized_layer_depth(pd.concat(all_results, ignore_index=True))
    results.to_csv(requested_output_file, index=False)
    return results


def add_normalized_layer_depth(results: pd.DataFrame) -> pd.DataFrame:
    """Add a 0-1 layer-depth coordinate within each model."""
    results = results.copy()
    max_layer = results.groupby("model")["layer"].transform("max")
    min_layer = results.groupby("model")["layer"].transform("min")
    denom = (max_layer - min_layer).replace(0, np.nan)
    results["layer_depth_norm"] = ((results["layer"] - min_layer) / denom).fillna(0)
    return results


def plot_layer_trajectory_by_model(
    results: pd.DataFrame,
    human_dataset: str,
    normalized: bool = True,
) -> None:
    y_col = "rsa_noise_ceiling_norm" if normalized else "rsa"
    y_label = "RSA / upper noise ceiling" if normalized else "RSA with human RDM"
    suffix = "normalized" if normalized else "raw"

    for model, model_df in results.groupby("model"):
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for human, human_df in model_df.groupby("human"):
            alpha = 1.0 if human == "average_human" else 0.35
            linewidth = 2.2 if human == "average_human" else 0.9
            ax.plot(
                human_df["layer"],
                human_df[y_col],
                marker="o" if human == "average_human" else None,
                markersize=3,
                linewidth=linewidth,
                alpha=alpha,
                label=human,
            )
        if normalized:
            ax.axhline(1, color="0.15", linestyle="--", linewidth=1, label="upper ceiling")
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set_title(model)
        ax.set_xlabel("Layer")
        ax.set_ylabel(y_label)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
        save_name = re.sub(r"[^\w.-]+", "_", model)
        save_figure(fig, output_prefix(human_dataset, f"layer_trajectory_{suffix}_{save_name}.png"))
        plt.close(fig)


def plot_average_human_trajectories(
    results: pd.DataFrame,
    human_dataset: str,
    normalized: bool = True,
) -> None:
    results = add_normalized_layer_depth(results)
    y_col = "rsa_noise_ceiling_norm" if normalized else "rsa"
    y_label = "RSA / upper noise ceiling" if normalized else "RSA with average human RDM"
    suffix = "normalized" if normalized else "raw"
    avg_df = results.query("human == 'average_human'")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for model, model_df in avg_df.groupby("model"):
        model_df = model_df.sort_values("layer_depth_norm")
        ax.plot(
            model_df["layer_depth_norm"],
            model_df[y_col],
            linewidth=1.6,
            label=model,
        )
    if normalized:
        ax.axhline(1, color="0.15", linestyle="--", linewidth=1, label="upper ceiling")
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    save_figure(fig, output_prefix(human_dataset, f"layer_trajectory_average_human_{suffix}.png"))
    plt.close(fig)


def plot_average_human_trajectories_by_dataset(
    results: pd.DataFrame,
    normalized: bool = True,
) -> None:
    """Facet average-human layer trajectories by model, colored by human data type."""
    results = add_normalized_layer_depth(results)
    y_col = "rsa_noise_ceiling_norm" if normalized else "rsa"
    y_label = "RSA / data-type upper noise ceiling" if normalized else "RSA with average human RDM"
    suffix = "normalized" if normalized else "raw"
    avg_df = results.query("human == 'average_human'").copy()

    models = sorted(avg_df["model"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(models) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, max(4, n_rows * 3.1)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()
    colors = {
        "frp": "#4C72B0",
        "rest_erp": "#55A868",
        "response_erp": "#C44E52",
    }

    for ax, model in zip(axes, models):
        model_df = avg_df.query("model == @model")
        for human_dataset, dataset_df in model_df.groupby("human_dataset"):
            dataset_df = dataset_df.sort_values("layer_depth_norm")
            ax.plot(
                dataset_df["layer_depth_norm"],
                dataset_df[y_col],
                linewidth=1.8,
                color=colors.get(human_dataset, "0.25"),
                label=human_dataset,
            )
        if normalized:
            ax.axhline(1, color="0.15", linestyle="--", linewidth=0.9)
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set_title(model, fontsize=9)
        ax.set_xlim(0, 1)

    for ax in axes[len(models):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.supxlabel("Normalized layer depth")
    fig.supylabel(y_label)
    save_figure(fig, EXPORT_DIR / f"layer_trajectory_average_human_by_dataset_{suffix}.png")
    plt.close(fig)


# * --------------------------------------------------------------------------
# * Pattern-level embeddings and common-subspace alignment
# * --------------------------------------------------------------------------
def pattern_labels(sequences_file: Path = SEQUENCES_FILE) -> list[str]:
    return sorted(pd.read_csv(sequences_file)["pattern"].unique())


def classical_mds_from_rdm(rdm: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Embed pattern distances with classical MDS/PCoA."""
    distance = np.asarray(rdm, dtype=float)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order[:n_components]], a_min=0, a_max=None)
    eigvecs = eigvecs[:, order[:n_components]]
    return eigvecs * np.sqrt(eigvals)


def tsne_from_rdm(
    rdm: np.ndarray,
    n_components: int = 2,
    perplexity: float = 2.0,
    random_state: int = 7,
) -> np.ndarray:
    """Embed pattern distances with t-SNE using the RDM as precomputed distances."""
    from sklearn.manifold import TSNE

    distance = np.asarray(rdm, dtype=float)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    max_perplexity = max(1, (distance.shape[0] - 1) / 3)
    perplexity = min(perplexity, max_perplexity)
    return TSNE(
        n_components=n_components,
        metric="precomputed",
        perplexity=perplexity,
        init="random",
        random_state=random_state,
        learning_rate="auto",
    ).fit_transform(distance)


def center_scale(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    centered = coords - coords.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered)
    if norm == 0:
        return centered
    return centered / norm


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Rotate source coordinates into target coordinates and score alignment."""
    source_scaled = center_scale(source)
    target_scaled = center_scale(target)
    u, _, vt = np.linalg.svd(source_scaled.T @ target_scaled, full_matrices=False)
    rotation = u @ vt
    aligned = source_scaled @ rotation
    disparity = float(np.sum((aligned - target_scaled) ** 2))
    coord_corr = float(np.corrcoef(aligned.ravel(), target_scaled.ravel())[0, 1])
    return aligned, disparity, coord_corr


def layer_file_for_model_layer(ann_root: Path, model: str, layer: int) -> Path:
    model_dir = ann_root / model
    matches = sorted(list_contents(model_dir, reg=rf"layers\.{layer}\.pkl"))
    if not matches:
        raise FileNotFoundError(f"Could not find layer {layer} activation file in {model_dir}")
    return matches[0]


def select_best_average_human_layers(results: pd.DataFrame) -> pd.DataFrame:
    avg = results.query("human == 'average_human'").copy()
    idx = avg.groupby(["human_dataset", "model"])["rsa"].idxmax()
    return avg.loc[idx].sort_values(["human_dataset", "model"]).reset_index(drop=True)


def plot_pattern_embedding_grid(
    human_dataset: str,
    human_rdm: np.ndarray,
    selected_layers: pd.DataFrame,
    patt_inds: dict[str, np.ndarray],
    method: str = "mds",
) -> pd.DataFrame:
    """Plot human and best-layer LLM pattern embeddings in a shared aligned 2D space."""
    if method == "mds":
        embed = classical_mds_from_rdm
        method_label = "MDS"
    elif method == "tsne":
        embed = tsne_from_rdm
        method_label = "t-SNE"
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    labels = list(patt_inds.keys())
    colors = dict(zip(labels, plt.cm.tab10(np.linspace(0, 1, len(labels)))))
    target_coords = embed(human_rdm)
    target_scaled = center_scale(target_coords)

    n_panels = len(selected_layers) + 1
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, max(4, n_rows * 3.6)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()

    rows = []

    def draw_embedding(ax, coords, title):
        for label, xy in zip(labels, coords):
            ax.scatter(xy[0], xy[1], color=colors[label], s=54)
            ax.text(xy[0], xy[1], f" {label}", fontsize=7, va="center")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    draw_embedding(axes[0], target_scaled, f"{human_dataset}: average human")

    for ax, (_, row) in zip(axes[1:], selected_layers.iterrows()):
        layer_file = layer_file_for_model_layer(ANN_DIR, row.model, int(row.layer))
        llm_rdm = layer_rdm_from_activations(layer_file, patt_inds)
        source_coords = embed(llm_rdm)
        aligned_coords, disparity, coord_corr = procrustes_align(source_coords, target_coords)

        rows.append(
            {
                "human_dataset": human_dataset,
                "embedding_method": method,
                "model": row.model,
                "layer": int(row.layer),
                "layer_depth_norm": row.layer_depth_norm,
                "rdm_rsa": row.rsa,
                "rdm_rsa_noise_ceiling_norm": row.rsa_noise_ceiling_norm,
                "procrustes_disparity": disparity,
                "aligned_coordinate_corr": coord_corr,
            }
        )
        title = f"{row.model}\nL{int(row.layer)}, RSA={row.rsa:.2f}, align={coord_corr:.2f}"
        draw_embedding(ax, aligned_coords, title)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.supxlabel(f"{method_label}1, Procrustes-aligned")
    fig.supylabel(f"{method_label}2, Procrustes-aligned")
    save_figure(fig, output_prefix(human_dataset, f"pattern_embedding_{method}_best_llm_layers.png"))
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_human_pattern_embeddings(
    human_average_rdms: dict[str, np.ndarray],
    method: str = "mds",
) -> None:
    if method == "mds":
        embed = classical_mds_from_rdm
        method_label = "MDS"
    elif method == "tsne":
        embed = tsne_from_rdm
        method_label = "t-SNE"
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    labels = pattern_labels()
    colors = dict(zip(labels, plt.cm.tab10(np.linspace(0, 1, len(labels)))))
    n_cols = len(human_average_rdms)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(4.5 * n_cols, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (human_dataset, rdm) in zip(axes, human_average_rdms.items()):
        coords = center_scale(embed(rdm))
        for label, xy in zip(labels, coords):
            ax.scatter(xy[0], xy[1], color=colors[label], s=54)
            ax.text(xy[0], xy[1], f" {label}", fontsize=7, va="center")
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        ax.set_title(human_dataset)
    fig.supxlabel(f"{method_label}1")
    fig.supylabel(f"{method_label}2")
    save_figure(fig, EXPORT_DIR / f"human_average_pattern_embeddings_{method}.png")
    plt.close(fig)


def plot_common_subspace_alignment(alignment: pd.DataFrame, method: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x_labels = [safe_model_name(model).replace("--", "\n") for model in alignment["model"].unique()]
    x = np.arange(len(x_labels))
    width = 0.24
    datasets = list(alignment["human_dataset"].unique())
    colors = {"frp": "#4C72B0", "rest_erp": "#55A868", "response_erp": "#C44E52"}

    for offset, human_dataset in enumerate(datasets):
        values = (
            alignment.query("human_dataset == @human_dataset")
            .set_index("model")
            .loc[alignment["model"].unique(), "aligned_coordinate_corr"]
        )
        ax.bar(
            x + (offset - (len(datasets) - 1) / 2) * width,
            values,
            width=width,
            color=colors.get(human_dataset, "0.4"),
            label=human_dataset,
        )
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xticks(x, x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Aligned coordinate correlation")
    ax.legend(frameon=False)
    ax.set_title(f"{method.upper()} pattern-space alignment")
    save_figure(fig, EXPORT_DIR / f"common_subspace_alignment_{method}_best_layers.png")
    plt.close(fig)


def analyze_human_dataset(
    human_dataset: str,
    humans_dir: Path,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    print(f"\n[{human_dataset}] Loading human RDMs from: {humans_dir}")
    human_rdms = load_human_pattern_rdms(humans_dir)

    human_sim = human_similarity_matrix(human_rdms)
    human_sim.to_csv(output_prefix(human_dataset, "human_human_similarity_matrix.csv"))
    plot_human_similarity_matrix(
        human_sim,
        output_prefix(human_dataset, "human_human_similarity_matrix.png"),
    )

    noise_per_human, noise_summary = estimate_noise_ceiling(human_rdms)
    noise_per_human.insert(0, "human_dataset", human_dataset)
    noise_summary.insert(0, "human_dataset", human_dataset)
    noise_per_human.to_csv(
        output_prefix(human_dataset, "human_noise_ceiling_per_subject.csv"),
        index=False,
    )
    noise_summary.to_csv(
        output_prefix(human_dataset, "human_noise_ceiling_summary.csv"),
        index=False,
    )
    plot_noise_ceiling(
        noise_per_human,
        noise_summary,
        output_prefix(human_dataset, "human_noise_ceiling.png"),
    )

    llm_results = compute_all_llm_layer_trajectories(
        ann_root=ANN_DIR,
        human_rdms=human_rdms,
        noise_summary=noise_summary,
        human_dataset=human_dataset,
        use_cache=use_cache,
    )
    plot_layer_trajectory_by_model(llm_results, human_dataset=human_dataset, normalized=False)
    plot_layer_trajectory_by_model(llm_results, human_dataset=human_dataset, normalized=True)
    plot_average_human_trajectories(llm_results, human_dataset=human_dataset, normalized=False)
    plot_average_human_trajectories(llm_results, human_dataset=human_dataset, normalized=True)

    patt_inds = pattern_indices()
    selected_layers = select_best_average_human_layers(llm_results)
    alignments = []
    for method in ["mds", "tsne"]:
        alignment = plot_pattern_embedding_grid(
            human_dataset=human_dataset,
            human_rdm=human_rdms["average_human"],
            selected_layers=selected_layers,
            patt_inds=patt_inds,
            method=method,
        )
        alignment.to_csv(
            output_prefix(human_dataset, f"common_subspace_alignment_{method}_best_layers.csv"),
            index=False,
        )
        alignments.append(alignment)

    pd.concat(alignments, ignore_index=True).to_csv(
        output_prefix(human_dataset, "common_subspace_alignment_best_layers.csv"),
        index=False,
    )
    return llm_results, human_rdms["average_human"]


def main() -> None:
    all_results = []
    human_average_rdms = {}
    for human_dataset, humans_dir in HUMAN_DATASET_DIRS.items():
        if not humans_dir.exists():
            print(f"[{human_dataset}] Skipping missing directory: {humans_dir}")
            continue
        llm_results, average_human_rdm = analyze_human_dataset(human_dataset, humans_dir, use_cache=True)
        all_results.append(llm_results)
        human_average_rdms[human_dataset] = average_human_rdm

    if not all_results:
        raise FileNotFoundError(
            "None of the configured human RDM directories were found. "
            f"Configured paths: {HUMAN_DATASET_DIRS}"
        )

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = add_normalized_layer_depth(combined_results)
    combined_results.to_csv(EXPORT_DIR / "llm_layer_all_human_datatypes_rsa.csv", index=False)
    plot_average_human_trajectories_by_dataset(combined_results, normalized=False)
    plot_average_human_trajectories_by_dataset(combined_results, normalized=True)
    plot_human_pattern_embeddings(human_average_rdms, method="mds")
    plot_human_pattern_embeddings(human_average_rdms, method="tsne")

    alignment_files = sorted(EXPORT_DIR.glob("*_common_subspace_alignment_*_best_layers.csv"))
    if alignment_files:
        alignment = pd.concat([pd.read_csv(path) for path in alignment_files], ignore_index=True)
        alignment.to_csv(EXPORT_DIR / "common_subspace_alignment_best_layers.csv", index=False)
        for method, method_alignment in alignment.groupby("embedding_method"):
            method_alignment.to_csv(
                EXPORT_DIR / f"common_subspace_alignment_{method}_best_layers.csv",
                index=False,
            )
            plot_common_subspace_alignment(method_alignment, method=method)


if __name__ == "__main__":
    main()
