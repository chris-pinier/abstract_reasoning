from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rsatoolbox.data import Dataset, load_dataset
from rsatoolbox.rdm import RDMs, compare
from tqdm.auto import tqdm

from ar_analysis.analysis_rsa import calc_rdm_clean
from ar_analysis.utils.analysis_utils import list_contents


EXPERIMENTS_DIR = Path("/Volumes/Realtek 1Tb/PhD Data/experiments")
CONDITIONS = ["frp", "frp_control", "response", "rest", "sequence_heatmap"]
RSA_LEVELS = {
    "trial_lvl": "item_id",
    "pattern_lvl": "pattern",
}
DISSIMILARITY_METRIC = "correlation"
SIMILARITY_METRIC = "corr"
SHOW_FIGS = False


def latest_experiment_dir(base_dir: Path = EXPERIMENTS_DIR) -> Path:
    run_dirs = sorted([path for path in base_dir.glob("*") if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No experiment folders found in {base_dir}")
    return run_dirs[-1]


def parse_export_stem(path: Path) -> tuple[str, str, str]:
    """Parse dataset-subj_02-trial_lvl-frp-frontal-no_baseline style names."""
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected level-aware export filename: {path.name}")
    return parts[1], parts[2], parts[3]


def load_datasets(
    folder: Path,
) -> dict[str, dict[str, dict[str, Dataset]]]:
    folder = latest_experiment_dir(folder)
    datasets: dict[str, dict[str, dict[str, Dataset]]] = {}
    for file in sorted(list_contents(folder, reg=r"dataset.+\.hdf5$")):
        subj_label, level, condition = parse_export_stem(file)
        datasets.setdefault(subj_label, {}).setdefault(level, {})[condition] = (
            load_dataset(str(file))
        )
    return datasets


def finite_observation_mask(dataset: Dataset) -> np.ndarray:
    measurements = np.asarray(dataset.get_measurements())
    return np.isfinite(measurements.reshape(measurements.shape[0], -1)).all(axis=1)


def subset_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    obs_descriptors = {
        key: np.asarray(value)[indices]
        for key, value in dataset.obs_descriptors.items()
    }
    return Dataset(
        measurements=dataset.get_measurements()[indices],
        descriptors=dataset.descriptors,
        obs_descriptors=obs_descriptors,
        channel_descriptors=dataset.channel_descriptors,
    )


def descriptor_values(dataset: Dataset, descriptor: str) -> np.ndarray:
    if descriptor not in dataset.obs_descriptors:
        raise KeyError(f"`{descriptor}` missing from dataset obs_descriptors.")
    values = np.asarray(dataset.obs_descriptors[descriptor])
    if len(np.unique(values)) != len(values):
        raise ValueError(f"`{descriptor}` contains duplicate values.")
    return values


def descriptor_universe(datasets_by_subject: dict[str, Dataset], descriptor: str) -> list:
    values = set()
    for dataset in datasets_by_subject.values():
        values.update(descriptor_values(dataset, descriptor).tolist())
    return sorted(values)


def expanded_rdm_matrix(
    dataset: Dataset,
    descriptor: str,
    universe: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one subject RDM and expand it to the descriptor universe with NaNs."""
    values = descriptor_values(dataset, descriptor)
    finite_mask = finite_observation_mask(dataset)
    finite_indices = np.flatnonzero(finite_mask)
    finite_values = values[finite_indices]

    expanded = np.full((len(universe), len(universe)), np.nan, dtype=float)
    valid_universe_mask = np.isin(universe, finite_values)
    if finite_indices.size < 2:
        return expanded, valid_universe_mask

    finite_dataset = subset_dataset(dataset, finite_indices)
    rdm = calc_rdm_clean(finite_dataset, DISSIMILARITY_METRIC).get_matrices()[0]

    universe_lookup = {value: idx for idx, value in enumerate(universe)}
    target_indices = np.asarray([universe_lookup[value] for value in finite_values])
    expanded[np.ix_(target_indices, target_indices)] = rdm
    return expanded, valid_universe_mask


def rdm_matrix_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> tuple[float, int]:
    """Correlate overlapping finite upper-triangle RDM cells."""
    tri = np.triu_indices_from(matrix1, k=1)
    vec1 = matrix1[tri]
    vec2 = matrix2[tri]
    mask = np.isfinite(vec1) & np.isfinite(vec2)
    if mask.sum() < 2:
        return np.nan, int(mask.sum())
    return float(np.corrcoef(vec1[mask], vec2[mask])[0, 1]), int(mask.sum())


def expanded_subject_rdms(
    datasets_by_subject: dict[str, Dataset],
    descriptor: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list]:
    universe = descriptor_universe(datasets_by_subject, descriptor)
    rdms = {}
    valid_masks = {}
    for subj_label, dataset in datasets_by_subject.items():
        rdms[subj_label], valid_masks[subj_label] = expanded_rdm_matrix(
            dataset,
            descriptor,
            universe,
        )
    return rdms, valid_masks, universe


def average_rdm_matrix(
    rdms_by_subject: dict[str, np.ndarray],
    exclude: str | None = None,
) -> np.ndarray:
    labels = [label for label in sorted(rdms_by_subject) if label != exclude]
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack([rdms_by_subject[label] for label in labels]), axis=0)


def participant_similarity_matrix(
    rdms_by_subject: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = sorted(rdms_by_subject)
    comparison_labels = labels + ["average"]
    sim = pd.DataFrame(np.nan, index=comparison_labels, columns=comparison_labels)
    n_cells = pd.DataFrame(0, index=comparison_labels, columns=comparison_labels)
    avg_matrix = average_rdm_matrix(rdms_by_subject)

    comparison_rdms = {**rdms_by_subject, "average": avg_matrix}
    for label1 in comparison_labels:
        for label2 in comparison_labels:
            if label1 == label2:
                sim.loc[label1, label2] = 1.0
                n_cells.loc[label1, label2] = int(
                    np.isfinite(comparison_rdms[label1][np.triu_indices_from(avg_matrix, k=1)]).sum()
                )
                continue
            value, count = rdm_matrix_similarity(
                comparison_rdms[label1],
                comparison_rdms[label2],
            )
            sim.loc[label1, label2] = value
            n_cells.loc[label1, label2] = count

    return sim, n_cells


def participant_to_average_summary(
    rdms_by_subject: dict[str, np.ndarray],
    valid_masks_by_subject: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    avg_all = average_rdm_matrix(rdms_by_subject)

    for label in sorted(rdms_by_subject):
        avg_loo = average_rdm_matrix(rdms_by_subject, exclude=label)
        sim_all, n_cells_all = rdm_matrix_similarity(rdms_by_subject[label], avg_all)
        sim_loo, n_cells_loo = rdm_matrix_similarity(rdms_by_subject[label], avg_loo)
        rows.append(
            {
                "subject": label,
                "n_valid_observations": int(valid_masks_by_subject[label].sum()),
                "similarity_to_average_all": sim_all,
                "n_cells_to_average_all": n_cells_all,
                "similarity_to_leave_one_out_average": sim_loo,
                "n_cells_to_leave_one_out_average": n_cells_loo,
            }
        )

    return pd.DataFrame(rows)


def plot_similarity_matrix(sim_matrix: pd.DataFrame, title: str):
    size = max(7, 0.45 * len(sim_matrix))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        sim_matrix,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        center=0,
        square=True,
        annot=len(sim_matrix) <= 16,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig, ax


def save_outputs(
    validation_dir: Path,
    condition: str,
    level: str,
    sim_matrix: pd.DataFrame,
    similarity_cell_counts: pd.DataFrame,
    avg_summary: pd.DataFrame,
    universe: list,
    fig,
) -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"participants_{level}_{condition}"
    sim_matrix.to_csv(validation_dir / f"{prefix}_similarity.csv")
    similarity_cell_counts.to_csv(validation_dir / f"{prefix}_similarity_cell_counts.csv")
    avg_summary.to_csv(validation_dir / f"{prefix}_to_average.csv", index=False)
    pd.Series(universe, name=RSA_LEVELS[level]).to_csv(
        validation_dir / f"{prefix}_observation_universe.csv",
        index=False,
    )
    fig.savefig(validation_dir / f"{prefix}_similarity.png", dpi=300)
    fig.savefig(validation_dir / f"{prefix}_similarity.pdf")


def datasets_for_condition_level(
    datasets_by_subject: dict[str, dict[str, dict[str, Dataset]]],
    condition: str,
    level: str,
) -> dict[str, Dataset]:
    selected = {}
    for subj_label, datasets_by_level in datasets_by_subject.items():
        if level in datasets_by_level and condition in datasets_by_level[level]:
            selected[subj_label] = datasets_by_level[level][condition]
    return selected


def main():
    run_dir = latest_experiment_dir(EXPERIMENTS_DIR)
    validation_dir = run_dir / "participant_similarity"
    print(f"Using experiment folder: {run_dir}")

    datasets_by_subject = load_datasets(EXPERIMENTS_DIR)
    summary_rows = []

    for level, descriptor in RSA_LEVELS.items():
        for condition in tqdm(CONDITIONS, desc=f"{level} conditions"):
            selected = datasets_for_condition_level(datasets_by_subject, condition, level)
            if len(selected) < 2:
                print(f"[{level}/{condition}] Skipping: need >=2 subjects, got {len(selected)}")
                continue

            try:
                rdms_by_subject, valid_masks_by_subject, universe = expanded_subject_rdms(
                    selected,
                    descriptor,
                )
            except (KeyError, ValueError) as exc:
                print(f"[{level}/{condition}] Skipping: {exc}")
                continue

            sim_matrix, similarity_cell_counts = participant_similarity_matrix(rdms_by_subject)
            avg_summary = participant_to_average_summary(
                rdms_by_subject,
                valid_masks_by_subject,
            )
            fig, _ = plot_similarity_matrix(
                sim_matrix,
                title=f"{level} {condition}: participant RSA similarity",
            )
            save_outputs(
                validation_dir,
                condition,
                level,
                sim_matrix,
                similarity_cell_counts,
                avg_summary,
                universe,
                fig,
            )

            summary_rows.append(
                {
                    "level": level,
                    "condition": condition,
                    "n_subjects": len(rdms_by_subject),
                    "n_observation_universe": len(universe),
                    "mean_valid_observations": avg_summary["n_valid_observations"].mean(),
                    "mean_leave_one_out_similarity": avg_summary[
                        "similarity_to_leave_one_out_average"
                    ].mean(),
                    "mean_cells_to_leave_one_out_average": avg_summary[
                        "n_cells_to_leave_one_out_average"
                    ].mean(),
                    "mean_similarity_to_average_all": avg_summary[
                        "similarity_to_average_all"
                    ].mean(),
                }
            )

    summary = pd.DataFrame(summary_rows)
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(validation_dir / "participant_similarity_summary.csv", index=False)
    print(summary)

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close("all")

    return summary


if __name__ == "__main__":
    summary = main()
