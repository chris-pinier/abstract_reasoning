from pathlib import Path
from typing import Literal
import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rsatoolbox.data import Dataset, load_dataset
from rsatoolbox.rdm import compare, load_rdm
from tqdm.auto import tqdm

from ar_analysis.analysis_rsa import calc_rdm_clean, match_datasets
from ar_analysis.utils.analysis_utils import list_contents


EXPERIMENTS_DIR = Path(
    # "/Users/chris/Documents/PhD-Local/abstract_reasoning/.temp/experiments/"
    "/Volumes/Realtek 1Tb/PhD Data/experiments"
)
CONDITIONS = ["frp", "frp_control", "response", "rest", "sequence_heatmap"]
RSA_CONDITIONS = CONDITIONS + ["random"]
RSA_LEVELS = {
    "trial_lvl": "item_id",
    "pattern_lvl": "pattern",
}
DISSIMILARITY_METRIC = "correlation"
SIMILARITY_METRIC = "corr"
RANDOM_SEED = 13
SHOW_FIGS = False
N_WORKERS = 1
PRINT_SUBJECT_RESULTS = False


def latest_experiment_dir(base_dir: Path = EXPERIMENTS_DIR) -> Path:
    run_dirs = sorted([path for path in base_dir.glob("*") if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No experiment folders found in {base_dir}")
    return run_dirs[-1]


def resolve_run_dir(experiments_dir: Path, run_dir: Path | None = None) -> Path:
    return Path(run_dir) if run_dir is not None else latest_experiment_dir(experiments_dir)


def safe_command_output(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            cmd,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def validation_config(args: argparse.Namespace, run_dir: Path) -> dict:
    return {
        "experiments_dir": str(args.experiments_dir),
        "run_dir": str(run_dir),
        "conditions": CONDITIONS,
        "rsa_conditions": RSA_CONDITIONS,
        "rsa_levels": RSA_LEVELS,
        "dissimilarity_metric": DISSIMILARITY_METRIC,
        "similarity_metric": SIMILARITY_METRIC,
        "random_seed": RANDOM_SEED,
        "n_workers": args.n_workers,
        "print_subject_results": args.print_subject_results,
    }


def write_validation_sidecar(
    validation_dir: Path,
    config: dict,
    argv: list[str] | None = None,
) -> None:
    sidecar = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script": str(Path(__file__).resolve()),
        "argv": sys.argv if argv is None else [str(Path(__file__).resolve()), *argv],
        "config": config,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "rsatoolbox": package_version("rsatoolbox"),
        },
        "git": {
            "commit": safe_command_output(["git", "rev-parse", "HEAD"]),
            "branch": safe_command_output(["git", "branch", "--show-current"]),
            "status_short": safe_command_output(["git", "status", "--short"]),
        },
    }
    validation_dir.mkdir(parents=True, exist_ok=True)
    with open(validation_dir / "validation_parameters_sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)


def parse_export_stem(path: Path) -> tuple[str, str, str]:
    """Parse dataset-subj_02-trial_lvl-frp-frontal-no_baseline style names."""
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected level-aware export filename: {path.name}")
    return parts[1], parts[2], parts[3]


def load_data(
    folder: Path,
    dtype: Literal["rdm", "ds"] = "ds",
) -> dict[str, dict[str, dict[str, Dataset]]]:
    folder = Path(folder)
    reg = r"rdm.+\.hdf5$" if dtype == "rdm" else r"dataset.+\.hdf5$"
    loader = load_rdm if dtype == "rdm" else load_dataset

    data: dict[str, dict[str, dict[str, Dataset]]] = {}
    for file in sorted(list_contents(folder, reg=reg)):
        subj_label, level, condition = parse_export_stem(file)
        data.setdefault(subj_label, {}).setdefault(level, {})[condition] = loader(str(file))

    return data


def create_random_dataset_like(
    template: Dataset,
    random_state: int | None = None,
    condition: str = "random",
) -> Dataset:
    """Create one reproducible white-noise sanity-check dataset like template."""
    rng = np.random.default_rng(random_state)
    measurements = rng.standard_normal(template.get_measurements().shape)
    obs_descriptors = {
        key: np.asarray(value).copy()
        for key, value in template.obs_descriptors.items()
    }
    descriptors = {
        **template.descriptors,
        "condition": condition,
        "random_seed": random_state,
        "random_control": "white_noise",
    }
    return Dataset(
        measurements=measurements,
        descriptors=descriptors,
        obs_descriptors=obs_descriptors,
        channel_descriptors=template.channel_descriptors,
    )


def add_random_control(
    datasets: dict[str, Dataset],
    random_state: int | None = None,
    template_condition: str = "rest",
) -> dict[str, Dataset]:
    datasets = dict(datasets)
    datasets["random"] = create_random_dataset_like(
        datasets[template_condition],
        random_state=random_state,
    )
    return datasets


def summarize_datasets(
    datasets_by_subject: dict[str, dict[str, dict[str, Dataset]]],
) -> pd.DataFrame:
    rows = []
    for subj_label, datasets_by_level in datasets_by_subject.items():
        for level, datasets in datasets_by_level.items():
            for condition, dataset in datasets.items():
                measurements = dataset.get_measurements()
                finite_rows = np.isfinite(
                    measurements.reshape(measurements.shape[0], -1)
                ).all(axis=1)
                match_descriptor = RSA_LEVELS.get(level)
                descriptor_unique = (
                    len(np.unique(dataset.obs_descriptors[match_descriptor]))
                    == measurements.shape[0]
                    if match_descriptor in dataset.obs_descriptors
                    else False
                )
                rows.append(
                    {
                        "subject": subj_label,
                        "level": level,
                        "condition": condition,
                        "n_observations": measurements.shape[0],
                        "n_features": measurements.shape[1],
                        "n_valid_observations": int(finite_rows.sum()),
                        "n_nan_observations": int((~finite_rows).sum()),
                        "n_nan_values": int(np.isnan(measurements).sum()),
                        "match_descriptor": match_descriptor,
                        "match_descriptor_unique": descriptor_unique,
                    }
                )
    return pd.DataFrame(rows)


def subject_rsa(
    datasets: dict[str, Dataset],
    match_descriptor: str,
    conditions: list[str] = RSA_CONDITIONS,
    dissimilarity_metric: str = DISSIMILARITY_METRIC,
    similarity_metric: str = SIMILARITY_METRIC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [condition for condition in conditions if condition in datasets]
    sim_matrix = pd.DataFrame(np.nan, columns=labels, index=labels)
    n_matched = pd.DataFrame(0, columns=labels, index=labels)

    for label1 in labels:
        for label2 in labels:
            ds1, ds2 = match_datasets(
                datasets[label1],
                datasets[label2],
                descriptor=match_descriptor,
                drop_nan=True,
            )
            n_matched.loc[label1, label2] = ds1.get_measurements().shape[0]
            rdm1 = calc_rdm_clean(ds1, dissimilarity_metric)
            rdm2 = calc_rdm_clean(ds2, dissimilarity_metric)
            sim_matrix.loc[label1, label2] = compare(
                rdm1,
                rdm2,
                similarity_metric,
            ).flatten().item()

    return sim_matrix, n_matched


def analyze_subject(
    subj_i: int,
    subj_label: str,
    datasets_by_level: dict[str, dict[str, Dataset]],
) -> dict:
    subject_results = {}
    subject_counts = {}
    messages = []

    for level, match_descriptor in RSA_LEVELS.items():
        if level not in datasets_by_level:
            messages.append(f"[{subj_label}] Skipping missing level: {level}")
            continue

        datasets = datasets_by_level[level]
        missing = sorted(set(CONDITIONS) - set(datasets))
        if missing:
            messages.append(f"[{subj_label}/{level}] Skipping missing conditions: {missing}")
            continue

        datasets = add_random_control(
            datasets,
            random_state=RANDOM_SEED + subj_i,
        )
        subject_results[level], subject_counts[level] = subject_rsa(
            datasets,
            match_descriptor=match_descriptor,
        )

    return {
        "subj_i": subj_i,
        "subj_label": subj_label,
        "rsa_results": subject_results,
        "matched_counts": subject_counts,
        "messages": messages,
    }


def analyze_subjects_parallel(
    datasets_by_subject: dict[str, dict[str, dict[str, Dataset]]],
    n_workers: int,
) -> list[dict]:
    subject_items = list(enumerate(datasets_by_subject.items()))
    if n_workers <= 1:
        return [
            analyze_subject(subj_i, subj_label, datasets_by_level)
            for subj_i, (subj_label, datasets_by_level) in tqdm(
                subject_items,
                desc="Validation participants",
            )
        ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                analyze_subject,
                subj_i,
                subj_label,
                datasets_by_level,
            ): subj_label
            for subj_i, (subj_label, datasets_by_level) in subject_items
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validation participants",
        ):
            results.append(future.result())

    return sorted(results, key=lambda result: result["subj_i"])


def fisher_z_mean(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    """Average correlation matrices using Fisher-z transform."""
    clipped = [matrix.clip(-0.999999, 0.999999) for matrix in matrices]
    mean_z = sum(np.arctanh(matrix) for matrix in clipped) / len(clipped)
    return np.tanh(mean_z)


def plot_subject_rsa(sim_matrix: pd.DataFrame, title: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        sim_matrix,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def save_rsa_outputs(
    validation_dir: Path,
    label: str,
    level: str,
    sim_matrix: pd.DataFrame,
    matched_counts: pd.DataFrame,
    fig,
) -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{label}_{level}"
    sim_matrix.to_csv(validation_dir / f"{prefix}_rsa_similarity.csv")
    matched_counts.to_csv(validation_dir / f"{prefix}_matched_observation_counts.csv")
    fig.savefig(validation_dir / f"{prefix}_rsa_similarity.png", dpi=300)
    fig.savefig(validation_dir / f"{prefix}_rsa_similarity.pdf")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Experiment 1 RSA exports.")
    parser.add_argument("--experiments-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific experiment run folder. Defaults to latest folder in --experiments-dir.",
    )
    parser.add_argument("--n-workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--print-subject-results",
        action="store_true",
        default=PRINT_SUBJECT_RESULTS,
        help="Print each subject RSA matrix and matched-count matrix to the terminal.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    run_dir = resolve_run_dir(args.experiments_dir, args.run_dir)
    validation_dir = run_dir / "validation"
    print(f"Using experiment folder: {run_dir}")
    print(f"Workers: {args.n_workers}")

    config = validation_config(args, run_dir)
    write_validation_sidecar(validation_dir, config, argv)

    datasets_by_subject = load_data(run_dir, "ds")
    summary = summarize_datasets(datasets_by_subject)
    validation_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(validation_dir / "dataset_summary.csv", index=False)
    print(summary)

    rsa_results = {level: {} for level in RSA_LEVELS}
    matched_counts = {level: {} for level in RSA_LEVELS}
    subject_results = analyze_subjects_parallel(
        datasets_by_subject,
        n_workers=max(1, args.n_workers),
    )
    for result in subject_results:
        subj_label = result["subj_label"]
        for message in result["messages"]:
            print(message)

        for level in result["rsa_results"]:
            rsa_results[level][subj_label] = result["rsa_results"][level]
            matched_counts[level][subj_label] = result["matched_counts"][level]

            if args.print_subject_results:
                print(f"\n{subj_label} {level} RSA")
                print(rsa_results[level][subj_label])
                print(f"\n{subj_label} {level} matched observation counts")
                print(matched_counts[level][subj_label])

            fig, _ = plot_subject_rsa(
                rsa_results[level][subj_label],
                title=f"{subj_label} {level} RSA",
            )

            save_rsa_outputs(
                validation_dir,
                subj_label,
                level,
                rsa_results[level][subj_label],
                matched_counts[level][subj_label],
                fig,
            )

    for level in RSA_LEVELS:
        if not rsa_results[level]:
            continue
        mean_rsa = fisher_z_mean(list(rsa_results[level].values()))
        mean_counts = sum(matched_counts[level].values()) / len(matched_counts[level])
        print(f"\nMean {level} RSA")
        print(mean_rsa)
        fig, _ = plot_subject_rsa(mean_rsa, title=f"Mean {level} RSA")
        save_rsa_outputs(validation_dir, "mean", level, mean_rsa, mean_counts, fig)

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close("all")

    return summary, rsa_results, matched_counts


if __name__ == "__main__":
    summary, rsa_results, matched_counts = main()
