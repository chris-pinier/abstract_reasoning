from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final
import os
import re

import pandas as pd
import plotly.express as px
import plotly.io as pio

from utils.analysis_utils import read_file


@dataclass(frozen=True)
class AnalysisConfig:
    project_root: Path
    save_disk: Path
    data_dir: Path
    export_dir: Path
    subj_lvl_data_dir: Path
    fixation_plots_dir: Path
    patterns: tuple[str, ...]
    target_inds: tuple[int, ...]
    group_specs: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def target_labels(self) -> list[str]:
        return [str(i) for i in self.target_inds]


def build_default_config(script_path: Path) -> AnalysisConfig:
    script_dir = script_path.resolve().parent
    project_root = (
        script_dir if (script_dir / "experiment-ANNs").exists() else script_dir.parent
    )

    save_disk = Path("/Volumes/Realtek 1Tb")
    assert save_disk.exists(), "WARNING: SSD not connected"

    data_dir = save_disk / "PhD Data/experiment1/data/"
    export_dir = save_disk / "PhD Data/experiment1-analysis/"

    return AnalysisConfig(
        project_root=project_root,
        save_disk=save_disk,
        data_dir=data_dir,
        export_dir=export_dir,
        subj_lvl_data_dir=export_dir / "Lab/analyzed/subj_lvl",
        fixation_plots_dir=project_root / "analysis/fixation_plots",
        patterns=(
            "AAABAAAB",
            "ABABCDCD",
            "ABBAABBA",
            "ABBACDDC",
            "ABBCABBC",
            "ABCAABCA",
            "ABCDDCBA",
            "ABCDEEDC",
        ),
        target_inds=tuple(range(8)),
        group_specs=(
            ("group", ()),
            ("participant", ("subj",)),
            ("session_by_participant", ("subj", "ses")),
            ("session_group", ("ses",)),
        ),
    )


def _configure_runtime_environment() -> None:
    # TEMP: locate ffmpeg for downstream workflows that depend on this script setup.
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]


def _as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def _slugify(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", text)).strip("_")


def _group_label(group_cols: tuple[str, ...], key: tuple) -> str:
    if not group_cols:
        return "group=all"
    return ", ".join(f"{col}={val}" for col, val in zip(group_cols, key))


def _group_slug(group_cols: tuple[str, ...], key: tuple) -> str:
    if not group_cols:
        return "group_all"
    parts = [f"{col}_{val}" for col, val in zip(group_cols, key)]
    return "__".join(_slugify(p) for p in parts)


def _parse_subject_session(fpath: Path) -> tuple[int, int]:
    subj_name, ses_name = fpath.parents[1].name, fpath.parents[0].name
    subj = int(subj_name.split("_")[1])
    ses = int(ses_name.split("_")[1])
    return subj, ses


def _load_behavior_patterns(
    subj: int,
    ses: int,
    data_dir: Path,
    cache: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    key = (subj, ses)
    if key in cache:
        return cache[key]

    sess_dir = data_dir / f"Lab/subj_{subj:02}/sess_{ses:02}"
    behav_df = pd.read_csv(next(sess_dir.glob("*-behav.csv")), index_col=0)
    behav_df = behav_df.reset_index(names="trial_N")[["trial_N", "pattern"]]
    cache[key] = behav_df
    return behav_df


def get_fixation_data_files(
    subj_lvl_data_dir: Path,
    data_dir: Path,
    file_types: tuple[str, ...] = ("gaze_target_fixation_sequence", "gaze_info"),
) -> dict[str, pd.DataFrame]:
    subj_lvl_data_files = list(subj_lvl_data_dir.rglob("*.pkl"))

    filtered_files = [
        f
        for f in subj_lvl_data_files
        if f.stem in file_types and not f.stem.startswith(".")
    ]
    files_by_type = {ftype: [] for ftype in file_types}
    for fpath in filtered_files:
        files_by_type[fpath.stem].append(fpath)

    behavior_cache: dict[tuple[int, int], pd.DataFrame] = {}
    outputs: dict[str, pd.DataFrame] = {}
    for ftype in file_types:
        entries: list[tuple[int, int, pd.DataFrame]] = []
        for fpath in files_by_type.get(ftype, []):
            subj, ses = _parse_subject_session(fpath)
            df = read_file(fpath).copy()
            df["subj"] = subj
            df["ses"] = ses
            behav_df = _load_behavior_patterns(subj, ses, data_dir, behavior_cache)
            merged_df = df.merge(behav_df, on="trial_N", how="left")
            entries.append((subj, ses, merged_df))

        if not entries:
            outputs[ftype] = pd.DataFrame()
            continue

        entries.sort(key=lambda x: (x[0], x[1]))
        outputs[ftype] = pd.concat([entry[2] for entry in entries], ignore_index=True)

    # TODO: URGENT FIX - currently the "first_fix_order" is 1-indexed; need to convert to 0-indexed for consistency
    if "gaze_info" in outputs and not outputs["gaze_info"].empty:
        outputs["gaze_info"] = outputs["gaze_info"].copy()
        outputs["gaze_info"]["first_fix_order"] -= 1

    return outputs


def build_matrices(
    df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...],
    patterns: tuple[str, ...],
    target_inds: tuple[int, ...],
    rank_within_pattern: bool = False,
) -> dict[tuple, pd.DataFrame]:
    by_cols = (
        [*group_cols, "pattern", "target_ind"]
        if group_cols
        else ["pattern", "target_ind"]
    )
    grouped = df.groupby(by_cols, as_index=False)[value_col].mean()

    if rank_within_pattern:
        rank_group_cols = [*group_cols, "pattern"] if group_cols else ["pattern"]
        grouped["plot_value"] = (
            grouped.groupby(rank_group_cols)[value_col]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
    else:
        grouped["plot_value"] = grouped[value_col]

    matrices: dict[tuple, pd.DataFrame] = {}
    iterator = (
        grouped.groupby(list(group_cols), sort=True)
        if group_cols
        else [(("all",), grouped)]
    )
    for key, group_df in iterator:
        key = _as_tuple(key)
        matrix = group_df.pivot(
            index="pattern", columns="target_ind", values="plot_value"
        ).reindex(index=patterns, columns=target_inds)
        matrices[key] = (
            matrix.astype("Int64") if rank_within_pattern else matrix.astype(float)
        )
    return matrices


def _write_matrix_table(df: pd.DataFrame, path_stem: Path) -> None:
    parquet_path = path_stem.with_suffix(".parquet")
    csv_path = path_stem.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path)
    except Exception as exc:
        print(f"WARNING: Failed to write {parquet_path.name} ({exc}); writing CSV.")
        df.to_csv(csv_path)


def export_plot_json_selection_to_images(
    json_paths: list[Path | str],
    output_format: str = "png",
    output_dir: Path | str | None = None,
) -> list[Path]:
    if output_format not in {"png", "pdf"}:
        raise ValueError("output_format must be one of: 'png', 'pdf'")

    exported_paths: list[Path] = []
    for json_path in json_paths:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"WARNING: JSON plot not found, skipping: {json_path}")
            continue

        target_dir = Path(output_dir) if output_dir is not None else json_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{json_path.stem}.{output_format}"

        try:
            fig = pio.from_json(json_path.read_text(encoding="utf-8"))
            fig.write_image(target_path, format=output_format)
            exported_paths.append(target_path)
        except Exception as exc:
            print(
                f"WARNING: Failed converting {json_path.name} -> {output_format}: {exc}"
            )

    return exported_paths


def _flatten_matrix(matrix: pd.DataFrame) -> pd.Series:
    return matrix.astype(float).stack(dropna=False)


def _build_named_matrices(
    matrices: dict[tuple, pd.DataFrame],
    group_cols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    named_matrices: dict[str, pd.DataFrame] = {}
    for key, matrix in matrices.items():
        key_tuple = _as_tuple(key)
        name = _group_slug(group_cols, key_tuple)
        named_matrices[name] = matrix
    return named_matrices


def _compute_pairwise_correlations(
    named_matrices: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(named_matrices) < 2:
        return pd.DataFrame(), pd.DataFrame()

    flat_df = pd.DataFrame(
        {name: _flatten_matrix(matrix) for name, matrix in named_matrices.items()}
    )
    corr_matrix = flat_df.corr(method="pearson")

    pairwise_rows: list[dict[str, float | int | str]] = []
    for matrix_a, matrix_b in combinations(flat_df.columns, 2):
        valid = flat_df[matrix_a].notna() & flat_df[matrix_b].notna()
        n_overlap = int(valid.sum())
        corr_value = flat_df.loc[valid, matrix_a].corr(flat_df.loc[valid, matrix_b])
        pairwise_rows.append(
            {
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "pearson_r": corr_value,
                "n_overlap": n_overlap,
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    return corr_matrix, pairwise_df


def _save_correlation_outputs(
    named_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    title: str,
) -> None:
    if len(named_matrices) < 2:
        print(f"[correlations] skipping '{title}' (need at least 2 matrices).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    corr_matrix, pairwise_df = _compute_pairwise_correlations(named_matrices)
    if corr_matrix.empty:
        print(f"[correlations] skipping '{title}' (correlation matrix is empty).")
        return

    _write_matrix_table(corr_matrix, output_dir / "correlation_matrix")
    _write_matrix_table(pairwise_df, output_dir / "pairwise_correlations")

    fig = px.imshow(
        corr_matrix,
        labels={"x": "Matrix", "y": "Matrix", "color": "Pearson r"},
        x=corr_matrix.columns,
        y=corr_matrix.index,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(title=title, xaxis_tickangle=45)
    fig.write_json(output_dir / "correlation_heatmap.json")


def run_correlation_pipeline(
    analysis_matrices: dict[str, dict[str, dict[tuple, pd.DataFrame]]],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for analysis_name, grouped_matrices in analysis_matrices.items():
        analysis_dir = output_root / analysis_name
        analysis_dir.mkdir(parents=True, exist_ok=True)

        participant_named = _build_named_matrices(
            grouped_matrices.get("participant", {}),
            ("subj",),
        )
        _save_correlation_outputs(
            participant_named,
            analysis_dir / "participant_vs_participant",
            title=f"{analysis_name}: Participant Matrix Correlations",
        )

        sessions_within_participant = grouped_matrices.get("session_by_participant", {})
        by_subj: dict[int, dict[str, pd.DataFrame]] = {}
        for key, matrix in sessions_within_participant.items():
            subj, ses = _as_tuple(key)
            by_subj.setdefault(int(subj), {})[f"ses_{ses}"] = matrix

        sessions_dir = analysis_dir / "sessions_within_participant"
        for subj, named_matrices in sorted(by_subj.items()):
            _save_correlation_outputs(
                named_matrices,
                sessions_dir / f"subj_{subj}",
                title=f"{analysis_name}: Session Correlations Within subj={subj}",
            )

        session_group_named = _build_named_matrices(
            grouped_matrices.get("session_group", {}),
            ("ses",),
        )
        _save_correlation_outputs(
            session_group_named,
            analysis_dir / "session_group_vs_session_group",
            title=f"{analysis_name}: Group Session Matrix Correlations",
        )

    group_type_named: dict[str, pd.DataFrame] = {}
    for analysis_name, grouped_matrices in analysis_matrices.items():
        group_matrices = grouped_matrices.get("group", {})
        if not group_matrices:
            continue
        first_key = next(iter(group_matrices))
        group_type_named[analysis_name] = group_matrices[first_key]

    _save_correlation_outputs(
        group_type_named,
        output_root / "group_across_types",
        title=(
            "Group Matrix Correlations Across Types "
            "(fixation_order_rank, total_duration, mean_pupil_diam)"
        ),
    )


def save_analysis_outputs(
    df: pd.DataFrame,
    value_col: str,
    analysis_name: str,
    color_label: str,
    text_auto: str | bool,
    group_specs: tuple[tuple[str, tuple[str, ...]], ...],
    patterns: tuple[str, ...],
    target_inds: tuple[int, ...],
    target_labels: list[str],
    output_root: Path,
    rank_within_pattern: bool = False,
    zmin=None,
    zmax=None,
) -> dict[str, dict[tuple, pd.DataFrame]]:
    analysis_dir = output_root / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    grouped_outputs: dict[str, dict[tuple, pd.DataFrame]] = {}

    for group_name, group_cols in group_specs:
        grouping_dir = analysis_dir / group_name
        matrices_dir = grouping_dir / "matrices"
        plots_dir = grouping_dir / "plots"
        matrices_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        matrices = build_matrices(
            df=df,
            value_col=value_col,
            group_cols=group_cols,
            patterns=patterns,
            target_inds=target_inds,
            rank_within_pattern=rank_within_pattern,
        )
        grouped_outputs[group_name] = matrices

        if group_cols:
            combined = pd.concat(matrices, names=[*group_cols, "pattern"])
        else:
            combined = next(iter(matrices.values()))
            combined.index.name = "pattern"
        _write_matrix_table(combined, matrices_dir / "all_matrices")

        for key, matrix in matrices.items():
            label = _group_label(group_cols, key)
            slug = _group_slug(group_cols, key)
            print(f"[{analysis_name}/{group_name}] exporting {slug}")

            _write_matrix_table(matrix, matrices_dir / f"matrix_{slug}")

            fig = px.imshow(
                matrix,
                labels={"x": "Target Index", "y": "Pattern", "color": color_label},
                x=target_labels,
                y=matrix.index,
                text_auto=text_auto,
                aspect="auto",
                color_continuous_scale="Viridis",
                zmin=zmin,
                zmax=zmax,
            )
            fig.update_layout(
                title=f"{color_label} Heatmap (Targets 0-7) - {group_name} ({label})",
                xaxis_tickangle=0,
            )
            fig.write_json(plots_dir / f"heatmap_{slug}.json")

    return grouped_outputs


def run_fixation_export_pipeline(config: AnalysisConfig) -> None:
    fixation_files = get_fixation_data_files(config.subj_lvl_data_dir, config.data_dir)
    gaze_info = fixation_files.get("gaze_info", pd.DataFrame())
    if gaze_info.empty:
        raise ValueError("No gaze_info data found in subject-level fixation files.")

    gaze_info_first8 = gaze_info.loc[
        gaze_info["target_ind"].isin(config.target_inds)
    ].copy()
    config.fixation_plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_matrices: dict[str, dict[str, dict[tuple, pd.DataFrame]]] = {}

    analysis_matrices["fixation_order_rank"] = save_analysis_outputs(
        df=gaze_info_first8,
        value_col="first_fix_order",
        analysis_name="fixation_order_rank",
        color_label="Fixation Order Rank",
        text_auto=True,
        group_specs=config.group_specs,
        patterns=config.patterns,
        target_inds=config.target_inds,
        target_labels=config.target_labels,
        output_root=config.fixation_plots_dir,
        rank_within_pattern=True,
        zmin=1,
        zmax=8,
    )

    for metric_col in ("total_duration", "mean_pupil_diam"):
        analysis_matrices[metric_col] = save_analysis_outputs(
            df=gaze_info_first8,
            value_col=metric_col,
            analysis_name=metric_col,
            color_label=metric_col.replace("_", " ").title(),
            text_auto=".2f",
            group_specs=config.group_specs,
            patterns=config.patterns,
            target_inds=config.target_inds,
            target_labels=config.target_labels,
            output_root=config.fixation_plots_dir,
            rank_within_pattern=False,
        )

    run_correlation_pipeline(
        analysis_matrices=analysis_matrices,
        output_root=config.fixation_plots_dir / "correlations",
    )


def main() -> None:
    _configure_runtime_environment()
    config = build_default_config(Path(__file__))
    run_fixation_export_pipeline(config)
    print(f"Saved analysis outputs to: {config.fixation_plots_dir}")


if __name__ == "__main__":
    main()
