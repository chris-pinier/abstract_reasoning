import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from rsatoolbox.rdm import calc_rdm, compare as compare_rdms, calc_rdm_movie, RDMs
from rsatoolbox.data import Dataset, TemporalDataset
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd


def sanitize_descriptor_value(value):
    """Convert 0-d numpy descriptors loaded from HDF5 back to Python scalars."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def sanitize_descriptors(descriptors: Optional[Dict]) -> Optional[Dict]:
    if descriptors is None:
        return None
    return {key: sanitize_descriptor_value(value) for key, value in descriptors.items()}


def sanitize_dataset_descriptors(dataset: Dataset) -> Dataset:
    """Sanitize dataset-level descriptors in-place and return the dataset."""
    dataset.descriptors = sanitize_descriptors(dataset.descriptors)
    return dataset


def _as_array_obs_descriptors(obs_descriptors: Dict) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in obs_descriptors.items()}


def _subset_dataset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    obs_descriptors = {
        key: np.asarray(value)[indices]
        for key, value in dataset.obs_descriptors.items()
    }
    return Dataset(
        measurements=dataset.get_measurements()[indices],
        descriptors=sanitize_descriptors(dataset.descriptors),
        obs_descriptors=obs_descriptors,
        channel_descriptors=dataset.channel_descriptors,
    )


def _finite_observation_mask(dataset: Dataset) -> np.ndarray:
    measurements = np.asarray(dataset.get_measurements())
    return np.isfinite(measurements.reshape(measurements.shape[0], -1)).all(axis=1)


def _assert_aligned_descriptor(
    ds1: Dataset, ds2: Dataset, descriptor: Optional[str] = None
) -> None:
    if descriptor is None:
        descriptor = "item_id" if "item_id" in ds1.obs_descriptors else None
    if descriptor is None:
        return
    if descriptor not in ds1.obs_descriptors or descriptor not in ds2.obs_descriptors:
        raise KeyError(f"`{descriptor}` must be present in both datasets.")
    vals1 = np.asarray(ds1.obs_descriptors[descriptor])
    vals2 = np.asarray(ds2.obs_descriptors[descriptor])
    if vals1.shape[0] != vals2.shape[0] or not np.array_equal(vals1, vals2):
        raise ValueError(f"Datasets are not aligned on `{descriptor}`.")


def calc_rdm_clean(dataset: Dataset, method: str, **kwargs):
    """Calculate an RDM after descriptor sanitization and finite-data validation."""
    dataset = sanitize_dataset_descriptors(dataset)
    if not _finite_observation_mask(dataset).all():
        raise ValueError("Dataset contains NaN or infinite observations before calc_rdm.")
    return calc_rdm(dataset, method=method, **kwargs)


def get_reference_rdms(
    n_trials: int = 400,
    n_trials_per_patt_type: int = 50,
    n_patt_type: int = 8,
    fmt: str = "rsatoolbox",
    dissimilarity_metric: Optional[str] = None,
):
    allowed_fmts = ["numpy", "rsatoolbox"]
    assert fmt in allowed_fmts, f"fmt must be one of {allowed_fmts}"

    arr_item_lvl = np.ones((n_trials, n_trials))

    inds = np.arange(0, n_trials + 1, n_trials_per_patt_type)
    inds = list(zip(inds[:-1], inds[1:]))
    inds = [slice(*i) for i in inds]

    for ind_slice in inds:
        arr_item_lvl[ind_slice, ind_slice] = 0

    arr_pattern_lvl = np.ones((n_patt_type, n_patt_type))

    inds = np.arange(0, n_patt_type + 1, 1)
    inds = list(zip(inds[:-1], inds[1:]))
    inds = [slice(*i) for i in inds]

    for ind_slice in inds:
        arr_pattern_lvl[ind_slice, ind_slice] = 0

    if fmt == "rsatoolbox":
        arr_item_lvl = RDMs(
            dissimilarities=arr_item_lvl[None, :],
            dissimilarity_measure=dissimilarity_metric,
        )

        arr_pattern_lvl = RDMs(
            dissimilarities=arr_pattern_lvl[None, :],
            dissimilarity_measure=dissimilarity_metric,
        )

    return arr_item_lvl, arr_pattern_lvl


def create_and_save_ds(
    measurements: np.ndarray,
    fpath: Optional[Path] = None,
    ds_type: str = "regular",
    descriptors: Optional[Dict] = None,
    obs_descriptors: Optional[Dict] = None,
    channel_descriptors: Optional[Dict] = None,
    time_descriptors: Optional[Dict] = None,
):
    allowed_ds_types = ["regular", "temporal"]
    assert ds_type in allowed_ds_types, f"ds_type must be one of {allowed_ds_types}"

    if ds_type == "regular":
        dataset = Dataset(
            measurements=measurements,
            descriptors=sanitize_descriptors(descriptors),
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
        )
    else:
        dataset = TemporalDataset(
            measurements=measurements,
            descriptors=sanitize_descriptors(descriptors),
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
            time_descriptors=time_descriptors,
        )
    if fpath:
        dataset.save(fpath, file_type="hdf5", overwrite=True)

    return dataset


def create_and_save_rdm(
    dataset,
    dissimilarity_metric: str,
    fpath: Path | None = None,
    ds_type: str = "regular",
):
    allowed_ds_types = ["regular", "temporal"]
    assert ds_type in allowed_ds_types, f"ds_type must be one of {allowed_ds_types}"

    if ds_type == "regular":
        rdm = calc_rdm_clean(dataset, method=dissimilarity_metric)
    else:
        rdm = calc_rdm_movie(dataset, method=dissimilarity_metric)

    if fpath:
        rdm.save(fpath, file_type="hdf5", overwrite=True)

    return rdm


def get_ds_and_rdm(
    measurements: np.ndarray,
    dissimilarity_metric: str,
    ds_fpath: Optional[Path] = None,
    rdm_fpath: Optional[Path] = None,
    ds_type: str = "regular",
    descriptors: Optional[Dict] = None,
    obs_descriptors: Optional[Dict] = None,
    channel_descriptors: Optional[Dict] = None,
    time_descriptors: Optional[Dict] = None,
):
    dataset = create_and_save_ds(
        measurements=measurements,
        fpath=ds_fpath,
        ds_type=ds_type,
        descriptors=descriptors,
        obs_descriptors=obs_descriptors,
        channel_descriptors=channel_descriptors,
        time_descriptors=time_descriptors,
    )

    rdm = create_and_save_rdm(dataset, dissimilarity_metric, rdm_fpath, ds_type)

    return dataset, rdm


def match_datasets_on_descriptor(
    ds1: Dataset, ds2: Dataset, descriptor: Optional[str] = None
):
    if descriptor is None:
        return ds1, ds2

    ds1_obs_desc = _as_array_obs_descriptors(ds1.obs_descriptors)
    ds2_obs_desc = _as_array_obs_descriptors(ds2.obs_descriptors)

    if descriptor not in ds1_obs_desc or descriptor not in ds2_obs_desc:
        raise KeyError(f"`{descriptor}` must be present in both datasets.")

    vals1 = ds1_obs_desc[descriptor]
    vals2 = ds2_obs_desc[descriptor]

    if len(np.unique(vals1)) != len(vals1):
        raise ValueError(f"`{descriptor}` contains duplicates in ds1.")
    if len(np.unique(vals2)) != len(vals2):
        raise ValueError(f"`{descriptor}` contains duplicates in ds2.")

    common = np.intersect1d(vals1, vals2)
    if common.size == 0:
        raise ValueError(f"No shared observations found for `{descriptor}`.")

    ds1_lookup = {value: idx for idx, value in enumerate(vals1)}
    ds2_lookup = {value: idx for idx, value in enumerate(vals2)}
    ds1_inds = np.asarray([ds1_lookup[value] for value in common])
    ds2_inds = np.asarray([ds2_lookup[value] for value in common])

    ds1_new = _subset_dataset(ds1, ds1_inds)
    ds2_new = _subset_dataset(ds2, ds2_inds)
    _assert_aligned_descriptor(ds1_new, ds2_new, descriptor)
    return ds1_new, ds2_new


def match_datasets_on_nan(
    ds1: Dataset, ds2: Dataset, verify_descriptor: Optional[str] = None
):
    ds1_arr = ds1.get_measurements()
    ds2_arr = ds2.get_measurements()

    if ds1_arr.shape[0] != ds2_arr.shape[0]:
        raise ValueError("Datasets shapes don't match. Align them first.")

    _assert_aligned_descriptor(ds1, ds2, verify_descriptor)

    mask = _finite_observation_mask(ds1) & _finite_observation_mask(ds2)
    return _subset_dataset(ds1, mask), _subset_dataset(ds2, mask)


def match_datasets(
    ds1: Dataset,
    ds2: Dataset,
    descriptor: Optional[str] = "item_id",
    drop_nan: bool = True,
) -> tuple[Dataset, Dataset]:
    """Align two datasets by descriptor and optionally drop NaN observations."""
    ds1_new, ds2_new = match_datasets_on_descriptor(ds1, ds2, descriptor)
    if drop_nan:
        ds1_new, ds2_new = match_datasets_on_nan(
            ds1_new, ds2_new, verify_descriptor=descriptor
        )
    return ds1_new, ds2_new


def simple_rsa(
    datasets: Dict[str, Dataset],
    dissimilarity_metric: str,
    similarity_metric: str,
    show_fig: bool = True,
    return_fig: bool = True,
    order: Optional[List[str]] = None,
    title: Optional[str] = None,
    pbar1: bool = True,
    pbar2: bool = False,
    descriptor_match: Optional[str] = "item_id",
):
    # # ! TEMP
    # human_data_dir = (
    #     EXPORT_DIR / f"Lab/analyzed/RSA-FRP-frontal-metric_correlation"
    # )

    # ANN_data_dir = (
    #     EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
    # )
    # level = "pattern"
    # humans_datasets, ANNs_datasets = load_representation_datasets(
    #     human_data_dir, ANN_data_dir, level
    # )
    # ANNs_datasets = {clean_ann_id(k): v for k, v in ANNs_datasets.items()}
    # dissimilarity_metric = "correlation"
    # similarity_metric = "corr"
    # datasets = {**ANNs_datasets, **humans_datasets}
    # # ! TEMP

    sim_labels = {
        "cosine": "Cosine similarity",
        "corr": "Pearson Correlation",
        "tau-a": "Kendall’s tau",
        "rho-a": "Spearman’s rho",
    }

    # assert len(set([ds.n_obs for ds in datasets.values()])) == 1, (
    #     "datasets have different number of observations"
    # )

    if order is not None:
        assert all([id in datasets.keys() for id in order]), (
            "All entries of `order` must be in the datasets keys"
        )

        datasets = {id: datasets[id] for id in order}

    labels = list(datasets.keys())
    similarities = np.zeros((len(datasets), len(datasets)))

    iterator1 = tqdm(datasets.items(), disable=not (pbar1), leave=True)
    for i, (id1, ds1) in enumerate(iterator1):
        iterator2 = tqdm(datasets.items(), disable=not (pbar2), leave=True)
        for j, (id2, ds2) in enumerate(iterator2):
            ds1_new, ds2_new = match_datasets(ds1, ds2, descriptor_match)

            rdm1, rdm2 = [
                calc_rdm_clean(ds, method=dissimilarity_metric)
                for ds in [ds1_new, ds2_new]
            ]

            similarities[i, j] = compare_rdms(
                rdm1, rdm2, method=similarity_metric
            ).item()

    if show_fig or return_fig:
        fig, ax = plt.subplots()
        heatmap = ax.imshow(similarities)
        ax.set_xticks(range(len(datasets)), labels, rotation=90)
        ax.set_yticks(range(len(datasets)), labels)
        ax.set_title(title)
        plt.colorbar(heatmap, label=f"{sim_labels[similarity_metric]}")

        if show_fig:
            plt.show()
        else:
            plt.close()

    similarities = pd.DataFrame(similarities, columns=labels, index=labels)

    if return_fig:
        return similarities, fig
    else:
        return similarities


def rsa_bootstrap(
    dataset1,
    dataset2,
    dissimilarity_metric: str,
    similarity_metric: str,
    n_boot: int = 10_000,
    ci_percentiles: tuple = (2.5, 97.5),
    random_state: Optional[int] = None,
    pbar: bool = False,
    descriptor_match: Optional[str] = "item_id",
):
    """
    Bootstrap‐estimate the similarity between two RSA datasets.

    On each bootstrap iteration we resample condition indices (with replacement)
    jointly for both datasets, rebuild them, recompute RDMs, and correlate.

    Parameters
    ----------
    dataset1, dataset2 : Dataset
        Two rsatoolbox‐style Dataset objects with the same number of observations.
    dissimilarity_metric : str
        Passed to your calc_rdm function (e.g. "correlation", "euclidean", …).
    similarity_metric : str
        Passed to your compare_rdms function (e.g. "corr", "spearman", …).
    n_boot : int
        Number of bootstrap samples (default 1000).
    ci_percentiles : tuple of float
        Percentiles for confidence interval, e.g. (2.5, 97.5) for a 95% CI.
    random_state : int or None
        Seed for reproducibility.
    pbar : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    r_obs : float
        Observed similarity on the full datasets.
    r_boot : ndarray, shape (n_boot,)
        Bootstrap distribution of similarities.
    ci : tuple of float
        The lower and upper percentile CI of the bootstrap distribution.
    """

    # ! TEMP
    # n_boot = 1000
    # ci_percentiles =  (2.5, 97.5)
    # random_state = None
    # pbar = False
    # ! TEMP

    dataset1, dataset2 = match_datasets(dataset1, dataset2, descriptor_match)

    # * 0. Check shapes
    arr1 = dataset1.get_measurements()
    arr2 = dataset2.get_measurements()

    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("Both datasets must have the same number of observations.")

    n_cond = arr1.shape[0]

    # * 1. Compute observed RDMs & similarity
    rdm1 = calc_rdm_clean(dataset1, dissimilarity_metric)
    rdm2 = calc_rdm_clean(dataset2, dissimilarity_metric)
    r_obs = compare_rdms(rdm1, rdm2, method=similarity_metric).item()

    # * 2. Prepare for bootstrap
    rng = np.random.default_rng(random_state)
    r_boot = np.empty(n_boot)
    inds_all = np.arange(n_cond)

    iterator = tqdm(
        range(n_boot), desc="Bootstrapping", leave=False, disable=not (pbar)
    )

    def resample_dataset(ds, samp):
        arr = ds.get_measurements()[samp]

        obs_descriptors = {k: v[samp] for k, v in ds.obs_descriptors.items()}

        resampled_ds = Dataset(
            measurements=arr,
            obs_descriptors=obs_descriptors,
            descriptors=sanitize_descriptors(ds.descriptors),
            channel_descriptors=ds.channel_descriptors,
        )

        return resampled_ds

    # sample_inds = np.zeros((n_boot, n_cond)).astype(int)
    # unique_samples = {}

    # * 3. Bootstrap loop
    for i in iterator:
        # * 3a. sample condition indices
        samp = sorted(rng.choice(inds_all, size=n_cond, replace=True))

        # * 3b. build new Datasets with resampled measurements & obs-descriptors
        bs_ds1 = resample_dataset(dataset1, samp)
        bs_ds2 = resample_dataset(dataset2, samp)

        # * 3c. recompute RDMs & correlation
        bs_rdm1 = calc_rdm_clean(bs_ds1, dissimilarity_metric)
        bs_rdm2 = calc_rdm_clean(bs_ds2, dissimilarity_metric)

        assert np.isnan(bs_rdm1.get_vectors()).sum() == 0
        assert np.isnan(bs_rdm2.get_vectors()).sum() == 0

        r_boot[i] = compare_rdms(bs_rdm1, bs_rdm2, method=similarity_metric).item()

    # * 4. percentile CIs
    ci_lower, ci_upper = np.percentile(r_boot, ci_percentiles)

    return r_obs, r_boot, (ci_lower, ci_upper)


def rsa_permutation(
    dataset1,
    dataset2,
    dissimilarity_metric: str,
    similarity_metric: str,
    tail: str = "two-sided",
    n_perm: int = 10_000,
    random_state: Optional[int] = None,
    pbar: bool = False,
    descriptor_match: Optional[str] = "item_id",
    rdm_descriptor: Optional[str] = None,
):
    # ! TEMP
    # level = "pattern"
    # eeg_chan_group = "frontal"
    # human_data_dir = (
    #     EXPORT_DIR / f"Lab/analyzed/RSA-FRP-{eeg_chan_group}-metric_correlation"
    # )
    # # human_data_dir = EXPORT_DIR / "Lab/analyzed/RSA-Response_ERP-Frontal"

    # ANN_data_dir = (
    #     EXPORT_DIR / "ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer"
    # )
    # humans_datasets, ANNs_datasets = load_representation_datasets(
    #     human_data_dir, ANN_data_dir, level
    # )
    # dataset1 = list(ANNs_datasets.values())[0]
    # dataset2 = list(humans_datasets.values())[0]
    # dissimilarity_metric = "correlation"
    # similarity_metric = "corr"
    # n_perm = 100
    # pbar = True
    # ! TEMP

    allowed_tails = ["less", "greater", "two-sided"]
    assert tail in allowed_tails, f"`tail` must be one of {allowed_tails}"
    assert n_perm > 0, "`n_perm` must be at least 1"

    rng = np.random.default_rng(random_state)
    dataset1, dataset2 = match_datasets(dataset1, dataset2, descriptor_match)

    # * Compare the original RDMs
    rdm1, rdm2 = [
        calc_rdm_clean(d, dissimilarity_metric, descriptor=rdm_descriptor)
        for d in [dataset1, dataset2]
    ]
    observed_similarity = compare_rdms(rdm1, rdm2, similarity_metric).item()

    permuted_similarities = np.zeros(n_perm)

    dataset2_arr = dataset2.get_measurements()
    n_obs = dataset2_arr.shape[0]

    # * Permutation: Shuffle measurements relative to the fixed descriptors.
    iterator = tqdm(
        range(n_perm), desc="Permutation test", leave=False, disable=not (pbar)
    )

    for i in iterator:
        permuted_inds = rng.permutation(n_obs)

        permuted_dataset2_arr = dataset2_arr[permuted_inds]

        permuted_dataset2 = Dataset(
            measurements=permuted_dataset2_arr,
            obs_descriptors=dataset2.obs_descriptors,
            descriptors=sanitize_descriptors(dataset2.descriptors),
            channel_descriptors=dataset2.channel_descriptors,
        )

        # * Recompute the RDM using the permuted data
        permuted_rdm2 = calc_rdm_clean(
            permuted_dataset2,
            dissimilarity_metric,
            descriptor=rdm_descriptor,
        )

        # * Compare the first RDM with the second, permuted RDM
        permuted_similarity = compare_rdms(
            rdm1, permuted_rdm2, similarity_metric
        ).item()

        permuted_similarities[i] = permuted_similarity

    # * Calculate p-values
    null_dist = np.array(permuted_similarities)

    if tail == "greater":
        count = np.sum(null_dist > observed_similarity)
    elif tail == "less":
        count = np.sum(null_dist < observed_similarity)
    else:  # * two-sided
        count = np.sum(np.abs(null_dist) > abs(observed_similarity))

    # * bias-corrected p:
    # * (count+1) / (n+1) ensures you never get exactly zero, and is the recommended
    # * unbiased estimator for permutation p-values.
    # * Source: https://arxiv.org/abs/1603.05766
    # TODO: double check claim and source above

    p_val = (count + 1) / (n_perm + 1)

    # * Plot the distribution of permuted similarities + observed similarity
    # fig, ax = plt.subplots()
    # sns.histplot(null_dist, stat="density", label="Null", ax=ax)
    # if tail == "two-sided":
    #     sns.histplot(
    #         np.abs(null_dist), stat="density", alpha=0.5, label="|Null|", ax=ax
    #     )
    # ax.axvline(observed_similarity, color="red", linestyle="--", label="Observed")
    # ax.set_title("Permutation null distribution")
    # ax.set_xlabel("Similarity")
    # ax.set_ylabel("Density")
    # ax.legend()

    return observed_similarity, permuted_similarities, p_val


@dataclass
class RSA:
    datasets: Dict[str, Dataset]

    def __post_init__(self):
        raise NotImplementedError("This class hasn't been implemented yet.")
