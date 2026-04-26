import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from rsatoolbox.rdm import calc_rdm, compare as compare_rdms, calc_rdm_movie, RDMs
from rsatoolbox.data import Dataset, TemporalDataset
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

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
            descriptors=descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
        )
    else:
        dataset = TemporalDataset(
            measurements=measurements,
            descriptors=descriptors,
            obs_descriptors=obs_descriptors,
            channel_descriptors=channel_descriptors,
            time_descriptors=time_descriptors,
        )
    if fpath:
        dataset.save(fpath, file_type="hdf5", overwrite=True)

    return dataset


def create_and_save_rdm(
    dataset,
    dissimilarity_metric,
    fpath: Optional[Path] = None,
    ds_type: str = "regular",
):
    allowed_ds_types = ["regular", "temporal"]
    assert ds_type in allowed_ds_types, f"ds_type must be one of {allowed_ds_types}"

    if ds_type == "regular":
        rdm = calc_rdm(dataset, method=dissimilarity_metric)
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
    # # ! TEMP
    # ds1 = ANNs_datasets['Qwen2.5-72B']
    # ds2 = humans_datasets['subj_03']
    # {k: ds.n_obs for k, ds in humans_datasets.items()}
    # descriptor = 'item_ids'
    # # ! TEMP
    if descriptor is None:
        return ds1, ds2

    ds1_arr = ds1.get_measurements()
    ds1_obs_desc = ds1.obs_descriptors

    ds2_arr = ds2.get_measurements()
    ds2_obs_desc = ds2.obs_descriptors

    assert all([isinstance(v, np.ndarray) for v in ds1_obs_desc.values()])
    assert all([isinstance(v, np.ndarray) for v in ds2_obs_desc.values()])

    common = np.intersect1d(ds1_obs_desc[descriptor], ds2_obs_desc[descriptor])
    ds1_inds = [i for i, v in enumerate(ds1_obs_desc[descriptor]) if v in common]
    ds2_inds = [i for i, v in enumerate(ds2_obs_desc[descriptor]) if v in common]

    assert np.all(
        ds1_obs_desc[descriptor][ds1_inds] == ds2_obs_desc[descriptor][ds2_inds]
    )

    ds1_obs_desc_new = {k: v[ds1_inds] for k, v in ds1_obs_desc.items()}
    ds2_obs_desc_new = {k: v[ds2_inds] for k, v in ds2_obs_desc.items()}

    ds1_arr = ds1_arr[ds1_inds]
    ds2_arr = ds2_arr[ds2_inds]

    ds1_new = Dataset(
        measurements=ds1_arr,
        descriptors=ds1.descriptors,
        obs_descriptors=ds1_obs_desc_new,
        channel_descriptors=ds1.channel_descriptors,
    )

    ds2_new = Dataset(
        measurements=ds2_arr,
        descriptors=ds2.descriptors,
        obs_descriptors=ds2_obs_desc_new,
        channel_descriptors=ds2.channel_descriptors,
    )
    return ds1_new, ds2_new


def match_datasets_on_nan(ds1: Dataset, ds2: Dataset):
    ds1_arr = ds1.get_measurements()
    ds1_obs_desc = ds1.obs_descriptors

    ds2_arr = ds2.get_measurements()
    ds2_obs_desc = ds2.obs_descriptors

    assert ds1_arr.shape[0] == ds2_arr.shape[0]
    assert all([isinstance(v, np.ndarray) for v in ds1_obs_desc.values()])
    assert all([isinstance(v, np.ndarray) for v in ds2_obs_desc.values()])

    nan_rows_ds1 = np.unique(np.where(np.isnan(ds1_arr))[0])
    nan_rows_ds2 = np.unique(np.where(np.isnan(ds2_arr))[0])
    nan_rows = np.concatenate([nan_rows_ds1, nan_rows_ds2])
    mask = np.ones(ds1_arr.shape[0])
    mask[nan_rows] = 0
    mask = mask.astype(bool)

    ds1_obs_desc_new = {k: v[mask] for k, v in ds1_obs_desc.items()}
    ds2_obs_desc_new = {k: v[mask] for k, v in ds2_obs_desc.items()}

    ds1_new = Dataset(
        measurements=ds1_arr[mask],
        descriptors=ds1.descriptors,
        obs_descriptors=ds1_obs_desc_new,
        channel_descriptors=ds1.channel_descriptors,
    )

    ds2_new = Dataset(
        measurements=ds2_arr[mask],
        descriptors=ds2.descriptors,
        obs_descriptors=ds2_obs_desc_new,
        channel_descriptors=ds2.channel_descriptors,
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
    descriptor_match: Optional[str] = None,
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

    similarities = np.zeros((len(datasets), len(datasets)))

    iterator1 = tqdm(datasets.items(), disable=not (pbar1), leave=True)
    for i, (id1, ds1) in enumerate(iterator1):
        iterator2 = tqdm(datasets.items(), disable=not (pbar2), leave=True)
        for j, (id2, ds2) in enumerate(iterator2):
            ds1_new, ds2_new = match_datasets_on_descriptor(ds1, ds2, descriptor_match)
            ds1_new, ds2_new = match_datasets_on_nan(ds1_new, ds2_new)

            rdm1, rdm2 = [
                calc_rdm(ds, method=dissimilarity_metric) for ds in [ds1_new, ds2_new]
            ]

            similarities[i, j] = compare_rdms(
                rdm1, rdm2, method=similarity_metric
            ).item()

    if show_fig or return_fig:
        labels = list(datasets.keys())
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

    # * 0. Check shapes
    arr1 = dataset1.get_measurements()
    arr2 = dataset2.get_measurements()

    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("Both datasets must have the same number of observations.")

    n_cond = arr1.shape[0]

    # * 1. Compute observed RDMs & similarity
    rdm1 = calc_rdm(dataset1, dissimilarity_metric)
    rdm2 = calc_rdm(dataset2, dissimilarity_metric)
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
            descriptors=ds.descriptors,
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
        bs_rdm1 = calc_rdm(bs_ds1, dissimilarity_metric)
        bs_rdm2 = calc_rdm(bs_ds2, dissimilarity_metric)

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

    # * Compare the original RDMs
    rdm1, rdm2 = [calc_rdm(d, dissimilarity_metric) for d in [dataset1, dataset2]]
    observed_similarity = compare_rdms(rdm1, rdm2, similarity_metric).item()

    permuted_similarities = np.zeros(n_perm)

    dataset2_arr = dataset2.get_measurements()
    n_obs = dataset2_arr.shape[0]

    # * Permutation: Shuffle the condition labels in the dataset
    iterator = tqdm(
        range(n_perm), desc="Permutation test", leave=False, disable=not (pbar)
    )

    for i in iterator:
        permuted_inds = rng.permutation(n_obs)

        permuted_obs_desc = {
            k: np.asarray(v)[permuted_inds] for k, v in dataset2.obs_descriptors.items()
        }

        permuted_dataset2_arr = dataset2_arr[permuted_inds]

        permuted_dataset2 = Dataset(
            measurements=permuted_dataset2_arr,
            obs_descriptors=permuted_obs_desc,
            descriptors=dataset2.descriptors,
            channel_descriptors=dataset2.channel_descriptors,
        )

        # * Recompute the RDM using the permuted data
        permuted_rdm2 = calc_rdm(permuted_dataset2, dissimilarity_metric)

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
