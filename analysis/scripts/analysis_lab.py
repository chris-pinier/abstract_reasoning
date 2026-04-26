# * ########################################
# * IMPORTS
# * ########################################
import os
from matplotlib import pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
# TODO: from loguru import logger

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from abstract_reasoning_analysis.data_loader.human_data import (
    HumanSessData,
    HumanSubjData,
    HumanGroupData,
)
from abstract_reasoning_analysis.utils.custom_type_hints import DATA_FMTS
from abstract_reasoning_analysis.utils.analysis_utils import (
    read_file,
    reorder_item_ids,
    list_contents,
)
from abstract_reasoning_analysis.analysis_rsa import get_ds_and_rdm
from abstract_reasoning_analysis.analysis_config import Config as c

# * ########################################
# * GLOBAL VARIABLES & CONFIG
# * ########################################
# ! TEMP: to locate and use ffmpeg
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

pd.set_option("future.no_silent_downcasting", True)

# * Get the tensorpac logger
# tensorpac_logger = logging.getLogger("tensorpac")

# # * Set the logging level to WARNING or ERROR to suppress INFO messages
# tensorpac_logger.setLevel(logging.WARNING)  # Or logging.ERROR

# TODO: WARNING: unknown channels detected. Dropping:  ['EMG1', 'EMG2', 'EMG3', 'EMG4']: modify appropriate code to adjust when first slot was used instead of 7


def combine_frps():
    frp_dir = c.EXPORT_DIR / "analyzed/subj_lvl"
    frp_files = list_contents(frp_dir, r".*sess_frps.pkl$", recurs=True)

    # all_frp_data = {}
    subj_frp_data = {}
    subj_behav_data = {}

    for frp_file in tqdm(frp_files):
        subj_N, sess_N = [int(i[-2:]) for i in str(frp_file.parent).split("/")[-2:]]

        # if not subj_frp_data.get(subj_N):
        #     subj_frp_data[subj_N] = {}

        # if not subj_frp_data[subj_N].get(sess_N):
        #     subj_frp_data[subj_N] = []

        # if not subj_behav_data.get(subj_N):
        #     subj_behav_data[subj_N] = {}

        # if not subj_behav_data[subj_N].get(sess_N):
        #     subj_behav_data[subj_N] = []

        id = f"{subj_N:02}{sess_N:02}"

        frp_data = read_file(frp_file)["sequence"]
        behav_data = HumanSessData(
            c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, subj_N, sess_N
        ).get_behav_data()
        # subj_frp_data[subj_N][sess_N].append(frp_data)
        # subj_behav_data[subj_N][sess_N].append(behav_data)
        subj_frp_data[id] = frp_data
        subj_behav_data[id] = behav_data

        # frp_data = [d for d in frp_data if d is not None]
        # if not frp_data:
        #     continue
        # frp_data = mne.combine_evoked(frp_data, "equal")
        # all_frp_data[id] = frp_data

    _subj_frp_data = {k: [] for k in sorted(set([i[:2] for i in subj_frp_data.keys()]))}
    _subj_behav_data = {
        k: [] for k in sorted(set([i[:2] for i in subj_frp_data.keys()]))
    }

    for id in subj_frp_data.keys():
        subj_N = id[:2]
        _subj_frp_data[subj_N].append(subj_frp_data[id])
        _subj_behav_data[subj_N].append(subj_behav_data[id])

    # del subj_frp_data, subj_behav_data

    for subj_N in _subj_frp_data.keys():
        frp_data = _subj_frp_data[subj_N]

        behav_data = _subj_behav_data[subj_N]

        behav_data = pd.concat(behav_data).reset_index(drop=True)

        reordered_inds = reorder_item_ids(
            original_order_df=behav_data,
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )

        behav_data = behav_data.iloc[reordered_inds].reset_index(drop=True)
        patts = behav_data["pattern"].to_numpy()
        assert np.unique(patts, return_index=True)[1].shape[0] == len(c.PATTERNS)

        frp_data = np.array(frp_data).flatten()[reordered_inds]
        frp_data = np.array(
            [
                f.interpolate_bads(verbose="WARNING") if f is not None else None
                for f in frp_data
            ]
        )

        behav_data = behav_data.merge(
            c.ITEM_ID_SORT[["item_id"]], on="item_id", how="right"
        )
        assert all(behav_data["item_id"] == c.ITEM_ID_SORT["item_id"])

        template_frp_matrix = np.zeros(400).astype("object")
        template_frp_matrix[:] = None

        nan_mask = behav_data["subj_N"].isna()
        template_frp_matrix[~nan_mask] = frp_data
        frp_data = template_frp_matrix

        invalid_inds_frp = np.where(template_frp_matrix == None)[0]
        invalid_inds_beh = behav_data[nan_mask].index.to_numpy()
        assert all([i in invalid_inds_frp for i in invalid_inds_frp])

        _subj_frp_data[subj_N] = frp_data
        _subj_behav_data[subj_N] = behav_data

    assert (
        len(list(set([i.shape[0] for i in _subj_frp_data.values() if i is not None])))
        == 1
    )
    assert (
        len(list(set([i.shape for i in _subj_behav_data.values() if i is not None])))
        == 1
    )

    patt_groups = c.ITEM_ID_SORT.groupby("pattern").groups
    group_frp_data = {patt: [] for patt in patt_groups.keys()}

    for subj_N, frp_data in _subj_frp_data.items():
        for patt, inds in patt_groups.items():
            group_frp_data[patt].extend([f for f in frp_data[inds] if f is not None])

    for patt, d in group_frp_data.items():
        evoked = mne.combine_evoked(d, "equal")
        group_frp_data[patt] = evoked

    avg_frp = {}
    all_chans = c.EEG_CHAN_GROUPS["all"]
    frontal_chans = c.EEG_CHAN_GROUPS["frontal"]
    frontal_chans_inds = [i for i, ch in enumerate(all_chans) if ch in frontal_chans]

    for patt, group_evoked in group_frp_data.items():
        # print(group_evoked.info['bads'])
        fig, ax = plt.subplots(figsize=(10, 4), dpi=500)
        # _data = group_evoked.pick(frontal_chans)
        _data = group_evoked.copy()
        # _data = _data.interpolate_bads()

        evoked_plot = _data.plot(axes=ax, show=False)
        evoked_plot.get_axes()[0].set_title(patt)
        lines = evoked_plot.get_axes()[0].get_lines()

        [
            l.set_color("black")
            for i, l in enumerate(lines)
            if i not in frontal_chans_inds
        ]
        [l.set_alpha(0.4) for i, l in enumerate(lines) if i not in frontal_chans_inds]
        [t.remove() for t in evoked_plot.get_axes()[0].texts]

        plt.show()
        plt.close()
        plots[patt] = fig

        avg_frp[patt] = _data.get_data(picks=frontal_chans).mean(axis=0)

    times = group_frp_data["AAABAAAB"].times
    mask = np.where(times > 0)[0]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=500)
    for patt, avg_group_evoked in avg_frp.items():
        ax.plot(times[mask], avg_group_evoked[mask] * 1e6, label=patt)
        # ax.plot(times, avg_group_evoked, label=patt)
    # ax.vlines(times[mask[0]], *ax.get_ylim(), ls="--")
    ax.set_ylabel("µV")
    ax.set_xlabel("Time (s)")
    ax.legend()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(2, 1)
    plt.show()


def main():
    # * ########################################
    # * Session Level Analysis
    # * ########################################
    s0101 = HumanSessData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 1, 1)
    self = s0101
    stim_flash_order = s0101.get_stim_flash_order()
    stim_flash_order["cube"]
    behav_data = s0101.get_behav_data()

    seq_order = behav_data["seq_order"].apply(lambda x: np.array([int(i) for i in x]))
    choice_order = behav_data["choice_order"].apply(
        lambda x: np.array([int(i) for i in x])
    )
    choice_order += 8

    s0102 = HumanSessData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 1, 2)
    # s0102.analyze_session(save_dir=WD / "test-export", preprocessed_dir=c.PREPROCESSED_DIR, raise_error=False)
    self = s0102
    print(s0102)

    # s0101.show_dir_struct()
    # self = HumanSessData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 3, 5)
    # sess_info, raw_behav, raw_eeg, raw_et, et_cals = s0101.get_raw_data()
    # sess_info, prepro_behav, prepro_eeg, prepro_et, et_cals = s0101.get_data()
    # s0101.check_data()

    # s0101_perf_res = s0101.analyze_perf()

    # sess_res = s0101.analyze_session(WD / "test-export", c.PREPROCESSED_DIR)
    # (
    #     sess_frps,
    #     fixation_data,
    #     eeg_fixation_data,
    #     gaze_info,
    #     gaze_target_fixation_sequence,
    # ) = sess_res

    # * ########################################
    # * Subject Level Analysis
    # * ########################################
    s01 = HumanSubjData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 1)
    print(s01)
    # s01.show_dir_struct()
    # self = s01

    # self.check_data()

    s02 = HumanSubjData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 4)
    self = s02

    # self = s01.sessions[1]
    all_sess_res = s01.analyze_sessions(
        WD / "test-export", c.PREPROCESSED_DIR, raise_error=False
    )

    # s01_perf_res = s01.analyze_perf(agg="suject")
    # s01_perf_res["acc_by_patt"]
    # behav_data = s01.get_behav_data()

    # * ########################################
    # * Group Level Analysis
    # * ########################################
    group = HumanGroupData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR)
    metadata = group.extract_eeg_metadata()

    self2 = group
    combined_stats, acc_fig, rt_fig, scatter = group.summarize_behav()
    # combined_stats['rt']

    behav_group = group.get_behav_data()
    behav_group.choice.unique()
    behav_group.choice_key.unique()
    behav_group.columns

    behav_group["iti"].value_counts()

    # stim_locs = group.get_stim_locs()
    # stim_locs["star"]

    behav_group.groupby(["subj_N"])["correct"].mean()

    behav_group.groupby(["subj_N", "sess_N", "pattern"])["correct"].mean().groupby(
        "pattern"
    ).mean()
    behav_group.groupby(["subj_N", "pattern"])["correct"].mean().groupby(
        "pattern"
    ).mean()
    behav_group.groupby(["sess_N", "pattern"])["correct"].mean().groupby(
        "pattern"
    ).mean()
    behav_group.groupby("pattern")["correct"].mean()

    behav_group.groupby("pattern")["rt"].mean()


if __name__ == "__main__":
    # main()
    pass
