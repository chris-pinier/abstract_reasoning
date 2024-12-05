import matplotlib.pyplot as plt
import numpy as np
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
# eeg_chan_groups =


def erp_analysis(raw_eeg, eeg_events, valid_events):
    # * ## SEQUENCE FLASHES ##
    epochs_stim_flash_seq = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=valid_events["stim-flash_sequence"],
        baseline=(-0.1, 0),
    )
    evoked_stim_flash_seq = epochs_stim_flash_seq.average()

    evoked_stim_flash_seq.plot()
    plt.show()

    # * ## CHOICE FLASHES ##
    epochs_stim_flash_choices = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=valid_events["stim-flash_choices"],
        baseline=(-0.1, 0),
    )
    evoked_stim_flash_choices = epochs_stim_flash_choices.average()

    evoked_stim_flash_choices.plot()
    plt.show()

    # * ## SEQUENCE + CHOICE FLASHES ##
    epochs_stim_flash_all = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=0.6,
        event_id=[
            valid_events[i] for i in ["stim-flash_choices", "stim-flash_choices"]
        ],
        baseline=(-0.1, 0),
    )

    evoked_stim_flash_all = epochs_stim_flash_all.average()

    evoked_stim_flash_all.plot()
    plt.show()

    # * ## ALL STIM PRESENTAION ##
    epochs_stim_flash_all = mne.Epochs(
        raw_eeg,
        eeg_events,
        tmin=-0.1,
        tmax=1,
        event_id=valid_events["stim-all_stim"],
        baseline=(-0.1, 0),
    )

    evoked_stim_flash_all = epochs_stim_flash_all.average()

    evoked_stim_flash_all.plot()
    plt.show()


def rsa_with_rsatoolbox(all_subj_pattern_erps, eeg_chan_groups):
    # import rsatoolbox

    # from scipy.signal import find_peaks

    # Example data: patterns x features x participants
    # Replace with your EEG or other data
    data = np.random.rand(10, 64, 20)  # 10 patterns, 64 features, 20 participants

    # Create an rsatoolbox Dataset
    labels = {"pattern_type": ["type1", "type2", "type3"]}  # Add your conditions
    dataset = Dataset(measurements=data, descriptors=labels)
    data.shape

    calc_rdm
    # Compute RDMs for each participant
    rdms = calc_rdm(dataset, method="correlation")  # Correlation-based dissimilarity
    # ##################################################################################

    p_t = list(all_subj_pattern_erps.keys())
    chan_names = all_subj_pattern_erps[p_t[0]].ch_names
    excl = chan_names[27]
    incl = [c for c in chan_names if c != excl]

    occipital_chans = [c for c in incl if c in eeg_chan_groups["occipital"]]

    evoked_times = all_subj_pattern_erps[p_t[0]].times
    time_window = (0.025, 0.2)
    time_mask = (evoked_times >= time_window[0]) & (evoked_times <= time_window[1])

    features = []

    for pattern, pattern_data in all_subj_pattern_erps.items():
        _data = pattern_data.get_data(picks=occipital_chans).copy()

        # plt.plot(_data.T, color="black")
        # plt.plot(_data.mean(axis=0), color="red")
        # vlines = np.where(time_mask)[0]
        # vlines = [vlines[0], vlines[-1]]
        # plt.vlines(vlines, ymin=_data.min(), ymax=_data.max(), color='blue')
        # plt.xticks(np.arange(0, len(evoked_times), 150), evoked_times[::150].round(3)*1000)

        _data = _data.mean(axis=0)
        neg_peak = _data[time_mask].argmin()
        peak_latency = evoked_times[time_mask][neg_peak]
        peak_amp = _data[time_mask][neg_peak]

        # plt.hlines(peak_amp, xmin=0, xmax=len(_data), color='r')
        # plt.vlines(np.where(evoked_times == peak_latency)[0], ymin=_data.min(), ymax=_data.max(), color='r')
        # plt.tight_layout()
        # plt.show()
        features.append([peak_latency, peak_amp])
        # features.append(peak_latency)

        print(f"peak latency = {peak_latency:.5f}  | peak amplitude = {peak_amp:.9f}")

    # Step 5: Create rsatoolbox Dataset
    dataset = Dataset(
        # measurements=np.array(features)[:, None]
        measurements=np.array(features),
        # descriptors={'pattern_type': p_t},
        # descriptors={'participants': participants},
        # obs_descriptors={'pattern_type': [pattern_type] * len(participants)},
        channel_descriptors={"feature": ["latency", "amplitude"]},
    )

    # np.array(features)[:, None].shape
    # rdms = calc_rdm(dataset, method='correlation')
    rdms = calc_rdm(dataset, method="correlation")

    this_rdm = rdms.get_matrices()[0]

    plt.imshow(this_rdm)
    plt.xticks(np.arange(0, len(p_t)), p_t, rotation=90)
    plt.yticks(
        np.arange(0, len(p_t)),
        p_t,
    )
    plt.title("RDM by pattern type")

    normalized_rdm = this_rdm / this_rdm.max()

    for (j, i), label in np.ndenumerate(normalized_rdm):
        plt.text(i, j, int(round(label, 2) * 100), ha="center", va="center")
    plt.colorbar()


def rsa_with_rsatoolbox2(subj_pattern_erps):
    import numpy as np
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm

    # Define the time window for analysis
    time_window = (0.025, 0.2)

    # Function to extract latency and amplitude of the first negative peak
    def extract_first_negative_peak(evoked, time_window):
        time_idx = (evoked.times >= time_window[0]) & (evoked.times <= time_window[1])
        data = evoked.get_data().mean(axis=0)  # Average across channels
        time_values = evoked.times[time_idx]
        data_values = data[time_idx]

        # Find first negative peak
        neg_peak_idx = np.argmin(data_values)  # Index of the most negative point
        neg_peak_latency = time_values[neg_peak_idx]
        neg_peak_amplitude = data_values[neg_peak_idx]

        return neg_peak_latency, neg_peak_amplitude

    # Dictionary to store RDMs
    rdms_by_participant = {}

    # Iterate over participants
    for participant, patterns in subj_pattern_erps.items():
        feature_matrix = []
        pattern_labels = []

        # Extract features for each pattern
        for pattern_name, evoked in patterns.items():
            latency, amplitude = extract_first_negative_peak(evoked, time_window)
            feature_matrix.append([latency, amplitude])
            pattern_labels.append(pattern_name)

        feature_matrix = np.array(feature_matrix)

        # Create rsatoolbox Dataset
        dataset = Dataset(
            measurements=feature_matrix,
            obs_descriptors={"pattern": pattern_labels},
            channel_descriptors={"features": ["latency", "amplitude"]},
        )

        # Calculate RDM
        rdm = calc_rdm(dataset, method="euclidean")
        rdms_by_participant[participant] = rdm

    # Display the RDMs for each participant
    for participant, rdm in rdms_by_participant.items():
        print(f"RDM for Participant {participant}:")
        # print(rdm)

    [i for i in dir(rdms_by_participant[1]) if not i.startswith("_")]
    rdms_by_participant[1].to_df()
    rdms_by_participant[1].n_rdm
    rdms_by_participant[1].n_cond

    rdms_by_participant[1].get_matrices()


def draft():
    montage = mne.channels.make_standard_montage("biosemi64")
    chans_pos_xy = np.array(list(montage.get_positions()["ch_pos"].values()))[:, :2]

    # * Select EEG channel groups to plot
    selected_chan_groups = {
        k: v
        for k, v in eeg_chan_groups.items()
        if k
        in [
            "frontal",
            "parietal",
            "temporal",
            "occipital",
        ]
    }

    group_colors = dict(
        zip(selected_chan_groups.keys(), ["red", "green", "purple", "orange"])
    )

    ch_group_inds = {
        region: [i for i, ch in enumerate(montage.ch_names) if ch in ch_group]
        for region, ch_group in selected_chan_groups.items()
    }

    eeg_baseline = 0.1

    analyzed_res_dir = wd / "results/analyzed/with_ica_eye_removal"
    analyzed_res = {}
    for subj_dir in analyzed_res_dir.glob("*"):
        if subj_dir.is_dir():
            subj = subj_dir.stem
            for sess_dir in subj_dir.glob("*"):
                if sess_dir.is_dir():
                    sess = sess_dir.stem
                    for res_file in sess_dir.glob("*.pkl"):
                        res_name = res_file.stem
                        with open(res_file, "rb") as f:
                            res = pickle.load(f)
                        # print(res)
                        analyzed_res[(subj, sess, res_name)] = res

    analyzed_res.keys()
    analyzed_res.keys()

    for i in range(0, len(analyzed_res), 3):
        subj_sess1_erps = list(analyzed_res.values())[i]

        sequence_erps = subj_sess1_erps["sequence"]
        sequence_erps = [arr for arr in sequence_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in sequence_erps])
        sequence_erps = [arr[:, :min_len] for arr in sequence_erps]

        choice_erps = subj_sess1_erps["choices"]
        choice_erps = [arr for arr in choice_erps if not np.isnan(arr).all()]
        min_len = min([arr.shape[1] for arr in choice_erps])
        choice_erps = [arr[:, :min_len] for arr in choice_erps]

        set([arr.shape for arr in sequence_erps])
        np.stack(sequence_erps, axis=0).shape

        mean_sequence_erp = np.mean(np.stack(sequence_erps, axis=0), axis=0)
        mean_choices_erp = np.mean(np.stack(choice_erps, axis=0), axis=0)

        xticks = np.arange(
            0, mean_sequence_erp.shape[1] + 0.1 * eeg_sfreq, 0.1 * eeg_sfreq
        )
        xlabels = np.arange(
            -0.1, (mean_sequence_erp.shape[1] / eeg_sfreq) - 0.1 + 0.1, 0.1
        ).round(2)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(mean_sequence_erp.T)
        ax1.set_xlim(0, mean_sequence_erp.shape[1])
        ax1.set_xticks(xticks, xlabels)

        ax2.plot(mean_choices_erp.T)
        ax2.set_xlim(0, mean_choices_erp.shape[1])
        ax2.set_xticks(xticks, xlabels)

        # for trial_erps in subj_sess1_erps:
        #     # i = 0 # ! TEMP
        #     trial_erps = [subj_sess1_erps["sequence"][i], subj_sess1_erps["choices"][i]]
        #     trial_erps = [
        #         np.mean(v, axis=0) if not np.isnan(v).all() else np.array([]) for v in trial_erps
        #     ]
        # trial_erps[0].shape

        # fig_titles = ["ERP - Sequence Icons", "ERP - Choice Icons"]

        # for eeg_data, title in zip(trial_erps, fig_titles):
        #     fig = plot_eeg(
        #         eeg_data * 1e6,
        #         chans_pos_xy,
        #         ch_group_inds,
        #         group_colors,
        #         eeg_sfreq,
        #         eeg_baseline,
        #         vlines=None,
        #         title=title,
        #     )


def draft2_behav():
    # from box import Box
    behav_files = list(data_dir.rglob("*behav*.csv"))
    behav_dfs = []

    for f in behav_files:
        df = pd.read_csv(f, index_col=0)
        df.insert(1, "sess", int(f.parent.stem.split("_")[-1]))
        df.reset_index(drop=False, inplace=True, names=["trial_N"])
        behav_dfs.append(df)

    group_behav_df = pd.concat(behav_dfs)
    group_behav_df.reset_index(drop=True, inplace=True)
    group_behav_df["correct"] = group_behav_df["correct"].astype(str)

    timeout_trials = group_behav_df.query("rt=='timeout'").copy()
    group_behav_df_clean_rt = group_behav_df.copy()

    group_behav_df_clean_rt["rt"] = (
        group_behav_df_clean_rt["rt"].replace({"timeout": np.nan}).astype(float)
    )

    group_behav_df_clean_correct = group_behav_df.copy()

    group_behav_df_clean_correct["correct"] = (
        group_behav_df_clean_correct["correct"]
        .replace({"invalid": False, "True": True, "False": False})
        .astype(bool)
    )
    group_behav_df_clean_correct["correct"].value_counts()

    group_behav_df_clean_rt.groupby("pattern")["rt"].mean().plot(kind="bar")
    group_behav_df_clean_correct.groupby("pattern")["correct"].mean().plot(kind="bar")

    def group_plot(rt_data, correct_data, rt_lim):
        fig, ax = plt.subplots(2, 1)
        rt_data.plot(kind="bar", ax=ax[0])
        correct_data.plot(kind="bar", ax=ax[1])

        ax[0].set_title(f"Mean RT per pattern (s)")
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)
        ax[0].grid(axis="y", ls="--")
        ax[0].set_ylim(0, rt_lim)
        ax[0].legend(
            title=ax[0].get_legend().get_title().get_text(),
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
        )

        ax[1].set_title(f"Mean accuracy per pattern")
        ax[1].grid(axis="y", ls="--")
        ax[1].set_ylim(0, 1)
        ax[1].legend(
            title=ax[1].get_legend().get_title().get_text(),
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
        )
        plt.tight_layout()
        # plt.show()
        return fig

    # * Goup Figure 1
    group_fig1 = group_plot(
        rt_data=(
            group_behav_df_clean_rt.groupby(["pattern", "sess"])["rt"].mean().unstack()
        ),
        correct_data=(
            group_behav_df_clean_correct.groupby(["pattern", "sess"])["correct"]
            .mean()
            .unstack()
        ),
        rt_lim=group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50,
    )

    # * Goup Figure 2
    group_fig2 = group_plot(
        rt_data=(
            group_behav_df_clean_rt.groupby(["pattern", "subj_id"])["rt"]
            .mean()
            .unstack()
        ),
        correct_data=(
            group_behav_df_clean_correct.groupby(["pattern", "subj_id"])["correct"]
            .mean()
            .unstack()
        ),
        rt_lim=group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50,
    )

    for participant in sorted(group_behav_df["subj_id"].unique()):
        fig, ax = plt.subplots(2, 1)
        # group_plot()

        group_behav_df_clean_rt.query("subj_id==@participant").groupby("pattern")[
            "rt"
        ].mean().sort_index().plot(kind="bar", ax=ax[0])
        # plt.show()
        group_behav_df_clean_correct.query("subj_id==@participant").groupby("pattern")[
            "correct"
        ].mean().sort_index().plot(kind="bar", ax=ax[1])

        ax[0].set_title(f"Participant {participant}\nMean RT per pattern (s)")
        ax[0].set_xticklabels([])
        ax[0].set_xlabel(None)

        ax[0].grid(axis="y", ls="--")
        ax[1].grid(axis="y", ls="--")

        ax[0].set_ylim(
            0, group_behav_df_clean_rt.groupby("subj_id")["rt"].mean().max() * 1.50
        )
        ax[1].set_ylim(0, 1)
        ax[1].set_title(f"Mean accuracy per pattern")
        plt.tight_layout()
        plt.show()

        group_behav_df_clean_correct.query("subj_id==@participant").groupby("sess")[
            "correct"
        ].mean().plot(kind="bar")
        plt.show()

        group_behav_df_clean_correct.query("subj_id==@participant")


# * ####################################################################################
# * ANALYZE FLASH PERIOD
# * METHOD 1
# * ####################################################################################


def analyze_flash_period_draft1(
    manual_et_epochs, manual_eeg_epochs, eeg_sfreq, et_sfreq, tracked_eye
):
    # for epoch_N in tqdm(range(len(manual_et_epochs))):
    for epoch_N in tqdm(range(2, 6)):
        # epoch_N = 6 # ! TEMP

        # * Extract epoch data
        et_epoch = manual_et_epochs[epoch_N]
        eeg_epoch = manual_eeg_epochs[epoch_N]

        # * GET SACCADE AND FIXATION EVENTS
        et_trial_evts, et_trial_evt_ids = mne.events_from_annotations(et_epoch)

        # et_trial_evts = pd.DataFrame(
        #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
        # )
        # et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

        # diff_indices = np.where(pd.Series(et_trial_evts[:, -1]).diff() != 0)[0]
        # et_trial_evts = et_trial_evts[diff_indices]
        # et_trial_evts = pd.DataFrame(
        #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
        # )
        # et_trial_evts["event_id"] = et_trial_evts["event_id"].replace({v:k for k,v in et_trial_evt_ids.items()})
        # # list(zip(et_trial_evts["sample_nb"][:-1], et_trial_evts["sample_nb"][1:]))
        # # fixation_evts = et_trial_evts[et_trial_evts["event_id"] == "fixation"]
        # # et_trial_evts['event_id'].tolist()

        # * Get channel positions for topomap
        # info = eeg_epoch.info

        # chans_pos_xy = np.array(
        #     list(info.get_montage().get_positions()["ch_pos"].values())
        # )[:, :2]

        trial_info = get_trial_info(epoch_N)
        stim_pos, stim_order = trial_info[:2]

        # * Resample eye-tracking data for the current trial
        x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
            et_epoch, tracked_eye, et_sfreq, eeg_sfreq
        )

        # * Synchronize data lengths
        eeg_data = eeg_epoch.get_data()

        # eeg_trial_evts, epoch_evt_ids = mne.events_from_annotations(eeg_epoch)
        eeg_trial_evts = pd.Series(eeg_data[-1, :])

        # * Find indices where consecutive events are different
        diff_indices = np.where(eeg_trial_evts.diff() != 0)[0]
        eeg_trial_evts = eeg_trial_evts[diff_indices]
        eeg_trial_evts = eeg_trial_evts[eeg_trial_evts != 0]
        eeg_trial_evts = eeg_trial_evts.replace(valid_events_inv)

        # * Drop EOG & Status channels
        eeg_data = eeg_data[:-5, :]

        # * Ensure that the data arrays have the same length
        min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
        eeg_data = eeg_data[:, :min_length]
        # avg_eeg_data = eeg_data.mean(axis=0)

        x_gaze_resampled = x_gaze_resampled[:min_length]
        y_gaze_resampled = y_gaze_resampled[:min_length]

        # * Heatmap of gaze data
        all_stim_onset = eeg_trial_evts[eeg_trial_evts == "stim-all_stim"].index[0]
        trial_end = eeg_trial_evts[eeg_trial_evts == "trial_end"].index[0]

        heatmap, _, _ = get_gaze_heatmap(
            x_gaze_resampled[all_stim_onset:trial_end],
            y_gaze_resampled[all_stim_onset:trial_end],
            screen_resolution,
            bin_size=100,
            show=True,
        )

        stim_flash_evts = eeg_trial_evts[eeg_trial_evts.str.contains("stim-")]
        event_bounds = stim_flash_evts.index
        event_bounds = list(zip(event_bounds[:-1], event_bounds[1:]))
        event_bounds = list(zip(stim_flash_evts.values, event_bounds))

        et_data = np.array([x_gaze_resampled, y_gaze_resampled]).T

        targets_fixation = {}

        for i, (event_id, ev_bounds) in enumerate(event_bounds):
            event_et_data = et_data[ev_bounds[0] : ev_bounds[1]]
            target_grid_loc = stim_order[i]
            target_id, target_coords = stim_pos[target_grid_loc]
            targ_left, targ_right, targ_bottom, targ_top = target_coords

            on_target_inds = [[]]
            for j, (eye_x, eye_y) in enumerate(event_et_data):
                on_target = (
                    targ_left <= eye_x <= targ_right
                    and targ_bottom <= eye_y <= targ_top
                )
                if on_target:
                    on_target_inds[-1].append(ev_bounds[0] + j)
                else:
                    if len(on_target_inds[-1]) > 0:
                        on_target_inds.append([])

            for inds_list in on_target_inds:
                if len(inds_list) < 5:
                    on_target_inds.remove(inds_list)

            targets_fixation[target_grid_loc] = on_target_inds

        # * TESTING
        for targ_ind in targets_fixation:
            target_grid_loc = targ_ind
            target_id, target_coords = stim_pos[target_grid_loc]
            targ_left, targ_right, targ_bottom, targ_top = target_coords

            for fixation in targets_fixation[targ_ind]:
                fig, ax = plt.subplots(3, 1, figsize=(10, 6))
                ax_et, ax_eeg, ax_eeg_avg = ax
                ax_et.set_xlim(0, screen_resolution[0])
                ax_et.set_ylim(screen_resolution[1], 0)
                ax_et.imshow(
                    icon_images[target_id],
                    extent=[targ_left, targ_right, targ_bottom, targ_top],
                    origin="lower",
                )

                eye_pos_inds = fixation
                x, y = et_data[eye_pos_inds].T
                ax_et.scatter(x, y, c="red", s=2)

                # duration_ms = len(eye_pos_inds) / eeg_sfreq * 1000

                ax_eeg.plot(eeg_data[:-5, eye_pos_inds].T)
                ax_eeg_avg.plot(eeg_data[:-5, eye_pos_inds].T.mean(axis=1))

                xticks = np.arange(
                    0, eeg_data[:-5, eye_pos_inds].T.mean(axis=1).shape[0], 100
                )
                ax_eeg_avg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

                plt.show()


# * ####################################################################################
# * ANALYZE FLASH PERIOD
# * METHOD 2
# * Problem: if fixation on same location as target before first flash,
# * fixation data before first flash will be excluded
# * ####################################################################################


def analyze_flash_period_draft2(et_epoch, eeg_epoch, stim_pos, stim_order):
    # ! IMPORTANT
    # TODO
    # * Right Now et_epoch_evts are extracted from raw ET data, but EEG data is sampled
    # * at a different rate. So EEG activity might not be aligned with ET data
    # * Possible solutions:
    # *     1. Use resampled ET data and identify events from this data (might not be possible)
    # *     2. Use time as index instead of sample number, and identify closest sample number
    # *        in EEG data for each event
    # TODO

    # # * Extract epoch data
    # et_epoch = manual_et_epochs[epoch_N]
    # eeg_epoch = manual_eeg_epochs[epoch_N]

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target

        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        first_flash = annotations.query("description == 'stim-flash_sequence'").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]
        last_fixation = annotations.query("description == 'fixation'").iloc[-1]

        # * Convert to seconds
        first_flash_onset = first_flash["onset"]
        all_stim_pres_onset = all_stim_pres["onset"]
        end_time = all_stim_pres_onset + last_fixation["duration"]

        # * Crop the data
        epoch = epoch.copy().crop(first_flash_onset, end_time)

        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # if annotations["description"].iloc[0] != "stim-flash_sequence":
        #     annotations = annotations.iloc[
        #         first_flash.name : all_stim_pres.name + 1
        #     ].copy()

        return epoch, annotations

    et_epoch, et_annotations = crop_et_epoch(et_epoch)

    eeg_epoch = eeg_epoch.copy().crop(
        et_annotations["onset"].iloc[0], et_annotations["onset"].iloc[-1]
    )

    # eeg_epoch.times[-1]
    # eeg_epoch.times[-1]

    # * ########
    # eeg_events, _ = mne.events_from_annotations(eeg_epoch, event_id=valid_events)
    # eeg_events[:, 0] -= eeg_events[0, 0]

    # first_flash_eeg = eeg_events[eeg_events[:, 2] == valid_events["stim-flash_sequence"]][0, 0]
    # all_stim_pres_eeg = eeg_events[eeg_events[:, 2] == valid_events["stim-all_stim"]][0, 0]

    # eeg_epoch = eeg_epoch.get_data()[:, first_flash_eeg : all_stim_pres_eeg + 1]
    # * ########

    # et_trial_evts = pd.DataFrame(
    #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
    # )
    # et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

    # diff_indices = np.where(pd.Series(et_trial_evts[:, -1]).diff() != 0)[0]
    # et_trial_evts = et_trial_evts[diff_indices]
    # et_trial_evts = pd.DataFrame(
    #     et_trial_evts, columns=["sample_nb", "prev", "event_id"]
    # )
    # et_trial_evts["event_id"] = et_trial_evts["event_id"].replace({v:k for k,v in et_trial_evt_ids.items()})
    # # list(zip(et_trial_evts["sample_nb"][:-1], et_trial_evts["sample_nb"][1:]))
    # # fixation_evts = et_trial_evts[et_trial_evts["event_id"] == "fixation"]
    # # et_trial_evts['event_id'].tolist()

    # * Get channel positions for topomap
    # info = eeg_epoch.info

    # chans_pos_xy = np.array(
    #     list(info.get_montage().get_positions()["ch_pos"].values())
    # )[:, :2]

    # trial_info = get_trial_info(epoch_N)
    # stim_pos, stim_order = trial_info[:2]

    # * Resample eye-tracking data for the current trial
    # x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
    #     et_epoch, tracked_eye, et_sfreq, eeg_sfreq
    # )

    # * Synchronize data lengths
    # eeg_data = eeg_epoch.get_data()

    # eeg_trial_evts, epoch_evt_ids = mne.events_from_annotations(eeg_epoch)
    # eeg_trial_evts = pd.Series(eeg_data[-1, :])

    # * Find indices where consecutive events are different
    # diff_indices = np.where(eeg_trial_evts.diff() != 0)[0]
    # eeg_trial_evts = eeg_trial_evts[diff_indices]
    # eeg_trial_evts = eeg_trial_evts[eeg_trial_evts != 0]
    # eeg_trial_evts = eeg_trial_evts.replace(valid_events_inv)

    # * Drop EOG & Status channels
    # eeg_data = eeg_data[:-5, :] # ! OLD VERSION
    eeg_data = eeg_epoch.copy().pick(["eeg"])
    # eog_data = eeg_epoch.copy().pick(["eog"])

    # * Ensure that the data arrays have the same length
    # min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
    # eeg_data = eeg_data[:, :min_length]
    # avg_eeg_data = eeg_data.mean(axis=0)

    # x_gaze_resampled = x_gaze_resampled[:min_length]
    # y_gaze_resampled = y_gaze_resampled[:min_length]

    # * Heatmap of gaze data
    # all_stim_onset = eeg_trial_evts[eeg_trial_evts == "stim-all_stim"].index[0]
    # trial_end = eeg_trial_evts[eeg_trial_evts == "trial_end"].index[0]

    # heatmap, _, _ = get_gaze_heatmap(
    #     x_gaze_resampled[all_stim_onset:trial_end],
    #     y_gaze_resampled[all_stim_onset:trial_end],
    #     screen_resolution,
    #     bin_size=100,
    #     show=True,
    # )

    # stim_flash_evts = eeg_trial_evts[eeg_trial_evts.str.contains("stim-")]
    # event_bounds = stim_flash_evts.index
    # event_bounds = list(zip(event_bounds[:-1], event_bounds[1:]))
    # event_bounds = list(zip(stim_flash_evts.values, event_bounds))

    # * ################################################################################
    ch_group_inds = {
        region: [i for i, ch in enumerate(eeg_epoch.ch_names) if ch in ch_group]
        for region, ch_group in eeg_chan_groups.items()
    }

    # * GET SACCADE AND FIXATION EVENTS
    # et_epoch_evts, _ = mne.events_from_annotations(et_epoch, event_id=et_events_dict)
    # et_epoch_evts = pd.DataFrame(et_epoch_evts, columns=["sample_nb", "prev", "description"])
    # et_epoch_evts['description'] = et_epoch_evts['description'].replace(et_events_dict_inv)

    # et_evt_df = et_epoch.annotations.to_data_frame()
    # # et_trial_start = et_evt_df[et_evt_df["description"] == "trial_start"].index[0]
    # # et_trial_end = et_evt_df[et_evt_df["description"] == "trial_end"].index[0]
    # et_first_flash = et_evt_df[et_evt_df["description"] == "stim-flash_sequence"].index[
    #     0
    # ]
    # et_all_stim_pres = et_evt_df[et_evt_df["description"] == "stim-all_stim"].index[0]

    # et_evt_df = et_evt_df.iloc[et_first_flash : et_all_stim_pres + 1]
    # et_evt_df.reset_index(drop=True, inplace=True)
    # et_evt_df["onset"] = (
    #     (et_evt_df["onset"] - et_evt_df["onset"].iloc[0]).dt.total_seconds().round(3)
    # )

    # eeg_evt_df = eeg_epoch.annotations.to_data_frame()
    # # eeg_trial_start = eeg_evt_df[eeg_evt_df["description"] == "trial_start"].index[0]
    # # eeg_trial_end = eeg_evt_df[eeg_evt_df["description"] == "trial_end"].index[0]

    # eeg_first_flash = eeg_evt_df[
    #     eeg_evt_df["description"] == "stim-flash_sequence"
    # ].index[0]

    # eeg_all_stim_pres = eeg_evt_df[eeg_evt_df["description"] == "stim-all_stim"].index[
    #     0
    # ]
    # eeg_evt_df = eeg_evt_df.iloc[eeg_first_flash : eeg_all_stim_pres + 1]
    # eeg_evt_df.reset_index(drop=True, inplace=True)
    # eeg_evt_df["onset"] = (
    #     (eeg_evt_df["onset"] - eeg_evt_df["onset"].iloc[0]).dt.total_seconds().round(3)
    # )
    # # eeg_evt_df["onset"] = eeg_evt_df["onset"].round(3)

    # # ! TEMP
    # et_evt_df.query("not description.str.contains('fixation|saccade')")
    # eeg_evt_df

    # et_evt_df.tail(20)
    # # ! TEMP

    # ! OLD VERSION
    # et_epoch_evts[:, 0] -= et_epoch_evts[0, 0]

    # et_epoch_evts = pd.DataFrame(
    #     et_epoch_evts, columns=["sample_nb", "prev", "event_id"]
    # )

    # et_epoch_evts["event_id"] = et_epoch_evts["event_id"].replace(et_events_dict_inv)
    # et_epoch_evts["time"] = et_epoch_evts["sample_nb"] / eeg_sfreq

    # trial_start_ind = et_epoch_evts[et_epoch_evts["event_id"] == "trial_start"].index[0]

    # trial_end_ind = et_epoch_evts[et_epoch_evts["event_id"] == "trial_end"].index[0]

    # et_epoch_evts = et_epoch_evts.iloc[trial_start_ind : trial_end_ind + 1]
    # et_epoch_evts = et_epoch_evts.reset_index(drop=True)

    # first_flash = et_epoch_evts.query("event_id == 'stim-flash_sequence'").index[0]
    # all_stim_pres = et_epoch_evts.query("event_id == 'stim-all_stim'").index[0]
    # last_sacc = (
    #     et_epoch_evts.iloc[all_stim_pres:].query("event_id == 'saccade'").index[0]
    # )

    # et_epoch_evts = et_epoch_evts.iloc[first_flash : last_sacc + 1]
    # et_epoch_evts = et_epoch_evts.reset_index(drop=True)

    # flash_event_ids = ["stim-flash_sequence", "stim-flash_choices", "stim-all_stim"]
    # # flash_events = et_epoch_evts[et_epoch_evts["event_id"].isin(flash_event_ids)]

    # fix_and_sac = et_epoch_evts.query("event_id.isin(['fixation', 'saccade'])").copy()

    # fixation_inds = fix_and_sac.query("event_id == 'fixation'").index
    # ! OLD VERSION

    flash_event_ids = ["stim-flash_sequence", "stim-flash_choices", "stim-all_stim"]

    # fix_and_sac = et_annotations.query("description.isin(['fixation', 'saccade'])")
    fixation_inds = et_annotations.query("description == 'fixation'").index

    fixation_data = {i: [] for i in stim_order}

    for fixation_ind in fixation_inds:
        # ! OLD VERSION
        # stim_flash_ind = (
        #     et_epoch_evts.iloc[:fixation_ind]
        #     .query(f"event_id.isin({flash_event_ids[:-1]})")
        #     .shape[0]
        #     - 1
        # )
        # ! OLD VERSION

        stim_flash_ind = (
            et_annotations.iloc[:fixation_ind]
            .query(f"description.isin({flash_event_ids[:-1]})")
            .shape[0]
            - 1
        )

        target_grid_loc = stim_order[stim_flash_ind]
        target_id, target_coords = stim_pos[target_grid_loc]
        targ_left, targ_right, targ_bottom, targ_top = target_coords

        # fixation = fix_and_sac.loc[fixation_ind]
        fixation = et_annotations.loc[fixation_ind]

        # ! OLD VERSION
        # fix_sample_ind = fixation["sample_nb"]
        # fix_sample_time = fixation["time"]

        # next_sacc = (
        #     et_epoch_evts.iloc[fixation_ind:].query("event_id == 'saccade'").iloc[0]
        # )
        # next_sacc_sample_ind = next_sacc["sample_nb"]
        # next_sacc_sample_time = next_sacc["time"]
        # ! OLD VERSION

        fixation_start = fixation["onset"]
        fixation_duration = fixation["duration"]
        fixation_stop = fixation_start + fixation_duration

        # start_sample_ind_et = int(fixation_start * et_sfreq)
        # stop_sample_ind_et = int(fixation_stop * et_sfreq)

        # gaze_x, gaze_y, pupil_diam = et_epoch.get_data()[
        #     :, start_sample_ind_et:stop_sample_ind_et
        # ]
        gaze_x, gaze_y, pupil_diam = (
            et_epoch.copy().crop(fixation_start, fixation_stop).get_data()
        )

        # ! OLD VERSION
        # gaze_x, gaze_y = manual_et_epochs[epoch_N].get_data()[
        #     :2, fix_sample_ind:next_sacc_sample_ind
        # ]
        #
        # # gaze_x = x_gaze_resampled[fix_sample_ind:next_sacc_sample_ind]
        # # gaze_y = y_gaze_resampled[fix_sample_ind:next_sacc_sample_ind]

        # fixation_duration = gaze_x.shape[0] / eeg_sfreq * 1000
        # ! OLD VERSION

        mean_x, mean_y = gaze_x.mean(), gaze_y.mean()

        on_target = (targ_left <= mean_x <= targ_right) and (
            targ_bottom <= mean_y <= targ_top
        )

        if fixation_duration >= 0.2 and on_target:
            discarded = False
            fixation_data[stim_flash_ind].append(np.array([gaze_x, gaze_y]))
        else:
            discarded = True

        title = f"stim-{stim_flash_ind + 1} ({fixation_duration * 1000:.0f} ms)"
        title += " " + ("DISCARDED" if discarded else "SAVED")

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax_et = fig.add_subplot(gs[0, :])
        ax_eeg = fig.add_subplot(gs[1, :])
        ax_eeg_avg = fig.add_subplot(gs[2, 0], sharex=ax_eeg)

        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_title(title)

        ax_et.imshow(
            icon_images[target_id],
            extent=[targ_left, targ_right, targ_bottom, targ_top],
            origin="lower",
        )

        rectangle = mpatches.Rectangle(
            (targ_left, targ_bottom),
            targ_right - targ_left,
            targ_top - targ_bottom,
            linewidth=1,
            linestyle="--",
            edgecolor="black",
            facecolor="none",
        )
        ax_et.add_patch(rectangle)

        ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
        ax_et.scatter(mean_x, mean_y, c="yellow", s=3)

        # eeg_slice = eeg_data[:, fix_sample_ind:next_sacc_sample_ind] # ! OLD VERSION
        # eeg_slice = eeg_data[:, start_sample_ind_et:stop_sample_ind_et]
        eeg_slice = eeg_data.copy().crop(fixation_start, fixation_stop).get_data()

        ax_eeg.plot(eeg_slice.T)

        # ax_eeg_avg.plot(eeg_slice.mean(axis=0))

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["occipital"]].mean(axis=0),
            color="red",
            label="occipital",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["parietal"]].mean(axis=0),
            color="green",
            label="parietal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["centro-parietal"]].mean(axis=0),
            color="purple",
            label="centro-parietal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["temporal"]].mean(axis=0),
            color="orange",
            label="temporal",
        )

        ax_eeg_avg.plot(
            eeg_slice[ch_group_inds["frontal"]].mean(axis=0),
            color="blue",
            label="frontal",
        )

        ax_eeg_avg.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg.set_xlim(0, eeg_slice.shape[1])

        xticks = np.arange(0, eeg_slice.shape[1], 100)
        ax_eeg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    # set([len(v) for v in fixation_data.values()])
    # fixation_data[11][0]


def rsa_behavior_individual_patterns(behav_data: pd.DataFrame, method="euclidean"):
    cleaned_behav_data = behav_data.copy()

    timeout_trials = cleaned_behav_data.query("choice == 'timeout'")
    cleaned_behav_data.loc[timeout_trials.index, "correct"] = "False"
    cleaned_behav_data.loc[timeout_trials.index, "rt"] = np.nan
    cleaned_behav_data["rt"] = cleaned_behav_data["rt"].astype(float)

    cleaned_behav_data.drop(
        cleaned_behav_data.query("correct == 'invalid'").index, inplace=True
    )

    cleaned_behav_data["correct"] = (
        cleaned_behav_data["correct"]
        .replace({"True": 1, "False": 0, True: 1, False: 0})
        .astype(int)
    )

    participants = [int(i) for i in behav_data["subj_N"].unique()]
    patterns = list(behav_data["pattern"].unique())

    # Min-max normalization
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # cleaned_behav_data = cleaned_behav_data.sort_values(
    #     by=["subj_N", "pattern", "item_id"]
    # )
    # cleaned_behav_data.reset_index(drop=True, inplace=True)

    cleaned_behav_data = (
        cleaned_behav_data.groupby(["subj_N", "pattern"])["correct"]
        .mean()
        .unstack()
        .T.sort_index()
        .to_dict()
        # .to_json("TEMP-cleaned_behav_data.json")
    )

    # cleaned_behav_data.to_json("TEMP-cleaned_behav_data.json")

    behav_features = {}
    for participant in participants:
        behav_features[participant] = []

        for pattern, mean_accuracy in cleaned_behav_data[participant].items():
            behav_features[participant].append(mean_accuracy)

    # * Normalize
    # normalized_behav_features = {p: normalize(mean_acc) for p, mean_acc in behav_features.items()}
    # behav_features = normalized_behav_features

    behav_rdms = {}

    for participant in participants:
        # * Convert to Dataset for RDM calculation
        behav_dataset = Dataset(
            measurements=np.array(behav_features[participant])[:, None],
            obs_descriptors={"pattern": patterns},
        )

        # * Compute RDMs
        behav_rdms[participant] = calc_rdm(behav_dataset, method=method)

    for participant in participants:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        behav_rdm = behav_rdms[participant].get_matrices()[0]

        plot_rdm(
            behav_rdm,
            labels=patterns,
            title=f"Behavior RDM for Participant {participant}",
            axis=ax,
            show_values=True,
        )
        # ax.set_yticklabels([])
        plt.show()
        plt.tight_layout()
        plt.close()

    return behav_rdms
