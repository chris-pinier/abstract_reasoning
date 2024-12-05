def analyze_decision_period_2(epoch_N):
    """
    This function uses a custom method to identify fixation events
    """

    # epoch_N = 0 # ! TEMP

    # * Extract epoch data
    et_epoch = manual_et_epochs[epoch_N]
    eeg_epoch = manual_eeg_epochs[epoch_N]

    def crop_et_epoch(epoch):
        # ! WARNING: We may not be capturing the first fixation if it is already on target
        # epoch = et_epoch.copy()

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

        response_ids = exp_config["lab"]["allowed_keys"] + ["timeout"]
        response = annotations.query(f"description.isin({response_ids})").iloc[0]

        # * Convert to seconds
        start_time = all_stim_pres["onset"]
        end_time = response["onset"]

        # * Crop the data
        # epoch = epoch.copy().crop(first_flash["onset"], last_fixation["onset"])
        epoch = epoch.copy().crop(start_time, end_time)

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        all_stim_idx = annotations.query("description == 'stim-all_stim'").index[0]
        annotations = annotations.iloc[all_stim_idx:]
        annotations.reset_index(drop=True, inplace=True)

        time_bounds = (start_time, end_time)

        return epoch, annotations, time_bounds

    # * Crop the data
    et_epoch, et_annotations, time_bounds = crop_et_epoch(et_epoch)
    eeg_epoch = eeg_epoch.copy().crop(*time_bounds)

    # * Get channel positions for topomap
    info = eeg_epoch.info

    chans_pos_xy = np.array(
        list(info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    # * Get channel indices for each channel group
    ch_group_inds = {
        group_name: [i for i, ch in enumerate(eeg_epoch.ch_names) if ch in group_chans]
        for group_name, group_chans in eeg_chan_groups.items()
    }

    # * Get info on the current trial
    trial_info = get_trial_info(epoch_N)
    stim_pos, stim_order, sequence, choices, response_ind, solution = trial_info
    response = choices.get(response_ind, "timeout")
    solution_ind = {v: k for k, v in choices.items()}[solution]
    correct = response == solution

    seq_and_choices = sequence.copy()
    seq_and_choices.update({k + len(sequence): v for k, v in choices.items()})

    # * Get the indices of the icons, reindex choices to simplify analysis
    sequence_icon_inds = list(sequence.keys())
    choice_icon_inds = [i + len(sequence) for i in choices.keys()]
    solution_ind += len(sequence)
    wrong_choice_icon_inds = [i for i in choice_icon_inds if i != solution_ind]

    def get_fixation_inds():
        fixations = []
        gaze_data = et_epoch.get_data().T
        time_data = et_epoch.times
        last_fix_ind = 0

        for idx_gaze, (x, y, pupil_diam) in enumerate(gaze_data):
            for idx_icon, (icon_name, pos) in enumerate(stim_pos):
                targ_left, targ_right, targ_bottom, targ_top = pos

                on_target = (
                    targ_left <= x <= targ_right and targ_bottom <= y <= targ_top
                )

                if on_target:
                    gaze_time = time_data[idx_gaze]
                    gaze_point_data = [gaze_time, x, y, pupil_diam, idx_icon]

                    if len(fixations) == 0:
                        fixations.append([gaze_point_data])
                        # last_fix_ind = idx_gaze
                    else:
                        last_x = fixations[-1][-1][1]
                        last_y = fixations[-1][-1][2]

                        x_change = abs(x - last_x)
                        y_change = abs(y - last_y)

                        x_threshold = 0.05 * abs(last_x)
                        y_threshold = 0.05 * abs(last_y)

                        x_10_pct = x_change <= x_threshold
                        y_10_pct = y_change <= y_threshold

                        same_gaze = idx_gaze - last_fix_ind <= 5
                        same_icon = fixations[-1][-1][-1] == idx_icon

                        # if x_10_pct and y_10_pct and same_gaze and same_icon:
                        if x_10_pct and y_10_pct and same_icon:
                            fixations[-1].append(gaze_point_data)
                        else:
                            fixations.append([gaze_point_data])

                    last_fix_ind = idx_gaze

        return fixations

    def testing(fixations, show_all=False):
        if show_all:
            fix_inds = range(len(fixations))
        else:
            fix_inds = np.random.choice(range(len(fixations)), 5)

        for fix_ind in fix_inds:
            this_fix = fixations[fix_ind]
            icon_ind = int(this_fix[0, -1])

            icon_name, pos = stim_pos[icon_ind]
            left, right, bottom, top = pos

            fig, ax = plt.subplots()
            ax.set_xlim(0, screen_resolution[0])
            ax.set_ylim(screen_resolution[1], 0)
            ax.imshow(
                icon_images[icon_name],
                extent=[left, right, bottom, top],
                origin="lower",
            )

            # this_fix_inds = this_fix[:, 0]
            # this_fix_x, this_fix_y = gaze_data[this_fix_inds].T
            this_fix_x, this_fix_y = this_fix[:, 1:3].T

            ax.scatter(this_fix_x, this_fix_y, c="red", s=2)
            plt.show()
            plt.close()

    fixations = [np.array(fixation) for fixation in get_fixation_inds()]
    print(len(fixations))

    testing(fixations, show_all=True)

    # * Indices of every gaze fixation event
    # fixation_inds = et_annotations.query("description == 'fixation'").index
    fixation_inds = range(len(fixations))

    # * Initialize data containers
    gaze_target_fixation_sequence = []
    fixation_data = {i: [] for i in range(len(stim_order))}
    eeg_fixation_data = {i: [] for i in range(len(stim_order))}

    for fixation_ind in fixation_inds:
        # * Get number of flash events before the current fixation; -1 to get the index

        # fixation = et_annotations.loc[fixation_ind]

        # fixation_start = fixation["onset"]
        # fixation_duration = fixation["duration"]
        # fixation_stop = fixation_start + fixation_duration
        fixation = fixations[fixation_ind]
        fixation_start = fixation[0, 0]
        fixation_stop = fixation[-1, 0]
        fixation_duration = fixation_stop - fixation_start

        end_time = min(fixation_stop, et_epoch.times[-1])

        # gaze_x, gaze_y, pupil_diam = (
        #     et_epoch.copy().crop(fixation_start, end_time).get_data()
        # )
        gaze_x, gaze_y, pupil_diam = fixation[:, 1:4].T

        mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

        on_target = False

        for i, (icon_name, pos) in enumerate(stim_pos):
            targ_left, targ_right, targ_bottom, targ_top = pos
            if (
                targ_left <= mean_gaze_x <= targ_right
                and targ_bottom <= mean_gaze_y <= targ_top
            ):
                on_target = True
                stim_ind = i
                break

        end_time = min(fixation_stop, eeg_epoch.times[-1])

        eeg_slice = eeg_epoch.copy().crop(fixation_start, end_time)
        # eeg_fixation_data[stim_flash_ind].append(eeg_slice)

        eeg_slice = eeg_slice.copy().pick(["eeg"]).get_data()

        if fixation_duration >= 0.1 and on_target:
            # if on_target:
            discarded = False
            fixation_data[stim_ind].append(np.array([gaze_x, gaze_y]))
            gaze_target_fixation_sequence.append(
                [stim_ind, fixation_start, fixation_duration, pupil_diam.mean()]
            )
        else:
            discarded = True

        title = f"FLASH-{stim_ind} ({fixation_duration * 1000:.0f} ms)"
        title += " - " + ("DISCARDED" if discarded else "SAVED")

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[1, 1])
        ax_et = fig.add_subplot(gs[0, 0])
        ax_topo = fig.add_subplot(gs[0, 1])
        ax_eeg = fig.add_subplot(gs[1, :])
        ax_eeg_group = fig.add_subplot(gs[2, :], sharex=ax_eeg)
        ax_eeg_avg = fig.add_subplot(gs[3, :], sharex=ax_eeg)

        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)
        ax_et.set_title(title)

        # * Plot target icon
        # ax_et.imshow(
        #     icon_images[target_id],
        #     extent=[targ_left, targ_right, targ_bottom, targ_top],
        #     origin="lower",
        # )
        for icon_name, pos in stim_pos:
            targ_left, targ_right, targ_bottom, targ_top = pos
            ax_et.imshow(
                icon_images[icon_name],
                extent=[targ_left, targ_right, targ_bottom, targ_top],
                origin="lower",
            )

            # # * Plot rectangle around target, with dimensions == img_size
            rectangle = mpatches.Rectangle(
                (targ_left, targ_bottom),
                img_size[0],
                img_size[1],
                linewidth=1,
                linestyle="--",
                edgecolor="black",
                facecolor="none",
            )
            ax_et.add_patch(rectangle)

        mne.viz.plot_topomap(
            eeg_slice.mean(axis=1),
            chans_pos_xy,
            ch_type="eeg",
            sensors=True,
            contours=0,
            outlines="head",
            sphere=None,
            image_interp="cubic",
            extrapolate="auto",
            border="mean",
            res=640,
            size=1,
            cmap=None,
            vlim=(None, None),
            cnorm=None,
            axes=ax_topo,
            show=False,
        )

        ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
        ax_et.scatter(mean_gaze_x, mean_gaze_y, c="yellow", s=3)

        ax_eeg.plot(eeg_slice.T)

        ax_eeg_avg.plot(eeg_slice.mean(axis=0))

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["occipital"]].mean(axis=0),
            color="red",
            label="occipital",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["parietal"]].mean(axis=0),
            color="green",
            label="parietal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["centro-parietal"]].mean(axis=0),
            color="purple",
            label="centro-parietal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["temporal"]].mean(axis=0),
            color="orange",
            label="temporal",
        )

        ax_eeg_group.plot(
            eeg_slice[ch_group_inds["frontal"]].mean(axis=0),
            color="blue",
            label="frontal",
        )

        ax_eeg_group.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg.set_xlim(0, eeg_slice.shape[1])
        xticks = np.arange(0, eeg_slice.shape[1], 100)
        ax_eeg.set_xticks(xticks, ((xticks / eeg_sfreq) * 1000).astype(int))

        plt.tight_layout()
        plt.show()

    gaze_target_fixation_sequence = pd.DataFrame(
        gaze_target_fixation_sequence,
        columns=["stim_ind", "onset", "duration", "pupil_diam"],
    )

    gaze_target_fixation_sequence["pupil_diam"] = gaze_target_fixation_sequence[
        "pupil_diam"
    ].round(2)

    mean_diam_per_target = (
        gaze_target_fixation_sequence.groupby("stim_ind")["pupil_diam"].mean().round(2)
    )

    fix_counts_per_target = gaze_target_fixation_sequence["stim_ind"].value_counts()

    total_fix_duration_per_target = gaze_target_fixation_sequence.groupby("stim_ind")[
        "duration"
    ].sum()

    mean_diam_per_target.name = "mean_pupil_diam"
    total_fix_duration_per_target.name = "total_duration"

    mean_diam_per_target.sort_values(ascending=False, inplace=True)
    fix_counts_per_target.sort_values(ascending=False, inplace=True)
    total_fix_duration_per_target.sort_values(ascending=False, inplace=True)

    # gaze_info = pd.merge(
    #     fix_counts_per_target,
    #     total_fix_duration_per_target,
    #     left_index=True,
    #     right_index=True,
    # )

    gaze_info = pd.concat(
        [fix_counts_per_target, total_fix_duration_per_target, mean_diam_per_target],
        axis=1,
    ).reset_index()

    gaze_info["target_name"] = gaze_info["stim_ind"].replace(seq_and_choices)
    gaze_info["trial_N"] = epoch_N

    gaze_info.query("stim_ind in @sequence_icon_inds")
    gaze_info.query("stim_ind == @choice_icon_inds")
    gaze_info.query("stim_ind == @wrong_choice_icon_inds")
    gaze_info.query("stim_ind == @solution_ind")

    return fixation_data, eeg_fixation_data
