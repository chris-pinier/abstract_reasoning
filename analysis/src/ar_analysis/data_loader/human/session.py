from ._common import *
from .base import HumanDataClass


@dataclass
class HumanSessData(HumanDataClass):
    subj_N: int
    sess_N: int

    def __post_init__(self):
        super().__post_init__()
        self.sess_dir = self.search_sess_dir()
        self.behav_data = None
        self.eeg_data = None
        self.et_data = None

        self.base_name = (
            f"sub-{self.subj_N:02}_ses-{self.sess_N:02}_task-{self.task_name}"
        )
        self.fnames = Box(
            {
                "prepro_eeg": {
                    "prepro": self.base_name + "_prepro-eeg-raw.fif",
                    "ica": self.base_name + "_fitted-ica.fif",
                }
            }
        )

    def __str__(self):
        s = f"""
            Class: "{type(self).__name__}"
            Subj_N: {self.subj_N}
            sess_N: {self.sess_N}
            Data Directory: '{self.sess_dir}'
            Data Format: {self.data_fmt}
        """
        return textwrap.dedent(s).strip()

    def show_dir_struct(self, stringRep: bool = False):
        print(f"Subject {self.subj_N:02} Sess {self.sess_N:02} - Directory Structure:")
        struct = DisplayTree(self.sess_dir, stringRep=stringRep)
        print(f"\nPath: {self.sess_dir}")
        if stringRep:
            return struct

    def search_sess_dir(self):
        if self.data_fmt == "bids":
            sess_dir = self.data_dir / f"sub-{self.subj_N:02}/ses-{self.sess_N:02}"
        else:
            sess_dir = self.data_dir / f"subj_{self.subj_N:02}/sess_{self.sess_N:02}"
        if not sess_dir.exists():
            raise FileNotFoundError(f"session directory not found: {sess_dir}")
        return sess_dir

    def _search_res_file(
        self, regex: str, label: str, directory: Path | None = None
    ) -> Path:
        directory = self.sess_dir if directory is None else directory
        results = list_contents(directory, reg=regex, incl="file")
        if len(results) > 1:
            raise ValueError(
                f"Multiple matches found for {label} file: \n\t{'\n\t'.join([str(r) for r in results])}"
            )
        if len(results) == 0:
            raise ValueError(f"{label} file not found in {self.sess_dir}")
        return results[0]

    def search_behav_file(self, regex=r".*beh/.*\.tsv$"):
        return self._search_res_file(regex=regex, label="Behav")

    def search_eeg_file(self, regex=r".*eeg/.*\.bdf$"):
        return self._search_res_file(regex=regex, label="EEG")

    def search_et_files(self, regex: dict | None = None):
        if regex is None:
            regex = dict(
                physio_json=r".*eeg/.*eye.*_physio\.json$",
                physio_tsv=r".*eeg/.*eye.*_physio\.tsv\.gz$",
                events_json=r".*eeg/.*eye.*_physioevents\.json$",
                events_tsv=r".*eeg/.*eye.*_physioevents\.tsv\.gz$",
            )
        # * Search for physio.json (sidecar file)
        physio_json = self._search_res_file(
            regex=regex["physio_json"], label="Eye-Tracking"
        )

        # * Search for physio.tsv
        physio_tsv = self._search_res_file(
            regex=regex["physio_tsv"], label="Eye-Tracking"
        )

        # * Search for physioevents.json
        events_json = self._search_res_file(
            regex=regex["events_json"], label="Eye-Tracking"
        )

        # * Search for physioevents.tsv
        events_tsv = self._search_res_file(
            regex=regex["events_tsv"], label="Eye-Tracking"
        )

        return physio_json, physio_tsv, events_json, events_tsv

    @staticmethod
    def _infer_et_mne_ch_names(
        physio_columns: list[str],
        physio_meta: dict[str, Any],
        physio_json_path: Path,
    ) -> list[str]:
        recorded_eye = str(physio_meta.get("RecordedEye", "")).strip().lower()
        if recorded_eye not in {"left", "right"}:
            if "recording-eye1" in physio_json_path.name:
                recorded_eye = "left"
            elif "recording-eye2" in physio_json_path.name:
                recorded_eye = "right"

        if recorded_eye in {"left", "right"}:
            rename_map = {
                "x_coordinate": f"xpos_{recorded_eye}",
                "y_coordinate": f"ypos_{recorded_eye}",
                "pupil_size": f"pupil_{recorded_eye}",
            }
            return [rename_map.get(col, col) for col in physio_columns]

        return physio_columns

    @staticmethod
    def _event_desc_from_mne_events(events: np.ndarray) -> dict[int, str]:
        observed_values = sorted({int(value) for value in events[:, 2]})
        return {
            value: c.VALID_EVENTS_INV.get(value, f"trigger_{value}")
            for value in observed_values
        }

    # * ########################################
    # * Get Raw Data
    # * ########################################
    def get_raw_behav_data(self) -> pd.DataFrame:
        supported_fmts = [".csv", ".tsv"]
        behav_fpath = self.search_behav_file()
        fext = behav_fpath.suffix
        if fext == ".csv":
            return pd.read_csv(behav_fpath, index_col=0)
        elif fext == ".tsv":
            return pd.read_csv(behav_fpath, sep="\t")
        else:
            raise ValueError(
                f"Behav file extension ({fext}) not supported. Supported formats: {supported_fmts}"
            )

    def get_raw_eog_data(
        self,
        raw_eeg: mne.io.Raw | None = None,
        verbose: bool = False,
        preload: bool = False,
        bad_chans: List[str] | None = None,
    ) -> mne.io.Raw:
        """Return the raw EOG channels stored in the EEG/BDF recording."""
        if raw_eeg is None:
            raw_eeg = self.get_raw_eeg_data(
                verbose=verbose,
                preload=preload,
                bad_chans=bad_chans,
            )

        missing_eog_chans = [ch for ch in c.EOG_CHANS if ch not in raw_eeg.ch_names]
        if missing_eog_chans:
            raise ValueError(
                f"Missing expected EOG channels in raw EEG data: {missing_eog_chans}"
            )

        raw_eog = raw_eeg.copy().pick(c.EOG_CHANS)
        raw_eog.set_channel_types({ch: "eog" for ch in raw_eog.ch_names})
        return raw_eog

    def get_raw_eeg_data(
        self,
        verbose: bool = False,
        preload: bool = False,
        bad_chans: List[str] | None = None,
    ) -> mne.io.Raw:
        eeg_fpath = self.search_eeg_file()
        raw_eeg = mne.io.read_raw_bdf(eeg_fpath, preload=preload, verbose=verbose)

        if bad_chans is not None:
            raw_eeg.info["bads"] = bad_chans

        set_eeg_montage(
            raw_eeg=raw_eeg,
            montage=self.get_eeg_montage(),
            eog_chans=c.EOG_CHANS,
            non_eeg_chans=c.NON_EEG_CHANS,
            verbose=verbose,
        )

        return raw_eeg

    def _get_raw_et_data_asc(
        self, verbose: str = "WARNING"
    ) -> mne.io.eyelink.eyelink.RawEyelink:
        # -> Tuple[mne.io.eyelink.eyelink.RawEyelink, list]:

        et_fpath = self.find_file(self.sess_dir, "*.asc", "Eye Tracking")
        raw_et = mne.io.read_raw_eyelink(et_fpath, verbose=verbose)
        # et_cals = self.get_et_calibration()

        return raw_et  # , et_cals

    def _get_raw_et_data_bids(self):
        physio_json, physio_tsv, events_json, events_tsv = self.search_et_files()

        # * Load the JSON sidecar to extract the metadata
        with open(physio_json, "r") as f:
            physio_meta = json.load(f)

        # * BIDS specs dictate that physio JSONs contain these keys
        sfreq = physio_meta["SamplingFrequency"]
        ch_names = self._infer_et_mne_ch_names(
            physio_meta["Columns"], physio_meta, physio_json
        )

        # * Load the continuous eye-tracking data
        # *  BIDS physio TSV files do NOT have a header row, so we apply the columns from the JSON
        df = pd.read_csv(physio_tsv, sep="\t", header=None, names=ch_names)

        # * MNE expects data to be shaped as (n_channels, n_samples)
        data = df.to_numpy().T

        # * Create the MNE Info object
        # * MNE requires specific channel types for its internal eye-tracking tools.
        # * We will guess the type based on standard BIDS naming conventions.
        ch_types = []
        for ch in ch_names:
            ch_lower = ch.lower()
            if any(x in ch_lower for x in ["x", "y", "gaze", "pos"]):
                ch_types.append("eyegaze")
            elif any(x in ch_lower for x in ["pupil", "size"]):
                ch_types.append("pupil")
            else:
                ch_types.append("misc")  # Catch-all for trigger channels

        misc_ch_inds = [i for i, ch in enumerate(ch_types) if ch == "misc"]
        misc_ch_names = [ch_names[i] for i in misc_ch_inds]
        print(f"Removing miscellaneous channels: {misc_ch_names}")

        ch_types = [ch for i, ch in enumerate(ch_types) if ch != "misc"]
        ch_names = [ch for i, ch in enumerate(ch_names) if i not in misc_ch_inds]
        data = np.delete(data, misc_ch_inds, axis=0)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # * Create the MNE Raw object
        raw = mne.io.RawArray(data, info)

        # * Load and attach the eye-tracking events
        with open(events_json, "r") as f:
            events_meta = json.load(f)
        ch_names = events_meta["Columns"]

        # * Unlike continuous physio, BIDS event files should have a header row.
        # * Older local conversions may be headerless, so fall back to sidecar columns.
        events_df = pd.read_csv(events_tsv, sep="\t")
        if not set(ch_names).issubset(events_df.columns):
            events_df = pd.read_csv(events_tsv, sep="\t", header=None, names=ch_names)

        for col in ("onset", "duration"):
            if col not in events_df.columns:
                raise ValueError(f"Missing required eye-tracking event column: {col}")

        events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
        events_df["duration"] = pd.to_numeric(
            events_df["duration"], errors="coerce"
        ).fillna(0)
        events_df.dropna(subset=["onset"], inplace=True)
        events_df.reset_index(drop=True, inplace=True)

        if events_df.empty:
            raw.set_annotations(mne.Annotations([], [], []))
            raw.filenames = [physio_tsv]
            return raw

        events_df["onset"] = (events_df["onset"] - events_df["onset"].iloc[0]) / 1000
        events_df["duration"] /= 1000

        for col in ("trial_type", "message"):
            if col not in events_df.columns:
                events_df[col] = pd.NA

        eye_events = events_df[events_df["trial_type"].notna()]
        exp_events = events_df[events_df["message"].notna()]

        eye_annotations = mne.Annotations(
            onset=eye_events["onset"].values,
            duration=eye_events["duration"].values,
            description=eye_events["trial_type"].astype(str).values,
        )

        exp_annotations = mne.Annotations(
            onset=exp_events["onset"].values,
            duration=exp_events["duration"].values,
            description=exp_events["message"].astype(str).values,
        )

        annotations = eye_annotations + exp_annotations

        unique_events = list(raw.annotations.to_data_frame()["description"].unique())
        print(f"Unique events found: {'\n'.join(unique_events)}")

        raw.set_annotations(annotations)

        raw.filenames = [physio_tsv]

        return raw

    def get_et_calibration(self):
        if self.data_fmt == "bids":
            raise NotImplementedError(
                ".asc file containing calibration data not ported to the BIDS converted dataset"
            )
        else:
            et_fpath = self._search_res_file(
                regex=r".+\.asc", label="Eye Tracking Calibration"
            )

            # * read_eyelink_calibration is too verbose by default, silencing it
            with contextlib.redirect_stdout(io.StringIO()):
                et_cals = read_eyelink_calibration(et_fpath)
                if not et_cals:
                    print(
                        f"WARNING: No calibration found for subj {self.subj_N}, sess {self.sess_N}"
                    )

    def get_raw_et_data(self):
        fmt = self.data_fmt
        if fmt == "bids":
            return self._get_raw_et_data_bids()
        elif fmt == "original":
            return self._get_raw_et_data_asc()
        else:
            raise ValueError(
                "Problem encountered when trying to load eye-tracking data. ",
                f"`self.data_fmt` must be set to {DATA_FMTS}",
            )

    def get_sess_info(self, fmt: DATA_FMTS | None = None) -> Dict:
        fmt = self.data_fmt if fmt is None else fmt
        if fmt == "bids":
            subj_dir = self.data_dir / f"sub-{self.subj_N:02}"
            fpath = self._search_res_file(
                directory=subj_dir,
                regex=rf".*sub-{self.subj_N:02}_sessions\.tsv$",
                label="Sess Info",
            )
            sess_info = pd.read_csv(fpath, sep="\t", keep_default_na=False)

            sess_info = (
                sess_info.query(f"session_id=='ses-{self.sess_N:02}'").iloc[0].to_dict()
            )

        elif fmt == "original":
            fpath = self._search_res_file(regex=r".*sess_info.json$", label="Sess Info")
            sess_info = pd.read_json(fpath, orient="index").iloc[:, 0].to_dict()

        # TODO: Not optimal, this should have been set during sess_info file creation. If fix, need to modify experiment's code as well
        sess_info = {str(k).lower(): v for k, v in sess_info.items()}
        return sess_info

    def is_sess_bad(self):
        subj_N, sess_N = self.subj_N, self.sess_N
        if f"sess_{sess_N}" in c.BAD_SESSIONS.get(f"subj_{subj_N}", []):
            print(
                f"WARNING: session {subj_N:02}{sess_N:02} labelled as bad/corrupted "
                "in configuration file. Skipping it"
            )
            return True
        return False

    def get_raw_data(
        self, bad_eeg_chans: List[str] | None = None, ignore_bad_sess: bool = True
    ) -> (
        Tuple[
            Dict[str, Any],
            pd.DataFrame,
            mne.io.Raw,
            mne.io.Raw,
            list,
        ]
        | Tuple[None, None, None, None, None]
    ):
        subj_N, sess_N = self.subj_N, self.sess_N

        if self.is_sess_bad() & ignore_bad_sess:
            return (None, None, None, None, None)

        # * Load data
        raw_behav = self.get_raw_behav_data()
        raw_eeg = self.get_raw_eeg_data(bad_chans=bad_eeg_chans)
        raw_et = self.get_raw_et_data()

        sess_info = self.get_sess_info()

        if len(sess_info["notes"]) > 0:
            print(f"notes for subj {subj_N}, sess {sess_N}: {sess_info['notes']}")

            # * Log all notes / problems in a single file
            # logger.warning(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")
            # with open(NOTES_FILE, "r") as f:
            #     notes = json.load(f)

            # notes.update({f"subj_{subj_N:02}-sess_{sess_N:02}": sess_info["Notes"]})

            # with open(NOTES_FILE, "w") as f:
            #     json.dump(notes, f)

        return sess_info, raw_behav, raw_eeg, raw_et  # , et_cals

    def generate_trial_video(
        self,
        manual_et_epochs: List[mne.Epochs],
        manual_eeg_epochs: List[mne.Epochs],
        raw_behav: pd.DataFrame,
        epoch_N: int,
        all_bad_chans: Optional[Dict] = None,
        eeg_chan_groups: Optional[Dict] = None,
        non_eeg_chans: Optional[List[str]] = None,
        et_sfreq: Optional[int | float] = None,
        valid_events_inv: Optional[Dict] = None,
        eeg_sfreq: Optional[int | float] = None,
        screen_resolution: Optional[Tuple[int, int]] = None,
        icon_images: Optional[Dict[str, np.ndarray]] = None,
        save_dir: Path | None = None,
        gaze_pt_size: int = 4,
        gaze_ln_size: int | None = None,
    ):
        gaze_ln_size = int(gaze_pt_size / 2) if gaze_ln_size is None else gaze_ln_size

        all_bad_chans = c.ALL_BAD_CHANS if all_bad_chans is None else all_bad_chans
        eeg_chan_groups = (
            c.EEG_CHAN_GROUPS if eeg_chan_groups is None else eeg_chan_groups
        )
        non_eeg_chans = c.NON_EEG_CHANS if non_eeg_chans is None else non_eeg_chans
        et_sfreq = c.ET_SFREQ if et_sfreq is None else et_sfreq
        valid_events_inv = (
            c.VALID_EVENTS_INV if valid_events_inv is None else valid_events_inv
        )
        eeg_sfreq = c.EEG_SFREQ if eeg_sfreq is None else eeg_sfreq
        screen_resolution = (
            c.SCREEN_RESOLUTION if screen_resolution is None else screen_resolution
        )
        icon_images = c.ICON_IMAGES if icon_images is None else icon_images

        # * Extract epoch data
        et_epoch = manual_et_epochs[epoch_N]
        eeg_epoch = manual_eeg_epochs[epoch_N]

        subj_N, sess_N = self.subj_N, self.sess_N

        tracked_eye = et_epoch.ch_names[0].split("_")[1]
        assert et_sfreq == et_epoch.info["sfreq"], (
            "Eye-tracking data has incorrect sampling rate"
        )
        assert eeg_sfreq == eeg_epoch.info["sfreq"], (
            "EEG data has incorrect sampling rate"
        )

        sess_bad_chans = all_bad_chans.get(f"subj_{subj_N}", {}).get(
            f"sess_{sess_N}", []
        )
        montage = eeg_epoch.get_montage()

        selected_chans = [
            i
            for i, ch in enumerate(montage.ch_names)
            if ch not in non_eeg_chans + sess_bad_chans
        ]
        selected_chans_names = [montage.ch_names[i] for i in selected_chans]

        # * Select EEG channel groups to plot
        selected_chan_groups = {
            k: v
            for k, v in eeg_chan_groups.items()
            if k in ["frontal", "parietal", "central", "temporal", "occipital"]
        }

        group_colors = dict(
            zip(
                selected_chan_groups.keys(),
                ["red", "green", "blue", "purple", "orange"],
            )
        )

        # * Get channel indices for each channel group
        ch_group_inds = {
            group_name: [
                i for i, ch in enumerate(selected_chans_names) if ch in group_chans
            ]
            for group_name, group_chans in selected_chan_groups.items()
        }

        # * Get channel positions for topomap
        eeg_info = mne.pick_info(
            eeg_epoch.info,
            [
                i
                for i, ch in enumerate(eeg_epoch.info.ch_names)
                if ch not in non_eeg_chans + eeg_epoch.info["bads"]
            ],
        )
        chans_pos_xy = np.array(
            list(eeg_info.get_montage().get_positions()["ch_pos"].values())
        )[:, :2]

        trial_info = get_trial_info(
            epoch_N,
            raw_behav,
            c.X_POS_STIM,
            c.Y_POS_CHOICES,
            c.Y_POS_SEQUENCE,
            c.SCREEN_RESOLUTION,
            c.IMG_SIZE,
        )
        stim_pos, stim_order = trial_info[:2]

        # * Resample eye-tracking data for the current trial
        x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
            et_epoch, tracked_eye, et_sfreq, eeg_sfreq
        )

        eeg_data = eeg_epoch.get_data(picks=selected_chans_names)
        epoch_evts = pd.Series(eeg_epoch.get_data(picks=[c.STIM_CHAN])[0])

        # * Find indices where consecutive events are different
        diff_indices = np.where(epoch_evts.diff() != 0)[0]
        epoch_evts = epoch_evts[diff_indices]
        epoch_evts = epoch_evts[epoch_evts != 0]
        epoch_evts = epoch_evts.replace(valid_events_inv)

        min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
        eeg_data = eeg_data[:, :min_length]
        eeg_data *= 1e6  # Convert to microvolts

        x_gaze_resampled = x_gaze_resampled[:min_length]
        y_gaze_resampled = y_gaze_resampled[:min_length]

        # * y-axis limits for the EEG plots
        eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
        y_eeg_min, y_eeg_max = eeg_min * 1.1, eeg_max * 1.1

        # * Heatmap of gaze data
        all_stim_onset = epoch_evts[epoch_evts == "stim-all_stim"].index[0]
        trial_end = epoch_evts[epoch_evts == "trial_end"].index[0]
        get_gaze_heatmap(
            x_gaze_resampled[all_stim_onset:trial_end],
            y_gaze_resampled[all_stim_onset:trial_end],
            screen_resolution,
            bin_size=20,
            show=True,
        )

        samples_per_1ms = eeg_sfreq / 1000
        samples_per_100ms = round(samples_per_1ms * 100)

        leftover = eeg_data.shape[1] % samples_per_100ms
        inds = np.arange(0, eeg_data.shape[1], samples_per_100ms)
        if leftover > 0:
            inds = np.append(inds, inds[-1] + leftover)

        step_size = samples_per_100ms
        steps = np.diff(inds)
        inds = np.array(list(zip(inds[:-1], inds[1:])))
        zfill_len = len(str(steps.shape[0]))

        # * Set up the directory to save the frames
        dir_name = f"eeg_frames-{subj_N}-{sess_N:02d}-ep{epoch_N:02d}"
        if save_dir is None:
            eeg_frames_dir = self.export_dir / dir_name
        else:
            eeg_frames_dir = save_dir / dir_name

        if eeg_frames_dir.exists():
            shutil.rmtree(eeg_frames_dir)

        eeg_frames_dir.mkdir(parents=True, exist_ok=True)

        # * Adjust figure size and subplot layout
        fig = plt.figure(figsize=(25.6, 14.4))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 3], width_ratios=[3, 2])
        ax_eeg1 = fig.add_subplot(gs[0, :])
        ax_eeg2 = fig.add_subplot(gs[1, :], sharex=ax_eeg1)
        ax_topo = fig.add_subplot(gs[2, 0])
        ax_et = fig.add_subplot(gs[2, 1])

        win_len_samples = steps.cumsum()[20]
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, eeg_data.shape[0]))

        x_ticks = np.arange(0, win_len_samples + 1, step_size)
        x_ticks_labels = [str(int(i / eeg_sfreq * 1000)) for i in x_ticks]

        ax_eeg1_line = ax_eeg1.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)
        ax_eeg2_line = ax_eeg2.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)

        et_scatter = None
        et_line = None
        last_eeg_x = 0
        last_plot_x = 0
        flash_N = 0

        ax_et_fix_cross = ax_et.scatter(
            screen_resolution[0] / 2,
            screen_resolution[1] / 2,
            s=80,
            marker="+",
            linewidths=1,
            color="black",
        )

        ax_et_plotted_icons = []
        for icon_name, icon_pos in stim_pos:
            left, right, bottom, top = icon_pos
            this_icon = ax_et.imshow(
                icon_images[icon_name],
                extent=[left, right, bottom, top],
                origin="lower",
            )
            ax_et_plotted_icons.append(this_icon)
            this_icon.set_visible(False)

        topo_plot_params = dict(
            ch_type="eeg",
            sensors=True,
            names=None,
            mask=None,
            mask_params=None,
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

        mne.viz.plot_topomap(
            data=np.zeros_like(eeg_data[:, 0]), pos=chans_pos_xy, **topo_plot_params
        )
        ax_topo.set_axis_off()

        eeg_group_data = {
            group: eeg_data[ch_group_inds[group]].mean(axis=0)
            for group in ch_group_inds
        }
        min_eeg_by_group = [group_data.min() for group_data in eeg_group_data.values()]
        max_eeg_by_group = [group_data.max() for group_data in eeg_group_data.values()]

        def reset_eeg_plot():
            ax_eeg1.clear()
            ax_eeg1.set_xticks(x_ticks)
            ax_eeg1.set_xticklabels([])
            ax_eeg1.set_ylim(y_eeg_min, y_eeg_max)
            ax_eeg1.set_xlim(0, win_len_samples)

            ax_eeg2.clear()
            ax_eeg2.set_xticks(x_ticks, x_ticks_labels)
            ax_eeg2.set_ylim(min(min_eeg_by_group), max(max_eeg_by_group))
            ax_eeg2.set_xlim(0, win_len_samples)
            ax_eeg2.hlines(0, 0, win_len_samples, color="black", linestyle="--")

        def reset_et_plot(show_icons: List[int] | bool = False, show_fix_cross=False):
            ax_et.set_xlim(0, screen_resolution[0])
            ax_et.set_ylim(screen_resolution[1], 0)

            ax_et.set_xticklabels([])
            ax_et.set_yticklabels([])
            ax_et.set_xticks([])
            ax_et.set_yticks([])
            ax_et.set_aspect("equal", adjustable="box")

            if isinstance(show_icons, bool):
                [
                    ax_et_plotted_icons[i].set_visible(show_icons)
                    for i in range(len(stim_order))
                ]
            else:
                [
                    ax_et_plotted_icons[i].set_visible(False)
                    for i in range(len(stim_order))
                ]
                [ax_et_plotted_icons[i].set_visible(True) for i in show_icons]

            ax_et_fix_cross.set_visible(show_fix_cross)

        dpi = 150
        reset_eeg_plot()
        reset_et_plot(show_icons=False, show_fix_cross=True)

        print(f"Saving frames in {eeg_frames_dir}")

        for idx_step, step in enumerate(tqdm(steps, desc="Generating frames")):
            ax_eeg1_line.remove()
            ax_eeg2_line.remove()

            bounds = (last_eeg_x, last_eeg_x + step)
            detected_event_inds = [i for i in epoch_evts.index if i in range(*bounds)]

            if detected_event_inds:
                if ax_et_fix_cross.get_visible():
                    ax_et_fix_cross.set_visible(False)

                detected_events = epoch_evts[detected_event_inds]
                event_desc = detected_events.values
                event_inds = detected_events.index - bounds[0] + last_plot_x

                ax_eeg1.vlines(
                    event_inds,
                    ymin=y_eeg_min,
                    ymax=y_eeg_max,
                    color="red",
                    linestyle="--",
                )
                ax_eeg2.vlines(
                    event_inds,
                    ymin=y_eeg_min,
                    ymax=y_eeg_max,
                    color="red",
                    linestyle="--",
                )

                for ind, desc in zip(event_inds, event_desc):
                    ax_eeg1.text(
                        ind,
                        y_eeg_max,
                        desc,
                        rotation=45,
                        verticalalignment="top",
                        horizontalalignment="right",
                        fontsize=8,
                        color="red",
                    )

                    if "stim-flash" in desc:
                        icon_ind = stim_order[flash_N]
                        reset_et_plot(show_icons=[icon_ind], show_fix_cross=False)
                        flash_N += 1
                    elif desc == "stim-all_stim":
                        reset_et_plot(show_icons=True, show_fix_cross=False)
                    elif desc in ["a", "x", "m", "l", "timeout", "trial_end"]:
                        reset_et_plot(show_icons=False, show_fix_cross=True)

            eeg_slice = eeg_data[:, bounds[0] : bounds[1]]
            x_gaze_slice = x_gaze_resampled[bounds[0] : bounds[1]]
            y_gaze_slice = y_gaze_resampled[bounds[0] : bounds[1]]

            if et_scatter:
                et_scatter.remove()
                [el.remove() for el in et_line]

            cmap = plt.get_cmap("Reds")
            norm = plt.Normalize(0, x_gaze_slice.shape[0])
            et_colors = cmap(
                norm(
                    np.linspace(0, x_gaze_slice.shape[0], x_gaze_slice.shape[0]) * 0.5
                    + 10
                )
            )

            et_scatter = ax_et.scatter(
                x_gaze_slice, y_gaze_slice, c=et_colors, s=gaze_pt_size, alpha=0.5
            )
            et_line = ax_et.plot(
                x_gaze_slice,
                y_gaze_slice,
                c="r",
                ls="-",
                linewidth=gaze_ln_size,
                alpha=0.3,
            )

            if last_plot_x == win_len_samples:
                last_plot_x = 0
                reset_eeg_plot()

            x = np.arange(last_plot_x, last_plot_x + step)
            last_eeg_x += step
            last_plot_x += step

            for i in range(eeg_slice.shape[0]):
                ax_eeg1.plot(x, eeg_slice[i], color=colors[i])

            for group_name, group_data in eeg_group_data.items():
                ax_eeg2.plot(
                    x,
                    group_data[bounds[0] : bounds[1]],
                    label=group_name,
                    color=group_colors[group_name],
                )
            if idx_step == 0:
                ax_eeg2_legend = ax_eeg2.get_legend_handles_labels()

            ax_eeg1_line = ax_eeg1.vlines(
                x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
            )
            ax_eeg2_line = ax_eeg2.vlines(
                x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
            )

            ax_topo.clear()
            mne.viz.plot_topomap(
                data=eeg_slice.mean(axis=1), pos=chans_pos_xy, **topo_plot_params
            )

            ax_eeg2.legend().remove()
            ax_eeg2.legend(
                *ax_eeg2_legend,
                bbox_to_anchor=(1.005, 1),
                loc="upper left",
                borderaxespad=0,
            )

            plt.tight_layout()
            plt.savefig(
                eeg_frames_dir / f"frame_{str(idx_step + 1).zfill(zfill_len)}.png",
                dpi=dpi,
            )

        reset_eeg_plot()
        reset_et_plot(show_icons=False, show_fix_cross=True)
        ax_eeg2.legend(
            *ax_eeg2_legend,
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()
        plt.savefig(eeg_frames_dir / f"frame_{str(0).zfill(zfill_len)}.png", dpi=dpi)
        plt.close(fig)

        fps = 3
        output_file = f"eeg_video-subj{subj_N:02d}-sess{sess_N:02d}-ep{epoch_N:02d}.mp4"
        if shutil.which("ffmpeg") is None:
            print(
                "WARNING: ffmpeg not found in PATH. "
                f"Saved frames only in: {eeg_frames_dir}"
            )
            return eeg_frames_dir

        create_video_from_frames(eeg_frames_dir, output_file, fps, zfill_len)

        return eeg_frames_dir / output_file

    # * ########################################
    # * Preprocess Data
    # * ########################################
    def _search_prepro_eeg_file(
        self,
        directory: Path | str,
        regex: str | None = None,
        raise_error: bool = False,
    ) -> Path | None:
        directory = Path(directory)

        if regex is None:
            regex = rf"sub.*{self.subj_N:02}.*ses.*{self.sess_N:02}.*-raw\.fif"

        fpaths = list_contents(directory, reg=regex, incl="file", recurs=False)

        # 1. Handle the "no match" case
        if not fpaths:
            if raise_error:
                raise FileNotFoundError(
                    "No preprocessed EEG file found for "
                    f"subj {self.subj_N:02} - sess {self.sess_N:02} in '{directory}'"
                )
            return None

        # 2. Handle the "multiple matches" case
        if len(fpaths) > 1:
            raise ValueError(
                "Multiple matches found for preprocessed EEG file of "
                f"subj {self.subj_N:02} - sess {self.sess_N:02} in '{directory}'"
            )

        # 3. Handle the successful case
        return fpaths[0]

    def _search_eeg_ica_file(
        self,
        directory: Path | str,
        regex: str | None = None,
        raise_error: bool = False,
    ) -> Path | None:
        directory = Path(directory)

        if regex is None:
            regex = rf"sub.*{self.subj_N:02}.*ses.*{self.sess_N:02}.*_fitted-ica\.fif"

        fpaths = list_contents(directory, reg=regex, incl="file", recurs=False)

        # 1. Handle the "no match" case
        if not fpaths:
            if raise_error:
                raise FileNotFoundError(
                    "No ICA file found for "
                    f"subj {self.subj_N:02} - sess {self.sess_N:02} in '{directory}'"
                )
            return None

        # 2. Handle the "multiple matches" case
        if len(fpaths) > 1:
            raise ValueError(
                "Multiple matches found for preprocessed EEG file of "
                f"subj {self.subj_N:02} - sess {self.sess_N:02} in '{directory}'"
            )

        # 3. Handle the successful case
        return fpaths[0]

    def _search_prepro_et_file(self):
        raise NotImplementedError

    def preprocess_behav(self) -> pd.DataFrame:
        behav_data = self.get_raw_behav_data()

        # sess_info = self.get_sess_info()

        # sess_date = pendulum.from_format(
        #     sess_info["date"], "YYYYMMDD_HHmmss", tz="Europe/Amsterdam"
        # )
        sess_N = self.sess_N
        behav_data.rename(columns={"subj_id": "subj_N"}, inplace=True)
        behav_data.insert(1, "sess_N", sess_N)
        behav_data.insert(2, "trial_N", list(range(len(behav_data))))
        behav_data.insert(3, "block_N", behav_data["blockN"])
        # behav_data.insert(behav_data.shape[1], "sess_date", sess_date)

        cols_to_drop = [
            "blockN",
            "trial_type",
            "trial_onset_time",
            "series_end_time",
            "choice_onset_time",
            "rt_global",
        ]

        behav_data.drop(columns=cols_to_drop, inplace=True)

        # * Identify timeout trials and mark them as incorrect
        behav_data["correct"] = behav_data["correct"].astype(object)
        timeout_trials = behav_data["choice_key"].eq("timeout")
        behav_data["rt"] = pd.to_numeric(behav_data["rt"], errors="coerce")
        behav_data.loc[timeout_trials, "correct"] = False
        behav_data.loc[timeout_trials, "rt"] = np.nan
        # behav_data.loc[timeout_trials.index, ['choice_key', "choice"]] = "invalid"

        behav_data["correct"] = (
            behav_data["correct"]
            .replace({"invalid": False, "True": True, "False": False})
            .astype(bool)
        )
        # behav_data["correct"] = behav_data["correct"].astype(bool)

        assert (
            behav_data["correct"].mean()
            == behav_data.query("choice==solution").shape[0] / behav_data.shape[0]
        ), "Error with cleaning of 'Correct' column"

        # * ----------------------------------------
        sequences_file = CONFIG_DIR / f"sequences/session_{sess_N}.csv"

        sequences = pd.read_csv(
            sequences_file, dtype={"choice_order": str, "seq_order": str}
        ).drop(columns=["pattern", "solution"])

        behav_data = behav_data.merge(sequences, on="item_id")

        return behav_data

    def preprocess_et_data(
        self,
        raw_et: mne.io.eyelink.eyelink.RawEyelink,
    ) -> mne.io.eyelink.eyelink.RawEyelink:
        return raw_et  # , et_cals

    def preprocess_eog_data(
        self,
        raw_eog: mne.io.Raw,
        l_freq: float | None = None,
        h_freq: float | None = None,
    ) -> mne.io.Raw:
        """Apply lightweight EOG preprocessing.

        By default this only returns a loaded copy with EOG channel types
        preserved. Optional filtering is available for analyses that need it.
        """
        prepro_eog = raw_eog.copy()
        prepro_eog.load_data(verbose="WARNING")
        prepro_eog.set_channel_types({ch: "eog" for ch in prepro_eog.ch_names})
        if l_freq is not None or h_freq is not None:
            prepro_eog.filter(l_freq=l_freq, h_freq=h_freq, verbose="WARNING")
        return prepro_eog

    def identify_bad_eeg_channels(self):
        raise NotImplementedError

    def preprocess_eeg_data(
        self,
        raw_eeg: mne.io.Raw,
        eeg_chan_groups: Dict[str, str],
        # raw_behav: pd.DataFrame,
        preprocessed_dir: Path | None = None,
        force: Optional[bool] = False,
        reuse_ica: Optional[bool] = True,
        # # TODO: bad_chs_method="interpolate",
    ) -> mne.io.Raw:
        # ! TEMP
        # eeg_chan_groups = c.EEG_CHAN_GROUPS
        # subj_N = 1
        # sess_N = 4
        # bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
        # sess_info, raw_behav, raw_eeg, raw_et, et_cals = load_raw_data(subj_N, sess_N, c.DATA_DIR, c.EEG_MONTAGE, bad_chans)
        # ! TEMP

        # fpath = Path(str(raw_eeg.filenames[0]))
        # subj_N = int(fpath.parents[1].name.split("_")[1])
        # sess_dir = fpath.parents[0]
        # sess_N = int(sess_dir.name.split("_")[1])

        subj_N, sess_N = self.subj_N, self.sess_N

        # * Create preprocessed_dir and ica_dir if they don't exist

        preprocessed_dir = (
            self.preprocessed_dir if preprocessed_dir is None else preprocessed_dir
        )
        ica_dir = preprocessed_dir / "ica"
        ica_dir.mkdir(exist_ok=True, parents=True)

        preprocessed_raw_fpath = self._search_prepro_eeg_file(preprocessed_dir)

        if preprocessed_raw_fpath is None or force is True:
            print(
                "Preprocessed file not found.\nPreprocessing raw data for:",
                f"'{str(raw_eeg.filenames[0])}'",
            )
            preprocessed_raw_fpath = preprocessed_dir / self.fnames.prepro_eeg.prepro
            # raw_eeg.load_data(verbose="WARNING")
            prepro_eeg = raw_eeg.copy()
            prepro_eeg.load_data(verbose="WARNING")
            del raw_eeg

            # # * Detecting events
            # eeg_events = mne.find_events(
            #     prepro_eeg,
            #     min_duration=0,
            #     initial_event=False,
            #     shortest_event=1,
            #     uint_cast=True,
            #     verbose="WARNING",
            # )

            # # # ! TEMP
            # # df = pd.DataFrame(eeg_events, columns=["sample_nb", "prev", "event_id"])
            # # df["event_id"] = df["event_id"].replace(VALID_EVENTS_INV)
            # # ! TEMP

            # # * Get annotations from events and add them to the raw data
            # annotations = mne.annotations_from_events(
            #     eeg_events,
            #     prepro_eeg.info["sfreq"],
            #     event_desc=c.VALID_EVENTS_INV,
            #     verbose="WARNING",
            # )

            # prepro_eeg.set_annotations(annotations, verbose="WARNING")
            # try:
            #     self.annotate_eeg_from_events(prepro_eeg)
            # except Exception as e:
            #     print(f"WARNING: eeg events not found. Error details: {e}")

            bad_chans = prepro_eeg.info["bads"]

            # TODO: automatically remove bad channels (e.g., amplitude cutoff)
            manually_set_bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(
                f"sess_{sess_N}"
            )

            if not bad_chans == manually_set_bad_chans:
                print(
                    "WARNING: raw EEG bad channels do not match expected bad channels, combining them"
                )

            bad_chans = list(set(bad_chans) | set(manually_set_bad_chans))

            prepro_eeg.info["bads"] = bad_chans

            # prepro_eeg.drop_channels(bad_chans)

            # * Check if channel groups include all channels present in the montage
            # * i.e., that there are no "orphan" channels
            check_ch_groups(prepro_eeg.get_montage(), eeg_chan_groups)

            # * Average Reference
            prepro_eeg = prepro_eeg.set_eeg_reference(
                ref_channels="average", verbose="WARNING"
            )

            # * Filter to remove power line noise
            prepro_eeg.notch_filter(freqs=np.arange(50, 251, 50), verbose="WARNING")

            # * ########################################
            # * EOG artifact rejection using ICA
            # * ########################################
            ica_fpath = self._search_eeg_ica_file(ica_dir)

            if ica_fpath is not None and reuse_ica is True:
                # * if ICA file exists, load it
                ica = mne.preprocessing.read_ica(ica_fpath)

            else:
                ica_fpath = ica_dir / self.fnames.prepro_eeg.ica
                eeg_filtered_for_ica = prepro_eeg.copy()

                # * Bandpass Filter: 1-100 Hz
                eeg_filtered_for_ica.filter(l_freq=1, h_freq=100, verbose="WARNING")

                # * Create a copy of the raw data to hihg-pass filter at 1Hz before ICA
                # * as recommended by MNE: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
                # prepro_eeg_copy_for_ica = prepro_eeg.copy()
                # prepro_eeg.filter(l_freq=1, h_freq=100, verbose="WARNING")

                ica = mne.preprocessing.ICA(
                    n_components=None,
                    noise_cov=None,
                    random_state=c.RAND_SEED,
                    # method="fastica",
                    method="infomax",
                    fit_params=dict(extended=True),
                    max_iter="auto",
                    verbose="WARNING",
                )

                ica.fit(eeg_filtered_for_ica, verbose="WARNING")

                ica.save(ica_fpath, verbose="WARNING")

            # eog_inds, eog_scores = ica.find_bads_eog(prepro_eeg)
            # ica.exclude = eog_inds

            # # * Label components using IClabel
            ic_labels = label_components(prepro_eeg, ica, method="iclabel")

            # df_labels = pd.DataFrame(
            #     list(zip(list(ic_labels["y_pred_proba"]), ic_labels["labels"])), columns=['prob', 'label']
            # ).sort_values(by='prob', ascending=False)

            # # * Get indices of components labeled as 'brain'
            brain_ic_indices = [
                idx for idx, label in enumerate(ic_labels["labels"]) if label == "brain"
            ]

            # # * Keep only brain components, effectively rejecting artifactual ones
            ica.exclude = [
                idx for idx in range(ica.n_components_) if idx not in brain_ic_indices
            ]

            # * Apply ICA to raw data
            prepro_eeg = ica.apply(prepro_eeg, verbose="WARNING")

            # * Bandpass Filter: 0.1-100 Hz
            prepro_eeg.filter(l_freq=0.1, h_freq=100, verbose="WARNING")

            # * Interpolate bad channels and remove them from "bads" list in eeg info
            prepro_eeg = prepro_eeg.interpolate_bads(reset_bads=True)

            # * Set average reference again, including interpolated bad channels
            prepro_eeg = prepro_eeg.set_eeg_reference(
                ref_channels="average", verbose="WARNING"
            )

            # * Add bad channels to the "bads" list in eeg info again
            prepro_eeg.info["bads"] = bad_chans

            # * Save preprocessed raw data
            prepro_eeg.save(preprocessed_raw_fpath, overwrite=True, verbose="WARNING")

        else:
            prepro_eeg = mne.io.read_raw_fif(
                preprocessed_raw_fpath, preload=False, verbose="WARNING"
            )

        # split_eeg_data_into_trials(prepro_eeg, raw_behav)

        return prepro_eeg

    def get_eeg_metadata(self) -> Dict[str, Any]:
        eeg_data = self.get_eeg_data()
        fpath = Path(eeg_data.filenames[0])
        fstem = fpath.stem
        fdir = fpath.parent

        try:
            events_counts = (
                eeg_data.annotations.to_data_frame()["description"]
                .value_counts()
                .to_dict()
            )
        except Exception as exc:
            logger.exception("Could not extract EEG annotation event counts: {}", exc)
            events_counts = {}
        try:
            ch_positions = eeg_data._get_channel_positions().tolist()
        except Exception as exc:
            logger.exception("Could not extract EEG channel positions: {}", exc)
            ch_positions = []
        try:
            dig_montage = ([str(i) for i in eeg_data.info["dig"]],)
        except Exception as exc:
            logger.exception("Could not extract EEG digitized montage: {}", exc)
            dig_montage = []

        metadata = dict(
            ch_types=eeg_data.get_channel_types(),
            ch_names=eeg_data.ch_names,
            ch_positions=ch_positions,
            ch_number=eeg_data.info.get("nchan"),
            sfreq=eeg_data.info.get("sfreq"),
            filter_highpass=eeg_data.info.get("highpass"),
            filter_lowpass=eeg_data.info.get("lowpass"),
            duration=str(timedelta(seconds=eeg_data.duration)),
            events_counts=events_counts,
        )
        metadata = {k: metadata[k] for k in sorted(metadata.keys())}

        # metadata_fp = fdir / f"{fstem}-metadata-eeg.json"
        # with open(metadata_fp, "w+") as f:
        #     json.dump(metadata, fp=f, indent=4)
        return metadata

    def get_et_metadata(self):  # -> Dict[str, Any]:
        # if self.data_fmt == 'bids':
        #     metadata = None
        # else:
        #     metadata = None
        # return metadata
        raise NotImplementedError

    def get_metadata(self, fmt: DATA_FMTS | None = None):
        sess_info = self.get_sess_info(fmt=fmt)
        eeg_metadata = self.get_eeg_metadata()
        # et_metadata = self.get_et_metadata()
        et_metadata = {}

        metadata = dict(sess_info=sess_info, eeg=eeg_metadata, et=et_metadata)
        metadata = Box(metadata)

        return metadata

    @staticmethod
    def annotate_eeg_from_events(data: mne.io.Raw, verbose: str = "WARNING"):
        # * Detecting events
        try:
            events = mne.find_events(
                data,
                min_duration=0,
                initial_event=False,
                shortest_event=1,
                uint_cast=True,
                verbose=verbose,
            )
        except ValueError as exc:
            if "Could not find any of the events" not in str(exc):
                raise
            data.set_annotations(mne.Annotations([], [], []), verbose=verbose)
            return

        if len(events) == 0:
            data.set_annotations(mne.Annotations([], [], []), verbose=verbose)
            return

        # * Get annotations from events and add them to the raw data
        annotations = mne.annotations_from_events(
            events,
            data.info["sfreq"],
            event_desc=HumanSessData._event_desc_from_mne_events(events),
            verbose=verbose,
        )

        data.set_annotations(annotations, verbose=verbose)

    # * ########################################
    # * Get Preprocessed Data
    # * ########################################
    def get_behav_data(self) -> pd.DataFrame:
        return self.preprocess_behav()

    def get_eog_data(
        self,
        raw_eeg: mne.io.Raw | None = None,
        l_freq: float | None = None,
        h_freq: float | None = None,
    ) -> mne.io.Raw:
        raw_eog = self.get_raw_eog_data(raw_eeg=raw_eeg)
        return self.preprocess_eog_data(raw_eog, l_freq=l_freq, h_freq=h_freq)

    def get_eeg_data(self) -> mne.io.Raw:
        raw_eeg = self.get_raw_eeg_data()

        return self.preprocess_eeg_data(
            raw_eeg=raw_eeg,
            eeg_chan_groups=c.EEG_CHAN_GROUPS,
            # raw_behav: pd.DataFrame,
            preprocessed_dir=None,
            force=False,
            reuse_ica=True,
        )

    def get_et_data(self) -> Tuple[mne.io.eyelink.eyelink.RawEyelink, list]:
        raw_et = self.get_raw_et_data()
        prepro_et_data = self.preprocess_et_data(raw_et)
        return prepro_et_data  # , et_cals

    def get_data(
        self, ignore_bad_sess: bool = True
    ) -> (
        Tuple[
            Dict[str, Any],
            pd.DataFrame,
            mne.io.Raw,
            mne.io.Raw,
            list,
        ]
        # | Tuple[None, None, None, None, None]
        | Tuple[None, None, None, None]
    ):
        subj_N, sess_N = self.subj_N, self.sess_N

        # if self.is_sess_bad() & ignore_bad_sess:
        #     return [], [], [], [], []
        if self.is_sess_bad() & ignore_bad_sess:
            return (None, None, None, None)

        # * Load data
        behav_data = self.get_behav_data()
        eeg_data = self.get_eeg_data()
        # et_data, et_cals = self.get_et_data()
        et_data = self.get_et_data()
        sess_info = self.get_sess_info()

        if len(sess_info["notes"]) > 0:
            print(f"notes for subj {subj_N}, sess {sess_N}: {sess_info['notes']}")

            # * Log all notes / problems in a single file
            # logger.warning(f"Notes for subj {subj_N}, sess {sess_N}: {sess_info['Notes']}")
            # with open(NOTES_FILE, "r") as f:
            #     notes = json.load(f)

            # notes.update({f"subj_{subj_N:02}-sess_{sess_N:02}": sess_info["Notes"]})

            # with open(NOTES_FILE, "w") as f:
            #     json.dump(notes, f)

        # return sess_info, behav_data, eeg_data, et_data, et_cals
        return sess_info, behav_data, eeg_data, et_data

    def check_data(self, remove_practice: bool = True) -> None:
        # TODO: optimize this code -> epoching of multiple events could be more efficient
        # * check for aborted trials, aborted experiments, mismatch in trials between files, etc.
        # sess_info, behav_data, eeg_data, et_data, et_cals = self.get_data(
        sess_info, behav_data, eeg_data, et_data = self.get_data(ignore_bad_sess=False)

        if eeg_data.info["bads"] != []:
            print(
                "WARNING: Found EEG channels marked as bad in "
                f"subject {self.subj_N} - session {self.sess_N}.\n"
                "\t These should be interpolated during preprocessing and removed from the 'bads' list. \n"
                "\t Reassigning them as good channels."
            )
            eeg_data.info["bads"] = []

        tmin, tmax, baseline = -0.1, 0.1, None

        for event_id in ["trial_start", "trial_end"]:
            eeg_epochs = mne.Epochs(
                eeg_data,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                verbose="WARNING",
            )

            et_epochs = mne.Epochs(
                et_data,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                verbose="WARNING",
            )

            if self.sess_N == 1 and remove_practice:
                if len(eeg_epochs.events) > 80:
                    eeg_epochs = eeg_epochs[3:]
                if len(et_epochs.events) > 80:
                    et_epochs = et_epochs[3:]

            n_et_epochs = len(et_epochs.events)
            n_eeg_epochs = len(eeg_epochs.events)
            n_behav_trials = len(behav_data)

            if len({n_eeg_epochs, n_et_epochs, n_behav_trials}) != 1:
                print(
                    "WARNING: Mismatch in number of trials between data files:"
                    f"\n - {n_eeg_epochs = }\n - {n_et_epochs = }\n - {n_behav_trials = }"
                )

        choice_events = ["a", "x", "m", "l", "timeout"]
        eeg_filtered_events = [
            a for a in eeg_data.annotations.description if a in choice_events
        ]
        eeg_filtered_events = list(set(eeg_filtered_events))

        eeg_epochs = mne.Epochs(
            eeg_data,
            event_id=eeg_filtered_events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            verbose="WARNING",
        )

        et_epochs = mne.Epochs(
            et_data,
            event_id=eeg_filtered_events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            verbose="WARNING",
        )

        eeg_annots = eeg_epochs.annotations
        et_annots = et_epochs.annotations

        if self.sess_N == 1 and remove_practice:
            if len(eeg_epochs.events) > 80:
                eeg_epochs = eeg_epochs[3:]
                eeg_annots = eeg_annots[3:]
            if len(et_epochs.events) > 80:
                et_epochs = et_epochs[3:]
                et_annots = et_annots[3:]

        q = f"description.isin({choice_events})"
        eeg_choice_keys = eeg_annots.to_data_frame().query(q)["description"].to_list()
        et_choice_keys = et_annots.to_data_frame().query(q)["description"].to_list()
        behav_choice_keys = behav_data["choice_key"].to_list()

        if not eeg_choice_keys == et_choice_keys == behav_choice_keys:
            print("WARNING: Mismatch in choice events between data files.")

    # * ########################################
    # * Eye Tracking Data processing
    # * ########################################
    def split_et_data_into_trials(self, raw_et, pb_on: bool = False):
        # TODO: Implement -> save preprocessed file

        # * Read events from annotations
        et_events, et_events_dict = mne.events_from_annotations(
            raw_et, verbose="WARNING"
        )

        # * Convert keys to strings (if they aren't already)
        et_events_dict = {str(k): v for k, v in et_events_dict.items()}

        if et_events_dict.get("exp_start"):
            et_events_dict["experiment_start"] = et_events_dict.pop("exp_start")

        # * Create a mapping from old event IDs to new event IDs
        # * that is, adding key-value pairs for events exracted from the eye tracker
        # * i.e., fixation, saccade, blink, etc.

        id_mapping = {}
        eye_events_idx = 60

        for event_name, event_id in et_events_dict.items():
            if event_name in c.VALID_EVENTS:
                new_id = c.VALID_EVENTS[event_name]
            else:
                eye_events_idx += 1
                new_id = eye_events_idx
            id_mapping[event_id] = new_id

        # # * Update event IDs in et_events
        for i in range(et_events.shape[0]):
            old_id = et_events[i, 2]
            if old_id in id_mapping:
                et_events[i, 2] = id_mapping[old_id]

        # * Update et_events_dict with new IDs
        et_events_dict = {k: id_mapping[v] for k, v in et_events_dict.items()}
        et_events_dict = {
            k: v for k, v in sorted(et_events_dict.items(), key=lambda x: x[1])
        }
        et_events_dict_inv = {v: k for k, v in et_events_dict.items()}

        inds_responses = np.where(
            np.isin(et_events[:, 2], [10, 11, 12, 13, 14, 15, 16])
        )
        choice_key_et = [c.VALID_EVENTS_INV[i] for i in et_events[inds_responses, 2][0]]

        et_events_df = pd.DataFrame(
            et_events, columns=["sample_nb", "prev", "event_id"]
        )
        et_events_df["event_id"] = et_events_df["event_id"].replace(et_events_dict_inv)

        # print("Eye tracking event counts:")
        # display(et_events_df["event_id"].value_counts())

        et_trial_bounds, et_trial_events_df = locate_trials(et_events, et_events_dict)

        # * Remove practice trials
        if self.sess_N == 1:
            choice_key_et = choice_key_et[3:]
            et_trial_bounds = et_trial_bounds[3:]
            et_trial_events_df = et_trial_events_df.query("trial_id >= 3").copy()
            et_trial_events_df["trial_id"] -= 3

        manual_et_epochs = []

        if pb_on is True:
            et_trial_bounds = tqdm(et_trial_bounds, "Creating EEG epochs")

        # * Loop through each trial
        for start, end in et_trial_bounds:
            # * Get start and end times in seconds
            start_time = (et_events[start, 0] / raw_et.info["sfreq"]) - c.PRE_TRIAL_TIME
            end_time = et_events[end, 0] / raw_et.info["sfreq"] + c.POST_TRIAL_TIME

            # * Crop the raw data to this time window
            epoch_data = raw_et.copy().crop(tmin=start_time, tmax=end_time)

            # * Add this epoch to our list
            manual_et_epochs.append(epoch_data)

        # * Print some information about our epochs
        # print(f"Number of epochs created: {len(manual_et_epochs)}")
        # for i, epoch in enumerate(manual_et_epochs):
        #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")

        # assert (
        #     len(manual_et_epochs) == 80
        # ), "Incorrect number of epochs created, should be 80"

        manual_et_epochs = (et_trial for et_trial in manual_et_epochs)

        return (
            manual_et_epochs,
            et_events_dict,
            et_events_dict_inv,
            et_trial_bounds,
            et_trial_events_df,
        )

    def get_et_epochs(self):
        raise NotImplementedError

    @staticmethod
    def is_fixation_on_target(
        gaze_x: np.ndarray, gaze_y: np.ndarray, targets_pos: List
    ) -> Tuple[bool, int | None]:
        """Check if the fixation is on target, and return the target index

        Args:
            gaze_x (np.ndarray): #TODO: _description_
            gaze_y (np.ndarray): #TODO: _description_
            targets_pos (list[str, list[float]]): #TODO: _description_

        Returns:
            tuple(bool, [int|None]): #TODO: _description_
        """

        # * Determine if fixation is on target
        mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

        on_target = False

        for target_ind, (target_name, target_pos) in enumerate(targets_pos):
            targ_left, targ_right, targ_bottom, targ_top = target_pos

            if (
                targ_left <= mean_gaze_x <= targ_right
                and targ_bottom <= mean_gaze_y <= targ_top
            ):
                on_target = True
                return (on_target, target_ind)

        return (on_target, None)

    @staticmethod
    def crop_et_trial(epoch: mne.Epochs):
        # ! WARNING: We may not be capturing the first fixation if it is already on target
        # epoch = et_trial.copy()

        # * Get annotations, convert to DataFrame, and adjust onset times
        annotations = epoch.annotations.to_data_frame(time_format="ms")
        annotations["onset"] -= annotations["onset"].iloc[0]
        annotations["onset"] /= 1000

        # first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

        response_ids = c.EXP_CONFIG.lab.allowed_keys + ["timeout", "invalid"]
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

        time_bounds = [start_time, end_time]

        return epoch, annotations, time_bounds

    @staticmethod
    def analyze_gaze_path_similarity():
        raise NotImplementedError
        # import multimatch_gaze as m

        # # read in data
        # fix_vector1 = np.recfromcsv(
        #     "data/fixvectors/segment_0_sub-01.tsv",
        #     delimiter="\t",
        #     dtype={"names": ("start_x", "start_y", "duration"), "formats": ("f8", "f8", "f8")},
        # )
        # fix_vector2 = np.recfromcsv(
        #     "data/fixvectors/segment_0_sub-19.tsv",
        #     delimiter="\t",
        #     dtype={"names": ("start_x", "start_y", "duration"), "formats": ("f8", "f8", "f8")},
        # )

        # # Optional - if the input data are produced by REMoDNaV
        # # pursuits = True is the equivalent of --pursuits 'keep', else specify False
        # fix_vector1 = m.remodnav_reader(
        #     "data/remodnav_samples/sub-01_task-movie_run-1_events.tsv",
        #     screensize=[1280, 720],
        #     pursuits=True,
        # )

        # # execution with multimatch-gaze's docomparison() function without grouping
        # m.docomparison(fix_vector1, fix_vector2, screensize=[1280, 720])

        # # execution with multimatch-gaze's docomparison() function with grouping
        # m.docomparison(
        #     fix_vector1,
        #     fix_vector2,
        #     screensize=[1280, 720],
        #     grouping=True,
        #     TDir=30.0,
        #     TDur=0.1,
        #     TAmp=100.1,
        # )

    @staticmethod
    def get_gaze_heatmaps_from_fixation_data(data_dir: Path, save_dir: Path):
        raise NotImplementedError

    # * ########################################
    # * EEG Data processing
    # * ########################################
    def get_erp(
        self,
        event_ids: List[str],
        tmin: float,
        tmax: float,
        raw: bool = False,
        baseline=None,
        detrend=None,
        method: Literal["mean", "median"] = "mean",
        verbose: str = "WARNING",
    ):
        if raw:
            data = self.get_raw_eeg_data()
            self.annotate_eeg_from_events(data, verbose=verbose)

        else:
            data = self.get_eeg_data()

        epochs = mne.Epochs(
            data,
            event_id=event_ids,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            detrend=detrend,
        )
        erp = epochs.average(picks=None, method=method, by_event_type=False)
        return erp

    def get_eeg_epochs(
        self,
        erp_events: list[str],
        erp_tmin: float,
        erp_tmax: float,
        # selected_chans: List[str] | None = None,
        baseline=None,
        # format: Literal["mne", "np"] = "mne",
        # trials_per_sess: int = 80,
        # epochs_name: str = "epochs",
        # preprocessed_dir: Path | None = None,
        # save_dir: Path | None = None,
        # combine: bool = False,
        # remove_practice: bool = True,
    ) -> Tuple[Any, pd.core.frame.DataFrame, List[int]]:
        # TODO: Remove practice trials if looking at session 1

        prepro_eeg = self.get_eeg_data()

        if prepro_eeg.info["bads"] != []:
            print(
                "WARNING: Found EEG channels marked as bad in "
                f"subject {self.subj_N} - session {self.sess_N}.\n"
                "\t These should be interpolated during preprocessing and removed"
                "\t from the 'bads' list. \n"
                "\t Reassigning them as good channels."
            )
            prepro_eeg.info["bads"] = []

        eeg_annotations = prepro_eeg.annotations.to_data_frame()["description"]

        eeg_filtered_events = [a for a in eeg_annotations if a in erp_events]
        eeg_filtered_events = list(set(eeg_filtered_events))

        eps = mne.Epochs(
            prepro_eeg,
            event_id=eeg_filtered_events,
            tmin=erp_tmin,
            tmax=erp_tmax,
            baseline=baseline,
            verbose="WARNING",
            # preload=True
        )

        # if remove_practice & self.sess_N == 1 & (len(eps.events) > trials_per_sess):
        #     # * Remove practice trials
        #     print("Removing practice trials from EEG data")
        #     eps = eps[3:]

        #     if n_eps != (n_trials := behav_df.query(f"sess_N == {sess_N}").shape[0]):
        #         print(
        #             f"WARNING: Different number of trials between "
        #             f"EEG ({n_eps}) and behavioral ({n_trials}) "
        #             f"files in session {sess_N}. Dropping session."
        #         )
        #         bad_sessions.append(sess_N)
        #     else:
        #         sess_epochs[sess_N] = eps

        return eps

    @staticmethod
    def split_eeg_data_into_trials(
        raw_eeg: mne.io.Raw,
        raw_behav: pd.DataFrame,
        remove_practice: bool = True,
        practice_ind: int = 3,
        n_trials: int = 80,
        incomplete: str = "error",
        pb_on=False,
    ):
        """_summary_ #TODO

        Args:
            raw_eeg (mne.io.Raw): _description_
            raw_behav (pd.DataFrame): _description_
            remove_practice (bool, optional): _description_. Defaults to True.
            practice_ind (int, optional): _description_. Defaults to 3.
            n_trials (int, optional): _description_. Defaults to 80.
            incomplete (str, optional): _description_. Defaults to "error".

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if incomplete not in ["allow", "error", "skip"]:
            raise ValueError(
                "incomplete must be either of the following: 'allow', 'error', 'skip'"
            )

        eeg_events, _ = mne.events_from_annotations(
            raw_eeg, c.VALID_EVENTS, verbose="WARNING"
        )

        if eeg_events.size == 0:
            warning_msg = "Error with EEG events: no valid EEG events found."
            if incomplete == "error":
                raise ValueError(warning_msg)
            elif incomplete == "skip":
                print(warning_msg)
                return [None] * 4
            elif incomplete == "allow":
                print(warning_msg)
                raw_behav["choice_key_eeg"] = pd.NA
                raw_behav["same"] = False
                eeg_trial_bounds = np.empty((0, 2), dtype=int)
                eeg_events_df = pd.DataFrame(
                    columns=["sample_nb", "event_id", "trial_id"]
                )
                return iter(()), eeg_trial_bounds, eeg_events, eeg_events_df

        choice_key_eeg = [
            c.VALID_EVENTS_INV[i]
            for i in eeg_events[:, 2]
            if i in [10, 11, 12, 13, 14, 15, 16]
        ]

        eeg_trial_bounds, eeg_events_df = locate_trials(eeg_events, c.VALID_EVENTS)

        if remove_practice is True:
            if len(choice_key_eeg) > n_trials and len(eeg_trial_bounds) > n_trials:
                choice_key_eeg = choice_key_eeg[practice_ind:]
                eeg_trial_bounds = eeg_trial_bounds[practice_ind:]

        if not len(choice_key_eeg) == len(eeg_trial_bounds) == n_trials:
            warning_msg = (
                "Error with EEG events: incorrect number of trials.\n"
                f"{len(choice_key_eeg) = }\n{len(eeg_trial_bounds) = }"
            )
            if incomplete == "error":
                raise ValueError(warning_msg)
            elif incomplete == "skip":
                print(warning_msg)
                # * Return a list of None matching the size of the expect output
                return [None] * 4
            elif incomplete == "allow":
                print(warning_msg)
                pass

        raw_behav["choice_key_eeg"] = choice_key_eeg
        raw_behav["same"] = raw_behav["choice_key"] == raw_behav["choice_key_eeg"]

        manual_eeg_trials = []

        if pb_on is True:
            eeg_trial_bounds = tqdm(eeg_trial_bounds, "Creating EEG epochs")

        # * Loop through each trial
        for start, end in eeg_trial_bounds:
            # * Get start and end times in seconds
            start_time = (
                eeg_events[start, 0] / raw_eeg.info["sfreq"]
            ) - c.PRE_TRIAL_TIME
            end_time = eeg_events[end, 0] / raw_eeg.info["sfreq"] + c.POST_TRIAL_TIME

            # * Crop the raw data to this time window
            epoch_data = raw_eeg.copy().crop(tmin=start_time, tmax=end_time)

            # * Add this epoch to our list
            manual_eeg_trials.append(epoch_data)

        # * Print some information about our epochs
        # print(f"Number of epochs created: {len(manual_eeg_trials)}")
        # for i, epoch in enumerate(manual_eeg_trials):
        #     print(f"\tEpoch {i+1} duration: {epoch.times[-1] - epoch.times[0]:.2f} seconds")
        manual_eeg_trials = (trial for trial in manual_eeg_trials)

        return manual_eeg_trials, eeg_trial_bounds, eeg_events, eeg_events_df

    def get_trials_data(
        self,
        preprocessed_dir: Path | None = None,
        raise_error: bool = False,
        eeg_incomplete: Literal["allow", "error", "skip"] = "error",
    ):
        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        if not preprocessed_dir.exists():
            raise FileNotFoundError("Preprocessed data directory not found")

        subj_N, sess_N = self.subj_N, self.sess_N

        bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])

        # * Load the data
        # sess_info, behav, eeg, et, et_cals = self.get_data()
        data = self.get_data()

        if any([v is None for v in data]):
            # if any([v is None for v in (sess_info, behav, eeg, et, et_cals)]):
            if raise_error:
                raise ValueError("Bad session")
            else:
                print("Bad session, skipping...")
                return [None] * (len(data) - 1)  # Removing sess_info

        sess_info, behav, eeg, et = data
        del sess_info

        # if notes := sess_info["Notes"]:
        #     print(f"SESSION NOTES:\n{notes}")

        if not c.ET_SFREQ == et.info["sfreq"]:
            raise ValueError("Eye-tracking data has incorrect sampling rate")

        if not c.EEG_SFREQ == eeg.info["sfreq"]:
            raise ValueError("EEG data has incorrect sampling rate")

        (
            manual_et_trials,
            *_,
            # et_events_dict,
            # et_events_dict_inv,
            # et_trial_bounds,
            # et_trial_events_df,
        ) = self.split_et_data_into_trials(et)

        (
            manual_eeg_trials,
            *_,
            # eeg_trial_bounds,
            # eeg_events,
            # eeg_events_df,
        ) = self.split_eeg_data_into_trials(eeg, behav, incomplete=eeg_incomplete)

        if manual_eeg_trials is None:
            msg = (
                "Skipping session trials due to incomplete EEG event parsing "
                f"(subj={subj_N:02}, sess={sess_N:02}, eeg_incomplete={eeg_incomplete!r})."
            )
            if raise_error:
                raise ValueError(msg)
            logger.warning(msg)
            return [None] * 3

        return behav, manual_et_trials, manual_eeg_trials

    @staticmethod
    def analyze_phase_coupling(
        eeg_data: np.ndarray,
        sfreq: int | float,
        f_pha,
        f_amp,
        idpac=(1, 2, 3),
        dcomplex="wavelet",
    ):
        """
        Analyze phase-amplitude coupling using Tensorpac.
        see: https://etiennecmb.github.io/tensorpac/auto_examples/erpac/plot_erpac.html#sphx-glr-auto-examples-erpac-plot-erpac-py
        Parameters:
        -----------
        eeg_data : np.ndarray
        f_pha : #TODO: write the description
        f_amp : #TODO: write the description

        Returns:
        --------
        pac : ndarray
            Phase-amplitude coupling results.
        p_obj : tensorpac.Pac
            PAC object with results.
        """

        # * alpha (8–13 Hz), beta (13–30 Hz), delta (0.5–4 Hz), and theta (4–7 Hz)
        # * Extract data and sampling frequency
        data = eeg_data.mean(axis=0)

        # * Suppress printed output
        with contextlib.redirect_stdout(io.StringIO()):
            p_obj = tensorpac.Pac(
                idpac=idpac, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex
            )

            # * Extract phase and amplitude
            # * filter func :
            # *     - expect x as array of data of shape (n_epochs, n_times)
            # *     - returns the filtered data of shape (n_freqs, n_epochs, n_times)
            pha_p = p_obj.filter(sf=sfreq, x=data, ftype="phase")
            amp_p = p_obj.filter(sfreq, data, ftype="amplitude")

            # * Compute PAC
            pac = p_obj.fit(pha_p, amp_p)

        return pac, p_obj

    def get_stim_flash_order(self, target_stim: str | None = None) -> pd.DataFrame:
        behav = self.get_behav_data()

        behav["all_stim_flash_seq"] = behav["seq_order"].apply(
            lambda x: [int(i) for i in x]
        ) + behav["choice_order"].apply(lambda x: [int(i) + 8 for i in x])

        stim_cols = [f"figure{i}" for i in range(1, 9)] + [
            f"choice{i}" for i in range(1, 5)
        ]

        unique_stims = set(behav[stim_cols].values.flatten())

        if target_stim is not None:
            if target_stim not in unique_stims:
                raise ValueError(f"`target_stim` must be one of {sorted(unique_stims)}")
            selected_stims = [target_stim]
        else:
            selected_stims = unique_stims

        all_stim_locs = {}

        for stim in selected_stims:
            res = behav[(behav[stim_cols] == stim).any(axis=1)].copy()
            # trial_inds = behav[(behav[stim_cols] == stim).any(axis=1)].index
            stim_locs = []
            stim_flash_order = []

            for i in res.index:
                row = res.loc[i]
                stims = row[stim_cols].values
                flash_order = row["all_stim_flash_seq"]
                _stim_locs = np.where(stims == stim)[0].tolist()
                _stim_flash_order = sorted([flash_order.index(i) for i in _stim_locs])

                stim_locs.append(_stim_locs)
                stim_flash_order.append(_stim_flash_order)

            res["stim_locs"] = stim_locs
            res["stim_flash_order"] = stim_flash_order
            res = res.merge(behav[["subj_N", "sess_N", "trial_N"]], how="outer")
            res["stim"] = stim

            all_stim_locs[stim] = res[
                [
                    "subj_N",
                    "sess_N",
                    "trial_N",
                    "all_stim_flash_seq",
                    "stim_locs",
                    "stim_flash_order",
                    "stim",
                ]
            ]

        # all_stim_locs = pd.concat(all_stim_locs.values()).reset_index(
        #     names="overall_trial_N", drop=False
        # )

        all_stim_locs = pd.concat(all_stim_locs.values()).reset_index(drop=True)

        return all_stim_locs

    def get_stim_flash_eeg_epochs(self, target_stim: str | None = None):
        subj_N, sess_N = self.subj_N, self.sess_N

        stim_locs = self.get_stim_flash_order()
        # stim_locs = stim_locs[target_stim] if target_stim is not None else stim_locs
        stim_locs = self.get_stim_flash_order(target_stim=target_stim)

        # stim_locs = {
        #     stim: df.query(f"subj_N == {subj_N} & sess_N=={sess_N}")
        #     for stim, df in stim_locs.items()
        # }

        prepro_eeg = self.get_eeg_data()

        eeg_annotations = prepro_eeg.annotations.to_data_frame()["description"]

        erp_events = ["stim-flash_sequence", "stim-flash_choices"]
        eeg_filtered_events = [a for a in eeg_annotations if a in erp_events]
        eeg_filtered_events = list(set(eeg_filtered_events))
        erp_tmin = 0
        erp_tmax = 0.6

        eeg_epochs = mne.Epochs(
            prepro_eeg,
            event_id=eeg_filtered_events,
            tmin=erp_tmin,
            tmax=erp_tmax,
            baseline=None,
            verbose=False,
        )

        flash_per_trial = 12

        if sess_N == 1:
            eeg_epochs = eeg_epochs[3 * flash_per_trial :]

        # assert len(eeg_epochs.events) / flash_per_trial == trial_start_inds.shape[0]

        trial_start_inds = np.arange(0, len(eeg_epochs.events) + 1, flash_per_trial)
        trial_bound_inds = np.array(
            list(zip(trial_start_inds[:-1], trial_start_inds[1:]))
        )

        stim_epochs_sess = {}
        for stim in stim_locs["stim"].unique():
            _stim_locs = (
                stim_locs.dropna()
                .query(f"stim == '{stim}'")[["trial_N", "stim_flash_order"]]
                .values
            )
            _ep = []
            for trial_ind, stim_flash_order in _stim_locs:
                try:
                    _ep.append(
                        eeg_epochs[slice(*trial_bound_inds[trial_ind])][
                            stim_flash_order
                        ]
                    )
                # * If there's a mismatch in number of expected trials and trials
                # * (i.e., missing data), ignore
                except IndexError:
                    print("WARNING: mismatch in number of expected trials and trials")

            stim_epochs_sess[stim] = _ep  # mne.concatenate_epochs(_ep)

        # TODO: Temporary, to be fixed in preprocessed files
        for stim, eps in stim_epochs_sess.items():
            for ep in eps:
                ep.info["bads"] = []
                ep.event_id = {"stim-flash_sequence": 11, "stim-flash_choices": 10}

        # stim_epochs = {
        #     s: mne.concatenate_epochs(ep) for s, ep in stim_epochs_sess.items()
        # }
        # TODO: For some unknown reason, MNE raises an error when trying to concat epoch objects here
        for stim, eps in stim_epochs_sess.items():
            dir(eps[1].info)
            eps[1].info.ch_names
            info = eps[0].info
            arr = np.concatenate([e.get_data() for e in eps])
            stim_epochs_sess[stim] = mne.EpochsArray(arr, info, verbose=False)
            # stim_epochs_sess[stim] = mne.concatenate_epochs(ep, verbose=False)

        # # ! validity test
        # a = {stim: len(e.events) for stim, e in stim_epochs_sess.items()}
        # a = {k: a[k] for k in sorted(a)}
        # stim_cols = [f"figure{i}" for i in range(1, 9)]
        # stim_cols += [f"choice{i}" for i in range(1, 5)]
        # uniq_counts = np.unique_counts(behav[stim_cols].values)

        # b = dict(zip(uniq_counts[0], uniq_counts[1]))
        # assert a == b
        # # ! validity test

        # stim_erps = {stim: ep.average() for stim, ep in stim_epochs_sess.items()}

        # for stim, erp in stim_erps.items():
        #     erp.plot(titles=stim)

        return stim_epochs_sess

    # * ########################################
    # * Represetational Similarity Analysis
    # * ########################################
    def get_erp_rdms(
        self,
        dissimilarity_metric: str,
        chan_group: str,
        erp_events: list[str],
        erp_tmin: float,
        erp_tmax: float,
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
    ):
        # # # ! TEMP
        # dissimilarity_metric = "correlation"
        # similarity_metric = "corr"
        # chan_group = "frontal"
        # erp_tmin = -1.0
        # erp_tmax = 0.0
        # erp_events = ["a", "x", "m", "l", "invalid", "timeout"]
        # save_dir = WD /'test-export/ERP-RDMs'
        # # ! TEMP
        subj_N, sess_N = self.subj_N, self.sess_N

        if save_dir is None:
            save_dir = self.export_dir / f"analyzed/{subj_N:02}{sess_N:02}"
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir

        if not preprocessed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed data directory not found, check path: {preprocessed_dir}"
            )

        selected_chans = c.EEG_CHAN_GROUPS.get(chan_group, "not found")
        if selected_chans == "not found":
            raise ValueError(
                f"chan_group '{chan_group}' not specified in c.EEG_CHAN_GROUPS"
            )

        prepro_eeg_file = (
            preprocessed_dir / f"subj_{subj_N:02}{sess_N:02}_preprocessed-raw.fif"
        )
        if not prepro_eeg_file.exists():
            raise FileNotFoundError(
                f"Preprocessed EEG file not found for subj {subj_N:02}, sess {sess_N:02}"
            )

        raw_behav = self.get_behav_data()

        prepro_eeg = mne.io.read_raw_fif(
            prepro_eeg_file, preload=False, verbose="WARNING"
        )

        eeg_annotations = prepro_eeg.annotations.to_data_frame()["description"]

        eeg_filtered_events = [a for a in eeg_annotations if a in erp_events]
        eeg_filtered_events = list(set(eeg_filtered_events))

        eeg_epochs = mne.Epochs(
            prepro_eeg,
            # event_id=["trial_end"],
            event_id=eeg_filtered_events,
            tmin=erp_tmin,
            tmax=erp_tmax,
            baseline=None,
            verbose=False,
        )
        evoked = eeg_epochs.average()

        # fig = evoked.plot()
        # fig.savefig(
        #     save_dir / f"response_lock_ERP-subj_{subj_N:02}{sess_N}", dpi=300
        # )

        eeg_data_seq_lvl = eeg_epochs.get_data(picks=selected_chans)

        # * Remove practice trials
        if eeg_data_seq_lvl.shape[0] > 80:
            print("Removing practice trials from EEG data")
            eeg_data_seq_lvl = eeg_data_seq_lvl[3:]

        # eeg_data_seq_lvl.shape
        # eeg_data_seq_lvl: np.ndarray = np.concatenate(eeg_data_seq_lvl)

        if eeg_data_seq_lvl.shape[0] != raw_behav.shape[0]:
            print(
                f"WARNING: different number of trials between EEG and behavioral data for subj {subj_N:02} sess {sess_N:02}. Skipping..."
            )

            ds_seq_lvl, rdm_seq_lvl = get_ds_and_rdm(
                measurements=np.array([np.nan] * raw_behav.shape[0])[:, None],
                dissimilarity_metric=dissimilarity_metric,
                # ds_fpath=ds_fpath,
                # rdm_fpath=rdm_fpath,
                descriptors={"id": id, "chan_group": [chan_group]},
                obs_descriptors={
                    "item_ids": list(raw_behav["item_id"]),
                    "patterns": list(raw_behav["pattern"]),
                    # "sessions": list(raw_behav["pattern"]),
                },
            )

            ds_patt_lvl, rdm_patt_lvl = get_ds_and_rdm(
                measurements=np.array([np.nan] * len(c.PATTERNS))[:, None],
                dissimilarity_metric=dissimilarity_metric,
                # ds_fpath=ds_fpath,
                # rdm_fpath=rdm_fpath,
                descriptors={"id": id},
                obs_descriptors={"patterns": c.PATTERNS},
            )
            return ds_seq_lvl, rdm_seq_lvl, ds_patt_lvl, rdm_patt_lvl

        # * Reorder the data
        reordered_inds = reorder_item_ids(
            original_order_df=raw_behav,
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )
        raw_behav = raw_behav.iloc[reordered_inds]
        raw_behav.reset_index(drop=True, inplace=True)
        eeg_data_seq_lvl = eeg_data_seq_lvl[reordered_inds]

        # * ----------------------------------------
        # * Sequence level analysis
        # * ----------------------------------------
        # base_fname = f"human-subj_{subj_N:02}-sequence_lvl.hdf5"
        # ds_fpath = save_dir / f"dataset-{base_fname}"
        # rdm_fpath = save_dir / f"rdm-{base_fname}"

        ds_seq_lvl, rdm_seq_lvl = get_ds_and_rdm(
            measurements=eeg_data_seq_lvl.reshape(eeg_data_seq_lvl.shape[0], -1),
            dissimilarity_metric=dissimilarity_metric,
            # ds_fpath=ds_fpath,
            # rdm_fpath=rdm_fpath,
            # descriptors={"subj_N": [subj_N], "chan_group": [chan_group]},
            descriptors={"id": [subj_N], "chan_group": [chan_group]},
            obs_descriptors={
                "item_ids": list(raw_behav["item_id"]),
                "patterns": list(raw_behav["pattern"]),
                # "sessions": list(raw_behav["pattern"]),
            },
        )

        # * Plot the RDM and save the figure
        # fig, ax = plot_rdm(rdm_seq_lvl, "patterns", True)
        # ax.set_title(f"RDM - subj {subj_N:02} - sequence level")
        # fig.savefig(rdm_fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.close()

        # * ----------------------------------------
        # * Pattern level analysis
        # * ----------------------------------------
        eeg_data_patt_lvl = {p: [] for p in c.PATTERNS}

        for i, patt in raw_behav["pattern"].items():
            eeg_data_patt_lvl[patt].append(eeg_data_seq_lvl[i])

        eeg_data_patt_lvl = np.array([np.array(v) for v in eeg_data_patt_lvl.values()])
        eeg_data_patt_lvl = np.nanmean(eeg_data_patt_lvl, axis=1)

        # base_fname = f"human-subj_{subj_N:02}-pattern_lvl.hdf5"
        # ds_fpath = save_dir / f"dataset-{base_fname}"
        # rdm_fpath = save_dir / f"rdm-{base_fname}"

        ds_patt_lvl, rdm_patt_lvl = get_ds_and_rdm(
            measurements=eeg_data_patt_lvl.reshape(eeg_data_patt_lvl.shape[0], -1),
            dissimilarity_metric=dissimilarity_metric,
            # ds_fpath=ds_fpath,
            # rdm_fpath=rdm_fpath,
            # descriptors={"subj_N": [subj_N]},
            descriptors={"id": [subj_N]},
            obs_descriptors={"patterns": c.PATTERNS},
        )

        # # * Plot the RDM and save the figure
        # fig, ax = plot_rdm(rdm_patt_lvl, "patterns", False)
        # ax.set_title(f"RDM - subj {subj_N:02} - pattern level \n all chans")
        # fig.savefig(rdm_fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.close()

        return ds_seq_lvl, rdm_seq_lvl, ds_patt_lvl, rdm_patt_lvl

    def get_behav_rdms(self):
        raise NotImplementedError

    def get_eye_tracking_rdms(self):
        raise NotImplementedError

    # * ########################################
    # * Analyze / Process the preprocessed data
    # * ########################################
    @staticmethod
    def _empty_stim_fixation_summary() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "stim_ind",
                "count",
                "first_fix_order",
                "total_duration",
                "mean_duration",
                "mean_pupil_diam",
                "stim_name",
                "trial_N",
                "stim_type",
            ]
        )

    @staticmethod
    def _stimulus_type_map(
        stim_labels: dict[int, str],
        sequence_items: dict[int, str],
        solution_stim_ind: int,
    ) -> dict[int, str]:
        stimulus_types = {}
        for stim_ind, icon_name in stim_labels.items():
            if stim_ind < 7:
                stimulus_types[stim_ind] = "sequence"
            elif stim_ind == 7:
                stimulus_types[stim_ind] = "question_mark"
            elif stim_ind == solution_stim_ind:
                stimulus_types[stim_ind] = "choice_correct"
            elif icon_name in sequence_items.values():
                stimulus_types[stim_ind] = "choice_incorrect_related"
            else:
                stimulus_types[stim_ind] = "choice_incorrect_unrelated"
        return stimulus_types

    @staticmethod
    def _fixation_events_table(
        fixation_rows: list[list],
        trial_N: int,
        stim_labels: dict[int, str],
        stimulus_types: dict[int, str],
    ) -> pd.DataFrame:
        fixation_events = pd.DataFrame(
            fixation_rows,
            columns=["stim_ind", "onset", "duration", "pupil_diam"],
        )
        fixation_events["trial_N"] = trial_N
        fixation_events["stim_name"] = fixation_events["stim_ind"].replace(stim_labels)
        fixation_events["stim_type"] = fixation_events["stim_ind"].replace(
            stimulus_types
        )
        fixation_events["duration"] = pd.to_numeric(
            fixation_events["duration"], errors="coerce"
        )
        fixation_events["pupil_diam"] = pd.to_numeric(
            fixation_events["pupil_diam"], errors="coerce"
        ).round(2)
        return fixation_events

    @classmethod
    def _stim_fixation_summary(
        cls,
        fixation_events: pd.DataFrame,
        trial_N: int,
        stim_labels: dict[int, str],
        stimulus_types: dict[int, str],
    ) -> pd.DataFrame:
        if fixation_events.empty:
            return cls._empty_stim_fixation_summary()

        first_fixation_order = (
            fixation_events.sort_values("onset")
            .groupby("stim_ind")
            .first()["onset"]
            .rank()
            .astype(int)
        )
        first_fixation_order.name = "first_fix_order"

        summary = pd.concat(
            [
                fixation_events["stim_ind"].value_counts().rename("count"),
                first_fixation_order,
                fixation_events.groupby("stim_ind")["duration"]
                .sum()
                .rename("total_duration"),
                fixation_events.groupby("stim_ind")["duration"]
                .mean()
                .rename("mean_duration"),
                fixation_events.groupby("stim_ind")["pupil_diam"]
                .mean()
                .round(2)
                .rename("mean_pupil_diam"),
            ],
            axis=1,
        ).reset_index()

        summary["stim_name"] = summary["stim_ind"].replace(stim_labels)
        summary["trial_N"] = trial_N
        summary["stim_type"] = summary["stim_ind"].replace(stimulus_types)
        summary.sort_values("stim_ind", inplace=True)
        return summary

    @staticmethod
    def _epochs_by_stim(
        eeg_arrays_by_stim: dict[int, list[np.ndarray]],
        eeg_info: mne.Info,
        eeg_baseline: float,
    ) -> dict[int, mne.EpochsArray]:
        epochs_by_stim = {}
        for stim_ind, epoch_arrays in eeg_arrays_by_stim.items():
            if not epoch_arrays:
                continue
            epochs_by_stim[stim_ind] = mne.EpochsArray(
                np.stack(epoch_arrays),
                eeg_info,
                tmin=-eeg_baseline,
                baseline=(None, 0),
                verbose="WARNING",
            )
            # TODO: If detrending is needed, pass detrend=1 when creating an
            # MNE Epochs object from events, or explicitly detrend epochs_by_stim
            # with a validated MNE/scipy routine here.
        return epochs_by_stim

    @staticmethod
    def _average_fixation_epochs(
        epochs_by_stim: dict[int, mne.EpochsArray],
        stim_inds: list[int],
    ) -> mne.Evoked | None:
        selected_epochs = [
            epochs
            for stim_ind, epochs in epochs_by_stim.items()
            if stim_ind in stim_inds and len(epochs) > 0
        ]
        if not selected_epochs:
            return None
        return mne.concatenate_epochs(selected_epochs, verbose="WARNING").average()

    @staticmethod
    def _stim_inds_for_scope(
        stim_scope: Literal["sequence", "choices", "both"],
        sequence_items: dict[int, str],
        choice_items: dict[int, str],
    ) -> list[int]:
        sequence_stim_inds = list(sequence_items.keys())
        choice_stim_inds = [i + len(sequence_items) for i in choice_items.keys()]

        if stim_scope == "sequence":
            return sequence_stim_inds
        if stim_scope == "choices":
            return choice_stim_inds
        if stim_scope == "both":
            return sequence_stim_inds + choice_stim_inds
        raise ValueError("stim_scope must be one of: 'sequence', 'choices', 'both'")

    def _plot_fixation_window(
        self,
        eeg_epoch_array: np.ndarray | None,
        eeg_info: mne.Info,
        cropped_eeg_trial: mne.io.Raw,
        fixation_is_valid: bool,
        on_target: bool,
        stim_ind: int | None,
        fixation_duration: float,
        eeg_fixation_start_sample: int,
        eeg_fixation_stop_sample: int,
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
        eeg_baseline: float,
        response_onset: float,
        stimulus_positions: list,
        plot_context: tuple,
    ) -> None:
        _, ch_group_inds, group_colors, chans_pos_xy = plot_context
        plot_title = f"ICON-{stim_ind}" if on_target else "OFF-TARGET"
        plot_title += f" ({fixation_duration * 1000:.0f} ms)"
        plot_title += " - " + ("SAVED" if fixation_is_valid else "DISCARDED")

        if eeg_epoch_array is None:
            plot_eeg_data_uv = np.empty((0, 0))
            eeg_start_time = np.nan
            eeg_end_time = np.nan
        else:
            eeg_epoch = mne.EpochsArray(
                [eeg_epoch_array],
                eeg_info,
                tmin=-eeg_baseline,
                baseline=(None, 0),
                verbose="WARNING",
            )
            plot_eeg_data_uv = eeg_epoch.get_data(picks="eeg", units="uV")[0]
            eeg_start_time = cropped_eeg_trial.times[eeg_fixation_start_sample]
            eeg_end_time = cropped_eeg_trial.times[eeg_fixation_stop_sample - 1]

        plot_eeg_and_gaze_fixations(
            eeg_data=plot_eeg_data_uv,
            eeg_sfreq=c.EEG_SFREQ,
            et_data=np.stack([gaze_x, gaze_y], axis=1).T,
            eeg_baseline=eeg_baseline,
            response_onset=response_onset,
            eeg_start_time=eeg_start_time,
            eeg_end_time=eeg_end_time,
            icon_images=c.ICON_IMAGES,
            img_size=c.IMG_SIZE,
            stim_pos=stimulus_positions,
            chans_pos_xy=chans_pos_xy,
            ch_group_inds=ch_group_inds,
            group_colors=group_colors,
            screen_resolution=c.SCREEN_RESOLUTION,
            title=plot_title,
            vlines=[
                eeg_baseline * c.EEG_SFREQ,
                eeg_baseline * c.EEG_SFREQ + fixation_duration * c.EEG_SFREQ,
            ],
        )

    def get_trial_frp(
        self,
        eeg_trial: mne.io.Raw,
        et_trial: mne.io.eyelink.eyelink.RawEyelink,
        raw_behav: pd.DataFrame,
        trial_N: int,
        stim_scope: Literal["sequence", "choices", "both"] = "sequence",
        tmin: float = -0.100,
        tmax: float = 0.600,
        baseline: tuple[float | None, float | None] | None = None,
        selected_chans: list[str] | str | None = "eeg",
        min_fixation_duration: float | None = None,
        return_epochs: bool = False,
    ) -> mne.Evoked | None | tuple[mne.Evoked | None, mne.EpochsArray | None]:
        """Compute one trial-level FRP from fixations on selected stimuli.

        The method finds valid fixation events on sequence icons, choice icons,
        or both, extracts EEG windows around fixation onset, concatenates those
        fixation-locked epochs, and returns their average.
        """
        if tmax <= tmin:
            raise ValueError("tmax must be larger than tmin")
        assert c.EEG_SFREQ == eeg_trial.info["sfreq"], (
            "EEG data has incorrect sampling rate"
        )

        min_fixation_duration = (
            c.MIN_FIXATION_DURATION
            if min_fixation_duration is None
            else min_fixation_duration
        )

        cropped_et_trial, et_annotations, time_bounds = self.crop_et_trial(et_trial)
        eeg_crop_start_time = max(eeg_trial.times[0], time_bounds[0] + min(tmin, 0))
        eeg_crop_stop_time = min(eeg_trial.times[-1], time_bounds[1] + max(tmax, 0))
        cropped_eeg_trial = eeg_trial.copy().crop(
            tmin=eeg_crop_start_time,
            tmax=eeg_crop_stop_time,
        )

        (
            stimulus_positions,
            _,
            sequence_items,
            choice_items,
            *_,
        ) = get_trial_info(
            trial_N,
            raw_behav,
            c.X_POS_STIM,
            c.Y_POS_CHOICES,
            c.Y_POS_SEQUENCE,
            c.SCREEN_RESOLUTION,
            c.IMG_SIZE,
        )
        selected_stim_inds = set(
            self._stim_inds_for_scope(stim_scope, sequence_items, choice_items)
        )

        et_data = cropped_et_trial.get_data()
        et_times = cropped_et_trial.times
        eeg_data = cropped_eeg_trial.get_data(picks=selected_chans)
        eeg_info = (
            cropped_eeg_trial.info
            if selected_chans is None
            else cropped_eeg_trial.copy().pick(selected_chans).info
        )
        sfreq = float(cropped_eeg_trial.info["sfreq"])
        n_epoch_samples = int(np.ceil((tmax - tmin) * sfreq)) + 1
        epoch_arrays = []

        for fixation_ind in et_annotations.query("description == 'fixation'").index:
            fixation = et_annotations.loc[fixation_ind]
            fixation_onset = float(fixation["onset"])
            fixation_duration = float(fixation["duration"])
            fixation_offset = min(fixation_onset + fixation_duration, et_times[-1])

            et_start_sample = int(
                np.searchsorted(et_times, fixation_onset, side="left")
            )
            et_stop_sample = int(
                np.searchsorted(et_times, fixation_offset, side="right")
            )
            if et_stop_sample <= et_start_sample:
                continue

            gaze_x, gaze_y = et_data[:2, et_start_sample:et_stop_sample]
            on_target, stim_ind = self.is_fixation_on_target(
                gaze_x, gaze_y, stimulus_positions
            )
            if (
                not on_target
                or stim_ind not in selected_stim_inds
                or fixation_duration < min_fixation_duration
            ):
                continue

            fixation_onset_in_eeg_crop = (
                time_bounds[0] + fixation_onset - eeg_crop_start_time
            )
            eeg_window_start_sample = int(
                np.round((fixation_onset_in_eeg_crop + tmin) * sfreq)
            )
            eeg_window_stop_sample = eeg_window_start_sample + n_epoch_samples
            if (
                eeg_window_start_sample < 0
                or eeg_window_stop_sample > eeg_data.shape[1]
            ):
                logger.warning(
                    "Skipping trial FRP fixation window beyond trial bounds "
                    f"(subj={self.subj_N:02}, sess={self.sess_N:02}, "
                    f"trial={trial_N}, onset={fixation_onset:.3f}, "
                    f"tmin={tmin:.3f}, tmax={tmax:.3f})."
                )
                continue
            epoch_arrays.append(
                eeg_data[:, eeg_window_start_sample:eeg_window_stop_sample]
            )

        if not epoch_arrays:
            return (None, None) if return_epochs else None

        fixation_epochs = mne.EpochsArray(
            np.stack(epoch_arrays),
            eeg_info,
            tmin=tmin,
            baseline=baseline,
            verbose="WARNING",
        )
        trial_frp = fixation_epochs.average()
        if return_epochs:
            return trial_frp, fixation_epochs
        return trial_frp

    def analyze_trial_decision_period(
        self,
        eeg_trial: mne.io.Raw,
        et_trial: mne.io.eyelink.eyelink.RawEyelink,
        raw_behav: pd.DataFrame,
        trial_N: int,
        eeg_baseline: float = 0.100,
        eeg_window: float = 0.600,
        frp_baseline: tuple[float | None, float | None] | None = None,
        show_plots: bool = True,
        pbar_off=True,
    ):
        """
        Analyze valid stimuli fixations during the trial decision period.

        The method first extracts lightweight fixation metadata and gaze traces,
        then builds fixation-locked EEG epochs in batches grouped by stimuli. This
        avoids repeated Raw.copy().crop() and one-EpochsArray-per-fixation work.
        """
        assert c.EEG_SFREQ == eeg_trial.info["sfreq"], (
            "EEG data has incorrect sampling rate"
        )

        cropped_et_trial, et_annotations, time_bounds = self.crop_et_trial(et_trial)
        eeg_crop_start_time = max(eeg_trial.times[0], time_bounds[0] - eeg_baseline)
        eeg_crop_stop_time = min(eeg_trial.times[-1], time_bounds[1] + eeg_window)
        cropped_eeg_trial = eeg_trial.copy().crop(
            tmin=eeg_crop_start_time,
            tmax=eeg_crop_stop_time,
        )

        eeg_data = cropped_eeg_trial.get_data()
        eeg_info = cropped_eeg_trial.info
        n_epoch_samples = int(np.ceil((eeg_window + eeg_baseline) * c.EEG_SFREQ)) + 1

        et_data = cropped_et_trial.get_data()
        et_times = cropped_et_trial.times

        (
            stimulus_positions,
            stimulus_order,
            sequence_items,
            choice_items,
            _,
            solution,
            _,
        ) = get_trial_info(
            trial_N,
            raw_behav,
            c.X_POS_STIM,
            c.Y_POS_CHOICES,
            c.Y_POS_SEQUENCE,
            c.SCREEN_RESOLUTION,
            c.IMG_SIZE,
        )

        response_onset = et_annotations.query(
            "description.isin(['a', 'x', 'm', 'l', 'timeout', 'invalid'])"
        ).iloc[0]["onset"]

        stim_labels = sequence_items.copy()
        stim_labels.update(
            {k + len(sequence_items): v for k, v in choice_items.items()}
        )
        sequence_stim_inds = list(sequence_items.keys())
        choice_stim_inds = [i + len(sequence_items) for i in choice_items.keys()]
        solution_stim_ind = {v: k for k, v in choice_items.items()}[solution]
        solution_stim_ind += len(sequence_items)
        stimulus_types = self._stimulus_type_map(
            stim_labels, sequence_items, solution_stim_ind
        )

        fixation_inds = et_annotations.query("description == 'fixation'").index
        gaze_traces_by_stim: dict[int, list[np.ndarray]] = {
            i: [] for i in range(len(stimulus_order))
        }
        eeg_arrays_by_stim: dict[int, list[np.ndarray]] = {
            i: [] for i in range(len(stimulus_order))
        }
        valid_fixation_rows = []

        plot_context = None
        if show_plots:
            channel_group_names = [
                "frontal",
                "parietal",
                "central",
                "temporal",
                "occipital",
            ]
            channel_group_colors = ["red", "green", "blue", "pink", "orange"]
            plot_context = prepare_eeg_data_for_plot(
                c.EEG_CHAN_GROUPS,
                c.EEG_MONTAGE,
                c.NON_EEG_CHANS,
                cropped_eeg_trial.info["bads"],
                channel_group_names,
                channel_group_colors,
            )

        pbar = tqdm(fixation_inds, leave=False, disable=pbar_off)

        for fixation_ind in pbar:
            fixation = et_annotations.loc[fixation_ind]
            fixation_onset = float(fixation["onset"])
            fixation_duration = float(fixation["duration"])
            fixation_offset = min(fixation_onset + fixation_duration, et_times[-1])

            et_start_sample = int(
                np.searchsorted(et_times, fixation_onset, side="left")
            )
            et_stop_sample = int(
                np.searchsorted(et_times, fixation_offset, side="right")
            )
            if et_stop_sample <= et_start_sample:
                continue

            gaze_x, gaze_y, pupil_diameter = et_data[:, et_start_sample:et_stop_sample]
            on_target, stim_ind = self.is_fixation_on_target(
                gaze_x, gaze_y, stimulus_positions
            )

            fixation_is_valid = (
                fixation_duration >= c.MIN_FIXATION_DURATION and on_target
            )
            absolute_fixation_onset = time_bounds[0] + fixation_onset
            eeg_fixation_start_sample = int(
                np.round(
                    (absolute_fixation_onset - eeg_baseline - eeg_crop_start_time)
                    * c.EEG_SFREQ
                )
            )
            eeg_fixation_stop_sample = eeg_fixation_start_sample + n_epoch_samples
            eeg_epoch_array = None

            if (
                0 <= eeg_fixation_start_sample
                and eeg_fixation_stop_sample <= eeg_data.shape[1]
            ):
                eeg_epoch_array = eeg_data[
                    :, eeg_fixation_start_sample:eeg_fixation_stop_sample
                ]
            elif fixation_is_valid:
                logger.warning(
                    "Skipping fixation EEG window beyond trial bounds "
                    f"(subj={self.subj_N:02}, sess={self.sess_N:02}, "
                    f"trial={trial_N}, onset={fixation_onset:.3f})."
                )
                fixation_is_valid = False

            if fixation_is_valid:
                gaze_traces_by_stim[stim_ind].append(np.array([gaze_x, gaze_y]))
                eeg_arrays_by_stim[stim_ind].append(eeg_epoch_array)
                valid_fixation_rows.append(
                    [
                        stim_ind,
                        fixation_onset,
                        fixation_duration,
                        np.nanmean(pupil_diameter),
                    ]
                )

            if show_plots:
                self._plot_fixation_window(
                    eeg_epoch_array=eeg_epoch_array,
                    eeg_info=eeg_info,
                    cropped_eeg_trial=cropped_eeg_trial,
                    fixation_is_valid=fixation_is_valid,
                    on_target=on_target,
                    stim_ind=stim_ind,
                    fixation_duration=fixation_duration,
                    eeg_fixation_start_sample=eeg_fixation_start_sample,
                    eeg_fixation_stop_sample=eeg_fixation_stop_sample,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    eeg_baseline=eeg_baseline,
                    response_onset=response_onset,
                    stimulus_positions=stimulus_positions,
                    plot_context=plot_context,
                )
        plt.close("all")

        eeg_epochs_by_stim = self._epochs_by_stim(
            eeg_arrays_by_stim, eeg_info, eeg_baseline
        )
        fixations_sequence_erp = self.get_trial_frp(
            eeg_trial=eeg_trial,
            et_trial=et_trial,
            raw_behav=raw_behav,
            trial_N=trial_N,
            stim_scope="sequence",
            tmin=-eeg_baseline,
            tmax=eeg_window,
            baseline=frp_baseline,
            selected_chans="eeg",
        )
        fixations_choices_erp = self.get_trial_frp(
            eeg_trial=eeg_trial,
            et_trial=et_trial,
            raw_behav=raw_behav,
            trial_N=trial_N,
            stim_scope="choices",
            tmin=-eeg_baseline,
            tmax=eeg_window,
            baseline=frp_baseline,
            selected_chans="eeg",
        )

        eeg_fixation_pac_data = {}
        fixation_events = self._fixation_events_table(
            valid_fixation_rows, trial_N, stim_labels, stimulus_types
        )
        stim_fixation_summary = self._stim_fixation_summary(
            fixation_events, trial_N, stim_labels, stimulus_types
        )

        return (
            gaze_traces_by_stim,
            eeg_epochs_by_stim,
            eeg_fixation_pac_data,
            fixation_events,
            stim_fixation_summary,
            fixations_sequence_erp,
            fixations_choices_erp,
        )

    def analyze_flash_period(self, et_epoch, eeg_epoch, raw_behav, epoch_N):
        raise NotImplementedError
        # # ! IMPORTANT
        # # TODO: can't remember, something related to eeg baseline ?
        # # * eeg_baseline

        # def crop_et_epoch(epoch):
        #     # ! WARNING: We may not be capturing the first fixation if it is already on target
        #     # * Get annotations, convert to DataFrame, and adjust onset times
        #     annotations = epoch.annotations.to_data_frame(time_format="ms")
        #     annotations["onset"] -= annotations["onset"].iloc[0]
        #     annotations["onset"] /= 1000

        #     first_flash = annotations.query("description.str.contains('flash')").iloc[0]
        #     all_stim_pres = annotations.query("description == 'stim-all_stim'").iloc[0]

        #     # * Convert to seconds
        #     start_time = (
        #         annotations.iloc[: first_flash.name]
        #         .query("description == 'fixation'")
        #         .iloc[-1]["onset"]
        #     )
        #     end_time = all_stim_pres["onset"]

        #     # * Crop the data
        #     epoch = epoch.copy().crop(start_time, end_time)

        #     # * Get annotations, convert to DataFrame, and adjust onset times
        #     annotations = epoch.annotations.to_data_frame(time_format="ms")
        #     annotations["onset"] -= annotations["onset"].iloc[0]
        #     annotations["onset"] /= 1000

        #     time_bounds = (start_time, end_time)

        #     return epoch, annotations, time_bounds

        # et_epoch, et_annotations, time_bounds = crop_et_epoch(et_epoch)

        # eeg_epoch = eeg_epoch.copy().crop(*time_bounds)

        # # * Get channel positions for topomap
        # info = eeg_epoch.info

        # chans_pos_xy = np.array(
        #     list(info.get_montage().get_positions()["ch_pos"].values())
        # )[:, :2]

        # # * Get channel indices for each region
        # ch_group_inds = {
        #     group_name: [i for i, ch in enumerate(eeg_epoch.ch_names) if ch in group_chans]
        #     for group_name, group_chans in c.EEG_CHAN_GROUPS.items()
        # }

        # # * Get positions and presentation order of stimuli
        # trial_info = get_trial_info(
        #     epoch_N,
        #     raw_behav,
        #     c.X_POS_STIM,
        #     c.Y_POS_CHOICES,
        #     c.Y_POS_SEQUENCE,
        #     c.SCREEN_RESOLUTION,
        #     c.IMG_SIZE,
        # )
        # stim_pos, stim_order = trial_info[:2]

        # flash_event_ids = ["stim-flash_sequence", "stim-flash_choices"]

        # fixation_inds = et_annotations.query("description == 'fixation'").index

        # fixation_data = {i: [] for i in stim_order}
        # eeg_fixation_data = {i: [] for i in stim_order}

        # for fixation_ind in fixation_inds:
        #     # * Get number of flash events before the current fixation; -1 to get the index
        #     flash_events = et_annotations.iloc[:fixation_ind].query(
        #         f"description.isin({flash_event_ids})"
        #     )

        #     n_flash_events = flash_events.shape[0]
        #     stim_flash_ind = n_flash_events - 1

        #     # * If first fixation is before the first flash, stim_flash_ind will be -1
        #     # * We set it to 0 in this case, and check if fixation is already on target
        #     stim_flash_ind = max(stim_flash_ind, 0)

        #     # * Get target location
        #     target_grid_loc = stim_order[stim_flash_ind]
        #     target_name, target_coords = stim_pos[target_grid_loc]
        #     targ_left, targ_right, targ_bottom, targ_top = target_coords

        #     fixation = et_annotations.loc[fixation_ind]

        #     fixation_start = fixation["onset"]
        #     fixation_duration = fixation["duration"]
        #     fixation_stop = fixation_start + fixation_duration

        #     end_time = min(fixation_stop, et_epoch.times[-1])

        #     gaze_x, gaze_y, pupil_diam = (
        #         et_epoch.copy().crop(fixation_start, end_time).get_data()
        #     )

        #     mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

        #     on_target = (targ_left <= mean_gaze_x <= targ_right) and (
        #         targ_bottom <= mean_gaze_y <= targ_top
        #     )

        #     end_time = min(fixation_stop, eeg_epoch.times[-1])

        #     eeg_slice = eeg_epoch.copy().crop(fixation_start, end_time)

        #     # eeg_annotations = eeg_slice.annotations.to_data_frame(time_format="ms")
        #     # mne_events, _ = mne.events_from_annotations(eeg_slice)
        #     # eeg_annotations.insert(1, 'sample_nb', mne_events[:, 0])

        #     eeg_fixation_data[stim_flash_ind].append(eeg_slice)

        #     eeg_slice = eeg_slice.copy().pick(["eeg"]).get_data()

        #     if fixation_duration >= c.MIN_FIXATION_DURATION and on_target:
        #         discarded = False
        #         fixation_data[stim_flash_ind].append(np.array([gaze_x, gaze_y]))
        #     else:
        #         discarded = True

        #     title = f"FLASH-{stim_flash_ind} ({fixation_duration * 1000:.0f} ms)"
        #     title += " - " + ("DISCARDED" if discarded else "SAVED")

        #     fig = plt.figure(figsize=(10, 6))
        #     gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[1, 1])
        #     ax_et = fig.add_subplot(gs[0, 0])
        #     ax_topo = fig.add_subplot(gs[0, 1])
        #     ax_eeg = fig.add_subplot(gs[1, :])
        #     ax_eeg_group = fig.add_subplot(gs[2, :], sharex=ax_eeg)
        #     ax_eeg_avg = fig.add_subplot(gs[3, :], sharex=ax_eeg)

        #     ax_et.set_xlim(0, c.SCREEN_RESOLUTION[0])
        #     ax_et.set_ylim(c.SCREEN_RESOLUTION[1], 0)
        #     ax_et.set_title(title)

        #     # * Plot target icon
        #     ax_et.imshow(
        #         c.ICON_IMAGES[target_name],
        #         extent=[targ_left, targ_right, targ_bottom, targ_top],
        #         origin="lower",
        #     )

        #     mne.viz.plot_topomap(
        #         eeg_slice.mean(axis=1),
        #         chans_pos_xy,
        #         ch_type="eeg",
        #         sensors=True,
        #         contours=0,
        #         outlines="head",
        #         sphere=None,
        #         image_interp="cubic",
        #         extrapolate="auto",
        #         border="mean",
        #         res=640,
        #         size=1,
        #         cmap=None,
        #         vlim=(None, None),
        #         cnorm=None,
        #         axes=ax_topo,
        #         show=False,
        #     )

        #     # * Plot rectangle around target, with dimensions == img_size
        #     rectangle = mpatches.Rectangle(
        #         (targ_left, targ_bottom),
        #         c.IMG_SIZE[0],
        #         c.IMG_SIZE[1],
        #         linewidth=1,
        #         linestyle="--",
        #         edgecolor="black",
        #         facecolor="none",
        #     )
        #     ax_et.add_patch(rectangle)

        #     # * Plot gaze data
        #     ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
        #     ax_et.scatter(mean_gaze_x, mean_gaze_y, c="yellow", s=3)

        #     # * Plot EEG data
        #     ax_eeg.plot(eeg_slice.T)

        #     # ax_eeg.vlines(eeg_annotations["sample_nb"], eeg_slice.T.min(), eeg_slice.T.max())

        #     ax_eeg_avg.plot(eeg_slice.mean(axis=0))

        #     ax_eeg_group.plot(
        #         eeg_slice[ch_group_inds["occipital"]].mean(axis=0),
        #         color="red",
        #         label="occipital",
        #     )

        #     ax_eeg_group.plot(
        #         eeg_slice[ch_group_inds["parietal"]].mean(axis=0),
        #         color="green",
        #         label="parietal",
        #     )

        #     ax_eeg_group.plot(
        #         eeg_slice[ch_group_inds["centro-parietal"]].mean(axis=0),
        #         color="purple",
        #         label="centro-parietal",
        #     )

        #     ax_eeg_group.plot(
        #         eeg_slice[ch_group_inds["temporal"]].mean(axis=0),
        #         color="orange",
        #         label="temporal",
        #     )

        #     ax_eeg_group.plot(
        #         eeg_slice[ch_group_inds["frontal"]].mean(axis=0),
        #         color="blue",
        #         label="frontal",
        #     )

        #     ax_eeg_group.legend(
        #         bbox_to_anchor=(1.005, 1),
        #         loc="upper left",
        #         borderaxespad=0,
        #     )

        #     ax_eeg.set_xlim(0, eeg_slice.shape[1])
        #     xticks = np.arange(0, eeg_slice.shape[1], 100)
        #     ax_eeg.set_xticks(xticks, ((xticks / c.EEG_SFREQ) * 1000).astype(int))

        #     plt.tight_layout()
        #     plt.show()

        # return fixation_data, eeg_fixation_data

    def analyze_session(
        self,
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
        force_preprocess: bool = False,
        reuse_ica: bool = True,
        raise_error: bool = True,
        pbar: bool = True,
        trial_pbar: Any | None = None,
        frp_baseline: tuple[float | None, float | None] | None = None,
    ):
        """ """
        # # ! TEMP: DEBUG
        # subj_N, sess_N = 1, 2
        # self = HumanSessData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, subj_N, sess_N)
        # save_dir = c.EXPORT_DIR / f"subj_{subj_N:02}-sess_{sess_N:02}"
        # preprocessed_dir = c.EXPORT_DIR / "preprocessed_data"
        # preprocessed_dir = self.preprocessed_dir
        # save_dir = WD / "test-export"
        # force_preprocess = False
        # reuse_ica = True
        # raise_error: bool = False
        # # ! TEMP: DEBUG

        should_save = save_dir is not None
        if should_save:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        if not preprocessed_dir.exists():
            raise FileNotFoundError("Preprocessed data directory not found")

        subj_N, sess_N = self.subj_N, self.sess_N

        bad_chans = c.ALL_BAD_CHANS.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])

        # * Load the data
        # sess_info, behav, eeg, et, et_cals = self.get_data()
        data = self.get_data()

        # if any([v is None for v in (sess_info, behav, eeg, et, et_cals)]):
        if any([v is None for v in data]):
            if raise_error:
                raise ValueError("Bad session")
            else:
                print("Bad session, skipping...")
                return {}

        sess_info, behav, eeg, et = data

        # TODO: Implement self.get_data() to get preprocessed data
        # sess_info, raw_behav, raw_eeg, raw_et, et_cals = self.get_raw_data(bad_chans)

        if notes := sess_info["notes"]:
            print(f"SESSION NOTES:\n{notes}")

        # sess_screen_resolution = sess_info["window_size"]
        # sess_img_size = sess_info["img_size"]
        # et_sfreq = raw_et.info["sfreq"]
        # tracked_eye = sess_info["eye"]
        # vision_correction = sess_info["vision_correction"]
        # eye_screen_distance = sess_info["eye_screen_dist"]

        if not c.ET_SFREQ == et.info["sfreq"]:
            raise ValueError("Eye-tracking data has incorrect sampling rate")

        if not c.EEG_SFREQ == eeg.info["sfreq"]:
            raise ValueError("EEG data has incorrect sampling rate")

        (
            manual_et_trials,
            *_,
            # et_events_dict,
            # et_events_dict_inv,
            # et_trial_bounds,
            # et_trial_events_df,
        ) = self.split_et_data_into_trials(et)

        (
            manual_eeg_trials,
            *_,
            # eeg_trial_bounds,
            # eeg_events,
            # eeg_events_df,
        ) = self.split_eeg_data_into_trials(eeg, behav)

        bad_chans = eeg.info["bads"]

        # * Initialize data containers
        sess_frps: Dict[str, List] = {"sequence": [], "choices": []}
        gaze_fixation_traces_all = []
        eeg_fixation_epochs_all = []
        stim_fixation_summary_all = []
        fixation_events_all = []
        eeg_fixation_pac_data_all = []

        if trial_pbar is not None:
            trial_iter = behav.index
            trial_pbar.reset(total=len(behav.index))
            trial_pbar.set_description(f"subj {subj_N:02} sess {sess_N:02} trials")
            trial_pbar.refresh()
        else:
            trial_iter = tqdm(
                behav.index, desc="Analyzing every trial", leave=False, disable=not pbar
            )

        for trial_N in trial_iter:
            # * Get the EEG and ET data for the current trial
            eeg_trial = next(manual_eeg_trials)
            et_trial = next(manual_et_trials)

            (
                gaze_fixation_traces,
                eeg_fixation_epochs,
                eeg_fixation_pac_data,
                fixation_events,
                stim_fixation_summary,
                fixations_sequence_erp,
                fixations_choices_erp,
            ) = self.analyze_trial_decision_period(
                eeg_trial,
                et_trial,
                behav,
                trial_N,
                eeg_baseline=c.EEG_BASELINE_FRP,
                eeg_window=c.FRP_WINDOW,
                frp_baseline=frp_baseline,
                show_plots=False,
            )

            sess_frps["sequence"].append(fixations_sequence_erp)
            sess_frps["choices"].append(fixations_choices_erp)
            gaze_fixation_traces_all.append(gaze_fixation_traces)
            eeg_fixation_epochs_all.append(eeg_fixation_epochs)
            stim_fixation_summary_all.append(stim_fixation_summary)
            fixation_events_all.append(fixation_events)
            eeg_fixation_pac_data_all.append(eeg_fixation_pac_data)
            if trial_pbar is not None:
                trial_pbar.update(1)

        # ic(len(eeg_fixation_pac_data_all))

        # * Concatenate fixation summary tables
        stim_fixation_summary_tables = [
            df for df in stim_fixation_summary_all if df.shape[0] > 0
        ]
        if stim_fixation_summary_tables:
            stim_fixation_summary = pd.concat(stim_fixation_summary_tables)
        else:
            stim_fixation_summary = pd.DataFrame(
                columns=[
                    "stim_ind",
                    "count",
                    "first_fix_order",
                    "total_duration",
                    "mean_duration",
                    "mean_pupil_diam",
                    "stim_name",
                    "trial_N",
                    "stim_type",
                ]
            )
        stim_fixation_summary.reset_index(drop=True, inplace=True)

        fixation_event_tables = [df for df in fixation_events_all if df.shape[0] > 0]
        if fixation_event_tables:
            fixation_events = pd.concat(fixation_event_tables)
        else:
            fixation_events = pd.DataFrame(
                columns=[
                    "stim_ind",
                    "onset",
                    "duration",
                    "pupil_diam",
                    "trial_N",
                    "stim_name",
                    "stim_type",
                ]
            )
        fixation_events.reset_index(drop=False, inplace=True, names=["fixation_N"])

        valid_frps = dict(
            subj_N=subj_N,
            sess_N=sess_N,
            n_seq_frps=len(sess_frps["sequence"]) - sess_frps["sequence"].count(None),
            n_choices_frps=len(sess_frps["choices"]) - sess_frps["choices"].count(None),
        )

        if should_save:
            # * Save the data to pickle files
            pd.DataFrame([valid_frps]).to_csv(save_dir / "valid_frps.csv", index=False)

            save_pickle(sess_frps, save_dir / "sess_frps.pkl")
            save_pickle(gaze_fixation_traces_all, save_dir / "gaze_fixation_traces.pkl")
            save_pickle(eeg_fixation_epochs_all, save_dir / "eeg_fixation_epochs.pkl")
            stim_fixation_summary.to_parquet(
                save_dir / "stim_fixation_summary.parquet", index=False
            )
            fixation_events.to_parquet(
                save_dir / "fixation_events.parquet", index=False
            )
            save_pickle(
                eeg_fixation_pac_data_all, save_dir / "eeg_fixation_pac_data.pkl"
            )

        return {
            "sess_frps": sess_frps,
            "gaze_fixation_traces": gaze_fixation_traces_all,
            "eeg_fixation_epochs": eeg_fixation_epochs_all,
            "stim_fixation_summary": stim_fixation_summary,
            "fixation_events": fixation_events,
            "gaze_info": stim_fixation_summary,
            "gaze_target_fixation_sequence": fixation_events,
        }

    def get_frp(self):
        raise NotImplementedError

    # * ########################################
    # * Analyze processed data
    # * ########################################
    def analyze_perf(
        self,
        return_raw: Optional[bool] = False,
        patterns: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        cleaned_behav_data: pd.DataFrame = self.get_behav_data()

        cleaned_behav_data["correct"] = cleaned_behav_data["correct"].astype(int)

        if patterns is None:
            patterns = sorted(cleaned_behav_data["pattern"].unique().tolist())

        # * --- Overall Results ---
        overall_acc = cleaned_behav_data["correct"].describe()
        overall_acc = pd.DataFrame(overall_acc).T
        overall_acc.reset_index(drop=True, inplace=True)
        # overall_acc.index = [self.sess_N]
        # overall_acc.index.name = "sess_N"

        overall_rt = cleaned_behav_data["rt"].describe()
        overall_rt = pd.DataFrame(overall_rt).T
        overall_rt.reset_index(drop=True, inplace=True)

        # * --- Detailed Results ---
        acc_by_patt = cleaned_behav_data.groupby("pattern")["correct"].describe()
        acc_by_patt = acc_by_patt.reindex(patterns, fill_value=np.nan)

        rt_by_patt = cleaned_behav_data.groupby("pattern")["rt"].describe()
        rt_by_patt = rt_by_patt.reindex(patterns, fill_value=np.nan)

        rt_by_crct = cleaned_behav_data.groupby("correct")["rt"].describe()
        rt_by_crct = rt_by_crct.reindex([0, 1], fill_value=np.nan)

        rt_by_crct_and_patt = cleaned_behav_data.groupby(["pattern", "correct"])[
            "rt"
        ].describe()
        index = pd.MultiIndex.from_product(
            [patterns, [0, 1]], names=["pattern", "correct"]
        )
        rt_by_crct_and_patt = rt_by_crct_and_patt.reindex(index, fill_value=np.nan)

        for df in [acc_by_patt, rt_by_patt, rt_by_crct, rt_by_crct_and_patt]:
            df.reset_index(drop=False, inplace=True)

        res = dict(
            overall_acc=overall_acc,
            overall_rt=overall_rt,
            acc_by_patt=acc_by_patt,
            rt_by_patt=rt_by_patt,
            rt_by_crct=rt_by_crct,
            rt_by_crct_and_patt=rt_by_crct_and_patt,
        )

        if return_raw:
            res["raw_cleaned"] = cleaned_behav_data
        return res

    def load_frp_data(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        frp_type: str = "sequence",
        data_fmt: str = "exp",
        time_window: Optional[Tuple[float, float]] = None,
        selected_chans: Optional[List[str]] = None,
    ):
        subj_N = self.subj_N

        allowed_data_fmts = ["sess", "exp"]
        allowed_frp_types = ["sequence", "choices"]

        assert data_fmt in allowed_data_fmts, (
            f"`frp_type` must be one of {allowed_data_fmts}"
        )
        assert frp_type in allowed_frp_types, (
            f"`frp_type` must be one of {allowed_frp_types}"
        )

        def _search_sess_N(fpath):
            res = re.findall(r"sess_(\d{2})", str(fpath))
            if res:
                return int(res[0])

        frp_files = sorted(
            list_contents(
                processed_data_dir,
                reg=rf"sub.*{subj_N:02}/ses*/sess_frps.pkl",
                incl="file",
                recurs=True,
            )
        )

        sess_Ns = [_search_sess_N(fpath) for fpath in frp_files]

        # * Load behavioral data from all sessions found
        behav_data = pd.concat(
            [
                load_and_clean_behav_data(raw_data_dir, subj_N, sess_N)
                for sess_N in sess_Ns
            ]
        ).reset_index(drop=True)

        missing_frps = []
        subj_data = {} if data_fmt == "sess" else [[], [], []]

        for sess_N, frp_file in zip(sess_Ns, frp_files):
            try:
                sess_df = behav_data.query(f"subj_N == {subj_N} and sess_N == {sess_N}")
                sess_item_ids = sess_df["item_id"].to_numpy()
                sess_patterns = sess_df["pattern"].to_numpy()

                frp_data = read_file(frp_file)[frp_type]

                if sess_missing_frps := [
                    i for i, s in enumerate(frp_data) if s is None
                ]:
                    n_missing_frps = len(sess_missing_frps)
                    pct_missing_frps = n_missing_frps / len(sess_df)
                    missing_frps.append(
                        (
                            subj_N,
                            sess_N,
                            sess_missing_frps,
                            n_missing_frps,
                            pct_missing_frps,
                        )
                    )

                if time_window is not None:
                    # * Assumes all FRPs have the same length
                    data_shape = (
                        [s for s in frp_data if s is not None][0]
                        .copy()
                        .crop(*time_window)
                        .get_data(picks=selected_chans)
                        .shape
                    )

                    empty_data = np.zeros(data_shape)
                    empty_data[:] = np.nan

                    frp_data = [
                        frp.copy().crop(*time_window).get_data(picks=selected_chans)
                        if frp is not None
                        else empty_data
                        for frp in frp_data
                    ]

                else:
                    data_shape = (
                        [s for s in frp_data if s is not None][0]
                        .copy()
                        .get_data(picks=selected_chans)
                        .shape
                    )

                    empty_data = np.zeros(data_shape)
                    empty_data[:] = np.nan

                    frp_data = [
                        frp.copy().get_data(picks=selected_chans)
                        if frp is not None
                        else empty_data
                        for frp in frp_data
                    ]

                if data_fmt == "sess":
                    subj_data[sess_N] = [
                        sess_item_ids,
                        sess_patterns,
                        frp_data,
                    ]
                else:
                    subj_data[0].extend(sess_item_ids)
                    subj_data[1].extend(sess_patterns)
                    subj_data[2].extend(frp_data)

            # TODO: Impprove this
            except Exception as E:
                print(
                    f"WARNING: error in loading data for subj_{subj_N:02} - sess {sess_N:02}"
                )
                logger.exception(
                    "Error loading FRP data for subj_{:02}, sess_{:02}: {}",
                    subj_N,
                    sess_N,
                    E,
                )

        # dict(zip(np.unique(sess_patterns[sess_missing_frps], return_counts=True)

        missing_frps = pd.DataFrame(
            missing_frps, columns=["subj_N", "sess_N", "trial_Ns", "count", "pct"]
        )

        if data_fmt == "exp":
            subj_data = [np.array(d) for d in subj_data]

        return subj_data, missing_frps
