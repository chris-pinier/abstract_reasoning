from ._common import *
from .base import HumanDataClass
from .session import HumanSessData


@dataclass
class HumanSubjData(HumanDataClass):
    subj_N: int
    sessions: Dict[int, "HumanSessData"] = field(default_factory=dict)
    behav_data = None
    eeg_data = None
    et_data = None

    def __post_init__(self):
        super().__post_init__()

        self.subj_dir = self.search_subj_dir()

        if not self.subj_dir.exists():
            raise FileNotFoundError(f"subject directory not found: {self.subj_dir}")

        self.sess_dirs = self.search_sess_dirs()
        # The self.sessions dictionary is already initialized by default_factory

        for sess_N, sess_dir in self.sess_dirs.items():
            sess_obj = HumanSessData(
                self.data_dir,
                self.preprocessed_dir,
                self.export_dir,
                self.subj_N,
                sess_N,
            )
            self.sessions[sess_N] = sess_obj  # Store in the dictionary

    def __str__(self):
        s = f"""
            Class: "{type(self).__name__}"
            Subj_N: {self.subj_N}
            Sessions: {list(self.sessions.keys())}
            Data Directory: '{self.subj_dir}'
            Preprocessed Directory: '{self.preprocessed_dir}'
            Export Directory: '{self.export_dir}'
            Data Format: {self.data_fmt}
        """
        # task_name
        return textwrap.dedent(s).strip()

    def search_subj_dir(self):
        if self.data_fmt == "bids":
            prefix = "sub-"
            subj_dir = self.data_dir / f"{prefix}{self.subj_N:02}/"
        else:
            prefix = "subj_"
            subj_dir = self.data_dir / f"{prefix}{self.subj_N:02}"
        if not subj_dir.exists():
            raise FileNotFoundError(f"session directory not found: {subj_dir}")
        return subj_dir

    def search_sess_dirs(self):
        if self.data_fmt == "bids":
            sess_pattern = re.compile(r"ses-(\d{2})$")
        else:
            sess_pattern = re.compile(r"sess_(\d{2})$")
        sess_dirs = sorted(
            d
            for d in list_contents(self.subj_dir, incl="folder", recurs=False)
            if sess_pattern.fullmatch(d.name)
        )
        sess_Ns = [int(sess_pattern.fullmatch(d.name).group(1)) for d in sess_dirs]
        sess_dirs = dict(zip(sess_Ns, sess_dirs))

        return sess_dirs

    def show_dir_struct(self, stringRep: bool = False):
        print(f"Subject {self.subj_N:02} - Directory Structure:")
        struct = DisplayTree(self.subj_dir, stringRep=stringRep)
        print(f"\nPath: {self.subj_dir}")
        if stringRep:
            return struct

    def check_data(self, remove_practice: bool = True):
        for sess_N, sess_obj in self.sessions.items():
            sess_obj.check_data(remove_practice=remove_practice)

    def get_sess_info(self):
        sessions_info = {}
        for sess_N, sess_obj in self.sessions.items():
            sessions_info[sess_N] = sess_obj.get_sess_info()
        return sessions_info

    def extract_eeg_metadata(self):
        metadata = {}
        for sess_N, sess_obj in self.sessions.items():
            metadata[sess_N] = sess_obj.extract_eeg_metadata()

        return metadata

    # * ########################################
    # * Eye Tracking Data Preprocessing
    # * ########################################

    # * ########################################
    # * EEG Data Preprocessing
    # * ########################################
    def get_eeg_epochs(
        self,
        selected_chans: List[str],
        erp_events: list[str],
        erp_tmin: float,
        erp_tmax: float,
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
        format: Literal["mne", "np"] = "mne",
        trials_per_sess: int = 80,
        epochs_name: str = "epochs",
        combine: bool = False,
    ) -> Tuple[Any, pd.core.frame.DataFrame, List[int]]:
        # ! TEMP
        # chan_group = "frontal"
        # selected_chans = c.EEG_CHAN_GROUPS[chan_group]
        # erp_tmin = -1.0
        # erp_tmax = 0.0
        # erp_events = ["a", "x", "m", "l", "invalid", "timeout"]
        # save_dir = WD /'test-export/ERP-RDMs'
        # preprocessed_dir = None
        # format = "mne"
        # trials_per_sess = 80
        # epochs_name = "epochs"
        # combine=False
        # ! TEMP

        subj_N = self.subj_N
        if format not in (formats := ["mne", "np"]):
            raise ValueError(f"format must be one of {formats}")

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir

        if not preprocessed_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed data directory not found, check path: {preprocessed_dir}"
            )

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        # prepro_eeg_files = list_contents(
        #     preprocessed_dir, reg=f".*subj_{subj_N:02}.*_preprocessed-raw.fif$"
        # )
        prepro_eeg_files = sorted(
            list_contents(
                preprocessed_dir, reg=rf".*sub-{subj_N:02}.*_prepro.*raw.fif$"
            )
        )

        if len(prepro_eeg_files) == 0:
            raise FileNotFoundError(
                f"Preprocessed EEG files not found for subj {subj_N:02}"
            )

        behav_df = self.get_behav_data()

        sess_epochs = {}
        bad_sessions = []
        # info = None

        for prepro_eeg_file in prepro_eeg_files:
            search_match = re.search(r"_ses-(\d*)_", prepro_eeg_file.stem)
            sess_N = int(search_match[1])

            prepro_eeg = mne.io.read_raw_fif(
                prepro_eeg_file, preload=False, verbose="WARNING"
            )
            if prepro_eeg.info["bads"] != []:
                print(
                    f"WARNING: Found channels marked as bad in session {sess_N}.\n"
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
                baseline=None,
                verbose="WARNING",
            )

            n_eps = eps.events.shape[0]

            if format == "mne":
                eps.load_data()
                eps = eps.pick(selected_chans)
            else:
                eps = eps.get_data(picks=selected_chans, verbose="WARNING")

            # * Remove practice trials
            if sess_N == 1 and n_eps > trials_per_sess:
                print("Removing practice trials from EEG data")
                n_eps -= 3
                eps = eps[3:]

            if n_eps != (n_trials := behav_df.query(f"sess_N == {sess_N}").shape[0]):
                print(
                    f"WARNING: Different number of trials between "
                    f"EEG ({n_eps}) and behavioral ({n_trials}) "
                    f"files in session {sess_N}. Dropping session."
                )
                bad_sessions.append(sess_N)
            else:
                sess_epochs[sess_N] = eps

        # ! Dropping sessions with a mismatch in number of trials btw behave & EEG
        behav_df = behav_df.query(f"sess_N not in {bad_sessions}")

        if combine:
            # sess_epochs = list(sess_epochs.values())
            # if format = "mne":
            #     # TODO: investigate problem below when running line above:
            #     # * ValueError: event_id values must be the same for identical keys for all
            #     # * concatenated epochs. Key "m" maps to 5 in some epochs and to 8 in others
            #     sess_epochs = mne.concatenate_epochs(sess_epochs)
            # else:
            #     sess_epochs = np.concatenate(sess_epochs)
            raise NotImplementedError("`combine` logic not yet implemented")

        if save_dir:
            _formats = {"mne": "mne.Epochs", "np": "numpy.ndarray"}
            info = {
                "epochs": f"epoched EEG data, saved as {_formats[format]} object",
                "behav_df": "Pandas DataFrame containing behavioral data",
                "bad_sessions": "List of indices of discarded EEG sessions (if "
                "mismatch between number of trials in EEG and Behavioral File)",
            }
            fpath = save_dir / f"subj_{subj_N:02}-{epochs_name}.pkl"
            save_pickle(
                {
                    "sess_epochs": sess_epochs,
                    "behav_df": behav_df,
                    "bad_sessions": bad_sessions,
                    "info": info,
                },
                fpath,
            )

            # TODO: SEPARATE SECTION BELOW IN ANOTHER FUNCTION
            # * -------------------------------------
            # * EVOKED
            # * -------------------------------------
            # def get_evoked():
            patt_inds = behav_df.groupby(["pattern"]).groups

            info = mne.create_info(
                ch_names=selected_chans, sfreq=self.eeg_info["sfreq"], ch_types="eeg"
            )
            sess_epochs = mne.concatenate_epochs(
                [
                    mne.EpochsArray(e, info, verbose="WARNING")
                    for e in sess_epochs.values()
                ],
                verbose="WARNING",
            )
            sess_epochs.set_montage(c.EEG_MONTAGE)

            evoked = sess_epochs.average()

            evoked_by_patt = {
                patt: sess_epochs[inds].average() for patt, inds in patt_inds.items()
            }
            evoked_by_patt = {patt: evoked_by_patt[patt] for patt in c.PATTERNS}

            base_name = f"subj_{subj_N:02}-evoked"
            save_pickle(evoked_by_patt, save_dir / f"{base_name}.pkl")
            base_name += "_by_pattern"
            save_pickle(evoked_by_patt, save_dir / f"{base_name}.pkl")

            for patt, _evoked in evoked_by_patt.items():
                fig, ax = plt.subplots(figsize=(10, 4), dpi=500)
                evoked_plot = _evoked.plot(axes=ax, show=False)
                evoked_plot.get_axes()[0].set_title(patt)
                # plt.show()
                fig.savefig(
                    save_dir / f"{base_name}-{patt}.png",
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

        return sess_epochs, behav_df, bad_sessions

    def get_erp(
        self,
        event_ids: List[str],
        tmin: float,
        tmax: float,
        raw: bool = False,
        baseline=None,
        detrend=None,
        method: Literal["mean", "median"] = "mean",
        erp_by_sess: bool = False,
        combine_weights: Literal["equal", "nave"] = "equal",
    ):
        erps = {}

        for sess_N, sess_obj in self.sessions.items():
            erp = sess_obj.get_erp(
                event_ids=event_ids,
                tmin=tmin,
                tmax=tmax,
                raw=raw,
                baseline=baseline,
                detrend=detrend,
                method=method,
            )
            erps[sess_N] = erp

        if not erp_by_sess:
            erps = mne.combine_evoked(list(erps.values()), weights=combine_weights)

        return erps

    def get_trials_data(
        self,
        preprocessed_dir: Path | None = None,
        raise_error: bool = False,
        eeg_incomplete: Literal["allow", "error", "skip"] = "error",
        pbar: bool = True,
    ):
        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        behav, manual_et_trials, manual_eeg_trials = [], [], []

        session_iter = tqdm(
            self.sessions.items(),
            desc=f"Loading subj {self.subj_N:02} session trials",
            total=len(self.sessions),
            disable=not pbar,
        )
        for sess_N, sess_obj in session_iter:
            beh, et, eeg = sess_obj.get_trials_data(
                preprocessed_dir=preprocessed_dir,
                raise_error=raise_error,
                eeg_incomplete=eeg_incomplete,
            )

            if any([i is None for i in [beh, et, eeg]]):
                continue
            else:
                behav.append(beh)
                manual_et_trials.extend(list(et))
                manual_eeg_trials.extend(list(eeg))

        if any([b is not None for b in behav]):
            behav = (
                pd.concat(behav)
                .reset_index(drop=True)
                .reset_index(names="overall_trial_N")
            )
        else:
            behav = pd.DataFrame([])

        return behav, manual_et_trials, manual_eeg_trials

    # * ########################################
    # * Behavioral Data Preprocessing
    # * ########################################

    def load_behav(self, combine: bool = True):
        behav_data = {}
        for sess_N, sess_obj in self.sessions.items():
            behav_data[sess_N] = sess_obj.get_behav_data(combine=combine)
        self.behav_data = behav_data

    def get_behav_data(self, combine: bool = True):
        if combine:
            return self.combine_behav()
        else:
            return self._get_behav_data()

    def _get_behav_data(self):
        if self.behav_data is None:
            behav_data = {}
            for sess_N, sess_obj in self.sessions.items():
                behav_data[sess_N] = sess_obj.get_behav_data()
            return behav_data
        else:
            return self.behav_data

    def combine_behav(self):
        behav_data = (
            self.behav_data if self.behav_data is not None else self._get_behav_data()
        ).values()
        behav_data = pd.concat(behav_data)
        behav_data.sort_values(["sess_N", "trial_N"], inplace=True)
        behav_data.reset_index(drop=True, inplace=True)

        return behav_data

    def analyze_perf(self, agg: Literal["session", "subject"] = "subject"):
        sessions_res = {}
        for sess_N, sess_obj in self.sessions.items():
            # res[sess_N] = sess_obj.analyze_perf(return_raw=True)
            sess_res = sess_obj.analyze_perf(return_raw=True)
            for res_name, df in sess_res.items():
                if not ("subj_N" in df.columns or "subj_N" in df.index.names):
                    df["subj_N"] = self.subj_N
                if not ("sess_N" in df.columns or "sess_N" in df.index.names):
                    df["sess_N"] = sess_N
            sessions_res[sess_N] = sess_res

        res_names = sessions_res[list(sessions_res.keys())[0]].keys()
        sessions_res = {
            res_name: pd.concat(
                [sess_res[res_name] for sess_res in sessions_res.values()]
            )
            for res_name in res_names
        }

        # if agg == "session":
        #     # Group by session number and calculate mean for each session
        #     res["overall_acc"] = (
        #         res["overall_acc"].groupby("sess_N").mean().reset_index()
        #     )

        #     res["acc_by_patt"].groupby('pattern')['mean'].mean().reset_index()
        #     res["raw_cleaned"].groupby("pattern")['correct'].mean().reset_index()

        #     res["acc_by_patt"].groupby(['pattern', "sess_N"]).mean()

        #     res["acc_by_patt"].groupby(["sess_N"])["mean"].describe()

        #     res["overall_acc"]["mean"].describe()
        #     res["overall_rt"]["mean"].describe()
        #     res["acc_by_patt"].groupby(["sess_N", "pattern"])["mean"].describe()
        #     res["rt_by_patt"]["mean"].describe()
        #     res["rt_by_crct"]["mean"].describe()
        #     res["rt_by_crct_and_patt"]["mean"].describe()
        #     res["raw_cleaned"]["mean"].describe()

        # elif agg == "subject":
        #     # Calculate mean across all the subjects' sessions
        #     pass

        raw_cleaned = sessions_res["raw_cleaned"]
        res = {}
        res["overall_acc"] = raw_cleaned["correct"].describe().rename("accuracy")
        res["overall_rt"] = raw_cleaned["rt"].describe()
        res["acc_by_patt"] = raw_cleaned.groupby("pattern")["correct"].describe()
        res["rt_by_patt"] = raw_cleaned.groupby("pattern")["rt"].describe()
        res["rt_by_crct"] = raw_cleaned.groupby("correct")["rt"].describe()
        res["rt_by_crct_and_patt"] = raw_cleaned.groupby(["correct", "pattern"])[
            "rt"
        ].describe()

        return res

    # * ########################################
    # * Represetational Similarity Analysis
    # * ########################################
    def get_erp_rdms(
        self,
        name: str,
        dissimilarity_metric: str = "correlation",
        chan_group: str = "frontal",
        erp_tmin: float = -1.0,
        erp_tmax: float = 0.0,
        erp_events: List[str] = ["a", "x", "m", "l", "invalid", "timeout"],
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
    ):
        # ! TEMP
        # save_dir = WD /'test-export/RDMs-EEG'
        # name = "Rest-EEG"
        # dissimilarity_metric: str = "correlation"
        # chan_group: str = "frontal"
        # erp_tmin:float = -1.0
        # erp_tmax:float = 0.0
        # erp_events :List[str]= ["a", "x", "m", "l", "invalid", "timeout"]
        # save_dir: Path|None = None
        # preprocessed_dir: Path|None = None
        # ! TEMP

        if save_dir is None:
            save_dir = self.export_dir / f"RDMs-EEG/{name}"
        save_dir.mkdir(exist_ok=True, parents=True)

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir

        ds_seq_lvl, ds_patt_lvl = {}, {}

        for sess_N, sess_obj in self.sessions.items():
            # if sess_N==3:break
            # self = self.sessions[1]
            ses_ds_seq_lvl, _, ses_ds_patt_lvl, _ = sess_obj.get_erp_rdms(
                dissimilarity_metric=dissimilarity_metric,
                chan_group=chan_group,
                erp_events=erp_events,
                erp_tmin=erp_tmin,
                erp_tmax=erp_tmax,
                save_dir=save_dir,
                preprocessed_dir=preprocessed_dir,
            )

            # ses_ds_seq_lvl=ses_ds_seq_lvl.get_measurements()
            # ses_ds_patt_lvl=ses_ds_patt_lvl.get_measurements()

            ds_seq_lvl[sess_N], ds_patt_lvl[sess_N] = ses_ds_seq_lvl, ses_ds_patt_lvl

        # ds_seq_lvl = [np.concat(ds_seq_lvl.get_measurements(), axis=0) for ds_seq_lvl in ds_seq_lvl.values()]
        # ds_patt_lvl = Dataset.from_measurements(ds_patt_lvl)

        # TODO: reorder_item_ids

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

    def get_erp_rdms2(
        self,
        dissimilarity_metric: str,
        chan_group: str,
        erp_events: list[str],
        erp_tmin: float,
        erp_tmax: float,
        erp_name: str = "ERP",
        save_dir: Path | None = None,
        save: bool = False,
        preprocessed_dir: Path | None = None,
    ):
        # # ! TEMP
        # dissimilarity_metric = "correlation"
        # chan_group = "frontal"
        # erp_events = ["a", "x", "m", "l", "invalid", "timeout"]
        # erp_tmin = -1.0
        # erp_tmax = 0.0
        # erp_name="ERP"
        # save_dir = WD /'test-export/ERP-RDMs'
        # preprocessed_dir = None
        # # ! TEMP
        ## TODO: implement this: save = {"rdm": True, "rdm_dataset": True, "plots": True}

        subj_N = self.subj_N

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        if save_dir is None:
            save_dir = self.export_dir / f"analyzed/{subj_N:02}"

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        selected_chans = c.EEG_CHAN_GROUPS.get(chan_group, "not found")

        if selected_chans == "not found":
            raise ValueError(
                f"chan_group '{chan_group}' not specified in c.EEG_CHAN_GROUPS"
            )

        sess_epochs, behav_df, bad_sessions = self.get_eeg_epochs(
            selected_chans=selected_chans,
            erp_events=erp_events,
            erp_tmin=erp_tmin,
            erp_tmax=erp_tmax,
            save_dir=None,
            preprocessed_dir=preprocessed_dir,
            format="np",
            epochs_name=erp_name,
        )

        info = mne.create_info(
            ch_names=selected_chans, sfreq=self.eeg_info["sfreq"], ch_types="eeg"
        )
        eeg_epochs = mne.concatenate_epochs(
            [mne.EpochsArray(e, info, verbose="WARNING") for e in sess_epochs.values()],
            verbose="WARNING",
        )
        eeg_epochs.set_montage(c.EEG_MONTAGE)
        del sess_epochs

        eeg_data_seq_lvl = eeg_epochs.get_data()
        del eeg_epochs

        # * Reorder the data
        reordered_inds = reorder_item_ids(
            original_order_df=behav_df,
            new_order_df=c.ITEM_ID_SORT[["pattern", "item_id"]],
        )
        behav_df = behav_df.iloc[reordered_inds]
        behav_df.reset_index(drop=True, inplace=True)
        eeg_data_seq_lvl = eeg_data_seq_lvl[reordered_inds]

        # * ----------------------------------------
        # * Sequence level analysis
        # * ----------------------------------------
        base_fname = f"human-subj_{subj_N:02}-sequence_lvl.hdf5"
        ds_fpath = save_dir / f"dataset-{base_fname}"
        rdm_fpath = save_dir / f"rdm-{base_fname}"

        ds_seq_lvl, rdm_seq_lvl = get_ds_and_rdm(
            measurements=eeg_data_seq_lvl.reshape(eeg_data_seq_lvl.shape[0], -1),
            dissimilarity_metric=dissimilarity_metric,
            ds_fpath=ds_fpath,
            rdm_fpath=rdm_fpath,
            # descriptors={"subj_N": [subj_N], "chan_group": [chan_group]},
            descriptors={"id": [subj_N], "chan_group": [chan_group]},
            obs_descriptors={
                "item_ids": list(behav_df["item_id"]),
                "patterns": list(behav_df["pattern"]),
                "sessions": list(behav_df["sessions"]),
            },
        )

        # * ----------------------------------------
        # * Pattern level analysis
        # * ----------------------------------------
        eeg_data_patt_lvl = {p: [] for p in c.PATTERNS}

        for i, patt in behav_df["pattern"].items():
            eeg_data_patt_lvl[patt].append(eeg_data_seq_lvl[i])

        eeg_data_patt_lvl = np.array([np.array(v) for v in eeg_data_patt_lvl.values()])
        eeg_data_patt_lvl = np.nanmean(eeg_data_patt_lvl, axis=1)

        base_fname = f"human-subj_{subj_N:02}-pattern_lvl.hdf5"
        ds_fpath = save_dir / f"dataset-{base_fname}"
        rdm_fpath = save_dir / f"rdm-{base_fname}"

        ds_patt_lvl, rdm_patt_lvl = get_ds_and_rdm(
            measurements=eeg_data_patt_lvl.reshape(eeg_data_patt_lvl.shape[0], -1),
            dissimilarity_metric=dissimilarity_metric,
            ds_fpath=ds_fpath,
            rdm_fpath=rdm_fpath,
            # descriptors={"subj_N": [subj_N]},
            descriptors={"id": [subj_N]},
            obs_descriptors={"patterns": c.PATTERNS},
        )

        return ds_seq_lvl, rdm_seq_lvl, ds_patt_lvl, rdm_patt_lvl

    def plot_generated_rdm(self, rdms_dir):
        # * ----------------------------------------
        # * Plotting
        # * ----------------------------------------
        # * Plot the RDM and save the Sequence-level figure
        # fig, ax = plot_rdm(rdm_seq_lvl, "patterns", True)
        # ax.set_title(f"RDM - subj {subj_N:02} - sequence level")
        # fig.savefig(rdm_fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.close()

        # # * Plot the RDM and save the figure
        # fig, ax = plot_rdm(rdm_patt_lvl, "patterns", False)
        # ax.set_title(f"RDM - subj {subj_N:02} - pattern level \n all chans")
        # fig.savefig(rdm_fpath.with_suffix(".png"), dpi=300, bbox_inches="tight")
        # plt.close()

        # return ds_seq_lvl, rdm_seq_lvl, ds_patt_lvl, rdm_patt_lvl

        raise NotImplementedError

    @staticmethod
    def apply_nan_template(eeg_data, et_data, behav_data):
        """Apply a NaN template to the all processed data (EEG, ET, Behavioral)."""
        raise NotImplementedError

    def get_behav_rdms(self):
        raise NotImplementedError

    def get_eye_tracking_rdms(self):
        raise NotImplementedError

    # * ########################################
    # * Rest of the code
    # * ########################################
    @staticmethod
    def _session_trial_count(sess_result: dict) -> int:
        candidate_counts = []

        sess_frps = sess_result.get("sess_frps", {})
        if isinstance(sess_frps, dict):
            candidate_counts.extend(
                len(sess_frps.get(key, [])) for key in ("sequence", "choices")
            )
        candidate_counts.extend(
            len(sess_result.get(key, []))
            for key in ("gaze_fixation_traces", "eeg_fixation_epochs")
        )

        for key in ("stim_fixation_summary", "fixation_events"):
            table = sess_result.get(key)
            if isinstance(table, pd.DataFrame) and "trial_N" in table.columns:
                trial_numbers = pd.to_numeric(table["trial_N"], errors="coerce")
                if trial_numbers.notna().any():
                    candidate_counts.append(int(trial_numbers.max()) + 1)

        return max(candidate_counts, default=0)

    @staticmethod
    def _session_trial_map(sess_result: dict, trial_offset: int) -> dict[Any, int]:
        trial_values = []
        for key in ("stim_fixation_summary", "fixation_events"):
            table = sess_result.get(key)
            if isinstance(table, pd.DataFrame) and "trial_N" in table.columns:
                trial_values.extend(table["trial_N"].dropna().tolist())
        trial_values = sorted(pd.Series(trial_values).drop_duplicates().tolist())

        trial_map = {}
        for overall_idx, trial_N in enumerate(trial_values):
            numeric_trial_N = pd.to_numeric(pd.Series([trial_N]), errors="coerce").iloc[
                0
            ]
            if pd.notna(numeric_trial_N):
                trial_map[trial_N] = trial_offset + int(numeric_trial_N)
            else:
                trial_map[trial_N] = trial_offset + overall_idx
        return trial_map

    @staticmethod
    def _add_session_trial_columns(
        table: pd.DataFrame,
        sess_N: int,
        trial_map: dict[Any, int],
    ) -> pd.DataFrame:
        """Add session and subject-level trial indices to a session table."""
        table = table.copy()
        if "sess_N" not in table.columns:
            table.insert(0, "sess_N", sess_N)
        else:
            table["sess_N"] = sess_N

        if "trial_N" not in table.columns:
            return table

        table.insert(
            table.columns.get_loc("trial_N"),
            "overall_trial_N",
            table["trial_N"].map(trial_map),
        )
        return table

    @classmethod
    def _concat_session_analysis_results(
        cls,
        sess_results: dict[int, dict],
    ) -> dict[str, Any]:
        """Concatenate session-level analysis outputs into subject-level outputs."""
        concatenated = {
            "sess_frps": {"sequence": [], "choices": []},
            "gaze_fixation_traces": [],
            "eeg_fixation_epochs": [],
            "stim_fixation_summary": pd.DataFrame(),
            "fixation_events": pd.DataFrame(),
            "gaze_info": pd.DataFrame(),
            "gaze_target_fixation_sequence": pd.DataFrame(),
        }

        stim_fixation_summary_tables = []
        fixation_event_tables = []
        trial_offset = 0

        for sess_N, sess_result in sorted(sess_results.items()):
            if not sess_result:
                continue
            trial_map = cls._session_trial_map(sess_result, trial_offset)
            trial_offset += cls._session_trial_count(sess_result)

            sess_frps = sess_result.get("sess_frps", {})
            concatenated["sess_frps"]["sequence"].extend(
                sess_frps.get("sequence", [])
            )
            concatenated["sess_frps"]["choices"].extend(sess_frps.get("choices", []))
            concatenated["gaze_fixation_traces"].extend(
                sess_result.get("gaze_fixation_traces", [])
            )
            concatenated["eeg_fixation_epochs"].extend(
                sess_result.get("eeg_fixation_epochs", [])
            )

            stim_fixation_summary = sess_result.get("stim_fixation_summary")
            if isinstance(stim_fixation_summary, pd.DataFrame):
                stim_fixation_summary = cls._add_session_trial_columns(
                    stim_fixation_summary,
                    sess_N=sess_N,
                    trial_map=trial_map,
                )
                if not stim_fixation_summary.empty:
                    stim_fixation_summary_tables.append(stim_fixation_summary)

            fixation_events = sess_result.get("fixation_events")
            if isinstance(fixation_events, pd.DataFrame):
                fixation_events = cls._add_session_trial_columns(
                    fixation_events,
                    sess_N=sess_N,
                    trial_map=trial_map,
                )
                if not fixation_events.empty:
                    fixation_event_tables.append(fixation_events)

        if stim_fixation_summary_tables:
            concatenated["stim_fixation_summary"] = pd.concat(
                stim_fixation_summary_tables, ignore_index=True
            )
        if fixation_event_tables:
            concatenated["fixation_events"] = pd.concat(
                fixation_event_tables, ignore_index=True
            )
        concatenated["gaze_info"] = concatenated["stim_fixation_summary"]
        concatenated["gaze_target_fixation_sequence"] = concatenated["fixation_events"]

        return concatenated
    def analyze_sessions(
        self,
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
        force_preprocess: bool = False,
        reuse_ica: bool = True,
        raise_error: bool = True,
        pbar: bool = True,
        concatenate: bool = False,
        frp_baseline: tuple[float | None, float | None] | None = None,
    ):
        should_save = save_dir is not None
        if should_save:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        """Analyze all sessions for the subject."""
        sess_results = {}
        sessions_iter = tqdm(
            self.sessions.items(),
            desc=f"Analyzing subj {self.subj_N:02} sessions",
            total=len(self.sessions),
            disable=not pbar,
        )
        trial_pbar = tqdm(
            total=0,
            desc="Session trials",
            leave=False,
            disable=not pbar,
        )
        try:
            for sess_N, sess_obj in sessions_iter:
                sess_save_dir = save_dir / f"sess_{sess_N:02}" if should_save else None
                sess_results[sess_N] = sess_obj.analyze_session(
                    save_dir=sess_save_dir,
                    preprocessed_dir=preprocessed_dir,
                    force_preprocess=force_preprocess,
                    reuse_ica=reuse_ica,
                    raise_error=raise_error,
                    pbar=False,
                    trial_pbar=trial_pbar,
                    frp_baseline=frp_baseline,
                )
        finally:
            trial_pbar.close()

        if concatenate:
            return self._concat_session_analysis_results(sess_results)
        return sess_results

    def get_gaze_heatmaps_from_fixation_data(self, data_dir: Path, save_dir: Path):
        raise NotImplementedError
        # # ! TEMP
        # # data_dir = c.EXPORT_DIR / f"analyzed/subj_lvl"
        # # save_dir = c.EXPORT_DIR / "analyzed/gaze_heatmaps"
        # # ! TEMP

        # def get_sess_heatmaps(data_dir: Path, subj_N: int, sess_N: int):
        #     # ! TEMP
        #     # subj_N = 4
        #     # sess_N = 5
        #     # # ! TEMP

        #     try:
        #         fixation_file = (
        #             data_dir / f"subj_{subj_N:02}/sess_{sess_N:02}/gaze_info.pkl"
        #         )

        #         fixation_data = load_pickle(fixation_file)
        #         behav_data = load_and_clean_behav_data(c.DATA_DIR, subj_N, sess_N)

        #         heatmap_data = {p: [] for p in c.PATTERNS}
        #         sequence_icon_inds = list(range(0, 8))

        #         for trial in behav_data["trial_N"].unique():
        #             pattern = behav_data.query("trial_N == @trial")["pattern"].iloc[0]

        #             trial_data = fixation_data.query(
        #                 f"trial_N == {trial} & target_ind in {sequence_icon_inds}"
        #             )

        #             trial_data = trial_data.merge(
        #                 behav_data[["trial_N", "pattern", "item_id"]], on="trial_N"
        #             )

        #             sequence_heatmap = np.zeros(len(sequence_icon_inds))

        #             if trial_data.shape[0] == 0:
        #                 sequence_heatmap[:] = np.nan
        #             else:
        #                 target_inds = trial_data["target_ind"].values
        #                 target_count = trial_data["count"].values
        #                 sequence_heatmap[target_inds] = target_count

        #             heatmap_data[pattern].append(sequence_heatmap)

        #         heatmap_data = {p: np.array(h) for p, h in heatmap_data.items()}

        #         avg_heatmaps = {
        #             p: np.nanmean(h, axis=0) for p, h in heatmap_data.items()
        #         }

        #     except Exception as E:
        #         print(f"ERROR ENCOUNTERED: subj_{subj_N:02} sess_{sess_N:02}")
        #         print(E)
        #         heatmap_data, avg_heatmaps = None, None

        #     return heatmap_data, avg_heatmaps

        # def get_subj_heatmaps(data_dir: Path, subj_N: int):
        #     # ! TEMP
        #     # subj_N = 1
        #     # ! TEMP

        #     subj_dir = data_dir / f"subj_{subj_N:02}"
        #     sess_dirs = sorted(
        #         [
        #             d
        #             for d in subj_dir.glob("*")
        #             if d.is_dir() and not d.name.startswith(".")
        #         ]
        #     )
        #     sess_Ns = [int(d.name.split("_")[1]) for d in sess_dirs]

        #     _subj_heatmaps = []
        #     # _avg_subj_heatmaps = []
        #     for sess_N in sess_Ns:
        #         sess_heatmaps, avg_sess_heatmaps = get_sess_heatmaps(
        #             data_dir, subj_N, sess_N
        #         )
        #         _subj_heatmaps.append(sess_heatmaps)
        #         # _avg_subj_heatmaps.append(avg_sess_heatmaps)

        #     subj_heatmaps = {p: [] for p in c.PATTERNS}
        #     # avg_subj_heatmaps = {p:[] for p in c.PATTERNS}

        #     for p in c.PATTERNS:
        #         for i in range(len(_subj_heatmaps)):
        #             sess_heatmaps = _subj_heatmaps[i]
        #             if sess_heatmaps is not None:
        #                 subj_heatmaps[p].append(sess_heatmaps[p])
        #             # avg_subj_heatmaps[p].append(_avg_subj_heatmaps[i][p])

        #     subj_heatmaps = {p: np.concatenate(v) for p, v in subj_heatmaps.items()}
        #     avg_subj_heatmaps = {
        #         p: np.nanmean(v, axis=0) for p, v in subj_heatmaps.items()
        #     }

        #     return subj_heatmaps, avg_subj_heatmaps

        # def get_all_subjs_heatmaps(data_dir: Path):
        #     subj_dirs = sorted(
        #         [
        #             d
        #             for d in data_dir.glob("*")
        #             if d.is_dir() and not d.name.startswith(".")
        #         ]
        #     )
        #     subj_Ns = [int(d.name.split("_")[1]) for d in subj_dirs]

        #     subjects_heatmaps, avg_subjects_heatmaps = {}, {}
        #     for subj_N in tqdm(subj_Ns):
        #         subj_heatmaps, avg_subj_heatmaps = get_subj_heatmaps(data_dir, subj_N)

        #         subjects_heatmaps[subj_N] = subj_heatmaps
        #         avg_subjects_heatmaps[subj_N] = avg_subj_heatmaps

        #     return subjects_heatmaps, avg_subjects_heatmaps

        # def sort_dict(dct, cstm_sort):
        #     return {k: dct[k] for k in cstm_sort if k in dct.keys()}

        # save_dir.mkdir(exist_ok=True, parents=True)
        # save_fig_params = c.SAVE_FIG_PARAMS

        # # * Get the Heatmaps over Sequence Icons
        # subjects_heatmaps, avg_subjects_heatmaps = get_all_subjs_heatmaps(data_dir)

        # subjects_heatmaps = {f"subj_{k:02}": v for k, v in subjects_heatmaps.items()}
        # avg_subjects_heatmaps = {
        #     f"subj_{k:02}": v for k, v in avg_subjects_heatmaps.items()
        # }

        # subjects_heatmaps = {
        #     k: sort_dict(dct, c.PATTERNS) for k, dct in subjects_heatmaps.items()
        # }
        # avg_subjects_heatmaps = {
        #     k: sort_dict(dct, c.PATTERNS) for k, dct in avg_subjects_heatmaps.items()
        # }

        # avg_subjects_heatmaps_arr = {
        #     k: np.array(list(heatmap_dict.values()))
        #     for k, heatmap_dict in avg_subjects_heatmaps.items()
        # }

        # avg_heatmap_arr = np.nanmean(list(avg_subjects_heatmaps_arr.values()), axis=0)

        # # avg_subjects_heatmaps["human_avg"] = dict(zip(c.PATTERNS, avg_heatmap_arr))
        # avg_subjects_heatmaps_arr["human_avg"] = avg_heatmap_arr

        # for subj_id, heatmap_arr in avg_subjects_heatmaps_arr.items():
        #     heatmap_fpath = save_dir / f"gaze_heatmap-{subj_id}.npy"

        #     np.save(heatmap_fpath, heatmap_arr)

        #     fig, ax = plt.subplots()
        #     heatmap_fig = ax.imshow(heatmap_arr)
        #     ax.set_yticks(range(len(c.PATTERNS)), c.PATTERNS)
        #     # ax.set_title(f"Gaze Heatmap - {subj_id}")
        #     plt.colorbar(heatmap_fig)
        #     fig.savefig(heatmap_fpath.with_suffix(".png"), **save_fig_params)
        #     plt.close()

    def get_stim_flash_order(
        self, target_stim: str | None = None
    ) -> Dict[str, pd.DataFrame]:
        sess_stim_locs = []
        for _, sess_obj in sorted(self.sessions.items()):
            try:
                sess_stim_locs.append(
                    sess_obj.get_stim_flash_order(target_stim=target_stim)
                )
            except ValueError as exc:
                if target_stim is None:
                    raise
                # Allow missing target in a specific session and aggregate available ones.
                print(
                    f"WARNING: {exc}. Skipping subj_{self.subj_N:02}, "
                    f"sess_{sess_obj.sess_N:02}."
                )

        if len(sess_stim_locs) == 0:
            if target_stim is not None:
                raise ValueError(
                    f"`target_stim` '{target_stim}' was not found in any session "
                    f"for subj_{self.subj_N:02}."
                )
            return {}

        all_stim_locs_df = pd.concat(sess_stim_locs, ignore_index=True)
        all_stim_locs_df.sort_values(
            ["subj_N", "sess_N", "trial_N"], inplace=True, ignore_index=True
        )

        all_stim_locs = {
            stim: df.reset_index(drop=True)
            for stim, df in all_stim_locs_df.groupby("stim", sort=True)
        }

        if target_stim is not None and target_stim not in all_stim_locs:
            raise ValueError(
                f"`target_stim` '{target_stim}' was not found in any session "
                f"for subj_{self.subj_N:02}."
            )

        # TODO: messy, clean this, but maintain final version of all_stim_locs as dataframe
        all_stim_locs = pd.concat(all_stim_locs.values()).reset_index(drop=True)

        return all_stim_locs

    def get_stim_flash_eeg_epochs(
        self,
        target_stim: str | None = None,
        silence: bool = True,
    ):
        # Choose the context based on the silence flag
        ctx = (
            contextlib.redirect_stdout(io.StringIO())
            if silence
            else contextlib.nullcontext()
        )
        stim_epochs = {}
        with ctx:
            for sess_N, sess_obj in tqdm(self.sessions.items()):
                sess_stim_epochs = sess_obj.get_stim_flash_eeg_epochs(
                    target_stim=target_stim
                )

                for stim, epochs in sess_stim_epochs.items():
                    if stim not in stim_epochs:
                        stim_epochs[stim] = []
                    stim_epochs[stim].append(epochs)

            for stim, epochs in stim_epochs.items():
                stim_epochs[stim] = mne.concatenate_epochs(epochs)

        return stim_epochs
