from ._common import *
from .base import HumanDataClass
from .subject import HumanSubjData


@dataclass
class HumanGroupData(HumanDataClass):
    subj_Ns: List[int] | None = None
    subjects: Dict[int, HumanSubjData] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.behav_data = None
        self.eeg_data = None
        self.et_data = None

        if self.subj_Ns is None:
            try:
                self.subj_Ns = self.search_subj_Ns()
            except Exception as e:
                raise ValueError(
                    "Couldn't find subject directories, make sure they exist at: "
                    f"{self.data_dir}.\nError Details: {e}"
                )

        for subj_N in self.subj_Ns:
            self.subjects[subj_N] = HumanSubjData(
                data_dir=self.data_dir,
                preprocessed_dir=self.preprocessed_dir,
                export_dir=self.export_dir,
                subj_N=subj_N,
            )

    def search_subj_Ns(self):
        if self.data_fmt == "bids":
            subj_Ns = [
                int(f.name.split("-")[1])
                for f in list_contents(self.data_dir, incl="folder", recurs=False)
            ]
        else:
            subj_Ns = [
                int(f.name.split("_")[1])
                for f in list_contents(self.data_dir, incl="folder", recurs=False)
            ]
        return subj_Ns

    def show_dir_struct(self, stringRep: bool = False, **kwargs):
        root = self.data_dir
        print(f"Human Data Root Directory Structure:")
        struct = DisplayTree(root, stringRep=stringRep, **kwargs)
        print(f"\nPath: {root}")
        if stringRep:
            return struct

    def extract_eeg_metadata(self, pbar: bool = True):
        metadata = {}
        for subj_N, subj_obj in tqdm(self.subjects.items(), disable=not pbar):
            metadata[subj_N] = subj_obj.extract_eeg_metadata()

        return metadata

    # * ########################################
    # * Eye Tracking Data
    # * ########################################

    # * ########################################
    # * EEG Data
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
    ):
        # # ! TEMP
        # chan_group = "frontal"
        # selected_chans = c.EEG_CHAN_GROUPS[chan_group]
        # erp_tmin = -1.0
        # erp_tmax = 0.0
        # erp_events = ["a", "x", "m", "l", "invalid", "timeout"]
        # save_dir = WD / "test-export/ERP-RDMs"
        # preprocessed_dir = None
        # format = "mne"
        # trials_per_sess = 80
        # epochs_name = "epochs"
        # combine = False
        # # ! TEMP

        for subj_N, subj_obj in self.subjects.items():
            # pass
            sess_epochs, behav_df, bad_sessions = subj_obj.get_eeg_epochs(
                selected_chans=selected_chans,
                erp_tmin=erp_tmin,
                erp_tmax=erp_tmax,
                erp_events=erp_events,
                save_dir=save_dir,
                preprocessed_dir=preprocessed_dir,
                format=format,
                trials_per_sess=trials_per_sess,
                epochs_name=epochs_name,
                combine=combine,
            )

        evoked_files = list_contents(save_dir, reg=r".+-evoked.pkl$")

        evoked_data = [read_file(f) for f in evoked_files]
        evoked_by_patt = {k: [] for k in evoked_data[0].keys()}

        for subj_evoked in evoked_data:
            for patt, evoked in subj_evoked.items():
                evoked_by_patt[patt].extend([evoked])

        evoked_by_patt = {
            patt: mne.combine_evoked(e, "equal") for patt, e in evoked_by_patt.items()
        }

        evoked_by_patt_avg = {}
        times = evoked_by_patt[list(evoked_by_patt.keys())[0]].times
        for patt, evoked in evoked_by_patt.items():
            fig, ax = plt.subplots(figsize=(10, 4), dpi=500)
            evoked_plot = evoked.plot(axes=ax, show=False)
            main_ax, topo_ax = evoked_plot.get_axes()
            # * Remove the "N_ave" label at the top right of the plot & set title
            [text.remove() for text in main_ax.texts]
            main_ax.set_title(patt)

            # topo_pos = [getattr(topo_ax.get_position(), i) for i in ['x0', "y0", "x1", 'y1']]
            # topo_pos = [getattr(topo_ax.get_position(), i) for i in ['x0', "y0", "width", 'height']]
            # topo_pos = [*topo_pos[:2], 0.2, 0.2]
            # topo_ax.set_position(topo_pos)

            plt.show()
            plt.close()

            evoked_by_patt_avg[patt] = evoked.get_data().mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=500)
        for patt, evoked in evoked_by_patt_avg.items():
            _evoked = evoked * 1e6
            ax.plot(times, _evoked, label=patt)

        y0, y1 = ax.get_ylim()
        y0, y1 = np.floor(y0), np.ceil(y1)
        ax.set_ylim(y0, y1)
        ax.set_ylabel("µV")
        yticks = np.arange(y0, y1 + 1)
        ax.set_yticks(yticks, [str(int(i)) for i in yticks])
        ax.set_xlim(times[0], times[-1])

        ax.legend()
        plt.show()

    def check_data(self, remove_practice: bool = True) -> None:
        raise NotImplementedError

    def get_sess_info(self):
        sessions_info = {}
        for sub_N, subj_obj in self.subjects.items():
            sessions_info[sub_N] = subj_obj.get_sess_info()
        return sessions_info

    # * ########################################
    # * Behavioral Data
    # * ########################################
    def get_behav_data(self):
        # TODO: rename to "get_behav_data()" to be consistent
        if self.behav_data is None:
            behav_data = {}
            for subj_N, subj_obj in self.subjects.items():
                behav_data[subj_N] = subj_obj.get_behav_data(combine=True)
            behav_data = pd.concat(behav_data.values())
            behav_data.sort_values(["subj_N", "sess_N", "trial_N"], inplace=True)
            behav_data.reset_index(drop=True, inplace=True)
            return behav_data
        else:
            return self.behav_data

    def summarize_behav(
        self,
    ) -> Tuple[
        pd.DataFrame,
        "plotly.graph_objs.Figure",
        "plotly.graph_objs.Figure",
        "plotly.graph_objs.Figure",
    ]:
        """
        Returns:
            combined: per-subject summary with acc_* and rt_* columns
            acc_fig:  bar chart of accuracy (mean ± sd)
            rt_fig:   bar chart of RT (mean ± sd)
            scatter:  accuracy vs RT scatter
        """
        import plotly.express as px

        behav_df = self.get_behav_data().copy()
        # ensure dtypes
        behav_df["correct"] = behav_df["correct"].astype(int)

        # single pass groupby with named aggregations
        g = behav_df.groupby("subj_N", as_index=False).agg(
            acc_count=("correct", "size"),
            acc_mean=("correct", "mean"),
            acc_std=("correct", "std"),
            rt_count=("rt", "size"),
            rt_mean=("rt", "mean"),
            rt_std=("rt", "std"),
        )

        # std is NaN for n=1; set to 0 so error bars don't break
        g[["acc_std", "rt_std"]] = g[["acc_std", "rt_std"]].fillna(0.0)

        # sort order (by accuracy mean); use same order for both bars
        subj_order = g.sort_values("acc_mean")["subj_N"].tolist()

        # plots (return figs; let caller decide to .show())
        acc_fig = px.bar(
            g,
            x="subj_N",
            y="acc_mean",
            error_y="acc_std",
            title="Accuracy",
            category_orders={"subj_N": subj_order},
            labels={"subj_N": "Subject", "acc_mean": "Accuracy (proportion)"},
        )
        rt_fig = px.bar(
            g,
            x="subj_N",
            y="rt_mean",
            error_y="rt_std",
            title="Response Time",
            category_orders={"subj_N": subj_order},
            labels={"subj_N": "Subject", "rt_mean": "RT (s)"},
        )
        scatter = px.scatter(
            g,
            x="rt_mean",
            y="acc_mean",
            trendline="ols",
            labels={"rt_mean": "RT (s)", "acc_mean": "Accuracy"},
            title="Accuracy vs. Response Time",
        )
        for fig in [acc_fig, rt_fig, scatter]:
            fig.show()

        # Build a two-level column index: ('acc'|'rt') x ('count','mean','std')
        acc = (
            g[["subj_N", "acc_count", "acc_mean", "acc_std"]]
            .set_index("subj_N")
            .rename(
                columns={"acc_count": "count", "acc_mean": "mean", "acc_std": "std"}
            )
        )
        rt = (
            g[["subj_N", "rt_count", "rt_mean", "rt_std"]]
            .set_index("subj_N")
            .rename(columns={"rt_count": "count", "rt_mean": "mean", "rt_std": "std"})
        )

        combined_stats = pd.concat({"acc": acc, "rt": rt}, axis=1)

        # optional: enforce column order within each block
        combined_stats = combined_stats.reindex(
            columns=pd.MultiIndex.from_product(
                [["acc", "rt"], ["count", "mean", "std"]]
            )
        )

        return combined_stats, acc_fig, rt_fig, scatter

    def make_behav_group_subj(self, save: bool = True):
        behav_data = self.get_behav_data()

        corr_and_rt = behav_data.groupby("item_id")[["correct", "rt"]].mean()

        col_filter = (
            ["item_id", "solution_key"]
            + [f"figure{i}" for i in range(1, 9)]
            + [f"choice{i}" for i in range(1, 5)]
            + ["masked_idx", "seq_order", "choice_order", "trial_type"]
        )

        behav_data[["item_id", "solution_key"]]
        group_behav = behav_data[col_filter].drop_duplicates()
        group_behav = group_behav.merge(corr_and_rt, on="item_id", how="inner")

        if save:
            raise NotImplementedError("`save` logic not yet implemented")
        #     save_dir = self.export_dir / "analyzed/group"
        #     save_dir.mkdir(exist_ok=True, parents=True)

    # * ########################################
    # * Rest of the code
    # * ########################################
    def get_trials_data(
        self,
        preprocessed_dir: Path | None = None,
        raise_error: bool = False,
        pbar: bool = True,
    ):
        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir
        preprocessed_dir = Path(preprocessed_dir)

        behav, manual_et_trials, manual_eeg_trials = {}, {}, {}

        for subj_N, subj_obj in tqdm(self.subjects.items()):
            beh, et, eeg = subj_obj.get_trials_data(
                preprocessed_dir=preprocessed_dir,
                raise_error=raise_error,
                pbar=pbar,
            )
            behav[subj_N] = beh
            manual_et_trials[subj_N] = et
            manual_eeg_trials[subj_N] = eeg

        return behav, manual_et_trials, manual_eeg_trials

    def get_frp_data(self):
        """Get the FRP RDMs for all subjects in the group."""
        raise NotImplementedError

    def get_frp_rdms(self):
        """Get the FRP RDMs for all subjects in the group."""
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def get_erp_rdms(
        self,
        dissimilarity_metric,
        chan_group,
        erp_events,
        erp_tmin,
        erp_tmax,
        save_dir,
        preprocessed_dir,
    ):
        # ! TEM
        # dissimilarity_metric="correlation"
        # chan_group="frontal"
        # erp_events=["a", "x", "m", "l", "invalid", "timeout"]
        # erp_tmin=-1.0
        # erp_tmax=0.0
        # save_dir=WD / "test-export/ERP-RDMs"
        # preprocessed_dir=None
        # ! TEMP

        def to_be_ported():
            for subj_N in tqdm(subjects):
                # * Group Level Analysis
                # * Sequence Level
                for level in ["sequence", "pattern"]:
                    ds_files = sorted(
                        save_dir.glob(f"dataset-*subj_*-{level}_lvl.hdf5")
                    )

                    datasets = [
                        rsatoolbox.data.dataset.load_dataset(f, file_type="hdf5")
                        for f in ds_files
                    ]

                    # datasets = {
                    #     f.name:rsatoolbox.data.dataset.load_dataset(f, file_type="hdf5") for f in ds_files
                    # }

                    obs_descriptors = datasets[0].obs_descriptors

                    group_ds = [ds.get_measurements() for ds in datasets]

                    if level == "sequence":
                        group_ds = [ds for ds in group_ds if ds.shape[0] == 400]

                    group_ds: np.ndarray = np.nanmean(group_ds, axis=0)

                    base_fname = f"human-group_avg-{level}_lvl.hdf5"
                    ds_fpath = save_dir / f"dataset-{base_fname}"
                    rdm_fpath = save_dir / f"rdm-{base_fname}"

                    ds, rdm = get_ds_and_rdm(
                        measurements=group_ds,
                        dissimilarity_metric=dissimilarity_metric,
                        ds_fpath=ds_fpath,
                        rdm_fpath=rdm_fpath,
                        # descriptors={"subj_N": ["group"], "chan_group": [chan_group]},
                        descriptors={"id": ["group"], "chan_group": [chan_group]},
                        obs_descriptors=obs_descriptors,
                    )

                    # * Plot the RDM and save the figure
                    if level == "sequence":
                        fig, ax = plot_rdm(rdm, "patterns", True)
                    else:
                        fig, ax = plot_rdm(rdm, "patterns", False)
                    # ax.set_title(f"RDM - group average - {level} level")
                    fig.savefig(
                        rdm_fpath.with_suffix(".png"), dpi=300, bbox_inches="tight"
                    )
                    plt.close()

        raise NotImplementedError("Group-level ERP RDMs are not implemented yet.")

    def get_stim_locs(self):
        raise NotImplementedError

    def get_stim_erp(self, subj_N):
        raise NotImplementedError

    def get_subj_info(self):
        raise NotImplementedError
