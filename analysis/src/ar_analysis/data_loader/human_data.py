# * ########################################
# * IMPORTS
# * ########################################
from pathlib import Path
import json
import re
import contextlib
import shutil
from matplotlib import pyplot as plt
import mne
from mne_icalabel import label_components
from mne.preprocessing.eyetracking import read_eyelink_calibration
import numpy as np
import pandas as pd
import tensorpac
import io
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
import plotly.express as px
from directory_tree import DisplayTree
import rsatoolbox
from datetime import timedelta
import textwrap
from box import Box
from loguru import logger
# from icecream import ic
# from pprint import pprint

# * ########################################
# * RELATIVE IMPORTS
# * ########################################
from ar_analysis.analysis_config import Config as c
from ar_analysis.paths import CONFIG_DIR
from ar_analysis.utils.custom_type_hints import DATA_FMTS

from ar_analysis.analysis_plotting import (
    create_video_from_frames,
    get_gaze_heatmap,
    plot_eeg_and_gaze_fixations,
    prepare_eeg_data_for_plot,
    plot_rdm,
)
from ar_analysis.utils.analysis_utils import (
    check_ch_groups,
    get_trial_info,
    locate_trials,
    resample_eye_tracking_data,
    set_eeg_montage,
    save_pickle,
    read_file,
    reorder_item_ids,
    list_contents,
)
from ar_analysis.analysis_rsa import get_ds_and_rdm

# * ########################################
# * GLOBAL VARIABLES
# * ########################################
CURRENT_DATA_FMT = c.CURRENT_DATA_FMT
TASK_NAME = c.TASK_NAME


# * ########################################
# * DATA LOADERS
# * ########################################
@dataclass
class HumanDataClass:
    data_dir: Path
    preprocessed_dir: Path
    export_dir: Path
    data_fmt: DATA_FMTS = field(default=CURRENT_DATA_FMT, kw_only=True)
    task_name: str = field(default=TASK_NAME, kw_only=True)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.EEG = {"sfreq": c.EEG_SFREQ}

        if self.data_fmt not in ["bids", "original"]:
            raise ValueError(f"`self.data_fmt` must be set to {DATA_FMTS}")

    # def load_and_clean_behav_data():
    #     raise NotImplementedError
    @staticmethod
    def get_eeg_montage():
        return mne.channels.make_standard_montage("biosemi64")

    @staticmethod
    def extract_subj_sess(s: str | Path):
        return re.findall(r"subj_(\d{2})(\d{2})", str(s))[0]

    @staticmethod
    def interactive_erp_plot(erp: mne.Evoked) -> None:
        fig = px.line(
            erp.to_data_frame().set_index("time"),
            labels={"value": "Amplitude", "index": "Time (samples)"},
        )
        fig.update_layout(legend_title_text="Channel")
        fig.show()


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
    def find_file(
        self,
        folder: Path,
        pat: str,
        search_label: str | None = None,
        dotfile: bool = False,
    ) -> Path:
        # TODO: move to base class
        subj_N, sess_N = self.subj_N, self.sess_N

        def filter_file(f):
            if not dotfile:
                return f.is_file() and not f.name.startswith(".")
            else:
                return f.is_file()

        files = [f for f in folder.glob(pat) if filter_file(f)]

        if search_label is not None:
            error_txt = f"[Error when searching for {search_label} file]\n"
        else:
            error_txt = ""

        if len(files) > 1:
            files_str = "\n- ".join([str(f) for f in files])
            raise ValueError(f"{error_txt}More than one files matched:{files_str}")

        try:
            file = files[0]
        except IndexError:
            raise FileNotFoundError(
                f"{error_txt}\n\tFile for subj {subj_N}, sess {sess_N} "
                f"with pattern `{pat}` not found in: \n\t{folder}"
            )
        return file

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

        raw.set_annotations(annotations)

        raw.filenames = [physio_tsv]

        return raw

    def get_sess_info(self, fmt: DATA_FMTS | None = None) -> Dict:
        fmt = self.data_fmt if fmt is None else fmt
        if fmt == "bids":
            fpath = self._search_res_file(
                directory=self.data_dir,
                regex=f".*sub-{self.subj_N:02}.*_sessions.tsv$",
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

    def preprocess_et_data(self) -> Tuple[mne.io.eyelink.eyelink.RawEyelink, list]:
        raw_et = self.get_raw_et_data()
        return raw_et  # , et_cals

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

    def extract_eeg_metadata(self) -> Dict[str, Any]:
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
        except:
            events_counts = {}
        try:
            ch_positions = eeg_data._get_channel_positions().tolist()
        except:
            ch_positions = []
        try:
            dig_montage = ([str(i) for i in eeg_data.info["dig"]],)
        except:
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

    def extract_et_metadata(self):  # -> Dict[str, Any]:
        # if self.data_fmt == 'bids':
        #     metadata = None
        # else:
        #     metadata = None
        # return metadata
        raise NotImplementedError

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
        prepro_et_data = self.get_raw_et_data()
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

            if n_eeg_epochs != n_et_epochs != n_behav_trials:
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
        preprocessed_dir: Path,
        raise_error: bool = False,
        eeg_incomplete: Literal["allow", "error", "skip"] = "error",
    ):
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
    def analyze_trial_decision_period(
        self,
        eeg_trial: mne.io.Raw,
        et_trial: mne.io.eyelink.eyelink.RawEyelink,
        raw_behav: pd.DataFrame,
        trial_N: int,
        eeg_baseline: float = 0.100,
        eeg_window: float = 0.600,
        show_plots: bool = True,
        pbar_off=True,
    ):
        """
        This function uses the Eye Tracker's label to identify fixation events
        """

        # # ! TEMP
        # eeg_baseline: float = 0.100
        # eeg_window: float = 0.600
        # show_plots = False
        # pbar_off =False

        # s0102 = HumanSessData(c.DATA_DIR, c.PREPROCESSED_DIR, c.EXPORT_DIR, 1, 2)
        # self = s0102
        # sess_info, raw_behav, raw_eeg, raw_et, et_cals = self.get_data()

        # (
        #     manual_et_trials,
        #     *_,
        #     # et_events_dict,
        #     # et_events_dict_inv,
        #     # et_trial_bounds,
        #     # et_trial_events_df,
        # ) = self.split_et_data_into_trials(raw_et, et_cals)

        # (
        #     manual_eeg_trials,
        #     *_,
        #     # eeg_trial_bounds,
        #     # eeg_events,
        #     # eeg_events_df,
        # ) = self.split_eeg_data_into_trials(raw_eeg, raw_behav)

        # bad_chans = raw_eeg.info["bads"]

        # # * Initialize data containers
        # sess_frps: Dict[str, List] = {"sequence": [], "choices": []}
        # fixation_data_all = []
        # eeg_fixation_data_all = []
        # gaze_info_all = []
        # gaze_target_fixation_sequence_all = []
        # eeg_fixation_pac_data_all = []

        # manual_et_trials = list(manual_et_trials)
        # manual_eeg_trials = list(manual_eeg_trials)
        # trial_N = 45
        # eeg_trial = manual_eeg_trials[trial_N]
        # et_trial = manual_et_trials[trial_N]
        # # ! TEMP

        # * ########################################
        # * "GLOBAL" VARIABLES
        # * ########################################

        # * Define frequency bands for Phase-Amplitude Coupling (PAC) analysis
        theta_band = [4, 7]  # Theta band: 4-7 Hz
        alpha_band = [8, 13]  # Alpha band: 8-13 Hz

        # * Get channel indices for PAC analysis
        picked_chs_pac = mne.pick_channels(
            eeg_trial.ch_names,
            include=c.EEG_CHAN_GROUPS.frontal,
            exclude=eeg_trial.info["bads"],
        )

        # * ########################################

        assert c.EEG_SFREQ == eeg_trial.info["sfreq"], (
            "EEG data has incorrect sampling rate"
        )

        # * Crop the data
        et_trial, et_annotations, time_bounds = self.crop_et_trial(et_trial)

        # * Adjust time bounds for EEG baseline and window
        # * Cropping with sample bounds
        trial_duration = (time_bounds[1] + eeg_window + eeg_baseline) - time_bounds[0]
        sample_bounds = [0, 0]
        sample_bounds[0] = int(time_bounds[0] * c.EEG_SFREQ)
        sample_bounds[1] = sample_bounds[0] + int(np.ceil(trial_duration * c.EEG_SFREQ))

        eeg_trial = eeg_trial.copy().crop(
            eeg_trial.times[sample_bounds[0]], eeg_trial.times[sample_bounds[1]]
        )  # TODO: Select only good channels here?

        # * Get channel positions for topomap
        eeg_info = eeg_trial.info

        chans_pos_xy = np.array(
            list(eeg_info.get_montage().get_positions()["ch_pos"].values())
        )[:, :2]

        # * Get info on the current trial
        stim_pos, stim_order, sequence, choices, _, solution, _ = get_trial_info(
            trial_N,
            raw_behav,
            c.X_POS_STIM,
            c.Y_POS_CHOICES,
            c.Y_POS_SEQUENCE,
            c.SCREEN_RESOLUTION,
            c.IMG_SIZE,
        )

        solution_ind = {v: k for k, v in choices.items()}[solution]

        # * Get the onset of the response event
        response_onset = et_annotations.query(
            "description.isin(['a', 'x', 'm', 'l', 'timeout', 'invalid'])"
        ).iloc[0]["onset"]

        seq_and_choices = sequence.copy()
        seq_and_choices.update({k + len(sequence): v for k, v in choices.items()})

        # * Get the indices of the icons, reindex choices to simplify analysis
        sequence_icon_inds = list(sequence.keys())
        choice_icon_inds = [i + len(sequence) for i in choices.keys()]
        solution_ind += len(sequence)
        # wrong_choice_icon_inds = [i for i in choice_icon_inds if i != solution_ind]

        # * Get the types of stimuli (e.g., choice related or unrelated to the sequence)
        stim_types = {}
        for i, icon_name in seq_and_choices.items():
            if i < 7:
                stim_types[i] = "sequence"
            elif i == 7:
                stim_types[i] = "question_mark"
            else:
                if i == solution_ind:
                    stim_types[i] = "choice_correct"
                else:
                    if icon_name in sequence.values():
                        stim_types[i] = "choice_incorrect_related"
                    else:
                        stim_types[i] = "choice_incorrect_unrelated"

        # * Indices of every gaze fixation event
        fixation_inds = et_annotations.query("description == 'fixation'").index

        # * Initialize data containers
        gaze_target_fixation_sequence = []
        fixation_data: dict = {i: [] for i in range(len(stim_order))}
        eeg_fixation_data: dict = {i: [] for i in range(len(stim_order))}
        # eeg_fixation_pac_data: dict = {i: [] for i in range(len(stim_order))}

        # TODO: get heatmap of fixation data
        # * Loop through each fixation event
        pbar = tqdm(fixation_inds, leave=False, disable=pbar_off)

        for idx_fix, fixation_ind in enumerate(pbar):
            # * Get number of flash events before the current fixation; -1 to get the index
            fixation = et_annotations.loc[fixation_ind]

            # * Get fixation start and end time, and duration
            fixation_start = fixation["onset"]
            fixation_duration = fixation["duration"]
            fixation_end = fixation_start + fixation_duration

            # * Make sure we don't go beyond the end of the trial, crop the data if needed
            end_time = min(fixation_end, et_trial.times[-1])

            # * Get gaze positions and pupil diameter during the fixation period
            gaze_x, gaze_y, pupil_diam = (
                et_trial.copy().crop(fixation_start, end_time).get_data()
            )

            # * Determine if gaze is on target
            on_target, target_ind = self.is_fixation_on_target(gaze_x, gaze_y, stim_pos)

            # * Get EEG data during the fixation period
            # * Convert time bounds to sample bounds
            eeg_start_sample = int(fixation_start * c.EEG_SFREQ)
            eeg_end_sample = eeg_start_sample + int(
                np.ceil((eeg_window + eeg_baseline) * c.EEG_SFREQ)
            )

            # * Convert EEG sample bounds back to time bounds
            eeg_start_time = eeg_trial.times[eeg_start_sample]
            eeg_end_time = eeg_trial.times[eeg_end_sample]
            # eeg_duration = eeg_end_time - eeg_start_time

            # ! Epoching on eeg_trial as epoching with MNE epochs object results in a
            # ! lot of epochs being dropped automatically by MNE, so we'll use the raw
            # ! data and crop it manually then convert it to an EpochsArray object
            # * Crop by sample bounds
            eeg_slice = eeg_trial.copy().crop(eeg_start_time, eeg_end_time).get_data()

            eeg_slice = mne.EpochsArray(
                [eeg_slice], eeg_trial.info, tmin=-eeg_baseline, verbose="WARNING"
            )

            # * Apply baseline correction and detrend
            eeg_slice = eeg_slice.apply_baseline(baseline=(None, 0), verbose="WARNING")
            eeg_slice.detrend = 1

            # * Check if fixation is on target and duration is above minimum
            if fixation_duration >= c.MIN_FIXATION_DURATION and on_target:
                # if on_target:
                discarded = False

                fixation_data[target_ind].append(np.array([gaze_x, gaze_y]))

                gaze_target_fixation_sequence.append(
                    [target_ind, fixation_start, fixation_duration, pupil_diam.mean()]
                )

                eeg_fixation_data[target_ind].append(eeg_slice)

            else:
                # * Only for visualization purposes
                discarded = True

            if show_plots:
                # * Select EEG channel groups to plot
                ch_group_names = [
                    "frontal",
                    "parietal",
                    "central",
                    "temporal",
                    "occipital",
                ]
                ch_group_colors = ["red", "green", "blue", "pink", "orange"]

                selected_chans_names, ch_group_inds, group_colors, chans_pos_xy = (
                    prepare_eeg_data_for_plot(
                        c.EEG_CHAN_GROUPS,
                        c.EEG_MONTAGE,
                        c.NON_EEG_CHANS,
                        eeg_trial.info["bads"],
                        ch_group_names,
                        ch_group_colors,
                    )
                )

                title = f"ICON-{target_ind}" if on_target else "OFF-TARGET"
                title += f" ({fixation_duration * 1000:.0f} ms)"
                title += " - " + ("DISCARDED" if discarded else "SAVED")

                # fig = plot_eeg_and_gaze_fixations_plotly(
                plot_eeg_and_gaze_fixations(
                    # * Convert to microvolts
                    # eeg_data=eeg_slice * 1e6,
                    eeg_data=eeg_slice.get_data(picks="eeg", units="uV")[0],  # * 1e6,
                    eeg_sfreq=c.EEG_SFREQ,
                    et_data=np.stack([gaze_x, gaze_y], axis=1).T,
                    eeg_baseline=eeg_baseline,
                    response_onset=response_onset,
                    eeg_start_time=eeg_start_time,
                    eeg_end_time=eeg_end_time,
                    icon_images=c.ICON_IMAGES,
                    img_size=c.IMG_SIZE,
                    stim_pos=stim_pos,
                    chans_pos_xy=chans_pos_xy,
                    ch_group_inds=ch_group_inds,
                    group_colors=group_colors,
                    screen_resolution=c.SCREEN_RESOLUTION,
                    title=title,
                    vlines=[
                        eeg_baseline * c.EEG_SFREQ,
                        eeg_baseline * c.EEG_SFREQ + fixation_duration * c.EEG_SFREQ,
                    ],
                )

                # plt.savefig(
                #     wd
                #     / f"subj_{subj_N:02}-sess_{sess_N:02}-trial_{epoch_N:02}-fixation{idx_fix:02}.png"
                # )
        plt.close("all")

        # * Getting Fixation Related Potentials (FRPs)
        # * FRPs here correspond to the average EEG signal during fixations on each icon

        # * Concatenate all fixations on each icon
        eeg_fixation_data = {
            target_ind: mne.concatenate_epochs(data, verbose="WARNING")
            for target_ind, data in eeg_fixation_data.items()
            if len(data) > 0
        }

        # * Concat all fixations on each icon from the sequence (top row in experiment)
        eeg_fixations_sequence = {
            k: v
            for k, v in eeg_fixation_data.items()
            if k in sequence_icon_inds and len(v) > 0
        }

        # * Concat all fixations on each icon from the choices (bottom row in experiment)
        eeg_fixations_choices = {
            k: v
            for k, v in eeg_fixation_data.items()
            if k in choice_icon_inds and len(v) > 0
        }

        # * Calculate ERPs to fixations on sequence icons
        if len(eeg_fixations_sequence) > 0:
            fixations_sequence_erp = mne.concatenate_epochs(
                list(eeg_fixations_sequence.values()), verbose="WARNING"
            ).average()
        else:
            # fixations_sequence_erp = np.array([])
            fixations_sequence_erp = None

        # * Calculate ERPs to fixations on choice icons
        if len(eeg_fixations_choices) > 0:
            fixations_choices_erp = mne.concatenate_epochs(
                list(eeg_fixations_choices.values()), verbose="WARNING"
            ).average()
        else:
            # fixations_choices_erp = np.array([])
            fixations_choices_erp = None

        # * Compute Phase-Amplitude Coupling (PAC)
        eeg_fixation_pac_data = {}
        # for target_ind, mne_data in eeg_fixation_data.items():
        #     # TODO: check which one to use for phase and amplitude
        #     pac, _ = self.analyze_phase_coupling(
        #         mne_data.get_data(picks=picked_chs_pac),
        #         sfreq=c.EEG_SFREQ,
        #         f_pha=theta_band,
        #         f_amp=alpha_band,
        #     )
        #     eeg_fixation_pac_data[target_ind] = pac

        # * ########################################
        # * GAZE ANALYSIS
        # * ########################################
        cols = ["target_ind", "onset", "duration", "pupil_diam"]

        gaze_target_fixation_sequence_df = pd.DataFrame(
            gaze_target_fixation_sequence,
        )
        del gaze_target_fixation_sequence

        if len(gaze_target_fixation_sequence_df) == 0:
            gaze_target_fixation_sequence_df = pd.DataFrame(
                [[pd.NA for _ in range(len(cols))]]
            )

        gaze_target_fixation_sequence_df.columns = cols

        gaze_target_fixation_sequence_df["trial_N"] = trial_N

        gaze_target_fixation_sequence_df["stim_name"] = (
            gaze_target_fixation_sequence_df["target_ind"].replace(seq_and_choices)
        )

        # gaze_target_fixation_sequence_df["pupil_diam"] = (
        #     gaze_target_fixation_sequence_df["pupil_diam"].round(2)
        # )

        gaze_target_fixation_sequence_df["stim_type"] = (
            gaze_target_fixation_sequence_df["target_ind"].replace(stim_types)
        )

        first_fixation_order = (
            gaze_target_fixation_sequence_df.sort_values("onset")
            .groupby("target_ind")
            .first()["onset"]
            .rank()
            .astype(int)
        )

        first_fixation_order.name = "first_fix_order"

        # gaze_target_fixation_sequence_df["pupil_diam"] = (
        #     gaze_target_fixation_sequence_df["pupil_diam"].round(2)
        # )

        # mean_duration_per_target = (
        #     gaze_target_fixation_sequence_df.groupby("target_ind")["duration"]
        #     .mean()
        #     .round(2)
        # )

        # mean_diam_per_target = (
        #     gaze_target_fixation_sequence_df.groupby("target_ind")["pupil_diam"]
        #     .mean()
        #     .round()
        # )

        # TODO: try the lines below
        # * Apply pd.to_numeric() here to ensure the data type is numeric
        gaze_target_fixation_sequence_df["pupil_diam"] = pd.to_numeric(
            gaze_target_fixation_sequence_df["pupil_diam"], errors="coerce"
        ).round(2)

        mean_duration_per_target = (
            pd.to_numeric(
                gaze_target_fixation_sequence_df.groupby("target_ind")["duration"],
                errors="coerce",
            )
            .mean()
            .round(2)
        )
        mean_duration_per_target = gaze_target_fixation_sequence_df.groupby(
            "target_ind"
        )["duration"].mean()  # .round(2)

        mean_diam_per_target = (
            pd.to_numeric(
                gaze_target_fixation_sequence_df.groupby("target_ind")["pupil_diam"],
                errors="coerce",
            )
            .mean()
            .round(2)
            # .round()
        )

        fix_counts_per_target = gaze_target_fixation_sequence_df[
            "target_ind"
        ].value_counts()

        total_fix_duration_per_target = gaze_target_fixation_sequence_df.groupby(
            "target_ind"
        )["duration"].sum()

        mean_duration_per_target.name = "mean_duration"
        mean_diam_per_target.name = "mean_pupil_diam"
        total_fix_duration_per_target.name = "total_duration"

        mean_diam_per_target.sort_values(ascending=False, inplace=True)
        fix_counts_per_target.sort_values(ascending=False, inplace=True)
        total_fix_duration_per_target.sort_values(ascending=False, inplace=True)

        gaze_info = pd.concat(
            [
                fix_counts_per_target,
                first_fixation_order,
                total_fix_duration_per_target,
                mean_duration_per_target,
                mean_diam_per_target,
            ],
            axis=1,
        ).reset_index()

        gaze_info["stim_name"] = gaze_info["target_ind"].replace(seq_and_choices)
        gaze_info["trial_N"] = trial_N
        gaze_info["stim_type"] = gaze_info["target_ind"].replace(stim_types)
        gaze_info.sort_values("target_ind", inplace=True)

        # gaze_info.query("target_ind in @sequence_icon_inds")
        # gaze_info.query("target_ind == @choice_icon_inds")
        # gaze_info.query("target_ind == @wrong_choice_icon_inds")
        # gaze_info.query("target_ind == @solution_ind")

        # gaze_info

        # gaze_target_fixation_sequence.query("target_ind == @sequence_icon_inds").groupby(
        #     "target_ind"
        # )["duration"].mean().plot(kind="bar")
        # gaze_target_fixation_sequence["duration"].plot()
        # gaze_target_fixation_sequence["pupil_diam"].plot()
        # gaze_target_fixation_sequence.groupby("target_ind")["pupil_diam"].plot()

        return (
            fixation_data,
            eeg_fixation_data,
            eeg_fixation_pac_data,
            gaze_target_fixation_sequence_df,
            gaze_info,
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
        save_dir: Path,
        preprocessed_dir: Path,
        force_preprocess: bool = False,
        reuse_ica: bool = True,
        raise_error: bool = True,
        pbar: bool = True,
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

        save_dir.mkdir(exist_ok=True, parents=True)

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
                return [None] * len(data)

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
        fixation_data_all = []
        eeg_fixation_data_all = []
        gaze_info_all = []
        gaze_target_fixation_sequence_all = []
        eeg_fixation_pac_data_all = []

        for trial_N in tqdm(
            behav.index, desc="Analyzing every trial", leave=False, disable=not pbar
        ):
            # * Get the EEG and ET data for the current trial
            eeg_trial = next(manual_eeg_trials)
            et_trial = next(manual_et_trials)

            (
                fixation_data,
                eeg_fixation_data,
                eeg_fixation_pac_data,
                gaze_target_fixation_sequence,
                gaze_info,
                fixations_sequence_erp,
                fixations_choices_erp,
            ) = self.analyze_trial_decision_period(
                eeg_trial,
                et_trial,
                behav,
                trial_N,
                eeg_baseline=c.EEG_BASELINE_FRP,
                eeg_window=c.FRP_WINDOW,
                show_plots=False,
            )

            sess_frps["sequence"].append(fixations_sequence_erp)
            sess_frps["choices"].append(fixations_choices_erp)
            fixation_data_all.append(fixation_data)
            eeg_fixation_data_all.append(eeg_fixation_data)
            gaze_info_all.append(gaze_info)
            gaze_target_fixation_sequence_all.append(gaze_target_fixation_sequence)
            eeg_fixation_pac_data_all.append(eeg_fixation_pac_data)

        # ic(len(eeg_fixation_pac_data_all))

        # * Concatenate the gaze data
        gaze_info = pd.concat([df for df in gaze_info_all if df.shape[0] > 0])
        gaze_info.reset_index(drop=True, inplace=True)

        gaze_target_fixation_sequence = pd.concat(
            [df for df in gaze_target_fixation_sequence_all if df.shape[0] > 0]
        )
        gaze_target_fixation_sequence.reset_index(
            drop=False, inplace=True, names=["fixation_N"]
        )

        valid_frps = dict(
            subj_N=subj_N,
            sess_N=sess_N,
            n_seq_frps=len(sess_frps["sequence"]) - sess_frps["sequence"].count(None),
            n_choices_frps=len(sess_frps["choices"]) - sess_frps["choices"].count(None),
        )

        # * Save the data to pickle files
        pd.DataFrame([valid_frps]).to_csv(save_dir / "valid_frps.csv", index=False)

        save_pickle(sess_frps, save_dir / "sess_frps.pkl")
        save_pickle(fixation_data_all, save_dir / "fixation_data.pkl")
        save_pickle(eeg_fixation_data_all, save_dir / "eeg_fixation_data.pkl")
        gaze_info.to_parquet(save_dir / "gaze_info.parquet", index=False)
        gaze_target_fixation_sequence.to_parquet(
            save_dir / "gaze_target_fixation_sequence.parquet", index=False
        )
        # save_pickle(gaze_info, save_dir / "gaze_info.pkl")
        # save_pickle(
        #     gaze_target_fixation_sequence,
        #     save_dir / "gaze_target_fixation_sequence.pkl",
        # )
        save_pickle(eeg_fixation_pac_data_all, save_dir / "eeg_fixation_pac_data.pkl")

        return (
            sess_frps,
            fixation_data,
            eeg_fixation_data,
            gaze_info,
            gaze_target_fixation_sequence,
            # eeg_fixation_pac_data_all,
        )

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

                if subj_data == "sess":
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

        # dict(zip(np.unique(sess_patterns[sess_missing_frps], return_counts=True)

        missing_frps = pd.DataFrame(
            missing_frps, columns=["subj_N", "sess_N", "trial_Ns", "count", "pct"]
        )

        if data_fmt == "exp":
            subj_data = [np.array(d) for d in subj_data]

        return subj_data, missing_frps


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
        """
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
        # if self.data_fmt == "bids":
        #     sess_dirs = sorted(
        #         list_contents(
        #             self.subj_dir, incl="folder", recurs=False, reg=r".*ses-*"
        #         )
        #     )
        # else:
        #     sess_dirs = sorted(
        #         list_contents(
        #             self.subj_dir, incl="folder", recurs=False, reg=r".*sess_*"
        #         )
        #     )

        sess_dirs = sorted(
            list_contents(self.subj_dir, incl="folder", recurs=False, reg=r".*ses.*")
        )
        sess_Ns = [int(re.search(r"\d{2}", d.name)[0]) for d in sess_dirs]
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

        prepro_eeg_files = list_contents(
            preprocessed_dir, reg=f".*subj_{subj_N:02}.*_preprocessed-raw.fif$"
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
            sess_N = int(prepro_eeg_file.stem.split("_")[1][2:])

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
                ch_names=selected_chans, sfreq=self.EEG["sfreq"], ch_types="eeg"
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
        preprocessed_dir: Path,
        raise_error: bool = False,
        eeg_incomplete: Literal["allow", "error", "skip"] = "error",
    ):
        behav, manual_et_trials, manual_eeg_trials = [], [], []

        for sess_N, sess_obj in self.sessions.items():
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
            ch_names=selected_chans, sfreq=self.EEG["sfreq"], ch_types="eeg"
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
    def analyze_sessions(
        self,
        save_dir: Path | None = None,
        preprocessed_dir: Path | None = None,
        force_preprocess: bool = False,
        reuse_ica: bool = True,
        raise_error: bool = True,
    ):
        if save_dir is None:
            # print("using default save directory for subject-level analysis")
            save_dir = c.EXPORT_DIR / f"subj_lvl/subj_{self.subj_N:02}"
            save_dir.mkdir(exist_ok=True, parents=True)

        if preprocessed_dir is None:
            preprocessed_dir = self.preprocessed_dir

        """Analyze all sessions for the subject."""
        sess_results = {}
        for sess_N, sess_obj in self.sessions.items():
            sess_results[sess_N] = sess_obj.analyze_session(
                save_dir=save_dir / f"sess_{sess_N:02}",
                preprocessed_dir=preprocessed_dir,
                force_preprocess=force_preprocess,
                reuse_ica=reuse_ica,
                raise_error=raise_error,
            )
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
        self, preprocessed_dir: Path, raise_error: bool = False, pbar: bool = True
    ):

        behav, manual_et_trials, manual_eeg_trials = {}, {}, {}

        for subj_N, subj_obj in tqdm(self.subjects.items()):
            beh, et, eeg = subj_obj.get_trials_data(
                preprocessed_dir=preprocessed_dir, raise_error=raise_error
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

        # behav_data = self.get_behav_data()
        # n_trials_behav = (
        #     behav_data.groupby("subj_N")["trial_N"].count().rename("n_trials")
        # )
        # mean_acc = behav_data.groupby("subj_N")["correct"].mean().rename("acc_mean")
        # sd_acc = behav_data.groupby("subj_N")["correct"].std().rename("acc_sd")


if __name__ == "__main__":
    # * -----------------------------
    # * -----------------------------
    # SSD_PATH = Path("/Volumes/Realtek 1Tb")
    # DATA_DIR = SSD_PATH / "PhD Data/experiment1/data/ANNs"

    # ANN_DIR = WD.parent / "experiment-ANNs"
    # SEQ_DIR = WD.parent / "config/sequences"

    # EXPORT_DIR = SSD_PATH / "PhD Data/experiment1-analysis/ANNs"

    # if not SSD_PATH.exists():
    #     print("WARNING: SSD not connected")
    # else:
    #     RDM_DIR = EXPORT_DIR / "RDMs"
    #     RDM_DIR.mkdir(parents=True, exist_ok=True)

    # with open(ANN_DIR / "config/instructions.txt", "r") as f:
    #     instructions = f.read()
    # * -----------------------------
    # * -----------------------------

    pass
