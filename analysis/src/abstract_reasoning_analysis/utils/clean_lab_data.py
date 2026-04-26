# * ########################################
# * IMPORTS
# * ########################################
from pathlib import Path
import json
import re
import contextlib
from matplotlib import pyplot as plt, patches as mpatches
import mne
from mne_icalabel import label_components
from mne.preprocessing.eyetracking import read_eyelink_calibration
from mne.io import BaseRaw
import numpy as np
import pandas as pd
import pendulum
import tensorpac
import io
import rsatoolbox
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
import json
from pprint import pprint
import plotly.express as px
from directory_tree import DisplayTree
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from icecream import ic

# *
from abstract_reasoning_analysis.analysis_plotting import (
    get_gaze_heatmap,
    plot_eeg,
    plot_eeg_and_gaze_fixations,
    plot_matrix,
    prepare_eeg_data_for_plot,
    show_ch_groups,
    plot_sequence_img,
    plot_rdm,
)
from abstract_reasoning_analysis.utils.analysis_utils import (
    check_ch_groups,
    get_stim_coords,
    get_trial_info,
    locate_trials,
    normalize,
    resample_and_handle_nans,
    resample_eye_tracking_data,
    set_eeg_montage,
    save_pickle,
    load_pickle,
    apply_df_style,
    read_file,
    reorder_item_ids,
    get_timestamp,
    list_contents,
)

from abstract_reasoning_analysis.analysis_config import Config as c
from abstract_reasoning_analysis.analysis_rsa import get_ds_and_rdm


DATA_DIR = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab-Copy")
DATA_DIR.exists()


def clean_behav_files():
    behav_files = sorted(list_contents(DATA_DIR, incl="file", reg="behav.csv"))

    dfs = []
    for f in behav_files:
        subj = f.parents[1].name.split("_")[1]
        ses = f.parents[0].name.split("_")[1]

        df = pd.read_csv(f)
        blockN = df["blockN"].copy()
        unnamed_cols = [c for c in df.columns if "unnamed" in c.lower()]
        df.drop(columns=unnamed_cols + ["subj_id", "blockN"], inplace=True)

        df.insert(0, "sub", subj)
        df.insert(1, "ses", ses)
        df.insert(2, "blockN", blockN)
        df.insert(3, "trialN", df.index)

        if ses == "01":
            behav_practice_file = list_contents(f.parent, reg="practice.csv")[0]
            df_practice = pd.read_csv(behav_practice_file)
            blockN = df_practice["blockN"].copy()
            unnamed_cols = [c for c in df_practice.columns if "unnamed" in c.lower()]
            df_practice.drop(columns=unnamed_cols + ["subj_id", "blockN"], inplace=True)

            df_practice.insert(0, "sub", subj)
            df_practice.insert(1, "ses", ses)
            df_practice.insert(2, "blockN", blockN)
            df_practice.insert(3, "trialN", df_practice.index - len(df_practice))

            df = pd.concat([df_practice, df]).reset_index(drop=True)

        dfs.append(df)

    df = pd.concat(dfs)
    df.groupby("sub")["trialN"].count()


def clean_filenames():
    for subj_folder in list_contents(DATA_DIR, "folder", False):
        # print(subj_folder)
        subj = subj_folder.name.split("_")[1]

        for sess_folder in list_contents(subj_folder, "folder", False):
            sess = sess_folder.name.split("_")[1]
            # print(f"\t{sess_folder}")
            behav_file = list_contents(sess_folder, reg="behav.csv")[0]
            behav_practice_file = list_contents(sess_folder, reg="practice.csv")

            sess_info_file = list_contents(sess_folder, reg="sess_info.json")[0]
            data_files = list_contents(sess_folder, reg=r"cp\d{4}")

            behav_file.rename(behav_file.parent / f"sub{subj}-ses{sess}-behav.csv")
            sess_info_file.rename(
                sess_info_file.parent / f"sub{subj}-ses{sess}-sess_info.json"
            )
            for f in data_files:
                f.rename(f.parent / f"sub{subj}-ses{sess}{f.suffix}")

            sess_folder.rename(sess_folder.parent / f"ses{sess}")

        subj_folder.rename(subj_folder.parent / f"subj{subj}")

    [f.name for f in list_contents(DATA_DIR, incl="file")]

    behav_files = sorted(list_contents(DATA_DIR, incl="file", reg="behav.csv"))
    behav_df = pd.concat([pd.read_csv(f) for f in behav_files])
    behav_df.head(30)

    behav_df.query("trialN.isna()")


# def eyelink_to_bids():
#     import pyxations as pyx

#     # 1) Convert raw files to BIDS
#     pyx.dataset_to_bids(
#         target_folder_path="/Volumes/Realtek 1Tb/PhD Data/experiment1/data/eyelink_bids",
#         files_folder_path="/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab-Copy",
#         dataset_name="TEST-EYELINK_BIDS",
#     )
