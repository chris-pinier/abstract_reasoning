from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import mne
from tqdm.auto import tqdm
from pprint import pprint
from abstract_reasoning_analysis.data_loader.human_data import (
    HumanSessData,
    HumanSubjData,
    HumanGroupData,
)
from abstract_reasoning_analysis.utils.analysis_utils import list_contents


MAIN_DIR = Path("/Volumes/SSD-512Go/PhD Data/experiment1")
DATA_DIR = MAIN_DIR / "data/Lab/raw"
PREPROCESSED_DIR = MAIN_DIR / "data/Lab/preprocessed"
EXPORT_DIR = MAIN_DIR / "data/export"


def test(
    data_dir=DATA_DIR,
    preprocessed_dir=PREPROCESSED_DIR,
    export_dir=EXPORT_DIR,
):
    # * ----------------------------------------
    # * I. Session Data
    # * ----------------------------------------
    sess = HumanSessData(
        subj_N=1,
        sess_N=1,
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        export_dir=export_dir,
    )
    # * ----------------------------------------
    # * ----------------------------------------
    pprint(sorted([i for i in dir(sess) if not i.startswith("_")]))

    # sess.get_et_calibration()

    sess.load_frp_data(sess.data_dir, sess.preprocessed_dir)

    eeg_montage = sess.get_eeg_montage()
    montage_plot = eeg_montage.plot()
    eeg_stim_flash_epochs = sess.get_stim_flash_eeg_epochs()

    sess_attrs = (
        "EEG base_name data_fmt data_dir export_dir preprocessed_dir sess_dir "
        "fnames subj_N sess_N task_name"
    )
    sess_attrs = sess_attrs.split(" ")

    for atr in sess_attrs:
        _atr = getattr(sess, atr)
        if isinstance(_atr, (Path, dict)):
            _atr = str(_atr)
        print(f"{atr}: {_atr:<30}")

    # * ----------------------------------------
    # * ----------------------------------------
    et_files = sess.search_et_files()
    eeg_file = sess.search_eeg_file()
    behav_file = sess.search_behav_file()

    eeg_metadata = sess.extract_eeg_metadata()
    # et_metadata = sess.extract_et_metadata()

    # sess.check_data()

    res = sess.analyze_perf()
    print(f"{res.keys() = }")

    sess.show_dir_struct()

    # * ----------------------------------------
    # * ----------------------------------------

    # * 1) Load Raw Data
    raw_behav = sess.get_raw_behav_data()
    raw_eeg = sess.get_raw_eeg_data()
    raw_et = sess.get_raw_et_data()
    sess_info = sess.get_sess_info()
    del raw_behav, raw_eeg, raw_et, sess_info

    raw_data = sess.get_raw_data()
    del raw_data

    # * 2) Load Preprocessed Data
    behav = sess.get_behav_data()
    et = sess.get_et_data()
    eeg = sess.get_eeg_data()
    del behav, et, eeg

    data = sess.get_data()
    del data

    # * 3)
    trials_data = sess.get_trials_data(preprocessed_dir=sess.preprocessed_dir)
    beh, et, eeg = trials_data

    del trials_data, beh, et, eeg

    # sess.get_behav_rdms()

    eeg_epochs = sess.get_eeg_epochs(erp_events="trial_start", erp_tmin=0, erp_tmax=2)
    eeg_epochs.plot()

    # * ----------------------------------------
    # * II. Subject Data
    # * ----------------------------------------
    subj = HumanSubjData(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        export_dir=export_dir,
        subj_N=1,
    )
    subj.show_dir_struct()

    # * 1)
    behav = subj.get_behav_data()
    del behav

    # * 2)

    # * 3)
    trials_data = subj.get_trials_data(preprocessed_dir=subj.preprocessed_dir)
    beh, et, eeg = trials_data

    del trials_data, beh, et, eeg

    # et = subj.get_et_data()
    # trials_data = subj.get_trials_data()
    # eeg = subj.get_eeg_data()

    #

    # * ----------------------------------------
    # * III. Group Data
    # * ----------------------------------------
    group = HumanGroupData(
        data_dir=data_dir,
        preprocessed_dir=preprocessed_dir,
        export_dir=export_dir,
    )
    group.show_dir_struct()

    group_behav = group.get_behav_data()

    # group.subjects[1].sessions[1].sess_dir
