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

# Shared constants copied from the legacy human_data module.
CURRENT_DATA_FMT = c.CURRENT_DATA_FMT
TASK_NAME = c.TASK_NAME
