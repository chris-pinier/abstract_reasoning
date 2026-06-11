# * IMPORTS
from pathlib import Path
from typing import Dict, Final
from box import Box
import pandas as pd
import os
from IPython.display import display
from pprint import pprint
import plotly.express as px
from contextlib import redirect_stdout
import mne
import sys

WD = Path(__file__).parent.resolve()
ROOT = Path("/".join(WD.parts[: WD.parts.index("abstract_reasoning") + 1]))

sys.path.append(WD)
os.chdir(WD)
assert WD == Path.cwd()

# * RELATIVE IMPORTS
# from analysis_conf import Config as c
# from data_loader.human_data import HumanSessData, HumanSubjData, HumanGroupData
# from utils.analysis_utils import read_file, list_contents
# from analysis_compare_clean import CombinedData
from ar_analysis.data_loader.human_data import (
    HumanSessData,
    HumanSubjData,
    HumanGroupData,
)
from ar_analysis.utils.custom_type_hints import DATA_FMTS
from ar_analysis.utils.analysis_utils import (
    read_file,
    reorder_item_ids,
    list_contents,
)
from ar_analysis.analysis_rsa import get_ds_and_rdm
from ar_analysis.analysis_config import Config as c

# ! TEMP: to locate and use ffmpeg
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

# * GLOBAL VARIABLES
PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER

DATASET = c.DATASET
SEQ_FILE = c.SEQ_FILE
# DIRECTORIES = c.DIRECTORIES

# SAVE_DISK: Final = Path("/Volumes/Realtek 1Tb")
SAVE_DISK: Final = Path("/Volumes/SSD-512Go")
assert SAVE_DISK.exists(), "WARNING: SSD not connected"
MAIN_DATA_DIR = SAVE_DISK / "PhD Data/experiment1/data/"
DIRECTORIES: Final = Box(
    {
        "ann": {
            "data": MAIN_DATA_DIR
            / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
            "prepro": None,
            "analyzed": MAIN_DATA_DIR / "ANNs/analyzed",
            "export": MAIN_DATA_DIR / "ANNs/analyzed",
        },
        "human": {
            "data": MAIN_DATA_DIR / "Lab/raw",
            "prepro": MAIN_DATA_DIR / "Lab/preprocessed",
            "analyzed": MAIN_DATA_DIR / "Lab/analyzed",
            "export": MAIN_DATA_DIR / "Lab/analyzed",
        },
    }
)
# * GLOBAL VARIABLES
PATTERNS = c.PATTERNS
ANN_ID_MAPPING = c.ANN_ID_MAPPING
ANN_ID_ORDER = c.ANN_ID_ORDER

DATASET = c.DATASET
SEQ_FILE = c.SEQ_FILE
# DIRECTORIES = c.DIRECTORIES

# SAVE_DISK: Final = Path("/Volumes/Realtek 1Tb")
SAVE_DISK: Final = Path("/Volumes/SSD-512Go")
assert SAVE_DISK.exists(), "WARNING: SSD not connected"
MAIN_DATA_DIR = SAVE_DISK / "PhD Data/experiment1/data/"
DIRECTORIES: Final = Box(
    {
        "ann": {
            "data": MAIN_DATA_DIR
            / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
            "prepro": None,
            "analyzed": MAIN_DATA_DIR / "ANNs/analyzed",
            "export": MAIN_DATA_DIR / "ANNs/analyzed",
        },
        "human": {
            "data": MAIN_DATA_DIR / "Lab/raw",
            "prepro": MAIN_DATA_DIR / "Lab/preprocessed",
            "analyzed": MAIN_DATA_DIR / "Lab/analyzed",
            "export": MAIN_DATA_DIR / "Lab/analyzed",
        },
    }
)

org_dir = Path("/Volumes/SSD-512Go/PhD Data/experiment1/data/Lab/raw-ORIGINAL")
bids_dir = Path("/Volumes/SSD-512Go/PhD Data/experiment1/data/Lab/raw-BIDS3")


org_files = {
    "eeg": sorted(org_dir.rglob("*.bdf")),
    "eyetrack": sorted(org_dir.rglob("*.asc")),
}
org_files = Box(org_files)

bids_files = {
    "eeg": sorted(bids_dir.rglob("*.bdf")),
    "eyetrack": sorted(bids_dir.rglob("*physioevents*.tsv*")),
}
bids_files = Box(bids_files)

len(org_files.eyetrack)
len(bids_files.eyetrack)

len(org_files.eeg)
len(bids_files.eeg)

sorted(bids_dir.rglob("*"))
sorted(bids_dir.rglob("*physioevents*.tsv*"))

org_files.eeg


sorted(org_dir.rglob("*"))
