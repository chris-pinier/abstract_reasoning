from pathlib import Path
import matplotlib.image as mpimg
import mne
import pendulum
from box import Box
import pandas as pd
from ar_analysis.paths import (
    ANALYSIS_DIR,
    CONFIG_DIR,
    REPO_ROOT,
    SCRIPTS_DIR,
)
from ar_analysis.utils.analysis_utils import read_file
# from loguru import logger


def config() -> Box:
    # * ########################################
    # * GLOBAL CONFIGURATION
    # * ########################################
    CREDENTIALS = Box(read_file(CONFIG_DIR / "credentials.toml"))
    # notif = email_sender(CREDENTIALS.email.email, CREDENTIALS.email.password)

    PATTERNS = tuple(
        sorted(
            [
                "AAABAAAB",
                "ABABCDCD",
                "ABBAABBA",
                "ABBACDDC",
                "ABBCABBC",
                "ABCAABCA",
                "ABCDDCBA",
                "ABCDEEDC",
            ]
        )
    )

    # TODO: description
    ITEM_ID_SORT = pd.read_csv(SCRIPTS_DIR / "item_ids_sort_for_rdm.csv")

    TASK_NAME = "AbsPattComp"

    SEQ_FILE = CONFIG_DIR / "sequences/sessions-1_to_5-masked_idx(7).csv"

    DATASET = pd.concat(
        [
            pd.read_csv(f)
            for f in (REPO_ROOT / "experiment-ANNs/sequences").glob("*session_*.csv")
        ]
    ).reset_index(drop=True)

    # * ########################################
    # * LAB DATA ANALYSIS CONFIGURATION
    # * ########################################
    CURRENT_DATA_FMT = "bids"

    EXP_CONFIG_FILE = CONFIG_DIR / "experiment_config.toml"
    ANAYSIS_CONFIG_FILE = CONFIG_DIR / "analysis_config.toml"

    TIMEZONE = "Europe/Amsterdam"
    TIMESTAMP = pendulum.now(TIMEZONE).format("YYYYMMDD_HHmmss")

    LOG_DIR = ANALYSIS_DIR / "analysis_logs"

    # * Load experiment config
    # with open(EXP_CONFIG_FILE, "rb") as f:
    EXP_CONFIG = Box(read_file(EXP_CONFIG_FILE))

    # * Load analysis config
    # with open(ANAYSIS_CONFIG_FILE, "rb") as f:
    ANALYSIS_CONFIG = Box(read_file(ANAYSIS_CONFIG_FILE))

    BAD_SESSIONS = ANALYSIS_CONFIG.get("bad_sessions", {})

    # * Set backend for MNE and Matplotlib
    MNE_BROWSER_BACKEND = "qt"
    MPL_BACKEND = "module://matplotlib_inline.backend_inline"  # "ipympl"
    SAVE_FIG_PARAMS = dict(dpi=300, bbox_inches="tight")

    # * Random seed
    RAND_SEED = 0

    # * EEG Montage and Channel Groups
    EEG_MONTAGE = mne.channels.make_standard_montage("biosemi64")
    EEG_CHAN_GROUPS = ANALYSIS_CONFIG.eeg.ch_groups
    ALL_BAD_CHANS = ANALYSIS_CONFIG.eeg.bad_channels
    EOG_CHANS = ANALYSIS_CONFIG.eeg.chans.eog
    EMG_CHANS = ANALYSIS_CONFIG.eeg.chans.get("emg", [])
    STIM_CHAN = ANALYSIS_CONFIG.eeg.chans.stim
    NON_EEG_CHANS = EOG_CHANS + EMG_CHANS + [STIM_CHAN]

    # * Sampling Frequencies for EEG and Eye Tracking
    EEG_SFREQ = 2048
    ET_SFREQ = 2000

    # * Minimum fixation duration for eye tracking data
    MIN_FIXATION_DURATION = 0.02  # * seconds
    # MIN_FIXATION_DURATION = 0.1  # * seconds

    # * Time window for the Fixation-Related Potential (FRP)
    FRP_WINDOW = 0.600  # * seconds

    # * EEG Baseline period for the FRP (pre-fixation duration for baseline average)
    EEG_BASELINE_FRP = 0.1  # * seconds

    # * Getting Valid Event IDs
    VALID_EVENTS = EXP_CONFIG["lab"]["event_IDs"]
    VALID_EVENTS_INV = {v: k for k, v in VALID_EVENTS.items()}

    # * Number of sequences per session
    N_SEQ_PER_SESS = 80

    # * Loading images
    ICON_IMAGES_DIR = REPO_ROOT / "experiment-Lab/images"

    ICON_IMAGES = {img.stem: mpimg.imread(img) for img in ICON_IMAGES_DIR.glob("*.png")}

    IMG_SIZE = (256, 256)
    SCREEN_RESOLUTION = (2560, 1440)

    # * Stimulus positions
    Y_POS_CHOICES = -IMG_SIZE[1]
    Y_POS_SEQUENCE = IMG_SIZE[1]
    X_POS_STIM = ANALYSIS_CONFIG["stim"]["x_pos_stim"]

    # * Time bounds (seconds) for separating trials
    # * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
    PRE_TRIAL_TIME = 1
    POST_TRIAL_TIME = 1

    # * ########################################
    # * ANN DATA ANALYSIS CONFIGURATION
    # * ########################################
    ANSWER_REGEX = r"Answer:\s?\n?(\w+)"

    ANN_ID_MAPPING = {
        "Qwen--Qwen2.5-72B-Instruct": "Qwen2.5-72B",
        "deepseek-ai--DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-Distill-Llama-70B",
        "google--gemma-2-27b-it": "Gemma-2-27B",
        "google--gemma-2-2b-it": "Gemma-2-2B",
        "google--gemma-2-9b-it": "Gemma-2-9B",
        "meta-llama--Llama-3.2-3B-Instruct": "Llama-3.2-3B",
        "meta-llama--Llama-3.3-70B-Instruct": "Llama-3.3-70B",
        "microsoft--phi-4": "Phi-4",
    }

    ANN_ID_ORDER = [
        "Phi-4",
        "Gemma-2-2B",
        "Gemma-2-9B",
        "Gemma-2-27B",
        "Llama-3.2-3B",
        "Llama-3.3-70B",
        "Qwen2.5-72B",
        "Deepseek-R1-Distill-Llama-70B",
    ]

    return Box(locals())


Config = config()
