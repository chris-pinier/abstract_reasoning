from pathlib import Path
import matplotlib.image as mpimg
import mne
import mne.baseline
import pendulum
import tomllib
from box import Box
from typing import Final
import pandas as pd
from analysis_utils import read_file, email_sender


def config():
    WD = Path(__file__).resolve().parent

    CREDENTIALS = Box(read_file(WD.parent / "config/credentials.toml"))
    # notif = email_sender(CREDENTIALS.email.email, CREDENTIALS.email.password)

    SSD_PATH: Final = Path("/Volumes/Realtek 1Tb")
    EXPORT_DIR: Final = SSD_PATH / "PhD Data/experiment1-analysis/Lab"

    EXP_CONFIG_FILE: Final = WD.parent / "config/experiment_config.toml"
    ANAYSIS_CONFIG_FILE: Final = WD.parent / "config/analysis_config.toml"

    # DATA_DIR = Path("/Users/chris/Documents/PhD-Local/PhD Data/experiment1/data/Lab")
    DATA_DIR: Final = Path("/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab")
    TIMEZONE: Final = "Europe/Amsterdam"
    TIMESTAMP: Final = pendulum.now(TIMEZONE).format("YYYYMMDD_HHmmss")

    LOG_DIR: Final = WD / "analysis_logs"

    # * Load experiment config
    # with open(EXP_CONFIG_FILE, "rb") as f:
    EXP_CONFIG: Final = Box(read_file(EXP_CONFIG_FILE))

    # * Load analysis config
    # with open(ANAYSIS_CONFIG_FILE, "rb") as f:
    ANALYSIS_CONFIG: Final = Box(read_file(ANAYSIS_CONFIG_FILE))

    # * Create an empty notes file
    # NOTES_FILE = WD / "notes.json"
    # with open(NOTES_FILE, "w") as f:
    #     json.dump({}, f)

    # * Set backend for MNE and Matplotlib
    MNE_BROWSER_BACKEND: Final = "qt"
    MPL_BACKEND: Final = "module://matplotlib_inline.backend_inline"  # "ipympl"

    # * Random seed
    RAND_SEED: Final = 0

    # * EEG Montage and Channel Groups
    EEG_MONTAGE: Final = mne.channels.make_standard_montage("biosemi64")
    EEG_CHAN_GROUPS: Final = ANALYSIS_CONFIG.eeg.ch_groups
    ALL_BAD_CHANS: Final = ANALYSIS_CONFIG.eeg.bad_channels
    EOG_CHANS: Final = ANALYSIS_CONFIG.eeg.chans.eog
    STIM_CHAN: Final = ANALYSIS_CONFIG.eeg.chans.stim
    NON_EEG_CHANS: Final = EOG_CHANS + [STIM_CHAN]

    # * Sampling Frequencies for EEG and Eye Tracking
    EEG_SFREQ: Final = 2048
    ET_SFREQ: Final = 2000

    # * Minimum fixation duration for eye tracking data
    MIN_FIXATION_DURATION: Final = 0.02  # * seconds
    # MIN_FIXATION_DURATION = 0.1  # * seconds

    # * Time window for the Fixation-Related Potential (FRP)
    FRP_WINDOW: Final = 0.600  # * seconds

    # * EEG Baseline period for the FRP (pre-fixation duration for baseline average)
    EEG_BASELINE_FRP: Final = 0.1  # * seconds

    # * Getting Valid Event IDs
    VALID_EVENTS: Final = EXP_CONFIG["lab"]["event_IDs"]
    VALID_EVENTS_INV: Final = {v: k for k, v in VALID_EVENTS.items()}

    # * Number of sequences per session
    N_SEQ_PER_SESS: Final = 80

    # * Loading images
    ICON_IMAGES_DIR: Final = WD.parent / "experiment-Lab/images"
    
    ICON_IMAGES: Final = {
        img.stem: mpimg.imread(img) for img in ICON_IMAGES_DIR.glob("*.png")
    }

    IMG_SIZE: Final = (256, 256)
    SCREEN_RESOLUTION: Final = (2560, 1440)

    # * Stimulus positions
    Y_POS_CHOICES: Final = -IMG_SIZE[1]
    Y_POS_SEQUENCE: Final = IMG_SIZE[1]
    X_POS_STIM: Final = ANALYSIS_CONFIG["stim"]["x_pos_stim"]

    PATTERNS: Final = sorted(
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

    # * Time bounds (seconds) for separating trials
    # * -> uses trial_start onset - pre_trial_time to trial_end onset + post_trial_time
    PRE_TRIAL_TIME: Final = 1
    POST_TRIAL_TIME: Final = 1

    # TODO: description
    ITEM_ID_SORT = pd.read_csv("item_ids_sort_for_rdm.csv")

    return Box(locals())


Config = config()
