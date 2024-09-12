from pathlib import Path
import sys
from psychopy import visual, core, event, logging, monitors, gui
from string import ascii_letters, digits
import sys
import json
from icecream import ic
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Union
from collections import namedtuple
import code
import traceback
# import platform
# import re
# import inspect
# import pickle
# from tqdm.auto import tqdm
# import time
# import pylink
# import os
from setup import prepare_images
from utils import prepare_sess, get_monitors_info, invert_dict, get_timestamp
from devices import EEGcap, EyeTracker

comments_desc = {
    "! TEMP": "signals a temporary change that should be reverted later",
    "! IMPLEMENT": "reminder to implement a feature",
}

# * ####################################################################################
# * GLOBAL VARIABLES
# * ####################################################################################
# logging.console.setLevel(logging.CRITICAL)
rand_seed = 0
rng = np.random.default_rng(seed=rand_seed)

# * IP address of the EyeLink tracker; default = "100.1.1.1"
eye_tracker_add = "100.1.1.1"

# * set to True if you are using a Mac with a retina display
use_retina = False

# * set to True if you want to check the configuration before starting the experiment
config_check = True

fullscr = True

# icecream config
ic.configureOutput(includeContext=False, prefix="")
# ic.enable()
ic.disable()

wd = Path.cwd()
results_dir = wd / "results/raw"
results_dir.mkdir(parents=True, exist_ok=True)

sequences_dir = wd / "sequences"

config_dir = wd.parent / "config"
img_dir = wd / "images"

# * Load experiment config
with open(config_dir / "experiment_config.json") as f:
    exp_config = json.load(f)

block_size = 20

timings = exp_config["global"]["timings"]

# ! TEMP
# timings = {
#     'feedback_duration': 0,
#     'intertrial_interval': [0, 0],
#     'pres_duration': 0.01,
#     'pre_pres_duration': None,
#     'resp_window': 1
#     }
# ! TEMP

timings = namedtuple("Timings", timings.keys())(*timings.values())
iti = timings.intertrial_interval

valid_events = exp_config["local"]["event_IDs"]

allowed_keys = exp_config["local"]["allowed_keys"]
allowed_keys_str = ", ".join(allowed_keys).upper()

refresh_rate = exp_config["local"]["monitor"].get("refresh_rate")

trial_count = 0

# * ####################################################################################
# * INSTRUCTIONS & MESSAGES
# * ####################################################################################

messages = {
    "welcome": (
        "You are going to solve {n} abstract reasoning problems. Your goal is to "
        "continue the sequence in the top row with one of the four options in the "
        "bottom row.\n"
        "Use the {keys} keys to select one of these options from left to right.\n"
        "Place your fingers on the keys and press one of them to start."
    ),
    "practice_start": ("Practice block.\n" "Press any of the {keys} keys to start."),
    "practice_end": (
        "Practice block completed.\n"
        "Place your fingers on the {keys} keys and press one of them to start."
    ),
    "block_end": (
        "End of block {blockN}\nTake a break\n"
        "Place your fingers on the {keys} keys and press one of them to continue."
    ),
    "end": "End of experiment. Thank you for participating!\nPress enter to quit",
    "abort_trial": (
        "Trial aborted.\n"
        "To continue the experiment: press any of the {keys} keys.\n"
        "To quit the experiment: press 'escape'."
    ),
    "timeout": "Timeout",
    "invalid_key": (
        "Invalid key pressed!\n"
        "Place your fingers on the {keys} keys and press one of them to continue."
    ),
    "error": (
        "An unexpected error occurred. Please contact the experimenter.\n"
        "Press enter to save the data and quit."
    ),
}


# * ####################################################################################
# * FUNCTIONS
# * ####################################################################################
def check_refresh_rate(win):
    win.recordFrameIntervals = True
    pass


def win_flip(
    win: visual.window.Window,
    cursor: bool = False,
    bg_color: Tuple[int] = (0, 0, 0),
    clear=True,
):
    """
    Flips the psychopy window, updating its content and controlling cursor visibility.

    Parameters:
    win (visual.window.Window): The psychopy window object.
    cursor (bool): Whether the cursor should be visible. Default is False.
    bg_color (Tuple[int]): The RGB background color. Default is black (0, 0, 0).
    """
    win.mouseVisible = cursor
    win.winHandle.set_mouse_visible(cursor)
    win.winHandle.set_mouse_position(-1, -1)
    win.fillColor = bg_color
    win.flip(clearBuffer=clear)


def record_event(event_name: str, eeg_device: EEGcap, eye_tracker: EyeTracker):
    """
    Records an event by sending it to both the EEG device and eye tracker.

    Parameters:
    event_name (str): The name of the event to be recorded.
    eeg_device (EEGcap): The EEG device object.
    eye_tracker (EyeTracker): The eye tracker object.
    """

    eeg_device.send(event_name)
    eye_tracker.send(event_name)
    # print(f"{event_name = } SENT")


def display_images_sequentially(
    win: visual.window.Window,
    images: List[visual.ImageStim],
    eeg_device: EEGcap,
    eye_tracker: EyeTracker,
    event_name: str,
    fix_cross: visual.TextStim = None,
    pres_duration: float = None,
    pres_frames: int = None,
    order: List[int] = None,
):
    """
    Displays a sequence of images on the psychopy window and records corresponding events.

    Parameters:
    win (visual.window.Window): The psychopy window object.
    images (List[visual.ImageStim]): List of image stimuli to be displayed.
    eeg_device (EEGcap): The EEG device object.
    eye_tracker (EyeTracker): The eye tracker object.
    event_name (str): The name of the event to be recorded for each image display.
    fix_cross (visual.TextStim, optional): A fixation cross to be displayed with each image.
    pres_duration (float, optional): The presentation duration for each image in seconds.
    pres_frames (int, optional): The presentation duration for each image in frames.
    order (List[int], optional): The order in which to display the images.

    Note: Either pres_duration or pres_frames must be specified, but not both.
    """
    # * Clear the screen
    win.mouseVisible = False
    win.winHandle.set_mouse_visible(False)
    win_flip(win)

    assert bool(pres_duration) ^ bool(
        pres_frames
    ), "You can only specify pres_duration or pres_frames, not both"

    if order is None:
        order = range(len(images))

    if pres_duration:
        for img_idx in order:
            images[img_idx].draw()
            if fix_cross is not None:
                fix_cross.draw()
            win_flip(win)

            record_event(event_name, eeg_device, eye_tracker)
            core.wait(pres_duration)

    elif pres_frames:
        for img_idx in order:
            for _ in range(pres_frames):
                images[img_idx].draw()
                if fix_cross is not None:
                    fix_cross.draw()
                win_flip(win)
                record_event(event_name, eeg_device, eye_tracker)
    win_flip(win)


def get_feedback(
    win: visual.window.Window,
    correct: bool,
    choice_x_pos,
    solution_x_pos,
    y_pos,
    img_size,
    all_imgs,
    duration,
):
    solution_rect = visual.rect.Rect(
        win,
        lineWidth=10,
        lineColor="green",
        fillColor=None,
        pos=(solution_x_pos, y_pos),
        size=img_size,
        units="pix",
    )
    solution_rect.draw()

    if not correct:
        choice_rect = visual.rect.Rect(
            win,
            lineWidth=10,
            lineColor="red",
            fillColor=None,
            pos=(choice_x_pos, y_pos),
            size=img_size,
            units="pix",
        )
        choice_rect.draw()

    for img in all_imgs:
        img.draw()

    win_flip(win)

    core.wait(duration)


def show_msg(
    win: visual.window.Window, text: str, kwargs: dict = None, keys: list = None
):
    """
    Displays a message on the psychopy window and optionally waits for a key press.

    Parameters:
    win (visual.window.Window): The psychopy window object.
    text (str): The message to be displayed.
    kwargs (dict, optional): Additional keyword arguments to be passed to visual.TextStim.
    keys (list, optional): List of keys to wait for before continuing.

    Returns:
    The pressed key if keys are specified, otherwise None.
    """

    win_flip(win)
    kwargs = {} if kwargs is None else kwargs
    msg = visual.TextStim(win, text, **kwargs)
    # TODO : color=txt_color, wrapWidth=scn_width / 2)
    msg.draw()
    win_flip(win)

    if keys:
        pressed_key = event.waitKeys(keyList=keys)
        win_flip(win)
        return pressed_key


def show_dialogue() -> dict:
    """
    Displays a dialogue box to collect subject and session information.

    Returns:
    dict: A dictionary containing the collected session information.
    """

    # * loop until we get a valid filename
    while True:
        dlg = gui.Dlg()
        dlg.addText("Subject info")
        dlg.addField(key="subj_id", label="ID:", required=True)

        dlg.addText("Experiment Info")
        dlg.addField("sess", "Session:", choices=["1", "2", "3", "4", "5"])

        dlg.addText("Eye tracking")
        dlg.addField(key="edf_fname", label="File Name:")
        dlg.addField("eye", "Eye tracked:", choices=["left", "right"])
        dlg.addField(key="eye_screen_dist", label="Eye to Screen Distance (mm):")

        # ! IMPLEMENT: validate fields -> subj_id should be required

        # * show dialog and wait for OK or Cancel
        sess_info = dlg.show()

        if dlg.OK:  # * if sess_info is not None
            print(f"EDF data filename: {sess_info['edf_fname']}.EDF")
        else:
            print("user cancelled")
            core.quit()
            sys.exit()

        # * strip trailing characters, ignore the ".edf" extension
        edf_fname = dlg.data["edf_fname"].rstrip().split(".")[0]

        # if dlg.data["subj_id"] == "": # TODO
        #     print("ERROR: Subject ID is required")

        # * check if the filename is valid (length <= 8 & no special char)
        allowed_char = ascii_letters + digits + "_"

        edf_name_conds = [
            all([c in allowed_char for c in edf_fname]),
            1 <= len(edf_fname) <= 8,
        ]

        if not all(edf_name_conds):
            print(
                "ERROR: Invalid EDF filename. Name should be 1 to 8 characters long",
                "and contain only letters, numbers, and underscores.",
            )
        else:
            break

    sess_info["edf_file"] = f"{edf_fname}.EDF"
    sess_info["subj_id"] = str(sess_info["subj_id"]).zfill(2)
    sess_info["sess"] = str(sess_info["sess"]).zfill(2)
    sess_info["date"] = get_timestamp("%Y%m%d_%H%M%S")
    sess_info["sess_id"] = (
        f'{sess_info["subj_id"]}-{sess_info["sess"]}-{sess_info["date"]}'
    )

    return sess_info


def terminate_exp(
    win,
    eeg_device: EEGcap,
    eye_tracker: EyeTracker,
    sess_data: dict,
    sess_info: dict,
    session_dir: str,
):
    """
    Gracefully terminates the experiment, saving data and closing connections.

    Parameters:
    win: The psychopy window object.
    eeg_device (EEGcap): The EEG device object.
    eye_tracker (EyeTracker): The eye tracker object.
    sess_data (dict): The session data.
    sess_info (dict): The session information.
    session_dir (str): The directory path for saving session data.
    """

    # print("TERMINATING TASK, SENDING 'experiment_end'...", end="")
    record_event("experiment_end", eeg_device, eye_tracker)
    # print("DONE")

    session_dir = Path(session_dir)
    eye_tracker.get_file(sess_info["edf_file"], session_dir)
    eye_tracker.disconnect(session_dir)
    eye_tracker.edf2asc(session_dir / sess_info["edf_file"])

    # # * ################ SAVE DATA ################
    behav_fpath = session_dir / f'{sess_info["sess_id"]}-behav.csv'
    print(
        f"Exporting behavioral results to: {behav_fpath.relative_to(session_dir.parents[3])}"
    )
    pd.DataFrame(sess_data).to_csv(behav_fpath)

    # # * ################ END OF EXPERIMENT ################
    show_msg(
        win,
        messages["end"],
        keys=["return"],
    )
    # * close the PsychoPy window
    win.close()

    # * Enter interactive mode
    code.interact(local=dict(globals(), **locals()), banner="Interactive mode. quit with: exit()")

    # * quit PsychoPy
    core.quit()
    # * quit Python
    sys.exit()


def abort_trial(
    win: visual.window.Window,
    eeg_device: EEGcap,
    eye_tracker: EyeTracker,
    keys: list,
) -> str:
    """
    Handles the abortion of a trial, allowing the user to continue or quit the experiment.

    Parameters:
    win: The psychopy window object.
    eeg_device: The EEG device object.
    eye_tracker: The eye tracker object.

    Returns:
    str: "abort" if the user chooses to quit, "continue" if they choose to continue.
    """

    record_event("trial_aborted", eeg_device, eye_tracker)

    choice = show_msg(
        win,
        text=messages["abort_trial"].format(keys=allowed_keys_str),
        keys=keys + ["escape"],
    )

    if choice[0] == "escape":
        return "abort"
    elif choice[0] == "return":
        return "continue"


def get_practice_sequences():
    file = wd / "sequences/practice_sequences.csv"
    practice_df = pd.read_csv(file, dtype="str")

    practice_df["item_id"] = practice_df["item_id"].astype(int)
    practice_df["masked_idx"] = practice_df["masked_idx"].astype(int)

    np.random.seed(rand_seed)
    practice_df = practice_df.groupby("pattern").sample(n=1, random_state=rand_seed)

    return practice_df


def modify_seq_csv():
    pass
    # file = wd / "sequences/practice_sequences.csv"
    # practice_df = pd.read_csv(file)

    # seq_orders = [
    #     "".join([str(i) for i in rng.choice(range(8), size=(8), replace=False)])
    #     for _ in range(len(practice_df))
    # ]

    # choice_orders = [
    #     "".join([str(i) for i in rng.choice(range(4), size=(4), replace=False)])
    #     for _ in range(len(practice_df))
    # ]

    # practice_df['choice_order'] = choice_orders
    # practice_df['seq_order'] = seq_orders

    # practice_df['trial_type'] = 'practice'
    # # practice_df.drop(columns=['Unnamed: 0'], inplace=True)
    # # practice_df.to_csv(file, index=False)

    # seq_files = [sequences_dir / f"session_{i}.csv" for i in range(1, 6)]
    # seq_dfs = {fpath: pd.read_csv(fpath, dtype=str) for fpath in seq_files}
    # col_types = practice_df.dtypes

    # for fpath, df in seq_dfs.items():

    #     df['trial_type'] = 'experiment'

    #     for col, dtype in col_types.items():

    #         if dtype == "object":
    #             df[col] = df[col].str.strip()

    #     df["choice_order"] = df["choice_order"].astype(str)
    #     df["seq_order"] = df["seq_order"].astype(str)

    #     seq_orders = [
    #         ''.join([str(i) for i in rng.choice(range(8), size=(8), replace=False)]) for _ in range(len(df))
    #     ]
    #     choice_orders = [
    #         ''.join([str(i) for i in rng.choice(range(4), size=(4), replace=False)]) for _ in range(len(df))
    #     ]
    #     df['choice_order'] = choice_orders
    #     df['seq_order'] = seq_orders

    #     df.to_csv(fpath, index=False)

def run_trial(
    win: visual.window.Window,
    trial: dict,
    # trialN: int,
    images,  # TODO: type hint
    img_size,  # TODO: type hint
    eeg_device: EEGcap,
    eye_tracker: EyeTracker,
    global_clock: core.Clock,
    sess_info: dict,
    y_pos_sequence: int,
    y_pos_choices: int,
) -> Union[dict, str]:
    """
    Runs a single trial of the experiment.

    Parameters:
    win (visual.window.Window): The psychopy window object.
    trial (dict): The trial information.
    # trialN (int): The trial number. #TODO: Remove if not used
    images: Dictionary mapping image names to file paths.
    img_size: The size of the images.
    eeg_device (EEGcap): The EEG device object.
    eye_tracker (EyeTracker): The eye tracker object.
    global_clock (core.Clock): Clock for timing the experiment.
    sess_info (dict): The session information.
    y_pos_sequence (int): The y-coordinate for displaying the sequence.
    y_pos_choices (int): The y-coordinate for displaying the choices.

    Returns:
    dict: The trial data, or "abort" if the trial was aborted.
    """

    # print("in run_trial")
    # print(images)
    # print("\n\n")

    sequence = [v for k, v in trial.items() if "figure" in k]

    masked_idx = trial["masked_idx"]
    solution = trial["solution"]
    sequence[masked_idx] = "question-mark"

    # ic(trialN, sequence, solution)

    x_pos = trial["x_pos"]
    avail_choices = list(trial["resp_map"].values())

    # * Load the sequence images for the current trial
    sequence_imgs = []
    for i, img_name in enumerate(sequence):
        img_path = images[img_name]
        img = visual.ImageStim(
            win,
            image=img_path,
            pos=(x_pos["items_set"][i], y_pos_sequence),
        )
        sequence_imgs.append(img)

    # * Load the choice images for the current trial
    avail_choices_imgs = []
    for i, img_name in enumerate(avail_choices):
        img_path = images[img_name]
        img = visual.ImageStim(
            win,
            image=img_path,
            pos=(x_pos["avail_choice"][i], y_pos_choices),
        )
        avail_choices_imgs.append(img)

    all_imgs = sequence_imgs + avail_choices_imgs

    # * Start of trial, record onset time
    trial_onset_time = global_clock.getTime()

    record_event("trial_start", eeg_device, eye_tracker)

    core.wait(1)
    win_flip(win)

    # * Displaying Sequence items one by one
    display_images_sequentially(
        win=win,
        images=sequence_imgs,
        eeg_device=eeg_device,
        eye_tracker=eye_tracker,
        event_name="stim-flash_sequence",
        fix_cross=None,
        pres_duration=timings.pres_duration,
        order=trial["seq_order"],
    )

    # * Displaying Choice items one by one
    display_images_sequentially(
        win=win,
        images=avail_choices_imgs,
        eeg_device=eeg_device,
        eye_tracker=eye_tracker,
        event_name="stim-flash_choices",
        fix_cross=None,
        pres_duration=timings.pres_duration,
        order=trial["choice_order"],
    )

    series_end_time = global_clock.getTime()

    # * Displaying all sequence items + choices at once
    for img in all_imgs:
        img.draw()
    win_flip(win)

    record_event("stim-all_stim", eeg_device, eye_tracker)

    choice_onset_time = global_clock.getTime()

    # * Start response clock
    response_clock = core.Clock()

    response = event.waitKeys(
        timeStamped=response_clock,
        maxWait=timings.resp_window,
        clearEvents=True,
    )

    rt_global = global_clock.getTime()

    if response:
        choice_key, response_time = response[0]
    else:
        choice_key, response_time = "timeout", "timeout"

    record_event(choice_key, eeg_device, eye_tracker)

    # * Check if pressed key is allowed and choice is correct
    if choice_key in allowed_keys:
        choice = trial["resp_map"][choice_key]
        correct = choice == solution

        # * Feedback
        if trial["trial_type"] == "practice":
            choice_idx = avail_choices.index(choice)
            choice_x_pos = x_pos["avail_choice"][choice_idx]

            solution_idx = avail_choices.index(solution)
            solution_x_pos = x_pos["avail_choice"][solution_idx]

            get_feedback(
                win,
                correct,
                choice_x_pos,
                solution_x_pos,
                y_pos_choices,
                img_size,
                all_imgs,
                timings.feedback_duration,
            )

    elif choice_key == "timeout":
        correct = "invalid"
        choice = "invalid"
        show_msg(win, text=messages["timeout"])
        core.wait(timings.feedback_duration)

    # * Abort trial during response period
    elif choice_key == "escape":
        abort_decision = abort_trial(win, eeg_device, eye_tracker, allowed_keys)
        if abort_decision == "abort":
            record_event("experiment_aborted", eeg_device, eye_tracker)
            return "abort"
        else:
            correct = "invalid"
            choice = "invalid"
    else:
        correct = "invalid"
        choice = "invalid"
        show_msg(
            win,
            text=messages["invalid_key"].format(keys=allowed_keys_str),
            keys=allowed_keys,
        )

    record_event("trial_end", eeg_device, eye_tracker)

    trial_data = {
        # "trial_idx": trialN,
        "subj_id": sess_info["subj_id"],
        "trial_type": trial["trial_type"],
        "item_id": trial["item_id"],
        "trial_onset_time": trial_onset_time,
        "series_end_time": series_end_time,
        "choice_onset_time": choice_onset_time,
        "rt": response_time,
        "rt_global": rt_global,
        "choice_key": choice_key,
        "solution_key": invert_dict(trial["resp_map"])[solution],
        "choice": choice,
        "solution": solution,
        "correct": correct,
        "pattern": trial["pattern"],
    }

    # * For debugging
    ic(
        trial["item_id"],
        choice,
        solution,
        correct,
        response_time,
    )

    return trial_data


def init_experiment(
    img_size: Tuple[int, int] = (256, 256),
    refresh_rate: int = refresh_rate,
) -> Tuple:
    """
    Initializes the experiment by setting up the psychopy window, loading images,
    and preparing the EEG device and eye tracker.

    Parameters:
    img_size (Tuple[int, int]): The size of the images. Default is (256, 256).
    refresh_rate (int): The desired refresh rate of the display.

    Returns:
    tuple: Various objects and parameters needed for running the experiment.
    """
    # * ################ DIALOG BOX -> PARTICIPANT & SESSION NUMBER ################
    sess_info = show_dialogue()
    edf_file = sess_info["edf_file"]

    session_dir = results_dir / f'subj_{sess_info["subj_id"]}/sess_{sess_info["sess"]}'
    session_dir.mkdir(parents=True)

    sess_info_file = session_dir / f'{sess_info["sess_id"]}.json'

    # * ############################ SETTING UP EXPERIMENT #############################

    # * Create a monitor object with your monitor's specifications
    config_res = tuple(exp_config["local"]["monitor"]["resolution"])
    monitors_info = get_monitors_info()
    my_monitor = [info for info in monitors_info if info["primary"] == True][0]
    resolution = my_monitor["res"]

    if resolution != config_res:
        print(
            "WARNING: Monitor resolution does not match the configuration file\n"
            f"Current resolution: {resolution}, configured resolution: {config_res}"
        )
        user_choice = input("continue? y/n: ").lower()
        if user_choice != "y":
            sys.exit()

    my_monitor = monitors.Monitor(
        name=my_monitor["name"],
        distance=sess_info["eye_screen_dist"] * 10,  # * psychopy requires cm
    )
    my_monitor.setSizePix(resolution)
    window_size = resolution

    max_row_items = 9
    max_width = resolution[0] / max_row_items
    height_factor = 3
    max_height = resolution[1] / height_factor

    imgs_info = prepare_images(
        config_dir / "images/standardized",
        img_dir,
        size=img_size,
    )

    # print(f"{imgs_info = }")
    images = {img_path.stem: img_path for img_path in img_dir.iterdir()}
    icon_names = list(images.keys())

    img_size = Image.open(images[icon_names[0]]).size

    assert int(window_size[1] / img_size[1]) >= height_factor, "Window height too small"

    # * Load sequences
    sess_n = int(sess_info["sess"])

    sequences_file = sequences_dir / f"session_{sess_n}.csv"
    sequences = pd.read_csv(sequences_file, dtype=str)
    n_sequences = len(sequences)

    np.random.seed(rand_seed)
    sequences = sequences.sample(frac=1, random_state=rand_seed)

    sequences["item_id"] = sequences["item_id"].astype(int)
    sequences["masked_idx"] = sequences["masked_idx"].astype(int)

    if sess_n == 1:
        practice_sequences = get_practice_sequences()

        practice_block = prepare_sess(
            images=images,
            icons=icon_names,
            sequences=practice_sequences,
            allowed_keys=allowed_keys,
            window_size=window_size,
            block_size=len(practice_sequences),
        )
    else:
        practice_sequences = None
        practice_block = None

    trial_blocks = prepare_sess(
        images=images,
        icons=icon_names,
        sequences=sequences,
        allowed_keys=allowed_keys,
        window_size=window_size,
        block_size=block_size,
    )

    # * Delete dataframes to free up memory
    del sequences, practice_sequences

    trial_blocks = (block for block in trial_blocks)

    sess_info.update({"window_size": window_size, "img_size": img_size, "Notes": ""})

    with open(sess_info_file, "w") as f:
        json.dump(sess_info, f, indent=4)

    # * ################ EEG & Eye Tracker  ################
    eeg_conf = exp_config["local"]["EEG"]
    eeg_conf = namedtuple("eeg_conf", eeg_conf.keys())(*eeg_conf.values())
    eeg_device = EEGcap(eeg_conf.read_add, eeg_conf.write_add, valid_events)
    eeg_device.connect()

    eye_tracker = EyeTracker(eye_tracker_add)
    eye_tracker.open_file(edf_file)

    win = visual.Window(
        window_size,
        winType="pyglet",
        monitor=my_monitor,
        fullscr=fullscr,
        screen=0,
        color=[0, 0, 0],
        units="pix",
    )

    # win.mouseVisible = False
    # win.winHandle.set_mouse_visible(False)
    # win.winHandle.set_mouse_position(-1, -1)

    win_flip(win)

    if refresh_rate is None:
        refresh_rate = win.getActualFrameRate()
        print("WARNING: Desired Refresh rate not specified. Using measured frame rate.")
        print(f"Measured frame rate: {refresh_rate} Hz")

    print(f"Configured refresh rate: {refresh_rate} Hz")

    eye_tracker.setup(
        win, eye=sess_info["eye"], screen_distance=sess_info["eye_screen_dist"]
    )
    eye_tracker.set_calib_env(win)

    show_msg(
        win,
        text="Press enter twice to start the eye tracker calibration",
        keys=["return"],
    )
    eye_tracker.calibrate()
    eye_tracker.start_recording()

    # * Create a fixation cross stimulus
    fix_cross = visual.TextStim(
        win,
        color="black",
        text="+",
        height=0.05 * window_size[1],
    )

    # * Welcome message
    pressed_key = show_msg(
        win,
        text=messages["welcome"].format(keys=allowed_keys_str, n=n_sequences),
        keys=allowed_keys + ["escape"],
    )

    if pressed_key[0] == "escape":
        abort_decision = abort_trial(win, eeg_device, eye_tracker, allowed_keys)

        if abort_decision == "abort":
            record_event("experiment_aborted", eeg_device, eye_tracker)
            terminate_exp(win, eeg_device, eye_tracker, {}, sess_info, session_dir)

    y_pos = [-img_size[1], img_size[1]]

    assert all([(pos + img_size[1]) <= window_size[1] for pos in y_pos])

    return (
        win,
        sess_info,
        session_dir,
        practice_block,
        trial_blocks,
        images,
        y_pos,
        fix_cross,
        img_size,
        window_size,
        eeg_device,
        eye_tracker,
    )


def run_trials(
    trials,
    win,
    fix_cross,
    images,
    img_size,
    global_clock,
    y_pos_sequence,
    y_pos_choices,
    blockN,
    eeg_device,
    eye_tracker,
    sess_data,
    sess_info,
    session_dir,
):
    for trial in trials:
        # trial_count += 1
        # trialN = trial_count

        # * Display the fixation cross in the middle of the screen
        fix_cross.draw()
        win_flip(win)

        intertrial_time = rng.integers(iti[0], iti[1] + 1, size=1)[0]

        intertrial_press = event.waitKeys(
            maxWait=intertrial_time, keyList=["escape"], clearEvents=True
        )

        # * Abort trial during intertrial period
        if intertrial_press:
            abort_decision = abort_trial(win, eeg_device, eye_tracker, allowed_keys)
            if abort_decision == "abort":
                # * Abort the experiment
                terminate_exp(
                    win,
                    eeg_device,
                    eye_tracker,
                    sess_data,
                    sess_info,
                    session_dir,
                )
            else:
                # * Continue the experiment
                core.wait(1)

                fix_cross.draw()
                win_flip(win)

                intertrial_press = event.waitKeys(
                    maxWait=intertrial_time,
                    keyList=["escape"],
                    clearEvents=True,
                )


        trial_data = run_trial(
            win,
            trial,
            # trialN,
            images,
            img_size,
            eeg_device,
            eye_tracker,
            global_clock,
            sess_info,
            y_pos_sequence,
            y_pos_choices,
        )

        # * Abort trial during response period
        if trial_data == "abort":
            # * Abort the experiment
            terminate_exp(
                win, eeg_device, eye_tracker, sess_data, sess_info, session_dir
            )
        else:
            trial_data.update({"blockN":blockN, "iti":intertrial_time})
            sess_data.append(trial_data)
            # sess_data[trialN] = {"blockN": blockN}
            # sess_data[trialN].update(trial_data)
            # sess_data[trialN].update({"intertrial_time": intertrial_time})


def main():
    (
        win,
        sess_info,
        session_dir,
        practice_block,
        trial_blocks,
        images,
        y_pos,
        fix_cross,
        img_size,
        window_size,
        eeg_device,
        eye_tracker,
    ) = init_experiment()

    # * Vertical positions for displaying the sequence and choices
    y_pos_choices, y_pos_sequence = y_pos

    # * ################################################################################
    # * START OF EXPERIMENT
    # * ################################################################################

    global_clock = core.Clock()

    # * Send signal to EEG amplifier & eye tracker to indicate start of experiment
    record_event("exp_start", eeg_device, eye_tracker)

    # practice_data = {}
    # sess_data = {}

    practice_data = []
    sess_data = []

    try:
        # * Main experiment loop
        if practice_block:
            record_event("block_start", eeg_device, eye_tracker)

            show_msg(
                win,
                messages["practice_start"].format(keys=allowed_keys_str),
                keys=allowed_keys,
            )

            trials = practice_block[0]
            blockN = -1

            run_trials(
                trials,
                win,
                fix_cross,
                images,
                img_size,
                global_clock,
                y_pos_sequence,
                y_pos_choices,
                blockN,
                eeg_device,
                eye_tracker,
                practice_data,
                sess_info,
                session_dir,
            )

            record_event("block_end", eeg_device, eye_tracker)

            pd.DataFrame(practice_data).to_csv(
                session_dir / f'{sess_info["sess_id"]}-practice.csv'
            )

            show_msg(
                win,
                messages["practice_end"].format(keys=allowed_keys_str),
                keys=allowed_keys,
            )

        for blockN, trials in enumerate(trial_blocks):
            record_event("block_start", eeg_device, eye_tracker)
            run_trials(
                trials,
                win,
                fix_cross,
                images,
                img_size,
                global_clock,
                y_pos_sequence,
                y_pos_choices,
                blockN,
                eeg_device,
                eye_tracker,
                sess_data,
                sess_info,
                session_dir,
            )
            # * END OF BLOCK -> PAUSE
            show_msg(
                win,
                messages["block_end"].format(blockN=blockN + 1, keys=allowed_keys_str),
                keys=allowed_keys,
            )

            record_event("block_end", eeg_device, eye_tracker)

        terminate_exp(win, eeg_device, eye_tracker, sess_data, sess_info, session_dir)

    except Exception as e:
        # * Print the error in the console
        print(f"\n\nUnexpected error:{e}\n\n")

        # * Display an error message on screen
        show_msg(
            win,
            messages["error"],
            keys=["return"],
        )

        print(traceback.format_exc(), "\n\n")

        # * Abort the experiment
        terminate_exp(win, eeg_device, eye_tracker, sess_data, sess_info, session_dir)


if __name__ == "__main__":
    main()
