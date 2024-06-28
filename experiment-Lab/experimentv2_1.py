from pathlib import Path
import sys
import time
import pylink
import os
from psychopy import visual, core, event, logging, monitors, gui
from string import ascii_letters, digits
import sys
import json
from icecream import ic
import numpy as np
import pandas as pd
from PIL import Image
import platform
import re
import inspect
import pickle
from typing import Dict, List, Tuple, Union
from collections import namedtuple
from tqdm.auto import tqdm
import code

wd = Path(__file__).parent
os.chdir(wd)
from utils import sess_prep2, get_monitors_info
from devices import EEGcap, EyeTracker

comments_desc = {
    "! TEMP": "signals a temporary change that should be reverted later",
    "! IMPLEMENT": "reminder to implement a feature",
}

# * ####################################################################################
# * GLOBAL VARIABLES
# * ####################################################################################
logging.console.setLevel(logging.CRITICAL)

# * IP address of the EyeLink tracker; default = "100.1.1.1"
eye_tracker_add = "100.1.1.1"

# * set to True if you are using a Mac with a retina display
use_retina = False

# * set to True if you want to check the configuration before starting the experiment
config_check = True

# icecream config
ic.configureOutput(includeContext=False, prefix="")

# * ################ CONFIGURATION ################
wd = Path(__file__).parent
results_dir = wd / "results"
results_dir.mkdir(exist_ok=True)

config_dir = wd.parent / "config"
img_dir = config_dir / "images/resized"

# * Load experiment config
with open(config_dir / "experiment_config.json") as f:
    exp_config = json.load(f)

block_size = 21

sequences_file = "abstract_reasoning-sequences.csv"
# * ####################################################################################


# * ################ FUNCTIONS ################
def check_refresh_rate(win):
    win.recordFrameIntervals = True
    pass


def check_config():
    from pprint import pprint

    print("Review experiment configuration:\n")
    pprint(exp_config, sort_dicts=False)

    choice = input("Continue? (y/n): ").lower()

    if choice != "y":
        raise SystemExit("Experiment aborted by user")


def win_flip(win, cursor=False, bg_color=(0, 0, 0)):
    win.fillColor = bg_color
    win.winHandle.set_mouse_visible(cursor)
    win.flip()


def display_images_sequentially(
    win,
    images,
    # fix_cross,
    eeg_device,
    eye_tracker,
    event_name,
    pres_duration=None,
    pres_frames=None,
    order=None,
):
    assert bool(pres_duration) ^ bool(
        pres_frames
    ), "You can only specify pres_duration or pres_frames, not both"

    # stim_pres_times = {}

    if order is None:
        order = range(len(images))

    if pres_duration:
        for img_idx in order:
            images[img_idx].draw()
            # fix_cross.draw()
            win_flip(win)

            eeg_device.send(event_name)
            eye_tracker.send(event_name)
            # stim_pres_times[f"{idx}-{img_name}-time"] = global_clock.getTime()
            core.wait(pres_duration)

    elif pres_frames:
        for img_idx in order:
            for _ in range(pres_frames):
                images[img_idx].draw()
                # fix_cross.draw()
                win_flip(win)
                eeg_device.send(event_name)
                eye_tracker.send(event_name)
    win_flip(win)  # ! Not sure if this is necessary


def get_feedback(
    win, correct, choice_x_pos, solution_x_pos, y_pos, img_size, all_imgs, duration
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


def end_block(win, blockN=None, keys=None):
    # !TODO: check and implement this function
    if blockN is None:
        text = "End of block."
    else:
        text = f"End of block {blockN}."

    text += (
        "\nTake a break.\n"
        "Place your fingers on the a, x, m, l keys and press one of them to continue."
    )

    show_msg(win, text, keys=keys)


def sess_prep(
    images: Dict[str, str],
    icons: List[str],
    sequences: pd.DataFrame,
    allowed_keys: List[str],
    window_size: List[int],
) -> Tuple[Dict, Dict]:

    assert (
        len(set([Image.open(im).size for im in images.values()])) == 1
    ), "Images must have same size"

    solution_mask = "question-mark"

    img_size = Image.open(images[icons[0]]).size
    x_positions = {}
    resp_mapping = {}

    match_cols_choice = lambda x: re.compile("choice\d{1,2}", re.IGNORECASE).search(x)
    match_cols_seq = lambda x: re.compile("figure\d{1,2}", re.IGNORECASE).search(x)

    seq_cols = [c for c in sequences.columns if match_cols_seq(c)]
    choice_cols = [c for c in sequences.columns if match_cols_choice(c)]

    x_pos_seq = {}
    new_sequences = sequences.copy()

    for seq_length in range(1, len(seq_cols) + 1):  # * +1 for 0 index
        # * Calculate the total width of images & blank spaces for the item set
        total_width = img_size[0] * seq_length
        empty_space_width = window_size[0] - total_width
        sep_space_width = empty_space_width / (seq_length + 1)
        x_shift = img_size[0] + sep_space_width
        start_pos = sep_space_width + img_size[0] / 2

        positions = [
            start_pos + (i * x_shift) - window_size[0] / 2 for i in range(seq_length)
        ]
        x_pos_seq[seq_length] = {}
        x_pos_seq[seq_length]["pos"] = [round(pos, 3) for pos in positions]  # positions
        x_pos_seq[seq_length]["sep_space_width"] = round(sep_space_width, 3)

    for idx_row, row in tqdm(new_sequences.iterrows()):
        x_positions[idx_row] = {}

        avail_choices = row.loc[choice_cols].dropna().tolist()
        sequence = row.loc[seq_cols].dropna().tolist()

        # * replace solution with the mask
        new_sequences.loc[idx_row, seq_cols[row["masked_idx"]]] = solution_mask

        # * Calculate the total width of images & blank spaces for the item set
        pos_info = x_pos_seq[len(sequence)]
        x_positions[idx_row]["items_set"] = pos_info["pos"]
        sep_space_width = x_pos_seq[len(sequence)]["sep_space_width"]

        # * Calculate the total width of images & blank spaces for the available choices
        # * keep same sep_space_width as for the item set => number of choices must be
        # * <= number of items
        x_shift = img_size[0] + sep_space_width
        total_width = x_shift * len(avail_choices)
        empty_space_width = window_size[0] - total_width
        start_pos = empty_space_width / 2 + x_shift / 2

        x_positions[idx_row]["avail_choice"] = [
            start_pos + (i * x_shift) - window_size[0] / 2
            for i in range(len(avail_choices))
        ]

        resp_mapping[idx_row] = {k: v for k, v in (zip(allowed_keys, avail_choices))}

    return new_sequences, x_positions, resp_mapping


def show_msg(
    win: visual.window.Window, text: str, kwargs: dict = None, keys: list = None
):
    """Show message on psychopy window"""
    # win_flip(win)
    kwargs = {} if kwargs is None else kwargs
    msg = visual.TextStim(win, text, **kwargs)
    # TODO : color=txt_color, wrapWidth=scn_width / 2)
    msg.draw()
    win.flip()

    if keys:
        pressed_key = event.waitKeys(keyList=keys)
        win_flip(win)
        return pressed_key


def show_dialogue():
    # Prompt user to specify an EDF data filename
    # before we open a fullscreen window
    # dlg_title = "Enter EDF File Name"
    # dlg_prompt = (
    #     "Please enter a file name with 8 or fewer characters\n"
    #     + "[letters, numbers, and underscore]."
    # )

    # loop until we get a valid filename
    while True:
        dlg = gui.Dlg()
        # dlg.addText(dlg_prompt)
        dlg.addText("Subject info")
        dlg.addField(key="subj_id", label="ID:", required=True)

        dlg.addText("Experiment Info")
        dlg.addField("group", "Group:", choices=["pilot", "experiment"])
        dlg.addField("sess", "Session:", choices=["1", "2", "3", "4", "5"])

        dlg.addText("Eye tracking")
        dlg.addField(key="fname", label="File Name:")
        dlg.addField(
            "mode",
            "Tracking mode:",
            choices=["monocular", "binocular"],
            initial="monocular",
        )
        dlg.addField("eye", "Eye tracked:", choices=["left", "right", "both"])
        dlg.addField(key="eye_screen_dist", label="Eye to Screen Distance (cm):")

        # ! IMPLEMENT: validate fields -> subj_id should be required

        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()

        if dlg.OK:  # if ok_data is not None
            print("EDF data filename: {}".format(ok_data["fname"]))
        else:
            print("user cancelled")
            core.quit()
            sys.exit()

        # get the string entered by the experimenter
        tmp_str = dlg.data["fname"]
        # strip trailing characters, ignore the ".edf" extension
        edf_fname = tmp_str.rstrip().split(".")[0]

        # if dlg.data["subj_id"] == "":
        #     print("ERROR: Subject ID is required")

        # check if the filename is valid (length <= 8 & no special char)
        allowed_char = ascii_letters + digits + "_"

        if not all([c in allowed_char for c in edf_fname]):
            print("ERROR: Invalid EDF filename")
        elif len(edf_fname) > 8:
            print("ERROR: EDF filename should not exceed 8 characters")
        else:
            break

    ok_data["edf_fname"] = edf_fname

    return ok_data


def get_timestamp(fmt="%Y_%m_%d-%H_%M_%S"):
    return time.strftime(fmt, time.localtime())


def terminate_task(win, eeg_device, eye_tracker, session_folder: str, edf_file: str):
    """Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """
    eeg_device.send("experiment_end")
    eye_tracker.send("experiment_end")

    eye_tracker.disconnect(session_folder)
    # eye_tracker.get_file(session_folder, edf_file)
    # eye_tracker.edf2asc(edf_file) # TODO

    # # Show a file transfer message on the screen
    # msg = "EDF data is transferring from EyeLink Host PC..."
    # # show_msg(win, msg, wait_for_keypress=False)
    # print(msg)

    # Download the EDF data file from the Host PC to a local data folder
    # parameters: source_file_on_the_host, destination_file_on_local_drive

    # TODO Implement commented logic below
    # # * ################ SAVE DATA ################
    # fname = f"subj_{subj_id}-sess_{sess_id}"
    # fpath = session_dir / f"{fname}.csv"
    # print(f"Exporting results to {fpath} ...")

    # pd.DataFrame(sessData).T.to_csv(fpath)

    # # * ################ END OF EXPERIMENT ################
    # end_screen = visual.TextStim(
    #     win,
    #     text="End of experiment. Thank you for participating!",
    #     # height=0.05 * window_size[1],
    # )
    # end_screen.draw()
    # win_flip(win)
    # core.wait(5)

    # terminate_task(win, eeg, eye_tracker, session_dir, edf_file)

    # eye_tracker.get_file(edf_file, session_dir)
    # eye_tracker.edf2asc(str(session_dir / edf_file))

    # code.interact(banner="Start", local=locals(), exitmsg="End")

    # * close the PsychoPy window
    win.close()

    # * quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial(win, eeg_device, eye_tracker, session_folder, edf_file):
    event_name = "trial_aborted"
    eye_tracker.send(event_name)
    eeg_device.send(event_name)

    choice = show_msg(
        win,
        text="Trial aborted. Press enter to continue the experiment or escape to quit.",
        keys=["return", "escape"],
    )

    # choice = event.waitKeys(keyList=["return", "escape"])

    if choice[0] == "escape":
        event_name = "experiment_aborted"
        eeg_device.send(event_name)
        eye_tracker.send(event_name)

        terminate_task(win, eeg_device, eye_tracker, session_folder, edf_file)

    elif choice[0] == "return":
        show_msg(win, "Starting next trial...")


def invert_dict(d: dict):
    return {v: k for k, v in d.items()}


def get_blocks(sequences: pd.DataFrame, block_size:int=20):        
    if (remainder := len(sequences) % block_size) != 0:
        n_blocks = (len(sequences) - remainder) / block_size
        blocks = np.array_split(sequences[:-remainder], n_blocks)
        block_trials = sequences.iloc[-remainder:]
        blocks.append(block_trials)
        print(f"WARNING: uneven block sizes: {[len(b) for b in blocks]}")

    trial_blocks = []
    for block in blocks:
        trials = []
        for idx_row, row in block.iterrows():
            trial = row.to_dict()
            trial.update(
                {
                    "item_id": row["itemid"],
                    "x_pos": x_positions[idx_row],
                    "resp_map": resp_mapping[idx_row],
                    "trial_type": "",
                }
            )
            trial["seq_order"] = [int(i) for i in trial["seq_order"] if i.isdigit()]
            trial["choice_order"] = [
                int(i) for i in trial["choice_order"] if i.isdigit()
            ]

            trials.append(trial)
        trial_blocks.append(trials)

    trial_blocks = (block for block in trial_blocks)


def main(results_dir, sequences_file):
    # * ################ SETTING UP EXPERIMENT ################
    # check_config() if config_check else None
    # db = Database(exp_config['global']["database_path"])

    allowed_keys_str = ", ".join(exp_config["local"]["allowed_keys"])

    images = {img_path.stem: img_path for img_path in img_dir.iterdir()}
    icon_names = list(images.keys())

    img_size = Image.open(images[icon_names[0]]).size

    sequences = pd.read_csv(sequences_file)
    sequences = sequences.sample(frac=1, random_state=0)

    # * Create a monitor object with your monitor's specifimcations
    monitors_info = get_monitors_info()
    my_monitor = [info for info in monitors_info if info["primary"] == True][0]
    res_pixels = my_monitor["res"]
    my_monitor = monitors.Monitor(name=my_monitor["name"])
    # TODO: add width=..., distance=...
    my_monitor.setSizePix(res_pixels)
    window_size = res_pixels

    assert int(window_size[1] / img_size[1]) > 3, "Window height too small"

    new_sequences, x_positions, resp_mapping = sess_prep(
        images=images,
        icons=icon_names,
        sequences=sequences,
        allowed_keys=exp_config["local"]["allowed_keys"],
        window_size=window_size,
    )

    blocks = get_blocks(sequences, block_size)

    # * Create a window
    # ! { TEMP
    # window_size = [1080, 720]
    fullscr = True
    # ! TEMP }

    # pres_frames * frame_dur
    # valid_timings = np.arange(frame_dur, 10, frame_dur)
    # precision = 3
    # # Select numbers that are 'accurate' after rounding
    # valid_timings = valid_timings[np.round(valid_timings, precision) == valid_timings]

    # * ################ DIALOG BOX -> PARTICIPANT & SESSION NUMBER ################
    sessInfo = show_dialogue()
    sessInfo["date"] = get_timestamp()

    edf_fname = sessInfo["edf_fname"]
    edf_file = f"{edf_fname}.EDF"

    subj_id = str(sessInfo["subj_id"]).zfill(2)
    sess_id = str(sessInfo["sess"]).zfill(2)

    sess_id = f"{sess_id}-{get_timestamp(fmt='%Y%m%d_%H%M%S')}"

    results_dir = Path(results_dir)
    session_dir = results_dir / f"subj_{subj_id}/sess_{sess_id}"
    session_dir.mkdir(parents=True)

    # * ################ EEG & Eye Tracker  ################
    eeg_conf = exp_config["local"]["EEG"]
    eeg_conf = namedtuple("eeg_conf", eeg_conf.keys())(*eeg_conf.values())
    eeg = EEGcap(
        eeg_conf.read_add, eeg_conf.write_add, exp_config["local"]["event_IDs"]
    )
    eeg.connect()

    eye_tracker = EyeTracker(eye_tracker_add)
    eye_tracker.open_file(edf_file)

    # * ################ START OF EXPERIMENT ################
    sessData = {}

    try:
        win = visual.Window(
            window_size,
            winType="pyglet",
            monitor=my_monitor,
            fullscr=fullscr,
            screen=0,
            color=[0, 0, 0],
            units="pix",
        )
        win.flip()  # TODO: check if this is necessary

        if not (refresh_rate := exp_config["local"]["monitor"].get("refresh_rate")):
            refresh_rate = win.getActualFrameRate()
            print(f"Measured frame rate: {refresh_rate}Hz")

        print(f"Configured refresh rate: {refresh_rate}Hz")

        # frame_dur = 1 / refresh_rate

        timings = exp_config["global"]["timings"]
        timings = namedtuple("Timings", timings.keys())(*timings.values())
        iti = timings.intertrial_interval

        # * Presentation time in frames
        # pres_frames = int(timings.pres_duration * refresh_rate)

        eye_tracker.setup(win, eye=sessInfo["eye"])
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
        instr1 = """You are going to solve __ abstract reasoning problems like the one below. Your goal is to continue the sequence in the top row with one of the four options in the bottom row.
        Use the keys a, x, m, l to select one of these options from left to right.
        You will perform two practice trials with feedback before the start of the experiment.
        Place your fingers on the a, x, m, l keys and press one of them to start the practice.
        """

        pressed_key = show_msg(
            win,
            text=instr1,
            keys=exp_config["local"]["allowed_keys"] + ["escape"],
        )

        if pressed_key[0] == "escape":
            abort_trial(win, eeg, eye_tracker, session_dir, edf_file)

        fix_cross.draw()
        win_flip(win)

        y_pos = [-img_size[1], img_size[1]]
        y_pos_choices, y_pos_sequence = y_pos

        assert all([(pos + img_size[1]) <= window_size[1] for pos in y_pos])

        # * Global clock
        global_clock = core.Clock()

        # * Trial Response clock
        response_clock = core.Clock()

        # * Send a signal to the EEG amplifier to indicate the start of the experiment
        eeg.send("exp_start")
        eye_tracker.send("exp_start")

        # * Main experiment loop
        for blockN, trials in enumerate(trial_blocks):
            eeg.send("block_start")
            eye_tracker.send("block_start")

            for trialN, trial in enumerate(trials):
                # * Intertrial interval
                sequence = [v for k, v in trial.items() if "figure" in k]

                masked_idx = trial["masked_idx"]
                solution = trial["solution"]
                sequence[masked_idx] = "question-mark"

                # ic(trialN, sequence, solution)

                x_pos = trial["x_pos"]
                avail_choices = list(trial["resp_map"].values())

                intertrial_time = np.random.randint(iti[0], iti[1] + 1, size=1)[0]
                core.wait(intertrial_time)

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

                # * Display the fixation cross in the middle of the screen
                fix_cross.draw()
                win_flip(win)

                eeg.send("trial_start")
                eye_tracker.send("trial_start")

                # * Displaying Sequence items one by one
                display_images_sequentially(
                    win=win,
                    images=sequence_imgs,
                    # fix_cross=fix_cross,
                    eeg_device=eeg,
                    eye_tracker=eye_tracker,
                    event_name="stim-flash_sequence",
                    pres_duration=timings.pres_duration,
                    order=trial["seq_order"],
                )

                # * Displaying Choice items one by one
                display_images_sequentially(
                    win=win,
                    images=avail_choices_imgs,
                    # fix_cross=fix_cross,
                    eeg_device=eeg,
                    eye_tracker=eye_tracker,
                    event_name="stim-flash_choices",
                    pres_duration=timings.pres_duration,
                    order=trial["choice_order"],
                )

                series_end_time = global_clock.getTime()

                # * Displaying all items + choices at once
                for img in all_imgs:
                    img.draw()
                win_flip(win)

                eeg.send("stim-all_stim")
                eye_tracker.send("stim-all_stim")

                # * Start response clock
                response_clock.reset()
                # * for safety / sanity check
                choice_onset_time = global_clock.getTime()

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

                eeg.send(choice_key)
                eye_tracker.send(choice_key)

                if choice_key == "escape":
                    abort_trial(win, eeg, eye_tracker, session_dir, edf_file)

                # * Check if pressed key is allowed and choice is correct
                elif choice_key in exp_config["local"]["allowed_keys"]:
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
                    text = "Timeout"
                    # visual.TextStim(win, text=text).draw()
                    # win_flip(win)
                    show_msg(win, text=text)
                    core.wait(timings.feedback_duration)

                else:
                    correct = "invalid"
                    choice = "invalid"
                    text = f"Invalid key pressed, make sure to use these keys: {allowed_keys_str}"
                    # visual.TextStim(win, text=text).draw()
                    # win_flip(win)
                    show_msg(win, text=text)
                    core.wait(timings.feedback_duration)

                eeg.send("trial_end")
                eye_tracker.send("trial_end")

                sessData[trialN] = {
                    "trial_idx": trialN,
                    "participant_id": sessInfo["subj_id"],
                    "trial_type": trial["trial_type"],
                    "item_id": trial["item_id"],
                    "trial_onset_time": trial_onset_time,  # float(trial_onset_time),
                    "series_end_time": series_end_time,  # float(series_end_time),
                    "choice_onset_time": choice_onset_time,  # float(choice_onset_time),
                    "rt": response_time,  # float(response_time),
                    "rt_global": rt_global,  # float(rt_global),
                    "choice_key": choice_key,
                    "solution_key": invert_dict(trial["resp_map"])[solution],
                    "choice": choice,
                    "solution": solution,
                    "correct": correct,
                    # "stim_pres_times": ", ".join([str(i) for i in (stim_pres_times.values())]),
                    "pattern": trial["pattern"],
                    "intertrial_time": intertrial_time,  # int(intertrial_time),
                }

                fix_cross.draw()
                win_flip(win)

                # * INSERT DATA INTO DATABASE
                # db.insert_results_pilot([list(sessData[trialN].values())])
                # * For debugging
                ic(trial["item_id"], choice, response_time, intertrial_time)

            # * BLOCK END -> PAUSE
            end_block(win, blockN=blockN, keys=exp_config["local"]["allowed_keys"])
            eeg.send("block_end")
            eye_tracker.send("block_end")

    except Exception as e:
        print("Unexpected error:", e)
        # terminate_task() #TODO


if __name__ == "__main__":
    results_dir = wd / "results"
    results_dir.mkdir(exist_ok=True)

    main(results_dir=results_dir, sequences_file=sequences_file)
