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

# * IP address of the EyeLink tracker; default = "100.1.1.1"
eye_tracker_add = "100.1.1.1"

# * set to True if you are using a Mac with a retina display
use_retina = False

# * set to True if you want to check the configuration before starting the experiment
config_check = True

fullscr = True

# icecream config
ic.configureOutput(includeContext=False, prefix="")

# * ################ CONFIGURATION ################
wd = Path(__file__).parent
results_dir = wd / "results/raw"
results_dir.mkdir(parents=True, exist_ok=True)

config_dir = wd.parent / "config"
img_dir = wd / "images"

# * Load experiment config
with open(config_dir / "experiment_config.json") as f:
    exp_config = json.load(f)

block_size = 20

sequences_file = wd / "sequences/sequences1.csv"

valid_events = exp_config["local"]["event_IDs"]
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
    win.winHandle.set_mouse_visible(cursor)
    win.mouseVisible = cursor
    win.winHandle.set_mouse_position(-1, -1)
    win.fillColor = bg_color
    win.flip()


def record_event(event_name, eeg_device, eye_tracker):
    print(f"{event_name = } SENT")
    eeg_device.send(event_name)
    eye_tracker.send(event_name)


def display_images_sequentially(
    win,
    images,
    eeg_device,
    eye_tracker,
    event_name,
    fix_cross: visual.TextStim = None,
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
            if fix_cross is not None:
                fix_cross.draw()
            win_flip(win)

            record_event(event_name, eeg_device, eye_tracker)
            # stim_pres_times[f"{idx}-{img_name}-time"] = global_clock.getTime()
            core.wait(pres_duration)

    elif pres_frames:
        for img_idx in order:
            for _ in range(pres_frames):
                images[img_idx].draw()
                if fix_cross is not None:
                    fix_cross.draw()
                win_flip(win)
                record_event(event_name, eeg_device, eye_tracker)
    # win_flip(win)  # ! Not sure if this is necessary


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


def show_msg(
    win: visual.window.Window, text: str, kwargs: dict = None, keys: list = None
):
    """Show message on psychopy window"""
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
        # tmp_str = dlg.data["edf_fname"]
        # edf_fname = tmp_str.rstrip().split(".")[0]
        edf_fname = dlg.data["edf_fname"].rstrip().split(".")[0]

        # if dlg.data["subj_id"] == "":
        #     print("ERROR: Subject ID is required")

        # * check if the filename is valid (length <= 8 & no special char)
        allowed_char = ascii_letters + digits + "_"

        edf_name_conds = [
            all([c in allowed_char for c in edf_fname]),
            1 <= len(edf_fname) <= 8,
        ]

        if not all(edf_name_conds):
            print(
                "ERROR: Invalid EDF filename. Name should be 1 to 8 characters long and contain only letters, numbers, and underscores."
            )
        else:
            break

    sess_info["edf_file"] = f"{edf_fname}.EDF"
    sess_info["subj_id"] = str(sess_info["subj_id"]).zfill(2)
    sess_info["sess"] = str(sess_info["sess"]).zfill(2)
    sess_info["date"] = get_timestamp("%Y%m%d_%H%M%S")
    sess_info["sess_id"] = (
        f"{sess_info['subj_id']}-{sess_info['sess']}-{sess_info['date']}"
    )

    return sess_info


def terminate_task(
    win,
    eeg_device,
    eye_tracker,
    sess_data: dict,
    sess_info: dict,
    session_dir: str,
):
    """Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """
    print("TERMINATING TASK, SENDING 'experiment_end'...", end="")

    event_name = "experiment_end"
    record_event(event_name, eeg_device, eye_tracker)
    print("DONE")

    session_dir = Path(session_dir)
    eye_tracker.get_file(sess_info["edf_file"], session_dir)
    eye_tracker.disconnect(session_dir)
    eye_tracker.edf2asc(session_dir / sess_info["edf_file"])

    # # * ################ SAVE DATA ################
    behav_fpath = session_dir / f"{sess_info['sess_id']}-behav.csv"
    print(
        f"Exporting behavioral results to: {behav_fpath.relative_to(session_dir.parents[3])}"
    )
    pd.DataFrame(sess_data).T.to_csv(behav_fpath)

    # # * ################ END OF EXPERIMENT ################
    show_msg(
        win,
        "End of experiment. Thank you for participating!\nPress enter to quit",
        keys=["return"],
    )
    # * close the PsychoPy window
    win.close()

    # * quit PsychoPy
    core.quit()
    # * quit Python
    sys.exit()


def abort_trial(
    win,
    eeg_device,
    eye_tracker,
    # sess_data: dict,
    # sess_info: dict,
    # session_dir: str,
):
    event_name = "trial_aborted"
    record_event(event_name, eeg_device, eye_tracker)

    choice = show_msg(
        win,
        text="Trial aborted. Press enter to continue the experiment or escape to quit.",
        keys=["return", "escape"],
    )

    if choice[0] == "escape":
        return "abort"
    elif choice[0] == "return":
        return "continue"


def testing():

    win = visual.Window(
        size=(2560, 1440),
        winType="pyglet",
        fullscr=True,
        screen=0,
        color=[0, 0, 0],
        units="pix",
    )
    # win.mouseVisible = False
    # win.winHandle.set_mouse_visible(False)
    # win_flip(win, cursor=True)
    core.wait(2)

    show_msg(win, "WELCOME, press enter to continue", keys=["return"])
    win_flip(win, cursor=True)

    for xpos in range(0, 900, 100):
        core.wait(1)
        win.winHandle.set_mouse_position(-1, -1)

    key = event.waitKeys()

    show_msg(win, "START")
    core.wait(2)
    win_flip(win)

    show_msg(win, "SCREEN 1")
    core.wait(2)
    win.flip(win)

    show_msg(win, "PRESS ANY KEY TO QUIT")
    key = event.waitKeys()

    win.close()
    core.quit()
    sys.exit()


def load_sequences(session: int, seed: int = None) -> pd.DataFrame:
    sequences_files = [f for f in (wd / "sequences").glob("*.csv")]
    sequences_files = {f.stem: f for f in sequences_files}

    sequences = pd.read_csv(sequences_files[f"sequences{session}"])
    return sequences


def main(results_dir, sequences_dir):
    abort = False

    # * ################ DIALOG BOX -> PARTICIPANT & SESSION NUMBER ################
    sess_info = show_dialogue()
    edf_file = sess_info["edf_file"]

    session_dir = results_dir / f"subj_{sess_info['subj_id']}/sess_{sess_info['sess']}"
    session_dir.mkdir(parents=True)

    sess_info_file = session_dir / f"{sess_info['sess_id']}.json"

    # * ################ SETTING UP EXPERIMENT ################
    # check_config() if config_check else None

    allowed_keys_str = ", ".join(exp_config["local"]["allowed_keys"])

    sequences_file = sequences_dir / f"session_{int(sess_info['sess'])}.csv"
    sequences = pd.read_csv(sequences_file)
    sequences = sequences.sample(frac=1, random_state=0)

    # * Create a monitor object with your monitor's specifimcations
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
        distance=sess_info["eye_screen_dist"] * 10,  # * psychopy wants cm
    )
    my_monitor.setSizePix(resolution)
    window_size = resolution

    max_row_items = 9
    max_width = resolution[0] / max_row_items
    height_factor = 3
    max_height = resolution[1] / height_factor

    imgs_info = prepare_images(
        # config_dir / "images/original",
        # config_dir / "images/selected_standardized",
        config_dir / "images/standardized",
        wd / "images",
        size=(256, 256),
        # size = (512, 512),
    )

    # print(f"{imgs_info = }")
    images = {img_path.stem: img_path for img_path in img_dir.iterdir()}
    icon_names = list(images.keys())

    img_size = Image.open(images[icon_names[0]]).size

    assert int(window_size[1] / img_size[1]) >= height_factor, "Window height too small"

    trial_blocks = prepare_sess(
        images=images,
        icons=icon_names,
        sequences=sequences,
        allowed_keys=exp_config["local"]["allowed_keys"],
        window_size=window_size,
        block_size=block_size,
    )

    sess_info.update({"window_size": window_size, "img_size": img_size, "Notes": ""})

    with open(sess_info_file, "w") as f:
        json.dump(sess_info, f)

    # * ################ EEG & Eye Tracker  ################
    eeg_conf = exp_config["local"]["EEG"]
    eeg_conf = namedtuple("eeg_conf", eeg_conf.keys())(*eeg_conf.values())
    eeg_device = EEGcap(eeg_conf.read_add, eeg_conf.write_add, valid_events)
    eeg_device.connect()

    eye_tracker = EyeTracker(eye_tracker_add)
    eye_tracker.open_file(edf_file)

    # * ################ START OF EXPERIMENT ################
    sess_data = {}

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
        # win.mouseVisible = False
        # win.winHandle.set_mouse_visible(False)
        # win.winHandle.set_mouse_position(-1, -1)

        win_flip(win)

        # TODO: fix this
        if not (refresh_rate := exp_config["local"]["monitor"].get("refresh_rate")):
            refresh_rate = win.getActualFrameRate()
            print(
                "WARNING: Refresh rate different from the one specified in config file"
            )
            print(f"Measured frame rate: {refresh_rate} Hz")

        print(f"Configured refresh rate: {refresh_rate} Hz")

        timings = exp_config["global"]["timings"]
        timings = namedtuple("Timings", timings.keys())(*timings.values())
        iti = timings.intertrial_interval

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

        # * ################ Run practice trials ################
        # if int(sess_info["sess"]) == 1:
        #     raise NotImplementedError("Practice trials not implemented yet")

        # * Welcome message
        instr1 = """You are going to solve {n} abstract reasoning problems. Your goal is to continue the sequence in the top row with one of the four options in the bottom row.
        Use the keys a, x, m, l to select one of these options from left to right.
        Place your fingers on the a, x, m, l keys and press one of them to continue.
        """.format(
            n=len(sequences)
        )

        pressed_key = show_msg(
            win,
            text=instr1,
            keys=exp_config["local"]["allowed_keys"] + ["escape"],
        )

        if pressed_key[0] == "escape":
            abort_decision = abort_trial(
                win, eeg_device, eye_tracker
            )  # , sess_data, sess_info, session_dir)

            if abort_decision == "abort":
                terminate_task(
                    win, eeg_device, eye_tracker, sess_data, sess_info, session_dir
                )

        y_pos = [-img_size[1], img_size[1]]
        y_pos_choices, y_pos_sequence = y_pos

        assert all([(pos + img_size[1]) <= window_size[1] for pos in y_pos])

        # * Global clock
        global_clock = core.Clock()

        # * Trial Response clock
        response_clock = core.Clock()

        # * Send a signal to the EEG amplifier to indicate the start of the experiment
        record_event("exp_start", eeg_device, eye_tracker)

        # * Main experiment loop
        for blockN, trials in enumerate(trial_blocks):
            record_event("block_start", eeg_device, eye_tracker)

            for trialN, trial in enumerate(trials):
                # * Display the fixation cross in the middle of the screen
                fix_cross.draw()
                win_flip(win)

                sequence = [v for k, v in trial.items() if "figure" in k]

                masked_idx = trial["masked_idx"]
                solution = trial["solution"]
                sequence[masked_idx] = "question-mark"

                # ic(trialN, sequence, solution)

                x_pos = trial["x_pos"]
                avail_choices = list(trial["resp_map"].values())

                intertrial_time = np.random.randint(iti[0], iti[1] + 1, size=1)[0]

                # core.wait(intertrial_time)
                intertrial_press = event.waitKeys(
                    maxWait=intertrial_time, keyList=["escape"], clearEvents=True
                )
                if intertrial_press:
                    abort_decision = abort_trial(win, eeg_device, eye_tracker)
                    if abort_decision == "abort":
                        abort = True
                        break
                    else:
                        intertrial_press = event.waitKeys(
                            maxWait=intertrial_time,
                            keyList=["escape"],
                            clearEvents=True,
                        )

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

                # * Displaying Sequence items one by one
                display_images_sequentially(
                    win=win,
                    images=sequence_imgs,
                    eeg_device=eeg_device,
                    eye_tracker=eye_tracker,
                    event_name="stim-flash_sequence",
                    # fix_cross=fix_cross,
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
                    # fix_cross=fix_cross,
                    pres_duration=timings.pres_duration,
                    order=trial["choice_order"],
                )

                series_end_time = global_clock.getTime()

                # * Displaying all items + choices at once
                for img in all_imgs:
                    img.draw()
                win_flip(win)

                record_event("stim-all_stim", eeg_device, eye_tracker)

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

                record_event(choice_key, eeg_device, eye_tracker)

                # * Check if pressed key is allowed and choice is correct
                if choice_key in exp_config["local"]["allowed_keys"]:
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
                    show_msg(win, text=text)
                    core.wait(timings.feedback_duration)

                elif choice_key == "escape":
                    abort_decision = abort_trial(win, eeg_device, eye_tracker)
                    if abort_decision == "abort":
                        abort = True
                    correct = "invalid"
                    choice = "invalid"

                else:
                    correct = "invalid"
                    choice = "invalid"
                    text = f"Invalid key pressed, make sure to use these keys: {allowed_keys_str}"
                    show_msg(win, text=text)
                    core.wait(timings.feedback_duration)

                record_event("trial_end", eeg_device, eye_tracker)

                sess_data[trialN] = {
                    "trial_idx": trialN,
                    "subj_id": sess_info["subj_id"],
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

                # * For debugging
                ic(
                    trial["item_id"],
                    choice,
                    solution,
                    correct,
                    response_time,
                    intertrial_time,
                )

                if abort:
                    terminate_task(
                        win, eeg_device, eye_tracker, sess_data, sess_info, session_dir
                    )

            # * BLOCK END -> PAUSE
            text = (
                f"End of block {blockN}"
                "\nTake a break\n"
                f"Place your fingers on the {', '.join(exp_config['local']['allowed_keys'])} keys \nand press one of them to continue"
            )
            show_msg(win, text, keys=exp_config["local"]["allowed_keys"])

            record_event("block_end", eeg_device, eye_tracker)

        terminate_task(win, eeg_device, eye_tracker, sess_data, sess_info, session_dir)

    except Exception as e:
        print("Unexpected error:", e)
        terminate_task(win, eeg_device, eye_tracker, sess_data, sess_info, session_dir)


if __name__ == "__main__":
    main(results_dir=results_dir, sequences_file=sequences_file)
