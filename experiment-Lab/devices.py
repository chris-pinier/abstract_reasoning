from icecream import ic
from psychopy import parallel, core
from utils import disable_decorator
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Union
import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from string import ascii_letters, digits
import sys

# ! TEMPORARY
disabled_funcs = {
    "EEGcap.connect": [False, "mock EEG device connected"],
    "EEGcap.send": [False, None],
    "EyeTracker.connect": [False, "mock Eye-tracker connected"],
    "EyeTracker.send": [False, None],
    "EyeTracker.get_file": [False, None],
    "EyeTracker.disconnect": [False, None],
    "EyeTracker.start_recording": [False, None],
    "EyeTracker.draw_boxes": [False, None],
    "EyeTracker.get_version": [False, None],
    "EyeTracker.open_file": [False, None],
    "EyeTracker.set_calib_env": [False, None],
    "EyeTracker.calibrate": [False, None],
    "EyeTracker.setup": [False, None],
    "EyeTracker.edf2asc": [False, None],
}
# disabled_funcs = {k: [True, f"DUMMY {k}"] for k in disabled_funcs.keys()} # ! TEMPORARY
# disabled_funcs = {k: [True, None] for k in disabled_funcs.keys()}  # ! TEMPORARY


@dataclass
class EEGcap:
    read_address: str
    write_address: str
    event_IDs: dict

    def __post_init__(self):
        if "invalid" not in self.event_IDs:
            raise ValueError(
                "Invalid response ID not found in the response IDs dictionary.\n"
                "Please add an entry with the key 'invalid' and its corresponding "
                "integer value."
            )

    @disable_decorator(*disabled_funcs.get("EEGcap.connect", (False, None)))
    def connect(self) -> None:
        try:
            self.port_read = parallel.ParallelPort(self.read_address)
            self.port_write = parallel.ParallelPort(self.write_address)
            self.reset_port()
            print("EEG device connected")
        except Exception as e:
            print("Error when trying to connect to EEG amplifier.\nDetails:", e)
            core.quit()
            sys.exit()

    @disable_decorator(*disabled_funcs.get("EEGcap.send", (False, None)))
    def send(self, data: str) -> None:
        data_int = self.event_IDs.get(data, self.event_IDs["invalid"])
        # print(f"{data} ({data_int}) sent to EEG device")
        self.port_write.setData(data_int)
        core.wait(0.01)
        self.port_write.setData(0)
        # print(f"{data} ({data_int})=> sent to {self.device_name} device")

    def reset_port(self):
        self.port_write.setData(255)  # Set all bits high
        core.wait(0.01)
        self.port_write.setData(0)  # Then set all low
        core.wait(0.01)


@dataclass
class EyeTracker:
    # * Overwrite the default value for device name & type
    ip_address: str
    # file: str
    device_name: str = field(default="Eyelink1000")
    device_type: str = field(default="EyeTracker")

    def __post_init__(self):
        self.connect()

    @disable_decorator(*disabled_funcs.get("EyeTracker.connect", (False, None)))
    def connect(self) -> None:
        try:
            self.device = pylink.EyeLink(self.ip_address)
            print("Eye Tracking device connected")
        except RuntimeError as error:
            print("encountered when trying to connect to EyeLink.\nERROR:", error)
            core.quit()
            sys.exit()

    @disable_decorator(*disabled_funcs.get("EyeTracker.get_file", (False, None)))
    def get_file(self, fname: str, local_dir: str = None) -> None:
        # if not self.device.isConnected():
        #     raise RuntimeError("The eye-tracker is not connected.")
        if self.device.isConnected():
            if self.device.isRecording():
                pylink.pumpDelay(200)
                self.device.stopRecording()

        if local_dir:
            local_fpath = str(Path(local_dir) / fname)
        else:
            local_fpath = fname

        try:
            self.device.receiveDataFile(fname, local_fpath)
        except RuntimeError as error:
            print("Error encountered when trying to get the file.\nERROR:", error)

    @disable_decorator(*disabled_funcs.get("EyeTracker.disconnect", (False, None)))
    def disconnect(self, session_folder: str, message: str = None) -> None:
        # el_tracker = pylink.getEYELINK()

        # * Stop recording
        if self.device.isConnected():
            if self.device.isRecording():
                # * add 100 ms to catch final trial events
                pylink.pumpDelay(100)
                self.device.stopRecording()

            if message is not None:
                self.device.sendMessage(message)

            # * Put tracker in Offline mode
            self.device.setOfflineMode()

            # * Clear the Host PC screen and wait for 500 ms
            self.device.sendCommand("clear_screen 0")
            pylink.msecDelay(500)

            # * Close the edf data file on the Host
            self.device.closeDataFile()

            export_file = str(Path(session_folder) / self.file)

            try:
                self.device.receiveDataFile(self.file, export_file)
            except RuntimeError as error:
                print("Error encountered when trying to get the file.\nERROR:", error)

            # * Close the link to the tracker.
            self.device.close()

    @disable_decorator(*disabled_funcs.get("EyeTracker.send", (False, None)))
    def send(self, msg: str) -> None:
        self.device.sendMessage(msg)

    @disable_decorator(*disabled_funcs.get("EyeTracker.start_recording", (False, None)))
    def start_recording(self):
        if not self.device.isConnected():
            print("ERROR: EyeLink not connected")
            return pylink.TRIAL_ERROR
        if not self.file:
            print("ERROR: EDF file not specified")
            return pylink.TRIAL_ERROR

        # * put tracker in idle/offline mode before recording
        self.device.setOfflineMode()

        try:
            # * Start recording
            # * arguments: sample_to_file, events_to_file, sample_over_link,
            # * event_over_link (1-yes, 0-no)
            self.device.startRecording(1, 1, 1, 1)
        except RuntimeError as error:
            print("Error encountered when trying to record.\nERROR:", error)
            # abort_trial()
            self.disconnect()
            return pylink.TRIAL_ERROR

        # * Allocate some time for the tracker to cache some samples
        pylink.pumpDelay(100)  # ! We might want this outside of this object / method

    @disable_decorator(*disabled_funcs.get("EyeTracker.draw_boxes", (False, None)))
    def draw_boxes(self):
        raise NotImplementedError
        # # get a reference to the currently active EyeLink connection
        # el_tracker = pylink.getEYELINK()

        # # put the tracker in the offline mode first
        # el_tracker.setOfflineMode()

        # # draw landmarks on the Host PC screen to mark the fixation cross,
        # # the visual target position, and the landing position of correct saccade
        # # The color codes supported on the Host PC range between 0-15
        # # 0 - black, 1 - blue, 2 - green, 3 - cyan, 4 - red, 5 - magenta,
        # # 6 - brown, 7 - light gray, 8 - dark gray, 9 - light blue,
        # # 10 - light green, 11 - light cyan, 12 - light red,
        # # 13 - bright magenta,  14 - yellow, 15 - bright white;
        # # see /elcl/exe/COMMANDs.INI on the Host
        # cross_coords = (int(scn_width/2.0), int(scn_height/2.0))
        # sac_tar_coords = (int(scn_width/2 + sac_x - 100),
        #                 int(scn_height/2 - 60),
        #                 int(scn_width/2 + sac_x + 100),
        #                 int(scn_height/2 + 60))
        # # the mirror location of the correct target
        # mir_tar_coords = (int(scn_width/2 - sac_x - 100),
        #                 int(scn_height/2 - 60),
        #                 int(scn_width/2 - sac_x + 100),
        #                 int(scn_height/2 + 60))
        # el_tracker.sendCommand('clear_screen 0')  # clear the host Display
        # el_tracker.sendCommand('draw_cross %d %d 10' % cross_coords)  # draw cross
        # el_tracker.sendCommand('draw_box %d %d %d %d 10' % sac_tar_coords)
        # el_tracker.sendCommand('draw_box %d %d %d %d 12' % mir_tar_coords)

        # # send a "TRIALID" message to mark the start of a trial, see Data
        # # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        # el_tracker.sendMessage('TRIALID %d' % trial_index)

        # # record_status_message : show some info on the Host PC
        # # here we show how many trial has been tested
        # status_msg = 'TRIAL number %d, %s' % (trial_index, cond)
        # el_tracker.sendCommand("record_status_message '%s'" % status_msg)

        # # drift check
        # # we recommend drift-check at the beginning of each trial
        # # the doDriftCorrect() function requires target position in integers
        # # the last two arguments:
        # # draw_target (1-default, 0-draw the target then call doDriftCorrect)
        # # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
        # #
        # # Skip drift-check if running the script in Dummy Mode
        # while not dummy_mode:
        #     # terminate the task if no longer connected to the tracker or
        #     # user pressed Ctrl-C to terminate the task
        #     if (not el_tracker.isConnected()) or el_tracker.breakPressed():
        #         terminate_task()
        #         return pylink.ABORT_EXPT

        #     # drift-check and re-do camera setup if ESCAPE is pressed
        #     try:
        #         error = el_tracker.doDriftCorrect(int(scn_width/2.0),
        #                                         int(scn_height/2.0), 1, 1)
        #         # break following a success drift-check
        #         if error is not pylink.ESC_KEY:
        #             break
        #     except:
        #         pass

        # # put tracker in idle/offline mode before recording
        # el_tracker.setOfflineMode()

        # # Start recording
        # # arguments: sample_to_file, events_to_file, sample_over_link,
        # # event_over_link (1-yes, 0-no)
        # try:
        #     el_tracker.startRecording(1, 1, 1, 1)
        # except RuntimeError as error:
        #     print("ERROR:", error)
        #     abort_trial()
        #     return pylink.TRIAL_ERROR

    @disable_decorator(*disabled_funcs.get("EyeTracker.open_file", (False, None)))
    def open_file(self, edf_file: str):
        # # * check if the filename is valid (length <= 8 & no special char)
        # if not edf_file.lower().endswith(".edf"):
        #     edf_file += ".EDF"

        # allowed_char = ascii_letters + digits + "_"
        # conditions = [
        #     all([c in allowed_char for c in edf_file[:-4]]),
        #     len(edf_file) <= 8,
        # ]

        # if not all(conditions):
        #     print("FILE CREATION ERROR: Invalid EDF filename")
        #     return

        try:
            self.device.openDataFile(edf_file)
            self.file = edf_file
        except RuntimeError as e:
            print(f"FILE CREATION ERROR: \n{e}")

            # * close the link if we have one open
            if self.device.isConnected():
                self.device.close()
            core.quit()
            sys.exit()

    @disable_decorator(*disabled_funcs.get("EyeTracker.set_calib_env", (False, None)))
    def set_calib_env(
        self,
        win,
        use_retina=False,
        fg_color=(-1, -1, -1),
        bg_color=(0, 0, 0),
        target_type="circle",
        target_size=24,
        calib_sounds=("", "", ""),
    ):
        # * Configure a graphics environment (genv) for tracker calibration
        genv = EyeLinkCoreGraphicsPsychoPy(self.device, win)
        print(genv)  # * print version number of CoreGraphics library

        # * Set background and foreground colors for the calibration target
        # * in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
        foreground_color = fg_color
        background_color = bg_color

        genv.setCalibrationColors(foreground_color, background_color)

        # * Set up the calibration target
        # * Use the default calibration target ('circle')
        genv.setTargetType(target_type)

        # * Configure size of calibration target (pixels); applies only to "circle" and "spiral"
        genv.setTargetSize(target_size)

        # * Beeps to play during calibration, validation and drift correction
        # genv.setCalibrationSounds(target="", good="", error="")
        genv.setCalibrationSounds(*calib_sounds)

        # * resolution fix for macOS retina display issues
        if use_retina:
            genv.fixMacRetinaDisplay()

        # * Request Pylink to use the PsychoPy window we opened above for calibration
        # ic("OPENING GRAPHICS")
        pylink.openGraphicsEx(genv)

    @disable_decorator(*disabled_funcs.get("EyeTracker.calibrate", (False, None)))
    def calibrate(self):
        # * Start the calibration process
        try:
            self.device.doTrackerSetup()
        except RuntimeError as err:
            print("ERROR:", err)
            self.device.exitCalibration()

    @disable_decorator(*disabled_funcs.get("EyeTracker.get_version", (False, None)))
    def get_version(self, verbose=False):
        vstr = self.device.getTrackerVersionString()
        vs_numb = int(vstr.split()[-1].split(".")[0])

        print(f"Running experiment on {vstr}, version {vs_numb}") if verbose else None
        return vs_numb

    @disable_decorator(*disabled_funcs.get("EyeTracker.setup", (False, None)))
    def setup(self, win, eye: str, use_retina=False, screen_distance: int = None):
        """ "
        eye: str: ["left", "right"]
        screen_distance: int: in millimeters
        """
        # * Put the tracker in offline mode before we change tracking parameters
        self.device.setOfflineMode()

        # * Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
        # * 5-EyeLink 1000 Plus, 6-Portable DUO
        # eyelink_ver = self.get_version()

        # * File and Link data control
        # * what eye events to save in the EDF file, include everything by default
        file_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT"
        # * what eye events to make available over the link, include everything by default
        link_event_flags = "LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT"

        # * what sample data to save in the EDF data file and to make available
        # * over the link, include the 'HTARGET' flag to save head target sticker
        # * data for supported eye trackers
        file_sample_flags = (
            "LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT"
        )
        link_sample_flags = "LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT"

        # if eyelink_ver <= 3:
        #     file_sample_flags = "LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT"
        #     link_sample_flags = "LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT"

        self.device.sendCommand(f"file_event_filter = {file_event_flags}")
        self.device.sendCommand(f"file_sample_data = {file_sample_flags}")
        self.device.sendCommand(f"link_event_filter = {link_event_flags}")
        self.device.sendCommand(f"link_sample_data = {link_sample_flags}")

        # ! 2B Tested:
        # binocular_tracking = "YES" if sess_data["mode"] == "binocular" else "NO"
        # el_tracker.sendCommand(f"binocular_enabled = {binocular_tracking}")
        self.device.sendCommand(f"active_eye = {eye}")

        self.device.sendCommand("sample_rate 2000")

        # Send the command to set the infrared illumination level
        # illumination_level = 3  # 0-15 # ! CHECK THIS
        # tracker.sendCommand(f"set_illumination={illumination_level}")

        # * Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
        self.device.sendCommand("calibration_type = HV9")

        # * ############################################################################
        # * set up a graphics environment for calibration
        # * Open a window, be sure to specify monitor parameters
        # mon = monitors.Monitor("myMonitor", width=53.0, distance=70.0)
        # win = visual.Window(fullscr=full_screen, monitor=mon, winType="pyglet", units="pix")

        if screen_distance:
            self.device.sendCommand(f"simulation_screen_distance = {screen_distance}")

        # * get the native screen resolution used by PsychoPy
        scn_width, scn_height = win.size
        # # resolution fix for Mac retina displays
        # if "Darwin" in platform.system():
        #     if use_retina:
        #         scn_width = int(scn_width / 2.0)
        #         scn_height = int(scn_height / 2.0)
        if use_retina:
            scn_width = int(scn_width / 2.0)
            scn_height = int(scn_height / 2.0)

        # * Pass the display pixel coordinates (left, top, right, bottom) to the tracker
        # * see the EyeLink Installation Guide, "Customizing Screen Settings"
        el_coords = f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}"
        self.device.sendCommand(el_coords)

        # Write a DISPLAY_COORDS message to the EDF file
        # * Data Viewer needs this piece of info for proper visualization, see Data
        # * Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        dv_coords = f"DISPLAY_COORDS  0 0 {scn_width - 1} {scn_height - 1}"
        self.device.sendMessage(dv_coords)

    # @disable_decorator(*disabled_funcs.get("EyeTracker.edf2asc", (False, None)))
    @staticmethod
    def edf2asc(edf_file: Union[str, Path], args: str = None):
        """
        Convert an EDF file to ASCII format using the EDF2ASC utility
        edf_file: type: str|Path ; The EDF file to be converted. Must be in the current working directory
        """
        import shutil

        edf_file = Path(edf_file)
        if not edf_file.exists():
            print(f"ERROR: {edf_file} not found")
            return

        # * check if the EDF2ASC utility is available on the system
        edf2asc_path = shutil.which("edf2asc")

        if not edf2asc_path:
            print("ERROR: EDF2ASC utility not found")
            return

        # print("CONVERTING FILE EDF FILE TO ASC...", end=" ")
        # * convert the EDF file to ASCII format
        if args is None:
            cmd = f"edf2asc {edf_file}"
        else:
            cmd = f"edf2asc {edf_file} {args}"
        os.system(cmd)

        # * check if the ASCII file was created
        asc_file = Path(str(edf_file).replace(".EDF", ".asc"))

        if asc_file.exists():
            print(f"EDF data file converted to ASCII format: {asc_file}")
        else:
            print("ERROR: Conversion failed")
