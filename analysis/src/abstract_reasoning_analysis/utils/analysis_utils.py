import numpy as np
import json
from pprint import pprint
from mne.preprocessing.eyetracking import read_eyelink_calibration
from pathlib import Path, PosixPath
from tqdm.auto import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from mne.channels import DigMontage
from typing import List, Dict, Union, Tuple, Optional, Literal, Any
import contextlib
import io
import pickle
import tomllib
import re
from rsatoolbox.data import Dataset, TemporalDataset
from datetime import datetime
import matplotlib.pyplot as plt
from mne.io.edf.edf import RawEDF
import inspect
from rich.table import Table
from rich.text import Text
from rich.console import Console

console = Console()


def xdir(
    obj,
    show_private=False,
    pattern=None,
    sort_by="name",
    show_doc=False,
    max_doc_len=60,
):
    """
    An enhanced alternative to `dir()` using `rich` for formatted output.

    Args:
        obj: The object to inspect.
        show_private (bool): If True, includes attributes starting with '_'. Defaults to False.
        pattern (str): Optional regex pattern to filter attribute names.
        sort_by (str): How to sort the results. Options: "name" (default), "type".
        show_doc (bool): If True, adds a column showing the docstring/description.
        max_doc_len (int): Maximum length of the docstring before truncating.
    """
    attributes = []

    for attr_name in dir(obj):
        # 1. Filter private/dunder methods
        if not show_private and attr_name.startswith("_"):
            continue

        # 2. Filter by regex pattern (case-insensitive)
        if pattern and not re.search(pattern, attr_name, re.IGNORECASE):
            continue

        docstring = ""

        # 3. Safely determine type, category, and docstring
        try:
            # Check for property at the class level to avoid triggering it
            is_prop = isinstance(getattr(type(obj), attr_name, None), property)

            if is_prop:
                category = "Property"
                color = "green"
                type_name = "property"
                prop_obj = getattr(type(obj), attr_name)
                docstring = inspect.getdoc(prop_obj) or ""
            else:
                attr_value = getattr(obj, attr_name)
                type_name = type(attr_value).__name__
                docstring = inspect.getdoc(attr_value) or ""

                if inspect.isroutine(attr_value):
                    category = "Method"
                    color = "cyan"
                elif inspect.isclass(attr_value):
                    category = "Class"
                    color = "magenta"
                else:
                    category = "Attribute"
                    color = "yellow"

        except Exception as e:
            category = "Dynamic/Error"
            color = "red"
            type_name = f"Error: {type(e).__name__}"
            docstring = str(e)

        # Clean and truncate docstring
        if docstring:
            # Grab just the first line to keep the table clean
            docstring = docstring.split("\n")[0].strip()
            if len(docstring) > max_doc_len:
                docstring = docstring[:max_doc_len] + "..."
        else:
            docstring = "-"

        attributes.append(
            {
                "name": attr_name,
                "category": category,
                "type": type_name,
                "color": color,
                "doc": docstring,
            }
        )

    # 4. Sorting logic
    if sort_by == "name":
        attributes.sort(key=lambda x: x["name"])
    elif sort_by == "type":
        attributes.sort(key=lambda x: (x["category"], x["name"]))

    # 5. Build and render the Rich Table
    obj_name = getattr(obj, "__name__", type(obj).__name__)
    table = Table(
        title=f"Attributes of [bold]{obj_name}[/bold]",
        show_header=True,
        header_style="bold white on grey27",
    )

    table.add_column("Name", style="bold")
    table.add_column("Category", justify="center")
    table.add_column("Type", style="dim")

    # Conditionally add the Docstring column
    if show_doc:
        table.add_column("Docstring", style="italic dim white")

    for attr in attributes:
        row = [
            Text(attr["name"], style=attr["color"]),
            Text(attr["category"], style=attr["color"]),
            attr["type"],
        ]
        if show_doc:
            row.append(attr["doc"])

        table.add_row(*row)

    if not attributes:
        console.print(
            f"[italic dim]No attributes found matching the criteria for {obj_name}.[/italic dim]"
        )
    else:
        console.print(table)


def clean_filename(filename: Union[str, PosixPath], case=None) -> str:
    """
    Cleans a filename by removing unwanted characters and replacing spaces with underscores.

    Args:
        filename (str): The original filename.

    Returns:
        str: The cleaned filename.
    """
    var_type = type(filename)

    if var_type not in [str, PosixPath]:
        raise TypeError(
            f"filename must be of type str or pathlib.PosixPath, not {var_type}"
        )

    _filename = filename.stem if var_type == PosixPath else filename

    # Replace unwanted characters with an empty string
    pattern = r"\n|\s|\\|\/|\:|\*|\?|\<|\>|\||\!|\"|\'|\'\'"
    cleaned_filename = re.sub(pattern, "_", _filename)

    # Remove multiple underscores
    cleaned_filename = re.sub(r"_{2,}", "_", cleaned_filename)

    # Replace spaces with underscores
    # cleaned_filename = cleaned_filename.replace(" ", "_")

    if case == "lower":
        cleaned_filename = cleaned_filename.lower()
    elif case == "upper":
        cleaned_filename = cleaned_filename.upper()

    if var_type == PosixPath:
        cleaned_filename = filename.with_stem(cleaned_filename)

    return cleaned_filename


def clean_ann_id(
    id: str, reg_pat: str = "default", flags=re.IGNORECASE, str_case="title"
):
    allowed_str_case = ["lower", "upper", "title", None]
    assert str_case in allowed_str_case, f"str_case must be one of {allowed_str_case}"

    if reg_pat == "default":
        # reg_pat = r".+\/|-(?:it|instruct)"
        reg_pat = r".+(?:--|/)|-(?:it|instruct)"

    new_id = re.sub(reg_pat, "", id, flags=flags)

    if str_case is not None:
        new_id = getattr(new_id, str_case)()

    return new_id


def save_pickle(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def normalize(data: np.ndarray, method: str = "min-max"):
    avail_methods = {
        "max": lambda data: data / np.max(data),
        "min-max": lambda data: (data - np.min(data)) / (np.max(data) - np.min(data)),
        "z-score": lambda data: (data - np.mean(data)) / np.std(data),
    }
    method_names = list(avail_methods.keys())

    if method not in method_names:
        raise ValueError(f"Invalid normalization method. Choose from: {method_names}")

    return avail_methods[method](data)


def get_stim_coords(
    x_pos_stim, y_pos_choices, y_pos_sequence, screen_resolution, img_size
):
    # # * Create the figure
    # fig, ax = plt.subplots(figsize=(12, 8))

    # # * Set the limits of the plot to match the screen resolution
    # ax.set_xlim(0, screen_resolution[0])
    # ax.set_ylim(0, screen_resolution[1])
    # ax.invert_yaxis()

    targets = {
        f"seq{i}": center_x for i, center_x in enumerate(x_pos_stim["items_set"])
    }
    targets.update(
        {
            f"choice{i}": center_x
            for i, center_x in enumerate(x_pos_stim["avail_choice"])
        }
    )

    for target_name, x in targets.items():
        x_ = x + screen_resolution[0] / 2

        x_left = x_ - img_size[0] / 2
        x_right = x_ + img_size[0] / 2

        y_ = y_pos_sequence if target_name.startswith("seq") else y_pos_choices
        y_ = -y_ + screen_resolution[1] / 2

        y_top = y_ - img_size[1] / 2
        y_bottom = y_ + img_size[1] / 2

        targets[target_name] = [[x_left, x_right], [y_top, y_bottom]]

        # rect = Rectangle(
        #     xy=(x_left, y_top),
        #     width=img_size[0],
        #     height=img_size[1],
        #     fill=False,
        #     edgecolor="blue",
        #     linewidth=2,
        # )
        # ax.add_patch(rect)

        # ax.text(
        #         x_,
        #         y_,
        #         target_name,
        #         ha="center",
        #         va="center",
        #         fontweight="bold",
        #         # color=color_map[target_name],
        #         alpha=0.5,
        #     )

        # * Can use the following to get the position of the rectangle
        # rect.get_xy()
        # rect.get_width()
        # rect.get_height()
        # ax.scatter(x_, y_, c="blue")
        # ax.scatter(x_left, y_, c='blue')
        # ax.scatter(x_right, y_, c="blue")
        # ax.scatter(x_, y_top, c="blue")
        # ax.scatter(x_, y_bottom, c="blue")
    return targets


def get_trial_info(
    epoch_N,
    raw_behav,
    x_pos_stim,
    y_pos_choices,
    y_pos_sequence,
    screen_resolution,
    img_size,
):
    # TODO: Finish this function
    trial_behav = raw_behav.iloc[epoch_N]

    trial_seq = {i: trial_behav[f"figure{i + 1}"] for i in range(8)}
    trial_seq[trial_behav["masked_idx"]] = "question-mark"

    trial_solution = trial_behav["solution"]
    trial_choices = {i: trial_behav[f"choice{i + 1}"] for i in range(4)}
    trial_response = trial_behav["choice"]
    rt = trial_behav["rt"]

    if trial_response in ["timeout", "invalid"]:
        choice_ind = None
    else:
        choice_ind = [k for k, v in trial_choices.items() if v == trial_response][0]

    # correct = trial_response == trial_solution
    trial_seq_order = [int(i) for i in str(trial_behav["seq_order"])]
    trial_choice_order = [int(i) for i in str(trial_behav["choice_order"])]

    icons_order = trial_seq_order + [i + 8 for i in trial_choice_order]

    icons_coords = get_stim_coords(
        x_pos_stim, y_pos_choices, y_pos_sequence, screen_resolution, img_size
    )

    icons_coords = [i[0] + i[1] for i in icons_coords.values()]

    trial_icons = list(trial_seq.values()) + list(trial_choices.values())

    icons_coords = [[trial_icons[i], icons_coords[i]] for i in range(len(trial_icons))]

    return (
        icons_coords,
        icons_order,
        trial_seq,
        trial_choices,
        choice_ind,
        trial_solution,
        rt,
    )


def get_memory_usage():
    import types
    from pympler import asizeof

    # List of variable names to exclude, particularly Jupyter artifacts
    exclude_vars = {
        "quit",
        "exit",
        "Out",
        "_oh",
        "_dh",
        "_",
        "__",
        "___",
        "get_ipython",
        "logger",
        "globals_snapshot",
    }

    # Snapshot of globals to prevent modification during iteration
    globals_snapshot = {
        name: value for name, value in globals().items() if name not in exclude_vars
    }

    # Filter globals for only objects that pympler can measure
    variables = {}
    for name, value in tqdm(globals_snapshot.items()):
        # Skip modules, functions, and classes to avoid issues
        if isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        # Attempt to get the memory size and skip if it fails
        try:
            variables[name] = asizeof.asizeof(value)
        except (ValueError, TypeError):
            pass  # Ignore variables that raise errors

    df = pd.DataFrame(
        [(name, size / 125000) for name, size in variables.items()],
        columns=["Variable", "Usage (Megabits)"],
    ).sort_values(by="Usage (Megabits)", ascending=False)

    df["Usage (Megabits)"] = df["Usage (Megabits)"].round(2)

    # Display the DataFrame
    df.reset_index(drop=True, inplace=True)

    return df


def clear_jupyter_artifacts():
    # List of known Jupyter and system variables to avoid deleting
    protected_vars = {
        "__name__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__builtins__",
        "__file__",
        "__cached__",
        "__annotations__",
        "__IPYTHON__",
        "__IPYTHON__.config",
    }

    # Loop through all variables in the global namespace
    for name in list(globals().keys()):
        # Delete only numbered underscore variables that are not in the protected list
        if name.startswith("_") and name[1:].isdigit() and name not in protected_vars:
            del globals()[name]


def console_log(console, msg, level=None):
    console.print(msg)
    # dir(console)
    # console.print("This is information", style="info")
    # console.print("[warning]WARNING[/warning]:The pod bay doors are locked")
    # console.print("Something terrible happened!", style="danger")


def check_notes(data_dir: Path, show: bool = True) -> dict:
    sess_info_files = sorted(data_dir.rglob("*sess_info.json"))
    notes = {}
    for f in sess_info_files:
        subj_N, sess_N = [int(d) for d in f.name.split("-")[:2]]

        with open(f, "r") as file:
            sess_info = json.load(file)

        if len(sess_info["Notes"]) > 0:
            notes[f"subj_{subj_N:02}-sess_{sess_N:02}"] = sess_info["Notes"]
    if show:
        pprint(notes)
    return notes


def check_et_calibrations(
    data_dir: Path | str, pat: str | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Check the eye tracker calibration files"""

    data_dir = Path(data_dir)

    if pat is not None:
        et_files = [
            f
            for f in data_dir.rglob("*.asc")
            if re.search(pat, f.name) and not f.name.startswith(".")
        ]
    else:
        et_files = [f for f in data_dir.rglob("*.asc") if not f.name.startswith(".")]

    cal_params = (
        "onset",
        "model",
        "eye",
        "avg_error",
        "max_error",
        "screen_size",
        "screen_distance",
        "screen_resolution",
        "positions",
        "offsets",
        "gaze",
    )

    # * Exclude positions and gaze from the dataframe
    df_cols = ["subj_N", "sess_N", "cal_N"] + [
        p for p in cal_params if p not in ["positions", "gaze"]
    ]

    cals_df = pd.DataFrame(columns=df_cols)

    # * Create a context manager to suppress stdout
    for f in tqdm(et_files, desc="Checking ET calibrations"):
        subj_N = int(f.parents[1].name.split("_")[1])
        sess_N = int(f.parents[0].name.split("_")[1])

        with contextlib.redirect_stdout(io.StringIO()):
            sess_cals = [dict(cal) for cal in read_eyelink_calibration(f)]

        if len(sess_cals) == 0:
            vals = [subj_N, sess_N] + [np.nan] * (len(df_cols) - 2)
            cal_df = pd.DataFrame([vals], columns=df_cols)
            cals_df = pd.concat([cals_df, cal_df])
        else:
            for cal_N, cal in enumerate(sess_cals, start=1):
                vals = [subj_N, sess_N, cal_N] + [
                    cal[p] for p in cal_params if p in df_cols
                ]
                cal_df = pd.DataFrame([vals], columns=df_cols)
                cals_df = pd.concat([cals_df, cal_df])

    cals_df.reset_index(drop=True, inplace=True)
    missing_cals = cals_df.copy().query("cal_N.isna()")
    valid_cals = cals_df.copy().query("cal_N.notna()")
    unvalid_cal_model = valid_cals.query('model != "HV9"')

    if missing_cals.shape[0] > 0:
        missing_cals = missing_cals[["subj_N", "sess_N"]]
        print(
            f"\nWARNING: NO ET CALIBRATION FOUND IN THE FOLLOWING FILES:\n{missing_cals}\n"
        )

    if unvalid_cal_model.shape[0] > 0:
        print("WARNING: ET CALIBRATION MODEL IS NOT HV6")

    et_cals_stats = valid_cals.describe()

    print(f"\nET CALIBRATION STATS:\n{et_cals_stats}\n")

    return (cals_df, et_cals_stats)


def locate_trials(events, valid_events):
    valid_events_inv = {v: k for k, v in valid_events.items()}

    # * Find trial start and end events
    trial_start_inds = np.where(events[:, 2] == valid_events["trial_start"])[0]
    events_df = pd.DataFrame(events[:, [0, 2]], columns=["sample_nb", "event_id"])
    events_df["event_id"] = events_df["event_id"].replace(valid_events_inv)

    trial_bounds = trial_start_inds.tolist()
    trial_bounds = trial_bounds + [events.shape[0] - 1]
    trial_bounds = list(zip(trial_bounds[:-1], trial_bounds[1:]))

    trial_end_inds = np.array([], dtype=int)
    trial_n = np.zeros(len(events_df), dtype=int) - 1

    for i, bound in enumerate(trial_bounds):
        temp = events_df.iloc[bound[0] : bound[1]]
        # if "trial_aborted" in temp["event_id"].values:
        #     trial_bounds.pop(i)
        # elif "trial_end" in temp["event_id"].values:
        if "trial_end" in temp["event_id"].values:
            trial_end_ind = temp[temp["event_id"] == "trial_end"].index[-1]
            trial_end_inds = np.append(trial_end_inds, trial_end_ind)
            trial_n[bound[0] : trial_end_ind + 1] = i

    events_df["trial_id"] = trial_n

    trial_bounds = np.array([i for i in zip(trial_start_inds, trial_end_inds)])

    # print("\n\nTRIAL BOUNDS: ", trial_bounds, "\n\n")

    assert all(events_df.loc[trial_bounds[:, 0]]["event_id"] == "trial_start")
    assert all(events_df.loc[trial_bounds[:, 1]]["event_id"] == "trial_end")

    events_df.query("trial_id != -1", inplace=True)

    return trial_bounds, events_df


def export_trials(events_df, save=False):
    # * Function to highlight rows
    def highlight_trials(row):
        if "trial_start" in str(row["event_id"]):
            return ["background-color: green" for col in row]
        elif "trial_end" in str(row["event_id"]):
            return ["background-color: red" for col in row]
        else:
            return ["" for col in row]

    # * Apply the function to the DataFrame
    styled_df = events_df.style.apply(highlight_trials, axis=1)
    if save:
        styled_df.to_excel("trials.xlsx", engine="openpyxl", index=False)
    return styled_df


def determine_bad_channels(raw_eeg):
    # TODO: Implement this function
    raise NotImplementedError


def resample_eye_tracking_data(et_epoch, tracked_eye, et_sfreq_original, eeg_sfreq):
    x_gaze = et_epoch[f"xpos_{tracked_eye}"][0][0]
    y_gaze = et_epoch[f"ypos_{tracked_eye}"][0][0]
    x_gaze_resampled, y_gaze_resampled = resample_and_handle_nans(
        x_gaze, y_gaze, et_sfreq_original, eeg_sfreq
    )
    return x_gaze_resampled, y_gaze_resampled


def resample_and_handle_nans(x_gaze, y_gaze, et_sfreq_original, eeg_sfreq):
    et_time = np.arange(len(x_gaze)) / et_sfreq_original
    new_et_time = np.arange(et_time[0], et_time[-1], 1 / eeg_sfreq)
    x_gaze_resampled = np.full_like(new_et_time, np.nan)
    y_gaze_resampled = np.full_like(new_et_time, np.nan)
    valid_mask = ~np.isnan(x_gaze)
    valid_indices = np.where(valid_mask)[0]

    from itertools import groupby
    from operator import itemgetter

    def get_valid_segments(valid_indices):
        segments = []
        for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            segments.append((group[0], group[-1]))
        return segments

    valid_segments = get_valid_segments(valid_indices)

    for start_idx, end_idx in valid_segments:
        segment_time = et_time[start_idx : end_idx + 1]
        x_segment = x_gaze[start_idx : end_idx + 1]
        y_segment = y_gaze[start_idx : end_idx + 1]
        x_interp = interp1d(
            segment_time,
            x_segment,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        y_interp = interp1d(
            segment_time,
            y_segment,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        segment_new_time_mask = (new_et_time >= segment_time[0]) & (
            new_et_time <= segment_time[-1]
        )
        segment_new_time = new_et_time[segment_new_time_mask]
        x_gaze_resampled[segment_new_time_mask] = x_interp(segment_new_time)
        y_gaze_resampled[segment_new_time_mask] = y_interp(segment_new_time)

    return x_gaze_resampled, y_gaze_resampled


def check_ch_groups(
    montage: DigMontage,
    ch_groups: Dict[str, list[str]],
) -> List[str] | None:
    """#TODO: _summary_

    Args:
        montage (DigMontage): #TODO: _description_
        ch_groups (Dict[str, list[str]]): #TODO: _description_
    """
    montage_chans = set(montage.ch_names)
    groupped_chans = set([ch for ch_group in ch_groups.values() for ch in ch_group])

    orphan_chans = list(montage_chans - groupped_chans)

    if orphan_chans:
        print(
            f"WARNING: orphan channels found in ch_groups:\n{'\t'.join(orphan_chans)}"
        )
        return orphan_chans
    else:
        return None


def set_eeg_montage(
    raw_eeg: RawEDF,
    montage: DigMontage,
    eog_chans: List[str] | None,
    non_eeg_chans: List[str] | None = None,
    verbose: bool = True,
):
    # TODO: Messy function and usage, to be fixed
    if eog_chans is not None:
        raw_eeg.set_channel_types({ch: "eog" for ch in eog_chans})

    # montage = mne.channels.make_standard_montage("biosemi64")
    non_eeg_chans = [] if non_eeg_chans is None else non_eeg_chans

    if other_chans := [
        ch for ch in raw_eeg.ch_names if ch not in montage.ch_names + eog_chans
    ]:
        if verbose:
            print("WARNING: Dropping non-EEG and non-EOG channels: ", other_chans)

        raw_eeg.drop_channels(other_chans)

    raw_eeg.set_montage(montage)


def read_file(
    file_path: Path | str, *, hdf5_key: str | None = None, verbose: bool = True
):
    """
    Reads a file based on its extension. Supports JSON, TOML, pickle, CSV, NPY, HDF5.
    Uses context managers for file handling.

    Args:
        file_path: A pathlib.Path object representing the file to read.
        hdf5_key: Optional HDF5 key/path (e.g., "group/dataset" or pandas table key).

    Returns:
        The loaded data.

    Raises:
        FileNotFoundError, TypeError, ValueError, Exception
    """
    file_path = Path(file_path)

    def _read_hdf5(file_path: Path, key: str | None = None) -> Any:
        """
        Read HDF5 (.h5/.hdf5).

        If `key` is provided, tries:
        1) pandas.read_hdf(file_path, key=key)
        2) h5py dataset at that path (e.g. "group/dataset")

        If `key` is not provided, tries:
        1) pandas (if exactly one key exists)
        2) returns a dict of datasets using h5py (leaf datasets only)
        """
        # First: try pandas HDFStore (common for DataFrames)
        try:
            if key is not None:
                return pd.read_hdf(file_path, key=key)

            with pd.HDFStore(file_path, mode="r") as store:
                keys = store.keys()
                if len(keys) == 1:
                    return store.get(keys[0])
                # If multiple tables, return dict of DataFrames
                if len(keys) > 1:
                    return {k.lstrip("/"): store.get(k) for k in keys}
        except Exception:
            # Not a pandas HDFStore (or key not found) -> fall back to h5py
            pass

        # Fallback: h5py for general HDF5
        try:
            import h5py
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Reading .h5/.hdf5 requires 'h5py' (and optionally numpy). "
                "Install with: pip install h5py"
            ) from e

        def _as_python(x):
            # Convert numpy scalars/bytes to Python types for nicer outputs
            if isinstance(x, (bytes, bytearray)):
                return x.decode("utf-8", errors="replace")
            if "numpy" in type(x).__module__:
                try:
                    return x.item()
                except Exception:
                    return x
            return x

        out: dict[str, Any] = {}

        with h5py.File(file_path, "r") as f:
            if key is not None:
                obj = f.get(key)
                if obj is None:
                    raise KeyError(f"HDF5 key/path not found: {key}")

                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    # data could be ndarray, scalar, or bytes
                    if hasattr(data, "shape") and data.shape == ():
                        return _as_python(data)
                    return data
                else:
                    # It's a group; return its immediate children names
                    return {"type": "group", "keys": list(obj.keys())}

            # No key: collect all leaf datasets into a flat dict
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if hasattr(data, "shape") and data.shape == ():
                        data = _as_python(data)
                    out[name] = data

            f.visititems(visitor)

        return out

    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a pathlib.Path object.")

    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    file_readers = {
        ".json": lambda f: json.load(f),
        ".toml": lambda f: tomllib.load(f),
        ".pickle": lambda f: pickle.load(f),
        ".pkl": lambda f: pickle.load(f),
        ".csv": lambda p: pd.read_csv(p),
        ".npy": lambda p: np.load(p, allow_pickle=True),
        ".h5": lambda p: _read_hdf5(p, key=hdf5_key),
        ".hdf5": lambda p: _read_hdf5(p, key=hdf5_key),
    }

    ext = file_path.suffix.lower()
    reader_func = file_readers.get(ext)

    if reader_func is None:
        raise ValueError(
            f"Unsupported file extension: {ext}. Supported extensions: {list(file_readers.keys())}"
        )

    print(f"Opening {ext} file...") if verbose else None
    binary_exts = [".toml", ".pickle", ".pkl"]
    try:
        if ext in (".toml", ".json", ".pickle", ".pkl"):
            with open(file_path, "rb" if ext in binary_exts else "r") as file:
                data = reader_func(file)
        elif ext in (".csv", ".npy", ".h5", ".hdf5"):
            data = reader_func(file_path)
        else:
            raise ValueError(f"Unsupported file extension despite earlier check {ext}.")
        dtype = type(data)
        print(f"data type = {dtype}") if verbose else None

        return data
    except (json.JSONDecodeError, pickle.UnpicklingError, pd.errors.ParserError) as e:
        raise Exception(f"Error reading file '{file_path}': {e}") from e
    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"Error reading file '{file_path}': {e}") from e


def apply_df_style(
    df: pd.DataFrame,
    style: int = 1,
    vmin: Optional[Union[int, str, None]] = None,
    vmax: Optional[Union[int, str, None]] = None,
):  # -> pd.io.formats.style.Styler:
    """Apply styling to a pandas DataFrame.

    Args:
        df: Input pandas DataFrame
        style: Style identifier (default=1)
            1: Yellow-Orange-Red gradient with 2 decimal formatting

    Returns:
        Styled pandas DataFrame

    Raises:
        ValueError: If style number is not supported
    """

    styles = {
        1: lambda x: x.style.background_gradient(
            cmap="YlOrRd", vmin=vmin, vmax=vmax
        ).format("{:.2f}"),
        2: lambda x: x.style.background_gradient(
            cmap="RdYlBu", vmin=vmin, vmax=vmax
        ).format("{:.2f}"),
        3: lambda x: x.style.background_gradient(
            cmap="coolwarm", vmin=vmin, vmax=vmax
        ).format("{:.2f}"),
        4: lambda x: x.style.background_gradient(
            cmap="bwr", vmin=vmin, vmax=vmax
        ).format("{:.2f}"),
    }

    if vmin == "auto":
        vmin = df.min().min()
    if vmax == "auto":
        vmax = df.max().max()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if style not in styles:
        raise ValueError(
            f"Style {style} is not supported. Choose from: {list(styles.keys())}"
        )

    return styles[style](df)


def reorder_item_ids(
    original_order_df: pd.DataFrame, new_order_df: pd.DataFrame
) -> np.ndarray:
    """TODO:_summary_

    Args:
        original_df (pd.DataFrame): DataFrame containing the columns: item_id, pattern
        new_df (pd.DataFrame): DataFrame containing the columns: item_id, pattern

    Returns:
        np.ndarray: reordered indices
    """

    original_order_df = original_order_df[["pattern", "item_id"]].reset_index(
        names=["original_order"]
    )

    new_order_df = new_order_df[["pattern", "item_id"]].reset_index(names=["new_order"])

    new_order_df = original_order_df.merge(
        new_order_df, on=["pattern", "item_id"], how="left"
    )

    new_order_df = new_order_df[["original_order", "new_order"]].sort_values(
        "new_order"
    )

    reordered_inds = new_order_df["original_order"].values

    return reordered_inds


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)


def email_sender(email: str, password: str, on: bool = True):
    """
    To use with Gmail, you need to create an app password in your Google account settings:
    https://myaccount.google.com/apppasswords

    Args:
        email (str): email address to send from
        password (str): password for the email address
        on (bool, optional): whether to enable the email sender. Defaults to True.

    Returns:
        EmailSender: object of EmailSender class
    """
    from email.message import EmailMessage
    import ssl
    import smtplib
    from dataclasses import dataclass

    @dataclass
    class EmailSender:
        """
        To use with Gmail, you need to create an app password in your Google account settings:
        https://myaccount.google.com/apppasswords

        Args:
            email (str): email address to send from
            password (str): password for the email address
            on (bool, optional): whether to enable the email sender. Defaults to True.
        """

        email: str
        password: str
        on: bool = True

        def __post_init__(self):
            if self.on:
                self.context = ssl.create_default_context()
                self.smtp = smtplib.SMTP_SSL(
                    "smtp.gmail.com", 465, context=self.context
                )
                self.smtp.login(self.email, self.password)

        def send(self, receiver: str, subject: str, body: str):
            if not self.on:
                return
            else:
                em = EmailMessage()
                em["From"] = self.email
                em["To"] = receiver
                em["Subject"] = subject
                em.set_content(body)

                self.smtp.sendmail(self.email, receiver, em.as_string())

    return EmailSender(email, password, on)


def show_seq_img(item_id: int, seq_img_dir: Path):
    # seq_img_dir = WD.parent / "config/sequence_images"

    img_path = seq_img_dir / f"sequence_{item_id}.png"

    if not img_path.exists():
        raise ValueError(f"The file path provided doesn't exist: {img_path}")

    fig, ax = plt.subplots()
    ax.imshow(plt.imread(str(img_path)))
    plt.axis("off")
    plt.show()
    plt.close(fig)


def list_contents(
    directory: Path | str,
    incl: Literal["file", "folder", "both"] = "both",
    recurs: bool = True,
    hidden: bool = False,
    reg: str | None = None,
) -> list[Path]:
    directory = Path(directory)

    if incl not in {"file", "folder", "both"}:
        raise ValueError("include must be 'file', 'folder', or 'both'")

    paths = directory.rglob("*") if recurs else directory.glob("*")

    if not hidden:
        paths = (p for p in paths if not p.name.startswith("."))

    if reg is not None:
        regex = re.compile(reg)
        reg_filter = lambda p: regex.search(str(p))
    else:
        reg_filter = lambda p: True

    if incl == "file":
        paths = (p for p in paths if p.is_file() and reg_filter(p))
    elif incl == "folder":
        paths = (p for p in paths if p.is_dir() and reg_filter(p))
    else:
        paths = (p for p in paths if reg_filter(p))

    return list(paths)
