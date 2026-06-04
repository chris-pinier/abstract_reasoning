from ._common import *


@dataclass
class HumanDataClass:
    data_dir: Path
    preprocessed_dir: Path
    export_dir: Path
    data_fmt: DATA_FMTS = field(default=CURRENT_DATA_FMT, kw_only=True)
    task_name: str = field(default=TASK_NAME, kw_only=True)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.EEG = {"sfreq": c.EEG_SFREQ}

        if self.data_fmt not in ["bids", "original"]:
            raise ValueError(f"`self.data_fmt` must be set to {DATA_FMTS}")

    # def load_and_clean_behav_data():
    #     raise NotImplementedError
    @staticmethod
    def get_eeg_montage():
        return mne.channels.make_standard_montage("biosemi64")

    @staticmethod
    def extract_subj_sess(s: str | Path):
        return re.findall(r"subj_(\d{2})(\d{2})", str(s))[0]

    @staticmethod
    def interactive_erp_plot(erp: mne.Evoked) -> None:
        fig = px.line(
            erp.to_data_frame().set_index("time"),
            labels={"value": "Amplitude", "index": "Time (samples)"},
        )
        fig.update_layout(legend_title_text="Channel")
        fig.show()

    def find_file(
        self,
        folder: Path | str,
        pattern: str,
        search_label: str | None = None,
        dotfile: bool = False,
    ) -> Path:
        """Find exactly one file in ``folder`` matching ``pattern``.

        Args:
            folder: Directory to search.
            pattern: Glob pattern, e.g. ``"*.asc"``.
            search_label: Optional human-readable label for error messages.
            dotfile: Whether to include files whose names start with ``.``.

        Raises:
            FileNotFoundError: If ``folder`` does not exist or no file matches.
            ValueError: If more than one file matches.
        """
        folder = Path(folder)
        label = f"{search_label} file" if search_label else "file"

        if not folder.exists():
            raise FileNotFoundError(
                f"Directory not found while searching for {label}: {folder}"
            )
        if not folder.is_dir():
            raise NotADirectoryError(
                f"Expected a directory while searching for {label}: {folder}"
            )

        files = sorted(
            f
            for f in folder.glob(pattern)
            if f.is_file() and (dotfile or not f.name.startswith("."))
        )

        if len(files) == 1:
            return files[0]

        context = []
        if hasattr(self, "subj_N"):
            context.append(f"subj {self.subj_N:02}")
        if hasattr(self, "sess_N"):
            context.append(f"sess {self.sess_N:02}")
        context_txt = f" for {', '.join(context)}" if context else ""

        if not files:
            raise FileNotFoundError(
                f"Could not find {label}{context_txt} matching pattern "
                f"`{pattern}` in: {folder}"
            )

        files_txt = "\n- ".join(str(f) for f in files)
        raise ValueError(
            f"Found multiple {label}s{context_txt} matching pattern "
            f"`{pattern}` in {folder}:\n- {files_txt}"
        )
