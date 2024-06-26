from pathlib import Path
from typing import Union, List, Dict, Callable, NamedTuple
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import inspect


def pickle_save(path, data):
    path = Path(path)
    suffix = path.suffix
    if suffix not in [".pickle", ".pkl"]:
        path = path.parent.resolve() / f"{path.stem}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def pickle_load(path):
    path = Path(path)
    suffix = path.suffix
    if suffix == "":
        path += ".pickle"

    if not path.exists():
        raise ValueError("File does not exist")

    elif suffix not in [".pickle", ".pkl"]:
        raise ValueError("Not a pickle file")

    if not Path(path).exists():
        raise ValueError("The file does not exist, check the path you provided.")
    with open(path, "rb") as f:
        return pickle.load(f)


def invert_dict(d: Dict) -> Dict:
    return {v: k for k, v in d.items()}


def to_named_tuple(d: dict, name):
    return NamedTuple(name, [(k, type(v)) for k, v in d.items()])(**d)


def create_report_doc(figs: dict = None, fname: str = None):
    from docx import Document

    figs = sorted(Path("./results").glob("*.png"))
    figs = figs[-10:]
    figs = {figs.stem: str(figs) for figs in figs}
    fname = "report"

    doc_name = f"{fname}.docx"

    document = Document()

    for title, fig in figs.items():
        title = title.replace("_", " ").title()
        document.add_heading(title, level=2)
        document.add_picture(fig)

    document.save(doc_name)

    # document.paragraphs[0].text


def custom_plot(
    sns_plots: List[Callable],
    plots_kwargs: List[Dict],
    fig_kwargs: Dict = None,
    aes_kwargs: List[Dict] = None,
    save_path: Union[str, Path] = None,
    ext: str = ".png",
    show: bool = True,
    to_pickle: bool = False,
):
    # ! TEMP
    # * just to show that it is expecting
    sns.color_palette()
    # ! TEMP

    for plot in sns_plots:
        assert (
            "ax" in inspect.signature(plot).parameters
        ), "Make sure sns_plots contain seaborn plots only"

    if fig_kwargs is None:
        fig_kwargs = {"figsize": (12, 8), "dpi": 300}

    if not sns_plots:
        raise ValueError("No plot types provided")
    elif len(sns_plots) > 1:
        # fig, ax = plt.subplots(**fig_kwargs)
        # for plot, kwargs in zip(sns_plots, plots_kwargs):
        #     plot(**kwargs, ax=ax)
        # TODO: implement this
        raise NotImplementedError("Only one plot type is supported at the moment")
    else:
        assert len(sns_plots) == len(plots_kwargs)
        if aes_kwargs is not None:
            assert isinstance(aes_kwargs, list)
            assert len(sns_plots) == len(aes_kwargs)
            aes_kwargs = aes_kwargs[0]

        sns_plot = sns_plots[0]
        plot_kwargs = plots_kwargs[0]

        fig, ax = plt.subplots(**fig_kwargs)

        sns_plot(**plot_kwargs, ax=ax)

        if aes_kwargs:
            for att, args in aes_kwargs.items():
                if hasattr(ax, att):
                    if isinstance(args, dict):
                        getattr(ax, att)(**args)
                    elif isinstance(args, list):
                        getattr(ax, att)(*args)
                    else:
                        getattr(ax, att)(args)

        # if ax.get_legend() is not None and aes_kwargs.get("legend") is None:
        #     ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

    if save_path:
        image_path = Path(save_path).with_suffix(ext)
        plt.savefig(
            image_path, dpi=fig_kwargs.get("dpi", 100)
        )  # , bbox_inches="tight")
        # TODO: Check if bbox_inches="tight" does not conflict with anything else
    if to_pickle:
        if save_path is None:
            raise ValueError("if to_pickle=True, you must provide a `save_path`")
        else:
            pickle_save(save_path, fig)

    plt.show() if show else None
    plt.close()

    return fig
