import base64
import io
import shutil
from typing import List, Optional, Dict, Tuple, Union
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import mplcursors
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
from analysis_utils import normalize, get_trial_info
from PIL import Image
import subprocess
from pathlib import Path
from matplotlib.figure import Figure
import pandas as pd
import re
from analysis_utils import get_trial_info

def plot_sequence_img(
    stim_pos: List,
    icon_images: dict,
    screen_resolution: tuple,
    save_dir: Union[str, Path],
    seq_id: Optional[str] = None,
):
    """_summary_

    Args:
        stim_pos (List): list of tuples containing the icon name and its left, right, bottom, top positions
        icon_images (dict): dictionary of icon images {icon_name: icon_image}
        screen_resolution (tuple): screen resolution in pixels (width, height)
        save_dir (Union[str, Path]): directory to save the image
        seq_id (str, optional): Defaults to None. Sequence ID to use for the filename.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, ax_et = plt.subplots(frameon=False)
    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticks([])
    ax_et.set_yticks([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # * Plot target icon
    for icon_name, pos in stim_pos:
        targ_left, targ_right, targ_bottom, targ_top = pos
        ax_et.imshow(
            icon_images[icon_name],
            extent=[targ_left, targ_right, targ_bottom, targ_top],
            origin="lower",
        )

        # # * Plot rectangle around target, with dimensions == img_size
        # rectangle = mpatches.Rectangle(
        #     (targ_left, targ_bottom),
        #     img_size[0],
        #     img_size[1],
        #     linewidth=0.8,
        #     linestyle="--",
        #     edgecolor="black",
        #     facecolor="none",
        # )
        # ax_et.add_patch(rectangle)

    ax_et.set_facecolor("lightgrey")
    # ax_et.axis("off")
    ax_et.spines[["left", "right", "top", "bottom"]].set_visible(False)
    # ax_et.splines = []
    fig.set_facecolor((0.5, 0.5, 0.5))  # "lightgrey")
    fig.tight_layout()

    fname = f"sequence_{seq_id}.png" if seq_id is not None else "sequence.png"
    fig.savefig(save_dir / fname, dpi=300)


def plot_sequence_video(
    type, icon_images, stim_pos, stim_order, screen_resolution, save_dir
):
    # save_dir = wd / "sequence_video"
    # if save_dir.exists():
    #     shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True)

    fps = 30
    zfill_len = 3
    frame_count = 0

    fig, ax_et = plt.subplots(frameon=False)
    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticks([])
    ax_et.set_yticks([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax_et.set_facecolor("lightgrey")
    ax_et.spines[["left", "right", "top", "bottom"]].set_visible(False)
    fig.set_facecolor((0.5, 0.5, 0.5))  # "lightgrey")
    fig.tight_layout()

    ax_et_fix_cross = ax_et.scatter(
        screen_resolution[0] / 2,
        screen_resolution[1] / 2,
        s=80,
        marker="+",
        linewidths=1,
        color="black",
    )
    ax_et_fix_cross.set_visible(False)

    ax_et_plotted_icons = []
    for icon_name, icon_pos in stim_pos:
        left, right, bottom, top = icon_pos

        this_icon = ax_et.imshow(
            icon_images[icon_name],
            extent=[left, right, bottom, top],
            origin="lower",
        )

        ax_et_plotted_icons.append(this_icon)
        this_icon.set_visible(True)

    plt.savefig(
        save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
    )  # , bbox_inches="tight")
    frame_count += 1

    [icon.set_visible(False) for icon in ax_et_plotted_icons]

    for frame in range(0, fps):
        ax_et_fix_cross.set_visible(True)
        plt.savefig(
            save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
        )  # , bbox_inches="tight")
        frame_count += 1

    ax_et_fix_cross.set_visible(False)

    for icon_idx in stim_order:
        for frame in range(0, int(0.6 * fps)):
            ax_et_plotted_icons[icon_idx].set_visible(True)
            plt.savefig(
                save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
            )  # , bbox_inches="tight")
            frame_count += 1
        ax_et_plotted_icons[icon_idx].set_visible(False)

    [icon.set_visible(True) for icon in ax_et_plotted_icons]

    for frame in range(0, int(fps * 2)):
        plt.savefig(
            save_dir / f"frame_{frame_count:0{zfill_len}}.png", dpi=300
        )  # , bbox_inches="tight")
        frame_count += 1

    [f.unlink() for f in save_dir.glob("*.png")]
    create_video_from_frames(save_dir, "sequence_video.mp4", fps, zfill_len)


def prepare_eeg_data_for_plot(
    eeg_chan_groups,
    eeg_montage,
    non_eeg_chans,
    sess_bad_chans: List[str],
    group_names: List[str],
    group_colors,
):
    selected_chans = [
        i
        for i, ch in enumerate(eeg_montage.ch_names)
        if ch not in non_eeg_chans + sess_bad_chans
    ]

    selected_chans_names = [eeg_montage.ch_names[i] for i in selected_chans]

    chans_pos_xy = np.array(
        [
            v
            for k, v in eeg_montage.get_positions()["ch_pos"].items()
            if k in selected_chans_names
        ]
    )[:, :2]

    # * Select EEG channel groups to plot
    selected_chan_groups = {
        k: v for k, v in eeg_chan_groups.items() if k in group_names
    }

    group_colors = dict(zip(selected_chan_groups.keys(), group_colors))

    # * Get channel indices for each channel group
    ch_group_inds = {
        group_name: [
            i for i, ch in enumerate(selected_chans_names) if ch in group_chans
        ]
        for group_name, group_chans in selected_chan_groups.items()
    }

    return selected_chans_names, ch_group_inds, group_colors, chans_pos_xy


def plot_eeg_and_gaze_fixations(
    eeg_data,
    eeg_sfreq,
    et_data,
    eeg_baseline,
    response_onset,
    eeg_start_time,
    eeg_end_time,
    icon_images,
    img_size,
    stim_pos,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    screen_resolution,
    title: str = None,
    vlines=None,
):
    """
    eeg_data: np.array, shape=(n_channels, n_samples)
    """

    gaze_x, gaze_y = et_data
    mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

    # * Set up the figure
    fig = plt.figure(figsize=(10, 6), dpi=200)
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 3, 3], width_ratios=[1, 1])
    ax_et = fig.add_subplot(gs[0, 0])
    ax_topo = fig.add_subplot(gs[0, 1])
    ax_eeg = fig.add_subplot(gs[1, :])
    ax_eeg_group = fig.add_subplot(gs[2, :])  # , sharex=ax_eeg)

    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticks([])
    ax_et.set_yticks([])

    ax_eeg.grid(axis="x", ls="--")
    ax_eeg_group.grid(axis="x", ls="--")
    ax_topo.set_axis_off()

    fig.suptitle(title)
    # ax_et.set_title(title)

    # * Plot target icon
    for icon_name, pos in stim_pos:
        targ_left, targ_right, targ_bottom, targ_top = pos
        ax_et.imshow(
            icon_images[icon_name],
            extent=(targ_left, targ_right, targ_bottom, targ_top),
            origin="lower",
        )

        # # * Plot rectangle around target, with dimensions == img_size
        rectangle = mpatches.Rectangle(
            (targ_left, targ_bottom),
            img_size[0],
            img_size[1],
            linewidth=0.8,
            linestyle="--",
            edgecolor="black",
            facecolor="none",
        )
        ax_et.add_patch(rectangle)

    # * Plot the topomap
    # TODO: Should we slice the topo data from after the baseline correction period?
    mne.viz.plot_topomap(
        eeg_data.mean(axis=1),
        chans_pos_xy,
        ch_type="eeg",
        sensors=True,
        contours=0,
        outlines="head",
        sphere=None,
        image_interp="cubic",
        extrapolate="auto",
        border="mean",
        res=640,
        size=1,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        axes=ax_topo,
        show=False,
    )

    # * Plot gaze data
    ax_et.scatter(gaze_x, gaze_y, c="red", s=2)
    ax_et.scatter(mean_gaze_x, mean_gaze_y, c="yellow", s=3)

    # * Plot EEG data
    ax_eeg.plot(eeg_data.T)

    for group_name, group_inds in ch_group_inds.items():
        ax_eeg_group.plot(
            eeg_data[group_inds].mean(axis=0),
            label=group_name,
            color=group_colors[group_name],
        )

    ax_eeg_group.legend(
        bbox_to_anchor=(1.005, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax_eeg.set_xlim(0, eeg_data.shape[1])
    ax_eeg_group.set_xlim(0, eeg_data.shape[1])

    tick_step_time = 0.05
    tick_step_sample = tick_step_time * eeg_sfreq
    x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)

    x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)
    # x_labels = ((x_ticks / eeg_sfreq - eeg_baseline)).round(3)  # .astype(int)
    # ax_eeg.set_xticks(x_ticks, x_labels)
    ax_eeg.set_xticks(x_ticks, [])
    ax_eeg_group.set_xticks(x_ticks, x_labels)
    ax_eeg_group.set_xlabel("Time (ms) relative to gaze fixation onset")

    eeg_sec_xaxis = ax_eeg.secondary_xaxis(location="top")
    t1 = -(response_onset - eeg_start_time)
    t2 = -(response_onset - eeg_end_time - tick_step_time * 0.9)
    x_labels2 = np.arange(t1, t2, tick_step_time).round(2)
    x_labels2 = [f"+{x}" if x >= 0 else f"{x}" for x in x_labels2]
    eeg_sec_xaxis.set_xticks(x_ticks, x_labels2)
    eeg_sec_xaxis.set_xlabel("Time (s) relative to response")

    if vlines is not None:
        for ax in [ax_eeg, ax_eeg_group]:
            ax.vlines(
                vlines,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color="black",
                ls="--",
                lw=1,
            )

    plt.tight_layout()
    # plt.show()

    return fig


def plot_eeg_and_gaze_fixations_plotly(
    eeg_data,
    et_data,
    ch_names,
    eeg_baseline,
    response_onset,
    eeg_start_time,
    eeg_end_time,
    icon_images,
    stim_pos,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    title: str = None,
    vlines=None,
    screen_resolution=(1920, 1080),
):
    """
    eeg_data: np.array, shape=(n_channels, n_samples)
    """

    def numpy_to_base64(img_array):
        """Converts a NumPy array image to a base64-encoded string after scaling to 0-255."""
        img_array = (img_array * 255).astype(np.uint8)  # Ensure image is uint8
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    # Check if icon_images is provided
    if icon_images is None:
        raise ValueError("icon_images must be provided and contain the required icons.")

    # Extract gaze data
    gaze_x, gaze_y = et_data
    mean_gaze_x, mean_gaze_y = gaze_x.mean(), gaze_y.mean()

    # # Create main figure layout
    fig = go.Figure()

    # # * Subplot 1: Gaze data with target icons
    # gaze_scatter = go.Scatter(
    #     x=gaze_x,
    #     y=gaze_y,
    #     mode="markers",
    #     marker=dict(color="red", size=2),
    #     name="Gaze Points",
    # )

    # mean_gaze_scatter = go.Scatter(
    #     x=[mean_gaze_x],
    #     y=[mean_gaze_y],
    #     mode="markers",
    #     marker=dict(color="yellow", size=8),
    #     name="Mean Gaze Point",
    # )
    # fig.add_trace(gaze_scatter)
    # fig.add_trace(mean_gaze_scatter)

    # # Define the layout for the gaze subplot
    # fig.update_xaxes(range=[0, screen_resolution[0]], title="X Position")
    # fig.update_yaxes(
    #     range=[screen_resolution[1], 0], title="Y Position", scaleanchor="x"
    # )

    # # Add icons to plot
    # for icon_name, pos in stim_pos:
    #     targ_left, targ_right, targ_bottom, targ_top = pos
    #     fig.add_shape(
    #         type="rect",
    #         x0=targ_left,
    #         x1=targ_right,
    #         y0=targ_bottom,
    #         y1=targ_top,
    #         line=dict(color="black", dash="dash"),
    #     )

    #     # Convert icon image to base64 and add to layout
    #     if icon_name in icon_images:
    #         base64_image = numpy_to_base64(icon_images[icon_name])
    #         fig.add_layout_image(
    #             dict(
    #                 source=base64_image,
    #                 xref="x",
    #                 yref="y",
    #                 x=targ_left,
    #                 y=targ_top,
    #                 sizex=(targ_right - targ_left),
    #                 sizey=(targ_top - targ_bottom),
    #                 opacity=0.8,
    #                 layer="below",
    #             )
    #         )
    #     else:
    #         print(f"Warning: {icon_name} not found in icon_images.")

    # * Subplot 2: EEG data (line plot for each channel)
    times = np.arange(0, eeg_data.shape[1])  # Sample indices
    for i, ch_data in enumerate(eeg_data):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=ch_data,
                mode="lines",
                name=ch_names[i],
                line=dict(width=1),
                opacity=0.5,
            )
        )

    # # Add gridlines and titles
    # fig.update_layout(
    #     title=title or "EEG and Gaze Fixations",
    #     xaxis=dict(title="Time (samples)", gridcolor="lightgray"),
    #     yaxis=dict(title="Amplitude (µV)", gridcolor="lightgray"),
    #     template="plotly_white",
    # )

    # # * Subplot 3: Grouped EEG channels
    # for group_name, group_inds in ch_group_inds.items():
    #     mean_group_data = eeg_data[group_inds].mean(axis=0)
    #     fig.add_trace(
    #         go.Scatter(
    #             x=times,
    #             y=mean_group_data,
    #             mode="lines",
    #             name=f"{group_name} Group",
    #             line=dict(color=group_colors[group_name], width=2),
    #         )
    #     )

    # * Add Vertical Lines for Specific Events
    if vlines is not None:
        for line_pos in vlines:
            fig.add_vline(
                x=line_pos, line=dict(color="black", dash="dash"), name="Event Line"
            )

    # Customize the legend
    fig.update_layout(
        showlegend=True, legend=dict(x=1.05, y=1), title=dict(x=0.5, y=0.95)
    )

    return fig


def plot_eeg(
    eeg_data: np.ndarray,
    chans_pos_xy: np.ndarray,
    ch_group_inds: Dict[str, List[int]],
    group_colors: Dict[str, str],
    eeg_sfreq: int,
    eeg_baseline: float = 0.0,
    chan_names=None,
    vlines: Optional[Union[float, List[float]]] = None,
    title: Optional[str] = None,
    plot_topo: bool = True,
    plot_eeg: bool = True,
    plot_eeg_group: bool = True,
    dpi: int = 100,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """_summary_

    Args:
        eeg_data (np.ndarray): _description_
        chans_pos_xy (np.ndarray): _description_
        ch_group_inds (Dict[str, List[int]]): _description_
        group_colors (Dict[str, str]): _description_
        eeg_sfreq (int): _description_
        eeg_baseline (float, optional): _description_. Defaults to 0.0.
        chan_names (_type_, optional): _description_. Defaults to None.
        vlines (Optional[Union[float, List[float]]], optional): _description_. Defaults to None.
        title (Optional[str], optional): _description_. Defaults to None.
        plot_topo (bool, optional): _description_. Defaults to True.
        plot_eeg (bool, optional): _description_. Defaults to True.
        plot_eeg_group (bool, optional): _description_. Defaults to True.
        dpi (int, optional): _description_. Defaults to 100.
        figsize (Tuple[int, int], optional): _description_. Defaults to (10, 6).

    Returns:
        Figure: _description_
    """

    if chan_names is None:
        chan_names = [f"Ch {i + 1}" for i in range(eeg_data.shape[0])]

    # tick_step_time = 0.05
    # tick_step_sample = tick_step_time * eeg_sfreq
    # x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)
    # x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)
    x = (np.arange(0, eeg_data.shape[1]) / eeg_sfreq - eeg_baseline) * 1000
    x_ticks = np.arange(round(x[0]), round(x[-1]) + 1, 50)

    # * Determine height ratios and rows based on flags
    height_ratios = []
    rows = 0
    if plot_topo:
        height_ratios.append(4)
        rows += 1
    if plot_eeg:
        height_ratios.append(3)
        rows += 1
    if plot_eeg_group:
        height_ratios.append(3)
        rows += 1

    # * Set up the figure and GridSpec based on the updated layout
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(rows, 3, height_ratios=height_ratios, width_ratios=[1, 1, 1])
    fig.suptitle(title) if title is not None else None

    ax_topo, ax_eeg, ax_eeg_group = None, None, None

    # * Conditionally add subplots
    current_row = 0
    if plot_topo:
        ax_topo = fig.add_subplot(gs[current_row, 1])
        ax_topo.set_axis_off()

        # * Plot the topomap
        # TODO: Should we slice the topo data from after the baseline correction period?
        mne.viz.plot_topomap(
            eeg_data.mean(axis=1),
            chans_pos_xy,
            # names=chan_names,
            ch_type="eeg",
            sensors=True,
            contours=0,
            outlines="head",
            sphere=None,
            image_interp="cubic",
            extrapolate="auto",
            border="mean",
            res=640,
            size=1,
            cmap=None,
            vlim=(None, None),
            cnorm=None,
            axes=ax_topo,
            show=False,
        )

        current_row += 1

    if plot_eeg:
        # * Plot EEG data, channel by channel
        ax_eeg = fig.add_subplot(gs[current_row, :])
        ax_eeg.grid(axis="x", ls="--")

        for i in range(eeg_data.shape[0]):
            # [i * eeg_sfreq * 1000 for i in eeg_data[i]]
            ax_eeg.plot(x, eeg_data[i], label=chan_names[i])

        ax_eeg.set_xlim(x[0], x[-1])

        if plot_eeg_group:
            ax_eeg.set_xticks(x_ticks)  # , [])
        else:
            ax_eeg.set_xticks(x_ticks)  # , x_labels)
            ax_eeg.set_xlabel("Time (ms) relative to gaze fixation onset")

        current_row += 1

    if plot_eeg_group:
        # TODO: make group_colors & ch_group_inds optional and uncomment below
        # if group_colors is None or ch_group_inds is None:
        #     raise ValueError(
        #         "group_colors and ch_group_inds must be provided when plot_eeg_group=True."
        #     )

        # * Plot EEG data, grouped by channel group
        ax_eeg_group = fig.add_subplot(gs[current_row, :])
        ax_eeg_group.grid(axis="x", ls="--")

        for group_name, group_inds in ch_group_inds.items():
            ax_eeg_group.plot(
                x,
                eeg_data[group_inds].mean(axis=0),
                label=group_name,
                color=group_colors[group_name],
            )

        ax_eeg_group.legend(
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        ax_eeg_group.set_xlim(x[0], x[-1])
        ax_eeg_group.set_xticks(x_ticks)  # , x_labels)

        ax_eeg_group.set_xlabel("Time (ms) relative to gaze fixation onset")

    if vlines is not None:
        for ax in [ax_eeg, ax_eeg_group]:
            if ax is not None:
                ax.vlines(
                    vlines,
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    color="black",
                    ls="--",
                    lw=1,
                )

    plt.tight_layout()
    mplcursors.cursor(hover=True)

    return fig


def plot_eeg_plotly_static(
    eeg_data,
    chans_pos_xy,
    ch_group_inds,
    group_colors,
    eeg_sfreq,
    eeg_baseline,
    vlines=None,
    title=None,
):
    import numpy as np

    # Set up figure with subplots
    fig = ps.make_subplots(
        rows=3,
        cols=3,
        subplot_titles=("Topomap", "EEG Time-Series", "Average EEG by Group"),
        row_heights=[0.5, 0.35, 0.35],
        shared_xaxes=True,
        specs=[
            [None, {"type": "scatter"}, None],
            [{"colspan": 3}, None, None],
            [{"colspan": 3}, None, None],
        ],
    )

    # * Topomap - placeholder, Plotly doesn't have built-in topomap functionality like matplotlib.
    fig.add_trace(
        go.Scatter(
            x=chans_pos_xy[:, 0],
            y=chans_pos_xy[:, 1],
            mode="markers+text",
            marker=dict(size=10),
            # text=[f"Ch {i+1}" for i in range(len(chans_pos_xy))],
            textposition="top center",
        ),
        row=1,
        col=2,
    )

    # * Add EEG time-series for each channel
    for i in range(eeg_data.shape[0]):
        fig.add_trace(
            go.Scatter(y=eeg_data[i], mode="lines", name=f"Channel {i + 1}"),
            row=2,
            col=1,
        )

    # * Plot average EEG for each group
    for group_name, group_inds in ch_group_inds.items():
        group_data_mean = eeg_data[group_inds].mean(axis=0)
        fig.add_trace(
            go.Scatter(
                y=group_data_mean,
                mode="lines",
                name=group_name,
                line=dict(color=group_colors[group_name]),
            ),
            row=3,
            col=1,
        )

    # Set x-axis properties
    tick_step_time = 0.05
    tick_step_sample = tick_step_time * eeg_sfreq
    x_ticks = np.arange(0, eeg_data.shape[1], tick_step_sample)
    x_labels = ((x_ticks / eeg_sfreq - eeg_baseline) * 1000).round().astype(int)

    fig.update_xaxes(
        tickvals=x_ticks,
        ticktext=x_labels,
        title_text="Time (ms) relative to gaze fixation onset",
        row=3,
        col=1,
    )

    # Set title
    if title:
        fig.update_layout(title=title)

    # Add vertical lines if provided
    if vlines is not None:
        for vline in vlines:
            fig.add_vline(
                x=vline,
                line=dict(color="black", dash="dash"),
                layer="below",
            )

    # * Save figure as a static image
    # fig.write_image("eeg_plotly_plot.png")

    # * Save figure as an interactive HTML file
    # fig.write_html("eeg_plotly_plot.html")

    # Show figure
    fig.update_layout(height=800, width=1000)
    fig.show()
    fig.update_layout(showlegend=False)

    return fig


def plot_matrix(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_values: bool = False,
    as_pct: bool = False,
    norm: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    text_color: Optional[str] = None,
    cbar: bool = False,
    ax: Optional[plt.Axes] = None,
):
    """
    Plots a correlation matrix with optional labels and value annotations.

    Args:
        - matrix (np.ndarray): # TODO
        - labels (Optional[List[str]]): List of labels for the rows/columns of the matrix.
        - show_values (bool): Whether to display values on the plot. Default is False.
        - ax (Optional[plt.Axes]): An existing matplotlib Axes object. If None, a new one is created.

    Returns:
        - None
    """
    m = matrix.copy()

    if norm is not None:
        m = normalize(m, norm)

    if ax is None:
        fig, ax = plt.subplots()

    if mask is not None:
        m = np.ma.masked_array(m, mask=mask)
        # m_masked = np.where(mask, np.nan, m)

    ax.set_title(title) if title is not None else None

    # * Create the heatmap
    # cax = ax.matshow(m, cmap=cmap)
    cax = ax.imshow(m, cmap=cmap)

    # * Add the colorbar
    if cbar:
        plt.colorbar(cax, ax=ax)

    # Add labels if provided
    if labels is not None:
        ax.set_xticks(range(m.shape[0]))
        ax.set_yticks(range(m.shape[0]))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

    # * Optionally display the values
    if show_values:
        if text_color is None:
            text_colors = np.array(
                ["black" if abs(i) < 0.5 else "white" for i in m.flatten()]
            ).reshape(m.shape)
        else:
            text_colors = np.array([text_color for i in m.flatten()]).reshape(m.shape)

        if as_pct:
            m = (np.round(m, 2) * 100).astype(int)

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}" if not as_pct else f"{m[i, j]}",
                    ha="center",
                    va="center",
                    color=text_colors[i, j],
                )


def create_base_figure(screen_resolution, targets, ax_et):
    ax_et.set_xlim(0, screen_resolution[0])
    ax_et.set_ylim(screen_resolution[1], 0)
    ax_et.set_xticklabels([])
    ax_et.set_yticklabels([])
    ax_et.set_xticks([])
    ax_et.set_yticks([])
    ax_et.set_aspect("equal", adjustable="box")
    for target in targets:
        ax_et.plot(target[0], target[1], "ko")


def create_video_from_frames(eeg_frames_dir, output_file, fps, zfill_len):
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        str(eeg_frames_dir / f"frame_%0{zfill_len}d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(eeg_frames_dir / output_file),
    ]
    print("Creating video with FFmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video created: {output_file}")


def get_gaze_heatmap(
    x_gaze: np.ndarray,
    y_gaze: np.ndarray,
    screen_res: tuple,
    bin_size: int = 50,
    show: bool = False,
    normalize: bool = True,
):
    """
    Generate a heatmap from gaze data.
    Parameters:
    x_gaze (array-like): Array of x-coordinates of gaze points.
    y_gaze (array-like): Array of y-coordinates of gaze points.
    screen_res (tuple): Screen resolution as (width, height).
    bin_size (int, optional): Size of the bins for the histogram. Default is 50.
    show (bool, optional): If True, display the heatmap. Default is False.
    normalize (bool, optional): If True, normalize the heatmap. Default is True.
    Returns:
    tuple: A tuple containing:
        - heatmap (2D array): The generated heatmap.
        - xedges (array): The bin edges along the x-axis.
        - yedges (array): The bin edges along the y-axis.
    """
    # TODO: Add possibility to show the heatmap over an existing plot

    screen_width, screen_height = screen_res

    valid_mask = ~np.isnan(x_gaze) & ~np.isnan(y_gaze)

    heatmap_gaze_x = x_gaze[valid_mask]
    heatmap_gaze_y = screen_height - y_gaze[valid_mask]

    # * Generate the heatmap using 2D histogram
    num_bins_x = screen_width // bin_size
    num_bins_y = screen_height // bin_size

    heatmap, xedges, yedges = np.histogram2d(
        heatmap_gaze_x,
        heatmap_gaze_y,
        bins=[num_bins_x, num_bins_y],
        range=[[0, screen_width], [0, screen_height]],
    )

    # * Optionally normalize the heatmap
    if normalize:
        heatmap = heatmap / np.max(heatmap)

    # * Plot the heatmap
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(
            heatmap.T,
            extent=[0, screen_width, 0, screen_height],
            origin="lower",
            cmap="hot",
            aspect="auto",
        )
        plt.colorbar(label="Normalized Gaze Density")
        plt.xlabel("X Position (pixels)")
        plt.ylabel("Y Position (pixels)")
        plt.title("Eye-Tracking Heatmap")
        plt.show()
        plt.close("all")

    return heatmap, xedges, yedges


def show_ch_groups(montage, ch_groups: Dict):
    """
    Display channel groups on a montage.
    Parameters:
    montage : mne.channels.DigMontage
        The montage object containing the electrode positions.
    ch_groups : dict
        A dictionary where keys are group names and values are lists of channel names to be displayed.
    Returns:
    None
    """

    for group_name, chans in ch_groups.items():
        fix, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title(group_name)
        montage.plot(show_names=chans, axes=ax, show=False)
        plt.show()


def custom_plot_montage(
    montage: mne.channels.DigMontage,
    ch_groups: Dict[str, List[str]],
    group_colors: Dict[str, str],
    show_names: bool = True,
    show_legend: bool = True,
    show: bool = True,
):
    # ! TEMP
    # ch_groups = ch_groups1.copy()
    # group_colors = ch_groups1_colors.copy()
    # ! TEMP

    # * Ensure channel names in ch_groups are non-overlapping
    all_chans = [ch for group in ch_groups.values() for ch in group]

    if len(all_chans) != len(set(all_chans)):
        raise ValueError(
            "Channel names in ch_groups must be non-overlapping in between groups."
        )

    chans_info = {ch: k for k, v in ch_groups.items() for ch in v}
    chans_info = {ch: chans_info.get(ch, "unassigned") for ch in montage.ch_names}
    ch_colors = {
        ch: group_colors.get(chans_info[ch], "black") for ch in montage.ch_names
    }
    ch_colors = list(ch_colors.values())

    # * Plot initially to capture the ax.lines and ax.texts (ch names)
    fig = montage.plot(show_names=True, show=False)
    ax = fig.get_axes()[0]
    plt.close()
    [group for group, chans in ch_groups.items() if "Cz" in chans]

    # * Get the lines (head, nose, ears) and positions from the initial plot
    head_lines = [line.get_xydata() for line in ax.lines]
    chans_plot_pos = ax.collections[0].get_offsets().data  # .T
    # chans_plot_pos_x, chans_plot_pos_y = chans_plot_pos
    chan_names = [i for i in ax.texts if i.get_text() in montage.ch_names]

    # * Now, replicate the plot, including the head, nose, and ears from `ax.lines`
    fig, ax = plt.subplots(figsize=(10, 10))

    # * Plot the head, nose, and ears using the lines stored in `ax.lines`
    for line_data in head_lines:
        ax.plot(line_data[:, 0], line_data[:, 1], color="black", lw=2)

    # * Plot the channel points
    ch_color_inds = pd.Series(ch_colors).groupby(ch_colors).groups

    for color, inds in ch_color_inds.items():
        # for i, color in enumerate(ch_colors):
        ax.scatter(
            chans_plot_pos[inds, 0],
            chans_plot_pos[inds, 1],
            color=color,
            s=100,
            zorder=2,
            label={v: k for k, v in group_colors.items()}.get(color, "unassigned"),
        )

    # * Plot the channel names
    if show_names:
        for i, ch_name in enumerate(chan_names):
            ax.text(
                ch_name.get_position()[0],
                ch_name.get_position()[1],
                ch_name.get_text(),
                fontsize=9,
                ha="left",
                va="center",
            )

    # * Set the aspect ratio, remove axes, and show the plot
    ax.set_aspect("equal", "box")
    ax.axis("off")

    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    if show:
        plt.show()
    plt.close()

    chans_info = [
        [i, ch_name, ch_group, ch_colors[i]]
        for i, (ch_name, ch_group) in enumerate(chans_info.items())
    ]

    return fig, chans_info


def custom_plot_montage_plotly(
    montage,  #: DigMontage,
    ch_groups: Dict[str, List[str]],
    group_colors: Dict[str, str],
    show_names: bool = True,
    show_legend: bool = True,
    show: bool = True,
):
    import numpy as np
    import plotly.graph_objs as go
    # import plotly.offline as pyo
    # from mne.channels import DigMontage

    # * Ensure channel names in ch_groups are non-overlapping
    all_chans = [ch for group in ch_groups.values() for ch in group]

    if len(all_chans) != len(set(all_chans)):
        raise ValueError(
            "Channel names in ch_groups must be non-overlapping between groups."
        )

    # * Map each channel to its group
    chans_info = {ch: k for k, v in ch_groups.items() for ch in v}
    chans_info = {ch: chans_info.get(ch, "unassigned") for ch in montage.ch_names}

    # * Map each channel to its color
    ch_colors = {
        ch: group_colors.get(chans_info[ch], "black") for ch in montage.ch_names
    }

    # * Get the 3D positions of the channels
    positions = montage.get_positions()
    ch_pos = positions["ch_pos"]

    # * Ensure the positions are in the same order as montage.ch_names
    pos_3d = np.array([ch_pos[ch] for ch in montage.ch_names])

    # * Convert 3D positions to 2D using azimuthal projection
    def _cart_to_sph(coords):
        x, y, z = coords.T
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        r = np.sqrt(x**2 + y**2 + z**2)
        return np.column_stack((azimuth, elevation, r))

    pos_sph = _cart_to_sph(pos_3d)
    xy = pos_sph[:, :2]  # Use azimuth and elevation as x and y

    # * Prepare the head outline (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    head_x = 0.5 * np.cos(theta)
    head_y = 0.5 * np.sin(theta)

    # * Define nose and ears
    nose = np.array([[0.0, 0.5], [-0.1, 0.6], [0.1, 0.6], [0.0, 0.5]])
    left_ear = np.array([[-0.5, 0.0], [-0.6, 0.1], [-0.6, -0.1], [-0.5, 0.0]])
    right_ear = np.array([[0.5, 0.0], [0.6, 0.1], [0.6, -0.1], [0.5, 0.0]])

    # * Create Plotly traces
    traces = []

    # * Head outline
    traces.append(
        go.Scatter(
            x=head_x,
            y=head_y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    # * Nose and ears
    traces.append(
        go.Scatter(
            x=nose[:, 0],
            y=nose[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )
    traces.append(
        go.Scatter(
            x=left_ear[:, 0],
            y=left_ear[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )
    traces.append(
        go.Scatter(
            x=right_ear[:, 0],
            y=right_ear[:, 1],
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    # * Group channels by color
    ch_color_series = pd.Series(ch_colors)
    ch_color_groups = ch_color_series.groupby(ch_colors).groups

    # * Plot channel points
    for color, inds in ch_color_groups.items():
        group_name = {v: k for k, v in group_colors.items()}.get(color, "unassigned")
        traces.append(
            go.Scatter(
                x=xy[inds, 0],
                y=xy[inds, 1],
                mode="markers",
                marker=dict(size=10, color=color),
                name=group_name if show_legend else "",
                showlegend=show_legend,
            )
        )

    # * Plot channel names
    if show_names:
        for i, ch_name in enumerate(montage.ch_names):
            traces.append(
                go.Scatter(
                    x=[xy[i, 0]],
                    y=[xy[i, 1]],
                    mode="text",
                    text=[ch_name],
                    textposition="top center",
                    showlegend=False,
                )
            )

    # * Create layout
    layout = go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.7, 0.7]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.7, 0.7]),
        width=600,
        height=600,
        showlegend=show_legend,
        legend=dict(x=1, y=1),
        margin=dict(t=20, b=20, l=20, r=20),
    )

    fig = go.Figure(data=traces, layout=layout)

    if show:
        fig.show()

    chans_info = [
        [i, ch_name, chans_info[ch_name], ch_colors[ch_name]]
        for i, ch_name in enumerate(montage.ch_names)
    ]

    return fig, chans_info


def generate_trial_video(
    manual_et_epochs,
    manual_eeg_epochs,
    raw_behav,
    epoch_N: int,
    all_bad_chans,
    eeg_chan_groups,
    non_eeg_chans,
    et_sfreq,
    valid_events_inv,
    eeg_sfreq,
    screen_resolution,
    icon_images,
):
    # * Extract epoch data
    et_epoch = manual_et_epochs[epoch_N]
    eeg_epoch = manual_eeg_epochs[epoch_N]

    fname = Path(eeg_epoch.filenames[0]).stem
    identifier = re.findall(r"subj_(\d+)", fname)[0]
    subj_N = int(identifier[:2])
    sess_N = int(identifier[2:])

    tracked_eye = et_epoch.ch_names[0].split("_")[1]
    # et_sfreq = et_epoch.info["sfreq"]
    # eeg_sfreq = eeg_epoch.info["sfreq"]
    assert et_sfreq == et_epoch.info["sfreq"], (
        "Eye-tracking data has incorrect sampling rate"
    )
    assert eeg_sfreq == eeg_epoch.info["sfreq"], "EEG data has incorrect sampling rate"

    sess_bad_chans = all_bad_chans.get(f"subj_{subj_N}", {}).get(f"sess_{sess_N}", [])
    montage = eeg_epoch.get_montage()

    selected_chans = [
        i
        for i, ch in enumerate(montage.ch_names)
        if ch not in non_eeg_chans + sess_bad_chans
    ]
    selected_chans_names = [montage.ch_names[i] for i in selected_chans]

    chans_pos_xy = np.array(
        [
            v
            for k, v in montage.get_positions()["ch_pos"].items()
            if k in selected_chans_names
        ]
    )[:, :2]

    # * Select EEG channel groups to plot
    selected_chan_groups = {
        k: v
        for k, v in eeg_chan_groups.items()
        if k
        in [
            "frontal",
            "parietal",
            "central",
            "temporal",
            "occipital",
        ]
    }

    group_colors = dict(
        zip(selected_chan_groups.keys(), ["red", "green", "blue", "purple", "orange"])
    )

    # * Get channel indices for each channel group
    ch_group_inds = {
        group_name: [
            i for i, ch in enumerate(selected_chans_names) if ch in group_chans
        ]
        for group_name, group_chans in selected_chan_groups.items()
    }

    # * Get channel positions for topomap
    eeg_info = eeg_epoch.info

    eeg_info = mne.pick_info(
        eeg_info,
        [
            i
            for i, ch in enumerate(eeg_info.ch_names)
            if ch not in non_eeg_chans + eeg_info["bads"]
        ],
    )

    chans_pos_xy = np.array(
        list(eeg_info.get_montage().get_positions()["ch_pos"].values())
    )[:, :2]

    trial_info = get_trial_info(epoch_N, raw_behav)
    stim_pos, stim_order = trial_info[:2]

    # * Resample eye-tracking data for the current trial
    x_gaze_resampled, y_gaze_resampled = resample_eye_tracking_data(
        et_epoch, tracked_eye, et_sfreq, eeg_sfreq
    )

    # * Synchronize data lengths
    eeg_data = eeg_epoch.get_data(picks=selected_chans_names)

    epoch_evts = pd.Series(eeg_epoch.get_data(picks=[stim_chan])[0])

    # * Find indices where consecutive events are different
    diff_indices = np.where(epoch_evts.diff() != 0)[0]
    epoch_evts = epoch_evts[diff_indices]
    epoch_evts = epoch_evts[epoch_evts != 0]
    epoch_evts = epoch_evts.replace(valid_events_inv)

    # * Drop EOG & Status channels
    # eeg_data = eeg_data[:-5, :]

    # * Ensure that the data arrays have the same length
    min_length = min(eeg_data.shape[1], len(x_gaze_resampled))
    eeg_data = eeg_data[:, :min_length]
    eeg_data *= 1e6  # Convert to microvolts
    # avg_eeg_data = eeg_data.mean(axis=0)

    x_gaze_resampled = x_gaze_resampled[:min_length]
    y_gaze_resampled = y_gaze_resampled[:min_length]

    # * y-axis limits for the EEG plots
    eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
    # avg_eeg_min, avg_eeg_max = avg_eeg_data.min(), avg_eeg_data.max()

    y_eeg_min, y_eeg_max = eeg_min * 1.1, eeg_max * 1.1
    # y_eeg_avg_min, y_eeg_avg_max = avg_eeg_min * 1.1, avg_eeg_max * 1.1

    # * Heatmap of gaze data
    all_stim_onset = epoch_evts[epoch_evts == "stim-all_stim"].index[0]
    trial_end = epoch_evts[epoch_evts == "trial_end"].index[0]

    heatmap, _, _ = get_gaze_heatmap(
        x_gaze_resampled[all_stim_onset:trial_end],
        y_gaze_resampled[all_stim_onset:trial_end],
        screen_resolution,
        bin_size=20,
        show=True,
    )

    # * Now you have resampled eye-tracking data that matches the EEG data in sampling rate and length
    # * Proceed with your analysis or visualization
    # * For example, you can plot the data or save it for later use
    samples_per_1ms = eeg_sfreq / 1000
    samples_per_100ms = round(samples_per_1ms * 100)

    # sample_window_ms = 50
    # samples_per_window = round(samples_per_1ms * sample_window_ms)

    # TODO: comment
    leftover = eeg_data.shape[1] % samples_per_100ms
    # n_splits = eeg_data.shape[1] // samples_per_100ms

    inds = np.arange(0, eeg_data.shape[1], samples_per_100ms)

    if leftover > 0:
        inds = np.append(inds, inds[-1] + leftover)

    # TODO: comment
    step_size = samples_per_100ms
    steps = np.diff(inds)
    inds = np.array(list(zip(inds[:-1], inds[1:])))

    zfill_len = len(str(steps.shape[0]))

    # * Set up the directory to save the frames
    eeg_frames_dir = Path.cwd() / f"eeg_frames-{subj_N}-{sess_N:02d}-ep{epoch_N:02d}"

    if eeg_frames_dir.exists():
        shutil.rmtree(eeg_frames_dir)
    eeg_frames_dir.mkdir()

    # * Adjust figure size and subplot layout
    fig = plt.figure(figsize=(25.6, 14.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 3], width_ratios=[3, 2])

    ax_eeg1 = fig.add_subplot(gs[0, :])  # * EEG traces plot (spans both columns)
    ax_eeg2 = fig.add_subplot(gs[1, :], sharex=ax_eeg1)  # * Avg EEG plot (both cols)
    ax_topo = fig.add_subplot(gs[2, 0])  # * Topomap (left side of bottom row)
    ax_et = fig.add_subplot(gs[2, 1])  # * Eye tracking plot (right side of bottom row)

    # win_len_seconds = 2
    # win_len_samples = int(win_len_seconds * eeg_sfreq)
    win_len_samples = steps.cumsum()[20]

    # samples_ticks = np.arange(0, win_len_samples + 101, samples_per_100ms)
    # time_ticks = [int(i / eeg_sfreq * 1000) for i in samples_ticks]
    # TODO: comment

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, eeg_data.shape[0]))

    x_ticks = np.arange(0, win_len_samples + 1, step_size)
    # x_ticks_labels = [None] * len(x_ticks)
    x_ticks_labels = [str(int(i / eeg_sfreq * 1000)) for i in x_ticks]

    ax_eeg1_line = ax_eeg1.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)
    ax_eeg2_line = ax_eeg2.vlines(-2, ymin=y_eeg_min, ymax=y_eeg_max)

    # TODO: comment
    et_scatter = None
    et_line = None
    last_eeg_x = 0
    last_plot_x = 0
    flash_N = 0

    ax_et_fix_cross = ax_et.scatter(
        screen_resolution[0] / 2,
        screen_resolution[1] / 2,
        s=80,
        marker="+",
        linewidths=1,
        color="black",
    )

    ax_et_plotted_icons = []
    for icon_name, icon_pos in stim_pos:
        left, right, bottom, top = icon_pos

        this_icon = ax_et.imshow(
            icon_images[icon_name],
            extent=[left, right, bottom, top],
            origin="lower",
        )

        ax_et_plotted_icons.append(this_icon)
        this_icon.set_visible(False)

    topo_plot_params = dict(
        ch_type="eeg",
        sensors=True,
        names=None,
        mask=None,
        mask_params=None,
        contours=0,
        outlines="head",
        sphere=None,
        image_interp="cubic",
        extrapolate="auto",
        border="mean",
        res=640,
        size=1,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        axes=ax_topo,
        show=False,
    )

    mne.viz.plot_topomap(
        data=np.zeros_like(eeg_data[:, 0]), pos=chans_pos_xy, **topo_plot_params
    )
    ax_topo.set_axis_off()

    eeg_group_data = {
        group: eeg_data[ch_group_inds[group]].mean(axis=0) for group in ch_group_inds
    }
    # min_eeg_by_group = [eeg_data[group_inds].mean().min() for group_inds in ch_group_inds.values()]
    # max_eeg_by_group = [
    #     eeg_data[group_inds].mean().max() for group_inds in ch_group_inds.values()
    # ]
    min_eeg_by_group = [group_data.min() for group_data in eeg_group_data.values()]
    max_eeg_by_group = [group_data.max() for group_data in eeg_group_data.values()]

    def reset_eeg_plot():
        ax_eeg1.clear()

        ax_eeg1.set_xticks(x_ticks)
        ax_eeg1.set_xticklabels([])

        ax_eeg1.set_ylim(y_eeg_min, y_eeg_max)
        ax_eeg1.set_xlim(0, win_len_samples)

        ax_eeg2.clear()
        ax_eeg2.set_xticks(x_ticks, x_ticks_labels)
        ax_eeg2.set_ylim(min(min_eeg_by_group), max(max_eeg_by_group))
        ax_eeg2.set_xlim(0, win_len_samples)
        ax_eeg2.hlines(0, 0, win_len_samples, color="black", linestyle="--")

    def reset_et_plot(show_icons: Union[List, bool] = False, show_fix_cross=False):
        ax_et.set_xlim(0, screen_resolution[0])
        ax_et.set_ylim(screen_resolution[1], 0)

        ax_et.set_xticklabels([])
        ax_et.set_yticklabels([])

        ax_et.set_xticks([])
        ax_et.set_yticks([])

        ax_et.set_aspect("equal", adjustable="box")

        if isinstance(show_icons, bool):
            [
                ax_et_plotted_icons[i].set_visible(show_icons)
                for i in range(len(stim_order))
            ]

        elif isinstance(show_icons, list):
            [ax_et_plotted_icons[i].set_visible(False) for i in range(len(stim_order))]
            [ax_et_plotted_icons[i].set_visible(True) for i in show_icons]

        ax_et_fix_cross.set_visible(show_fix_cross)

    dpi = 150
    reset_eeg_plot()
    reset_et_plot(show_icons=False, show_fix_cross=True)

    for idx_step, step in enumerate(tqdm(steps, desc="Generating frames")):
        ax_eeg1_line.remove()
        ax_eeg2_line.remove()

        bounds = (last_eeg_x, last_eeg_x + step)

        detected_event_inds = [i for i in epoch_evts.index if i in range(*bounds)]

        if detected_event_inds:
            if ax_et_fix_cross.get_visible():
                ax_et_fix_cross.set_visible(False)

            # * Get name, detected events and their descriptions
            detected_events = epoch_evts[detected_event_inds]
            event_desc = detected_events.values
            # event_inds = detected_events.index
            event_inds = detected_events.index - bounds[0] + last_plot_x

            # * Vertical lines to mark events
            ax_eeg1.vlines(
                event_inds, ymin=y_eeg_min, ymax=y_eeg_max, color="red", linestyle="--"
            )
            ax_eeg2.vlines(
                event_inds, ymin=y_eeg_min, ymax=y_eeg_max, color="red", linestyle="--"
            )

            for i, (ind, desc) in enumerate(zip(event_inds, event_desc)):
                ax_eeg1.text(
                    ind,
                    y_eeg_max,
                    desc,
                    rotation=45,
                    verticalalignment="top",
                    horizontalalignment="right",
                    fontsize=8,
                    color="red",
                )

                if "stim-flash" in desc:
                    icon_ind = stim_order[flash_N]
                    reset_et_plot(show_icons=[icon_ind], show_fix_cross=False)
                    # ax_et_plotted_icons[icon_ind].set_visible(True)
                    flash_N += 1

                elif desc == "stim-all_stim":
                    reset_et_plot(show_icons=True, show_fix_cross=False)

                elif desc in ["a", "x", "m", "l", "timeout", "trial_end"]:
                    reset_et_plot(show_icons=False, show_fix_cross=True)

        # * Get EEG data slice
        eeg_slice = eeg_data[:, bounds[0] : bounds[1]]

        # * Get gaze data slice
        x_gaze_slice = x_gaze_resampled[bounds[0] : bounds[1]]
        y_gaze_slice = y_gaze_resampled[bounds[0] : bounds[1]]

        # * Remove previous gaze data
        if et_scatter:
            et_scatter.remove()
            [el.remove() for el in et_line]

        # * Plot gaze data
        cmap = plt.get_cmap("Reds")
        norm = plt.Normalize(0, x_gaze_slice.shape[0])
        et_colors = cmap(
            norm(
                np.linspace(0, x_gaze_slice.shape[0], x_gaze_slice.shape[0]) * 0.5 + 10
            )
        )

        et_scatter = ax_et.scatter(
            x_gaze_slice, y_gaze_slice, c=et_colors, s=2, alpha=0.5
        )
        # segments = np.stack((x_gaze_slice, y_gaze_slice), axis=1)
        et_line = ax_et.plot(
            x_gaze_slice, y_gaze_slice, c="r", ls="-", linewidth=1, alpha=0.3
        )

        # * Plot EEG data
        if last_plot_x == win_len_samples:
            last_plot_x = 0
            reset_eeg_plot()

        x = np.arange(last_plot_x, last_plot_x + step)

        last_eeg_x += step
        last_plot_x += step

        for i in range(eeg_slice.shape[0]):
            ax_eeg1.plot(x, eeg_slice[i], color=colors[i])

        # ax_eeg2.plot(x, avg_eeg_data[bounds[0] : bounds[1]], color="black")
        # for group_name, group_inds in ch_group_inds.items():
        for group_name, group_data in eeg_group_data.items():
            ax_eeg2.plot(
                x,
                group_data[bounds[0] : bounds[1]],
                label=group_name,
                color=group_colors[group_name],
            )
        if idx_step == 0:
            ax_eeg2_legend = ax_eeg2.get_legend_handles_labels()
        # ax_eeg2.legend(
        #     bbox_to_anchor=(1.005, 1),
        #     loc="upper left",
        #     borderaxespad=0,
        # )

        # * Plot vertical lines to mark the end of the current slice
        ax_eeg1_line = ax_eeg1.vlines(
            x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
        )

        ax_eeg2_line = ax_eeg2.vlines(
            x[-1], ymin=y_eeg_min, ymax=y_eeg_max, color="black", linestyle="--"
        )

        # * Plot EEG Topomap
        ax_topo.clear()
        mne.viz.plot_topomap(
            data=eeg_slice.mean(axis=1), pos=chans_pos_xy, **topo_plot_params
        )

        ax_eeg2.legend().remove()
        ax_eeg2.legend(
            *ax_eeg2_legend,
            bbox_to_anchor=(1.005, 1),
            loc="upper left",
            borderaxespad=0,
        )

        # * Save the figure
        plt.tight_layout()

        plt.savefig(
            eeg_frames_dir / f"frame_{str(idx_step + 1).zfill(zfill_len)}.png", dpi=dpi
        )

    reset_eeg_plot()
    reset_et_plot(show_icons=False, show_fix_cross=True)

    ax_eeg2.legend(
        *ax_eeg2_legend,
        bbox_to_anchor=(1.005, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(eeg_frames_dir / f"frame_{str(0).zfill(zfill_len)}.png", dpi=dpi)

    fps = 3
    create_video_from_frames(eeg_frames_dir, "eeg_video.mp4", fps, zfill_len)


if __name__ == "__main__":
    pass

    # ! ----------------------------------------
    # ! Demonstration purposes
    # def example_show_ch_groups():
    #     # # * Show channel groups
    #     show_ch_groups(raw_eeg.get_montage(), eeg_chan_groups)

    # def example_custom_plot_montage():
    #     ch_groups1 = {
    #         g: chans
    #         for g, chans in eeg_chan_groups.items()
    #         if g
    #         in ["frontal", "parietal", "occipital", "temporal", "central", "unassigned"]
    #     }
    #     ch_groups1_colors = dict(
    #         zip(ch_groups1.keys(), ["red", "green", "blue", "purple", "orange"])
    #     )
    #     ch_groups2 = {
    #         g: chans
    #         for g, chans in eeg_chan_groups.items()
    #         if g in ["frontal", "occipital"]
    #     }

    #     ch_groups2_colors = dict(
    #         zip(
    #             ch_groups2.keys(),
    #             [
    #                 "red",
    #                 "green",
    #             ],
    #         )
    #     )

    #     ch_groups1_montage, _ = custom_plot_montage(
    #         montage, ch_groups1, ch_groups1_colors, show_names=False
    #     )

    #     ch_groups2_montage, _ = custom_plot_montage(
    #         montage,
    #         ch_groups2,
    #         ch_groups2_colors,
    #         show_names=True,
    #     )
    # ! Demonstration purposes
    # ! ----------------------------------------
