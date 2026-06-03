import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configuration
    """)
    return


@app.cell
def _():
    # * IMPORTS
    from pathlib import Path
    from typing import Dict, Final
    from box import Box
    import pandas as pd
    import os
    from IPython.display import display
    from pprint import pprint
    import plotly.express as px
    from contextlib import redirect_stdout
    import mne
    import sys

    WD = Path(__file__).parent
    ROOT = "/".join(WD.parts[:WD.parts.index("abstract_reasoning")+1])
    sys.path.append(WD)
    os.chdir(WD)
    assert WD == Path.cwd()

    # * RELATIVE IMPORTS
    # from analysis_conf import Config as c
    # from data_loader.human_data import HumanSessData, HumanSubjData, HumanGroupData
    # from utils.analysis_utils import read_file, list_contents
    # from analysis_compare_clean import CombinedData
    from ar_analysis.data_loader.human_data import (
        HumanSessData,
        HumanSubjData,
        HumanGroupData,
    )
    from ar_analysis.data_loader.ann_data import (
        ANNSubjData,
        ANNGroupData,
    )
    from ar_analysis.utils.custom_type_hints import DATA_FMTS
    from ar_analysis.utils.analysis_utils import (
        read_file,
        reorder_item_ids,
        list_contents,
    )
    from ar_analysis.analysis_rsa import get_ds_and_rdm
    from ar_analysis.analysis_config import Config as c

    # ! TEMP: to locate and use ffmpeg
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

    return (
        ANNGroupData,
        ANNSubjData,
        Box,
        Final,
        HumanGroupData,
        HumanSessData,
        HumanSubjData,
        Path,
        WD,
        c,
        display,
        list_contents,
        mne,
        os,
        pd,
        px,
        read_file,
        redirect_stdout,
    )


@app.cell
def _(Box, Final, Path, c):
    # * GLOBAL VARIABLES
    PATTERNS = c.PATTERNS
    ANN_ID_MAPPING = c.ANN_ID_MAPPING
    ANN_ID_ORDER = c.ANN_ID_ORDER

    DATASET = c.DATASET
    SEQ_FILE = c.SEQ_FILE
    # DIRECTORIES = c.DIRECTORIES

    # SAVE_DISK: Final = Path("/Volumes/Realtek 1Tb")
    SAVE_DISK: Final = Path("/Volumes/SSD-512Go")
    assert SAVE_DISK.exists(), "WARNING: SSD not connected"
    MAIN_DATA_DIR = SAVE_DISK / "PhD Data/experiment1/data/"
    DIRECTORIES: Final = Box(
        {
            "ann": {
                "data": MAIN_DATA_DIR
                / "ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts",
                "prepro": None,
                "analyzed": MAIN_DATA_DIR / "ANNs/analyzed",
                "export": MAIN_DATA_DIR / "ANNs/analyzed",
            },
            "human": {
                "data": MAIN_DATA_DIR / "Lab/raw-BIDS3",
                "prepro": MAIN_DATA_DIR / "Lab/preprocessed",
                "analyzed": MAIN_DATA_DIR / "Lab/analyzed",
                "export": MAIN_DATA_DIR / "Lab/analyzed",
            },
        }
    )
    return (
        ANN_ID_MAPPING,
        ANN_ID_ORDER,
        DATASET,
        DIRECTORIES,
        PATTERNS,
        SEQ_FILE,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Human Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Data
    """)
    return


@app.cell
def _(DIRECTORIES: "Final", HumanGroupData, HumanSessData, HumanSubjData):
    this_sess = HumanSessData(
        DIRECTORIES.human.data,
        DIRECTORIES.human.prepro,
        DIRECTORIES.human.export,
        subj_N=1,
        sess_N = 1
    )

    this_subj = HumanSubjData(
        DIRECTORIES.human.data,
        DIRECTORIES.human.prepro,
        DIRECTORIES.human.export,
        subj_N=1,
    )

    # this_sess = this_subj.sessions[1]

    human_group = HumanGroupData(
        DIRECTORIES.human.data,
        DIRECTORIES.human.prepro,
        DIRECTORIES.human.export,
    )
    return human_group, this_sess, this_subj


@app.cell
def _(human_group):
    human_group.show_dir_struct()
    return


@app.cell
def _(this_sess):
    list(this_sess.get_raw_behav_data().columns)
    return


@app.cell
def _(human_group):
    sess_info = human_group.get_sess_info()
    return (sess_info,)


@app.cell
def _(pd, sess_info):
    _df_sess_info = []
    for _subj_N, subj_info in sess_info.items():
        for sess_N, sessions_info in subj_info.items():
            _df = pd.DataFrame.from_dict(sessions_info, orient='index').T
            _df['subj_N'] = _subj_N
            _df['sess_N'] = sess_N
            _df_sess_info.append(_df)
    df_sess_info = pd.concat(_df_sess_info).reset_index(drop=True)
    print(f"df_sess_info['img_size'].unique() = {df_sess_info['img_size'].unique()!r}")
    print(f"df_sess_info['window_size'].unique() = {df_sess_info['window_size'].unique()!r}")
    print(f"df_sess_info['eye'].unique() = {df_sess_info['eye'].unique()!r}")
    print(f"df_sess_info['vision_correction'].unique() = {df_sess_info['vision_correction'].unique()!r}")
    df_sess_info
    return


@app.cell
def _(human_group, list_contents, read_file):
    physio_jsons = [read_file(f) for f in list_contents(human_group.data_dir, reg=r".+_physio.json$")]
    physio_jsons
    return


@app.cell
def _():
    # ks = ["AverageCalibrationError", "MaximalCalibrationError"]
    # res = []
    # for pj in physio_jsons:
    #     for k in ks:
    #         res.extend(pj[k][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyze Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####  Subject Level
    """)
    return


@app.cell
def _(this_subj):
    this_subj.show_dir_struct()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's look at this subject's performance
    """)
    return


@app.cell
def _(display, this_subj):
    this_subj_behav = this_subj.get_behav_data()
    display(this_subj_behav.head())

    agg_level = "session"  # "subject" or "session"
    this_subj_perf_res = this_subj.analyze_perf(agg="subject")
    perf_res_keys = this_subj_perf_res.keys()

    print(f"Perfomance results' keys:{''.join([f'\n  - {k}' for k in perf_res_keys])}")

    display(this_subj_perf_res["acc_by_patt"].head())
    return (this_subj_behav,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Create trial video
    """)
    return


@app.cell
def _(Path, this_sess):
    sess_info_1, _behav, eeg, et = this_sess.get_data()
    manual_et_trials, *_ = this_sess.split_et_data_into_trials(et)
    manual_eeg_trials, *_ = this_sess.split_eeg_data_into_trials(eeg, _behav)
    manual_et_trials = list(manual_et_trials)
    manual_eeg_trials = list(manual_eeg_trials)
    _trial_N = 30
    save_dir = Path(f'./TEST/subj{this_sess.subj_N:02}_ses{this_sess.sess_N:02}/trial{_trial_N}_frames')
    save_dir.mkdir(exist_ok=True, parents=True)
    this_sess.generate_trial_video(manual_et_trials, manual_eeg_trials, _behav, _trial_N, save_dir=save_dir, gaze_pt_size=10)
    return (save_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Sesssion Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's get locate in which trials and in which presentation order each stimulus is shown
    """)
    return


@app.cell
def _(this_subj):
    stim_flash_order = this_subj.get_stim_flash_order()
    # display(stim_flash_order.dropna())
    stim_flash_order.dropna()
    return (stim_flash_order,)


@app.cell
def _(Box, display, stim_flash_order):
    ind = 0
    row = stim_flash_order.dropna().iloc[ind:ind + 1]
    display(row)
    row = Box(row.iloc[0].to_dict())
    _trial_N = row.trial_N
    # trial_N = row.overall_trial_N
    stim = row.stim
    print(f'\n`{row.stim}` is present in trial {row.trial_N} of session {row.sess_N} for subject\n{row.subj_N} and is flashed at time slot(s) {row.stim_flash_order} of the encoding phase\n'.replace('\n', ' ').strip())
    seq = ['[]' if i not in row.stim_locs else row.stim for i in range(8)]
    options = ['[]' if i not in row.stim_locs else row.stim for i in range(8, 13)]
    print(f"\nSpatial arrangment of this stimulus in the trial:\n\tSequence: {' '.join(seq)}\n\tOptions: {' '.join(options)}")
    return row, stim


@app.cell
def _(row, this_subj_behav):
    fig_cols = [f"figure{i}" for i in range(1, 9)]
    choice_cols = [f"choice{i}" for i in range(1, 5)]

    trial_data = sequence = this_subj_behav.query(
        f"sess_N == {row['sess_N']} & trial_N=={row['trial_N']}"
    )
    trial_sequence = trial_data[fig_cols]
    trial_choices = trial_data[choice_cols]

    print(f"Sequence: {' '.join(trial_sequence.values[0].tolist())}")
    print(f"Options: {' '.join(trial_choices.values[0].tolist())}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's the average EEG activity for that stimulus across all trials of this session
    """)
    return


@app.cell
def _(os, redirect_stdout, stim, this_sess):
    with redirect_stdout(open(os.devnull, "w")):
        stim_flash_eeg_sess_epochs = this_sess.get_stim_flash_eeg_epochs(stim)
        sess_stim_erp = stim_flash_eeg_sess_epochs[stim].average()
        sess_stim_erp_plot = sess_stim_erp.plot_joint()
        sess_stim_erp_plot_psd = sess_stim_erp.plot_topomap(times='peaks', ch_type='eeg')
        _all_sess_stim_erp_plot = sess_stim_erp.compute_psd().plot_topomap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And here's the average EEG activity for that stimulus across all trials and sessions
    """)
    return


@app.cell
def _(stim, this_subj):
    stim_flash_eeg_all_sess_epochs = this_subj.get_stim_flash_eeg_epochs(stim)
    all_sess_stim_erp = stim_flash_eeg_all_sess_epochs[stim].average()
    _all_sess_stim_erp_plot = all_sess_stim_erp.plot_joint()
    all_sess_stim_erp_plot_topo = all_sess_stim_erp.plot_topomap(times='peaks', ch_type='eeg')
    all_sess_stim_erp_plot_psd = all_sess_stim_erp.compute_psd().plot_topomap()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also get the ERP associated with any recorded EEG events
    """)
    return


@app.cell
def _(this_subj_behav):
    response_events = this_subj_behav["solution_key"].unique().tolist()
    print(f"{response_events = }")
    return (response_events,)


@app.cell
def _(mne, os, redirect_stdout, response_events, this_subj):
    selected_events = response_events  # ['trial_start']

    with redirect_stdout(open(os.devnull, "w")):
        this_subj_erps = this_subj.get_erp(
            selected_events, tmin=-0.2, tmax=0.5, erp_by_sess=True
        )
    # this_subj.get_erp?

    all_sess_erp = mne.combine_evoked(list(this_subj_erps.values()), weights="equal")
    erp_plot = all_sess_erp.plot()
    return (this_subj_erps,)


@app.cell
def _(pd, px, this_subj_erps):
    this_sess_erp, this_sess_ch_names = (
        this_subj_erps[1].get_data(),
        this_subj_erps[1].info["ch_names"],
    )
    this_subj_erps[1].plot()

    this_sess_erp = pd.DataFrame(this_sess_erp.T, columns=this_sess_ch_names)

    px.line(
        this_sess_erp,
        labels={"value": "Amplitude", "index": "Time (ms)", "variable": "Channel"},
        title="ERP Subj 1 - Sess 1",
        width=650,
        height=500,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Analyze all sessions
    """)
    return


@app.cell
def _(DIRECTORIES: "Final", WD, this_sess):
    this_sess_res = this_sess.analyze_session(
        save_dir=WD / "analysis/test-export",
        preprocessed_dir=DIRECTORIES.human.prepro,
    )
    return (this_sess_res,)


@app.cell
def _(this_sess_res):
    (
        sess_frps,
        fixation_data,
        eeg_fixation_data,
        gaze_info,
        gaze_target_fixation_sequence,
    ) = this_sess_res
    return gaze_info, gaze_target_fixation_sequence


@app.cell
def _(DIRECTORIES: "Final", WD, this_subj):
    all_sess_res = this_subj.analyze_sessions(
        save_dir=WD / "analysis/test-export",
        preprocessed_dir=DIRECTORIES.human.prepro,
        force_preprocess=False,
        reuse_ica=True,
        raise_error=False,
    )
    return


@app.cell
def _():
    # (
    #     sess_frps,
    #     fixation_data,
    #     eeg_fixation_data,
    #     gaze_info,
    #     gaze_target_fixation_sequence,
    # ) = all_sess_res[5]
    return


@app.cell
def _(gaze_info, gaze_target_fixation_sequence):
    # pprint(sess_frps)
    # sess_frps['sequence'][0].plot()
    gaze_info
    gaze_target_fixation_sequence.groupby("trial_N")["target_ind"].value_counts()
    gaze_target_fixation_sequence.query("trial_N==3")
    # _fix = fixation_data[1][0]
    # plt.plot(_fix[0], _fix[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Stimuli ERPs
    """)
    return


@app.cell
def _(this_subj):
    stim_flash_eeg_eps = this_subj.get_stim_flash_eeg_epochs()
    stim_flash_erps = {stim: eps.average() for stim, eps in stim_flash_eeg_eps.items()}
    return (stim_flash_erps,)


@app.cell
def _(stim_flash_erps):
    print(f"Stimuli: {', '.join(stim_flash_erps.keys())}")
    return


@app.cell
def _(os, redirect_stdout, stim_flash_erps):
    selected_stims = {stim: stim_flash_erps[stim] for stim in ['helicopter', 'smile', 'eye']}
    with redirect_stdout(open(os.devnull, 'w')):
        for i, (stim_1, erp) in enumerate(selected_stims.items()):
            erp.plot_joint(title=stim_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Group level
    """)
    return


@app.cell
def _(human_group):
    human_group.show_dir_struct()
    return


@app.cell
def _(human_group):
    eeg_metadata = human_group.extract_eeg_metadata()
    return


@app.cell
def _(human_group):
    combined_stats, acc_fig, rt_fig, scatter = human_group.summarize_behav()
    return


@app.cell
def _(display, human_group):
    human_group_behav = human_group.get_behav_data()
    human_group_behav["correct"] = human_group_behav["correct"].astype(int)

    human_group_acc_stats = human_group_behav.groupby(["sess_N", "subj_N"], observed=False)[
        "correct"
    ].describe()

    display(human_group_acc_stats)
    return human_group_acc_stats, human_group_behav


@app.cell
def _(human_group_behav, plt, sns):
    sns.lineplot(data=human_group_behav, x='subj_N', y='correct', hue='sess_N', errorbar=None)
    plt.show()
    plots_kwargs = [dict(data=human_group_behav, x='sess_N', y='correct'), dict(data=human_group_behav, x='sess_N', y='correct', hue='pattern')]
    for kwargs in plots_kwargs:
        _fig, _ax = plt.subplots()
        sns.lineplot(ax=_ax, **kwargs)
        _ax.set_xticks(range(1, 6))
        plt.show()
    return


@app.cell
def _(human_group_acc_stats):
    human_group_acc_stats.query("subj_N==1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Data
    """)
    return


@app.cell
def _(ANNGroupData, ANNSubjData, DIRECTORIES: "Final", WD):
    ann_data_dir = DIRECTORIES.ann.data
    ann_export_dir = WD / "TEST/ANN_ANALYSIS"

    qwen2_5_72B = ANNSubjData(
        ann_data_dir, ann_export_dir, ann_id="Qwen--Qwen2.5-72B-Instruct"
    )
    ann_group = ANNGroupData(ann_data_dir, ann_export_dir)


    # qwen2_5_72B = ANNSubjData(
    #     DIRECTORIES.ann.data, ann_export_dir, ann_id="Qwen--Qwen2.5-72B-Instruct"
    # )
    # ann_group = ANNGroupData(ann_data_dir, ann_export_dir)

    return ann_group, qwen2_5_72B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyze Data
    """)
    return


@app.cell
def _(qwen2_5_72B):
    print("\n".join([i for i in dir(qwen2_5_72B) if not i.startswith("_")]))
    return


@app.cell
def _(qwen2_5_72B):
    qwen2_5_72B_behav = qwen2_5_72B.get_behav_data()
    qwen2_5_72B_behav.head()
    # qwen2_5_72B.get_behav_rdms()
    # qwen2_5_72B.get_layer_acts()
    # qwen2_5_72B.list_contents()
    # qwen2_5_72B.get_run_info()
    # qwen2_5_72B.get_rdms_on_all_layers()
    return


@app.cell
def _(DATASET, ann_group):
    ann_group_behav = ann_group.get_behav_data().merge(
        DATASET, on=["item_id", "masked_idx", "pattern", "solution"]
    )

    # category_cols = {i:"category" for i in (
    #     ["ann_id", "pattern", "item_id", "trial_type"]
    #     + [f"figure{i}" for i in range(1, 9)]
    #     + [f"choice{i}" for i in range(1, 5)]
    # )}
    # ann_group_behav = ann_group_behav.astype(category_cols)

    ann_group_behav.head(3)
    return (ann_group_behav,)


@app.cell
def _(DATASET, PATTERNS, ann_group_behav, display, pd):
    n_items = len(DATASET)
    n_item_per_patt = list(set(DATASET.groupby(['pattern'])['item_id'].nunique()))[0]
    ann_ids = ann_group_behav['ann_id'].unique()
    pattern_index = pd.MultiIndex.from_product([PATTERNS, ann_ids], names=['pattern', 'ann_id'])
    q = 'cleaned_response==figure7 & cleaned_response!=solution'
    copying_df = ann_group_behav.query(q).groupby(['ann_id'])['pattern'].value_counts().reset_index()
    copying_df['count'] = copying_df['count'] / n_item_per_patt
    copying_df = copying_df.rename(columns={'count': 'pct_copying'})
    copying_df = pattern_index.to_frame(index=False).merge(copying_df, on=['pattern', 'ann_id'], how='left').fillna(0)
    copying_df['ann_id'] = copying_df['ann_id'].str.replace('.+--', '', regex=True)
    display(copying_df.head(5))
    pivot_df = copying_df.pivot(index='pattern', columns='ann_id', values='pct_copying').T
    print('Percent of trials where model repeated last visible element, per pattern type:')
    pivot_df.style.background_gradient(axis=1, cmap='YlOrRd').format(precision=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Combined Analysis
    """)
    return


@app.cell
def _(CombinedData, DIRECTORIES: "Final", SEQ_FILE):
    combined_data = CombinedData(DIRECTORIES.ann, DIRECTORIES.human, SEQ_FILE)
    return (combined_data,)


@app.cell
def _(combined_data):
    print("combined_data available methods/properties:")
    print("\n".join([f"\t- {i}" for i in dir(combined_data) if not i.startswith("_")]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare Performance
    """)
    return


@app.cell
def _(combined_data, display):
    perf_data_all = combined_data.get_perf_data()
    print(f"columns: {perf_data_all.columns.tolist()} ")
    display(perf_data_all.head())
    return (perf_data_all,)


@app.cell
def _(perf_data_all):
    perf_data_all.groupby("type")["correct"].describe()
    perf_data_all.groupby(["pattern", "type"])["correct"].mean()
    # perf_data_all.groupby("id")['correct'].mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Representational Similarity Analysis
    """)
    return


@app.cell
def _(DIRECTORIES: "Final", combined_data):
    human_ds, llm_ds = combined_data.get_rsa_datasets(
        human_data_dir=DIRECTORIES.human.analyzed / "RSA-FRP-frontal",
        ann_data_dir=DIRECTORIES.ann.analyzed
        / "RSA-seq_tokens-metric_correlation/best_layer",
        level="pattern",
    )
    return human_ds, llm_ds


@app.cell
def _(WD, combined_data, human_ds, llm_ds):
    _rsa_repr = combined_data.compare_representations(human_ds, llm_ds, n_perm=0, n_boot=0, boot_conf_int=(2.5, 97.5), random_state=None, pbar=True, pbar_perm=True, pbar_boot=True, descriptor_match=None, similarity_metric='corr', dissimilarity_metric='correlation', tail='two-sided', save_dir=WD / 'TEST')
    _observed_corrs, _permuted_corrs, _bootstrap_corrs, df_res = _rsa_repr
    return (df_res,)


@app.cell
def _(df_res):
    df_res.query("id1=='group'")
    return


@app.cell
def _(df_res):
    df_res["corr"].plot(kind="hist")
    return


@app.cell
def _(DIRECTORIES: "Final", WD, combined_data, pd, plt):
    res = []
    res_dfs = []
    for _human_dir in ['RSA-FRP-frontal', 'RSA-Response_ERP-frontal', 'RSA-Rest_ERP-frontal']:
        human_dir = DIRECTORIES.human.analyzed / _human_dir
        human_ds_1, llm_ds_1 = combined_data.get_rsa_datasets(human_data_dir=human_dir, ann_data_dir=DIRECTORIES.ann.analyzed / 'RSA-seq_tokens-metric_correlation/best_layer', level='pattern')
        _rsa_repr = combined_data.compare_representations(human_ds_1, llm_ds_1, n_perm=0, n_boot=0, boot_conf_int=(2.5, 97.5), random_state=None, pbar=True, pbar_perm=True, pbar_boot=True, descriptor_match=None, similarity_metric='corr', dissimilarity_metric='correlation', tail='two-sided', save_dir=WD / 'TEST')
        _observed_corrs, _permuted_corrs, _bootstrap_corrs, df_res_1 = _rsa_repr
        df_res_1['corr'].plot(kind='hist')
        plt.show()
        df_res_1 = df_res_1.query("id1=='group'").copy()
        df_res_1['type'] = _human_dir.split('-')[1]
        res_dfs.append(df_res_1)
    df_res_1 = pd.concat(res_dfs).reset_index(drop=True)
    return df_res_1, llm_ds_1


@app.cell
def _(ANN_ID_MAPPING, ANN_ID_ORDER, df_res_1, plt, sns):
    df_res_1['id2'] = df_res_1['id2'].replace(ANN_ID_MAPPING)
    _fig, _ax = plt.subplots()
    sns.barplot(df_res_1, x='corr', y='id2', hue='type', order=ANN_ID_ORDER, ax=_ax)
    _ax.grid(axis='x', ls='--', lw=0.75)
    _ax.legend(bbox_to_anchor=(1, 1))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DRAFT
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TESTING
    """)
    return


@app.cell
def _(human_group):
    human_group.get_frp_data()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Controls
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Control 1 - EEG averaging influence
    """)
    return


@app.cell
def _():
    import numpy as np
    from tqdm.auto import tqdm
    from ar_analysis.utils.analysis_utils import save_pickle
    from mne.io import concatenate_raws
    from rsatoolbox.data import Dataset
    from rsatoolbox.rdm import calc_rdm, compare_cosine, compare as compare_rdm
    from analysis_rsa import get_reference_rdms
    from rsatoolbox.rdm.rdms import load_rdm
    from rsatoolbox.data.dataset import load_dataset
    import re

    return Dataset, calc_rdm, compare_rdm, np, re, tqdm


@app.cell
def _(WD):
    save_dir_1 = WD / 'analysis/TEST/RSA-Random_EEG_Windows'
    save_dir_1.mkdir(parents=True, exist_ok=True)
    return (save_dir_1,)


@app.cell
def _(DIRECTORIES: "Final", human_group, mne, save_dir, tqdm):
    import contextlib
    import gc
    from loguru import logger

    for subj_N, subj_obj in tqdm(human_group.subjects.items()):
        try:
            behav, et_trials, eeg_trials = subj_obj.get_trials_data(
                DIRECTORIES.human.prepro,
                eeg_incomplete="skip",
            )
        except Exception:
            logger.exception(
                f"Control 1 export failed for subj_{subj_N:02}; skipping subject."
            )
            continue

        if len(behav) == 0 or len(et_trials) == 0 or len(eeg_trials) == 0:
            logger.warning(
                f"Control 1 export has empty trial data for subj_{subj_N:02}; skipping subject."
            )
            continue

        behav.to_csv(save_dir / f"subj{subj_N:02}-behav.csv")

        et_trials_concat = None
        eeg_trials_concat = None
        et_memmap = save_dir / f".subj{subj_N:02}-et-concat.dat"
        eeg_memmap = save_dir / f".subj{subj_N:02}-eeg-concat.dat"

        try:
            et_trials_concat = mne.concatenate_raws(et_trials, preload=str(et_memmap))
            et_trials_concat.save(
                save_dir / f"{subj_N:02}-trials_et.fif",
                overwrite=True,
            )

            eeg_trials_concat = mne.concatenate_raws(eeg_trials, preload=str(eeg_memmap))
            eeg_trials_concat.save(
                save_dir / f"{subj_N:02}-trials_eeg.fif",
                overwrite=True,
            )
        finally:
            for raw in (et_trials_concat, eeg_trials_concat):
                if raw is not None:
                    with contextlib.suppress(Exception):
                        raw.close()

            for raw in et_trials + eeg_trials:
                with contextlib.suppress(Exception):
                    raw.close()

            for fpath in (et_memmap, eeg_memmap):
                with contextlib.suppress(FileNotFoundError):
                    fpath.unlink()

            del behav, et_trials, eeg_trials, et_trials_concat, eeg_trials_concat
            gc.collect()
    return


@app.cell
def _(save_dir_1):
    save_dir_1

    return


@app.cell
def _(re, save_dir_1):
    _files = [f for f in save_dir_1.glob('*') if f.is_file() and (not f.name.startswith('.'))]
    existing = []
    subj_Ns = [int(re.search('subj_(\\d{2})', f.name).groups()[0]) for f in _files]
    for _subj_N in set(subj_Ns):
        if subj_Ns.count(_subj_N) == 4:
            existing.append(_subj_N)
    print(existing)
    return (existing,)


@app.cell
def _(
    DIRECTORIES: "Final",
    Dataset,
    PATTERNS,
    c,
    calc_rdm,
    existing,
    human_group,
    mne,
    np,
    save_dir_1,
    tqdm,
):
    from loguru import logger

    fixation_estmd_duration = 0.6
    eeg_sfreq = 2048
    window_duration = round(fixation_estmd_duration * eeg_sfreq)

    def save_ds_and_rdm(ds, rdm, level, save_dir=save_dir_1):
        base_fname = f"human-subj_{_ds.descriptors['subj_N']:02}-{level}_lvl.hdf5"
        _ds.save(save_dir / f'dataset-{base_fname}', file_type='hdf5', overwrite=True)
        rdm.save(save_dir / f'rdm-{base_fname}', file_type='hdf5', overwrite=True)

    for _subj_N, subj_obj in tqdm(human_group.subjects.items()):
        if _subj_N in existing:
            continue
        try:
            _behav, et_trials, eeg_trials = subj_obj.get_trials_data(
                DIRECTORIES.human.prepro,
                eeg_incomplete="skip",
            )
        except Exception:
            logger.exception(
                f"Control 1 RDM computation failed while loading subj_{_subj_N:02}; skipping."
            )
            continue

        if len(_behav) == 0 or len(eeg_trials) == 0:
            logger.warning(
                f"Control 1 RDM computation has empty EEG data for subj_{_subj_N:02}; skipping."
            )
            continue
        info = eeg_trials[0].info
        montage = eeg_trials[0].get_montage()
        processed_eeg_trials = []

        for eeg_trial in tqdm(eeg_trials):
            eeg_trial_arr = eeg_trial.get_data()
            n_samples = int(eeg_trial_arr.shape[1] / window_duration)
            eeg_trial_windows = np.array_split(eeg_trial_arr, n_samples, axis=1)
            duration = min([e.shape[-1] for e in eeg_trial_windows])
            eeg_trial_windows = [e[:, :duration] for e in eeg_trial_windows]
            selected_window_inds = sorted(np.random.choice(len(eeg_trial_windows), size=np.random.randint(3, round(n_samples * 0.9)), replace=False))
            selected_eeg_trial_windows = [eeg_trial_windows[i] for i in selected_window_inds]
            eeg_trial_windows_avg = np.stack(selected_eeg_trial_windows).mean(axis=0)
            eeg_trial_windows_avg = eeg_trial_windows_avg[np.newaxis, :, :window_duration]
            eeg_trial_windows_avg_mne = mne.EpochsArray(eeg_trial_windows_avg, info, verbose=False)
            eeg_trial_windows_avg_mne.set_montage(montage)
            processed_eeg_trials.append(eeg_trial_windows_avg_mne)

        obs_descriptors = _behav['pattern'].to_numpy()
        measurements = [i.get_data(picks=c.EEG_CHAN_GROUPS.frontal).squeeze().flatten() for i in processed_eeg_trials]
        measurements = np.stack(measurements)

        _ds = Dataset(measurements=measurements, descriptors={'subj_N': _subj_N}, obs_descriptors={'patterns': obs_descriptors})
        rdm = calc_rdm(_ds, 'correlation')
        save_ds_and_rdm(_ds, rdm, level='sequence', save_dir=save_dir_1)

        measurements = [i.get_data(picks=c.EEG_CHAN_GROUPS.frontal).squeeze().flatten() for i in processed_eeg_trials]
        measurements = np.stack(measurements)

        pattern_inds = _behav.groupby('pattern').groups
        pattern_inds = {p: pattern_inds[p] for p in PATTERNS}

        measurements = np.stack([measurements[inds].mean(axis=0) for inds in pattern_inds.values()])
    
        _ds = Dataset(measurements=measurements, descriptors={'subj_N': _subj_N}, obs_descriptors={'patterns': PATTERNS})
        rdm = calc_rdm(_ds, 'correlation')
    
        save_ds_and_rdm(_ds, rdm, level='pattern', save_dir=save_dir_1)
    return


@app.cell
def _():
    # subj_obj = human_group.subjects[1]
    # subj_N = subj_obj.subj_N
    # # behav, et_trials, eeg_trials = subj_obj.get_trials_data(DIRECTORIES.human.prepro)
    return


@app.cell
def _():
    # get_ds_and_rdm(
    #     measurements: np.ndarray,
    #     dissimilarity_metric: str,
    #     ds_fpath: Optional[Path] = None,
    #     rdm_fpath: Optional[Path] = None,
    #     ds_type: str = "regular",
    #     descriptors: Optional[Dict] = None,
    #     obs_descriptors: Optional[Dict] = None,
    #     channel_descriptors: Optional[Dict] = None,
    #     time_descriptors: Optional[Dict] = None,
    # )
    return


@app.cell
def _(human_random_segmt_rdm_patt_level):
    import matplotlib.pyplot as plt
    plt.imshow(human_random_segmt_rdm_patt_level.get_matrices()[0])
    return (plt,)


@app.cell
def _(calc_rdm, compare_rdm, human_random_segmt_rdm_patt_level, llm_ds_1):
    llm_rdms = []
    corr_vals = []
    for llm, _ds in llm_ds_1.items():
        llm_rdm = calc_rdm(_ds, 'correlation')
        comp = compare_rdm(llm_rdm, human_random_segmt_rdm_patt_level, method='corr')
        corr_vals.append(comp[0])
    corr_vals
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Control 2 -
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
