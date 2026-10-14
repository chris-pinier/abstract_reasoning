# Paper 1 figure sources and outputs

This is the canonical home of the migrated eLife figure generators, plotting
inputs, and retained alternatives. The paper repository holds selected exports.

## Layout

- `redesign/`: active generators, numeric inputs, palette, and checksums.
- `optimal_layer_rdms/`: Figure 17 generator, matrices, and provenance.
- `assets/`: the stimulus image used by Figure 1, copied from the paper.
- `generated/`: PDF and PNG exports of the six supported manuscript figures.
- `variants/figure_1b/`: standalone accuracy heatmap and grouped-bar alternatives.
- `variants/redesign_2026-09-30/`: historical plots and design notes.
- `redesign/archive/`: all 27 historical redesign/extraction scripts.

Sources, small plotting extracts, and useful exports belong in Git. The ignored
`.temp` copies in the paper repository are legacy working copies, not the editing
or regeneration location.

## Regenerate

Run from the coding repository using Python with numpy, pandas, matplotlib, and
Pillow. These seven entry points need neither the external drive nor the paper checkout:

```sh
python analysis/figures/paper1/redesign/generate_experiment_overview_v5_real_data.py
python analysis/figures/paper1/redesign/generate_task_rdms_restyled.py
python analysis/figures/paper1/redesign/generate_layer_profiles_reordered.py
python analysis/figures/paper1/redesign/generate_frp_infographic_real_data.py
python analysis/figures/paper1/redesign/generate_eeg_measures_real_data_no_title.py
python analysis/figures/paper1/optimal_layer_rdms/generate.py
python analysis/figures/paper1/redesign/generate_accuracy_alternatives.py
```

The first six write to `generated/`; the last writes to `variants/figure_1b/`.
All accept `PAPER_FIGURE_OUTPUT` to override the output directory. Inputs resolve
relative to the scripts, independent of the working directory.

## Copy selected outputs into the manuscript

Edit and regenerate here, inspect the PDF/PNG, then copy selected exports into
the paper's `Versions/eLife/figures/`. Update its README when the selected source
changes. Commit code/data here and exported figures in the paper repository;
record the source commit in the paper README when available.

Example from the coding repo using the current sibling-checkout layout:

```sh
cp analysis/figures/paper1/generated/main_experiment_overview_v5_frp_llm_construction_no_title_real_data_panel_e_ci.pdf \
  ../paper/Paper1-LLM_and_Human_Abs_Reasoning/Versions/eLife/figures/
```

The Figure 1B alternatives have not replaced the manuscript panel.

## Inputs, provenance, and limits

`redesign/data` preserves the exact plotting inputs, extraction metadata, and
10,000 paired participant-bootstrap replicates for Figure 1E. These extracts do
not replace the full dataset. `redesign/input_checksums.json` records SHA-256
checksums, including the stimulus asset. Figure 17 has its own provenance file.

Human examples use participant 1, session 1, trial 1, Fz; pattern means use that
session's frontal averages. Human group RDMs use 24 participants and 96 matched
sessions. The LLM activation example uses Gemma-2-2B on the matched item; displayed
feature subsets do not replace full features used for RDM calculation. Panel E
intervals quantify participant uncertainty, holding the eight models fixed.

Sources were migrated from the paper's `Versions/eLife/figure_sources/` after
paper commit `6ddd7a2`; the Figure 1B candidates were added subsequently. The
initial source-preservation commit in the paper was `cc3b8c1`.

Historical scripts retain their original path assumptions and may require the
external drive, original folder layout, or MNE/h5py. Old notes may describe
superseded schematic figures. These are historical records, not supported offline
entry points. This is not yet a regeneration pipeline for every legacy paper figure.
