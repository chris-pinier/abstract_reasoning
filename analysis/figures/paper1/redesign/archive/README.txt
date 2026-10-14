MAIN-TEXT FIGURE REDESIGNS - TEMPORARY REVIEW

Temporary design drafts; manuscript source and production figures are unchanged.

Figure 2: the recovered plot contains 25 participant values per pattern. Their equal-weight mean is 82.4708%; the old pooled-trial human line averages 86.7976%. Point indices are not participant IDs, and no paired cross-pattern analysis is inferred from their order.

Figure 3A-C preserve the exact current scatterplot points to isolate layout choice. Figure 3D is a separately labeled sensitivity calculation using the equal-participant profile from Figure 2. It changes correlations (e.g., Phi-4: .688 to .784; Llama-70B: .266 to .426). Choose and document the behavioral estimand before publication.

Figures 6-7 both use matched participants/sessions. The original Figure 7 used condition-specific sets. This accounts for the response mean discrepancy (.285 versus .192). The draft is harmonized to Figure 6 and the matched comparison described in the text.

Figure 8 retains the saved Fisher-z estimator and existing participant confidence intervals. It does not silently switch to the arithmetic mean of raw correlations.

All plot labels and marks are vector in PDF. Figure 1 retains the original stimulus raster; the 400-by-400 empirical RDM is rasterized without interpolation. No new raw-data cleaning or model inference was run.

DeepSeek-R1-70B abbreviates DeepSeek-R1-Distill-Llama-70B throughout. Each output has a PNG preview and source data in data/.

01_trial_example
Original stimulus with pattern letters, correct-answer annotation and a schematic fixation-locked EEG window. The letters and annotations are not part of the experimental display. Image content is raster; annotations are vector.

02_pattern_accuracy
Human means give each of 25 participants equal weight. Whiskers are marginal 95% percentile bootstrap CIs (10,000 resamples of participant values within each pattern; seed 20260930). Model points are descriptive accuracies, without sampling intervals. The 25% line is the uniform four-choice baseline; invalid model outputs can produce below-baseline accuracy. Overall human mean across patterns: 82.47%.

03a_accuracy_profile_aligned
Aligned dot plots, ordered by accuracy. Layout comparison: values preserved from the current manuscript scatterplot. Exact values are annotated. Separate axes avoid implying that accuracy and correlation are the same quantity.

03b_accuracy_profile_compact_scatter
Compact scatter with direct model labels. Current manuscript values are preserved. The vertical axis uses the displayed range around the data, rather than the full possible correlation range. This remains the clearest option for judging the relationship between the two metrics.

03c_accuracy_profile_rank
Rank comparison using current manuscript values. It emphasizes changes in ordering, but overstates tiny rank differences if interpreted without the printed values: Phi-4 and DeepSeek correlations differ by about 0.00006. This is a descriptive alternative, not evidence of statistically distinguishable model ranks.

03d_accuracy_profile_equal_participant
Aligned dot plots, ordered by accuracy. Analysis sensitivity: correlations recomputed against Figure 2's equally weighted human profile. Exact values are annotated. Separate axes avoid implying that accuracy and correlation are the same quantity.

04_task_structure_rdms
Reference uses a binary same/different-pattern key; measured LLM RDM uses correlation distance. Panel C averages unique within- and between-pattern pairs separately for each of eight models; these are descriptive summaries, not independent pair-level inferential tests. The full trial matrix is preserved, not replaced by an RDM of pattern-averaged features.

05_layer_profiles
One panel per model with shared axes. Gray trajectories provide context; the colored dot marks the maximum task-design similarity. Normalized depth follows the existing figure convention: layer index divided by the maximum index, not an independently redefined depth measure.

06_matched_rsa_heatmap
Same 24 participants and 96 complete participant-session cells across conditions. Each regional cell is the arithmetic mean of eight model RSA values. The regional mean is separated from anatomical regions. Frontal is marked as primary. No cellwise significance is implied.

07_model_rsa_regions
Five regional values per model and condition; frontal values are diamonds and regional means are open squares. Uses exactly the matched data underlying Figure 6. This deliberately replaces the condition-specific participant sets used by the previous Figure 7, so response means change from about .285 to .192. The five regions are not treated as independent participant replicates.

08_gaze_timing_intervals
Participant dots and the existing Fisher-z mean / t-based 95% confidence intervals, back-transformed to r. N=24 in every region-measure cell. Intervals are marginal, not simultaneous or multiplicity-adjusted. The source heatmap also uses Fisher-z summaries; these are preserved. Associations do not establish that gaze/timing explains away LLM-FRP correspondence.