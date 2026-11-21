# Full-data annotated-beat MESA improvement experiment

Authorized on 2026-11-02 after discussion of removing the initial epoch caps and improving feature/sequence learning. The prior benchmark and all results are preserved.

## Cohort, objective and limitations

Reuse the verified 1,828 published NSRR R-point files in the existing participant split (1,279 train, 274 validation, 275 test); 79 of 1,907 prepared recordings have no published annotation. The original common target masks are preserved: 877,099 training, 186,559 validation and 194,310 test epochs, each 30 seconds. No per-participant epoch cap is used in new fitting or validation. All usable training participants contribute, including longer recordings. Validation and test participants never enter model fitting. Sleep-stage columns inside beat CSVs are ignored; only RPoint and Type feed features.

The reference test partition was inspected in the earlier experiment and motivated this development. Therefore new reference-test comparisons are exploratory, not an independent confirmation. A genuinely independent confirmation requires a new external cohort or prospective data. Gaussian timing jitter is a simulation, not validation against measured ballistocardiography.

Offline protocol: centered windows, future sequence context and per-night normalization use the complete unlabeled recording. These methods cannot be described as real-time staging. Population normalization uses training participants only.

## Features and extraction

The existing 54 HRV/QC/context measures remain unchanged. Complete definitions, units and literature are in `../bcg_jitter_staging/FEATURES.md`.

Add eight measures from each centered 30-s and 120-s window: mean RR (s), sample SD of RR (s), mean instantaneous HR (bpm), RR IQR (s), RMSSD (s), pNN50 (fraction), accepted-normal RR fraction, and number of candidate intervals. At least five accepted intervals are required for the six physiological measures. Differences use adjacent accepted original intervals and never bridge rejected intervals. Normal intervals require both beat types normal and RR between 0.3 and 2 s. Window membership remains anchored to clean beat times so jitter does not move target identities.

This produces 70 raw features. Append 70 per-night relative copies: subtract that recording's unlabeled median and divide by its IQR, separately for each jitter condition; use scale 1 for constant columns and clip relative values to ±10. Total: **140 HRV/QC/context values**. Trees impute missing values with missingness indicators for forests; XGBoost handles missing values natively. TCNs receive the 140 values normalized with the median/IQR of all expanded clean training rows, clipped to ±10, with missing entries zero-filled plus 140 missingness indicators. No population normalization is fitted on validation or test.

For dual TCNs, interpolate accepted RR intervals at 2 Hz on clean time anchors; three channels are (RR−1 s)/0.3 s, per-night median/IQR-relative RR (IQR floor 0.03 s), and a validity indicator. Clip the first two channels to ±10, set invalid samples to zero, and never interpolate across anchor gaps exceeding 3 s or invalid ECG epochs. The stream contains 60 samples per epoch. An explicit validity mask distinguishes missing values from genuine normalized zeros.

Expanded targets retain every original-common epoch and add eligible epochs with valid ECG and at least 80% valid RR samples at **every** jitter level. Five-minute features may be missing for recovered targets. Missing long-window HRV remains explicit. The exact new coverage is determined and reported during preparation. No synthetic labels are generated, and unannotated recordings are not silently included.

## Jitter and augmentation

The existing fixed simulations target interval mean absolute error 0, 20, 40 and 60 ms. Independent Gaussian beat offsets have sigma = target interval MAE × sqrt(pi)/2, in consistent seconds. Existing peaks/features and realized jitter are verified and reused. Tree augmentation assigns one deterministic condition to every training epoch (each target occurs once). TCN augmentation assigns one condition per contiguous center chunk, resampled deterministically each training pass. Validation covers every common target at all four fixed conditions.

## Fixed candidate comparison

Six new tree candidates:

- Random Forest with original 54 inputs, 200 trees, maximum depth 20, minimum leaf 20, inverse-frequency class weights; this isolates removing caps relative to the previous forest.
- XGBoost with original 54 inputs, 300 rounds, depth 5, learning rate .05, subsample .8, column fraction .8, minimum child weight 20, L2 penalty 5 and inverse-frequency weights.
- Random Forest with 140 inputs and minimum leaf 100, otherwise as above, at weight powers .5 and 1.
- XGBoost with 140 inputs and otherwise the same fixed settings, at weight powers .5 and 1.

Class weights are (N/(5 × class count)) raised to the specified power and normalized to weighted mean one. The prior capped random forest winner is a **saved validation-only control**, evaluated on the same full validation targets. It is not refitted, and if it wins, its saved test predictions are reused without new test inference.

Three temporal candidates: HRV TCN on common targets, dual RR+HRV TCN on common targets, and dual RR+HRV TCN on expanded targets. The last two isolate training coverage while keeping architecture fixed. Each has 64 hidden channels, four residual blocks with two kernel-three convolutions per block, dilations 1/2/4/8, per-timestep LayerNorm, GELU, dropout .2, and a five-class output. The dual branch encodes each epoch's RR with 3→16→32 channel kernel-five convolutions, stride two in the second layer, then mean pooling and validity gating. Center chunks have 64 epochs plus 30 halo epochs on each side; convolutional receptive field is 61 epochs (30.5 minutes), and halo predictions are discarded. Labels are targets only. Chunks never join recordings or compress gaps. Every selected training target occurs exactly once each pass, with no dropped final batch.

TCN fitting: AdamW learning rate .0003, weight decay .01, class-weight power .5, batch 16, gradient clipping 1, maximum 30 complete passes, early stopping after five passes without improvement in mean common-validation macro-F1 across four noise levels. Record best and last checkpoints plus optimizer/RNG/progress for recovery. Full validation is performed every pass. CPU jobs avoid competing with the existing GPU Transformer queue.

## Selection, evaluation and diagnostics

Choose the largest mean five-class macro-F1 over all common validation targets and all four jitter levels among the ten entries, with lexical tie-break. Lock selection before reference-test inference. Only the winner receives new reference-test inference. Recover completed prediction files and regenerate reports from saved values; never repeatedly use test scores for fitting or model selection. Expanded test results for a temporal winner are separate from the common-cohort comparison. Verify unique participant/epoch identities, exact target coverage, probabilities and confusion totals.

Learning curves are optimization histories, not training-set-size experiments. Tree loss diagnostics use a fixed ≤5,000-target training monitor and all clean common validation targets. Forest curves count trees, boosting curves count rounds. TCN curves use every mixed-noise training target and every common validation target averaged over four noise levels. Training diagnostics use unweighted loss while fitting uses weighted loss. Plot all saved histories, validation ranking, jitter robustness with participant bootstrap intervals, confusion matrices, stage F1, participant variation, feature importance, ROC/PR, reliability, coverage/jitter audit and identifier-selected example hypnograms. Export PNG/PDF, CSV/JSON, Markdown/HTML report and results ZIP.

For a tree winner, validation permutation importance uses a fixed ≤4,000-target diagnostic subset at 40 ms, three repeats. For a temporal winner, mask one HRV group (including its relative copies and missingness indicators) or the RR branch and score all common validation epochs at 40 ms. This is grouped masking ablation, not permutation importance or causal attribution.

## Literature motivating the design

- Suliman et al., 2023: [Sensors 23(5), 2693](https://doi.org/10.3390/s23052693): simulated beat timing error for cardiac sleep staging; do not equate synthetic ECG timing errors with measured BCG.
- Sridhar et al., 2020: [Deep learning for automated sleep staging using instantaneous heart rate](https://www.nature.com/articles/s41746-020-0291-x): local and dilated temporal convolutions motivate learning longer cardiac context.
- Garingo et al., 2024: [MESA cardiac sleep staging study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186678/): HRV and sequence modeling motivate the feature/temporal comparisons.

The architecture and numerical settings above are our prespecified experiment choices, not a claim to exactly reproduce those papers or guarantee a globally best model.
