# RUN-006: five-stage balanced accuracy and literature-based method comparison

## Objective and reasons for the preceding result

User authorized active improvements targeting at least 70% balanced accuracy and requested broad use of literature and model families. This protocol treats 70% as a target, never a guaranteed score or a criterion for hiding failed candidates. All five stages remain separate. Balanced accuracy is the unweighted mean recall across Wake/N1/N2/N3/REM. It is different from ordinary accuracy and macro-F1.

The preceding dual CNN achieved 64.28% ordinary accuracy but 59.65% balanced accuracy at 0 ms. Its N1 recall was 34.70%; N3 recall 52.87%. The earlier objective selected macro-F1 and used square-root class weights. Weak rare-stage recall, restricted beat-only physiology, signal quality and subject variability are plausible limitations; these are hypotheses rather than proof of implementation errors. The new audit rechecks original XML labels and EDF clocks for every annotated recording before fitting. No label/time shift is optimized from model scores.

## Data, split and safeguards

Reuse immutable preparation from `bcg_full_training_runs/20261102T213108Z/prepared`. The original participant split remains: 1279 annotated training, 274 validation and 275 test recordings; 79 additional prepared participants lack published beat annotations. Common usable epochs remain 877099 / 186559 / 194310; expanded targets remain 1305764 / 277650 / 281712. There are no per-participant fitting/validation epoch caps.

The validation participant list is split once by seed 20261103 into 137 calibration and 137 selection participants. Every common epoch from each group is used at every noise level. Calibration participants tune only the single class-prior adjustment coefficient. Selection participants select checkpoints, base methods, and ensembles. A participant never belongs to both validation roles. These are development sets, not independent confirmation: previous experiments already used the original full validation partition.

All original-common targets are retained, including ones with unavailable extra sensor features (represented as missing). Model performance does not determine quality exclusions. Expanded model training uses all expanded targets. No annotation stage fields, scored respiratory events, EEG, EOG, chin EMG, participant labels, or true neighboring stages enter predictors. XML labels supply targets and identity audits only. Inputs and offline normalization never use those labels.

The old reference test has been inspected and influenced development. Lock selection in all three input tracks before any new test inference. Evaluate one validation winner per prespecified track, including its common and expanded coverage; label all test comparisons exploratory. Never choose candidates, class adjustments, sensor exclusions or ensembles from test scores. Reuse complete cached predictions on recovery; the old dual-CNN reference predictions are transformed from saved outputs if needed, without repeating its inference. Independent confirmation requires a new cohort or prospective recordings.

## Three input tracks

1. **Beat-only**: the unchanged 140 HRV/QC/context values, their missingness indicators for neural models, and the 2-Hz RR/relative-RR/validity stream. Full definitions are in the preceding `bcg_full_training/PROTOCOL.md` and `bcg_jitter_staging/FEATURES.md`.
2. **ECG-derived breathing**: beat-only plus 36 features obtained from QRS amplitude modulation. For accepted normal intervals, take the maximum ECG within ±8 samples of the ending R-point minus the minimum in samples -32 through -8 at 256 Hz. Interpolate amplitudes at 4 Hz using clean time anchors, reject gaps longer than 3 s, and extract the breathing features below. This is an additional ECG-waveform input, not beat-only information.
3. **Multiple sensors**: 326 feature values total: original 140, ECG-derived breathing 36, thorax/abdomen/airflow 108, SpO2 24 and leg movement 18. Leg EMG-derived movement is explicitly not wrist actigraphy. Channels are selected by exact EDF labels `Thor`, `Abdo`, `Flow`, `SpO2`, `Leg`. EEG/EOG/chin EMG are excluded. Missing channels remain explicit and do not remove participants.

Extra-signal features use clean waveforms and do not change with the synthetic RR jitter. Their results therefore measure sensitivity to RR timing error in a system with additional intact sensors. They must never be presented as pure BCG timing-only results. All methods are offline: centered windows, whole-night relative statistics and bidirectional context use future unlabeled samples.

### Additional features and extraction

For each of EDR, thorax, abdomen and airflow, use centered 30/120/300-s windows. Resample measured respiratory channels from 32 to 4 Hz with polyphase anti-alias filtering. Require at least 80% valid expected samples, interpolate small missing stretches within the window for feature calculation, apply a second-order 0.08–0.6-Hz bandpass, then Welch PSD (up to 512 samples per segment). Extract six values per window:

- dominant respiratory frequency in 0.08–0.6 Hz;
- normalized spectral entropy in that band;
- 0.1–0.4-Hz power as a fraction of 0.04–1-Hz power;
- IQR of filtered amplitude;
- coefficient of variation of detected interbreath intervals (peaks at least 1 s apart, prominence 0.2×SD; at least three intervals);
- valid window coverage relative to the complete requested duration.

For SpO2, mark values outside 50–100% missing. At each of the three windows extract mean, minimum, SD and fraction below 90%. For leg movement extract RMS, IQR and fraction with absolute magnitude exceeding the recording's median absolute signal plus twice its SD. Require 80% expected valid samples. These simple movement features are our implementation choices, not actigraphic ENMO/LIDS reproductions.

There are 93 extra raw values: 72 respiratory, 12 oxygenation and 9 movement. Append 93 per-night median/IQR-relative copies, clipped ±10; constant-column scale is 1. This yields 186 extra values. Population median/IQR scaling is fitted on all clean expanded training rows only. Missing neural inputs are zero-filled after population scaling with explicit missingness indicators. Forest/linear models impute from training data; boosting methods use their documented missing-value handling. Feature availability and EDF/XML/source signatures are recorded per participant.

## Twenty-four base comparisons

### Twelve classical candidates

All use every 877099 common training target, each once with a deterministic jitter assignment. Class weights are inverse training class frequency. Full validation uses every common epoch separately at four fixed noise levels.

- Random Forest, Extra Trees: 300 estimators, max depth 24, min leaf 30; training median imputation plus indicators.
- Histogram gradient boosting: 300 iterations, 63 leaves, min leaf 100, learning rate .05, L2 5, no early stopping.
- XGBoost: 500 rounds, depth 6, learning rate .035, subsample/column fraction .8, child weight 30, L2 10.
- LightGBM: 500 rounds, 63 leaves, min child 100, learning rate .035, L2 10.
- CatBoost: 500 rounds, depth 6, learning rate .035, L2 10, multiclass loss.
- AdaBoost: 150 depth-two trees, min leaf 100, learning rate .05, median imputation.
- Linear SVM, approximate RBF SVM, logistic classifier: SGD with inverse-frequency sample weights and 10 complete passes; training imputation/scaling; alpha .0001. RBF uses 256 Nyström landmarks (gamma .01) fitted from up to 4096 deterministic training rows, while classifier fitting uses every target. This is a scalable approximation, not an exact full RBF-kernel SVM reproduction.
- Additional XGBoost and LightGBM candidates use the multiple-sensor inputs.

### Ten neural candidates

- Beat-only dual TCN with inverse-frequency weights.
- Beat-only dual TCN with focal gamma 2 and square-root inverse-frequency weights.
- Beat-only dual TCN using all expanded targets, inverse-frequency weights and lower loss weights for recovered targets: .1 for first three passes, .5 subsequently; original-common targets retain weight 1. This quality-based curriculum is prespecified and does not inspect validation/test errors.
- Beat-only two-layer bidirectional LSTM and two-layer bidirectional GRU, 64 units per direction, inverse-frequency weights.
- Beat-only three-layer Transformer, width 96, 4 heads, feedforward 256, sinusoidal positions and padding mask; inverse-frequency weights.
- Beat-only per-epoch MLP, without RR waveform branch, inverse-frequency weights.
- ECG-derived-breathing dual TCN.
- Multiple-sensor dual TCN and bidirectional GRU.

All neural models encode the feature/missingness vector to 96 channels. Except the MLP, they also encode the three-channel RR stream with kernel-five convolutions 3→16→32 and validity gating, then fuse to 96 channels. TCN blocks have two kernel-three convolutions per block, dilations 1/2/4/8 and dropout .2. The receptive field is 61 epochs (30.5 min). Chunks contain 64 center targets plus 30 halo epochs per side; predictions outside the center are discarded. Recurrent/attention models use the same full 124-epoch input span. Chunks preserve recording identities, actual timeline gaps, and the final partial chunk. Every selected training target is visited exactly once per complete pass.

Training: AdamW .0005, weight decay .01, batch16, gradient clipping1, at most60 passes, patience10 on selection balanced accuracy, learning-rate halving after3 unimproved passes down to1e-6. Save best/last checkpoints including optimizer, scheduler, RNG, current batch and history. No dropout-time test averaging is used.

### Two saved controls and ensembles

Evaluate the existing full-data HRV-only and dual TCN checkpoints on the new validation roles without refitting. This supplies 24 base candidates total. As prespecified additional candidates, average the adjusted class probabilities of the top three selection-ranked models within a track (or all if only two). Fit the ensemble's single extra prior coefficient on calibration participants and score it on selection participants. No ensemble is created for a single-candidate track. One winner per track is locked before exploratory test evaluation.

## Balanced decision rule and reporting

For training priors pi and raw class scores z, use `softmax(z − beta log(pi))`. Beta candidates are [-.5,0,.25,.5,.75,1]. Select one beta per checkpoint across all four noise levels using mean calibration balanced accuracy; ties prefer the smallest absolute beta. Rank candidates by mean selection balanced accuracy over 0/20/40/60 ms. Lexical names break exact candidate ties. Accuracy, macro-F1, kappa, per-stage recall/F1 and common/expanded coverage remain reported. Softmax SVM scores and class-weighted/post-adjusted probabilities are not guaranteed calibrated; reliability diagnostics must disclose this.

All charts and numeric reports are retained, including failed-to-reach-target outcomes. Plot optimization histories, full validation comparisons, target line, noise robustness, per-stage recall, confusion matrices, participant variation, feature-group masking ablations, ROC/PR, reliability, coverage and identifier-selected hypnograms. Distinguish neural epochs, SGD passes, tree counts and boosting rounds. Fixed training diagnostic subsets (≤5000) do not cap fitting. CatBoost learn loss is a weighted objective. Group masking at 40 ms uses selection validation only and can create out-of-distribution inputs; it is sensitivity analysis, not causal importance. Bootstrap intervals resample participants.

## Literature map and applicability

This is a concrete representative suite spanning relevant families, not an assertion that every published architecture has been reproduced or that adding methods guarantees an outcome. Numerical settings and the extra feature set above are our prespecified adaptation.

- [Suliman et al., Sensors 2023](https://doi.org/10.3390/s23052693): cardiac timing jitter, HRV and SVM reference. Preserve RR-noise simulations and beat-only comparison.
- [Xiao et al., 2013](https://doi.org/10.1016/j.bspc.2013.06.001): broader HRV and Random Forest; participant-separated comparison matters.
- [Sridhar et al., npj Digital Medicine 2020](https://www.nature.com/articles/s41746-020-0291-x): local/dilated convolution on instantaneous heart rate. Motivates temporal cardiac encoding; their four-class results are not five-class balanced accuracy.
- [Garingo et al., J Clin Neurophysiol 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186678/): MESA HRV, motion, oxygenation, RNN/XGBoost. Their four-class multisensor result is not an equivalent beat-only five-class benchmark.
- [Sun et al., SLEEP 2020](https://pubmed.ncbi.nlm.nih.gov/31863111/): ECG and respiration with convolution/recurrent models and five-stage evaluation. Motivates additional respiratory inputs.
- [MESA respiratory-signal benchmark, SLEEP 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12417017/): compares SVM/MLP/AdaBoost/RF/XGBoost/LSTM/TCN, HRV/actigraphy, measured breathing and ECG-derived respiration. We use leg movement, not wrist actigraphy, and our feature set/quality masks differ from that paper.
- [Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314): class-frequency logit adjustment for imbalanced classification. Here the coefficient is fitted on calibration participants only.
- [Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002): focal loss downweights easy examples. Its original application was detection; its application here is an experiment rather than established sleep-staging improvement.

Extra Trees, LightGBM, CatBoost, approximate kernels, attention, quality weighting and ensembles broaden this empirical comparison. They should not be attributed as exact implementations of the cardiac papers above. More complex SSM/Mamba or generative architectures and cross-cohort pretraining require additional implementation/data validation; they are not silently counted as trained candidates.
