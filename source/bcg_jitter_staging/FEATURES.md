# Feature extraction and model comparison, version 2

## Literature behind the choices

- [Suliman et al., 2023](https://doi.org/10.3390/s23052693): Gaussian perturbations of beat timestamps; HRV features from five-minute windows; SVM reference. Our original six-feature, 50-person experiment is preserved in `experiment.py`.
- [Xiao et al., 2013](https://doi.org/10.1016/j.bspc.2013.06.001): Random Forest classification from 41 HRV features and explicit comparison of subject-specific and subject-independent evaluation. This supports a wider HRV feature set and a participant-separated comparison.
- [Garingo et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11186678/): MESA HRV, five-minute feature extraction, LSTM and XGBoost. Their four-class result also used motion and oxygenation, so it is not directly comparable with our five-class, beat-only experiment.
- [2025 SLEEP benchmark](https://academic.oup.com/sleep/article/48/9/zsaf091/8112037): compares SVM, MLP, AdaBoost, Random Forest, XGBoost, LSTM and TCN across sensor combinations. It motivates comparing model families rather than assuming one wins universally.
- [NSRR MESA HRV documentation](https://sleepdata.org/datasets/mesa/pages/hrv-analysis.md): Somte-derived R-point locations at 256 Hz and beat types: artifact=0, normal sinus=1, ventricular ectopic=2, supraventricular ectopic=3.

## Inputs and extraction

The active benchmark reads downloaded `mesa-sleep-<ID>-rpoint.csv` files. It uses only `RPoint` and `Type`; any annotation sleep-stage column is ignored. R-point samples divided by 256 give beat times in seconds. Consecutive differences give RR intervals. Only intervals bounded by two normal-sinus annotations and between 0.3 and 2 seconds contribute to HRV. The resulting features are estimates from annotated normal intervals, with the limitations of the original automated beat classification.

The official CSV rows are sorted chronologically as paired RPoint/Type observations before RR calculation. The first run found 35 exports with nonchronological row order, although their timestamps were finite, unique and within recording bounds. Sorting preserves both the timestamps and their beat types. Duplicate, negative, nonfinite or out-of-range times remain errors. Coverage records include whether sorting was needed and the original number of backward transitions. The recovery rebuilds features under a new source hash.

The reference targets remain the repository's prepared XML sleep labels: Wake, N1, N2, N3, REM. The fixed training/validation/test participant split is preserved. Of 1,907 prepared participants, NSRR publishes beat annotations for 1,828; the other 79 are explicitly excluded by file availability before modeling. Raw ECG quality flags exclude windows containing nonfinite or constant ECG.

Each 30-second target epoch uses a five-minute window centered on its midpoint. Require at least 80 intervals, at least 90% normal/plausible intervals, at least 270 seconds of beat coverage, no beat gaps longer than 3 seconds, and valid ECG across the covered epochs. Coverage and exclusions are saved per participant. The same intersection of usable epochs is used at all four jitter levels, preventing differences in sample inclusion from driving the robustness comparison.

For each recording, independent Gaussian offsets are added to all beat timestamps. Target interval MAEs are 0, 20, 40 and 60 ms; offset SD is target MAE × sqrt(pi)/2. The code records realized MAE before interval rejection. Beat identities, normal-beat flags and window anchors remain fixed across levels. This simulates timing errors, without simulating BCG waveforms, missed beats, extra beats, or correlated sensor noise.

## The 30 base features

All time statistics are calculated on accepted RR intervals. Differences between successive intervals never bridge rejected intervals.

| Feature | Calculation / units |
|---|---|
| `mean_rr_s` | Mean RR interval, seconds |
| `median_rr_s` | Median RR interval, seconds |
| `sdnn_s` | Sample standard deviation of accepted normal RR intervals, seconds |
| `iqr_rr_s` | RR 75th percentile minus 25th percentile, seconds |
| `mad_rr_s` | Median absolute deviation from median RR, seconds |
| `cv_rr` | SDNN divided by mean RR |
| `mean_hr_bpm` | Mean of 60/RR, beats/minute |
| `sd_hr_bpm` | Standard deviation of 60/RR, beats/minute |
| `rmssd_s` | Root mean square of successive accepted RR differences, seconds |
| `sdsd_s` | Sample SD of successive accepted RR differences, seconds |
| `pnn20` | Fraction of successive accepted RR differences exceeding 20 ms |
| `pnn50` | Fraction exceeding 50 ms |
| `poincare_sd1_s` | SDSD divided by sqrt(2), seconds |
| `poincare_sd2_s` | sqrt(max(0, 2 SDNN² − SDSD²/2)), seconds |
| `poincare_ratio` | SD1 / SD2 |
| `sample_entropy` | m=2, r=0.2×SD, Chebyshev matches, excluding self matches; longest consecutive accepted RR run; missing if fewer than 30 intervals or no matches |
| `rr_skewness` | Third standardized RR moment |
| `rr_excess_kurtosis` | Fourth standardized RR moment minus 3 |
| `lf_power_s2` | Integrated tachogram PSD from 0.04 to <0.15 Hz, seconds² |
| `hf_power_s2` | Integrated PSD from 0.15 to 0.4 Hz, seconds² |
| `lf_hf_ratio` | LF/HF |
| `lf_fraction` | LF/(LF+HF) |
| `hf_fraction` | HF/(LF+HF) |
| `lf_peak_hz` | Frequency of largest LF PSD bin |
| `hf_peak_hz` | Frequency of largest HF PSD bin |
| `spectral_entropy` | Shannon entropy of normalized 0.04–0.4 Hz power, divided by log(number of bins) |
| `n_intervals` | Number of candidate RR intervals in the window |
| `plausible_rr_fraction` | Fraction of candidates with normal endpoints and 0.3–2 s RR |
| `coverage_fraction` | Accepted beat time span / 300 s |
| `valid_ecg_fraction` | Fraction of covered ECG epochs passing finite/nonconstant checks |

For spectral features, the accepted RR tachogram is linearly interpolated at 4 Hz. Welch PSD uses a Hann window, up to 1,024 samples per segment, 50% overlap and constant detrending. Frequency bands are integrated with the trapezoid rule. This differs from the original pilot's Lomb–Scargle approach and is recorded as a versioned extraction change. Band limits refer to PSD bin centers. No VLF estimate is used from these short windows.

## The 24 context features (54 total)

For each of eight base features—mean RR, SDNN, RMSSD, mean HR, LF power, HF power, LF/HF and sample entropy—calculate three additional values: mean and SD over the five neighboring epochs (±2 epochs, finite values only), and next epoch minus previous epoch. Names are `context_mean_<base>`, `context_std_<base>` and `context_delta_<base>`.

Context never crosses participant boundaries. These are offline features: centered windows and neighboring epochs use future signal context. No previous true sleep-stage labels enter the model.

## Models and selection

Train Random Forest, Extra Trees, histogram gradient boosting, XGBoost and an MLP, each with clean training features and with features sampled across all four jitter levels: ten candidates. SVM remains the original pilot reference. Trees model nonlinear interactions in tabular features; the MLP provides a neural-network comparison. This run does not train an LSTM or TCN.

All training participants contribute up to 128 uniformly sampled usable epochs, with identical sampled epoch identities for clean and augmented training. Augmentation replaces each selected epoch with one randomly chosen jitter level, keeping training budgets equal. More eligible epochs exist than are fitted; the cap bounds tree and neural training cost and balances participant contributions. Validation similarly uses up to 128 epochs per validation participant, identically sampled at each jitter level. The final selected model evaluates every usable test epoch.

Fixed model settings are recorded with each run. Random Forest / Extra Trees use 200 trees, depth≤20, minimum 20 samples per leaf and balanced class weights. Histogram boosting uses 200 iterations, 31 leaves, minimum leaf size 100, rate 0.05, L2=1 and balanced class weights. XGBoost uses 300 trees, depth 5, rate 0.05, row/column sampling 0.8, minimum child weight 20, L2=5 and balanced sample weights. The MLP uses hidden layers 128/64, ReLU/Adam, batch size 512, L2=0.01, and balanced resampling. Its imputation and scaling fit only training data; clean validation macro-F1 chooses the training epoch, with patience 5 and maximum 30 passes. It never uses a random split of neighboring training epochs for early stopping.

The winning candidate maximizes mean five-class validation macro-F1 over the four jitter levels. Only that winner is evaluated on the test participants. Report accuracy, macro-F1, kappa, per-stage scores, confusion matrices, participant results and feature coverage. Validation permutation importance at 40 ms ranks features by macro-F1 loss, with three repeats on up to 4,000 validation epochs. Correlated features can share or mask importance; rankings describe model use, not physiological causation.

Cohen's kappa is reported as null (undefined), with an explanatory note, when its expected-disagreement denominator is zero, such as an individual whose usable reference and predicted epochs are all Wake. Such participants remain included in accuracy, macro-F1, confusion matrices and coverage. This reporting policy does not change fitting, feature extraction or model selection.

Completed features can be reused with `--reuse-features-from <PRIOR_RUN_DIRECTORY>`. Reuse requires identical feature/ECG helper sources and extraction function, identical cohort/input inventory/seed/jitter settings, complete coverage and all cache files. Every cache signature is rechecked against annotation and prepared-data provenance. The new manifest records the original feature source version separately from the current training/reporting source version.

Artifacts: `manifest.json`, `coverage.json`, `validation_results.json`, `selection.json`, `feature_importance_validation.json`, `test_results.json`, test prediction CSVs, fitted candidates, saved source and frozen dependencies. Feature arrays and beat times remain under the cluster's ignored `data/derived/` directories. The old 50-person pilot is retained as a preliminary experiment; its scores are not directly comparable because detector, features and cohort differ.

## Charts and downloadable report

`reporting.py` creates PNG and PDF versions of the figures, `REPORT.md`, `REPORT.html`, summary CSVs, and `results_bundle.zip`. The bundle contains reports, figures, predictions and source; large fitted models and feature caches stay on the cluster.

- Ten learning plots: MLP training epochs, XGBoost/HGB boosting rounds, and Random Forest/Extra Trees tree-count curves. The last two are ensemble-growth diagnostics, not neural optimization epochs. They are not training-set-size learning curves.
- Training and clean-validation unweighted log loss and, where recorded, macro-F1. The training monitor uses a fixed sample of up to 5,000 training epochs. These diagnostics do not change selection based on final validation macro-F1 averaged across all jitter levels.
- Validation comparison of all candidates and robustness curves across 0/20/40/60 ms.
- Selected-model test macro-F1 with 95% intervals from 1,000 participant bootstrap samples.
- Validation and test confusion matrices at each noise level: counts and row-normalized views.
- Per-stage F1, participant score distributions, feature coverage and requested-versus-realized jitter checks.
- Top-25 validation permutation importance plot and a CSV containing all 54 feature rankings.
- Selected-model test one-vs-rest ROC and precision–recall curves, AUROC/AP summaries, confidence reliability diagrams, ECE and multiclass Brier score. These measure the existing predictions and do not fit a calibration model.
- Reference/predicted hypnograms for the first three test participant identifiers, at each jitter level; missing epochs appear as gaps.

To regenerate figures from existing results without retraining or reevaluating models:

```bash
.venv-transformer/bin/python bcg_jitter_staging/reporting.py data/derived/bcg_benchmark_runs/run_<JOB_ID>
```
