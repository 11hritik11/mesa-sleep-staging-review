# How the reported model works

This explains the archived implementation. Links point to readable copies with identical Python syntax trees. Execution uses the frozen files in `source/`.

## Data entering the model

The multiple-sensor setting uses 326 feature values per 30-second epoch: 140 heartbeat/HRV/context values, 36 ECG-derived breathing values, 108 measured breathing values, 24 oxygen-saturation values and 18 leg-movement values. A separate missingness flag accompanies each value for the neural models. The RR branch has three channels sampled at 2 Hz: standardized RR, night-relative RR and validity.

Neural population scaling is fitted on expanded training targets. Each recording's label-free relative features use whole-night statistics. A neural input chunk contains 64 output epochs and a 30-epoch context halo on each side: 124 input epochs total. Gaps and missingness remain explicit. This is offline context; the BiGRU does not process an entire night as one sequence in this experiment.

Read [data indexing and masks](readable/bcg_balanced_staging/balanced_data.py), [base sequence construction](readable/bcg_full_training/sequences.py), and the [feature protocol](source/bcg_balanced_staging/PROTOCOL.md).

## The three winning components

| Component | Implementation |
|---|---|
| BiGRU | A linear layer maps 652 feature/missingness inputs to 96 dimensions. A small convolutional RR encoder produces 32 more values, then a fusion layer maps the combined 128 values to 96. A two-layer bidirectional GRU uses 64 hidden units per direction. A five-class linear output retains the central 64 epochs. |
| Temporal CNN | The same feature/RR encoders and 96-dimensional fusion feed residual temporal convolution blocks with dilations 1, 2, 4 and 8. A five-class output retains the central 64 epochs. |
| LightGBM | A weighted multiclass boosted-tree model consumes the 326 feature values, with native missing-value handling. Settings include 500 boosting rounds, 63 leaves, minimum 100 child samples, learning rate 0.035 and L2 regularization 10. It does not receive the neural RR convolution stream. |

Read [BalancedNetwork](readable/bcg_balanced_staging/balanced_models.py), [TemporalBlock](readable/bcg_full_training/temporal.py), and the [LightGBM training branch](readable/bcg_balanced_staging/train_balanced_classic.py).

## Fitting and checkpoint selection

Neural fitting uses AdamW with learning rate 0.0005, weight decay 0.01, gradient clipping at 1, batch size 16, maximum 60 passes and patience 10. Each complete pass visits every declared training target. Both winning neural components use inverse-frequency class weighting. The scheduler halves the learning rate after three stale selection scores, with a minimum of 1e-6.

The balanced-study initialization seed is 20261103. The inherited temporal-loader seed remains 20261102; the original code preserves these separately. Training samples different jitter settings through its deterministic sampler. Classical training assigns one jitter setting to each training epoch with a deterministic participant-specific seed.

Validation participants are divided into calibration and selection roles. For each candidate/checkpoint, calibration chooses one shared prior-correction coefficient across the four noise conditions. Selection ranks the resulting mean balanced accuracy. Reference-test labels are not used by those selection functions. The original reference set had been inspected in earlier development, so the result remains exploratory.

Read [neural training](readable/bcg_balanced_staging/train_balanced_deep.py) and [calibration utilities](readable/bcg_balanced_staging/balanced_utils.py).

## Combining predictions and computing 71.15%

Each selected member produces a five-class probability vector after its own calibration correction. The ensemble averages these three vectors equally. Its additional correction coefficient is zero. The predicted stage is the class with the largest averaged probability.

For each synthetic RR timing-error condition, a confusion matrix is built on the identical 194,310 common reference epochs. Balanced accuracy is the arithmetic mean of five stage recalls. The four balanced accuracies are then averaged:

```text
0 ms:  0.7134442674381861
20 ms: 0.7127012941927934
40 ms: 0.7118140737303531
60 ms: 0.7080198045525663
mean:  0.7114948599784747 = 71.15% rounded
```

Read [ensemble inference](readable/bcg_balanced_staging/balanced_inference.py) and [selection/evaluation](readable/bcg_balanced_staging/finalize_balanced.py). `balanced_report.py` renders the already-computed results into reports and plots; it does not fit these models.

## What to reproduce

[REPRODUCING.md](REPRODUCING.md) supplies commands to load the original weights and recompute probabilities, as well as commands to fit candidates anew from the audited prepared features. The executable runner never replaces the original archive with the readable copies. See [verification status](VERIFICATION_STATUS.md) for which checks have actually finished.
