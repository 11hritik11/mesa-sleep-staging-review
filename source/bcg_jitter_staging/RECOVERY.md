# November 2, 2026: annotation row-order recovery

Download job **1272580** completed successfully: 1,828 MESA R-point CSVs were downloaded or verified against the official size and MD5. The token remains outside the repository.

Benchmark job **1272581** stopped during feature extraction when the parser encountered nonchronological RPoint rows in `mesa-sleep-0167-rpoint.csv`. Inspection of all 35 recordings without completed feature caches found the same row-order issue; all their beat timestamps were finite, unique, nonnegative and within recording bounds. The other 1,793 recordings had completed feature caches. No classifier fitting or benchmark test evaluation had started.

The parser now sorts each timestamp together with its beat type before forming adjacent intervals. It still rejects duplicate, invalid or out-of-range timestamps. Coverage records save whether sorting was required and the number of backward transitions in the source. Regression checks verify that shuffling input rows preserves times, normal-interval flags, and paired beat types after sorting. A real-recording check passed for both an already ordered export (0001) and a reordered export (0167), including feature extraction for the latter.

Replacement benchmark job **1272584** was submitted with the same participant split, features, jitter settings, model families, training budgets, selection rule and chart export. Features are rebuilt under the updated source hash; earlier caches and logs remain preserved. Outputs: `data/derived/bcg_benchmark_runs/run_1272584/`. Logs: `logs/mesa-bcg-models-1272584.{out,err}`.

## Recovery after complete feature extraction

Job **1272584** completed feature extraction for all 1,828 annotated recordings. Twenty-eight recordings have zero usable epochs under the existing quality rules; their coverage remains explicit. It fitted the first Random Forest candidate, then failed while writing validation reports because Cohen's kappa is undefined for an individual whose usable reference and predicted stages are all the same. Strict JSON correctly rejected the resulting NaN. No test evaluation had started.

Undefined kappa is now saved as JSON null with an explanatory note. Accuracy, macro-F1, confusion matrices, participants and selection rules are unchanged. A regression check verifies strict JSON serialization for the degenerate case, and a report from the saved forest checkpoint also serialized successfully.

Replacement job **1272588** uses `--reuse-features-from .../run_1272584`. Reuse checks exact extraction source compatibility, participant/input inventory, seed, jitter settings, complete coverage, all cache files, and per-file input signatures. The feature cache is reused without changing features or repeating extraction. The new run records separate training/reporting and feature source versions. Active outputs: `data/derived/bcg_benchmark_runs/run_1272588/`; logs: `logs/mesa-bcg-models-1272588.{out,err}`.
