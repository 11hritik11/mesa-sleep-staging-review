# Presentation claims and evidence

This map covers the supplied screenshot's visible slides 1–9 and the saved meeting guide's proposed slides 10–11. The full Keynote deck and hidden speaker notes were not inspected.

| Slide | Claim and result | Code and evidence |
|---|---|---|
| 1 | Mean five-stage balanced accuracy: 71.15% | `aggregate_results.json`, track `multi`; `verify_review.py` averages the four common-condition balanced accuracies. |
| 2 | Beat-only 64.19%, ECG-derived breathing 63.95%, multiple sensors 71.15% | All three tracks in `aggregate_results.json`. Multi minus beat is 6.96 percentage points; winning model families differ. |
| 3 | Common usable train 1,256 people/877,099 epochs; validation 273/186,559; reference 271/194,310 | Original protocol and preparation code; individual membership records remain in the separate review ZIP. Public metrics verify reference epoch totals only. |
| 4 | Signals/labels → features → candidate models → selection → locked evaluation | `prepare_balanced.py`, `train_balanced_deep.py`, `train_balanced_classic.py`, `finalize_balanced.py`; 24 base candidates plus two ensembles. Detailed histories/selection evidence are in the separate review ZIP. |
| 5 | Equal probability average of BiGRU, temporal CNN and LightGBM | `balanced_models.py`, LightGBM branch in `train_balanced_classic.py`, `Predictor.predict` in `balanced_inference.py`, `selection_summary.json`. |
| 6 | Balanced accuracy 71.34/71.27/71.18/70.80% at 0/20/40/60 ms | `aggregate_results.json` → `multi` → each condition → `common`. Only RR timing is perturbed; other sensor signals remain clean. |
| 7 | N2 has lowest clean recall: 42.62% | `aggregate_results.json` → `multi` → `0` → `common` → `per_stage`. |
| 8 | Clean balanced accuracy 71.34%; accuracy 64.49%; macro-F1 0.6246. N1 recall 67.32%, precision 28.45% | Same clean aggregate report; the checker independently recalculates stage precision, recall and F1 from its confusion matrix. |
| 9 | Three work phases | Preparation, training and verification are documented. Literal Week 1/2/3 dates are unverified: use Phase 1/2/3 unless independently documented. |
| 10–11 | Controlled follow-up and conclusion | Matched sensor-addition retraining is a proposal, not a completed result. Independent confirmation requires new independent data. |

All code paths in the table are under `source/bcg_balanced_staging/` unless otherwise specified. Original feature extraction dependencies are also archived in `source/bcg_full_training/` and `source/bcg_jitter_staging/`.

## What a reviewer can establish

The public repository supports review of the archived scientific implementation and recomputation of metrics from aggregate matrices. The source hashes match the completed run's saved hash list. A dated server check matched the 12 prediction files and five result/source records with the locally reviewed copies; the hashes are recorded in `provenance.json`.

Individual saved prediction rows can be checked with the separate full review ZIP. Full retraining requires the original authorized data and numerical environment. Neither aggregate checks nor matching hashes establish independent clinical validity.
