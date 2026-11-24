# MESA sleep staging: code for professor review

[![Engineering checks](https://github.com/11hritik11/mesa-sleep-staging-review/actions/workflows/reproduction.yml/badge.svg)](https://github.com/11hritik11/mesa-sleep-staging-review/actions/workflows/reproduction.yml)

Archived experiment code and aggregate evidence supporting **71.15% mean five-stage balanced accuracy** in the presentation (RUN-006).

**Original multiple-sensor checkpoint replay passed:** all four timing conditions reproduce the saved probabilities exactly, with zero class disagreements. Mean balanced accuracy is **0.7114948599784747**. See [the evidence](checkpoint_replay_evidence.json) and [verification status](VERIFICATION_STATUS.md). This establishes replay from trained weights, not a completed fresh training rerun.

**Executable reproduction:** [REPRODUCING.md](REPRODUCING.md) documents the locked environment, fitting tests, original-checkpoint replay and fresh training from authorized prepared features. Full raw-data reconstruction and a completed fresh training rerun are not claimed.

**Start with [the model walkthrough](MODEL_WALKTHROUGH.md)** for inputs, model architecture, training, selection and the exact calculation of 71.15%. [Verification status](VERIFICATION_STATUS.md) distinguishes completed checks from pending work.

**Participant leakage review:** [participant separation and the exact files to inspect](PARTICIPANT_SEPARATION.md), with a [saved audit](participant_split_audit.json) and a runnable private-manifest checker. Declared train/validation/reference-test memberships have zero overlap; prior reference-test inspection still makes the study exploratory.

## Clone and check

```bash
git clone https://github.com/11hritik11/mesa-sleep-staging-review.git
cd mesa-sleep-staging-review
python3 verify_review.py
```

Python 3.9 or later; no additional packages needed for this review check. Expected output includes `"status": "PASS"`, 49 verified archived source files, and mean balanced accuracy `0.7114948599784747` for `multi`.

The public check recalculates accuracy, balanced accuracy, macro-F1, kappa and per-stage scores from aggregate confusion matrices. It checks the original source-file hashes. It does **not** rerun training or verify individual prediction rows.

## Code to review

Start with the commented, formatted [readable model definitions](readable/bcg_balanced_staging/balanced_models.py), [neural training](readable/bcg_balanced_staging/train_balanced_deep.py), and [ensemble inference](readable/bcg_balanced_staging/balanced_inference.py). All 29 readable copies have the same Python syntax trees as their archived originals; run `python3 tools/check_readable.py` to check this.

The `source/` directory is copied from the completed experiment archive, rather than rewritten example code. The reproduction runner executes those unchanged originals and records the effective paths/configuration. Newly written runners and review tools are outside that directory.

| Topic | Original implementation |
|---|---|
| Full experimental protocol | [PROTOCOL.md](source/bcg_balanced_staging/PROTOCOL.md) |
| Frozen settings | [balanced_config.json](source/bcg_balanced_staging/balanced_config.json) |
| Signal extraction and preparation | [prepare_balanced.py](source/bcg_balanced_staging/prepare_balanced.py) |
| BiGRU and temporal CNN | [balanced_models.py](source/bcg_balanced_staging/balanced_models.py) |
| Neural training | [train_balanced_deep.py](source/bcg_balanced_staging/train_balanced_deep.py) |
| LightGBM and other classical models | [train_balanced_classic.py](source/bcg_balanced_staging/train_balanced_classic.py) |
| Ensemble probability averaging | [balanced_inference.py](source/bcg_balanced_staging/balanced_inference.py) |
| Selection and final evaluation | [finalize_balanced.py](source/bcg_balanced_staging/finalize_balanced.py) |
| Original feature definitions | [FEATURES.md](source/bcg_jitter_staging/FEATURES.md) |

## Verified results

| Input setting | Mean five-stage balanced accuracy |
|---|---:|
| Beat-derived features | 64.19% |
| Beat features + ECG-derived breathing | 63.95% |
| Multiple sensors | **71.15%** |

The multiple-sensor winner averages probabilities equally from BiGRU, temporal CNN and LightGBM. Each component may have its own fitted prior correction; the ensemble's additional correction is zero. The mean covers four synthetic RR timing-error conditions (0/20/40/60 ms). Common reference evaluation uses 194,310 epochs from 271 participants at each condition.

Multiple-sensor balanced accuracy at those four levels is 71.34%, 71.27%, 71.18%, and 70.80%. In the clean condition, ordinary accuracy is 64.49%, macro-F1 is 0.6246, and N2 recall is 42.62%. Balanced accuracy means average recall across Wake/N1/N2/N3/REM; it differs from ordinary accuracy.

Full aggregate matrices and stage metrics are in [aggregate_results.json](aggregate_results.json). The recorded winning methods are in [selection_summary.json](selection_summary.json). [provenance.json](provenance.json) contains the dated server comparison and file hashes. Hashes establish consistency with the saved records, rather than independent proof of execution.

## Presentation evidence

See [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for the slide-by-slide map.

These are exploratory findings on previously inspected reference data. Additional measured breathing, oxygen and leg-movement inputs remain clean while RR timing is perturbed. The result does not establish measured load-cell BCG performance. Comparing differently selected model families does not isolate the causal contribution of each sensor.

## Full reproduction and separate review evidence

This public repository contains source code and aggregate results. Individual participant lists, prediction probabilities, epoch reference labels, raw signals and trained weights are excluded.

The separately prepared full review ZIP includes the saved prediction files, training histories and `verify_presentation.py`. An authorized reviewer with that ZIP can verify individual predictions by extracting it and running:

```bash
python3 verify_presentation.py
```

Those saved-output checks passed before publication. They do not rerun training or independently compare the saved reference labels with original clinical annotations.

The repository now includes a pinned recovery environment (`pyproject.toml`, `uv.lock`) and an explicit-path runner (`reproduce.py`). Checkpoint replay and fresh fitting start from authorized audited prepared features; see [the exact commands and data contract](REPRODUCING.md). Full raw-to-result reconstruction additionally needs the private split, raw EDF/XML/beat data, upstream ECG preparation and frozen feature caches, and has not been made portable here. No bit-for-bit fresh-training claim is made. The Keynote presentation itself is not included.
