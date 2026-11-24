# Reproducing RUN-006

There are three different checks. Aggregate arithmetic checks need no individual data. Checkpoint replay requires the original authorized prepared features and trained weights. Fresh fitting additionally takes the original training time and does not guarantee identical floating-point results.

## 1. Install the pinned recovery environment

Install `uv`, then run from the cloned repository:

```bash
# macOS: the XGBoost/LightGBM runtime requires OpenMP.
brew install libomp

# Linux instead: install the distribution's OpenMP runtime, e.g. libgomp1.
uv sync --frozen
uv run --frozen python reproduce.py doctor
uv run --frozen python tools/check_readable.py
uv run --frozen python verify_review.py
```

Python is pinned to 3.11.18. `uv.lock` pins the Python packages and distribution hashes, with a CPU PyTorch index on Linux. The twelve main package versions were checked against the surviving server environment. The original server uses Python 3.9.16, so this is a newly pinned recovery environment, not a claim that the entire historical operating system was reconstructed. System OpenMP libraries are prerequisites outside the Python lock.

The original CPU training scripts are preserved without changes in `source/`. `readable/` contains formatted review copies with explanatory comments above the main functions/classes and at participant-separation, normalization and fitting checks. The comments describe inputs, outputs, masks and validation roles without modifying executable statements. `tools/check_readable.py` checks their hashes and proves that all 29 copies have the same Python syntax trees. Reproduction commands execute the archived originals.

## 2. Test the executable pipeline without MESA data

```bash
uv run --frozen python reproduce.py smoke
uv run --frozen python tools/test_replay.py
```

The original smoke suite tests the actual fitting APIs for all twelve classical candidates and ten neural candidates on artificial fixtures, serialization, all five neural architectures, gradients, complete target visits, calibration separation, completed-job skipping and report generation. It reduces the fixture runs to one neural pass. These are engineering tests, not sleep-staging performance estimates.

The replay integration test loads actual serialized BiGRU, TCN and fitted LightGBM models on an artificial dataset. It tests path relocation and uncached ensemble inference, then corrupts a saved probability and requires replay to fail. It does not contain clinical records.

## 3. Obtain authorized experiment artifacts

These files must be supplied separately through approved study access. Public GitHub access does not grant MESA data access.

```text
BASE_RUN/
  prepared/
    manifest.json
    population_scaler.npz
    records/{partition}_{participant}.npz
  temporal/
    tcn_dual_common/best.pt
    tcn_hrv_common/best.pt

BALANCED_RUN/
  prepared/
    balanced_manifest.json
    extra_scaler.npz
    extra/{partition}_{participant}.npz
  deep/{candidate}/best.pt
  classic/{candidate}/model.joblib
  results/
    selection.json
    validation_results.json
    tracks/{track}/test_predictions_{level}ms.csv
```

`BASE_RUN` is the original RUN-005 directory ending `20261102T213108Z`; `BALANCED_RUN` is RUN-006 ending `20261103T061410Z`. These survive under the original server's `/home/rdb200/nsrr-setup/data/derived/bcg_full_training_runs/` and `bcg_balanced_runs/` respectively. Preserve the relative directory layout when transferring authorized artifacts. The runner relocates historical model paths and the base-run setting in memory.

For checkpoint replay only the selected model files are needed. New fitting needs prepared features; the historical weights of the new candidates are not needed. The two RUN-005 temporal checkpoints are needed only to rescore the inherited comparison controls. Final report generation also uses `prepared/audit.json` and the base preparation's `coverage.json`.

The runner verifies source fingerprints, parent feature identity, validation-role separation, scalers and expected record-file presence. It does not re-establish the clinical annotation audit merely from a manifest. Keep the original audits with the data.

## 4. Replay the original multiple-sensor winner

Set these two shell variables to your authorized local paths:

```bash
BASE_RUN=/path/to/bcg_full_training_runs/20261102T213108Z
BALANCED_RUN=/path/to/bcg_balanced_runs/20261103T061410Z

uv run --frozen python reproduce.py inventory \
  --base-run "$BASE_RUN" --balanced-run "$BALANCED_RUN"

uv run --frozen python reproduce.py replay \
  --base-run "$BASE_RUN" --balanced-run "$BALANCED_RUN" \
  --output runs/checkpoint-replay --tracks multi --levels 0 20 40 60 --threads 8
```

This **reruns model inference** from original weights with no inference cache. It compares every expanded target identity, reference label, predicted class and probability with the saved CSVs. Default pass criteria are zero predicted-class disagreements and maximum absolute probability error at most `1e-6`. A mismatch is recorded as `MISMATCH` and exits unsuccessfully. Do not increase tolerances merely to make a mismatch disappear.

Results are saved in `runs/checkpoint-replay/replay_results.json`, including checkpoint hashes and aggregate metrics. `reproduction.json` records effective configuration, manifest hashes, environment and the path adaptation. Original sources, prepared data, model files and results are left untouched. Use a new output directory outside the original runs.

Successful full replay should give multiple-sensor common balanced accuracies approximately 0.7134443, 0.7127013, 0.7118141 and 0.7080198; their mean is 0.7114948599784747. `--tracks beat edr multi` requests replay of all three tracks.

Only load PyTorch/joblib files from the trusted experiment archive: these serialization formats can execute code when loaded.

## 5. Fit the original candidates anew from prepared features

```bash
# 22 new fits: 12 classical + 10 neural, with original full settings/targets.
uv run --frozen python reproduce.py fit \
  --base-run "$BASE_RUN" --balanced-run "$BALANCED_RUN" \
  --output runs/fresh-fit --candidate all --threads 8

# Rescore the two already-trained RUN-005 controls on the RUN-006 roles.
uv run --frozen python reproduce.py fit-controls \
  --base-run "$BASE_RUN" --balanced-run "$BALANCED_RUN" \
  --output runs/fresh-fit --threads 8

# Requires all 24 completed base entries before model selection/evaluation.
uv run --frozen python reproduce.py finalize \
  --base-run "$BASE_RUN" --balanced-run "$BALANCED_RUN" \
  --output runs/fresh-fit --threads 8
```

For scheduling separate jobs, replace `--candidate all` with the exact candidate name from `source/bcg_balanced_staging/balanced_utils.py`, for example `bigru_multisensor`, `tcn_multisensor` or `lightgbm_multi`. Use the same new output directory for all candidates. Per-candidate locks prevent overlapping workers, and the original training routines retain checkpoint resume/completed-job behavior. Invoke `finalize` only after all candidates and both controls finish. Original launch scripts contain the old server paths; use the runner for relocated execution.

These are full CPU training jobs. Original allocations used eight CPUs, 32 GB for neural training, 64 GB for classical training and up to 48 hours per allocation. They are not quick laptop demos. Checkpoint replay is less expensive than fitting all models.

## Boundary of this reproduction

The executable rerun starts at the **audited prepared-feature artifacts**, not at a newly downloaded raw NSRR cohort. Complete raw-to-result reproduction additionally needs the exact private participant split, the upstream ECG preparation and audit scripts, original EDF/XML/R-point files and frozen feature caches. The archived extraction code is present for review, but the upstream raw-data reconstruction has not been made portable or revalidated here.

Prior reference-test inspection makes the result exploratory. Successfully reproducing an existing score does not establish independent clinical validity or measured load-cell BCG performance. Fresh training may produce numerical or checkpoint differences; retain and report them.
