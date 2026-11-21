# BCG peak jitter and MESA sleep staging

## Active expansion: annotated beats and five model families

`benchmark.py` uses NSRR MESA R-point/normal-beat annotations for the 1,828 prepared participants with published annotations (79 of 1,907 lack these files). It extracts 54 features and compares XGBoost, histogram gradient boosting, Random Forest, Extra Trees and MLP, each trained with clean or jitter-augmented inputs. Validation chooses the winning candidate before it evaluates test participants. See [FEATURES.md](FEATURES.md) for every feature, formula, window, quality rule, model setting and literature source. The SVM pilot below remains preserved.

Save the NSRR token privately at `~/.config/nsrr/token` (permissions 600). The downloader verifies file size and MD5 against the official NSRR listing. Run on the cluster:

```bash
download_job=$(sbatch --parsable bcg_jitter_staging/run_download.slurm)
sbatch --dependency=afterok:$download_job bcg_jitter_staging/run_benchmark.slurm
```

New outputs: `data/derived/bcg_benchmark_runs/run_<JOB_ID>/`. Shared feature caches: `data/derived/bcg_hrv_features_v2/<SOURCE_HASH>/`. Source files, cohort provenance, coverage, validation comparisons, the selection decision, validation feature importance, test reports and predictions are saved per run. Fitting uses a documented epoch cap per participant; test evaluation uses all usable test epochs. The benchmark requires every file in the published annotation inventory to download successfully; unavailable recordings are explicitly excluded using the official listing.

## Original SVM pilot

Paper: Ahmad Suliman et al., *Effects of Ballistocardiogram Peak Detection Jitters on the Quality of Heart Rate Variability Features: A Simulation-Based Case Study in the Context of Sleep Staging*, Sensors 2023, DOI [10.3390/s23052693](https://doi.org/10.3390/s23052693).

The paper perturbs ECG R-peak timestamps with independent zero-mean Gaussian offsets. The resulting adjacent heartbeat intervals approximate timing errors from BCG peak detection. It measures interval error up to about 60 ms, computes five-minute HRV features every 30 seconds (mean interval, SDNN, LF, HF, LF/HF, median-filtered LF/HF), and uses an RBF SVM for five-stage and three-stage sleep classification. Its 50 NSRR participants and 10-fold cross-validation do not identify an exact MESA subset. The paper's 17% to 25% error example is for its three-class scenario and is not a target for this code.

This is a **paper-inspired MESA experiment**, not an exact reproduction: a deterministic 50-person subset of the already prepared MESA cohort is chosen from the fixed participant split (34 train, 8 validation, 8 test); ECG peaks are detected heuristically; four jitter levels (0, 20, 40, 60 ms target interval MAE) are used; and centered five-minute windows yield the six paper features. The code records realized interval MAE. It omits the paper's label smoothing, Bayes-error analysis, and 98-level sweep. The ECG does not contain real BCG, so this tests sensitivity to simulated timing error only.

The improvement is a second SVM trained on a mixture of clean and jittered HRV features. Both models use the same 12,000-example training budget and are evaluated on identical held-out participants at each error level. `results.json` includes epoch accuracy, macro-F1, kappa, confusion matrices, and mean participant macro-F1. The claim of improvement should be judged on held-out test results, especially 40 and 60 ms, after the job completes. Do not tune against test results.

Run on the cluster from `/home/rdb200/nsrr-setup`:

```bash
sbatch bcg_jitter_staging/run.slurm
```

Outputs are under `data/derived/bcg_jitter_runs/run_<JOB_ID>/`; Slurm logs are under `logs/`. The peak detector is a heuristic and should be validated against annotated R peaks before any physiological claim. The source split and data never leave the cluster.
