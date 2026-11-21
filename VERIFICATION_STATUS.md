# Verification status

Updated November 21, 2026. Engineering fixtures are artificial; their scores are not scientific results.

| Check | Status |
|---|---|
| Archived source hashes and aggregate metric arithmetic | PASS: 49 archived source files; mean multi-sensor balanced accuracy 0.7114948599784747 |
| Readable copies preserve original Python syntax trees | PASS: 29 files |
| Locked environment and numerical imports on macOS ARM, Python 3.11.18 | PASS |
| Original all-family fixture suite on macOS | PASS: 12 classical fits, 10 one-pass neural fits, architecture/gradient/coverage/calibration/serialization/report checks |
| Uncached checkpoint replay integration with corruption rejection | PASS on an artificial BiGRU/TCN/LightGBM ensemble |
| Original server's numerical package versions | Recovered: the twelve main versions match the pinned recovery environment; server Python is 3.9.16 |
| Clean GitHub Linux environment and full engineering suite | PASS: [run 44876448611](https://github.com/11hritik11/mesa-sleep-staging-review/actions/runs/44876448611), commit 2d35101 |
| Original Linux server engineering suite | PASS in Slurm job 1604043 |
| Original multiple-sensor checkpoint replay | **PASS** in Slurm job 1604043: all four timing conditions; 1,126,848 expanded epoch-condition predictions; zero class disagreements and zero probability differences |
| Fresh full-cohort fitting of all 22 new candidates | Commands supplied; not rerun in this update |
| Complete raw EDF/XML-to-result reconstruction | Not completed; requires upstream private artifacts and preparation dependencies |

See GitHub Actions for independently executed Linux engineering checks. A passing fixture or aggregate check does not establish reproduction of the clinical result from model weights. [Checkpoint replay evidence](checkpoint_replay_evidence.json) records original model hashes, per-condition aggregate metrics and comparisons. Recomputed mean common balanced accuracy is **0.7114948599784747 (71.15% rounded)**. This is inference replay from original trained weights; full fresh training has not been rerun.
