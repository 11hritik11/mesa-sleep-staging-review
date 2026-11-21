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
| Original Linux server engineering suite and real checkpoint replay | Submitted as Slurm job 1604043; result pending |
| Fresh full-cohort fitting of all 22 new candidates | Commands supplied; not rerun in this update |
| Complete raw EDF/XML-to-result reconstruction | Not completed; requires upstream private artifacts and preparation dependencies |

See GitHub Actions for independently executed Linux engineering checks. A passing fixture or aggregate check does not establish reproduction of the clinical result from model weights. Real checkpoint replay status will be reported separately.
