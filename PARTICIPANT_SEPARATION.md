# What the professor should check for participant leakage

## Which code ran on Linux?

`source/` is the scientific source archived with the original run. Its 49 file hashes match the archived run checksums. The original trained ensemble was replayed on Linux and reproduced all four saved probability files exactly. `readable/` contains formatted and commented copies with identical Python syntax trees. `reproduce.py`, `verify_review.py`, the audit tools and the walkthroughs were added later for review; they are not presented as historical training code.

## Participant membership result

`reference test` and `test` name the **same partition** in RUN-006, not two independent held-out groups.

| Declared partition | Annotated participants |
|---|---:|
| Training | 1,279 |
| Validation | 274 |
| Reference/test | 275 |

All three pairwise ID intersections are **zero**. The 274 validation participants split into 137 calibration and 137 selection participants, with zero overlap. Common usable participants are a different count: 1,256 training, 273 validation and 271 test. Expanded reference predictions include all 275 test participants.

The [saved audit](participant_split_audit.json) was computed from the archived private membership manifests, all 26 candidates' validation outputs, and all twelve track/noise prediction CSVs. The output publishes counts and file hashes, not individual participant memberships. The code also canonicalizes numeric IDs so `0001` and `1` cannot evade overlap checks.

## Read these implementations

| Concern | Implementation to inspect |
|---|---|
| Fail if train/validation/test IDs overlap | [`common.py`](readable/bcg_full_training/common.py), `verify_prepared` |
| Load only the requested partition | [`common.py`](readable/bcg_full_training/common.py), `matrix`; [`sequences.py`](readable/bcg_full_training/sequences.py), `SequenceDataset.__init__` |
| Neural fitting uses `train`; validation uses `validation` | [`train_balanced_deep.py`](readable/bcg_balanced_staging/train_balanced_deep.py), `train` |
| Classical fitting matrix comes from `train` | [`train_balanced_classic.py`](readable/bcg_balanced_staging/train_balanced_classic.py), `train` |
| Base population scaler uses training records only | [`prepare.py`](readable/bcg_full_training/prepare.py), loop over `ids['train']` |
| Extra-sensor population scaler uses training records only | [`prepare_balanced.py`](readable/bcg_balanced_staging/prepare_balanced.py), loop over `ids['train']` |
| Calibration and selection are separate validation roles | [`balanced_utils.py`](readable/bcg_balanced_staging/balanced_utils.py), `subset_labels`, `choose_beta_across_levels` |
| Lock winners before reference inference | [`finalize_balanced.py`](readable/bcg_balanced_staging/finalize_balanced.py), selection lock before `predict(..., 'test', ...)` |

Centered context and whole-night normalization stay within each recording. Their use of future unlabeled samples makes this an offline method; it does not itself mix participants. Population-scaler code uses training records. Actual scaler values have not been independently recomputed in this new audit.

## Recompute the audit with private evidence

The professor needs authorized access to the original membership records and saved outputs. The public repository intentionally does not publish those identities or per-epoch reference labels. The previously prepared review ZIP contains these manifests and prediction files.

From the cloned repository, with `EVIDENCE` pointing to the extracted review ZIP:

```bash
EVIDENCE=/path/to/professor_code_review_RUN006
python3 audit_participant_split.py \
  --base-manifest "$EVIDENCE/base_preparation_manifest.json" \
  --roles-manifest "$EVIDENCE/manifest.json" \
  --predictions-dir "$EVIDENCE/tracks" \
  --validation-results "$EVIDENCE/validation_results.json" \
  --output /tmp/rechecked-participant-split.json
```

Only Python's standard library is required. Failure raises an error. Passing verifies declared membership separation, validation-output role membership, and that every saved test-output identity belongs to the declared test partition, with no training/validation IDs. It does not prove original optimizer access from a historical execution trace, nor establish that distinct IDs always represent distinct real people without the dataset's identity mapping.

## Important qualification: test inspection

Zero participant overlap and an untouched final test are different requirements. This project inspected reference/test results during earlier development. Consequently, reference performance may have influenced research choices even though the recorded training partitions are disjoint.

If the professor's condition is **no held-out participant IDs in fitting**, the declared memberships and loading code support it, subject to reviewing the private artifacts and execution provenance. If the condition is **an entirely untouched test set that influenced no development**, this experiment does **not** meet that stronger condition. An independent cohort or appropriately untouched new test is needed for confirmation. Replaying checkpoints cannot repair earlier test inspection.
