"""Audit private run manifests and prediction membership; standard library only.

The public report contains counts and artifact hashes, never participant IDs.
Disjoint IDs do not prove that checkpoints followed the declared manifests or
that the test set was untouched during earlier research decisions.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def person(value):
    require(
        isinstance(value, str) and value.isascii() and value.isdigit(),
        "Participant IDs must be digit strings",
    )
    return str(int(value))


def members(values):
    require(isinstance(values, list), "Membership must be a list")
    ids = [person(v) for v in values]
    require(
        len(ids) == len(set(ids)), "Duplicate or differently padded participant IDs"
    )
    return set(ids)


def audit(base_path, roles_path, predictions=None, validation_path=None):
    base = json.loads(base_path.read_text())
    roles_manifest = json.loads(roles_path.read_text())
    groups = {
        k: members(base["participants"][k]) for k in ("train", "validation", "test")
    }
    require(all(groups.values()), "Empty train/validation/test partition")
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[a + "_vs_" + b] = len(groups[a] & groups[b])
    require(
        not any(overlaps.values()), "Participant overlap across train/validation/test"
    )
    role_ids = {
        k: members(roles_manifest["validation_participants"][k])
        for k in ("calibration", "selection")
    }
    require(all(role_ids.values()), "Empty calibration or selection role")
    require(
        not role_ids["calibration"] & role_ids["selection"], "Validation role overlap"
    )
    require(
        role_ids["calibration"] | role_ids["selection"] == groups["validation"],
        "Validation roles do not exactly partition validation participants",
    )
    report = {
        "status": "PASS",
        "partition_participants": {k: len(v) for k, v in groups.items()},
        "pairwise_overlaps": overlaps,
        "validation_role_participants": {k: len(v) for k, v in role_ids.items()},
        "calibration_selection_overlap": 0,
        "artifact_sha256": {
            "base_manifest": digest(base_path),
            "roles_manifest": digest(roles_path),
        },
        "prediction_files": {},
        "validation_candidates_checked": None,
        "scope": "Declared participant identities, role memberships and optional saved-output membership",
        "not_established": [
            "Historical optimizer access to data",
            "Untouched test-set independence",
            "Distinct real people behind different source IDs",
            "Training-only scaler recomputation",
        ],
    }
    if validation_path:
        validation = json.loads(validation_path.read_text())
        for candidate in validation.values():
            for condition in candidate["validation"].values():
                for role in ("calibration", "selection"):
                    actual = {person(v) for v in condition[role]["per_participant"]}
                    require(
                        actual <= role_ids[role],
                        "Candidate validation output has a participant outside its role",
                    )
                actual = {
                    person(v) for v in condition["full_validation"]["per_participant"]
                }
                require(
                    actual <= groups["validation"],
                    "Full validation output has a non-validation participant",
                )
        report["validation_candidates_checked"] = len(validation)
        report["artifact_sha256"]["validation_results"] = digest(validation_path)
    if predictions:
        for track in ("beat", "edr", "multi"):
            for level in (0, 20, 40, 60):
                path = predictions / track / f"test_predictions_{level}ms.csv"
                seen, people, common_people = set(), set(), set()
                common_count = 0
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        pid, epoch = person(row["participant"]), int(row["epoch"])
                        require(
                            pid in groups["test"],
                            "Prediction participant is outside the test partition",
                        )
                        require(
                            pid not in groups["train"] | groups["validation"],
                            "Prediction participant was assigned to development",
                        )
                        require(
                            epoch >= 0 and (pid, epoch) not in seen,
                            "Invalid or repeated test epoch",
                        )
                        require(row["common"] in ("0", "1"), "Invalid common mask")
                        seen.add((pid, epoch))
                        people.add(pid)
                        if row["common"] == "1":
                            common_count += 1
                            common_people.add(pid)
                require(
                    len(seen) == base["counts"]["test"]["expanded"],
                    "Expanded test epoch coverage differs",
                )
                require(
                    common_count == base["counts"]["test"]["common"],
                    "Common test epoch coverage differs",
                )
                key = track + "/" + path.name
                report["prediction_files"][key] = {
                    "expanded_epochs": len(seen),
                    "expanded_participants": len(people),
                    "common_epochs": common_count,
                    "common_participants": len(common_people),
                    "participants_outside_test": 0,
                    "sha256": digest(path),
                }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument(
        "--roles-manifest",
        required=True,
        type=Path,
        help="RUN-006 manifest.json or prepared/balanced_manifest.json",
    )
    parser.add_argument(
        "--predictions-dir", type=Path, help="Directory containing beat/, edr/, multi/"
    )
    parser.add_argument("--validation-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.base_manifest,
        args.roles_manifest,
        args.predictions_dir,
        args.validation_results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
