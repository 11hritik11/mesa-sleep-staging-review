"""Require the split auditor to reject real leakage patterns on artificial IDs."""

import json
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_participant_split import audit


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        base, roles = root / "base.json", root / "roles.json"
        original = {
            "participants": {
                "train": ["0001"],
                "validation": ["0002", "0003"],
                "test": ["0004"],
            }
        }
        valid_roles = {
            "validation_participants": {"calibration": ["0002"], "selection": ["0003"]}
        }
        base.write_text(json.dumps(original))
        roles.write_text(json.dumps(valid_roles))
        assert audit(base, roles)["status"] == "PASS"
        cases = []
        repeated_test = json.loads(json.dumps(original))
        repeated_test["participants"]["train"].append("0004")
        cases.append((repeated_test, valid_roles))
        padded_alias = json.loads(json.dumps(original))
        padded_alias["participants"]["test"] = ["1"]
        cases.append((padded_alias, valid_roles))
        repeated_role = {
            "validation_participants": {
                "calibration": ["0002", "0003"],
                "selection": ["0003"],
            }
        }
        cases.append((original, repeated_role))
        for manifest, role in cases:
            base.write_text(json.dumps(manifest))
            roles.write_text(json.dumps(role))
            try:
                audit(base, roles)
            except ValueError:
                pass
            else:
                raise AssertionError("Auditor accepted overlapping people/roles")
        original["counts"] = {"test": {"common": 1, "expanded": 1}}
        base.write_text(json.dumps(original))
        roles.write_text(json.dumps(valid_roles))
        for track in ("beat", "edr", "multi"):
            (root / track).mkdir()
            for level in (0, 20, 40, 60):
                with (root / track / f"test_predictions_{level}ms.csv").open(
                    "w"
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["participant", "epoch", "common"])
                    writer.writerow(["0004", 0, 1])
        assert len(audit(base, roles, root)["prediction_files"]) == 12
        (root / "beat/test_predictions_0ms.csv").write_text(
            "participant,epoch,common\n0001,0,1\n"
        )
        try:
            audit(base, roles, root)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Auditor accepted a training person in test predictions"
            )
        print(
            "PASS: rejects train/test overlap, padded-ID aliases, and calibration/selection overlap"
        )
        print(
            "PASS: checks all 12 prediction files and rejects training people in test outputs"
        )


if __name__ == "__main__":
    main()
