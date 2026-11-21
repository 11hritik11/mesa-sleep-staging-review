"""Integration test of uncached checkpoint replay and mismatch detection.

All records and weights here are artificial. This is not a MESA result.
The original all-family smoke suite separately exercises actual fitting.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce


def main():
    with tempfile.TemporaryDirectory(prefix="mesa-replay-test-") as temp:
        root = Path(temp)
        base, balanced = root / "base", root / "balanced"
        shared = reproduce.activate(base)
        import joblib
        import numpy as np
        import torch
        from lightgbm import LGBMClassifier
        from tests import fixture
        from balanced_inference import Predictor
        from balanced_models import BalancedNetwork

        torch.set_num_threads(1)
        torch.manual_seed(73)
        settings = fixture(base / "prepared")
        prepared = balanced / "prepared"
        (prepared / "extra").mkdir(parents=True)
        rng = np.random.default_rng(73)
        for part, pid in (("train", "0001"), ("validation", "0002"), ("test", "0003")):
            np.savez(
                prepared / "extra" / f"{part}_{pid}.npz",
                x=rng.normal(size=(130, len(shared.EXTRA_NAMES))).astype("float32"),
            )
        np.savez(
            prepared / "extra_scaler.npz",
            center=np.zeros(len(shared.EXTRA_NAMES)),
            scale=np.ones(len(shared.EXTRA_NAMES)),
        )
        reproduce.save(
            prepared / "balanced_manifest.json",
            {
                "status": "COMPLETE",
                "source_version": shared.fingerprint(),
                "base_feature_version": "fixture",
                "validation_participants": {"calibration": ["0002"], "selection": []},
            },
        )
        results = {}
        members = ["bigru_multisensor", "tcn_multisensor", "lightgbm_multi"]
        width = len(shared.feature_names("multi"))
        for name, architecture in zip(members[:2], ("gru", "tcn")):
            output = balanced / "deep" / name
            output.mkdir(parents=True)
            model = BalancedNetwork(width, architecture, settings)
            torch.save({"model": model.state_dict()}, output / "best.pt")
            results[name] = {
                "kind": "deep",
                "source": shared.fingerprint(),
                "specification": {"modality": "multi", "architecture": architecture},
                "output": str(output),
                "prior": [0.2] * 5,
                "beta": 0,
            }
        output = balanced / "classic/lightgbm_multi"
        output.mkdir(parents=True)
        x = rng.normal(size=(100, width))
        tree = LGBMClassifier(
            n_estimators=3,
            num_leaves=7,
            min_child_samples=2,
            n_jobs=1,
            verbosity=-1,
            random_state=73,
        ).fit(x, np.arange(100) % 5)
        joblib.dump(tree, output / "model.joblib")
        results[members[2]] = {
            "kind": "classic",
            "source": shared.fingerprint(),
            "specification": {"modality": "multi"},
            "output": str(output),
            "prior": [0.2] * 5,
            "beta": 0,
        }
        results["ensemble_multi"] = {
            "kind": "ensemble",
            "source": shared.fingerprint(),
            "members": members,
            "prior": [0.2] * 5,
            "beta": 0,
        }
        folder = balanced / "results"
        (folder / "tracks/multi").mkdir(parents=True)
        # Deliberately use stale historical paths to exercise relocation.
        historical = json.loads(json.dumps(results))
        for value in historical.values():
            if "output" in value:
                value["output"] = "/old/unavailable/cluster/location"
        reproduce.save(folder / "validation_results.json", historical)
        reproduce.save(
            folder / "selection.json",
            {
                "source": shared.fingerprint(),
                "tracks": {"multi": {"winner": "ensemble_multi"}},
            },
        )
        predictor = Predictor(prepared, results, cache=None)
        probability, y, people, epochs = predictor.predict(
            "ensemble_multi", "test", 0, expanded=True
        )
        with np.load(base / "prepared/records/test_0003.npz") as record:
            common = record["common"]
        path = folder / "tracks/multi/test_predictions_0ms.csv"
        header = ["participant", "epoch", "reference", "prediction", "common"]
        header += ["p_" + stage for stage in shared.STAGES]
        with path.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for prob, truth, person, epoch in zip(probability, y, people, epochs):
                writer.writerow(
                    [person, epoch, truth, prob.argmax(), int(common[epoch]), *prob]
                )
        command = [
            sys.executable,
            str(reproduce.ROOT / "reproduce.py"),
            "replay",
            "--base-run",
            str(base),
            "--balanced-run",
            str(balanced),
            "--output",
            str(root / "replay"),
            "--levels",
            "0",
            "--threads",
            "1",
        ]
        passed = subprocess.run(command, text=True, capture_output=True)
        if passed.returncode:
            raise RuntimeError(passed.stderr)
        require = reproduce.require
        require(
            reproduce.load(root / "replay/replay_results.json")["status"] == "PASS",
            "Clean replay failed",
        )
        rows = list(csv.reader(path.open()))
        rows[1][5] = str(float(rows[1][5]) + 0.01)
        with path.open("w") as stream:
            csv.writer(stream).writerows(rows)
        failed = subprocess.run(command, text=True, capture_output=True)
        require(
            failed.returncode != 0
            and reproduce.load(root / "replay/replay_results.json")["status"]
            == "MISMATCH",
            "Corrupted saved probabilities were not rejected",
        )
        print(
            "PASS: real checkpoint loading, path relocation, uncached inference, coverage and mismatch rejection"
        )
        print("Synthetic engineering fixture only; no clinical performance estimate")


if __name__ == "__main__":
    main()
