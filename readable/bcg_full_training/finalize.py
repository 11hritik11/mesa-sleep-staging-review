#!/usr/bin/env python3
"""Lock validation selection, infer once, verify saved outputs and render reports."""
import argparse
import csv
import shutil
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.inspection import permutation_importance
from common import (
    config,
    read_json,
    save,
    verify_prepared,
    source_hash,
    snapshot,
    matrix,
    report,
    NAMES,
    LEVELS_MS,
    STAGES,
)
from temporal import TemporalCNN, evaluate
from sequences import SequenceDataset


def prediction_report(path, common_only=False):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if common_only:
        rows = [r for r in rows if int(r.get("common", "1"))]
    ids = [(r["participant"], int(r["epoch"])) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction identities")
    probabilities = np.array([[float(r["p_" + s]) for s in STAGES] for r in rows])
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(1), 1, atol=2e-5)
    ):
        raise ValueError("Invalid saved probabilities")
    y = np.array([int(r["reference"]) for r in rows])
    pred = np.array([int(r["prediction"]) for r in rows])
    if not np.array_equal(probabilities.argmax(1), pred):
        raise ValueError("Probability/prediction mismatch")
    return report(y, pred, np.array([r["participant"] for r in rows])), ids, rows


def write_rows(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def feature_groups():
    # Original features and their nightly copies are ablated together.
    groups = {
        "HR/time": list(range(12)),
        "Nonlinear/distribution": list(range(12, 18)),
        "Spectral": list(range(18, 26)),
        "Quality": list(range(26, 30)),
        "Temporal HRV summaries": list(range(30, 54)),
        "Short-window HRV": list(range(54, 70)),
    }
    return {
        name: columns + [i + 70 for i in columns] for name, columns in groups.items()
    }


def run(args):
    cfg = config()
    manifest = verify_prepared(args.prepared)
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source": source_hash(),
        "features": manifest["feature_version"],
        "reference_test_policy": cfg["reference_test_policy"],
    }
    if (args.output / "manifest.json").exists() and read_json(
        args.output / "manifest.json"
    ) != provenance:
        raise ValueError("Changed finalizer provenance")
    save(args.output / "manifest.json", provenance)
    snapshot(args.output)
    validation = read_json(args.trees / "results.json")
    for variant in ("tcn_hrv_common", "tcn_dual_common", "tcn_dual_expanded"):
        value = read_json(args.temporal / variant / "result.json")
        if not value["completed"]:
            raise ValueError("Incomplete temporal candidate")
        validation[variant] = value
    if len(validation) != 10:
        raise ValueError(
            "Expected six new tree, three temporal and one prior control candidates"
        )
    for name, r in validation.items():
        if not r["completed"]:
            raise ValueError("Incomplete candidate")
        for score in r["validation"].values():
            if score["n_epochs"] != manifest["counts"]["validation"]["common"]:
                raise ValueError("Validation coverage mismatch")
        if not np.isclose(
            r["selection_score"],
            np.mean([r["validation"][str(n)]["macro_f1"] for n in LEVELS_MS]),
        ):
            raise ValueError("Selection score mismatch")
    winner = sorted(validation, key=lambda n: (-validation[n]["selection_score"], n))[0]
    selection = {
        "winner": winner,
        "validation_score": validation[winner]["selection_score"],
        "selection_complete": True,
        "criterion": cfg["selection"],
        "reference_test_policy": cfg["reference_test_policy"],
    }
    if (args.output / "selection.json").exists() and read_json(
        args.output / "selection.json"
    ) != selection:
        raise ValueError("Selection changed after lock")
    save(args.output / "validation_results.json", validation)
    save(args.output / "selection.json", selection)
    save(args.output / "coverage.json", read_json(args.prepared / "coverage.json"))
    save(args.output / "preparation_manifest.json", manifest)
    for name, r in validation.items():
        if r["history"]:
            save(args.output / f"training_history_{name}.json", r["history"])
    temporal = winner.startswith("tcn_")
    control = winner == "previous_capped_rf_control"
    chosen = validation[winner]
    if temporal:
        model = TemporalCNN(cfg, rr_branch=winner != "tcn_hrv_common")
        state = torch.load(
            args.temporal / winner / "best.pt", map_location="cpu", weights_only=False
        )
        if state["fingerprint"]["source"] != source_hash():
            raise ValueError("Temporal source mismatch")
        model.load_state_dict(state["model"])
        model.eval()
    else:
        model = joblib.load(chosen["model_path"])
    test = {}
    expanded = {}
    ids_by_level = {}
    for k, level in enumerate(LEVELS_MS):
        target = args.output / f"test_predictions_{level}ms.csv"
        full = args.output / f"expanded_test_predictions_{level}ms.csv"
        if control:
            if not target.exists():
                shutil.copy2(Path(cfg["source_run"]) / target.name, target)
        elif temporal:
            if not full.exists():
                temp = full.with_suffix(".incomplete.csv")
                evaluate(
                    model,
                    SequenceDataset(args.prepared, "test", "expanded"),
                    cfg,
                    k,
                    predictions_path=temp,
                )
                temp.replace(full)
            expanded[str(level)], _, _ = prediction_report(full)
            common_report, _, rows = prediction_report(full, True)
            write_rows(target, rows)
        elif not target.exists():
            x, y, p, e = matrix(args.prepared, "test", level)
            prob = model.predict_proba(x[:, : chosen["features"]])
            rows = [
                dict(
                    zip(
                        ["participant", "epoch", "reference", "prediction"]
                        + ["p_" + s for s in STAGES],
                        [
                            str(pid),
                            int(epoch),
                            int(ref),
                            int(pr.argmax()),
                            *map(float, pr),
                        ],
                    )
                )
                for pid, epoch, ref, pr in zip(p, e, y, prob)
            ]
            temp = target.with_suffix(".incomplete.csv")
            write_rows(temp, rows)
            temp.replace(target)
        test[str(level)], ids, rows = prediction_report(target)
        ids_by_level[str(level)] = set(ids)
        # Verify exact eligible target identities against prepared masks, using saved predictions only.
        expected = {}
        for pid in manifest["participants"]["test"]:
            with np.load(args.prepared / "records" / f"test_{pid}.npz") as f:
                expected.update(
                    {(pid, int(i)): int(f["y"][i]) for i in np.flatnonzero(f["common"])}
                )
        if set(ids) != set(expected) or any(
            int(r["reference"]) != expected[(r["participant"], int(r["epoch"]))]
            for r in rows
        ):
            raise ValueError("Reference test identities or labels changed")
        save(
            args.output / "test_results.json",
            {
                "five_stage": test,
                "expanded": expanded,
                "policy": cfg["reference_test_policy"],
                "reused_previous_predictions": control,
            },
        )
    if any(v != ids_by_level["0"] for v in ids_by_level.values()):
        raise ValueError("Noise comparison target mismatch")
    importance_path = args.output / "feature_importance_validation.json"
    if not importance_path.exists():
        if temporal:
            base = chosen["validation"]["40"]["macro_f1"]
            importance = []
            for name, columns in list(feature_groups().items()) + [
                ("RR waveform branch", [])
            ]:
                if name == "RR waveform branch" and winner == "tcn_hrv_common":
                    continue
                ds = SequenceDataset(
                    args.prepared,
                    "validation",
                    "expanded",
                    drop_features=columns,
                    drop_rr=name == "RR waveform branch",
                )
                value = evaluate(model, ds, cfg, 2)["common"]["macro_f1"]
                importance.append(
                    {"feature": name, "mean_macro_f1_drop": base - value, "std": 0.0}
                )
            kind = "Validation group masking ablation on all common epochs at 40 ms; not permutation importance"
        else:
            x, y, _, _ = matrix(args.prepared, "validation", 40)
            rng = np.random.default_rng(cfg["seed"])
            ix = rng.choice(len(y), min(4000, len(y)), replace=False)
            width = chosen["features"]
            perm = permutation_importance(
                model,
                x[ix, :width],
                y[ix],
                scoring="f1_macro",
                n_repeats=3,
                n_jobs=1,
                random_state=cfg["seed"],
            )
            importance = [
                {"feature": NAMES[i], "mean_macro_f1_drop": float(a), "std": float(b)}
                for i, (a, b) in enumerate(
                    zip(perm.importances_mean, perm.importances_std)
                )
            ]
            kind = "Validation permutation importance on fixed <=4000-epoch diagnostic subset at 40 ms, three repeats"
        importance.sort(key=lambda r: -r["mean_macro_f1_drop"])
        save(importance_path, importance)
        save(
            args.output / "feature_importance_method.json",
            {"method": kind, "test_used": False},
        )
    save(
        args.output / "verification.json",
        {
            "status": "PASS",
            "unique_predictions": True,
            "exact_common_test_identities": True,
            "probabilities_valid": True,
            "all_common_validation_epochs": True,
            "selection_from_validation_only": True,
            "reference_test_policy": cfg["reference_test_policy"],
            "test_epochs_each_level": len(ids_by_level["0"]),
            "test_confusion_totals": {
                n: int(np.array(r["confusion_matrix"]).sum()) for n, r in test.items()
            },
        },
    )
    from full_report import build_report

    build_report(args.output)
    save(
        args.output / "COMPLETE.json",
        {"winner": winner, "status": "COMPLETE", "source": source_hash()},
    )
    print("FULL-DATA COMPARISON COMPLETE", winner, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--trees", type=Path, required=True)
    p.add_argument("--temporal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
