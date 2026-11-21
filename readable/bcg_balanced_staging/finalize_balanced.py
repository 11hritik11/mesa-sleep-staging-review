#!/usr/bin/env python3
"""Validation-only per-modality selection and one exploratory reference evaluation."""
import argparse, csv
from pathlib import Path
import numpy as np, torch
from balanced_utils import (
    cfg,
    base,
    read_json,
    save,
    fingerprint,
    snapshot,
    feature_names,
    choose_beta_across_levels,
    probability_transform,
    report,
    subset_labels,
    CLASSIC_SPECS,
    DEEP_SPECS,
)
from balanced_inference import Predictor


def validate_prediction(path, expected=None):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    ids = [(r["participant"], int(r["epoch"])) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated participant/epoch")
    p = np.array(
        [[float(r["p_" + s]) for s in ["Wake", "N1", "N2", "N3", "REM"]] for r in rows]
    )
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1, atol=2e-5)
    ):
        raise ValueError("Invalid probabilities")
    y = np.array([int(r["reference"]) for r in rows])
    pred = np.array([int(r["prediction"]) for r in rows])
    pids = np.array([r["participant"] for r in rows])
    if not np.array_equal(pred, p.argmax(1)):
        raise ValueError("Probability prediction mismatch")
    if expected is not None and (
        set(ids) != set(expected)
        or any(int(r["reference"]) != expected[k] for r, k in zip(rows, ids))
    ):
        raise ValueError("Prediction target identity/label mismatch")
    common = np.array([int(r["common"]) == 1 for r in rows])
    return {
        "expanded": report(y, pred, pids),
        "common": report(y[common], pred[common], pids[common]),
    }


def feature_groups(modality):
    names = feature_names(modality)
    groups = {}
    for i, name in enumerate(names):
        raw = name.removeprefix("night_relative_")
        if raw.startswith("edr_"):
            g = "ECG-derived respiration"
        elif raw.startswith(("thorax_", "abdomen_", "airflow_")):
            g = "Respiratory sensors"
        elif raw.startswith("spo2_"):
            g = "Oxygenation"
        elif raw.startswith("leg_"):
            g = "Leg movement"
        elif raw.startswith("local_"):
            g = "Short-window HRV"
        elif raw.startswith("context_"):
            g = "HRV temporal summaries"
        elif name.startswith("night_relative_"):
            g = "Other HRV and quality"
        else:
            g = "Other HRV and quality"
        groups.setdefault(g, []).append(i)
    return groups


def run(args):
    torch.set_num_threads(args.threads)
    version = fingerprint()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    prepared = read_json(args.prepared / "balanced_manifest.json")
    if prepared["source_version"] != version or prepared["status"] != "COMPLETE":
        raise ValueError("Changed prepared provenance")
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")[
        "source"
    ] != version:
        raise ValueError("Changed output source")
    save(
        out / "manifest.json",
        {
            "source": version,
            "config": cfg(),
            "feature_names": {m: feature_names(m) for m in ("beat", "edr", "multi")},
            "validation_participants": prepared["validation_participants"],
            "test_policy": "Previously inspected reference test: exploratory only",
        },
    )
    snapshot(out)
    results = {}
    for folder in (args.classic, args.deep, args.controls):
        for p in folder.glob("*/result.json"):
            value = read_json(p)
            if not value["completed"] or value["source"] != version:
                raise ValueError("Incomplete or changed candidate")
            results[value["name"]] = value
    expected = len(CLASSIC_SPECS) + len(DEEP_SPECS) + 2
    if len(results) != expected:
        raise ValueError(
            f"Expected {expected} completed base candidates, got {len(results)}"
        )
    # Equal-weight ensembles are prespecified extra candidates, using top three selection-ranked members per track.
    for modality in ("beat", "edr", "multi"):
        eligible = sorted(
            [
                n
                for n, v in results.items()
                if v["specification"]["modality"] == modality
            ],
            key=lambda n: (-results[n]["selection_score"], n),
        )
        if len(eligible) < 2:
            continue
        members = eligible[:3]
        evidence = {}
        identities = {}
        for level in (0, 20, 40, 60):
            probabilities = []
            reference = None
            for member in members:
                value = results[member]
                with np.load(
                    Path(value["output"]) / f"validation_logits_{level}ms.npz"
                ) as f:
                    scores, y, pids, epochs = (
                        f["scores"],
                        f["y"],
                        f["pids"],
                        f["epochs"],
                    )
                if reference is not None and not all(
                    np.array_equal(a, b) for a, b in zip(reference, (y, pids, epochs))
                ):
                    raise ValueError("Validation ensemble identities differ")
                reference = y, pids, epochs
                probabilities.append(
                    probability_transform(
                        scores, np.array(value["prior"]), value["beta"]
                    )
                )
            evidence[level] = (
                np.log(np.maximum(np.mean(probabilities, axis=0), 1e-12)),
                y,
                pids,
            )
        prior = np.array(results[members[0]]["prior"])
        beta, reports, score = choose_beta_across_levels(evidence, args.prepared, prior)
        name = "ensemble_" + modality
        results[name] = {
            "name": name,
            "kind": "ensemble",
            "specification": {"modality": modality},
            "members": members,
            "prior": prior.tolist(),
            "beta": beta,
            "selection_score": score,
            "validation": reports,
            "source": version,
            "completed": True,
            "history": [],
        }
    save(out / "validation_results.json", results)
    selection = {}
    for modality in ("beat", "edr", "multi"):
        names = [
            n for n, v in results.items() if v["specification"]["modality"] == modality
        ]
        name = sorted(names, key=lambda n: (-results[n]["selection_score"], n))[0]
        selection[modality] = {
            "winner": name,
            "validation_balanced_accuracy": results[name]["selection_score"],
            "target_70_met_validation": results[name]["selection_score"] >= 0.7,
        }
    lock = {
        "tracks": selection,
        "criterion": cfg()["selection"],
        "locked_before_test": True,
        "source": version,
        "test_policy": "Previously inspected reference partition; exploratory. No test-driven fitting or model selection.",
    }
    if (out / "selection.json").exists() and read_json(out / "selection.json") != lock:
        raise ValueError("Selection changed after lock")
    save(out / "selection.json", lock)
    predictor = Predictor(args.prepared, results, out / "inference_cache")
    expected_labels = {}
    common_ids = set()
    manifest = read_json(base() / "manifest.json")
    for pid in manifest["participants"]["test"]:
        with np.load(base() / "records" / f"test_{pid}.npz") as f:
            expected_labels.update(
                {(pid, int(i)): int(f["y"][i]) for i in np.flatnonzero(f["expanded"])}
            )
            common_ids.update((pid, int(i)) for i in np.flatnonzero(f["common"]))
    test = {}
    importance = {}
    for modality, selection_info in selection.items():
        name = selection_info["winner"]
        track = out / "tracks" / modality
        track.mkdir(parents=True, exist_ok=True)
        test[modality] = {}
        for level in (0, 20, 40, 60):
            path = track / f"test_predictions_{level}ms.csv"
            if not path.exists():
                p, y, pids, epochs = predictor.predict(
                    name, "test", level, expanded=True
                )
                temporary = path.with_suffix(".incomplete.csv")
                with temporary.open("w") as f:
                    w = csv.writer(f)
                    w.writerow(
                        ["participant", "epoch", "reference", "prediction", "common"]
                        + ["p_" + s for s in ["Wake", "N1", "N2", "N3", "REM"]]
                    )
                    for prob, ref, pid, epoch in zip(p, y, pids, epochs):
                        w.writerow(
                            [
                                pid,
                                int(epoch),
                                int(ref),
                                int(prob.argmax()),
                                int((pid, int(epoch)) in common_ids),
                                *map(float, prob),
                            ]
                        )
                temporary.replace(path)
            test[modality][str(level)] = validate_prediction(path, expected_labels)
        save(track / "test_results.json", test[modality])
        save(out / "test_results.json", test)
        # Masked-group sensitivity on selection validation only at 40ms; no test attribution.
        importance_path = track / "importance_validation.json"
        if importance_path.exists():
            importance[modality] = read_json(importance_path)
        else:
            baseline = results[name]["validation"]["40"]["selection"][
                "balanced_accuracy"
            ]
            values = []
            groups = feature_groups(modality)
            if results[name]["kind"] != "classic":
                groups["RR stream"] = []
            for group, columns in groups.items():
                p, y, pids, epochs = predictor.predict(
                    name,
                    "validation",
                    40,
                    drop_features=tuple(columns),
                    drop_rr=group == "RR stream",
                )
                take = subset_labels(pids, args.prepared) == "selection"
                score = report(y[take], p.argmax(1)[take], pids[take])[
                    "balanced_accuracy"
                ]
                values.append(
                    {
                        "group": group,
                        "balanced_accuracy_drop": baseline - score,
                        "masked_score": score,
                    }
                )
            values.sort(key=lambda r: -r["balanced_accuracy_drop"])
            importance[modality] = values
            save(importance_path, values)
    verification = {
        "status": "PASS",
        "base_candidates": expected,
        "total_candidates_including_ensembles": len(results),
        "unique_predictions": True,
        "exact_expanded_identities": True,
        "expanded_epochs": len(expected_labels),
        "common_epochs": len(common_ids),
        "valid_probabilities": True,
        "selection_locked": True,
        "test_evaluation": "Only prespecified validation winners, one per modality; saved test inference caches reused on recovery",
        "target_mean_balanced_accuracy": 0.7,
        "achieved_reference_common": {
            m: float(
                np.mean([r["common"]["balanced_accuracy"] for r in scores.values()])
            )
            for m, scores in test.items()
        },
    }
    save(out / "verification.json", verification)
    from balanced_report import build_report

    build_report(out, args.prepared)
    save(
        out / "COMPLETE.json",
        {"status": "COMPLETE", "tracks": selection, "source": version},
    )
    print("BALANCED COMPARISON COMPLETE", selection, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--classic", type=Path, required=True)
    p.add_argument("--deep", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
