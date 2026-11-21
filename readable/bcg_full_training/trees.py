#!/usr/bin/env python3
"""Full common-cohort forests and boosting, with fixed validation-only comparisons."""
import argparse
import gc
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from common import (
    config,
    verify_prepared,
    matrix,
    class_weights,
    report,
    save,
    read_json,
    source_hash,
    snapshot,
    LEVELS_MS,
)

SPECS = [
    ("rf_raw54", 54, 1.0, 20),
    ("xgb_raw54", 54, 1.0, None),
    ("rf_multiscale140_w05", 140, 0.5, 100),
    ("rf_multiscale140_w10", 140, 1.0, 100),
    ("xgb_multiscale140_w05", 140, 0.5, None),
    ("xgb_multiscale140_w10", 140, 1.0, None),
]


def diagnostic(model, x, y):
    p = model.predict_proba(x)
    return {
        "log_loss": float(log_loss(y, p, labels=range(5))),
        "macro_f1": report(y, p.argmax(1), np.repeat("monitor", len(y)))["macro_f1"],
    }


def train(args):
    cfg = config()
    manifest = verify_prepared(args.prepared)
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "source": source_hash(),
        "features": manifest["feature_version"],
        "config": cfg,
    }
    old = args.output / "manifest.json"
    if old.exists() and read_json(old)["fingerprint"] != fingerprint:
        raise ValueError("Changed tree source/configuration")
    snapshot(args.output)
    save(
        old,
        {
            "fingerprint": fingerprint,
            "specifications": SPECS,
            "training_cap": None,
            "validation_cap": None,
        },
    )
    x, y, _, _ = matrix(args.prepared, "train", augmented=True, seed=cfg["seed"])
    vx, vy, vp, _ = matrix(args.prepared, "validation", 0)
    rng = np.random.default_rng(cfg["seed"])
    monitor = rng.choice(len(y), min(5000, len(y)), replace=False)
    results = {}
    for name, width, power, leaf in SPECS:
        path = args.output / name
        path.mkdir(exist_ok=True)
        if (path / "result.json").exists():
            results[name] = read_json(path / "result.json")
            continue
        started = time.time()
        xx = x[:, :width]
        vv = vx[:, :width]
        weight = class_weights(y, power)[y]
        history = []
        if name.startswith("rf_"):
            model = make_pipeline(
                SimpleImputer(add_indicator=True),
                RandomForestClassifier(
                    n_estimators=25,
                    max_depth=20,
                    min_samples_leaf=leaf,
                    n_jobs=args.threads,
                    random_state=cfg["seed"],
                    warm_start=True,
                ),
            )
            for step in [25, 50, 100, 150, 200]:
                model[-1].set_params(n_estimators=step)
                model.fit(xx, y, randomforestclassifier__sample_weight=weight)
                a, b = diagnostic(model, xx[monitor], y[monitor]), diagnostic(
                    model, vv, vy
                )
                history.append(
                    {
                        "step": step,
                        "axis": "Number of trees",
                        **{"train_" + k: v for k, v in a.items()},
                        **{"validation_" + k: v for k, v in b.items()},
                        "monitoring_scope": "Train monitor <=5000; all common clean validation",
                    }
                )
                save(path / "history.json", history)
                print("TREES", name, step, b, flush=True)
        else:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=20,
                reg_lambda=5,
                tree_method="hist",
                n_jobs=args.threads,
                random_state=cfg["seed"],
                objective="multi:softprob",
                num_class=5,
                eval_metric="mlogloss",
            )
            model.fit(
                xx,
                y,
                sample_weight=weight,
                eval_set=[(xx[monitor], y[monitor]), (vv, vy)],
                verbose=False,
            )
            ev = model.evals_result()
            history = [
                {
                    "step": i + 1,
                    "axis": "Boosting round",
                    "train_log_loss": float(a),
                    "validation_log_loss": float(b),
                    "monitoring_scope": "Train monitor <=5000; all common clean validation",
                }
                for i, (a, b) in enumerate(
                    zip(ev["validation_0"]["mlogloss"], ev["validation_1"]["mlogloss"])
                )
            ]
            # Macro-F1 at fixed milestones, without changing the selected final-round model.
            for step in [25, 50, 100, 150, 200, 300]:
                for label, dx, dy in [
                    ("train", xx[monitor], y[monitor]),
                    ("validation", vv, vy),
                ]:
                    pred = model.predict_proba(dx, iteration_range=(0, step)).argmax(1)
                    history[step - 1][label + "_macro_f1"] = report(
                        dy, pred, np.repeat("monitor", len(dy))
                    )["macro_f1"]
            save(path / "history.json", history)
        joblib.dump(model, path / "model.joblib")
        validation = {}
        for level in LEVELS_MS:
            if level == 0:
                dx, dy, dp = vx, vy, vp
            else:
                dx, dy, dp, _ = matrix(args.prepared, "validation", level)
            validation[str(level)] = report(
                dy, model.predict_proba(dx[:, :width]).argmax(1), dp
            )
            if level != 0:
                del dx
        result = {
            "model": name,
            "selection_score": float(
                np.mean([r["macro_f1"] for r in validation.values()])
            ),
            "validation": validation,
            "training_epochs": len(y),
            "features": width,
            "weight_power": power,
            "history": history,
            "seconds": time.time() - started,
            "completed": True,
            "model_path": str(path / "model.joblib"),
        }
        save(path / "result.json", result)
        results[name] = result
        save(args.output / "results.json", results)
        print("FINISHED", name, result["selection_score"], flush=True)
        del model
        gc.collect()
    # The capped winner is a validation-only control; never fit it again or rerun its test.
    name = "previous_capped_rf_control"
    parent = Path(cfg["source_run"])
    if name not in results:
        selection = read_json(parent / "selection.json")
        oldname = selection["winner"]
        candidates = [parent / (oldname + ".pkl")]
        if not candidates[0].exists():
            raise ValueError("Missing saved prior control model")
        model = joblib.load(candidates[0])
        validation = {}
        for level in LEVELS_MS:
            dx, dy, dp, _ = matrix(args.prepared, "validation", level)
            validation[str(level)] = report(
                dy, model.predict_proba(dx[:, :54]).argmax(1), dp
            )
        results[name] = {
            "model": name,
            "selection_score": float(
                np.mean([r["macro_f1"] for r in validation.values()])
            ),
            "validation": validation,
            "training_epochs": 154949,
            "features": 54,
            "model_path": str(candidates[0]),
            "completed": True,
            "control": True,
            "history": [],
        }
    save(args.output / "results.json", results)
    print("ALL TREE CANDIDATES COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    train(p.parse_args())
