#!/usr/bin/env python3
"""Full-epoch classical baselines with balanced-accuracy validation selection."""
import argparse, time, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import log_loss
from balanced_utils import (
    cfg,
    CLASSIC_SPECS,
    read_json,
    save,
    fingerprint,
    snapshot,
    matrix,
    choose_beta_across_levels,
    ba_from_cm,
    subset_labels,
    save_evidence,
)


class ApproximateClassifier:
    def __init__(self, family, seed):
        self.family = family
        self.seed = seed
        self.imputer = SimpleImputer(add_indicator=True)
        self.scale = StandardScaler()
        self.kernel = (
            Nystroem(kernel="rbf", gamma=0.01, n_components=256, random_state=seed)
            if family == "rbf_approx_svm"
            else None
        )
        self.estimator = SGDClassifier(
            loss="log_loss" if family == "logistic" else "hinge",
            alpha=0.0001,
            random_state=seed,
            average=True,
        )
        self.classes_ = np.arange(5)

    def fit_full(self, x, y, weights, callback):
        transformed = self.imputer.fit_transform(x)
        self.scale.fit(transformed)
        if self.kernel:
            ix = np.random.default_rng(self.seed).choice(
                len(x), min(4096, len(x)), replace=False
            )
            self.kernel.fit(self.scale.transform(transformed[ix]))
        for epoch in range(10):
            # Shuffling contiguous blocks controls memory while retaining all targets each pass.
            starts = np.random.default_rng(self.seed + epoch).permutation(
                np.arange(0, len(y), 4096)
            )
            for start in starts:
                z = self.scale.transform(transformed[start : start + 4096])
                if self.kernel:
                    z = self.kernel.transform(z)
                self.estimator.partial_fit(
                    z,
                    y[start : start + 4096],
                    classes=self.classes_,
                    sample_weight=weights[start : start + 4096],
                )
            callback(epoch + 1, self)
        return self

    def decision_function(self, x):
        output = []
        for start in range(0, len(x), 4096):
            z = self.scale.transform(self.imputer.transform(x[start : start + 4096]))
            if self.kernel:
                z = self.kernel.transform(z)
            output.append(self.estimator.decision_function(z))
        return np.concatenate(output)

    def predict_proba(self, x):
        z = self.decision_function(x)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)


def raw_scores(model, x):
    if isinstance(model, ApproximateClassifier):
        return model.decision_function(x)
    return np.log(np.maximum(model.predict_proba(x), 1e-12))


def train(args):
    c = cfg()
    spec = CLASSIC_SPECS[args.index]
    out = args.output / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    version = fingerprint()
    prepared = read_json(args.prepared / "balanced_manifest.json")
    if prepared["status"] != "COMPLETE" or prepared["source_version"] != version:
        raise ValueError("Changed preparation source")
    signature = {"source": version, "specification": spec}
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")[
        "signature"
    ] != signature:
        raise ValueError("Output source changed")
    if (out / "result.json").exists():
        print("ALREADY COMPLETE", spec["name"])
        return
    x, y, _, _ = matrix(
        args.prepared, "train", augmented=True, modality=spec["modality"]
    )
    vx, vy, vp, ve = matrix(args.prepared, "validation", 0, modality=spec["modality"])
    counts = np.bincount(y, minlength=5).astype(float)
    prior = counts / counts.sum()
    weights = 1 / (5 * prior[y])
    save(
        out / "manifest.json",
        {
            "signature": signature,
            "training_targets": len(y),
            "prior": prior.tolist(),
            "config": c,
        },
    )
    snapshot(out)
    started = time.time()
    history = []
    rng = np.random.default_rng(c["seed"])
    monitor = rng.choice(len(y), min(5000, len(y)), replace=False)
    val_take = subset_labels(vp, args.prepared) == "selection"

    def trace(step, model, axis="Fitting pass"):
        a = model.predict_proba(x[monitor])
        b = model.predict_proba(vx[val_take])
        truth = vy[val_take]
        cm = np.bincount(truth * 5 + b.argmax(1), minlength=25).reshape(5, 5)
        history.append(
            {
                "step": step,
                "axis": axis,
                "train_log_loss": float(log_loss(y[monitor], a, labels=range(5))),
                "validation_log_loss": float(log_loss(truth, b, labels=range(5))),
                "validation_balanced_accuracy": ba_from_cm(cm),
                "train_balanced_accuracy": ba_from_cm(
                    np.bincount(y[monitor] * 5 + a.argmax(1), minlength=25).reshape(
                        5, 5
                    )
                ),
                "monitoring_scope": "Fixed <=5000 training diagnostic rows; all clean selection-validation rows; raw uncorrected predictions",
            }
        )
        save(out / "history.json", history)
        print(
            "HISTORY",
            spec["name"],
            step,
            history[-1]["validation_balanced_accuracy"],
            flush=True,
        )

    family = spec["family"]
    if family in ("random_forest", "extra_trees"):
        cls = (
            RandomForestClassifier
            if family == "random_forest"
            else ExtraTreesClassifier
        )
        model = make_pipeline(
            SimpleImputer(add_indicator=True),
            cls(
                n_estimators=50,
                max_depth=24,
                min_samples_leaf=30,
                n_jobs=args.threads,
                random_state=c["seed"],
                warm_start=True,
            ),
        )
        for step in [50, 100, 200, 300]:
            model[-1].set_params(n_estimators=step)
            model.fit(x, y, **{model.steps[-1][0] + "__sample_weight": weights})
            trace(step, model, "Number of trees")
    elif family == "hist_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=50,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            learning_rate=0.05,
            l2_regularization=5,
            early_stopping=False,
            warm_start=True,
            random_state=c["seed"],
        )
        for step in [50, 100, 200, 300]:
            model.set_params(max_iter=step)
            model.fit(x, y, sample_weight=weights)
            trace(step, model, "Boosting round")
    elif family == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=30,
            reg_lambda=10,
            tree_method="hist",
            n_jobs=args.threads,
            random_state=c["seed"],
            num_class=5,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
        model.fit(
            x,
            y,
            sample_weight=weights,
            eval_set=[(x[monitor], y[monitor]), (vx[val_take], vy[val_take])],
            verbose=False,
        )
        ev = model.evals_result()
        history = [
            {
                "step": i + 1,
                "axis": "Boosting round",
                "train_log_loss": float(a),
                "validation_log_loss": float(b),
            }
            for i, (a, b) in enumerate(
                zip(ev["validation_0"]["mlogloss"], ev["validation_1"]["mlogloss"])
            )
        ]
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=100,
            learning_rate=0.035,
            reg_lambda=10,
            n_jobs=args.threads,
            random_state=c["seed"],
            verbosity=-1,
        )
        model.fit(
            x,
            y,
            sample_weight=weights,
            eval_set=[(x[monitor], y[monitor]), (vx[val_take], vy[val_take])],
            eval_metric="multi_logloss",
        )
        ev = model.evals_result_
        history = [
            {
                "step": i + 1,
                "axis": "Boosting round",
                "train_log_loss": float(a),
                "validation_log_loss": float(b),
            }
            for i, (a, b) in enumerate(
                zip(ev[next(iter(ev))]["multi_logloss"], ev["valid_1"]["multi_logloss"])
            )
        ]
    elif family == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=10,
            thread_count=args.threads,
            random_seed=c["seed"],
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            x,
            y,
            sample_weight=weights,
            eval_set=(vx[val_take], vy[val_take]),
            use_best_model=False,
        )
        ev = model.get_evals_result()
        history = [
            {
                "step": i + 1,
                "axis": "Boosting round",
                "train_log_loss": float(a),
                "validation_log_loss": float(b),
                "loss_note": "CatBoost training objective is class-weighted; validation objective is unweighted",
            }
            for i, (a, b) in enumerate(
                zip(ev["learn"]["MultiClass"], ev["validation"]["MultiClass"])
            )
        ]
    elif family == "adaboost":
        model = make_pipeline(
            SimpleImputer(add_indicator=True),
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=100),
                n_estimators=150,
                learning_rate=0.05,
                random_state=c["seed"],
            ),
        )
        model.fit(x, y, adaboostclassifier__sample_weight=weights)
        tm = model[0].transform(x[monitor])
        vm = model[0].transform(vx[val_take])
        for i, (a, b) in enumerate(
            zip(model[-1].staged_predict_proba(tm), model[-1].staged_predict_proba(vm))
        ):
            if i + 1 not in (10, 25, 50, 100, 150):
                continue
            truth = vy[val_take]
            history.append(
                {
                    "step": i + 1,
                    "axis": "Boosting round",
                    "train_log_loss": float(log_loss(y[monitor], a, labels=range(5))),
                    "validation_log_loss": float(log_loss(truth, b, labels=range(5))),
                    "validation_balanced_accuracy": ba_from_cm(
                        np.bincount(truth * 5 + b.argmax(1), minlength=25).reshape(5, 5)
                    ),
                    "train_balanced_accuracy": ba_from_cm(
                        np.bincount(y[monitor] * 5 + a.argmax(1), minlength=25).reshape(
                            5, 5
                        )
                    ),
                }
            )
    else:
        model = ApproximateClassifier(family, c["seed"])
        model.fit_full(
            x, y, weights, lambda step, m: trace(step, m, "Full-data SGD pass")
        )
    save(out / "history.json", history)
    joblib.dump(model, out / "model.joblib")
    evidence = {}
    full_evidence = {}
    for level in [0, 20, 40, 60]:
        if level == 0:
            xx, yy, pp, ee = vx, vy, vp, ve
        else:
            xx, yy, pp, ee = matrix(
                args.prepared, "validation", level, modality=spec["modality"]
            )
        evidence[level] = (raw_scores(model, xx), yy, pp)
        full_evidence[level] = (*evidence[level], ee)
    save_evidence(out, full_evidence)
    beta, reports, score = choose_beta_across_levels(evidence, args.prepared, prior)
    result = {
        "name": spec["name"],
        "specification": spec,
        "selection_score": score,
        "validation": reports,
        "beta": beta,
        "prior": prior.tolist(),
        "training_targets": len(y),
        "source": version,
        "history": history,
        "completed": True,
        "kind": "classic",
        "output": str(out),
        "probability_note": "SVM softmax scores and post-hoc prior corrections are not guaranteed calibrated probabilities",
        "seconds": time.time() - started,
    }
    save(out / "result.json", result)
    print("COMPLETE", spec["name"], score, flush=True)


ApproximateClassifier.__module__ = "train_balanced_classic"
if __name__ == "__main__":
    sys.modules["train_balanced_classic"] = sys.modules[__name__]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--threads", type=int, default=8)
    train(p.parse_args())
