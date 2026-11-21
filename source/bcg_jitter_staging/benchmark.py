#!/usr/bin/env python3
"""Full prepared MESA cohort: HRV features and validation-selected classifiers."""
import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from experiment import LEVELS_MS, STAGES, ROOT, jitter_peaks, load_recording, metrics as base_metrics
from features import FEATURE_NAMES, BASE_NAMES, read_annotations, base_features, add_context


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, prediction):
    # Kappa is undefined when reference and prediction both contain one identical class.
    # Preserve that fact as JSON null rather than assigning an invented numeric score.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = base_metrics(y, prediction)
    if not np.isfinite(result["kappa"]):
        result["kappa"] = None
        result["kappa_note"] = "Undefined: zero expected-disagreement denominator"
    return result


def verified_feature_reuse(previous_run, data, ids, cohort, seed, inventory_hash):
    """Reuse only complete caches with identical extraction code, cohort and inputs."""
    previous = json.loads((previous_run / "manifest.json").read_text())
    for name in ("features.py", "experiment.py"):
        if digest(previous_run / "source" / name) != digest(Path(__file__).with_name(name)):
            raise ValueError(f"Feature reuse refused: changed {name}")
    def extraction_ast(path):
        module = ast.parse(path.read_text())
        return ast.dump(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "extract_one"))
    if extraction_ast(previous_run / "source/benchmark.py") != extraction_ast(Path(__file__)):
        raise ValueError("Feature reuse refused: changed extraction function")
    if (previous["participants"] != ids or previous["seed"] != seed or
            previous["levels_ms"] != list(LEVELS_MS) or
            previous["annotation_inventory_sha256"] != inventory_hash or
            any(previous["cohort"][k] != cohort[k] for k in ("split_sha256", "prepared_inventory_sha256"))):
        raise ValueError("Feature reuse refused: cohort, seed, jitter or source inventory changed")
    coverage = json.loads((previous_run / "coverage.json").read_text())
    lookup = {(row["partition"], row["participant"]): row for row in coverage}
    expected = {(part, pid) for part, group in ids.items() for pid in group}
    if len(coverage) != len(lookup) or set(lookup) != expected:
        raise ValueError("Feature reuse requires complete participant coverage")
    cache = Path(previous["cache"])
    if any(not (cache / f"{part}_{pid}.npz").exists() for part, pid in expected):
        raise ValueError("Feature reuse requires every completed cache file")
    version = previous.get("feature_source_version", previous["source_version"])
    return cache, version, lookup


def extract_one(task):
    data, cache, part, pid, seed, version = task
    data, cache = Path(data), Path(cache)
    folder = data / "recordings" / f"mesa-sleep-{pid}"
    array_stat = (folder / "ecg_windows.npy").stat()
    annotation = ROOT / "data/raw/mesa/polysomnography/annotations-rpoints" / f"mesa-sleep-{pid}-rpoint.csv"
    provenance = {"version": version, "annotation_sha256": digest(annotation),
                  "summary_sha256": digest(folder / "summary.json"),
                  "epochs_sha256": digest(folder / "epochs.csv"), "seed": seed,
                  "array_bytes": array_stat.st_size, "array_mtime_ns": array_stat.st_mtime_ns}
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    target = cache / f"{part}_{pid}.npz"
    if target.exists():
        with np.load(target) as f:
            if str(f["signature"]) == signature:
                return {"participant": pid, "partition": part, "cached": True,
                        "eligible": int((f["y"] >= 0).sum()), "usable": int(f["common"].sum())}
        raise ValueError(f"Stale feature cache for {pid}")
    windows, labels, valid = load_recording(data, pid, part)
    peaks, normal_intervals, annotation_audit = read_annotations(annotation, len(labels) * 30,
                                                               return_audit=True)
    if len(peaks) < 100:
        raise ValueError(f"Insufficient detections in {pid}: {len(peaks)}")
    rng = np.random.default_rng(seed + int(pid))
    arrays, actual = {}, {}
    common = labels >= 0
    for level in LEVELS_MS:
        jittered, mae = jitter_peaks(peaks, level, rng)
        # Anchor window membership to clean beat times so jitter never changes the evaluation epochs.
        base = base_features(jittered, len(labels), valid, anchors=peaks[1:],
                             normal_intervals=normal_intervals)
        arrays[f"x{level}"] = add_context(base)
        common &= np.isfinite(base[:, 0])
        actual[str(level)] = mae
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, y=labels, common=common, signature=signature,
                        peaks=peaks, realized_mae_json=json.dumps(actual), **arrays)
    temporary.replace(target)
    return {"participant": pid, "partition": part, "cached": False,
            "eligible": int((labels >= 0).sum()), "usable": int(common.sum()),
        "n_peaks": len(peaks), "annotation_audit": annotation_audit,
        "normal_interval_fraction": float(normal_intervals.mean()),
        "realized_mae_ms": actual}


def partition_arrays(cache, ids, part, seed, cap=None, augmented=False, level=0):
    xx, yy, pp, ee = [], [], [], []
    for pid in ids:
        rng = np.random.default_rng(seed + int(pid))
        with np.load(cache / f"{part}_{pid}.npz") as f:
            indices = np.flatnonzero(f["common"])
            if cap and len(indices) > cap:
                indices = np.sort(rng.choice(indices, cap, replace=False))
            if not len(indices):
                continue
            x = f[f"x{level}"][indices].copy()
            if augmented:
                assigned = rng.integers(0, len(LEVELS_MS), len(indices))
                for k, jitter in enumerate(LEVELS_MS):
                    chosen = assigned == k
                    x[chosen] = f[f"x{jitter}"][indices[chosen]]
            xx.append(x); yy.append(f["y"][indices]); pp.extend([pid] * len(indices)); ee.extend(indices)
    if not xx:
        raise ValueError(f"No usable {part} epochs")
    return np.concatenate(xx), np.concatenate(yy), np.asarray(pp), np.asarray(ee)


def make_models(seed, workers):
    common = dict(n_estimators=200, max_depth=20, min_samples_leaf=20,
                  class_weight="balanced", n_jobs=workers, random_state=seed)
    return {
        "random_forest": make_pipeline(SimpleImputer(add_indicator=True), RandomForestClassifier(**common)),
        "extra_trees": make_pipeline(SimpleImputer(add_indicator=True), ExtraTreesClassifier(**common)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31,
            min_samples_leaf=100, learning_rate=.05, l2_regularization=1, early_stopping=False,
            class_weight="balanced", random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
            subsample=.8, colsample_bytree=.8, min_child_weight=20, reg_lambda=5,
            tree_method="hist", n_jobs=workers, random_state=seed, objective="multi:softprob",
            num_class=5, eval_metric="mlogloss"),
        "mlp": make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=.01, batch_size=512,
                          learning_rate_init=.001, random_state=seed, early_stopping=False)),
    }


def report(y, prediction, participants):
    result = metrics(y, prediction)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction,
        labels=range(5), zero_division=0)
    result["per_stage"] = {name: {"precision": float(precision[i]), "recall": float(recall[i]),
        "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(STAGES)}
    result["per_participant"] = {pid: metrics(y[participants == pid], prediction[participants == pid])
                                 for pid in sorted(set(participants))}
    result["participant_macro_f1_mean"] = float(np.mean([
        v["macro_f1"] for v in result["per_participant"].values()]))
    return result


def monitor_metrics(model, train, validation):
    result = {}
    for name, (x, y) in (("train", train), ("validation", validation)):
        probabilities = model.predict_proba(x)
        result[name + "_log_loss"] = float(log_loss(y, probabilities, labels=np.arange(5)))
        result[name + "_macro_f1"] = metrics(y, model.classes_[np.argmax(probabilities, axis=1)])["macro_f1"]
    return result


def fit_mlp(model, x, y, validation, seed, history_path=None):
    # Fit preprocessing exclusively on training. Participant validation controls epochs.
    transformed = model[:-1].fit_transform(x, y)
    validation_x = model[:-1].transform(validation[0])
    estimator = model[-1]
    rng = np.random.default_rng(seed)
    monitor = rng.choice(len(y), min(5000, len(y)), replace=False)
    weights = compute_sample_weight("balanced", y)
    best, best_score, stale, history = None, -1, 0, []
    for epoch in range(30):
        sampled = rng.choice(len(y), len(y), replace=True, p=weights / weights.sum())
        estimator.partial_fit(transformed[sampled], y[sampled], classes=np.arange(5))
        score = metrics(validation[1], estimator.predict(validation_x))["macro_f1"]
        row = {"step": epoch + 1, "axis": "Training epoch", "training_objective": float(estimator.loss_),
               "validation_clean_macro_f1": score,
               **monitor_metrics(estimator, (transformed[monitor], y[monitor]), (validation_x, validation[1]))}
        history.append(row)
        if history_path:
            save(history_path, history)
        if score > best_score:
            best, best_score, stale = pickle.dumps(estimator), score, 0
        else:
            stale += 1
        if stale >= 5:
            break
    model.steps[-1] = (model.steps[-1][0], pickle.loads(best))
    return model, history


def fit_with_history(model, family, x, y, validation, seed, path):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(y), min(5000, len(y)), replace=False)
    train_monitor = (x[chosen], y[chosen])
    val_monitor = (validation[0], validation[1])
    if family == "mlp":
        return fit_mlp(model, x, y, validation, seed, path)
    if family == "xgboost":
        model.fit(x, y, sample_weight=compute_sample_weight("balanced", y),
                  eval_set=[train_monitor, val_monitor], verbose=False)
        traces = model.evals_result()
        history = [{"step": i + 1, "axis": "Boosting round", "train_log_loss": float(train),
                    "validation_log_loss": float(val)} for i, (train, val) in enumerate(zip(
                    traces["validation_0"]["mlogloss"], traces["validation_1"]["mlogloss"]))]
    else:
        forest = family in ("random_forest", "extra_trees")
        estimator = model[-1] if forest else model
        parameter = "n_estimators" if forest else "max_iter"
        steps = [25, 50, 100, 150, 200]
        estimator.set_params(warm_start=True)
        history = []
        for step in steps:
            estimator.set_params(**{parameter: step})
            model.fit(x, y)
            history.append({"step": step, "axis": "Number of trees" if forest else "Boosting iteration",
                            **monitor_metrics(model, train_monitor, val_monitor)})
            save(path, history)
    save(path, history)
    return model, history


def run(args):
    sys.path.insert(0, str(ROOT / "ecg_staging"))
    from train_full_transformer import load_cohort
    ids, cohort = load_cohort(args.data)
    download_manifest = ROOT / "data/raw/mesa/polysomnography/annotations-rpoints/download_manifest.json"
    inventory = json.loads(download_manifest.read_text())
    if inventory.get("limited_smoke") or inventory["expected"] != inventory["downloaded_or_verified"]:
        raise ValueError("Complete the full verified annotation download before training")
    published_missing = set(inventory["missing_from_nsrr_listing"])
    annotated = {row["file"].removeprefix("mesa-sleep-").removesuffix("-rpoint.csv")
                 for row in inventory["files"]}
    all_ids = set(sum(ids.values(), []))
    if annotated & published_missing or annotated | published_missing != all_ids:
        raise ValueError("Annotation inventory does not match the prepared cohort")
    ids = {part: [pid for pid in group if pid in annotated] for part, group in ids.items()}
    if args.limit:
        # Engineering smoke only: never submit this as full-cohort evidence.
        ids = {part: group[:args.limit] for part, group in ids.items()}
    missing = [pid for group in ids.values() for pid in group if not (
        ROOT / "data/raw/mesa/polysomnography/annotations-rpoints" / f"mesa-sleep-{pid}-rpoint.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing beat annotations for {len(missing)} participants; run download_annotations.py first")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for name in ("benchmark.py", "features.py", "experiment.py", "FEATURES.md", "run_benchmark.slurm",
                 "download_annotations.py", "reporting.py"):
        shutil.copy2(Path(__file__).parent / name, source / name)
    version = hashlib.sha256((digest(Path(__file__)) + digest(Path(__file__).with_name("features.py")) +
                              digest(Path(__file__).with_name("experiment.py"))).encode()).hexdigest()
    cache = ROOT / "data/derived/bcg_hrv_features_v2" / version[:16]
    feature_version, reused_coverage = version, {}
    inventory_hash = digest(download_manifest)
    if args.reuse_features_from:
        cache, feature_version, reused_coverage = verified_feature_reuse(args.reuse_features_from,
            args.data, ids, cohort, args.seed, inventory_hash)
    cache.mkdir(parents=True, exist_ok=True)
    save(args.output / "manifest.json", {"cohort": cohort, "participants": ids,
        "annotation_inventory_sha256": digest(download_manifest),
        "excluded_without_published_annotations": sorted(published_missing),
        "source_version": version, "features": FEATURE_NAMES, "cache": str(cache),
        "feature_source_version": feature_version,
        "reused_feature_run": str(args.reuse_features_from) if args.reuse_features_from else None,
        "seed": args.seed, "levels_ms": LEVELS_MS, "train_cap_per_participant": args.train_cap,
        "validation_cap_per_participant": args.validation_cap,
        "selection": "mean validation five-stage macro-F1 across 0/20/40/60 ms; sampled identically per level",
        "test_policy": "evaluate only validation winner; no test-based fitting or selection",
        "scope": "NSRR R-point and normal-beat annotations, 30 base HRV/QC features plus 24 context features"})
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output / "requirements_frozen.txt").write_text(frozen)
    tasks = [(str(args.data), str(cache), part, pid, args.seed, feature_version)
             for part, group in ids.items() for pid in group]
    coverage = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(extract_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            if reused_coverage:
                old = reused_coverage[(row["partition"], row["participant"])]
                if any(old[k] != row[k] for k in ("eligible", "usable")):
                    raise ValueError("Reused feature coverage differs from saved evidence")
                row = {**old, **row}
            coverage.append(row)
            save(args.output / "coverage.json", coverage)
            print(f"FEATURES {len(coverage)}/{len(tasks)} {coverage[-1]}", flush=True)
    validation = {level: partition_arrays(cache, ids["validation"], "validation", args.seed,
                         cap=args.validation_cap, level=level) for level in LEVELS_MS}
    results, winner, best = {}, None, -1
    from reporting import plot_history, build_report
    for augmented in (False, True):
        x, y, _, _ = partition_arrays(cache, ids["train"], "train", args.seed,
                                     cap=args.train_cap, augmented=augmented)
        if set(y) != set(range(5)):
            raise ValueError("Training must contain all five stages")
        for family, model in make_models(args.seed, args.workers).items():
            name = family + ("_augmented" if augmented else "_clean")
            start = time.time()
            history = None
            print("FIT", name, len(y), flush=True)
            model, history = fit_with_history(model, family, x, y, validation[0], args.seed,
                                              args.output / f"training_history_{name}.json")
            plot_history(args.output, name, history)
            reports = {str(level): report(v[1], model.predict(v[0]), v[2])
                       for level, v in validation.items()}
            score = float(np.mean([r["macro_f1"] for r in reports.values()]))
            results[name] = {"validation": reports, "selection_score": score,
                             "fit_seconds": time.time() - start, "training_epochs": len(y),
                             "training_history": history, "parameters": str(model.get_params())}
            with (args.output / f"{name}.pkl").open("wb") as file:
                pickle.dump(model, file)
            save(args.output / "validation_results.json", results)
            print("VALIDATION", name, score, flush=True)
            if score > best:
                winner, best = name, score
    save(args.output / "selection.json", {"winner": winner, "validation_score": best,
        "criterion": "mean validation macro-F1 across four jitter levels", "selection_complete": True})
    with (args.output / f"{winner}.pkl").open("rb") as file:
        model = pickle.load(file)
    # Attribution uses validation data exclusively and does not change the model.
    vx, vy, _, _ = validation[40]
    chosen = np.random.default_rng(args.seed).choice(len(vy), min(4000, len(vy)), replace=False)
    relevance = permutation_importance(model, vx[chosen], vy[chosen], scoring="f1_macro",
                                      n_repeats=3, n_jobs=1, random_state=args.seed)
    save(args.output / "feature_importance_validation.json", sorted([
        {"feature": name, "mean_macro_f1_drop": float(mean), "std": float(sd)}
        for name, mean, sd in zip(FEATURE_NAMES, relevance.importances_mean, relevance.importances_std)],
        key=lambda item: -item["mean_macro_f1_drop"]))
    test_results = {}
    for level in LEVELS_MS:
        tx, ty, participants, epochs = partition_arrays(cache, ids["test"], "test", args.seed, level=level)
        probabilities = model.predict_proba(tx)
        prediction = model.classes_[np.argmax(probabilities, axis=1)]
        test_results[str(level)] = report(ty, prediction, participants)
        with (args.output / f"test_predictions_{level}ms.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["participant", "epoch", "reference", "prediction"] + ["p_" + s for s in STAGES])
            writer.writerows((pid, int(epoch), int(reference), int(pred), *map(float, probability))
                             for pid, epoch, reference, pred, probability in
                             zip(participants, epochs, ty, prediction, probabilities))
    save(args.output / "test_results.json", {"winner": winner, "five_stage": test_results})
    build_report(args.output)
    print("COMPLETED", winner, best, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/derived/full_cohort_windows_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/derived/bcg_benchmark_runs" /
                        ("run_" + os.environ.get("SLURM_JOB_ID", "local")))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-cap", type=int, default=128)
    parser.add_argument("--validation-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--limit", type=int, help="Engineering smoke: max participants per partition")
    parser.add_argument("--reuse-features-from", type=Path, help="Verify and reuse a completed extraction run")
    run(parser.parse_args())
