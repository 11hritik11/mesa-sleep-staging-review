#!/usr/bin/env python3
"""Paper-inspired MESA ECG peak-jitter sleep staging with subject-held-out evaluation."""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, lombscargle, sosfiltfilt
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
LEVELS_MS = (0, 20, 40, 60)
FS = 256
SOS = butter(3, [5, 20], fs=FS, btype="bandpass", output="sos")
FREQUENCIES = np.r_[np.linspace(0.04, 0.15, 17), np.linspace(0.151, 0.4, 25)]


def beat_times(windows, input_valid):
    """Heuristic QRS detector; never treat these estimates as validated annotations."""
    found = []
    for i, ok in enumerate(input_valid):
        if not ok:
            continue
        x = np.asarray(windows[i], dtype=np.float64)
        x -= np.median(x)
        filtered = sosfiltfilt(SOS, x)
        energy = np.convolve(
            np.diff(filtered, prepend=filtered[0]) ** 2, np.ones(31) / 31, mode="same"
        )
        mid = np.median(energy)
        mad = np.median(np.abs(energy - mid))
        threshold = max(3 * mad, 0.05 * np.percentile(energy, 95), 1e-20)
        candidates, _ = find_peaks(
            energy,
            distance=int(0.25 * FS),
            height=mid + threshold,
            prominence=threshold,
        )
        for candidate in candidates:
            lo, hi = max(0, candidate - 20), min(len(x), candidate + 21)
            peak = lo + np.argmax(np.abs(filtered[lo:hi]))
            found.append(i * 30 + peak / FS)
    return np.unique(np.asarray(found, dtype=np.float64))


def jitter_peaks(peaks, target_mae_ms, rng):
    if target_mae_ms == 0:
        return peaks.copy(), 0.0
    # For independent N(0,sigma^2) beat offsets, E|delta_(i+1)-delta_i|=2*sigma/sqrt(pi).
    sigma = target_mae_ms * np.sqrt(np.pi) / 2000
    jittered = peaks + rng.normal(0, sigma, len(peaks))
    mae = float(np.mean(np.abs(np.diff(jittered) - np.diff(peaks))) * 1000)
    return jittered, mae


def hrv_features(peaks, count):
    """Centered five-minute windows, one prediction per 30-second epoch."""
    answer = np.full((count, 6), np.nan, dtype=np.float32)
    if len(peaks) < 25:
        return answer
    rr = np.diff(peaks)
    times = peaks[1:]
    good = (rr >= 0.3) & (rr <= 2.0)
    for epoch in range(count):
        start, end = epoch * 30 - 135, epoch * 30 + 165
        window = (times >= start) & (times < end)
        if window.sum() < 80 or (good & window).sum() / window.sum() < 0.9:
            continue
        t, r = times[window & good], rr[window & good]
        if t[-1] - t[0] < 240:
            continue
        centered = r - np.mean(r)
        power = lombscargle(t - t[0], centered, 2 * np.pi * FREQUENCIES, normalize=True)
        lf = np.trapz(power[:17], FREQUENCIES[:17])
        hf = np.trapz(power[17:], FREQUENCIES[17:])
        answer[epoch, :5] = (np.mean(r), np.std(r), lf, hf, lf / max(hf, 1e-12))
    answer[:, 5] = median_filter(answer[:, 4], size=5, mode="nearest")
    return answer


def selected_ids(data, count, seed):
    split_path = data / "participant_split.json"
    split = json.loads(split_path.read_text())
    rng = np.random.default_rng(seed)
    quotas = {
        "train": count - 2 * round(count * 0.15),
        "validation": round(count * 0.15),
        "test": round(count * 0.15),
    }
    result = {}
    for part, n in quotas.items():
        available = [
            pid
            for pid in split[part]
            if (data / "recordings" / f"mesa-sleep-{pid}" / "epochs.csv").exists()
        ]
        if len(available) < n:
            raise ValueError(
                f"Only {len(available)} prepared {part} participants; need {n}"
            )
        result[part] = sorted(rng.choice(sorted(available), n, replace=False).tolist())
    return result, hashlib.sha256(split_path.read_bytes()).hexdigest()


def load_recording(data, pid, part):
    folder = data / "recordings" / f"mesa-sleep-{pid}"
    with (folder / "epochs.csv").open() as file:
        rows = list(csv.DictReader(file))
    if any(
        row["participant_id"] != pid
        or row["partition"] != part
        or int(row["epoch_index"]) != i
        for i, row in enumerate(rows)
    ):
        raise ValueError(f"Bad epoch identity: {pid}")
    labels = np.array(
        [int(r["stage_code"]) if r["eligible"] == "True" else -1 for r in rows],
        dtype=np.int8,
    )
    valid = np.array(
        [
            r["finite_samples"] == "True" and r["constant_signal"] == "False"
            for r in rows
        ],
        dtype=bool,
    )
    if np.any((labels >= 0) & ~valid):
        raise ValueError(f"Eligible epoch with invalid ECG: {pid}")
    windows = np.load(folder / "ecg_windows.npy", mmap_mode="r")
    if windows.shape != (len(rows), 30 * FS):
        raise ValueError(f"Bad ECG array: {pid}")
    return windows, labels, valid


def extract(data, out, count, seed):
    ids, split_hash = selected_ids(data, count, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cohort.json").write_text(
        json.dumps(
            {
                "participants": ids,
                "split_sha256": split_hash,
                "seed": seed,
                "levels_target_hbi_mae_ms": LEVELS_MS,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "paper_deviations": [
                    "MESA cohort sampled from fixed existing split",
                    "subject-held-out evaluation",
                    "heuristic ECG peak detection",
                    "four error levels rather than 98",
                    "centered HRV windows",
                ],
            },
            indent=2,
        )
    )
    for part, group in ids.items():
        for pid in group:
            target = out / f"{part}_{pid}.npz"
            if target.exists():
                continue
            windows, labels, valid = load_recording(data, pid, part)
            peaks = beat_times(windows, valid)
            rng = np.random.default_rng(seed + int(pid))
            features = {}
            realized = {}
            for level in LEVELS_MS:
                jittered, mae = jitter_peaks(peaks, level, rng)
                features[f"x{level}"] = hrv_features(jittered, len(labels))
                realized[str(level)] = mae
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                y=labels,
                peaks=len(peaks),
                realized_mae_json=json.dumps(realized),
                **features,
            )
            temporary.replace(target)
            print(
                f"extracted {part} {pid}: {len(labels)} epochs, {len(peaks)} peaks",
                flush=True,
            )


def arrays(out, part, ids, level):
    xx, yy, pp = [], [], []
    for pid in ids:
        with np.load(out / f"{part}_{pid}.npz") as saved:
            x, y = saved[f"x{level}"], saved["y"]
        ok = (y >= 0) & np.isfinite(x[:, 0]) & np.isfinite(x[:, 1])
        xx.append(x[ok])
        yy.append(y[ok])
        pp.extend([pid] * int(ok.sum()))
    return np.concatenate(xx), np.concatenate(yy), np.asarray(pp)


def metrics(y, predicted):
    return {
        "n_epochs": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=range(5), average="macro", zero_division=0)
        ),
        "kappa": float(cohen_kappa_score(y, predicted, labels=range(5))),
        "confusion_matrix": confusion_matrix(y, predicted, labels=range(5)).tolist(),
    }


def fit_and_evaluate(out, ids, seed):
    rng = np.random.default_rng(seed)
    x0, y0, _ = arrays(out, "train", ids["train"], 0)
    results = {
        "cohort": ids,
        "stages": STAGES,
        "levels_ms": LEVELS_MS,
        "models": {},
        "actual_mae_ms": {},
    }
    for method in ("clean_svm", "jitter_augmented_svm"):
        if method == "clean_svm":
            x, y = x0, y0
        else:
            extra = [
                arrays(out, "train", ids["train"], level)[:2] for level in LEVELS_MS[1:]
            ]
            x = np.concatenate([x0] + [item[0] for item in extra])
            y = np.concatenate([y0] + [item[1] for item in extra])
        # Fixed budget for the RBF SVM, equal across models. Every participant remains in training.
        if len(y) > 12000:
            chosen = rng.choice(len(y), 12000, replace=False)
            x, y = x[chosen], y[chosen]
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            SVC(
                C=10,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovo",
            ),
        )
        model.fit(x, y)
        item = {"train_epochs": len(y), "validation": {}, "test": {}}
        for part in ("validation", "test"):
            for level in LEVELS_MS:
                test_x, test_y, test_pid = arrays(out, part, ids[part], level)
                pred = model.predict(test_x)
                report = metrics(test_y, pred)
                report["participant_macro_f1_mean"] = float(
                    np.mean(
                        [
                            f1_score(
                                test_y[test_pid == pid],
                                pred[test_pid == pid],
                                labels=range(5),
                                average="macro",
                                zero_division=0,
                            )
                            for pid in ids[part]
                        ]
                    )
                )
                item[part][str(level)] = report
        results["models"][method] = item
        (out / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False)
        )
        print(method, json.dumps(item["validation"]), flush=True)
    for part, group in ids.items():
        for level in LEVELS_MS:
            values = []
            for pid in group:
                with np.load(out / f"{part}_{pid}.npz") as saved:
                    values.append(
                        json.loads(str(saved["realized_mae_json"]))[str(level)]
                    )
            results["actual_mae_ms"][f"{part}_{level}"] = float(np.mean(values))
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/derived/full_cohort_windows_v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/derived/bcg_jitter_runs"
        / ("run_" + os.environ.get("SLURM_JOB_ID", "local")),
    )
    parser.add_argument("--participants", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    ids, _ = selected_ids(args.data, args.participants, args.seed)
    extract(args.data, args.output, args.participants, args.seed)
    fit_and_evaluate(args.output, ids, args.seed)


if __name__ == "__main__":
    main()
