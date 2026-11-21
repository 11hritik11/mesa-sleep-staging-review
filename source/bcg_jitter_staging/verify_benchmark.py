"""Targeted engineering checks: physiology formulas, missing beats, and model APIs."""
import csv
import pickle
import tempfile
from pathlib import Path

import numpy as np

from experiment import jitter_peaks
from features import FEATURE_NAMES, BASE_NAMES, read_annotations, base_features, add_context
from benchmark import make_models, fit_with_history, partition_arrays, report


def main():
    import json
    single = report(np.zeros(8, dtype=int), np.zeros(8, dtype=int), np.array(["0001"] * 8))
    assert single["kappa"] is None and single["per_participant"]["0001"]["kappa"] is None
    json.dumps(single, allow_nan=False)
    rng = np.random.default_rng(5)
    peaks = np.arange(0, 150000, .8)
    noisy, mae = jitter_peaks(peaks, 60, rng)
    assert abs(mae - 60) < 1, mae
    # RR sinusoid with known frequency should yield a matching LF spectral peak.
    rr = .9 + .05 * np.sin(2 * np.pi * .1 * np.arange(900) * .9)
    beats = np.r_[0, np.cumsum(rr)]
    base = base_features(beats, 27, np.ones(27, dtype=bool))
    middle = base[13]
    assert abs(middle[BASE_NAMES.index("lf_peak_hz")] - .1) < .02
    assert middle[BASE_NAMES.index("lf_power_s2")] > middle[BASE_NAMES.index("hf_power_s2")]
    assert add_context(base).shape == (27, 54)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 54
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        path = folder / "beats.csv"
        with path.open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["RPoint", "Type", "stage"])
            writer.writerows((i * 256, 2 if i == 100 else 1, 99) for i in range(500))
        times, normal = read_annotations(path, 600)
        assert normal.sum() == len(times) - 3  # Both sides of one ectopic beat rejected.
        shuffled = np.random.default_rng(9).permutation(500)
        with path.open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["RPoint", "Type", "stage"])
            writer.writerows((i * 256, 2 if i == 100 else 1, 99) for i in shuffled)
        sorted_times, sorted_normal, audit = read_annotations(path, 600, return_audit=True)
        assert np.array_equal(times, sorted_times) and np.array_equal(normal, sorted_normal)
        assert audit["chronological_sort_applied"]
        target = folder / "train_0001.npz"
        np.savez(target, y=np.arange(100) % 5, common=np.ones(100, dtype=bool),
                 **{f"x{level}": np.full((100, 54), level, dtype=np.float32) for level in (0,20,40,60)})
        clean = partition_arrays(folder, ["0001"], "train", 42, cap=40)
        augmented = partition_arrays(folder, ["0001"], "train", 42, cap=40, augmented=True)
        assert np.array_equal(clean[3], augmented[3])
        assert np.array_equal(clean[1], augmented[1])
    x = rng.normal(size=(600, 54)).astype(np.float32)
    y = np.arange(600) % 5
    x[:, 0] += y * 2
    x[::10, 5] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        from reporting import plot_history, build_report
        for name, model in make_models(42, 1).items():
            model, history = fit_with_history(model, name, x, y, (x, y), 42,
                                              folder / f"training_history_{name}.json")
            assert history and np.isfinite(history[-1]["validation_log_loss"])
            plot_history(folder, name, history)
            prediction = model.predict(x[:10])
            assert prediction.shape == (10,)
            assert np.array_equal(prediction, pickle.loads(pickle.dumps(model)).predict(x[:10]))
            print("PASS", name, "and learning history", flush=True)
        ids = np.repeat(["0001", "0002", "0003"], 200)
        r = report(y, y, ids)
        reports = {str(n): r for n in (0,20,40,60)}
        def write(name, value):
            (folder / name).write_text(json.dumps(value))
        write("validation_results.json", {"mlp_clean": {"selection_score":1., "validation":reports}})
        write("selection.json", {"winner":"mlp_clean"})
        write("test_results.json", {"five_stage":reports})
        write("feature_importance_validation.json", [{"feature":n,"mean_macro_f1_drop":.01,"std":.001} for n in FEATURE_NAMES])
        write("coverage.json", [{"partition":p,"eligible":600,"usable":500,
                "realized_mae_ms":{str(n):float(n) for n in (0,20,40,60)}} for p in ("train","validation","test")])
        probabilities = np.full((len(y),5), .025)
        probabilities[np.arange(len(y)),y] = .9
        for level in (0,20,40,60):
            with (folder / f"test_predictions_{level}ms.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(["participant","epoch","reference","prediction"] + ["p_"+s for s in ["Wake","N1","N2","N3","REM"]])
                writer.writerows((pid,i%200,int(label),int(label),*p) for i,(pid,label,p) in enumerate(zip(ids,y,probabilities)))
        build_report(folder)
        assert (folder / "results_bundle.zip").stat().st_size > 0
        assert len(list((folder / "figures").glob("*.png"))) >= 30
        assert len(list((folder / "figures").glob("*.pdf"))) >= 30
        print("PASS complete report, probability plots, timelines, and downloadable bundle", flush=True)
    print("PASS feature formulas, annotation types, identical sampling, and all five model APIs", flush=True)


if __name__ == "__main__":
    main()
