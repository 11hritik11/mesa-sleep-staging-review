#!/usr/bin/env python3
"""Run the archived pipeline without editing it or its original input artifacts.

Paths are explicit. Scientific settings, fitting code and source fingerprints
remain frozen. Only cfg()['base_run'] is relocated in memory; this adaptation
and the effective configuration are recorded in each new output directory.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "scikit-learn",
    "matplotlib",
    "xgboost",
    "lightgbm",
    "catboost",
    "pyedflib",
    "joblib",
    "wfdb",
    "pandas",
)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def activate(base_run=None):
    """Import frozen modules and relocate the dataset root, never model settings."""
    # Original training imports NumPy and PyTorch before balanced_utils.
    # Preserve that order so macOS does not initialize a second OpenMP runtime
    # through XGBoost before PyTorch's bundled runtime is loaded.
    import numpy
    import torch

    for folder in ("bcg_jitter_staging", "bcg_full_training", "bcg_balanced_staging"):
        sys.path.insert(0, str(ROOT / "source" / folder))
    import balanced_utils as shared

    expected = load(ROOT / "source_checksums.json")
    for name, sha in expected.items():
        require(
            digest(ROOT / "source" / name) == sha, "Changed archived source: " + name
        )
    require(
        shared.fingerprint() == load(ROOT / "provenance.json")["scientific_source"],
        "Scientific source fingerprint mismatch",
    )
    if base_run is not None:
        config = shared.cfg()
        config["base_run"] = str(Path(base_run).resolve())
        shared.cfg = lambda: dict(config)
    return shared


def inputs(args, shared):
    base = args.base_run.resolve()
    balanced = args.balanced_run.resolve()
    prepared = balanced / "prepared"
    parent = shared.verify_prepared(base / "prepared")
    manifest = load(prepared / "balanced_manifest.json")
    require(manifest["status"] == "COMPLETE", "Incomplete extra-feature preparation")
    require(
        manifest["source_version"] == shared.fingerprint(), "Prepared source mismatch"
    )
    require(
        manifest["base_feature_version"] == parent["feature_version"], "Parent mismatch"
    )
    roles = manifest["validation_participants"]
    cal, selection = set(roles["calibration"]), set(roles["selection"])
    require(
        not cal & selection
        and cal | selection == set(parent["participants"]["validation"]),
        "Calibration/selection roles are invalid",
    )
    for part, people in parent["participants"].items():
        for person in people:
            require(
                (base / "prepared/records" / f"{part}_{person}.npz").is_file(),
                "Missing prepared base record",
            )
            require(
                (prepared / "extra" / f"{part}_{person}.npz").is_file(),
                "Missing prepared extra-feature record",
            )
    for path in (
        base / "prepared/population_scaler.npz",
        prepared / "extra_scaler.npz",
    ):
        require(path.is_file(), "Missing scaler: " + str(path))
    return base, balanced, prepared


def output_root(args, shared, base, balanced):
    out = args.output.resolve()
    for protected in (base, balanced, ROOT / "source"):
        require(
            out != protected and protected not in out.parents,
            "Output must be outside original input/source directories",
        )
    marker = out / "reproduction.json"
    identity = {
        "scientific_source": shared.fingerprint(),
        "base_run": str(base),
        "balanced_run": str(balanced),
        "effective_config": shared.cfg(),
        "base_manifest_sha256": digest(base / "prepared/manifest.json"),
        "extra_manifest_sha256": digest(balanced / "prepared/balanced_manifest.json"),
        "relocation": "Only base_run is overridden in memory; source bytes unchanged.",
    }
    if marker.exists():
        require(
            load(marker)["identity"] == identity, "Output belongs to different inputs"
        )
    else:
        require(
            not out.exists() or not any(out.iterdir()),
            "Use a new empty output directory",
        )
        out.mkdir(parents=True, exist_ok=True)
        save(
            marker,
            {
                "identity": identity,
                "environment": environment(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    return out


@contextmanager
def lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def replay(args, shared, base, balanced, prepared, out):
    """Recompute probabilities from original weights; never use inference caches."""
    import numpy as np
    import torch
    from balanced_inference import Predictor

    torch.set_num_threads(args.threads)
    results = load(balanced / "results/validation_results.json")
    selected = load(balanced / "results/selection.json")
    require(selected["source"] == shared.fingerprint(), "Selected source mismatch")
    used = set()

    def relocate(name):
        used.add(name)
        info = results[name]
        require(info["source"] == shared.fingerprint(), "Model source mismatch")
        if info["kind"] == "ensemble":
            for member in info["members"]:
                relocate(member)
        elif info["kind"] in ("deep", "classic"):
            info["output"] = str(balanced / info["kind"] / name)
        else:
            raise ValueError("Winner depends on an unsupported historical control")

    summaries = {}
    for track in args.tracks:
        winner = selected["tracks"][track]["winner"]
        relocate(winner)
        predictor = Predictor(prepared, results, cache=None)
        summaries[track] = {}
        for level in args.levels:
            probability, truth, people, epochs = predictor.predict(
                winner, "test", level, expanded=True
            )
            require(
                probability.shape == (len(truth), 5) and np.isfinite(probability).all(),
                "Invalid replay probabilities",
            )
            require(
                np.all(probability >= 0)
                and np.allclose(probability.sum(1), 1, atol=1e-10),
                "Invalid replay probability normalization",
            )
            keys = list(zip(people.tolist(), epochs.tolist()))
            require(len(set(keys)) == len(keys), "Duplicate replay identities")
            lookup = {key: i for i, key in enumerate(keys)}
            common = np.zeros(len(truth), dtype=bool)
            seen = set()
            max_error = 0.0
            saved_predictions = np.empty(len(truth), dtype=np.int64)
            path = (
                balanced / "results/tracks" / track / f"test_predictions_{level}ms.csv"
            )
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    key = row["participant"], int(row["epoch"])
                    require(
                        key in lookup and key not in seen,
                        "Saved/replay identity mismatch",
                    )
                    seen.add(key)
                    i = lookup[key]
                    require(
                        int(row["reference"]) == int(truth[i]),
                        "Reference label mismatch",
                    )
                    common[i] = bool(int(row["common"]))
                    saved_predictions[i] = int(row["prediction"])
                    expected = np.array(
                        [float(row["p_" + stage]) for stage in shared.STAGES]
                    )
                    max_error = max(
                        max_error, float(np.max(np.abs(probability[i] - expected)))
                    )
            require(seen == set(keys), "Replay coverage mismatch")
            predicted = probability.argmax(1)
            disagreement = int(np.sum(predicted != saved_predictions))
            common_report = shared.report(
                truth[common], predicted[common], people[common]
            )
            # Public output contains aggregate metrics only.
            common_report.pop("per_participant", None)
            summaries[track][str(level)] = {
                "max_probability_absolute_error": max_error,
                "prediction_disagreements": disagreement,
                "expanded_epochs": len(truth),
                "common": common_report,
                "matches_saved_outputs": max_error <= args.atol and disagreement == 0,
            }
            save(
                out / "replay_results.json",
                {"status": "IN_PROGRESS", "tracks": summaries},
            )
        del predictor
    models = {}
    for name in sorted(used):
        info = results[name]
        if info["kind"] in ("deep", "classic"):
            path = Path(info["output"]) / (
                "best.pt" if info["kind"] == "deep" else "model.joblib"
            )
            models[name] = digest(path)
    passed = all(
        r["matches_saved_outputs"]
        for values in summaries.values()
        for r in values.values()
    )
    save(
        out / "replay_results.json",
        {
            "status": "PASS" if passed else "MISMATCH",
            "test_inference_rerun": True,
            "inference_cache_used": False,
            "tolerance": args.atol,
            "checkpoint_sha256": models,
            "tracks": summaries,
        },
    )
    require(
        passed,
        "Replay differs from saved outputs; inspect replay_results.json, do not change tolerances to hide it",
    )
    print("CHECKPOINT REPLAY PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="Check numerical imports, versions and archived sources"
    )
    sub.add_parser(
        "smoke", help="Run original synthetic fitting/serialization/report tests"
    )
    for command in ("inventory", "fit", "fit-controls", "finalize", "replay"):
        p = sub.add_parser(command)
        p.add_argument(
            "--base-run",
            type=Path,
            required=True,
            help="RUN-005 directory containing prepared/",
        )
        p.add_argument(
            "--balanced-run",
            type=Path,
            required=True,
            help="Original RUN-006 directory",
        )
        p.add_argument("--threads", type=int, default=8)
        if command != "inventory":
            p.add_argument(
                "--output", type=Path, required=True, help="New reproduction directory"
            )
        if command == "fit":
            p.add_argument(
                "--candidate",
                required=True,
                help="Exact candidate name, or all (22 new fits)",
            )
        if command == "replay":
            p.add_argument(
                "--tracks",
                nargs="+",
                choices=("beat", "edr", "multi"),
                default=["multi"],
            )
            p.add_argument(
                "--levels",
                nargs="+",
                type=int,
                choices=(0, 20, 40, 60),
                default=[0, 20, 40, 60],
            )
            p.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    shared = activate(getattr(args, "base_run", None))
    if args.command == "doctor":
        import torch, scipy, xgboost, lightgbm, catboost, pyedflib, matplotlib, joblib

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "environment": environment(),
                    "scientific_source": shared.fingerprint(),
                },
                indent=2,
            )
        )
        return
    if args.command == "smoke":
        require(not sys.flags.optimize, "Do not use Python -O with original assertions")
        from tests_balanced import run

        run(real=False)
        print("SYNTHETIC ENGINEERING CHECK ONLY; no scientific performance claim")
        return
    require(args.threads > 0, "threads must be positive")
    base, balanced, prepared = inputs(args, shared)
    if args.command == "inventory":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "base_run": str(base),
                    "balanced_run": str(balanced),
                    "counts": load(base / "prepared/manifest.json")["counts"],
                },
                indent=2,
            )
        )
        return
    out = output_root(args, shared, base, balanced)
    key = args.candidate if args.command == "fit" else args.command
    require(Path(key).name == key, "Invalid candidate")
    with lock(out / "locks" / (key + ".lock")):
        if args.command == "replay":
            replay(args, shared, base, balanced, prepared, out)
        elif args.command == "fit":
            from train_balanced_deep import train as deep
            from train_balanced_classic import train as classic

            specs = [("classic", i, s) for i, s in enumerate(shared.CLASSIC_SPECS)]
            specs += [("deep", i, s) for i, s in enumerate(shared.DEEP_SPECS)]
            require(
                args.candidate == "all"
                or any(s["name"] == args.candidate for _, _, s in specs),
                "Unknown candidate; see reproduce.py fit --help and balanced_utils.py",
            )
            for kind, index, spec in specs:
                if args.candidate in ("all", spec["name"]):
                    with lock(out / "locks" / ("worker-" + spec["name"] + ".lock")):
                        (classic if kind == "classic" else deep)(
                            SimpleNamespace(
                                prepared=prepared,
                                output=out / kind,
                                index=index,
                                threads=args.threads,
                            )
                        )
        elif args.command == "fit-controls":
            from previous_controls import run

            run(SimpleNamespace(prepared=prepared, output=out / "controls"))
        elif args.command == "finalize":
            from finalize_balanced import run

            run(
                SimpleNamespace(
                    prepared=prepared,
                    classic=out / "classic",
                    deep=out / "deep",
                    controls=out / "controls",
                    output=out / "results",
                    threads=args.threads,
                )
            )


if __name__ == "__main__":
    main()
