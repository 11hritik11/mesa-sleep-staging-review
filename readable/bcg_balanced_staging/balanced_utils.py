"""Shared frozen provenance and explicit five-class balanced-accuracy utilities."""

import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bcg_full_training"))
from common import (
    read_json,
    save,
    config as oldconfig,
    verify_prepared,
    matrix as oldmatrix,
    report as oldreport,
    STAGES,
    LEVELS_MS,
    NAMES as BASE_NAMES,
    relative_features,
)
from sequences import SequenceDataset, ChunkSampler
from temporal import loader, checkpoint, TemporalCNN, TemporalBlock

HERE = Path(__file__).parent
EXTRA_RAW = [
    f"{s}_{w}s_{n}"
    for s in ("edr", "thorax", "abdomen", "airflow")
    for w in (30, 120, 300)
    for n in (
        "peak_hz",
        "spectral_entropy",
        "resp_band_fraction",
        "iqr",
        "breath_interval_cv",
        "coverage",
    )
]
EXTRA_RAW += [
    f"spo2_{w}s_{n}"
    for w in (30, 120, 300)
    for n in ("mean", "minimum", "sd", "fraction_below90")
]
EXTRA_RAW += [
    f"leg_{w}s_{n}" for w in (30, 120, 300) for n in ("rms", "iqr", "activity_fraction")
]
EXTRA_NAMES = EXTRA_RAW + ["night_relative_" + n for n in EXTRA_RAW]
DEEP_SPECS = [
    {
        "name": "tcn_inverse",
        "architecture": "tcn",
        "modality": "beat",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "tcn_focal",
        "architecture": "tcn",
        "modality": "beat",
        "mask": "common",
        "power": 0.5,
        "focal": 2.0,
    },
    {
        "name": "tcn_quality_expanded",
        "architecture": "tcn",
        "modality": "beat",
        "mask": "expanded",
        "power": 1.0,
        "quality": True,
    },
    {
        "name": "bilstm_inverse",
        "architecture": "lstm",
        "modality": "beat",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "bigru_inverse",
        "architecture": "gru",
        "modality": "beat",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "transformer_inverse",
        "architecture": "transformer",
        "modality": "beat",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "mlp_inverse",
        "architecture": "mlp",
        "modality": "beat",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "tcn_edr",
        "architecture": "tcn",
        "modality": "edr",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "tcn_multisensor",
        "architecture": "tcn",
        "modality": "multi",
        "mask": "common",
        "power": 1.0,
    },
    {
        "name": "bigru_multisensor",
        "architecture": "gru",
        "modality": "multi",
        "mask": "common",
        "power": 1.0,
    },
]
CLASSIC_SPECS = [
    {"name": n + "_" + m, "family": n, "modality": m}
    for n, m in [
        ("random_forest", "beat"),
        ("extra_trees", "beat"),
        ("hist_boosting", "beat"),
        ("xgboost", "beat"),
        ("lightgbm", "beat"),
        ("catboost", "beat"),
        ("adaboost", "beat"),
        ("rbf_approx_svm", "beat"),
        ("linear_svm", "beat"),
        ("logistic", "beat"),
        ("xgboost", "multi"),
        ("lightgbm", "multi"),
    ]
]


def cfg():
    return read_json(HERE / "balanced_config.json")


def base():
    return Path(cfg()["base_run"]) / "prepared"


def fingerprint():
    h = hashlib.sha256()
    for folder in (HERE, ROOT / "bcg_full_training", ROOT / "bcg_jitter_staging"):
        for p in sorted(folder.iterdir()):
            if (
                p.is_file()
                and p.suffix in (".py", ".json")
                and p.name
                not in (
                    "balanced_report.py",
                    "full_report.py",
                    "reporting.py",
                    "launch.py",
                    "verify_source.py",
                    "tests.py",
                    "tests_balanced.py",
                )
            ):
                h.update(str(p.relative_to(ROOT)).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def snapshot(out):
    for folder in (HERE, ROOT / "bcg_full_training", ROOT / "bcg_jitter_staging"):
        target = out / "source" / folder.name
        target.mkdir(parents=True, exist_ok=True)
        for p in folder.iterdir():
            if p.is_file() and p.suffix in (".py", ".json", ".md", ".slurm"):
                shutil.copy2(p, target / p.name)


def report(y, p, pids):
    r = oldreport(y, p, pids)
    cm = np.asarray(r["confusion_matrix"])
    support = cm.sum(1)
    if np.any(support == 0):
        r["balanced_accuracy"] = None
        r["balanced_accuracy_note"] = (
            "Undefined five-class balanced accuracy: one or more reference stages absent"
        )
    else:
        r["balanced_accuracy"] = float(np.mean(np.diag(cm) / support))
    return r


def ba_from_cm(cm):
    cm = np.asarray(cm)
    den = cm.sum(1)
    if np.any(den == 0):
        raise ValueError("All five classes required in aggregate balanced accuracy")
    return float((np.diag(cm) / den).mean())


def extra_columns(modality):
    if modality == "beat":
        return []
    if modality == "multi":
        return list(range(len(EXTRA_NAMES)))
    return [
        i
        for i, n in enumerate(EXTRA_NAMES)
        if n.startswith("edr_") or n.startswith("night_relative_edr_")
    ]


def feature_names(modality):
    return BASE_NAMES + [EXTRA_NAMES[i] for i in extra_columns(modality)]


def partitions(prepared):
    return read_json(prepared / "balanced_manifest.json")["validation_participants"]


def subset_labels(pid, prepared):
    groups = partitions(prepared)
    calibration = set(groups["calibration"])
    selection = set(groups["selection"])
    if calibration & selection or not set(pid.tolist()) <= calibration | selection:
        raise ValueError("Invalid validation role membership")
    return np.array(["calibration" if p in calibration else "selection" for p in pid])


def matrix(
    prepared, part, level=0, augmented=False, modality="beat", mask_name="common"
):
    x, y, p, e = oldmatrix(
        base(),
        part,
        level,
        augmented=augmented,
        seed=cfg()["seed"],
        mask_name=mask_name,
    )
    columns = extra_columns(modality)
    if columns:
        extra = np.empty((len(y), len(columns)), dtype=np.float32)
        for pid in dict.fromkeys(p.tolist()):
            take = np.flatnonzero(p == pid)
            with np.load(prepared / "extra" / f"{part}_{pid}.npz") as f:
                extra[take] = f["x"][e[take]][:, columns]
        x = np.column_stack([x, extra])
    return x, y, p, e


def probability_transform(scores, prior, beta):
    scores = np.asarray(scores, dtype=float) - beta * np.log(np.maximum(prior, 1e-8))
    scores -= scores.max(1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(1, keepdims=True)


def calibrate(logits, y, pids, prepared, prior):
    group = subset_labels(pids, prepared)
    cal = group == "calibration"
    sel = ~cal
    candidates = []
    for beta in cfg()["bias_betas"]:
        pred = probability_transform(logits, prior, beta).argmax(1)
        cm = np.bincount(y[cal] * 5 + pred[cal], minlength=25).reshape(5, 5)
        candidates.append((ba_from_cm(cm), -abs(beta), beta))
    beta = max(candidates)[2]
    pred = probability_transform(logits, prior, beta).argmax(1)
    return {
        "beta": beta,
        "calibration": report(y[cal], pred[cal], pids[cal]),
        "selection": report(y[sel], pred[sel], pids[sel]),
        "full_validation": report(y, pred, pids),
    }


def choose_beta_across_levels(evidence, prepared, prior):
    # One shared beta for all levels; never choose a separate correction for each noise condition.
    values = []
    for beta in cfg()["bias_betas"]:
        scores = []
        for logits, y, pids in evidence.values():
            take = subset_labels(pids, prepared) == "calibration"
            pred = probability_transform(logits[take], prior, beta).argmax(1)
            scores.append(
                ba_from_cm(np.bincount(y[take] * 5 + pred, minlength=25).reshape(5, 5))
            )
        values.append((float(np.mean(scores)), -abs(beta), beta))
    beta = max(values)[2]
    reports = {}
    for level, (logits, y, pids) in evidence.items():
        pred = probability_transform(logits, prior, beta).argmax(1)
        group = subset_labels(pids, prepared)
        reports[str(level)] = {
            name: report(y[group == name], pred[group == name], pids[group == name])
            for name in ("calibration", "selection")
        }
        reports[str(level)]["full_validation"] = report(y, pred, pids)
    return (
        beta,
        reports,
        float(np.mean([r["selection"]["balanced_accuracy"] for r in reports.values()])),
    )


def save_evidence(out, evidence):
    for level, (scores, y, pids, epochs) in evidence.items():
        temporary = out / f"validation_logits_{level}ms.tmp.npz"
        np.savez_compressed(temporary, scores=scores, y=y, pids=pids, epochs=epochs)
        temporary.replace(out / f"validation_logits_{level}ms.npz")
