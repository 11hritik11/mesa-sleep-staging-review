"""Saved-result reports: balanced accuracy, per-stage recall and all optimization histories."""

import csv, html, zipfile
from pathlib import Path
import numpy as np
from balanced_utils import read_json, save, cfg, base, STAGES
import reporting as legacy
import matplotlib.pyplot as plt


def export(run, name, fig):
    fig.text(
        0.5,
        0.005,
        "Five stages; participant-separated; previously inspected test is exploratory",
        ha="center",
        fontsize=8,
        color="#744",
    )
    legacy.export(run, name, fig)


def bootstrap(report, seed=20261103):
    cms = np.array([v["confusion_matrix"] for v in report["per_participant"].values()])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        cm = cms[rng.integers(0, len(cms), len(cms))].sum(0)
        den = cm.sum(1)
        if np.all(den > 0):
            values.append(float(np.mean(np.diag(cm) / den)))
    return np.quantile(values, [0.025, 0.975]).tolist()


def build_report(run, prepared):
    results = read_json(run / "validation_results.json")
    selection = read_json(run / "selection.json")["tracks"]
    test = read_json(run / "test_results.json")
    audit = read_json(prepared / "audit.json")
    base_manifest = read_json(base() / "manifest.json")
    save(run / "coverage.json", read_json(base() / "coverage.json"))
    save(run / "signal_audit.json", audit)
    save(run / "base_preparation_manifest.json", base_manifest)
    lines = [
        "# Balanced-accuracy MESA comparison",
        "",
        "**Target: 70% mean balanced accuracy across 0/20/40/60 ms. This is a target, not a guaranteed outcome.**",
        "",
        "Five classes are retained: Wake, N1, N2, N3, REM. Balanced accuracy is mean stage recall. Macro-F1 and ordinary accuracy are separate metrics. Model selection and class-bias correction use disjoint sets of validation participants. The reference test has already been inspected during earlier development, so all new comparisons are exploratory. No claim of measured BCG validation is made.",
        "",
        "## Selected results",
        "",
        "| Inputs | Selected method | Selection balanced accuracy | Reference test mean balanced accuracy | Target met on reference? |",
        "|---|---|---:|---:|---|",
    ]
    for modality, s in selection.items():
        score = float(
            np.mean([r["common"]["balanced_accuracy"] for r in test[modality].values()])
        )
        lines.append(
            f"| {modality} | {s['winner']} | {s['validation_balanced_accuracy']:.4f} | {score:.4f} | {'Yes' if score>=.7 else 'No'} |"
        )
    lines += [
        "",
        "Beat-only, ECG waveform-derived breathing and extra-sensor results must be compared as different input settings. Respiration, oxygenation and leg movement remain unperturbed while simulated jitter affects RR inputs only; these settings cannot be described as beat-only BCG performance.",
        "",
        "## Complete validation comparison",
        "",
        "| Candidate | Inputs | Balanced accuracy | Class-bias beta |",
        "|---|---|---:|---:|",
    ]
    ordered = sorted(results, key=lambda n: -results[n]["selection_score"])
    for name in ordered:
        r = results[name]
        lines.append(
            f"| {name} | {r['specification']['modality']} | {r['selection_score']:.4f} | {r['beta']} |"
        )
    with (run / "validation_comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "candidate",
                "modality",
                "selection_balanced_accuracy",
                "beta",
                "training_targets",
            ]
        )
        w.writerows(
            (
                n,
                results[n]["specification"]["modality"],
                results[n]["selection_score"],
                results[n]["beta"],
                results[n].get("training_targets"),
            )
            for n in ordered
        )
    for modality in ("beat", "edr", "multi"):
        names = [
            n for n in ordered if results[n]["specification"]["modality"] == modality
        ]
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
        ax.barh(
            names[::-1],
            [results[n]["selection_score"] for n in names[::-1]],
            color=[
                "#e78b28" if n == selection[modality]["winner"] else "#4477aa"
                for n in names[::-1]
            ],
        )
        ax.axvline(0.7, color="red", ls="--", label="70% target")
        ax.set(
            xlim=(0, 1),
            xlabel="Mean five-class selection-validation balanced accuracy",
            title=modality,
        )
        ax.legend()
        export(run, "comparison_" + modality, fig)
    for name, r in results.items():
        h = r.get("history", [])
        if not h:
            continue
        save(run / f"training_history_{name}.json", h)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for prefix in ("train", "validation"):
            rows = [v for v in h if prefix + "_log_loss" in v]
            if rows:
                axes[0].plot(
                    [v["step"] for v in rows],
                    [v[prefix + "_log_loss"] for v in rows],
                    label=prefix,
                )
            rows = [v for v in h if prefix + "_balanced_accuracy" in v]
            if rows:
                axes[1].plot(
                    [v["step"] for v in rows],
                    [v[prefix + "_balanced_accuracy"] for v in rows],
                    label=prefix,
                )
        axes[0].set(ylabel="Recorded loss (see objective notes)", xlabel=h[0]["axis"])
        axes[0].legend()
        axes[1].set(ylabel="Balanced accuracy", xlabel=h[0]["axis"], ylim=(0, 1))
        axes[1].axhline(0.7, color="red", ls="--")
        if axes[1].lines:
            axes[1].legend()
        fig.suptitle(name)
        export(run, "learning_" + name, fig)
    uncertainty = {}
    probability = {}
    for modality, tracks in test.items():
        folder = run / "tracks" / modality
        folder.mkdir(exist_ok=True, parents=True)
        uncertainty[modality] = {}
        probability[modality] = {}
        lines += [
            "",
            f"## {modality}: common reference results",
            "",
            "| MAE | Accuracy | Balanced accuracy | Macro-F1 | Kappa |",
            "|---|---:|---:|---:|---:|",
        ]
        with (folder / "metrics.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "noise_ms",
                    "coverage",
                    "epochs",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "kappa",
                ]
                + [s + "_recall" for s in STAGES]
                + [s + "_f1" for s in STAGES]
            )
            for level in (0, 20, 40, 60):
                for coverage in ("common", "expanded"):
                    r = tracks[str(level)][coverage]
                    w.writerow(
                        [
                            level,
                            coverage,
                            r["n_epochs"],
                            r["accuracy"],
                            r["balanced_accuracy"],
                            r["macro_f1"],
                            r["kappa"],
                        ]
                        + [r["per_stage"][s]["recall"] for s in STAGES]
                        + [r["per_stage"][s]["f1"] for s in STAGES]
                    )
                r = tracks[str(level)]["common"]
                k = "undefined" if r["kappa"] is None else f"{r['kappa']:.4f}"
                lines.append(
                    f"| {level} ms | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {k} |"
                )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for coverage in ("common", "expanded"):
            rr = [tracks[str(n)][coverage] for n in (0, 20, 40, 60)]
            point = [r["balanced_accuracy"] for r in rr]
            bounds = np.array([bootstrap(r) for r in rr])
            axes[0].plot([0, 20, 40, 60], point, "o-", label=coverage)
            axes[0].vlines([0, 20, 40, 60], bounds[:, 0], bounds[:, 1], alpha=0.5)
            uncertainty[modality][coverage] = {
                str(n): bounds[k].tolist() for k, n in enumerate((0, 20, 40, 60))
            }
        axes[0].axhline(0.7, color="red", ls="--")
        axes[0].set(
            xlabel="Target interval MAE (ms)",
            ylabel="Balanced accuracy",
            ylim=(0, 1),
            title="95% participant bootstrap intervals",
        )
        axes[0].legend()
        for level in (0, 20, 40, 60):
            r = tracks[str(level)]["common"]
            axes[1].plot(
                STAGES,
                [r["per_stage"][s]["recall"] for s in STAGES],
                "o-",
                label=f"{level}ms",
            )
            legacy.confusion(folder, "reference_common", level, r)
        axes[1].set(ylabel="Stage recall", ylim=(0, 1))
        axes[1].legend()
        fig.suptitle(modality)
        export(folder, "robustness_and_recall", fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(
            [
                [
                    np.mean(
                        np.diag(np.array(v["confusion_matrix"]))
                        / np.maximum(np.array(v["confusion_matrix"]).sum(1), 1)
                    )
                    for v in tracks[str(n)]["common"]["per_participant"].values()
                    if np.all(np.array(v["confusion_matrix"]).sum(1) > 0)
                ]
                for n in (0, 20, 40, 60)
            ]
        )
        ax.set(
            xticks=range(1, 5),
            xticklabels=["0ms", "20ms", "40ms", "60ms"],
            ylabel="Five-class participant balanced accuracy",
            ylim=(0, 1),
            title="Only participants with all five reference stages",
        )
        export(folder, "participant_variation", fig)
        importance = read_json(folder / "importance_validation.json")
        fig, ax = plt.subplots(figsize=(9, max(4, len(importance) * 0.35)))
        ax.barh(
            [r["group"] for r in importance[::-1]],
            [r["balanced_accuracy_drop"] for r in importance[::-1]],
        )
        ax.axvline(0, color="gray")
        ax.set(
            xlabel="Selection-validation balanced-accuracy decrease",
            title="Grouped masking ablation, 40ms; not causal importance",
        )
        export(folder, "feature_group_ablation", fig)
        for level in (0, 20, 40, 60):
            with (folder / f"test_predictions_{level}ms.csv").open() as f:
                rows = [r for r in csv.DictReader(f) if int(r["common"])]
            probability[modality][str(level)] = legacy.probability_plots(
                folder, level, rows
            )
            for pid in sorted({r["participant"] for r in rows})[:2]:
                record = sorted(
                    [r for r in rows if r["participant"] == pid],
                    key=lambda r: int(r["epoch"]),
                )
                epochs = np.array([int(r["epoch"]) for r in record])
                timeline = np.arange(epochs.min(), epochs.max() + 1)
                fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 4))
                for ax, key in zip(axes, ("reference", "prediction")):
                    values = np.full(len(timeline), np.nan)
                    values[epochs - timeline[0]] = [int(r[key]) for r in record]
                    ax.step(timeline / 120, values, where="post")
                    ax.set(
                        yticks=range(5),
                        yticklabels=STAGES,
                        ylim=(4.5, -0.5),
                        ylabel=key,
                    )
                axes[1].set_xlabel("Hours from recording start; gaps excluded")
                fig.suptitle(f"{modality}: participant {pid}, {level}ms")
                export(folder, f"hypnogram_{pid}_{level}ms", fig)
    save(run / "test_uncertainty.json", uncertainty)
    save(run / "probability_diagnostics.json", probability)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, part in enumerate(("train", "validation", "test")):
        for j, key in enumerate(("eligible", "common", "expanded")):
            axes[0].bar(
                k + (j - 1) * 0.25,
                base_manifest["counts"][part][key],
                0.25,
                label=key if k == 0 else None,
            )
    axes[0].set(
        xticks=range(3),
        xticklabels=["Train", "Validation", "Test"],
        ylabel="Epochs",
        title="Coverage retained",
    )
    axes[0].legend()
    sensors = ["Thor", "Abdo", "Flow", "SpO2", "Leg"]
    axes[1].bar(
        sensors, [sum(r["channels"].get(s, False) for r in audit) for s in sensors]
    )
    axes[1].set(ylabel="Recordings", title="Additional channel availability")
    export(run, "coverage", fig)
    source_coverage = read_json(base() / "coverage.json")
    if source_coverage:
        means = [
            np.mean([r["realized_mae_ms"][str(n)] for r in source_coverage])
            for n in (0, 20, 40, 60)
        ]
        stds = [
            np.std([r["realized_mae_ms"][str(n)] for r in source_coverage])
            for n in (0, 20, 40, 60)
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar([0, 20, 40, 60], means, yerr=stds, fmt="o")
        ax.plot([0, 60], [0, 60], "k:")
        ax.set(
            xlabel="Target RR interval MAE (ms)",
            ylabel="Realized MAE (ms)",
            title="Jitter audit; error bars = participant SD",
        )
        export(run, "jitter_audit", fig)
    lines += [
        "",
        "## Methods and limits",
        "",
        "Features: original140 beat HRV/context values; ECG-derived respiration adds36; measured breathing, oxygenation and leg movement bring the total to326 in the multi-sensor track. Complete feature definitions, signal provenance, quality rules, methods and literature are in source/bcg_balanced_staging/PROTOCOL.md. All usable training epochs are used. No labels or scored events enter input features. No EEG, EOG or chin EMG inputs are used.",
        "",
        "Class-frequency logit correction has one beta per candidate across all four noise levels, fitted only on calibration-validation participants. Selection participants rank model checkpoints, base models and prespecified equal-weight ensembles. These repeated development comparisons require external confirmation. Softmax-transformed SVM scores and class-weighted predictions need not be calibrated; reliability plots show their behavior.",
        "",
        "Neural histories use complete training passes; trees count estimators and boosting histories count rounds. Classical training diagnostics use up to5000 rows while fitting uses all training rows. CatBoost learn loss is its weighted training objective; its validation loss is unweighted. Other displayed diagnostic losses are unweighted. These are optimization histories, not training-set-size learning curves.",
        "",
        "Per-participant five-class balanced accuracy is undefined when a stage is absent; aggregate balanced accuracy requires all five stages. Undefined kappa is null with its explanation. Group masking is a sensitivity diagnostic and may create out-of-distribution missingness. Bootstrap intervals resample participants, not epochs.",
        "",
        "## Figures",
        "",
    ]
    images = sorted(run.glob("figures/*.png")) + sorted(
        run.glob("tracks/*/figures/*.png")
    )
    for p in images:
        lines += [
            f'### {p.stem.replace("_"," ")}',
            f"![{p.stem}]({p.relative_to(run)})",
            "",
        ]
    (run / "REPORT.md").write_text("\n".join(lines))
    body = "".join(
        f'<h2>{html.escape(str(p.relative_to(run)))}</h2><img loading="lazy" src="{p.relative_to(run)}">'
        for p in images
    )
    (run / "REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>MESA balanced accuracy</title><style>body{max-width:1200px;margin:32px auto;font:16px system-ui}img{width:100%}</style><h1>Balanced-accuracy comparison</h1><p>Five stages; separate sensor tracks. Previously inspected test: exploratory. Read REPORT.md for tables and limitations.</p>'
        + body
    )
    with zipfile.ZipFile(run / "results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.rglob("*")):
            if (
                p.is_file()
                and p.suffix
                in (
                    ".json",
                    ".csv",
                    ".md",
                    ".html",
                    ".png",
                    ".pdf",
                    ".py",
                    ".slurm",
                    ".txt",
                )
                and not p.name.endswith(".incomplete.csv")
            ):
                z.write(p, p.relative_to(run))
