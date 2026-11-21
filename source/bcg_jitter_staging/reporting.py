"""Render saved benchmark evidence as PNG/PDF charts and a portable results bundle."""
import argparse
import csv
import html
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
LEVELS = [0, 20, 40, 60]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 180})


def load(path):
    return json.loads(path.read_text())


def export(run, name, fig):
    folder = run / "figures"
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / (name + ".png"), bbox_inches="tight")
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_history(run, name, history):
    x = [r["step"] for r in history]
    f1_available = "train_macro_f1" in history[0]
    fig, axes = plt.subplots(1, 2 if f1_available else 1, figsize=(11 if f1_available else 7, 4), squeeze=False)
    ax = axes[0, 0]
    for source in ("train", "validation"):
        ax.plot(x, [r[source + "_log_loss"] for r in history], label=source.title())
    ax.set(xlabel=history[0]["axis"], ylabel="Unweighted log loss", title="Loss on fixed monitoring samples")
    ax.legend()
    if f1_available:
        ax = axes[0, 1]
        for source in ("train", "validation"):
            ax.plot(x, [r[source + "_macro_f1"] for r in history], label=source.title())
        ax.set(xlabel=history[0]["axis"], ylabel="Five-stage macro-F1", ylim=(0, 1))
        if name.startswith("mlp"):
            best = x[int(np.argmax([r["validation_clean_macro_f1"] for r in history]))]
            ax.axvline(best, color="gray", linestyle=":", label="Selected epoch")
        ax.legend()
    fig.suptitle(name.replace("_", " ") + "\nTraining monitor ≤5,000 epochs; validation at 0 ms")
    export(run, "learning_" + name, fig)


def bootstrap_interval(report, seed=42):
    cms = np.array([r["confusion_matrix"] for r in report["per_participant"].values()])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(1000):
        cm = cms[rng.integers(0, len(cms), len(cms))].sum(axis=0)
        denominator = cm.sum(0) + cm.sum(1)
        estimates.append(np.divide(2 * np.diag(cm), denominator, out=np.zeros(5), where=denominator > 0).mean())
    return np.quantile(estimates, [.025, .975]).tolist()


def confusion(run, part, level, report):
    cm = np.array(report["confusion_matrix"])
    normalized = np.divide(cm, cm.sum(1)[:, None], out=np.zeros_like(cm, dtype=float), where=cm.sum(1)[:, None] > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, values, title in zip(axes, [cm, normalized], ["Epoch counts", "Fraction within reference stage"]):
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if values is normalized else None)
        fig.colorbar(im, ax=ax, fraction=.046)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=STAGES, yticklabels=STAGES,
               xlabel="Predicted", ylabel="Reference", title=title)
        for i in range(5):
            for j in range(5):
                label = f"{values[i,j]:.2f}" if values is normalized else str(values[i,j])
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if values[i,j] > values.max() / 2 else "black", fontsize=9)
    fig.suptitle(f"{part.title()} — selected model — {level} ms interval error")
    export(run, f"confusion_{part}_{level}ms", fig)


def probability_plots(run, level, rows):
    y = np.array([int(r["reference"]) for r in rows])
    probabilities = np.array([[float(r["p_" + s]) for s in STAGES] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    summary = {}
    for k, stage in enumerate(STAGES):
        binary = y == k
        if binary.all() or not binary.any():
            continue
        fpr, tpr, _ = roc_curve(binary, probabilities[:, k])
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, k])
        auc = float(roc_auc_score(binary, probabilities[:, k]))
        ap = float(average_precision_score(binary, probabilities[:, k]))
        axes[0].plot(fpr, tpr, label=f"{stage}: {auc:.3f}")
        axes[1].plot(recall, precision, label=f"{stage}: {ap:.3f}")
        summary[stage] = {"roc_auc": auc, "average_precision": ap}
    axes[0].plot([0, 1], [0, 1], "k:")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC (legend: AUC)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision–recall (legend: AP)")
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle(f"Selected model — test probabilities at {level} ms")
    export(run, f"roc_pr_test_{level}ms", fig)
    confidence, prediction = probabilities.max(1), np.argmax(probabilities, axis=1)
    centers, accuracy, sizes = [], [], []
    ece = 0.
    for low, high in zip(np.arange(0, 1, .1), np.arange(.1, 1.1, .1)):
        mask = (confidence >= low) & ((confidence < high) if high < .999 else (confidence <= 1))
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            accuracy.append(float((prediction[mask] == y[mask]).mean()))
            sizes.append(int(mask.sum()))
            ece += mask.mean() * abs(accuracy[-1] - centers[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], "k:"); axes[0].plot(centers, accuracy, "o-")
    axes[0].set(xlabel="Mean prediction confidence", ylabel="Observed accuracy", xlim=(0,1), ylim=(0,1), title="Reliability diagram (10 bins)")
    axes[1].hist(confidence, bins=np.linspace(0,1,11))
    axes[1].set(xlabel="Prediction confidence", ylabel="Epoch count", title="Confidence distribution")
    fig.suptitle(f"Test probability diagnostics — {level} ms; ECE={ece:.3f}")
    export(run, f"calibration_test_{level}ms", fig)
    summary["confidence_ece_10_bins"] = float(ece)
    summary["multiclass_brier_score"] = float(np.mean(np.sum((probabilities - np.eye(5)[y]) ** 2, axis=1)))
    summary["reliability_bins"] = {"mean_confidence": centers, "accuracy": accuracy, "n": sizes}
    return summary


def build_report(run):
    run = Path(run)
    validation = load(run / "validation_results.json")
    selection = load(run / "selection.json")
    test = load(run / "test_results.json")["five_stage"]
    winner = selection["winner"]
    for path in sorted(run.glob("training_history_*.json")):
        plot_history(run, path.stem.removeprefix("training_history_"), load(path))
    names = sorted(validation, key=lambda name: -validation[name]["selection_score"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names[::-1], [validation[n]["selection_score"] for n in names[::-1]])
    for bar, name in zip(bars, names[::-1]):
        if name == winner:
            bar.set_color("#e78b28")
    ax.set(xlabel="Mean validation macro-F1 across 0, 20, 40, 60 ms", xlim=(0,1), title="Model comparison; orange = selected model")
    export(run, "validation_model_comparison", fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in names:
        axes[0].plot(LEVELS, [validation[name]["validation"][str(n)]["macro_f1"] for n in LEVELS], label=name)
    axes[0].set(title="All candidates: validation", xlabel="Target interval MAE (ms)", ylabel="Macro-F1", ylim=(0,1))
    axes[0].legend(fontsize=6)
    intervals = {str(level): bootstrap_interval(test[str(level)]) for level in LEVELS}
    point = np.array([test[str(level)]["macro_f1"] for level in LEVELS])
    bounds = np.array([intervals[str(level)] for level in LEVELS])
    # Plot bounds explicitly; percentile intervals need not bracket the point in small samples.
    axes[1].plot(LEVELS, point, "o-", label="Test")
    axes[1].vlines(LEVELS, bounds[:,0], bounds[:,1], color="C0", label="95% participant bootstrap interval")
    axes[1].plot(LEVELS, [validation[winner]["validation"][str(n)]["macro_f1"] for n in LEVELS], "s--", label="Validation")
    axes[1].set(title="Selected model", xlabel="Target interval MAE (ms)", ylabel="Macro-F1", ylim=(0,1))
    axes[1].legend(fontsize=8)
    export(run, "jitter_robustness", fig)
    (run / "test_uncertainty.json").write_text(json.dumps({"method":"1000 participant bootstrap replicates", "macro_f1_95_intervals":intervals}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, part, reports in ((axes[0], "Validation", validation[winner]["validation"]), (axes[1], "Test", test)):
        for level in LEVELS:
            report = reports[str(level)]
            confusion(run, part.lower(), level, report)
            ax.plot(STAGES, [report["per_stage"][s]["f1"] for s in STAGES], "o-", label=f"{level} ms")
        ax.set(title=part, ylabel="Stage F1", ylim=(0,1)); ax.legend()
    export(run, "per_stage_f1", fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([[r["macro_f1"] for r in test[str(n)]["per_participant"].values()] for n in LEVELS])
    ax.set(xticks=range(1,5), xticklabels=[f"{n} ms" for n in LEVELS], ylabel="Participant macro-F1", ylim=(0,1), title="Test performance variation across participants")
    export(run, "participant_variation", fig)
    importance = load(run / "feature_importance_validation.json")
    with (run / "feature_importance.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "mean_macro_f1_drop", "std"])
        writer.writeheader(); writer.writerows(importance)
    top = importance[:25][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([r["feature"] for r in top], [r["mean_macro_f1_drop"] for r in top], xerr=[r["std"] for r in top])
    ax.axvline(0, color="gray", linewidth=.5)
    ax.set(xlabel="Validation macro-F1 decrease after permutation", title="Feature importance at 40 ms; error bars = SD over 3 repeats")
    export(run, "feature_importance_top25", fig)
    coverage = load(run / "coverage.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, part in enumerate(("train", "validation", "test")):
        records = [r for r in coverage if r["partition"] == part]
        eligible, usable = sum(r["eligible"] for r in records), sum(r["usable"] for r in records)
        axes[0].bar(k-.16, eligible, .32, label="Eligible labels" if k==0 else None)
        axes[0].bar(k+.16, usable, .32, label="Usable features" if k==0 else None)
    axes[0].set(xticks=range(3), xticklabels=["Train","Validation","Test"], ylabel="Epoch count", title="Feature coverage"); axes[0].legend()
    realized = [r for r in coverage if "realized_mae_ms" in r]
    if realized:
        means = [np.mean([r["realized_mae_ms"][str(n)] for r in realized]) for n in LEVELS]
        stds = [np.std([r["realized_mae_ms"][str(n)] for r in realized]) for n in LEVELS]
        axes[1].errorbar(LEVELS, means, yerr=stds, fmt="o"); axes[1].plot(LEVELS, LEVELS, "k:")
    axes[1].set(xlabel="Target interval MAE (ms)", ylabel="Realized interval MAE (ms)", title="Jitter check; error bars = participant SD")
    export(run, "coverage_and_jitter", fig)
    diagnostics = {}
    for level in LEVELS:
        with (run / f"test_predictions_{level}ms.csv").open() as file:
            rows = list(csv.DictReader(file))
        diagnostics[str(level)] = probability_plots(run, level, rows)
        # Choose example participants by identifier before inspecting performance.
        examples = sorted({r["participant"] for r in rows})[:3]
        for pid in examples:
            record = [r for r in rows if r["participant"] == pid]
            epochs = np.array([int(r["epoch"]) for r in record])
            timeline = np.arange(epochs.min(), epochs.max() + 1)
            fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
            for ax, field in zip(axes, ["reference", "prediction"]):
                values = np.full(len(timeline), np.nan)
                values[epochs - timeline[0]] = [int(r[field]) for r in record]
                ax.step(timeline / 120, values, where="post")
                ax.set(yticks=range(5), yticklabels=STAGES, ylim=(4.5,-.5), ylabel=field.title())
            axes[1].set_xlabel("Hours from recording start; gaps = excluded epochs")
            fig.suptitle(f"Example participant {pid} — {level} ms")
            export(run, f"hypnogram_{pid}_{level}ms", fig)
    (run / "probability_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    with (run / "validation_comparison.csv").open("w") as file:
        writer = csv.writer(file); writer.writerow(["candidate","mean_validation_macro_f1"])
        writer.writerows((name, validation[name]["selection_score"]) for name in names)
    lines = ["# MESA annotated HRV benchmark", "", f"Selected model: **{winner}**", "",
             "Selection: mean validation five-stage macro-F1 across four jitter levels. Test results did not select the model.", "",
             "## Test results", "", "| Target MAE | Accuracy | Macro-F1 | Kappa |", "|---|---:|---:|---:|"]
    lines += [f"| {n} ms | {test[str(n)]['accuracy']:.3f} | {test[str(n)]['macro_f1']:.3f} | {test[str(n)]['kappa']:.3f} |" for n in LEVELS]
    lines += ["", "## Plot notes", "", "Learning curves use a fixed training monitor of up to 5,000 epochs and the clean validation sample. Loss is unweighted, while some fitting objectives use class weights. Forest curves show performance as trees accumulate; they are not neural training epochs. Selection still uses final validation macro-F1 across all four noise levels. These are iteration curves, not performance-versus-training-set-size experiments.", "", "ROC/PR and reliability plots describe the selected model's test probabilities. Class weighting can affect calibration. Feature importance uses validation only. Bootstrap intervals resample participants, rather than treating neighboring epochs as independent. Example participants are chosen by identifier, without selecting on performance.", "", "## Figures", ""]
    for path in sorted((run / "figures").glob("*.png")):
        lines += [f"### {path.stem.replace('_',' ')}", "", f"![{path.stem}](figures/{path.name})", ""]
    (run / "REPORT.md").write_text("\n".join(lines))
    body = "".join(f'<h2>{html.escape(p.stem.replace("_"," "))}</h2><img loading="lazy" src="figures/{p.name}">' for p in sorted((run / "figures").glob("*.png")))
    (run / "REPORT.html").write_text(f'<!doctype html><meta charset="utf-8"><title>MESA HRV results</title><style>body{{max-width:1200px;margin:32px auto;font:16px system-ui}}img{{width:100%}}</style><h1>MESA HRV benchmark</h1><p>Selected model: {html.escape(winner)}. Full methods and tables: REPORT.md and source/FEATURES.md.</p>{body}')
    with zipfile.ZipFile(run / "results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".md", ".html", ".png", ".pdf", ".py", ".slurm", ".txt"}:
                archive.write(path, path.relative_to(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    build_report(parser.parse_args().run)
