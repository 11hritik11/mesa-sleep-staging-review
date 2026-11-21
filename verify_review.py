"""Check public aggregate arithmetic and archived source hashes; Python >=3.9.

This verifies aggregate confusion matrices, not individual prediction rows.
Use the separately shared full review ZIP to verify individual predictions.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ("Wake", "N1", "N2", "N3", "REM")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    check(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), message)


def verify_metrics(report):
    cm = report["confusion_matrix"]
    check(len(cm) == 5 and all(len(row) == 5 for row in cm), "Matrix shape")
    check(all(type(x) is int and x >= 0 for row in cm for x in row), "Matrix counts")
    rows = [sum(row) for row in cm]
    cols = [sum(cm[i][j] for i in range(5)) for j in range(5)]
    n = sum(rows)
    check(n == report["n_epochs"] and n > 0 and all(rows), "Population counts")
    accuracy = sum(cm[i][i] for i in range(5)) / n
    recalls = [cm[i][i] / rows[i] for i in range(5)]
    f1 = [2 * cm[i][i] / (rows[i] + cols[i]) for i in range(5)]
    close(accuracy, report["accuracy"], "Accuracy")
    close(sum(recalls) / 5, report["balanced_accuracy"], "Balanced accuracy")
    close(sum(f1) / 5, report["macro_f1"], "Macro F1")
    pe = sum(rows[i] * cols[i] for i in range(5)) / n ** 2
    if abs(1 - pe) > 1e-15:
        close((accuracy - pe) / (1 - pe), report["kappa"], "Kappa")
    else:
        check(report["kappa"] is None, "Undefined kappa")
    for i, stage in enumerate(STAGES):
        saved = report["per_stage"][stage]
        check(saved["support"] == rows[i], "Stage support")
        close(recalls[i], saved["recall"], "Stage recall")
        close(cm[i][i] / cols[i] if cols[i] else 0, saved["precision"], "Stage precision")
        close(f1[i], saved["f1"], "Stage F1")
    return sum(recalls) / 5


def main():
    checks = json.loads((ROOT / "source_checksums.json").read_text())
    for name, expected in checks.items():
        actual = hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest()
        check(actual == expected, "Source SHA-256: " + name)
    results = json.loads((ROOT / "aggregate_results.json").read_text())
    means = {}
    for track in ("beat", "edr", "multi"):
        common = []
        for level in ("0", "20", "40", "60"):
            for population, epochs in (("common", 194310), ("expanded", 281712)):
                report = results[track][level][population]
                check(report["n_epochs"] == epochs, "Epoch count")
                score = verify_metrics(report)
                if population == "common":
                    common.append(score)
        means[track] = sum(common) / 4
    check([round(means[k] * 100, 2) for k in ("beat", "edr", "multi")]
          == [64.19, 63.95, 71.15], "Presentation headline comparison")
    selection = json.loads((ROOT / "selection_summary.json").read_text())
    check(set(selection["multi"]["members"]) ==
          {"bigru_multisensor", "tcn_multisensor", "lightgbm_multi"}, "Ensemble membership")
    check(selection["multi"]["beta"] == 0, "Ensemble correction")
    print(json.dumps({"status": "PASS", "source_files_verified": len(checks),
                      "scope": "Aggregate arithmetic and archived source consistency",
                      "individual_predictions_verified": False,
                      "reference_mean_balanced_accuracy": means}, indent=2))


if __name__ == "__main__":
    main()
