#!/usr/bin/env python3
"""Audit XML/ECG identity and extract ECG-derived breathing / peripheral sensor features."""
import argparse, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pyedflib
from scipy.signal import resample_poly, butter, sosfiltfilt, welch, find_peaks
from balanced_utils import (
    ROOT,
    base,
    cfg,
    read_json,
    save,
    verify_prepared,
    relative_features,
    EXTRA_RAW,
    EXTRA_NAMES,
    fingerprint,
    snapshot,
)
from common import read_annotations, digest


def respiratory_features(signal, epochs, hz=4):
    answer = np.full((epochs, 18), np.nan, dtype=np.float32)
    for epoch in range(epochs):
        center = epoch * 30 + 15
        for k, window in enumerate((30, 120, 300)):
            lo = max(0, int((center - window / 2) * hz))
            hi = min(len(signal), int((center + window / 2) * hz))
            x = signal[lo:hi]
            good = np.isfinite(x)
            offset = k * 6
            answer[epoch, offset + 5] = float(good.sum() / (window * hz))
            if good.sum() < max(40, 0.8 * window * hz):
                continue
            x = np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])
            if np.std(x) < 1e-8:
                continue
            filtered = sosfiltfilt(
                butter(2, [0.08, 0.6], fs=hz, btype="band", output="sos"), x
            )
            frequencies, power = welch(filtered, fs=hz, nperseg=min(512, len(x)))
            band = (frequencies >= 0.08) & (frequencies <= 0.6)
            full = (frequencies >= 0.04) & (frequencies <= 1)
            values = power[band]
            total = values.sum()
            if total <= 1e-15:
                continue
            normalized = values / total
            peaks, _ = find_peaks(
                filtered, distance=int(hz), prominence=max(np.std(filtered) * 0.2, 1e-8)
            )
            intervals = np.diff(peaks) / hz
            answer[epoch, offset : offset + 5] = [
                frequencies[band][values.argmax()],
                -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)))
                / np.log(len(normalized)),
                power[(frequencies >= 0.1) & (frequencies <= 0.4)].sum()
                / max(power[full].sum(), 1e-15),
                np.subtract(*np.percentile(filtered, [75, 25])),
                (
                    np.std(intervals) / np.mean(intervals)
                    if len(intervals) >= 3
                    else np.nan
                ),
            ]
    return answer


def simple_features(signal, epochs, hz, kind):
    width = 4 if kind == "spo2" else 3
    answer = np.full((epochs, width * 3), np.nan, dtype=np.float32)
    if kind == "leg":
        good = np.isfinite(signal)
        threshold = np.nanmedian(np.abs(signal)) + 2 * np.nanstd(signal)
    for epoch in range(epochs):
        center = epoch * 30 + 15
        for k, window in enumerate((30, 120, 300)):
            lo = max(0, int((center - window / 2) * hz))
            hi = min(len(signal), int((center + window / 2) * hz))
            x = signal[lo:hi]
            x = x[np.isfinite(x)]
            if len(x) < 0.8 * window * hz:
                continue
            if kind == "spo2":
                values = [x.mean(), x.min(), x.std(), np.mean(x < 90)]
            else:
                values = [
                    np.sqrt(np.mean(x * x)),
                    np.subtract(*np.percentile(x, [75, 25])),
                    np.mean(np.abs(x) > threshold),
                ]
            answer[epoch, k * width : (k + 1) * width] = values
    return answer


def extract(task):
    out, part, pid, version = task
    source = base() / "records" / f"{part}_{pid}.npz"
    edf = ROOT / "data/raw/mesa/polysomnography/edfs" / f"mesa-sleep-{pid}.edf"
    xml = (
        ROOT
        / "data/raw/mesa/polysomnography/annotations-events-nsrr"
        / f"mesa-sleep-{pid}-nsrr.xml"
    )
    signature = hashlib.sha256(
        (
            version
            + digest(source)
            + str(edf.stat().st_size)
            + str(edf.stat().st_mtime_ns)
            + digest(xml)
        ).encode()
    ).hexdigest()
    target = out / "extra" / f"{part}_{pid}.npz"
    if target.exists():
        with np.load(target) as f:
            if str(f["signature"]) != signature:
                raise ValueError("Changed extraction input " + pid)
            return json.loads(str(f["audit_json"]))
    with np.load(source) as f:
        y = f["y"]
        common = f["common"]
        expanded = f["expanded"]
    # Re-read independent original XML labels and verify EDF/XML clocks and origin.
    sys.path.insert(0, str(ROOT / "ecg_staging"))
    from prepare_full_cohort import read_labels

    with pyedflib.EdfReader(str(edf)) as reader:
        labels, clock = read_labels(xml, edf, reader.file_duration)
        if len(labels) != len(y) or not np.array_equal(labels[y >= 0], y[y >= 0]):
            raise ValueError("Stage alignment mismatch " + pid)
        names = reader.getSignalLabels()
        channels = {
            s: names.index(s)
            for s in ("Thor", "Abdo", "Flow", "SpO2", "Leg")
            if s in names
        }
        raw = []
        availability = {}
        for name in ("Thor", "Abdo", "Flow"):
            if name not in channels:
                raw.append(np.full((len(y), 18), np.nan))
                availability[name] = False
                continue
            i = channels[name]
            rate = reader.getSampleFrequency(i)
            values = reader.readSignal(i)
            if rate != 32:
                raise ValueError("Unexpected respiratory sample rate " + pid)
            values[~np.isfinite(values)] = np.nan
            if np.isfinite(values).mean() < 0.8:
                signal = np.full(len(values) // 8, np.nan)
            else:
                finite = np.isfinite(values)
                signal = resample_poly(
                    np.nan_to_num(values, nan=float(np.nanmedian(values))), 1, 8
                )
                # Restore missingness after anti-alias filtering; imputation must not create valid coverage.
                usable = (
                    np.convolve(finite.astype(float), np.ones(8) / 8, mode="same")[::8]
                    >= 0.8
                )
                signal[: len(usable)][~usable[: len(signal)]] = np.nan
            raw.append(respiratory_features(signal, len(y)))
            availability[name] = True
        rest = []
        for name, kind in [("SpO2", "spo2"), ("Leg", "leg")]:
            if name not in channels:
                rest.append(np.full((len(y), 12 if kind == "spo2" else 9), np.nan))
                availability[name] = False
                continue
            i = channels[name]
            rate = reader.getSampleFrequency(i)
            values = reader.readSignal(i)
            if kind == "spo2":
                values[(values < 50) | (values > 100)] = np.nan
            rest.append(simple_features(values, len(y), rate, kind))
            availability[name] = True
    # ECG-derived respiration uses QRS amplitude and is an explicit ECG-waveform branch.
    peaks, normal = read_annotations(
        ROOT
        / "data/raw/mesa/polysomnography/annotations-rpoints"
        / f"mesa-sleep-{pid}-rpoint.csv",
        len(y) * 30,
    )
    ecg = np.load(
        ROOT
        / "data/derived/full_cohort_windows_v1/recordings"
        / f"mesa-sleep-{pid}"
        / "ecg_windows.npy",
        mmap_mode="r",
    ).reshape(-1)
    ix = np.rint(peaks[1:][normal] * 256).astype(int)
    times = peaks[1:][normal]
    keep = (ix >= 32) & (ix + 12 < len(ecg))
    ix, times = ix[keep], times[keep]
    qrs = ecg[ix[:, None] + np.arange(-8, 9)].max(1) - ecg[
        ix[:, None] + np.arange(-32, -7)
    ].min(1)
    good = np.isfinite(qrs)
    qrs, times = qrs[good], times[good]
    query = (np.arange(len(y) * 120) + 0.5) / 4
    edr = np.full(len(query), np.nan)
    if len(times) > 2:
        right = np.searchsorted(times, query)
        inside = (right > 0) & (right < len(times))
        r = np.clip(right, 0, len(times) - 1)
        l = np.clip(right - 1, 0, len(times) - 1)
        inside &= times[r] - times[l] <= 3
        edr[inside] = np.interp(query[inside], times, qrs)
    edr_features = respiratory_features(edr, len(y))
    features = np.column_stack([edr_features, *raw, *rest])
    assert features.shape == (len(y), len(EXTRA_RAW))
    features, center, scale = relative_features(features)
    row = {
        "participant": pid,
        "partition": part,
        "common": int(common.sum()),
        "expanded": int(expanded.sum()),
        "xml_identity_verified": True,
        "clock_checks_pass": clock["timing_checks_pass"],
        "channels": availability,
        "edr_finite_fraction": float(np.isfinite(edr).mean()),
        "edf_bytes": edf.stat().st_size,
        "edf_mtime_ns": edf.stat().st_mtime_ns,
        "xml_sha256": digest(xml),
    }
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        x=features,
        center=center,
        scale=scale,
        signature=signature,
        audit_json=json.dumps(row),
    )
    temporary.replace(target)
    return row


def run(args):
    manifest = verify_prepared(base())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "extra").mkdir(exist_ok=True)
    version = fingerprint()
    if (out / "balanced_manifest.json").exists() and read_json(
        out / "balanced_manifest.json"
    )["source_version"] != version:
        raise ValueError("Changed preparation source")
    ids = manifest["participants"]
    rng = np.random.default_rng(cfg()["seed"])
    ordered = rng.permutation(sorted(ids["validation"])).tolist()
    split = int(len(ordered) * cfg()["calibration_fraction"])
    groups = {
        "calibration": sorted(ordered[:split]),
        "selection": sorted(ordered[split:]),
    }
    record = {
        "source_version": version,
        "base_feature_version": manifest["feature_version"],
        "extra_names": EXTRA_NAMES,
        "validation_participants": groups,
        "config": cfg(),
        "base": str(base()),
        "status": "PREPARING",
    }
    save(out / "balanced_manifest.json", record)
    snapshot(out)
    tasks = [
        (out, part, pid, version) for part, people in ids.items() for pid in people
    ]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        for f in as_completed([pool.submit(extract, t) for t in tasks]):
            rows.append(f.result())
            save(out / "audit.json", rows)
            if len(rows) % 20 == 0:
                print("EXTRACTED", len(rows), "/", len(tasks), flush=True)
    arrays = []
    for pid in ids["train"]:
        with np.load(base() / "records" / f"train_{pid}.npz") as b, np.load(
            out / "extra" / f"train_{pid}.npz"
        ) as f:
            arrays.append(f["x"][b["expanded"]])
    x = np.concatenate(arrays)
    del arrays
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    for j in range(x.shape[1]):
        v = x[:, j][np.isfinite(x[:, j])]
        if len(v):
            center[j] = np.median(v)
            a, b = np.percentile(v, [25, 75])
            scale[j] = b - a if b - a > 1e-6 else 1
    np.savez(
        out / "extra_scaler.npz",
        center=center.astype("float32"),
        scale=scale.astype("float32"),
    )
    record["status"] = "COMPLETE"
    record["audit_records"] = len(rows)
    save(out / "balanced_manifest.json", record)
    print("PREPARATION COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    run(p.parse_args())
