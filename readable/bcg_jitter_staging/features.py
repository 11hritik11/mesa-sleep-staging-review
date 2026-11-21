"""Explicit, versioned ECG-derived RR features for simulated BCG timing errors."""

import csv
import numpy as np
from scipy.signal import welch
from scipy.spatial import cKDTree
from wfdb.processing import xqrs_detect

FS = 256
BASE_NAMES = [
    "mean_rr_s",
    "median_rr_s",
    "sdnn_s",
    "iqr_rr_s",
    "mad_rr_s",
    "cv_rr",
    "mean_hr_bpm",
    "sd_hr_bpm",
    "rmssd_s",
    "sdsd_s",
    "pnn20",
    "pnn50",
    "poincare_sd1_s",
    "poincare_sd2_s",
    "poincare_ratio",
    "sample_entropy",
    "rr_skewness",
    "rr_excess_kurtosis",
    "lf_power_s2",
    "hf_power_s2",
    "lf_hf_ratio",
    "lf_fraction",
    "hf_fraction",
    "lf_peak_hz",
    "hf_peak_hz",
    "spectral_entropy",
    "n_intervals",
    "plausible_rr_fraction",
    "coverage_fraction",
    "valid_ecg_fraction",
]
CONTEXT_BASE = [
    "mean_rr_s",
    "sdnn_s",
    "rmssd_s",
    "mean_hr_bpm",
    "lf_power_s2",
    "hf_power_s2",
    "lf_hf_ratio",
    "sample_entropy",
]
FEATURE_NAMES = BASE_NAMES + [
    f"context_{stat}_{name}"
    for stat in ("mean", "std", "delta")
    for name in CONTEXT_BASE
]


def read_annotations(path, duration, return_audit=False):
    """Only RPoint and Type are inputs. Annotation sleep-stage columns are ignored."""
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"RPoint", "Type"} <= set(reader.fieldnames):
            raise ValueError(f"Missing RPoint/Type columns in {path}")
        rows = [(float(row["RPoint"]) / FS, int(float(row["Type"]))) for row in reader]
    times = np.array([r[0] for r in rows], dtype=np.float64)
    normal = np.array([r[1] == 1 for r in rows], dtype=bool)
    reordered = int(np.sum(np.diff(times) < 0))
    # Some official exports contain valid beats in nonchronological row order.
    # Sort time/type pairs together before forming adjacent-beat intervals.
    order = np.argsort(times, kind="stable")
    times, normal = times[order], normal[order]
    if (
        len(times) < 100
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or times[0] < 0
        or times[-1] > duration + 30
    ):
        raise ValueError(f"Invalid annotation times in {path}")
    keep = times < duration
    times, normal = times[keep], normal[keep]
    result = (times, normal[:-1] & normal[1:])
    if return_audit:
        return (
            *result,
            {
                "source_rows": len(rows),
                "source_backward_transitions": reordered,
                "chronological_sort_applied": bool(reordered),
                "trailing_beats_outside_prepared_epochs": int((~keep).sum()),
            },
        )
    return result


def contiguous_runs(valid):
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def detect_beats(windows, valid):
    """Adaptive XQRS on continuous valid runs, in overlapping 10-minute blocks."""
    beats = []
    for start, end in contiguous_runs(valid):
        for block in range(start, end, 20):
            left, right = max(start, block - 1), min(end, block + 21)
            x = np.array(windows[left:right], dtype=np.float64).reshape(-1)
            inds = xqrs_detect(x, FS, verbose=False)
            times = left * 30 + inds / FS
            # Disjoint 10-minute ownership; overlap only supplies detector context.
            beats.extend(
                times[(times >= block * 30) & (times < min(end, block + 20) * 30)]
            )
    return np.unique(np.asarray(beats, dtype=np.float64))


def sampen(rr):
    """m=2, Chebyshev distance, r=0.2*SD; equal-length template sets."""
    sd = np.std(rr)
    if len(rr) < 30 or sd <= 1e-12:
        return np.nan
    template = np.lib.stride_tricks.sliding_window_view(rr, 3)
    b = len(cKDTree(template[:, :2]).query_pairs(0.2 * sd, p=np.inf))
    a = len(cKDTree(template).query_pairs(0.2 * sd, p=np.inf))
    return float(-np.log(a / b)) if a and b else np.nan


def base_features(peaks, count, input_valid, anchors=None, normal_intervals=None):
    answer = np.full((count, len(BASE_NAMES)), np.nan, dtype=np.float32)
    rr = np.diff(peaks)
    times = peaks[1:] if anchors is None else anchors
    good = (rr >= 0.3) & (rr <= 2.0)
    if normal_intervals is not None:
        good &= normal_intervals
    for epoch in range(5, count - 5):
        start, end = epoch * 30 - 135, epoch * 30 + 165
        lo, hi = np.searchsorted(times, [start, end])
        raw, t, mask = rr[lo:hi], times[lo:hi], good[lo:hi]
        valid_fraction = np.mean(input_valid[epoch - 5 : epoch + 6])
        if (
            len(raw) < 80
            or mask.mean() < 0.9
            or valid_fraction < 1
            or t[-1] - t[0] < 270
            or np.max(np.diff(t)) > 3
        ):
            continue
        r, t = raw[mask], t[mask]
        if len(r) < 80 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > 3:
            continue
        # Successive-interval statistics never bridge a rejected interval.
        d = np.diff(raw)[mask[1:] & mask[:-1]]
        if len(d) < 20:
            continue
        mean, sd = np.mean(r), np.std(r, ddof=1)
        sdsd = np.std(d, ddof=1)
        sd1 = sdsd / np.sqrt(2)
        sd2 = np.sqrt(max(0, 2 * sd**2 - 0.5 * sdsd**2))
        scaled = (r - mean) / max(sd, 1e-12)
        # Standard tachogram PSD: interpolate at 4 Hz, subtract mean, Welch Hann windows.
        regular_t = np.arange(t[0], t[-1], 0.25)
        tachogram = np.interp(regular_t, t, r)
        f, psd = welch(
            tachogram,
            fs=4,
            window="hann",
            nperseg=min(1024, len(tachogram)),
            detrend="constant",
            scaling="density",
        )
        lf_mask, hf_mask = (f >= 0.04) & (f < 0.15), (f >= 0.15) & (f <= 0.4)
        lf, hf = np.trapz(psd[lf_mask], f[lf_mask]), np.trapz(psd[hf_mask], f[hf_mask])
        band = (f >= 0.04) & (f <= 0.4)
        p = psd[band] / max(psd[band].sum(), 1e-20)
        runs = list(contiguous_runs(mask))
        a, b = max(runs, key=lambda pair: pair[1] - pair[0])
        value = [
            mean,
            np.median(r),
            sd,
            np.subtract(*np.percentile(r, [75, 25])),
            np.median(np.abs(r - np.median(r))),
            sd / mean,
            np.mean(60 / r),
            np.std(60 / r),
            np.sqrt(np.mean(d**2)),
            sdsd,
            np.mean(np.abs(d) > 0.02),
            np.mean(np.abs(d) > 0.05),
            sd1,
            sd2,
            sd1 / max(sd2, 1e-12),
            sampen(raw[a:b]),
            np.mean(scaled**3),
            np.mean(scaled**4) - 3,
            lf,
            hf,
            lf / max(hf, 1e-12),
            lf / max(lf + hf, 1e-12),
            hf / max(lf + hf, 1e-12),
            f[lf_mask][np.argmax(psd[lf_mask])],
            f[hf_mask][np.argmax(psd[hf_mask])],
            -np.sum(p * np.log(np.maximum(p, 1e-20))) / np.log(len(p)),
            len(raw),
            mask.mean(),
            (t[-1] - t[0]) / 300,
            valid_fraction,
        ]
        answer[epoch] = value
    answer[~np.isfinite(answer)] = np.nan
    return answer


def add_context(base):
    """Five nearby epochs: finite-only means/SDs and next-minus-previous changes."""
    selected = base[:, [BASE_NAMES.index(name) for name in CONTEXT_BASE]]
    means, stds = np.full_like(selected, np.nan), np.full_like(selected, np.nan)
    deltas = np.full_like(selected, np.nan)
    for i in range(len(base)):
        chunk = selected[max(0, i - 2) : i + 3]
        for j in range(len(CONTEXT_BASE)):
            values = chunk[:, j][np.isfinite(chunk[:, j])]
            if len(values):
                means[i, j], stds[i, j] = values.mean(), values.std()
        if 0 < i < len(base) - 1:
            deltas[i] = selected[i + 1] - selected[i - 1]
    return np.column_stack([base, means, stds, deltas]).astype(np.float32)
