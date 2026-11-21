"""Shared provenance, normalization, input preparation and full-cohort readers."""
import hashlib
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bcg_jitter_staging"))
from benchmark import report, save, digest, metrics
from experiment import jitter_peaks, load_recording, STAGES, LEVELS_MS
from features import FEATURE_NAMES, read_annotations

LOCAL_NAMES = [f"local_{seconds}s_{name}" for seconds in (30,120) for name in
               ("mean_rr_s","sd_rr_s","mean_hr_bpm","iqr_rr_s","rmssd_s","pnn50","normal_fraction","interval_count")]
RAW_NAMES = FEATURE_NAMES + LOCAL_NAMES
NAMES = RAW_NAMES + ["night_relative_" + name for name in RAW_NAMES]


def read_json(path):
    return json.loads(Path(path).read_text())


def config():
    return read_json(Path(__file__).with_name("config.json"))


def source_hash():
    digestor = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*")):
        if p.is_file() and p.suffix in (".py", ".json", ".md", ".slurm"):
            digestor.update(p.name.encode()); digestor.update(p.read_bytes())
    return digestor.hexdigest()


def snapshot(out):
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source"
    source.mkdir(exist_ok=True)
    for folder, target in ((Path(__file__).parent, source / "bcg_full_training"),
                           (ROOT / "bcg_jitter_staging", source / "bcg_jitter_staging")):
        target.mkdir(exist_ok=True)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in (".py", ".json", ".md", ".slurm"):
                shutil.copy2(path, target / path.name)


def relative_features(raw):
    """Label-free offline per-night median/IQR, computed separately for each jitter level."""
    center, scale = np.zeros(raw.shape[1]), np.ones(raw.shape[1])
    for j in range(raw.shape[1]):
        values = raw[:, j][np.isfinite(raw[:, j])]
        if len(values):
            center[j] = np.median(values)
            q25, q75 = np.percentile(values, [25,75])
            scale[j] = q75-q25 if q75-q25 > 1e-6 else 1.
    normalized = np.clip((raw-center)/scale, -10, 10)
    return np.column_stack([raw, normalized]).astype(np.float32), center, scale


def rr_inputs(peaks, normal, count, valid, cfg, anchors):
    """2-Hz RR input and 30/120-s features, without interpolating across long gaps."""
    rr = np.diff(peaks)
    good = normal & (rr >= .3) & (rr <= 2)
    accepted, times = rr[good], anchors[good]
    samples = int(cfg["rr_hz"] * 30)
    query = (np.arange(count*samples)+.5)/cfg["rr_hz"]
    output = np.zeros((count*samples,3), dtype=np.float32)
    if len(accepted) >= 2:
        right = np.searchsorted(times,query)
        inside = (right>0) & (right<len(times))
        left_ix = np.clip(right-1,0,len(times)-1)
        right_ix = np.clip(right,0,len(times)-1)
        inside &= (times[right_ix]-times[left_ix] <= 3)
        inside &= valid[np.minimum((query/30).astype(int),count-1)]
        values = np.interp(query,times,accepted)
        med = float(np.median(accepted)); q25,q75 = np.percentile(accepted,[25,75])
        spread = max(float(q75-q25), .03)
        output[inside,0] = np.clip((values[inside]-1.)/.3,-10,10)
        output[inside,1] = np.clip((values[inside]-med)/spread,-10,10)
        output[inside,2] = 1
    output = output.reshape(count,samples,3).transpose(0,2,1).copy()
    local = np.full((count,16),np.nan,dtype=np.float32)
    for epoch in range(count):
        if not valid[epoch]:
            continue
        midpoint=epoch*30+15
        for k, width in enumerate(cfg["short_windows_seconds"]):
            lo,hi=np.searchsorted(anchors,[midpoint-width/2,midpoint+width/2])
            raw,mask=rr[lo:hi],good[lo:hi]
            offset=k*8
            local[epoch,offset+6:offset+8]=[float(mask.mean()) if len(mask) else 0.,len(raw)]
            if mask.sum()<5:
                continue
            selected=raw[mask]
            changes=np.diff(raw)[mask[1:] & mask[:-1]]
            local[epoch,offset:offset+6]=[
                selected.mean(),selected.std(ddof=1),np.mean(60/selected),
                np.subtract(*np.percentile(selected,[75,25])),
                np.sqrt(np.mean(changes**2)) if len(changes) else np.nan,
                np.mean(np.abs(changes)>.05) if len(changes) else np.nan]
    return output,local


def verify_prepared(prepared):
    manifest = read_json(prepared / "manifest.json")
    if manifest["status"] != "COMPLETE" or manifest["feature_names"] != NAMES:
        raise ValueError("Incomplete preparation or unexpected feature schema")
    if any(set(manifest["participants"][a]) & set(manifest["participants"][b])
           for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise ValueError("Participant partitions overlap")
    return manifest


def matrix(prepared, part, level=0, augmented=False, mask_name="common", seed=20261102):
    """Every requested target, once; no epoch cap or subsampling."""
    manifest=verify_prepared(prepared)
    xx,yy,pp,ee=[],[],[],[]
    for pid in manifest["participants"][part]:
        with np.load(prepared / "records" / f"{part}_{pid}.npz") as f:
            indices=np.flatnonzero(f[mask_name])
            if not len(indices): continue
            if augmented:
                assigned=np.random.default_rng(seed+int(pid)).integers(0,4,len(indices))
                all_x=f["x"]
                x=all_x[assigned,indices]
            else:
                x=f["x"][LEVELS_MS.index(level),indices]
            xx.append(x); yy.append(f["y"][indices]); pp.extend([pid]*len(indices)); ee.extend(indices.tolist())
    if not xx: raise ValueError(f"No {part} targets")
    return np.concatenate(xx),np.concatenate(yy),np.asarray(pp),np.asarray(ee)


def class_weights(y, power):
    counts=np.bincount(y,minlength=5).astype(float)
    if np.any(counts==0): raise ValueError("All five stages must occur in training")
    weights=(counts.sum()/(5*counts))**power
    return weights / np.average(weights,weights=counts)
