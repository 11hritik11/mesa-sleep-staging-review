#!/usr/bin/env python3
"""Prepare normalized, multiscale and RR inputs; retain common and expanded targets."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from common import (ROOT,NAMES,config,read_json,save,digest,snapshot,relative_features,
                    rr_inputs,read_annotations,load_recording,jitter_peaks,LEVELS_MS)
from benchmark import verified_feature_reuse,extract_one


def prepare_record(task):
    previous_cache,data,out,part,pid,seed,version,cfg=task
    status=extract_one((str(data),str(previous_cache),part,pid,seed,version))
    if not status["cached"]: raise ValueError("Expected existing verified baseline cache")
    target=out / "records" / f"{part}_{pid}.npz"
    prior_signature=digest(previous_cache / f"{part}_{pid}.npz")
    if target.exists():
        with np.load(target) as f:
            if str(f["parent_cache_sha256"]) != prior_signature:
                raise ValueError("Changed parent feature cache")
            return json.loads(str(f["coverage_json"]))
    with np.load(previous_cache / f"{part}_{pid}.npz") as old:
        y,common=old["y"].copy(),old["common"].copy()
        old_x=[old[f"x{n}"].copy() for n in LEVELS_MS]
        old_peaks=old["peaks"].copy()
    windows,labels,valid=load_recording(data,pid,part)
    if not np.array_equal(labels,y): raise ValueError("Changed target labels")
    annotation=ROOT / "data/raw/mesa/polysomnography/annotations-rpoints" / f"mesa-sleep-{pid}-rpoint.csv"
    peaks,normal=read_annotations(annotation,len(y)*30)
    if not np.array_equal(peaks,old_peaks): raise ValueError("Changed beat annotations")
    rng=np.random.default_rng(seed+int(pid))
    xx,rr,coverage,mae,centers,scales=[],[],[],{},[],[]
    for k,level in enumerate(LEVELS_MS):
        jittered,error=jitter_peaks(peaks,level,rng)
        stream,local=rr_inputs(jittered,normal,len(y),valid,cfg,peaks[1:])
        x,center,scale=relative_features(np.column_stack([old_x[k],local]))
        xx.append(x);rr.append(stream);coverage.append(stream[:,2].mean(1));mae[str(level)]=error
        centers.append(center);scales.append(scale)
    expanded=(y>=0)&valid&(common | (np.min(coverage,axis=0)>=cfg["expanded_min_rr_coverage"]))
    if np.any(common & ~expanded): raise ValueError("Expanded targets lost original common epochs")
    row={"participant":pid,"partition":part,"eligible":int((y>=0).sum()),
         "common":int(common.sum()),"expanded":int(expanded.sum()),"realized_mae_ms":mae,
         "common_stage_counts":np.bincount(y[common],minlength=5).tolist(),
         "expanded_stage_counts":np.bincount(y[expanded],minlength=5).tolist()}
    temporary=target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,x=np.stack(xx),rr=np.stack(rr),y=y,common=common,expanded=expanded,
        night_center=np.stack(centers),night_scale=np.stack(scales),coverage_json=json.dumps(row),
        parent_cache_sha256=prior_signature)
    temporary.replace(target)
    return row


def run(args):
    cfg=config(); previous=Path(cfg["source_run"])
    parent=read_json(previous / "manifest.json")
    sys.path.insert(0,str(ROOT/"ecg_staging"))
    from train_full_transformer import load_cohort
    accepted,cohort=load_cohort(ROOT/"data/derived/full_cohort_windows_v1")
    ids=parent["participants"]
    if any(not set(ids[p])<=set(accepted[p]) for p in ids): raise ValueError("Changed participant split")
    cache,version,_=verified_feature_reuse(previous,ROOT/"data/derived/full_cohort_windows_v1",ids,
        cohort,cfg["seed"],parent["annotation_inventory_sha256"])
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"records").mkdir(exist_ok=True)
    snapshot(args.output)
    feature_version=hashlib.sha256((digest(Path(__file__)) + digest(Path(__file__).with_name("common.py"))+
                                   json.dumps(cfg,sort_keys=True)+version).encode()).hexdigest()
    manifest={"status":"PREPARING","config":cfg,"participants":ids,"feature_names":NAMES,
              "feature_version":feature_version,"parent_feature_version":version,"parent_run":str(previous),
              "parent_cache":str(cache),"cohort":cohort,"train_epoch_cap":None,"validation_epoch_cap":None}
    existing=args.output/"manifest.json"
    if existing.exists() and read_json(existing)["feature_version"]!=feature_version:
        raise ValueError("Preparation destination belongs to different source/configuration")
    save(existing,manifest)
    tasks=[(cache,ROOT/"data/derived/full_cohort_windows_v1",args.output,part,pid,cfg["seed"],version,cfg)
           for part,group in ids.items() for pid in group]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(prepare_record,t) for t in tasks]):
            rows.append(future.result());save(args.output/"coverage.json",rows)
            print("PREPARED",len(rows),"/",len(tasks),rows[-1],flush=True)
    expected=sum(r["usable"] for r in read_json(previous/"coverage.json") if r["partition"]=="train")
    if sum(r["common"] for r in rows if r["partition"]=="train")!=expected:
        raise ValueError("Training common target coverage changed")
    # Population scaling fits every expanded clean-training target, without a sample cap.
    arrays=[]
    for pid in ids["train"]:
        with np.load(args.output/"records"/f"train_{pid}.npz") as f:
            arrays.append(f["x"][0,f["expanded"]])
    all_x=np.concatenate(arrays);del arrays
    center,scale=np.zeros(len(NAMES)),np.ones(len(NAMES))
    for j in range(len(NAMES)):
        values=all_x[:,j][np.isfinite(all_x[:,j])]
        if len(values):
            center[j]=np.median(values);a,b=np.percentile(values,[25,75]);scale[j]=b-a if b-a>1e-6 else 1
    np.savez(args.output/"population_scaler.npz",center=center.astype(np.float32),scale=scale.astype(np.float32))
    manifest["counts"]={part:{key:sum(r[key] for r in rows if r["partition"]==part)
                              for key in ("eligible","common","expanded")} for part in ids}
    manifest["status"]="COMPLETE";save(existing,manifest)
    (args.output/"requirements_frozen.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    print("PREPARATION COMPLETE",manifest["counts"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--workers",type=int,default=8)
    run(p.parse_args())
