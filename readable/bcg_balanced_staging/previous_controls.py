#!/usr/bin/env python3
"""Rescore existing CNN checkpoints on the new disjoint validation roles only."""
import argparse
from pathlib import Path
import numpy as np, torch
from balanced_utils import (
    base,
    cfg,
    oldconfig,
    read_json,
    save,
    fingerprint,
    TemporalCNN,
    choose_beta_across_levels,
    save_evidence,
)
from balanced_data import BalancedDataset
from train_balanced_deep import validation_logits


# Load two already-trained parent CNN controls and rescore them on the new validation roles.
# This step writes validation evidence/results; it does not fit those control weights anew.
def run(args):
    torch.set_num_threads(8)
    settings = oldconfig()
    train = BalancedDataset(args.prepared, "train")
    prior = train.counts / train.counts.sum()
    val = BalancedDataset(args.prepared, "validation", "expanded")
    for name in ["tcn_dual_common", "tcn_hrv_common"]:
        out = args.output / ("previous_" + name)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "result.json").exists():
            continue
        original = Path(cfg()["base_run"]) / "temporal" / name
        state = torch.load(original / "best.pt", map_location="cpu", weights_only=False)
        model = TemporalCNN(settings, rr_branch=name != "tcn_hrv_common")
        model.load_state_dict(state["model"])
        full_evidence = {
            n: validation_logits(model, val, settings, k)
            for k, n in enumerate([0, 20, 40, 60])
        }
        evidence = {n: v[:3] for n, v in full_evidence.items()}
        save_evidence(out, full_evidence)
        beta, reports, score = choose_beta_across_levels(evidence, args.prepared, prior)
        save(
            out / "result.json",
            {
                "name": "previous_" + name,
                "specification": {"modality": "beat", "mask": "common"},
                "selection_score": score,
                "validation": reports,
                "beta": beta,
                "prior": prior.tolist(),
                "training_targets": int(train.counts.sum()),
                "source": fingerprint(),
                "history": [],
                "completed": True,
                "kind": "control",
                "checkpoint": str(original / "best.pt"),
                "variant": name,
                "output": str(out),
            },
        )
        print("CONTROL", name, score, beta, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
