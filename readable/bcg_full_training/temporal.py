#!/usr/bin/env python3
"""Full-target temporal CNN with optional RR branch, validation-selected checkpoints."""
import argparse
import csv
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from common import (
    config,
    verify_prepared,
    report,
    save,
    snapshot,
    source_hash,
    LEVELS_MS,
    NAMES,
    read_json,
)
from sequences import SequenceDataset, ChunkSampler

STOP = False


# Residual temporal-convolution block with dilation, layer normalization and dropout.
class TemporalBlock(nn.Module):
    # Create two same-width dilated convolutions and their normalization/dropout layers.
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.first = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.second = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    # Input/output shape: batch x channels x time. Return the residual block activations.
    def forward(self, x):
        y = self.norm1(F.gelu(self.first(x)).transpose(1, 2)).transpose(1, 2)
        y = self.dropout(y)
        y = self.norm2(F.gelu(self.second(y)).transpose(1, 2)).transpose(1, 2)
        return F.gelu(x + self.dropout(y))


# Earlier temporal-CNN baseline used by the inherited comparison controls.
class TemporalCNN(nn.Module):
    # Construct feature encoding, the optional RR branch, temporal blocks and five-stage classifier from cfg.
    def __init__(self, cfg, rr_branch=True):
        super().__init__()
        self.halo = cfg["context_halo_epochs"]
        self.core = cfg["context_core_epochs"]
        width = cfg["hidden_channels"]
        self.rr_branch = rr_branch
        self.features = nn.Conv1d(2 * len(NAMES), width, 1)
        if rr_branch:
            self.rr_encoder = nn.Sequential(
                nn.Conv1d(3, 16, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(16, 32, 5, padding=2, stride=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.combine = nn.Conv1d(width + 32, width, 1)
        self.blocks = nn.Sequential(
            *[TemporalBlock(width, d, cfg["dropout"]) for d in cfg["dilations"]]
        )
        self.classifier = nn.Conv1d(width, 5, 1)

    # Encode feature/RR context and return five raw class scores for the central output epochs.
    def forward(self, features, rr):
        x = F.gelu(self.features(features.transpose(1, 2)))
        if self.rr_branch:
            b, t, c, n = rr.shape
            encoded = self.rr_encoder(rr.reshape(b * t, c, n)).reshape(b, t, 32)
            encoded = encoded * rr[:, :, 2].mean(-1, keepdim=True)
            x = F.gelu(self.combine(torch.cat([x, encoded.transpose(1, 2)], dim=1)))
        result = self.classifier(self.blocks(x)).transpose(1, 2)
        return result[:, self.halo : self.halo + self.core]


# Wrap a sequence dataset and deterministic chunk sampler in a PyTorch DataLoader.
# training controls sampler behavior; workers controls parallel input loading, not participant membership.
def loader(dataset, cfg, training=False, epoch=0, level=0, workers=None):
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=ChunkSampler(dataset, training, epoch, level, cfg["seed"]),
        num_workers=cfg["loader_workers"] if workers is None else workers,
        pin_memory=False,
        drop_last=False,
    )


@torch.no_grad()
# Run the model without gradients and compute common/expanded evaluation reports.
# Optionally write predictions; return a dictionary with common/expanded reports and common-target log loss.
def evaluate(
    model, dataset, cfg, level=0, device="cpu", predictions_path=None, workers=None
):
    model.eval()
    ys = {"common": [], "expanded": []}
    ps = {"common": [], "expanded": []}
    pids = {"common": [], "expanded": []}
    loss_sum, count = 0.0, 0
    file = predictions_path.open("w") if predictions_path else None
    writer = csv.writer(file) if file else None
    if writer:
        writer.writerow(
            ["participant", "epoch", "reference", "prediction", "common"]
            + ["p_" + s for s in ["Wake", "N1", "N2", "N3", "REM"]]
        )
    try:
        for feat, rr, _, common, expanded, record_ids, starts in loader(
            dataset, cfg, level=level, workers=workers
        ):
            logits = model(feat.to(device), rr.to(device))
            prob = logits.softmax(-1).cpu().numpy()
            pred = prob.argmax(-1)
            for name, target in (
                ("common", common.numpy()),
                ("expanded", expanded.numpy()),
            ):
                mask = target >= 0
                ys[name].extend(target[mask].tolist())
                ps[name].extend(pred[mask].tolist())
                for b, index in enumerate(record_ids.tolist()):
                    pids[name].extend(
                        [dataset.records[index]["pid"]] * int(mask[b].sum())
                    )
            mask = common >= 0
            if mask.any():
                loss_sum += float(
                    F.cross_entropy(logits.cpu()[mask], common[mask], reduction="sum")
                )
                count += int(mask.sum())
            if writer:
                target = expanded.numpy()
                for b, index in enumerate(record_ids.tolist()):
                    pid = dataset.records[index]["pid"]
                    for k in np.flatnonzero(target[b] >= 0):
                        writer.writerow(
                            [
                                pid,
                                int(starts[b]) + int(k),
                                int(target[b, k]),
                                int(pred[b, k]),
                                int(common[b, k] >= 0),
                                *map(float, prob[b, k]),
                            ]
                        )
    finally:
        if file:
            file.close()
    output = {
        name: report(np.asarray(ys[name]), np.asarray(ps[name]), np.asarray(pids[name]))
        for name in ys
    }
    if output["common"]["n_epochs"] != int(dataset.common_counts.sum()):
        raise ValueError("Incomplete common validation coverage")
    if output["expanded"]["n_epochs"] != int(
        sum((r["y"][r["expanded"]] >= 0).sum() for r in dataset.records)
    ):
        raise ValueError("Incomplete expanded validation coverage")
    output["log_loss"] = loss_sum / count
    return output


# Save training state through a temporary file, then replace the destination checkpoint.
def checkpoint(path, state):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


# Fit an inherited temporal-CNN variant, select checkpoints on validation performance and save state/results.
def train(args):
    global STOP
    cfg = config()
    manifest = verify_prepared(args.prepared)
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    mask = "expanded" if args.variant == "tcn_dual_expanded" else "common"
    training = SequenceDataset(args.prepared, "train", mask)
    validation = SequenceDataset(args.prepared, "validation", "expanded")
    model = TemporalCNN(cfg, rr_branch=args.variant != "tcn_hrv_common").to(args.device)
    counts = training.counts.astype(float)
    weights = (counts.sum() / (5 * counts)) ** cfg["tcn_class_weight_power"]
    weights /= np.average(weights, weights=counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    fingerprint = {
        "source": source_hash(),
        "features": manifest["feature_version"],
        "variant": args.variant,
        "config": cfg,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists() and read_json(
        args.output / "manifest.json"
    )["fingerprint"] != fingerprint:
        raise ValueError("Existing temporal output source/configuration changed")
    if (args.output / "result.json").exists():
        print("ALREADY COMPLETE", args.variant, flush=True)
        return
    snapshot(args.output)
    save(
        args.output / "manifest.json",
        {
            "variant": args.variant,
            "fingerprint": fingerprint,
            "target_mask": mask,
            "training_epochs_per_pass": int(training.counts.sum()),
            "training_stage_counts": training.counts.tolist(),
            "validation_common_epochs": int(validation.common_counts.sum()),
            "reference_test_policy": cfg["reference_test_policy"],
        },
    )
    epoch, start_batch, history, best, stale = 0, 0, [], -1.0, 0
    cumulative = {
        "unweighted_loss_sum": 0.0,
        "n": 0,
        "cm": np.zeros((5, 5), dtype=np.int64),
    }
    last = args.output / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=args.device, weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Resume source/configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, start_batch, history, best, stale = [
            state[k] for k in ("epoch", "batch", "history", "best", "stale")
        ]
        cumulative = state["cumulative"]
        torch.set_rng_state(state["torch_rng"])

    # Persist model, optimizer, scheduler, RNG and position so this fit can resume.
    def dump(next_epoch, next_batch):
        checkpoint(
            last,
            {
                "fingerprint": fingerprint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": next_epoch,
                "batch": next_batch,
                "history": history,
                "best": best,
                "stale": stale,
                "cumulative": cumulative,
                "torch_rng": torch.get_rng_state(),
            },
        )

    # Set a stop flag when a scheduler signal arrives; the training loop handles checkpointing.
    def stop(signum, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    for epoch in range(epoch, cfg["max_epochs"] if stale < cfg["patience"] else epoch):
        model.train()
        started = time.time()
        for batch, (feat, rr, target, _, _, _, _) in enumerate(
            loader(training, cfg, training=True, epoch=epoch)
        ):
            if batch < start_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(feat.to(args.device), rr.to(args.device))
            target = target.to(args.device)
            valid = target >= 0
            loss = F.cross_entropy(logits[valid], target[valid], weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                cumulative["unweighted_loss_sum"] += float(
                    F.cross_entropy(logits[valid], target[valid], reduction="sum")
                )
                cumulative["n"] += int(valid.sum())
                true = target[valid].cpu().numpy()
                pred = logits.argmax(-1)[valid].cpu().numpy()
                cumulative["cm"] += np.bincount(true * 5 + pred, minlength=25).reshape(
                    5, 5
                )
            if batch % 100 == 0:
                save(
                    args.output / "progress.json",
                    {
                        "epoch": epoch + 1,
                        "batch": batch + 1,
                        "batches": (len(training) + cfg["batch_size"] - 1)
                        // cfg["batch_size"],
                        "targets_seen_this_pass": cumulative["n"],
                        "expected_targets": int(training.counts.sum()),
                        "loss": float(loss),
                    },
                )
                print(
                    "PROGRESS",
                    args.variant,
                    epoch + 1,
                    batch + 1,
                    cumulative["n"],
                    float(loss),
                    flush=True,
                )
            if batch % 200 == 0 or STOP:
                dump(epoch, batch + 1)
            if STOP:
                raise SystemExit(75)
        if cumulative["n"] != int(training.counts.sum()):
            raise ValueError("Not every training target was visited exactly once")
        scores = {}
        losses = []
        expanded = {}
        for k, level in enumerate(LEVELS_MS):
            evaluated = evaluate(model, validation, cfg, k, args.device)
            scores[str(level)] = evaluated["common"]
            expanded[str(level)] = evaluated["expanded"]
            losses.append(evaluated["log_loss"])
        score = float(np.mean([r["macro_f1"] for r in scores.values()]))
        cm = cumulative["cm"]
        denom = cm.sum(0) + cm.sum(1)
        train_f1 = float(
            np.divide(2 * np.diag(cm), denom, out=np.zeros(5), where=denom > 0).mean()
        )
        history.append(
            {
                "step": epoch + 1,
                "axis": "Training epoch",
                "train_log_loss": cumulative["unweighted_loss_sum"] / cumulative["n"],
                "validation_log_loss": float(np.mean(losses)),
                "train_macro_f1": train_f1,
                "validation_macro_f1": score,
                "targets_seen": cumulative["n"],
                "seconds": time.time() - started,
                "monitoring_scope": "All training targets in mixed-noise pass; all common validation targets across four levels",
            }
        )
        save(args.output / "history.json", history)
        if score > best:
            best, stale = score, 0
            checkpoint(
                args.output / "best.pt",
                {
                    "model": model.state_dict(),
                    "fingerprint": fingerprint,
                    "epoch": epoch + 1,
                    "score": score,
                },
            )
            save(
                args.output / "best_validation.json",
                {
                    "model": args.variant,
                    "selection_score": score,
                    "validation": scores,
                    "expanded_validation": expanded,
                    "training_epochs": int(training.counts.sum()),
                    "target_mask": mask,
                    "epoch": epoch + 1,
                    "history": history,
                },
            )
        else:
            stale += 1
        print("VALIDATION", args.variant, epoch + 1, score, flush=True)
        cumulative = {
            "unweighted_loss_sum": 0.0,
            "n": 0,
            "cm": np.zeros((5, 5), dtype=np.int64),
        }
        start_batch = 0
        dump(epoch + 1, 0)
        if stale >= cfg["patience"]:
            break
    result = json.loads((args.output / "best_validation.json").read_text())
    result["history"] = history
    result["completed"] = True
    save(args.output / "result.json", result)
    print("COMPLETED", args.variant, best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--variant",
        choices=["tcn_hrv_common", "tcn_dual_common", "tcn_dual_expanded"],
        required=True,
    )
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    train(p.parse_args())
