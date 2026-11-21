#!/usr/bin/env python3
"""Full-data training selected by disjoint-validation five-class balanced accuracy."""
import argparse, json, signal, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from balanced_utils import (
    cfg,
    oldconfig,
    verify_prepared,
    base,
    read_json,
    save,
    fingerprint,
    snapshot,
    DEEP_SPECS,
    feature_names,
    loader,
    checkpoint,
    choose_beta_across_levels,
    ba_from_cm,
    save_evidence,
)
from balanced_data import BalancedDataset
from balanced_models import BalancedNetwork

STOP = False


@torch.no_grad()
def validation_logits(
    model, dataset, settings, level=0, workers=None, collect_expanded=False
):
    model.eval()
    logits, ys, pp, ee = [], [], [], []
    for f, r, _, common, expanded, indices, starts in loader(
        dataset, settings, level=level, workers=workers
    ):
        scores = model(f, r).numpy()
        target = expanded.numpy() if collect_expanded else common.numpy()
        for b, index in enumerate(indices.tolist()):
            take = np.flatnonzero(target[b] >= 0)
            logits.append(scores[b, take])
            ys.extend(target[b, take].tolist())
            pp.extend([dataset.records[index]["pid"]] * len(take))
            ee.extend((int(starts[b]) + take).tolist())
    values = (
        np.concatenate(logits),
        np.asarray(ys, dtype=np.int64),
        np.asarray(pp),
        np.asarray(ee),
    )
    expected = sum(
        int(r["expanded" if collect_expanded else "common"].sum())
        for r in dataset.records
    )
    if len(ys) != expected or len(set(zip(values[2], values[3]))) != expected:
        raise ValueError("Incomplete or repeated evaluation targets")
    return values


def train(args):
    global STOP
    c = cfg()
    settings = oldconfig()
    settings.update(batch_size=c["batch_size"], loader_workers=c["loader_workers"])
    spec = DEEP_SPECS[args.index]
    out = args.output / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    version = fingerprint()
    prepared = read_json(args.prepared / "balanced_manifest.json")
    if prepared["status"] != "COMPLETE" or prepared["source_version"] != version:
        raise ValueError("Feature/source provenance changed")
    torch.set_num_threads(args.threads)
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    signature = {
        "source": version,
        "specification": spec,
        "base_features": prepared["base_feature_version"],
    }
    if (out / "manifest.json").exists() and read_json(out / "manifest.json")[
        "signature"
    ] != signature:
        raise ValueError("Changed output provenance")
    if (out / "result.json").exists():
        print("ALREADY COMPLETE", spec["name"])
        return
    train = BalancedDataset(args.prepared, "train", spec["mask"], spec["modality"])
    val = BalancedDataset(args.prepared, "validation", "expanded", spec["modality"])
    model = BalancedNetwork(len(train.names), spec["architecture"], settings)
    counts = train.counts.astype(float)
    prior = counts / counts.sum()
    weight = (counts.sum() / (5 * counts)) ** spec["power"]
    weight /= np.average(weight, weights=counts)
    weight = torch.tensor(weight, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    save(
        out / "manifest.json",
        {
            "signature": signature,
            "training_targets_per_pass": int(counts.sum()),
            "stage_counts": counts.tolist(),
            "feature_names": train.names,
            "config": c,
            "data_config": settings,
        },
    )
    snapshot(out)
    start_epoch, start_batch, best, stale, history = 0, 0, -1.0, 0, []
    cumulative = {"n": 0, "loss_sum": 0.0, "cm": np.zeros((5, 5), dtype=np.int64)}
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_batch, best, stale, history = [
            state[k] for k in ("epoch", "batch", "best", "stale", "history")
        ]
        cumulative = state["cumulative"]
        torch.set_rng_state(state["rng"])

    def dump(epoch, batch):
        checkpoint(
            out / "last.pt",
            {
                "signature": signature,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "best": best,
                "stale": stale,
                "history": history,
                "rng": torch.get_rng_state(),
                "cumulative": cumulative,
            },
        )

    def stop(signum, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    for epoch in range(
        start_epoch, c["epochs"] if stale < c["patience"] else start_epoch
    ):
        model.train()
        started = time.time()
        for batch, (f, r, y, common, _, _, _) in enumerate(
            loader(train, settings, training=True, epoch=epoch)
        ):
            if batch < start_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(f, r)
            valid = y >= 0
            targets = y[valid]
            scores = logits[valid]
            unweighted = F.cross_entropy(scores, targets, reduction="none")
            multiplier = weight[targets]
            denominator = multiplier.clone()
            if spec.get("focal"):
                multiplier = multiplier * (1 - torch.exp(-unweighted)) ** spec["focal"]
            if spec.get("quality"):
                common_quality = common[valid] >= 0
                quality = torch.where(common_quality, 1.0, 0.1 if epoch < 3 else 0.5)
                multiplier = multiplier * quality
                denominator = denominator * quality
            loss = (multiplier * unweighted).sum() / denominator.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                cumulative["n"] += int(valid.sum())
                cumulative["loss_sum"] += float(unweighted.sum())
                truth = targets.numpy()
                pred = scores.argmax(-1).numpy()
                cumulative["cm"] += np.bincount(truth * 5 + pred, minlength=25).reshape(
                    5, 5
                )
            if batch % 100 == 0:
                save(
                    out / "progress.json",
                    {
                        "epoch": epoch + 1,
                        "batch": batch + 1,
                        "targets_seen": cumulative["n"],
                        "expected_targets": int(counts.sum()),
                        "weighted_loss": float(loss),
                    },
                )
                print(
                    "PROGRESS",
                    spec["name"],
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
        if cumulative["n"] != int(counts.sum()):
            raise ValueError("Training targets omitted/repeated")
        full_evidence = {
            level: validation_logits(model, val, settings, k)
            for k, level in enumerate([0, 20, 40, 60])
        }
        evidence = {n: value[:3] for n, value in full_evidence.items()}
        beta, reports, score = choose_beta_across_levels(evidence, args.prepared, prior)
        val_loss = []
        from balanced_utils import subset_labels

        for scores, y, pids in evidence.values():
            take = subset_labels(pids, args.prepared) == "selection"
            val_loss.append(
                float(
                    F.cross_entropy(
                        torch.from_numpy(scores[take]), torch.from_numpy(y[take])
                    )
                )
            )
        history.append(
            {
                "step": epoch + 1,
                "axis": "Neural training epoch",
                "train_log_loss": cumulative["loss_sum"] / cumulative["n"],
                "validation_log_loss": float(np.mean(val_loss)),
                "train_balanced_accuracy": ba_from_cm(cumulative["cm"]),
                "validation_balanced_accuracy": score,
                "beta": beta,
                "targets_seen": cumulative["n"],
                "seconds": time.time() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        save(out / "history.json", history)
        scheduler.step(score)
        if score > best:
            best, stale = score, 0
            save_evidence(out, full_evidence)
            checkpoint(
                out / "best.pt",
                {
                    "model": model.state_dict(),
                    "signature": signature,
                    "epoch": epoch + 1,
                    "score": score,
                    "beta": beta,
                },
            )
            save(
                out / "best_validation.json",
                {
                    "name": spec["name"],
                    "specification": spec,
                    "selection_score": score,
                    "validation": reports,
                    "epoch": epoch + 1,
                    "beta": beta,
                    "prior": prior.tolist(),
                    "training_targets": int(counts.sum()),
                    "source": version,
                },
            )
        else:
            stale += 1
        print(
            "VALIDATION",
            spec["name"],
            epoch + 1,
            "BALANCED_ACCURACY",
            score,
            "BETA",
            beta,
            flush=True,
        )
        start_batch = 0
        cumulative = {"n": 0, "loss_sum": 0.0, "cm": np.zeros((5, 5), dtype=np.int64)}
        dump(epoch + 1, 0)
        if stale >= c["patience"]:
            break
    result = read_json(out / "best_validation.json")
    result.update(history=history, completed=True, kind="deep", output=str(out))
    save(out / "result.json", result)
    print("COMPLETED", spec["name"], best, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--threads", type=int, default=8)
    train(p.parse_args())
