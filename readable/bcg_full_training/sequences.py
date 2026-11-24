"""Chronological chunks with halos, fixed target identities and noise-aware indexing."""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from common import verify_prepared


# Represent one participant partition as chronological chunks with within-recording context.
# Targets and identity metadata are separate from the feature inputs supplied to the network.
class SequenceDataset(Dataset):
    # Load the requested partition and saved population scaler; build record/chunk indexes and target counts.
    # part selects people before chunk construction, so chunks do not cross participant boundaries.
    def __init__(
        self, prepared, part, target_mask="common", drop_features=(), drop_rr=False
    ):
        self.prepared, self.part, self.target_mask = prepared, part, target_mask
        self.manifest = verify_prepared(prepared)
        cfg = self.manifest["config"]
        self.core, self.halo = cfg["context_core_epochs"], cfg["context_halo_epochs"]
        self.drop_features, self.drop_rr = tuple(drop_features), drop_rr
        self.cache = OrderedDict()
        self.records = []
        self.chunks = []
        self.counts = np.zeros(5, dtype=np.int64)
        self.common_counts = np.zeros(5, dtype=np.int64)
        with np.load(prepared / "population_scaler.npz") as f:
            self.center, self.scale = f["center"], f["scale"]
        # Build chunks only after choosing the participant partition.
        for pid in self.manifest["participants"][part]:
            path = prepared / "records" / f"{part}_{pid}.npz"
            with np.load(path) as f:
                y = f["y"].copy()
                common = f["common"].copy()
                expanded = f["expanded"].copy()
            mask = common if target_mask == "common" else expanded
            self.counts += np.bincount(y[mask], minlength=5)
            self.common_counts += np.bincount(y[common], minlength=5)
            index = len(self.records)
            record = {
                "pid": pid,
                "path": path,
                "y": y,
                "common": common,
                "expanded": expanded,
                "chunks": [],
            }
            for start in range(0, len(y), self.core):
                if mask[start : start + self.core].any():
                    record["chunks"].append((index, start))
                    self.chunks.append((index, start))
            self.records.append(record)
        if self.counts.sum() == 0:
            raise ValueError("No sequence targets")

    # Return the number of eligible output chunks, not the number of people.
    def __len__(self):
        return len(self.chunks)

    # Input: (record index, output-start epoch, jitter-level index). Return normalized inputs, target masks and identities.
    # Pad context at recording boundaries; -100 targets are ignored by the training loss.
    def __getitem__(self, request):
        index, start, level = request
        record = self.records[index]
        if index not in self.cache:
            with np.load(record["path"]) as f:
                self.cache[index] = (f["x"], f["rr"])
            while len(self.cache) > 20:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        all_x, all_rr = self.cache[index]
        # Clamp context to this recording: never borrow samples from another person.
        lo, hi = max(0, start - self.halo), min(
            len(record["y"]), start + self.core + self.halo
        )
        offset = lo - (start - self.halo)
        length = self.core + 2 * self.halo
        width = len(self.center)
        feature = np.zeros((length, width * 2), dtype=np.float32)
        feature[:, width:] = 1
        rr = np.zeros((length, 3, all_rr.shape[-1]), dtype=np.float32)
        x = all_x[level, lo:hi]
        missing = ~np.isfinite(x)
        scaled = np.clip((x - self.center) / self.scale, -10, 10)
        feature[offset : offset + len(x), :width] = np.nan_to_num(
            scaled, nan=0, posinf=0, neginf=0
        )
        feature[offset : offset + len(x), width:] = missing
        rr[offset : offset + len(x)] = all_rr[level, lo:hi]
        if self.drop_features:
            feature[:, self.drop_features] = 0
            feature[:, np.asarray(self.drop_features) + width] = 1
        if self.drop_rr:
            rr[:] = 0
        n = min(self.core, len(record["y"]) - start)
        targets = {}
        for name in ("common", "expanded"):
            y = np.full(self.core, -100, dtype=np.int64)
            take = record[name][start : start + n]
            y[:n] = np.where(take, record["y"][start : start + n], -100)
            targets[name] = y
        return (
            feature,
            rr,
            targets[self.target_mask],
            targets["common"],
            targets["expanded"],
            np.int64(index),
            np.int64(start),
        )


# Generate reproducible chunk requests, with shuffled records and jitter choices during fitting.
class ChunkSampler(Sampler):
    # Store the dataset, fitting/evaluation mode, training-pass number, fixed evaluation level and seed.
    def __init__(self, dataset, training=False, epoch=0, level=0, seed=20261102):
        self.dataset, self.training, self.epoch, self.level, self.seed = (
            dataset,
            training,
            epoch,
            level,
            seed,
        )

    # Return the number of chunk requests in one complete pass.
    def __len__(self):
        return len(self.dataset)

    # Yield (record index, start epoch, jitter index) for every declared chunk.
    # Evaluation preserves order and uses a fixed level; fitting shuffles and samples jitter deterministically.
    def __iter__(self):
        if not self.training:
            yield from ((i, s, self.level) for i, s in self.dataset.chunks)
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.dataset.records))
        size = self.dataset.manifest["config"]["record_shuffle_group"]
        for begin in range(0, len(order), size):
            group = [
                chunk
                for i in order[begin : begin + size]
                for chunk in self.dataset.records[i]["chunks"]
            ]
            rng.shuffle(group)
            for index, start in group:
                yield index, start, int(rng.integers(0, 4))
