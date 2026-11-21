"""Chronological feature sequences with explicit input modality and no target caps."""

from collections import OrderedDict
import numpy as np
from balanced_utils import SequenceDataset, base, extra_columns, feature_names


class BalancedDataset(SequenceDataset):
    def __init__(self, prepared, part, mask="common", modality="beat"):
        super().__init__(base(), part, mask)
        self.drop_features = ()
        self.drop_rr = False
        self.extra_prepared = prepared
        self.modality = modality
        self.extra_columns = extra_columns(modality)
        self.names = feature_names(modality)
        if self.extra_columns:
            with np.load(prepared / "extra_scaler.npz") as f:
                self.center = np.r_[self.center, f["center"][self.extra_columns]]
                self.scale = np.r_[self.scale, f["scale"][self.extra_columns]]

    def __getitem__(self, request):
        index, start, level = request
        record = self.records[index]
        if index not in self.cache:
            with np.load(record["path"]) as f:
                x, rr = f["x"], f["rr"]
            if self.extra_columns:
                with np.load(
                    self.extra_prepared / "extra" / f'{self.part}_{record["pid"]}.npz'
                ) as f:
                    extra = f["x"][:, self.extra_columns]
                x = np.concatenate(
                    [x, np.broadcast_to(extra, (4, *extra.shape))], axis=2
                )
            self.cache[index] = (x, rr)
            while len(self.cache) > 20:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        all_x, all_rr = self.cache[index]
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
        scaled = np.clip((x - self.center) / self.scale, -10, 10)
        feature[offset : offset + len(x), :width] = np.nan_to_num(
            scaled, nan=0, posinf=0, neginf=0
        )
        feature[offset : offset + len(x), width:] = ~np.isfinite(x)
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
