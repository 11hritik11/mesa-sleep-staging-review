"""Representative temporal convolution, recurrent, attention and dense model families."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from balanced_utils import TemporalBlock


class BalancedNetwork(nn.Module):
    def __init__(self, width, architecture, cfg):
        super().__init__()
        self.architecture = architecture
        self.width = width
        self.core, self.halo = cfg["context_core_epochs"], cfg["context_halo_epochs"]
        self.feature_encoder = nn.Sequential(
            nn.Linear(2 * width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.2)
        )
        self.rr_encoder = nn.Sequential(
            nn.Conv1d(3, 16, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fuse = nn.Linear(128, 96)
        if architecture == "tcn":
            self.temporal = nn.Sequential(
                *[TemporalBlock(96, d, 0.2) for d in [1, 2, 4, 8]]
            )
        elif architecture in ("gru", "lstm"):
            cls = nn.GRU if architecture == "gru" else nn.LSTM
            self.temporal = cls(
                96, 64, num_layers=2, dropout=0.2, bidirectional=True, batch_first=True
            )
        elif architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                96, 4, 256, 0.2, batch_first=True, norm_first=True, activation="gelu"
            )
            self.temporal = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(96, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 96),
                nn.GELU(),
            )
        self.output = nn.Linear(128 if architecture in ("gru", "lstm") else 96, 5)

    def forward(self, features, rr):
        x = self.feature_encoder(features)
        if self.architecture != "mlp":
            b, t, c, n = rr.shape
            z = self.rr_encoder(rr.reshape(b * t, c, n)).reshape(b, t, 32)
            z = z * rr[:, :, 2].mean(-1, keepdim=True)
            x = F.gelu(self.fuse(torch.cat([x, z], -1)))
        if self.architecture == "tcn":
            x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        elif self.architecture in ("gru", "lstm"):
            x, _ = self.temporal(x)
        elif self.architecture == "transformer":
            t = x.shape[1]
            position = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
            div = torch.exp(
                torch.arange(0, 96, 2, device=x.device, dtype=x.dtype)
                * (-math.log(10000) / 96)
            )
            pe = torch.zeros(t, 96, device=x.device, dtype=x.dtype)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            pad = (features[:, :, self.width :].sum(-1) == self.width) & (
                rr[:, :, 2].sum(-1) == 0
            )
            x = self.temporal(x + pe[None], src_key_padding_mask=pad)
        else:
            x = self.temporal(x)
        return self.output(x[:, self.halo : self.halo + self.core])
