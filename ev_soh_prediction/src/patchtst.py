"""Channel-independent PatchTST for scalar SOH regression."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTST(nn.Module):
    """Input (B, L, F) → scalar (B,). Channel-independent patch encoder."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        num_patches = (seq_len - patch_len) // stride + 1
        if num_patches < 1:
            raise ValueError(f"seq_len={seq_len} too short for patch_len={patch_len}")
        self.num_patches = num_patches
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = _PositionalEncoding(d_model, num_patches + 1)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(n_features * num_patches * d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        b, _, f = x.shape
        x = x.permute(0, 2, 1)  # (B, F, L)
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B, F, P, patch)
        p = patches.shape[2]
        z = patches.reshape(b * f, p, self.patch_len)
        z = self.patch_embed(z)
        z = self.pos(z)
        z = self.dropout(z)
        z = self.encoder(z)
        z = z.reshape(b, -1)
        return self.head(z).squeeze(-1)
