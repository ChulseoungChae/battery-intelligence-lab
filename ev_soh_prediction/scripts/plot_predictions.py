#!/usr/bin/env python3
"""Generate LOVO prediction figures under outputs/figs/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_patchtst import (  # noqa: E402
    ACC_TOLS,
    FEATURES,
    accuracy_at,
    build_windows,
    predict,
    train_model,
)
from src.patchtst import PatchTST  # noqa: E402

FIG = ROOT / "outputs" / "figs"
DOC = ROOT / "docs" / "figs"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(ROOT / "outputs/traj/daily_trajectories.csv")
    df["date"] = pd.to_datetime(df["date"])
    L, H = 14, 7
    W = build_windows(df, L, H)

    model_kw = dict(
        patch_len=4, stride=2, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1
    )
    train_kw = dict(
        L=L,
        epochs=60,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=64,
        seed=42,
        model_kw=model_kw,
        device=device,
    )

    rows = []
    for te_vid in np.unique(W["vid"]):
        te = W["vid"] == te_vid
        tr = ~te
        model, meta = train_model(W["X"][tr], W["y"][tr] - W["last"][tr], **train_kw)
        pred = predict(model, W["X"][te], W["last"][te], meta, device)
        for j, i in enumerate(np.where(te)[0]):
            rows.append(
                dict(
                    device=str(W["vid"][i]),
                    date=W["date"][i],
                    y_true=float(W["y"][i]),
                    y_pred=float(pred[j]),
                    last=float(W["last"][i]),
                )
            )

    pred_df = pd.DataFrame(rows)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values(["device", "date"])
    pred_df.to_csv(FIG / "lovo_predictions.csv", index=False)

    devices = sorted(pred_df["device"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dev in zip(axes.ravel(), devices):
        g = pred_df[pred_df["device"] == dev]
        ax.plot(g["date"], g["y_true"], label="Actual SOH", color="#1f4e79", lw=2)
        ax.plot(g["date"], g["y_pred"], label="PatchTST pred", color="#c45c26", lw=1.5)
        ax.plot(g["date"], g["last"], label="Hold-last", color="#7a7a7a", ls="--", lw=1)
        ax.set_title(f"device {dev}")
        ax.set_ylabel("SOH (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"LOVO SOH forecast (L={L}, H={H} days)", fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "lovo_pred_timeseries.png", dpi=140, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for c, dev in zip(plt.cm.tab10(np.linspace(0, 1, len(devices))), devices):
        g = pred_df[pred_df["device"] == dev]
        ax.scatter(g["y_true"], g["y_pred"], s=18, alpha=0.65, label=dev, color=c)
    lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min()) - 0.3
    hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.fill_between(
        [lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#1f4e79", alpha=0.08, label="±1.0 band"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.set_title("LOVO: Actual vs Predicted")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "lovo_pred_scatter.png", dpi=140, bbox_inches="tight")
    plt.close()

    err = (pred_df["y_pred"] - pred_df["y_true"]).abs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(err, bins=30, color="#1f4e79", edgecolor="white")
    for t, ls in [(0.5, ":"), (1.0, "--"), (2.0, "-")]:
        axes[0].axvline(t, color="#c45c26", ls=ls, lw=1.5, label=f"±{t}")
    axes[0].set_xlabel("|error| (SOH %p)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Absolute error distribution (LOVO)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    acc_m = [accuracy_at(pred_df["y_true"], pred_df["y_pred"], t) for t in ACC_TOLS]
    acc_h = [accuracy_at(pred_df["y_true"], pred_df["last"], t) for t in ACC_TOLS]
    x = np.arange(len(ACC_TOLS))
    w = 0.35
    axes[1].bar(x - w / 2, acc_m, w, label="PatchTST", color="#c45c26")
    axes[1].bar(x + w / 2, acc_h, w, label="Hold-last", color="#7a7a7a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Acc@±{t}" for t in ACC_TOLS])
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Tolerance accuracy (LOVO pooled)")
    for i, (a, b) in enumerate(zip(acc_m, acc_h)):
        axes[1].text(i - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
        axes[1].text(i + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "lovo_pred_accuracy.png", dpi=140, bbox_inches="tight")
    plt.close()

    DOC.mkdir(parents=True, exist_ok=True)
    for name in (
        "lovo_pred_timeseries.png",
        "lovo_pred_scatter.png",
        "lovo_pred_accuracy.png",
    ):
        src = FIG / name
        if src.exists():
            (DOC / name).write_bytes(src.read_bytes())
    print(f"[saved] figures under {FIG} (and {DOC})")


if __name__ == "__main__":
    main()
