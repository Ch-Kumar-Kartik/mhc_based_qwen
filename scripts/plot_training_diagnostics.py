#!/usr/bin/env python3
"""Plot training diagnostics + mHC mapping gains.

Generates the requested graphs:
1) Absolute training loss gap vs training steps
2) Gradient norm vs training steps
3) Single-layer mapping + composite mapping: amax(|mapping|) vs layer index l

Inputs:
- Training logs (standard and mhc) produced by scripts/train.py or scripts/train_amd.py
  Expected to contain lines like:
    Step 123: loss=..., lr=..., grad_norm=...
- An mHC checkpoint directory for mapping plots (optional but needed for #3)

Example:
  python -m scripts.plot_training_diagnostics \
    --standard-log /path/to/standard/train.log \
    --mhc-log /path/to/mhc/train.log \
    --mhc-model /path/to/mhc/checkpoint-1000 \
    --outdir ./plots

If matplotlib is missing, the script will print an install hint.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_STEP_RE = re.compile(
    r"Step\s+(?P<step>\d+):\s+loss=(?P<loss>[0-9]*\.?[0-9]+)"  # loss
    r"(?:,\s+lr=(?P<lr>[0-9eE+\-\.]+))?"  # optional lr
    r"(?:,\s+grad_norm=(?P<grad_norm>[0-9eE+\-\.]+))?"  # optional grad_norm
)


@dataclass
class Series:
    steps: np.ndarray
    loss: np.ndarray
    grad_norm: Optional[np.ndarray]


def _load_step_series_from_log(path: str) -> Series:
    steps: List[int] = []
    loss: List[float] = []
    grad: List[float] = []
    grad_present = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group("step")))
            loss.append(float(m.group("loss")))
            g = m.group("grad_norm")
            if g is not None:
                grad_present = True
                grad.append(float(g))
            else:
                grad.append(np.nan)

    if not steps:
        raise ValueError(f"No training step lines parsed from log: {path}")

    steps_arr = np.asarray(steps, dtype=np.int64)
    loss_arr = np.asarray(loss, dtype=np.float64)
    grad_arr = np.asarray(grad, dtype=np.float64) if grad_present else None

    # If grad was partially present, keep it and let NaNs exist.
    if grad_present:
        return Series(steps=steps_arr, loss=loss_arr, grad_norm=grad_arr)
    return Series(steps=steps_arr, loss=loss_arr, grad_norm=None)


def _align_by_step(a: Series, b: Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return common_steps, a_loss_aligned, b_loss_aligned."""
    a_map = {int(s): float(v) for s, v in zip(a.steps, a.loss)}
    b_map = {int(s): float(v) for s, v in zip(b.steps, b.loss)}
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common:
        raise ValueError("No overlapping steps between the two runs")
    a_loss = np.asarray([a_map[s] for s in common], dtype=np.float64)
    b_loss = np.asarray([b_map[s] for s in common], dtype=np.float64)
    return np.asarray(common, dtype=np.int64), a_loss, b_loss


def _compute_mhc_mappings(mhc_model_path: str) -> Dict[str, np.ndarray]:
    import torch

    from src.conversion import load_mhc_model
    from src.sinkhorn import sinkhorn_knopp

    mhc_model, _ = load_mhc_model(mhc_model_path, device="cpu", torch_dtype=torch.float32)
    mhc_model.eval()

    n_layers = len(mhc_model.model.layers)
    single_amax: List[float] = []
    composite_amax: List[float] = []

    composite = torch.eye(4, dtype=torch.float32)

    for layer in mhc_model.model.layers:
        # Use static (parameter-derived) H_res for attention + mlp
        h_attn = sinkhorn_knopp(layer.mhc_attn.b_res).squeeze().detach().cpu()  # (4,4)
        h_mlp = sinkhorn_knopp(layer.mhc_mlp.b_res).squeeze().detach().cpu()    # (4,4)

        # Per-transformer-layer mapping: attention mixing then mlp mixing
        m_l = h_mlp @ h_attn

        single_amax.append(float(m_l.abs().max().item()))

        composite = m_l @ composite
        composite_amax.append(float(composite.abs().max().item()))

    return {
        "layer_idx": np.arange(n_layers, dtype=np.int64),
        "single_layer_amax": np.asarray(single_amax, dtype=np.float64),
        "composite_amax": np.asarray(composite_amax, dtype=np.float64),
    }


def main():
    p = argparse.ArgumentParser(description="Plot loss gap, grad norm, and mapping gains")
    p.add_argument("--standard-log", type=str, default=None, help="Path to standard model train.log")
    p.add_argument("--mhc-log", type=str, default=None, help="Path to mHC model train.log")
    p.add_argument("--mhc-model", type=str, default=None, help="Path to mHC checkpoint dir (for mapping plots)")
    p.add_argument("--outdir", type=str, default="./plots", help="Output directory for PNGs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        has_mpl = True
    except Exception:
        has_mpl = False

    if not has_mpl:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib\n"
        )

    import matplotlib.pyplot as plt

    # --- Training plots (loss gap + grad norm) ---
    if args.standard_log and args.mhc_log:
        std = _load_step_series_from_log(args.standard_log)
        mhc = _load_step_series_from_log(args.mhc_log)

        common_steps, std_loss, mhc_loss = _align_by_step(std, mhc)
        loss_gap = np.abs(mhc_loss - std_loss)

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(common_steps, loss_gap, linewidth=1.5)
        ax1.set_title("Absolute training loss gap")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("|loss_mhc - loss_standard|")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(1, 2, 2)
        # Prefer mHC grad_norm if available; otherwise standard.
        grad_series = None
        grad_steps = None
        if mhc.grad_norm is not None:
            grad_series = mhc.grad_norm
            grad_steps = mhc.steps
            label = "mHC grad_norm"
        elif std.grad_norm is not None:
            grad_series = std.grad_norm
            grad_steps = std.steps
            label = "standard grad_norm"
        else:
            label = None

        if grad_series is not None:
            ax2.plot(grad_steps, grad_series, linewidth=1.5)
            ax2.set_title("Gradient norm vs steps")
            ax2.set_xlabel("training step")
            ax2.set_ylabel("grad_norm")
            ax2.grid(True, alpha=0.3)
            if label:
                ax2.legend([label])
        else:
            ax2.text(0.5, 0.5, "No grad_norm found in logs", ha="center", va="center")
            ax2.set_axis_off()

        out_path = os.path.join(args.outdir, "training_loss_gap_and_grad_norm.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    else:
        if args.standard_log or args.mhc_log:
            print("Note: provide BOTH --standard-log and --mhc-log to plot loss gap.")

    # --- Mapping plots (single-layer + composite amax gain) ---
    if args.mhc_model:
        mapping = _compute_mhc_mappings(args.mhc_model)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mapping["layer_idx"], mapping["single_layer_amax"], marker="o", markersize=3, label="single-layer mapping amax")
        ax.plot(mapping["layer_idx"], mapping["composite_amax"], marker="s", markersize=3, label="composite mapping amax")
        ax.set_title("amax(|mapping|) vs layer index")
        ax.set_xlabel("layer index l")
        ax.set_ylabel("amax(|M|)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = os.path.join(args.outdir, "mapping_amax_gain_vs_layer.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        print("Note: provide --mhc-model to plot mapping gains.")


if __name__ == "__main__":
    main()
