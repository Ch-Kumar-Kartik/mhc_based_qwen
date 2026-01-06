#!/usr/bin/env python3
"""Plot training diagnostics + mHC mapping gains.

Generates the requested graphs:
1) Absolute training loss gap vs training steps
2) Gradient norm vs training steps
3) Single-layer mapping + composite mapping: Amax Gain Magnitude vs layer index l

For (3), each standard Transformer block is unrolled into two layers on the x-axis:
Attention then FFN.

The Amax Gain Magnitude (y-axis) is computed per mapping matrix H as:
        0.5 * (max_abs_row_sum(H) + max_abs_col_sum(H))
and then averaged over all tokens in a selected sequence.

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
        --sequence "Hello world" \
        --outdir ./plots

If matplotlib is missing, the script will print an install hint.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


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


def _amax_gain_magnitude_from_h_res(h_res: "torch.Tensor") -> "torch.Tensor":
    """Compute Amax Gain Magnitude per token for H_res.

    Args:
        h_res: (B, S, n, n) doubly-stochastic mixing matrices.

    Returns:
        gain: (B, S) tensor.
    """
    # Forward signal bound: max absolute row sum
    max_abs_row_sum = h_res.abs().sum(dim=-1).max(dim=-1).values  # (B, S)
    # Backward gradient bound: max absolute column sum
    max_abs_col_sum = h_res.abs().sum(dim=-2).max(dim=-1).values  # (B, S)
    return 0.5 * (max_abs_row_sum + max_abs_col_sum)


def _collect_h_res_per_sublayer(
    mhc_model: "torch.nn.Module",
    input_ids: "torch.Tensor",
    attention_mask: Optional["torch.Tensor"],
    device: str,
) -> "torch.Tensor":
    """Run one forward pass and collect per-token H_res for each Attention/FFN sublayer.

    Returns:
        h_all: (L2, B, S, n, n) where L2 = 2*num_hidden_layers (attn then ffn).
    """
    import torch

    mhc_model.eval()

    # We'll collect H_res in strict execution order: layer0.attn, layer0.ffn, layer1.attn, ...
    collected: List[torch.Tensor] = []
    hooks = []

    def _make_pre_hook(layer_name: str):
        def _pre_hook(module, args, kwargs):
            # args[0] is x: (B, S, n, C)
            x = args[0]
            B, S, n, C = x.shape
            x_flat = x.view(B, S, n * C)
            with torch.no_grad():
                _, _, h_res = module.compute_coefficients(x_flat)
            collected.append(h_res.detach().to("cpu"))
        return _pre_hook

    # Register hooks
    for layer in mhc_model.model.layers:
        hooks.append(layer.mhc_attn.register_forward_pre_hook(_make_pre_hook("attn"), with_kwargs=True))
        hooks.append(layer.mhc_mlp.register_forward_pre_hook(_make_pre_hook("mlp"), with_kwargs=True))

    try:
        with torch.no_grad():
            mhc_model(
                input_ids=input_ids.to(device),
                attention_mask=None if attention_mask is None else attention_mask.to(device),
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()

    if not collected:
        raise RuntimeError("Failed to collect any H_res matrices (no hooks fired)")

    return torch.stack(collected, dim=0)


def _compute_mhc_mappings(
    mhc_model_path: str,
    sequence: str,
    base_model: str,
    device: str,
    torch_dtype: "torch.dtype",
    max_length: int,
) -> Dict[str, np.ndarray]:
    import torch

    from src.conversion import load_mhc_model

    mhc_model, tokenizer = load_mhc_model(
        mhc_model_path,
        device=device,
        torch_dtype=torch_dtype,
        base_model=base_model,
    )
    mhc_model.eval()

    enc = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    h_all = _collect_h_res_per_sublayer(
        mhc_model=mhc_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
    )  # (L2, B, S, n, n)

    # Single-layer gain: average over tokens in the selected sequence.
    single_gain = _amax_gain_magnitude_from_h_res(h_all).mean(dim=(1, 2)).cpu()  # (L2,)

    # Composite suffix mapping per l:  Π_{i=1}^{L2-l} H_res^{L2-i}
    # which corresponds to: H_res[L2-1] @ ... @ H_res[l]
    L2 = h_all.shape[0]
    B, S, n, _ = h_all.shape[1:]
    suffix = torch.eye(n, dtype=h_all.dtype).view(1, 1, n, n).expand(B, S, n, n).clone()
    composite = torch.empty_like(h_all)
    for k in range(L2 - 1, -1, -1):
        suffix = torch.matmul(suffix, h_all[k])
        composite[k] = suffix

    composite_gain = _amax_gain_magnitude_from_h_res(composite).mean(dim=(1, 2)).cpu()  # (L2,)

    return {
        "layer_idx": np.arange(L2, dtype=np.int64),
        "single_layer_gain": single_gain.numpy().astype(np.float64),
        "composite_gain": composite_gain.numpy().astype(np.float64),
    }


def main():
    p = argparse.ArgumentParser(description="Plot loss gap, grad norm, and mapping gains")
    p.add_argument("--standard-log", type=str, default=None, help="Path to standard model train.log")
    p.add_argument("--mhc-log", type=str, default=None, help="Path to mHC model train.log")
    p.add_argument("--mhc-model", type=str, default=None, help="Path to mHC checkpoint dir (for mapping plots)")
    p.add_argument("--sequence", type=str, default=None, help="Text sequence used to compute token-averaged H_res gain metrics")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B", help="Base HF model name/path for tokenizer")
    p.add_argument("--device", type=str, default="cpu", help="Device for mapping computation (e.g. cpu, cuda)")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for mapping computation",
    )
    p.add_argument("--max-length", type=int, default=256, help="Max sequence length for mapping computation")
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
    std = None
    mhc = None
    
    if args.standard_log:
        std = _load_step_series_from_log(args.standard_log)
    if args.mhc_log:
        mhc = _load_step_series_from_log(args.mhc_log)

    # Loss gap requires BOTH logs
    if std is not None and mhc is not None:
        common_steps, std_loss, mhc_loss = _align_by_step(std, mhc)
        loss_gap = np.abs(mhc_loss - std_loss)

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(common_steps, loss_gap, linewidth=1.5)
        ax1.set_title("Absolute training loss gap")
        ax1.set_xlabel("training step")
        ax1.set_ylabel("|loss_mhc - loss_standard|")
        ax1.grid(True, alpha=0.3)

        out_path = os.path.join(args.outdir, "training_loss_gap.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    elif args.standard_log or args.mhc_log:
        print("Note: provide BOTH --standard-log and --mhc-log to plot loss gap.")

    # Training loss curve can be plotted from a SINGLE log
    loss_series = None
    loss_steps = None
    loss_label = None
    if mhc is not None:
        loss_series = mhc.loss
        loss_steps = mhc.steps
        loss_label = "mHC training loss"
    elif std is not None:
        loss_series = std.loss
        loss_steps = std.steps
        loss_label = "standard training loss"

    if loss_series is not None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(loss_steps, loss_series, linewidth=1.5, label=loss_label, alpha=0.8)
        ax.set_title("Training loss vs steps")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = os.path.join(args.outdir, "training_loss_curve.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Grad norm can be plotted from a SINGLE log
    grad_series = None
    grad_steps = None
    label = None
    if mhc is not None and mhc.grad_norm is not None:
        grad_series = mhc.grad_norm
        grad_steps = mhc.steps
        label = "mHC grad_norm"
    elif std is not None and std.grad_norm is not None:
        grad_series = std.grad_norm
        grad_steps = std.steps
        label = "standard grad_norm"

    if grad_series is not None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(grad_steps, grad_series, linewidth=1.5, label=label)
        ax.set_title("Gradient norm vs steps")
        ax.set_xlabel("training step")
        ax.set_ylabel("grad_norm")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = os.path.join(args.outdir, "training_grad_norm.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    elif args.standard_log or args.mhc_log:
        print("Note: no grad_norm data found in the provided log(s).")

    # --- Mapping plots (single-layer + composite amax gain) ---
    if args.mhc_model:
        if not args.sequence:
            raise SystemExit("--sequence is required when using --mhc-model (needed for token-averaged H_res metrics).")

        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        mapping = _compute_mhc_mappings(
            mhc_model_path=args.mhc_model,
            sequence=args.sequence,
            base_model=args.base_model,
            device=args.device,
            torch_dtype=dtype_map[args.dtype],
            max_length=args.max_length,
        )

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mapping["layer_idx"], mapping["single_layer_gain"], marker="o", markersize=3, label="single-layer H_res gain")
        ax.plot(mapping["layer_idx"], mapping["composite_gain"], marker="s", markersize=3, label="composite suffix gain")
        ax.set_title("Amax Gain Magnitude vs unrolled layer index")
        ax.set_xlabel("layer index l")
        ax.set_ylabel("Amax Gain Magnitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = os.path.join(args.outdir, "mapping_gain_vs_layer.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")
    else:
        print("Note: provide --mhc-model to plot mapping gains.")


if __name__ == "__main__":
    main()
