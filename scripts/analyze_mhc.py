#!/usr/bin/env python3
"""
Analyze and visualize mHC model internals.

Shows:
- H_pre (squeeze 4→1), H_post (expand 1→4), H_res (stream mixing) matrices
- Stream evolution through layers
- How much streams have diverged from initialization
- Alpha gating values evolution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np

from src.conversion import load_mhc_model
from src.sinkhorn import sinkhorn_knopp, compute_identity_distance


def print_matrix(name: str, matrix: torch.Tensor, precision: int = 4):
    """Pretty print a small matrix."""
    print(f"\n{name}:")
    if matrix.dim() == 1:
        print("  [" + ", ".join(f"{v:.{precision}f}" for v in matrix.tolist()) + "]")
    elif matrix.dim() == 2:
        for row in matrix.tolist():
            print("  [" + ", ".join(f"{v:.{precision}f}" for v in row) + "]")
    else:
        print(f"  Shape: {matrix.shape}")


def colorize_value(val: float, low: float = 0.0, high: float = 1.0) -> str:
    """Return ANSI colored representation of value."""
    # Normalize to 0-1
    norm = (val - low) / (high - low + 1e-8)
    norm = max(0, min(1, norm))
    
    # Color gradient: blue (low) -> white (mid) -> red (high)
    if norm < 0.5:
        # Blue to white
        intensity = int(255 * (norm * 2))
        return f"\033[48;2;{intensity};{intensity};255m{val:6.3f}\033[0m"
    else:
        # White to red
        intensity = int(255 * ((1 - norm) * 2))
        return f"\033[48;2;255;{intensity};{intensity}m{val:6.3f}\033[0m"


def print_heatmap(name: str, matrix: torch.Tensor):
    """Print a matrix as a colored heatmap in terminal."""
    print(f"\n{name} (blue=low, red=high):")
    mat = matrix.squeeze().cpu()
    if mat.dim() == 1:
        mat = mat.unsqueeze(0)
    
    vmin, vmax = mat.min().item(), mat.max().item()
    
    for i, row in enumerate(mat.tolist()):
        row_str = " ".join(colorize_value(v, vmin, vmax) for v in row)
        print(f"  Stream {i}: {row_str}")
    print(f"  Range: [{vmin:.4f}, {vmax:.4f}]")


def analyze_layer_parameters(mhc_layer, layer_idx: int, layer_type: str):
    """Analyze mHC layer parameters (static analysis, no forward pass needed)."""
    
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} - {layer_type}")
    print(f"{'='*60}")
    
    # Alpha values
    print(f"\n📊 Alpha gating values (init=0.01):")
    print(f"   α_pre:  {mhc_layer.alpha_pre.item():.6f}")
    print(f"   α_post: {mhc_layer.alpha_post.item():.6f}")
    print(f"   α_res:  {mhc_layer.alpha_res.item():.6f}")
    
    # b_pre -> H_pre
    h_pre = torch.softmax(mhc_layer.b_pre, dim=-1).squeeze()
    print(f"\n📥 H_pre (squeeze weights, init=[0.25]*4):")
    print(f"   {[f'{v:.4f}' for v in h_pre.tolist()]}")
    print(f"   Sum: {h_pre.sum().item():.4f}")
    
    # b_post -> H_post  
    h_post = (2.0 * torch.sigmoid(mhc_layer.b_post)).squeeze()
    print(f"\n📤 H_post (broadcast weights, init=[1.0]*4):")
    print(f"   {[f'{v:.4f}' for v in h_post.tolist()]}")
    
    # b_res -> H_res
    h_res = sinkhorn_knopp(mhc_layer.b_res).squeeze()
    print(f"\n🔀 H_res (stream mixing, init=identity):")
    for i in range(4):
        print(f"   Row {i}: {[f'{v:.4f}' for v in h_res[i].tolist()]}")
    
    # Identity distance
    identity = torch.eye(4)
    dist = (h_res - identity).abs().mean().item()
    print(f"   Identity distance: {dist:.4f}")
    
    return h_pre, h_post, h_res


def analyze_alpha_evolution(mhc_model):
    """Analyze how alpha gating values evolved across layers."""
    print(f"\n{'='*60}")
    print("Alpha Gating Values (per layer)")
    print("  init=0.01, controls dynamic vs static coefficient balance")
    print(f"{'='*60}")
    
    print(f"\n{'Layer':<6} {'α_pre':>10} {'α_post':>10} {'α_res':>10} {'H_res dist':>12}")
    print("-" * 52)
    
    for i, layer in enumerate(mhc_model.model.layers):
        # Attention mHC
        mhc = layer.mhc_attn
        alpha_pre = mhc.alpha_pre.item()
        alpha_post = mhc.alpha_post.item()
        alpha_res = mhc.alpha_res.item()
        
        # H_res identity distance
        h_res = sinkhorn_knopp(mhc.b_res).squeeze()
        identity = torch.eye(4)
        dist = (h_res - identity).abs().mean().item()
        
        marker = " *" if dist > 0.01 else ""
        print(f"{i:<6} {alpha_pre:>10.4f} {alpha_post:>10.4f} {alpha_res:>10.4f} {dist:>12.4f}{marker}")
    
    print("\n(* = layer with H_res significantly different from identity)")


def analyze_b_res_matrices(mhc_model, num_layers: int = 4):
    """Visualize raw b_res matrices before Sinkhorn."""
    print(f"\n{'='*60}")
    print(f"Raw b_res Matrices (before Sinkhorn)")
    print(f"Init: diagonal=20, off-diagonal=0")
    print(f"{'='*60}")
    
    for i, layer in enumerate(mhc_model.model.layers[:num_layers]):
        print(f"\n--- Layer {i} ---")
        
        for name, mhc in [("Attention", layer.mhc_attn), ("MLP", layer.mhc_mlp)]:
            b_res = mhc.b_res.squeeze().detach()
            print(f"\n{name} b_res:")
            print_heatmap(f"  {name}", b_res)
            
            # Show diagonal dominance
            diag = b_res.diag()
            off_diag = b_res[~torch.eye(4, dtype=torch.bool)]
            print(f"    Diagonal mean: {diag.mean().item():.2f}, Off-diag mean: {off_diag.mean().item():.2f}")


def summarize_training_drift(mhc_model):
    """Quick summary of how much the model changed from initialization."""
    print(f"\n{'='*60}")
    print("TRAINING DRIFT SUMMARY")
    print(f"{'='*60}")
    
    total_h_res_drift = 0
    total_alpha_drift = 0
    total_b_post_drift = 0
    
    for layer in mhc_model.model.layers:
        for mhc in [layer.mhc_attn, layer.mhc_mlp]:
            # H_res drift
            h_res = sinkhorn_knopp(mhc.b_res).squeeze()
            identity = torch.eye(4)
            total_h_res_drift += (h_res - identity).abs().mean().item()
            
            # Alpha drift from 0.01
            total_alpha_drift += abs(mhc.alpha_pre.item() - 0.01)
            total_alpha_drift += abs(mhc.alpha_post.item() - 0.01)
            total_alpha_drift += abs(mhc.alpha_res.item() - 0.01)
            
            # b_post drift from 0
            total_b_post_drift += mhc.b_post.abs().mean().item()
    
    n_layers = len(mhc_model.model.layers)
    
    print(f"\nTotal H_res drift from identity: {total_h_res_drift:.4f}")
    print(f"Total alpha drift from 0.01: {total_alpha_drift:.4f}")
    print(f"Total b_post drift from 0: {total_b_post_drift:.4f}")
    
    if total_h_res_drift < 0.1 and total_alpha_drift < 0.1:
        print(f"\n⚠️  Model appears mostly UNTRAINED - parameters very close to init!")
    else:
        print(f"\n✅ Model shows training - parameters have drifted from initialization.")


def main():
    parser = argparse.ArgumentParser(description="Analyze mHC model internals")
    parser.add_argument("--mhc", type=str, required=True, help="Path to mHC model")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of layers to analyze in detail")
    parser.add_argument("--no-color", action="store_true", 
                        help="Disable colored output")
    args = parser.parse_args()
    
    print("Loading mHC model...")
    mhc_model, tokenizer = load_mhc_model(args.mhc, device="cpu", torch_dtype=torch.float32)
    mhc_model.eval()
    
    # 1. Quick summary
    summarize_training_drift(mhc_model)
    
    # 2. Alpha evolution across layers
    analyze_alpha_evolution(mhc_model)
    
    # 3. Raw b_res matrices
    analyze_b_res_matrices(mhc_model, args.layers)
    
    # 4. Detailed parameter analysis for first few layers
    print(f"\n\n{'#'*60}")
    print("# DETAILED LAYER ANALYSIS")
    print(f"{'#'*60}")
    
    for layer_idx in range(min(args.layers, len(mhc_model.model.layers))):
        layer = mhc_model.model.layers[layer_idx]
        analyze_layer_parameters(layer.mhc_attn, layer_idx, "ATTENTION")
        analyze_layer_parameters(layer.mhc_mlp, layer_idx, "MLP")
    
    print(f"\n\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
