#!/usr/bin/env python3
"""
Visualize the mHC squeeze/unsqueeze operations.

Shows the transformation:
  4 streams → squeeze → 1 stream → sublayer → 1 stream → expand → 4 streams

This script does STATIC analysis of the mHC parameters without running
full forward passes through attention (which requires complex setup).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch

from src.conversion import load_mhc_model
from src.conversionV2 import load_mhc_model_v2
from src.sinkhorn import sinkhorn_knopp


def _load_mhc_model_auto(model_path: str):
    """Load either V1 or V2 mHC checkpoint based on config.json model_type."""
    config_path = Path(model_path) / "config.json"
    model_type = ""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_type = str(config.get("model_type", ""))

    if model_type == "qwen3_mhc_v2":
        return load_mhc_model_v2(model_path, device="cpu", torch_dtype=torch.float32)

    return load_mhc_model(model_path, device="cpu", torch_dtype=torch.float32)


def _get_h_res_matrix(mhc) -> torch.Tensor:
    """Get residual mixing matrix for either V1 (`b_res`) or V2 (`static_alpha`)."""
    if hasattr(mhc, "b_res"):
        return sinkhorn_knopp(mhc.b_res).squeeze()

    if hasattr(mhc, "static_alpha"):
        num_input = mhc.num_input_views * mhc.num_fracs
        num_residual = mhc.num_residual_streams * mhc.num_fracs

        alpha_residual = mhc.static_alpha.float()[..., num_input : num_input + num_residual]
        alpha_residual = alpha_residual / float(getattr(mhc, "residual_mix_temperature", 1.0))
        alpha_residual = torch.clamp(alpha_residual, -5.0, 5.0)
        return mhc.residual_mix_constraint_fn(alpha_residual).squeeze()

    raise AttributeError("Unsupported mHC module type: cannot extract residual mixing matrix")


def _is_v1_style_module(mhc) -> bool:
    return hasattr(mhc, "alpha_pre") and hasattr(mhc, "b_pre") and hasattr(mhc, "b_res")


def summarize_all_layers(mhc_model):
    """Quick summary of parameter changes across all layers."""
    print(f"\n{'='*70}")
    print("SUMMARY: PARAMETER DRIFT FROM INITIALIZATION (all layers)")
    print(f"{'='*70}")
    
    print(f"\n{'Layer':<6} {'α_pre':>8} {'α_post':>8} {'α_res':>8} {'b_post':>10} {'H_res dist':>10}")
    print("-" * 60)
    
    total_drift = 0
    max_drift_layer = 0
    max_drift = 0
    
    for i, layer in enumerate(mhc_model.model.layers):
        mhc = layer.mhc_attn

        if _is_v1_style_module(mhc):
            alpha_pre = mhc.alpha_pre.item()
            alpha_post = mhc.alpha_post.item()
            alpha_res = mhc.alpha_res.item()
            b_post_drift = mhc.b_post.abs().mean().item()
        else:
            alpha_pre = mhc.pre_branch_scale.item() if hasattr(mhc, "pre_branch_scale") else float("nan")
            alpha_post = mhc.h_post_scale.item() if hasattr(mhc, "h_post_scale") else float("nan")
            alpha_res = mhc.residual_scale.item() if hasattr(mhc, "residual_scale") else float("nan")
            b_post_drift = mhc.static_beta.abs().mean().item() if hasattr(mhc, "static_beta") else float("nan")

        # H_res distance from identity
        h_res = _get_h_res_matrix(mhc)
        identity = torch.eye(4)
        h_res_dist = (h_res - identity).abs().mean().item()
        
        total_drift += h_res_dist
        if h_res_dist > max_drift:
            max_drift = h_res_dist
            max_drift_layer = i
        
        # Mark layers with significant changes
        marker = " *" if h_res_dist > 0.01 else ""
        
        print(f"{i:<6} {alpha_pre:>8.4f} {alpha_post:>8.4f} {alpha_res:>8.4f} {b_post_drift:>10.4f} {h_res_dist:>10.4f}{marker}")
    
    print("-" * 60)
    print(f"Total H_res drift: {total_drift:.4f}")
    print(f"Max drift at layer: {max_drift_layer} ({max_drift:.4f})")
    print(f"(* = layer with significant drift from identity)")
    
    if total_drift < 0.1:
        print(f"\n  Very little training detected! Model looks mostly unchanged from init.")
    else:
        print(f"\n Model shows training - parameters have drifted from initialization.")
    
    return max_drift_layer


def visualize_layer_static(mhc_model, layer_idx: int):
    """Visualize a single layer's mHC parameters (static, no forward pass)."""
    
    layer = mhc_model.model.layers[layer_idx]
    mhc_attn = layer.mhc_attn
    mhc_mlp = layer.mhc_mlp
    
    print(f"\n{'='*70}")
    print(f"LAYER {layer_idx} DETAILED VIEW")
    print(f"{'='*70}")
    
    for name, mhc in [("ATTENTION", mhc_attn), ("MLP", mhc_mlp)]:
        print(f"\n{'─'*70}")
        print(f"{name}")
        print(f"{'─'*70}")

        if not _is_v1_style_module(mhc):
            print("\n  V2 module detected (hyper/frac-connections parameterization).")
            print("  Scalar gates:")
            print(f"    pre_branch_scale: {mhc.pre_branch_scale.item():.6f}")
            print(f"    residual_scale:   {mhc.residual_scale.item():.6f}")
            if hasattr(mhc, "h_post_scale"):
                print(f"    h_post_scale:     {mhc.h_post_scale.item():.6f}")

            h_res = _get_h_res_matrix(mhc)
            print("\n  Residual mixing matrix (effective H_res from static_alpha residual block):")
            for i in range(4):
                row = h_res[i].tolist()
                row_str = " ".join(f"{v:6.4f}" for v in row)
                print(f"    [{row_str}]  <- from stream {i}")

            identity = torch.eye(4)
            dist = (h_res - identity).abs().mean().item()
            print(f"\n  Identity distance: {dist:.4f} (0=no mixing, higher=more mixing)")
            continue
        
        # Alpha values
        print(f"\n  Alpha gating (init=0.01):")
        print(f"    α_pre:  {mhc.alpha_pre.item():.6f}")
        print(f"    α_post: {mhc.alpha_post.item():.6f}")
        print(f"    α_res:  {mhc.alpha_res.item():.6f}")
        
        # H_pre from b_pre
        h_pre = torch.softmax(mhc.b_pre, dim=-1).squeeze()
        print(f"\n   H_pre (squeeze 4→1, init=[0.25]*4):")
        print(f"     Weights: {[f'{v:.4f}' for v in h_pre.tolist()]}")
        print(f"     Sum: {h_pre.sum().item():.4f}")
        print(f"     Interpretation: Stream contributions to sublayer input")
        
        # Visualize as bar
        print(f"     Stream 0: {'█' * int(h_pre[0]*40):<40} {h_pre[0]:.3f}")
        print(f"     Stream 1: {'█' * int(h_pre[1]*40):<40} {h_pre[1]:.3f}")
        print(f"     Stream 2: {'█' * int(h_pre[2]*40):<40} {h_pre[2]:.3f}")
        print(f"     Stream 3: {'█' * int(h_pre[3]*40):<40} {h_pre[3]:.3f}")
        
        # H_post from b_post
        h_post = (2.0 * torch.sigmoid(mhc.b_post)).squeeze()
        print(f"\n   H_post (expand 1→4, init=[1.0]*4, range [0,2]):")
        print(f"     Weights: {[f'{v:.4f}' for v in h_post.tolist()]}")
        print(f"     Interpretation: How sublayer output is distributed to streams")
        
        # Visualize as bar (scale for 0-2 range)
        print(f"     Stream 0: {'█' * int(h_post[0]*20):<40} {h_post[0]:.3f}")
        print(f"     Stream 1: {'█' * int(h_post[1]*20):<40} {h_post[1]:.3f}")
        print(f"     Stream 2: {'█' * int(h_post[2]*20):<40} {h_post[2]:.3f}")
        print(f"     Stream 3: {'█' * int(h_post[3]*20):<40} {h_post[3]:.3f}")
        
        # H_res from b_res via Sinkhorn
        h_res = sinkhorn_knopp(mhc.b_res).squeeze()
        print(f"\n   H_res (stream mixing, init=identity, doubly stochastic):")
        print(f"     Matrix (row i = how stream i mixes into new streams):")
        for i in range(4):
            row = h_res[i].tolist()
            row_str = " ".join(f"{v:6.4f}" for v in row)
            # Highlight diagonal
            print(f"       [{row_str}]  ← from stream {i}")
        
        # Identity distance
        identity = torch.eye(4)
        dist = (h_res - identity).abs().mean().item()
        print(f"\n     Identity distance: {dist:.4f} (0=no mixing, higher=more mixing)")
        
        # Show mixing interpretation
        print(f"\n     Mixing interpretation:")
        for i in range(4):
            main_contrib = h_res[i, i].item()
            other_contribs = [(j, h_res[i, j].item()) for j in range(4) if j != i and h_res[i, j].item() > 0.01]
            if other_contribs:
                others_str = ", ".join(f"{v:.1%} from S{j}" for j, v in other_contribs)
                print(f"       New S{i}: {main_contrib:.1%} self + {others_str}")
            else:
                print(f"       New S{i}: {main_contrib:.1%} self (no significant mixing)")


def compare_init_vs_trained(mhc_model, layer_idx: int = 0):
    """Compare what the coefficients look like vs initialization."""
    print(f"\n\n{'='*70}")
    print(f"INITIALIZATION VS TRAINED (Layer {layer_idx})")
    print(f"{'='*70}")
    
    layer = mhc_model.model.layers[layer_idx]
    mhc = layer.mhc_attn

    if not _is_v1_style_module(mhc):
        print("\nV2 module detected. Showing V2-focused drift summary.")
        print(f"\n{'Parameter':<24} {'Expected Init':<20} {'Actual':<20} {'Drift':<10}")
        print("─"*78)

        pre_branch_scale = mhc.pre_branch_scale.item()
        residual_scale = mhc.residual_scale.item()
        print(f"{'pre_branch_scale':<24} {'0.0000':<20} {pre_branch_scale:<20.4f} {abs(pre_branch_scale):<10.4f}")
        print(f"{'residual_scale':<24} {'0.0000':<20} {residual_scale:<20.4f} {abs(residual_scale):<10.4f}")

        if hasattr(mhc, "h_post_scale"):
            h_post_scale = mhc.h_post_scale.item()
            print(f"{'h_post_scale':<24} {'0.0000':<20} {h_post_scale:<20.4f} {abs(h_post_scale):<10.4f}")

        h_res = _get_h_res_matrix(mhc)
        identity = torch.eye(4)
        h_res_dist = (h_res - identity).abs().mean().item()
        print(f"{'H_res identity dist':<24} {'0.0000':<20} {h_res_dist:<20.4f} {h_res_dist:<10.4f}")
        return
    
    print(f"\n{'Parameter':<20} {'Expected Init':<20} {'Actual':<20} {'Drift':<10}")
    print("─"*70)
    
    # Alpha values
    for name, param in [("α_pre", mhc.alpha_pre), ("α_post", mhc.alpha_post), ("α_res", mhc.alpha_res)]:
        val = param.item()
        drift = abs(val - 0.01)
        print(f"{name:<20} {'0.0100':<20} {val:<20.4f} {drift:<10.4f}")
    
    # b_pre (should be 0)
    b_pre_mean = mhc.b_pre.mean().item()
    print(f"{'b_pre mean':<20} {'0.0000':<20} {b_pre_mean:<20.4f} {abs(b_pre_mean):<10.4f}")
    
    # b_post (should be 0)
    b_post_mean = mhc.b_post.mean().item()
    print(f"{'b_post mean':<20} {'0.0000':<20} {b_post_mean:<20.4f} {abs(b_post_mean):<10.4f}")
    
    # b_res diagonal (should be 20)
    b_res = mhc.b_res.squeeze()
    diag_mean = b_res.diag().mean().item()
    off_diag_mean = b_res[~torch.eye(4, dtype=torch.bool)].mean().item()
    print(f"{'b_res diagonal':<20} {'20.0000':<20} {diag_mean:<20.4f} {abs(diag_mean-20):<10.4f}")
    print(f"{'b_res off-diag':<20} {'0.0000':<20} {off_diag_mean:<20.4f} {abs(off_diag_mean):<10.4f}")
    
    # H_res identity distance
    h_res = _get_h_res_matrix(mhc)
    identity = torch.eye(4)
    h_res_dist = (h_res - identity).abs().mean().item()
    print(f"{'H_res identity dist':<20} {'0.0000':<20} {h_res_dist:<20.4f} {h_res_dist:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize mHC stream transformations")
    parser.add_argument("--mhc", type=str, required=True, help="Path to mHC model")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index to analyze (default: auto-select layer with most change)")
    args = parser.parse_args()
    
    print("Loading mHC model...")
    mhc_model, tokenizer = _load_mhc_model_auto(args.mhc)
    mhc_model.eval()
    
    # Quick summary first - also returns layer with most drift
    max_drift_layer = summarize_all_layers(mhc_model)
    
    # Which layer to analyze in detail
    layer_idx = args.layer if args.layer is not None else max_drift_layer
    print(f"\n📍 Analyzing layer {layer_idx}" + (" (auto-selected: most drift)" if args.layer is None else ""))
    
    # Detailed visualization
    visualize_layer_static(mhc_model, layer_idx)
    
    # Compare to init
    compare_init_vs_trained(mhc_model, layer_idx)
    
    print("\n Visualization complete!")


if __name__ == "__main__":
    main()
