#!/usr/bin/env python3
"""
Generate visual heatmaps of mHC matrices across all layers.

Creates matplotlib figures showing:
- H_res mixing matrices for all layers
- Alpha values evolution
- H_pre/H_post weight distribution

This uses STATIC parameter analysis (no forward passes needed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np

from src.conversionV2 import load_mhc_model_v2
from src.sinkhorn import sinkhorn_knopp

# Try to import matplotlib, fall back to ASCII if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not found, using ASCII visualization")


def collect_layer_data(mhc_model):
    """Collect H_res matrices and other data from all layers (static analysis)."""
    n_layers = len(mhc_model.model.layers)

    def extract_static_matrices(mhc):
        """Derive static constrained matrices from V2 parameters."""
        s = mhc.num_residual_streams
        f = mhc.num_fracs
        v = mhc.num_input_views

        static_alpha = mhc.static_alpha.detach().float()
        sf = s * f
        vf = v * f

        alpha_pre = torch.sigmoid(static_alpha[:, :vf])
        alpha_residual = mhc.residual_mix_constraint_fn(static_alpha[:, vf:])

        # Collapse frac/view dimensions for per-stream summary visualizations.
        alpha_pre = alpha_pre.reshape(s, f, v, f).mean(dim=(1, 2, 3))
        alpha_residual = alpha_residual.reshape(s, f, s, f).mean(dim=(1, 3))

        if hasattr(mhc, "static_beta"):
            beta = (2.0 * torch.sigmoid(mhc.static_beta.detach().float())).mean(dim=(1, 2))
        else:
            beta = torch.full((s,), float("nan"))

        return (
            alpha_pre.cpu().numpy(),
            alpha_residual.cpu().numpy(),
            beta.cpu().numpy(),
        )
    
    # Storage
    h_res_attn = []
    h_res_mlp = []
    h_pre_attn = []
    h_post_attn = []
    alphas_pre = []
    alphas_post = []
    alphas_res = []
    
    for layer in mhc_model.model.layers:
        # Attention mHC
        mhc = layer.mhc_attn

        # Compute static constrained matrices from V2 parameters.
        h_pre, h_res, h_post = extract_static_matrices(mhc)
        
        h_res_attn.append(h_res)
        h_pre_attn.append(h_pre)
        h_post_attn.append(h_post)

        # MLP mHC
        _, h_res_mlp_mat, _ = extract_static_matrices(layer.mhc_mlp)
        h_res_mlp.append(h_res_mlp_mat)

        # V2 scalar gates controlling dynamic coefficient magnitude.
        alphas_pre.append(mhc.pre_branch_scale.detach().item())
        alphas_post.append(mhc.h_post_scale.detach().item() if hasattr(mhc, "h_post_scale") else float("nan"))
        alphas_res.append(mhc.residual_scale.detach().item())
    
    # Compute H_res identity distances
    identity_distances = []
    for h_res in h_res_attn:
        identity = np.eye(h_res.shape[0])
        dist = np.abs(h_res - identity).mean()
        identity_distances.append(dist)
    
    return {
        'h_res_attn': np.array(h_res_attn),  # (n_layers, 4, 4)
        'h_res_mlp': np.array(h_res_mlp),
        'h_pre_attn': np.array(h_pre_attn),  # (n_layers, 4)
        'h_post_attn': np.array(h_post_attn),
        'alphas_pre': np.array(alphas_pre),
        'alphas_post': np.array(alphas_post),
        'alphas_res': np.array(alphas_res),
        'identity_distances': np.array(identity_distances),
        'n_layers': n_layers,
    }


def plot_matplotlib(data: dict, output_path: str):
    """Create matplotlib visualization."""
    n_layers = data['n_layers']
    n_streams = data['h_res_attn'].shape[-1]
    
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.2)
    
    # 1. H_res matrices for all layers (attention) - top left
    ax1 = fig.add_subplot(gs[0, 0])

    # Stack H_res matrices horizontally
    split_idx = min(16, n_layers)
    h_res_combined = np.hstack(data['h_res_attn'][:split_idx])
    im1 = ax1.imshow(h_res_combined, cmap='RdBu_r', vmin=0, vmax=0.5, aspect='auto')
    ax1.set_title(f'H_res Attention (layers 0-{split_idx-1})\nDark blue=0, Red=0.5, White=0.25 (identity)', fontsize=10)
    ax1.set_ylabel('From Stream')
    ax1.set_xlabel('To Stream (grouped by layer)')
    
    # Add layer separators
    for i in range(1, split_idx):
        ax1.axvline(x=i * n_streams - 0.5, color='black', linewidth=0.5)
    
    plt.colorbar(im1, ax=ax1, label='Weight')
    
    # 2. H_res matrices for layers 16-27 - top right
    ax2 = fig.add_subplot(gs[0, 1])
    if n_layers > split_idx:
        h_res_combined2 = np.hstack(data['h_res_attn'][split_idx:])
        im2 = ax2.imshow(h_res_combined2, cmap='RdBu_r', vmin=0, vmax=0.5, aspect='auto')
        ax2.set_title(f'H_res Attention (layers {split_idx}-{n_layers-1})', fontsize=10)
        ax2.set_ylabel('From Stream')
        ax2.set_xlabel('To Stream (grouped by layer)')

        for i in range(1, n_layers - split_idx):
            ax2.axvline(x=i * n_streams - 0.5, color='black', linewidth=0.5)

        plt.colorbar(im2, ax=ax2, label='Weight')
    else:
        ax2.axis('off')
    
    # 3. Alpha values across layers
    ax3 = fig.add_subplot(gs[1, :])
    layers = np.arange(n_layers)
    ax3.plot(layers, data['alphas_pre'], 'b-o', label='α_pre', markersize=3)
    ax3.plot(layers, data['alphas_post'], 'g-s', label='α_post', markersize=3)
    ax3.plot(layers, data['alphas_res'], 'r-^', label='α_res', markersize=3)
    ax3.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='init (0.01)')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Alpha Value')
    ax3.set_title('Alpha Gating Values (controls dynamic vs static coefficients)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. H_pre weights across layers
    ax4 = fig.add_subplot(gs[2, 0])
    h_pre = data['h_pre_attn']
    for i in range(h_pre.shape[1]):
        ax4.plot(layers, h_pre[:, i], label=f'Stream {i}', marker='o', markersize=2)
    ax4.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='init (0.25)')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Weight')
    ax4.set_title('H_pre: Squeeze Weights (4→1)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. H_post weights across layers
    ax5 = fig.add_subplot(gs[2, 1])
    h_post = data['h_post_attn']
    for i in range(h_post.shape[1]):
        ax5.plot(layers, h_post[:, i], label=f'Stream {i}', marker='o', markersize=2)
    ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='init (1.0)')
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Weight')
    ax5.set_title('H_post: Broadcast Weights (1→4)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. H_res identity distance across layers
    ax6 = fig.add_subplot(gs[3, :])
    ax6.bar(layers, data['identity_distances'], color='purple', alpha=0.7)
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Identity Distance')
    ax6.set_title('H_res Deviation from Identity (0=no mixing, higher=more stream mixing)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('mHC Model Analysis', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    plt.close()


def ascii_heatmap(matrix: np.ndarray, title: str = "", vmin: float = None, vmax: float = None):
    """Print an ASCII heatmap."""
    chars = " ░▒▓█"
    
    if vmin is None:
        vmin = matrix.min()
    if vmax is None:
        vmax = matrix.max()
    
    print(f"\n{title}")
    print("─" * (matrix.shape[1] * 6 + 4))
    
    for i, row in enumerate(matrix):
        row_str = f"{i}│ "
        for val in row:
            norm = (val - vmin) / (vmax - vmin + 1e-8)
            norm = max(0, min(1, norm))
            char_idx = int(norm * (len(chars) - 1))
            row_str += f"{chars[char_idx]}{val:4.2f} "
        print(row_str)
    
    print("─" * (matrix.shape[1] * 6 + 4))
    print(f"  Range: [{vmin:.3f}, {vmax:.3f}]")


def plot_ascii(data: dict):
    """Create ASCII visualization."""
    n_layers = data['n_layers']
    n_streams = data['h_pre_attn'].shape[1]
    
    print("\n" + "="*70)
    print("H_res MATRICES (sampled layers)")
    print("="*70)
    
    # Show a few representative layers
    for layer_idx in [0, 7, 14, 21, n_layers-1]:
        if layer_idx < n_layers:
            ascii_heatmap(
                data['h_res_attn'][layer_idx],
                f"Layer {layer_idx} H_res (Attention)",
                vmin=0, vmax=0.5
            )
    
    print("\n" + "="*70)
    print("ALPHA VALUES")
    print("="*70)
    print(f"\n{'Layer':<6} {'α_pre':>10} {'α_post':>10} {'α_res':>10} {'H_res dist':>12}")
    print("-" * 52)
    for i in range(n_layers):
        print(f"{i:<6} {data['alphas_pre'][i]:>10.4f} {data['alphas_post'][i]:>10.4f} {data['alphas_res'][i]:>10.4f} {data['identity_distances'][i]:>12.4f}")
    
    print("\n" + "="*70)
    print("H_res IDENTITY DISTANCE (0 = identity, higher = more mixing)")
    print("="*70)
    
    # ASCII bar chart
    max_dist = max(data['identity_distances'].max(), 0.1)  # Avoid div by zero
    for i, dist in enumerate(data['identity_distances']):
        bar_len = int(dist / max_dist * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " *" if dist > 0.01 else ""
        print(f"L{i:02d} │{bar}│ {dist:.4f}{marker}")
    
    print("\n" + "="*70)
    print("H_pre WEIGHTS (squeeze: how much each stream contributes)")
    print("="*70)
    header = " ".join(f"S{i:>1}".rjust(8) for i in range(n_streams))
    print(f"\n{'Layer':<6} {header}")
    print("-" * (7 + 1 + 9 * n_streams))
    for i in range(0, n_layers, 4):  # Every 4th layer
        w = data['h_pre_attn'][i]
        vals = " ".join(f"{x:>8.4f}" for x in w)
        print(f"{i:<6} {vals}")


def plot_single_layer_detail(data: dict, layer_idx: int):
    """Detailed view of a single layer."""
    print(f"\n{'='*70}")
    print(f"DETAILED VIEW: LAYER {layer_idx}")
    print(f"{'='*70}")
    
    h_res = data['h_res_attn'][layer_idx]
    h_pre = data['h_pre_attn'][layer_idx]
    h_post = data['h_post_attn'][layer_idx]
    n_streams = h_res.shape[0]
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print(f"│                        H_res ({n_streams}×{n_streams})                              │")
    print("│            Stream Mixing Matrix (doubly stochastic)            │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    to_streams = " ".join([f"S{i}".rjust(7) for i in range(n_streams)])
    print(f"│         To: {to_streams}                    │")
    print("│        ┌────────────────────────────────────┐                  │")
    for i in range(n_streams):
        row = " ".join(f"{v:7.4f}" for v in h_res[i])
        print(f"│  From S{i} │ {row} │                  │")
    print("│        └────────────────────────────────────┘                  │")
    
    # Identity distance
    identity = np.eye(n_streams)
    dist = np.abs(h_res - identity).mean()
    print(f"│  Identity distance: {dist:.4f} (0=identity, higher=mixing)       │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  H_pre (squeeze weights)      │  H_post (broadcast weights)    │")
    print("├─────────────────────────────────────────────────────────────────┤")
    for idx in range(n_streams):
        init_hint = "  (init: 0.25)" if idx == 0 else ""
        init_post_hint = "  (init: 1.0)" if idx == 0 else ""
        print(f"│  S{idx}: {h_pre[idx]:.4f}{init_hint:<15} │  S{idx}: {h_post[idx]:.4f}{init_post_hint:<16} │")
    print(f"│  Sum: {h_pre.sum():.4f}              │  Avg: {h_post.mean():.4f}               │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Alpha values (gating: dynamic vs static)                      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  α_pre:  {data['alphas_pre'][layer_idx]:.4f}  (init: 0.01)                           │")
    print(f"│  α_post: {data['alphas_post'][layer_idx]:.4f}  (init: 0.01)                           │")
    print(f"│  α_res:  {data['alphas_res'][layer_idx]:.4f}  (init: 0.01)                           │")
    print("└─────────────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="Visualize mHC matrices across layers")
    parser.add_argument("--mhc", type=str, required=True, help="Path to mHC model")
    parser.add_argument("--output", type=str, default="mhc_analysis.png",
                        help="Output path for matplotlib figure")
    parser.add_argument("--ascii", action="store_true",
                        help="Force ASCII output even if matplotlib available")
    parser.add_argument("--layer", type=int, default=None,
                        help="Show detailed view of specific layer")
    args = parser.parse_args()
    
    print("Loading mHC V2 model...")
    mhc_model, tokenizer = load_mhc_model_v2(
        model_path=args.mhc,
        device="cpu",
        torch_dtype=torch.float32,
    )
    mhc_model.eval()
    
    print(f"Collecting data from {len(mhc_model.model.layers)} layers...")
    data = collect_layer_data(mhc_model)
    
    # Detailed single layer view
    if args.layer is not None:
        plot_single_layer_detail(data, args.layer)
    
    # Full visualization
    if HAS_MATPLOTLIB and not args.ascii:
        plot_matplotlib(data, args.output)
        print(f"\n📊 Open {args.output} to see the visualization!")
    else:
        plot_ascii(data)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # How much did alphas change?
    alpha_change = np.abs(data['alphas_res'] - 0.01).mean()
    print(f"\nAlpha drift from init (0.01): {alpha_change:.4f}")
    
    # How much did H_res deviate from identity?
    h_res_drift = data['identity_distances']
    print(f"H_res identity distance: mean={h_res_drift.mean():.4f}, max={h_res_drift.max():.4f}")
    
    # Which layer has most mixing?
    max_mix_layer = h_res_drift.argmax()
    print(f"Most mixing at layer: {max_mix_layer} (dist={h_res_drift[max_mix_layer]:.4f})")
    
    total_drift = h_res_drift.sum()
    if total_drift < 0.1:
        print(f"\n⚠️  Very little training detected! Model looks mostly unchanged from init.")
    else:
        print(f"\n✅ Model shows training - H_res matrices have drifted from identity.")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
