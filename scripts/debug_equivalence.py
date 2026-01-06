"""
Debug script to trace exactly where the mHC equivalence breaks.

This script systematically tests each component of the mHC transformation
to identify the source of numerical discrepancy.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/ch_kumar_kartik/qwen3-mhc')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MHCConfig
from src.mhc_layer import MHCLayer
from src.stream_ops import StreamExpand, StreamCollapse
from src.sinkhorn import sinkhorn_knopp

def test_stream_expand_collapse():
    """Test that expand -> collapse = identity."""
    print("\n" + "="*60)
    print("TEST 1: Stream Expand/Collapse Identity")
    print("="*60)
    
    n_streams = 4
    expand = StreamExpand(n_streams)
    collapse = StreamCollapse(n_streams)
    
    # Test with random tensor
    x = torch.randn(2, 10, 1024)  # (B, S, C)
    
    x_expanded = expand(x)  # (B, S, n, C)
    x_collapsed = collapse(x_expanded)  # (B, S, C)
    
    diff = (x - x_collapsed).abs().max().item()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Expanded shape: {x_expanded.shape}")
    print(f"  Collapsed shape: {x_collapsed.shape}")
    print(f"  Max difference after round-trip: {diff:.6e}")
    print(f"  ✓ PASS" if diff < 1e-6 else f"  ✗ FAIL")
    
    return diff < 1e-6


def test_sinkhorn_identity():
    """Test that Sinkhorn produces identity matrix from diagonal initialization."""
    print("\n" + "="*60)
    print("TEST 2: Sinkhorn-Knopp Identity Initialization")
    print("="*60)
    
    n = 4
    diagonal_value = 20.0
    
    # Create initialization similar to b_res in MHCLayer
    b_res = torch.zeros(1, 1, n, n)
    for i in range(n):
        b_res[0, 0, i, i] = diagonal_value
    
    print(f"  b_res diagonal value: {diagonal_value}")
    print(f"  b_res before Sinkhorn:\n{b_res[0, 0]}")
    
    # Apply Sinkhorn
    H_res = sinkhorn_knopp(b_res, num_iterations=20, eps=1e-8)
    
    print(f"\n  H_res after Sinkhorn:\n{H_res[0, 0]}")
    
    # Check how close to identity
    identity = torch.eye(n)
    identity_diff = (H_res[0, 0] - identity).abs().max().item()
    
    print(f"\n  Max difference from identity: {identity_diff:.6e}")
    print(f"  ✓ PASS" if identity_diff < 1e-4 else f"  ✗ FAIL - Not close to identity!")
    
    # Also test doubly stochastic properties
    row_sums = H_res.sum(dim=-1)
    col_sums = H_res.sum(dim=-2)
    
    print(f"\n  Row sums: {row_sums[0, 0].tolist()}")
    print(f"  Col sums: {col_sums[0, 0].tolist()}")
    
    return identity_diff < 1e-4


def test_mhc_layer_equivalence():
    """Test that MHC layer with equivalence initialization acts as identity + sublayer."""
    print("\n" + "="*60)
    print("TEST 3: MHC Layer Equivalence Check")
    print("="*60)
    
    n_streams = 4
    hidden_size = 64  # Smaller for debugging
    
    config = MHCConfig(
        n_streams=n_streams,
        hidden_size=hidden_size,
        alpha_init=0.01,
        phi_init_std=0.0,
        b_res_diagonal_init=20.0,
    )
    
    # Create a simple identity sublayer for testing
    class IdentitySublayer(nn.Module):
        def forward(self, x, **kwargs):
            return x
    
    class IdentityNorm(nn.Module):
        def forward(self, x):
            return x
    
    mhc_layer = MHCLayer(
        config=config,
        sublayer=IdentitySublayer(),
        layernorm=IdentityNorm(),
    )
    
    # Check initialization values
    print(f"  alpha_pre: {mhc_layer.alpha_pre.item():.4f}")
    print(f"  alpha_post: {mhc_layer.alpha_post.item():.4f}")
    print(f"  alpha_res: {mhc_layer.alpha_res.item():.4f}")
    print(f"  phi_pre max weight: {mhc_layer.phi_pre.weight.abs().max().item():.6e}")
    print(f"  phi_post max weight: {mhc_layer.phi_post.weight.abs().max().item():.6e}")
    print(f"  phi_res max weight: {mhc_layer.phi_res.weight.abs().max().item():.6e}")
    print(f"  b_pre: {mhc_layer.b_pre[0, 0].tolist()}")
    print(f"  b_post: {mhc_layer.b_post[0, 0].tolist()}")
    print(f"  b_res diagonal: {[mhc_layer.b_res[0, 0, i, i].item() for i in range(n_streams)]}")
    
    # Test with multi-stream input (as if it just came from StreamExpand)
    x_single = torch.randn(1, 5, hidden_size)  # (B, S, C)
    expand = StreamExpand(n_streams)
    x = expand(x_single)  # (B, S, n, C)
    
    print(f"\n  Input shape: {x.shape}")
    print(f"  Input (all streams identical): {x[0, 0, :, :3]}")  # First 3 features
    
    # Run through MHC layer
    with torch.no_grad():
        output = mhc_layer(x)
    
    print(f"\n  Output shape: {output.shape}")
    print(f"  Output streams: {output[0, 0, :, :3]}")  # First 3 features
    
    # For equivalence, output should equal input (identity sublayer + identity norm)
    # Because: h_in = average(streams) = single_stream
    #          h_out = sublayer(norm(h_in)) = h_in (identity)
    #          h_post = H_post * h_out = broadcast to n streams
    #          h_res = H_res * x = x (identity)
    #          output = h_res + h_post = x + broadcast(h_in)
    
    # Wait, this is NOT identity! Let me trace through the math more carefully...
    # Actually, for a layer like attention or MLP that produces OUTPUT, the
    # equivalence is: output should match original_layer(input)
    
    # For identity sublayer: h_in = average of streams = original x
    #                       h_out = identity(x) = x
    #                       h_post = H_post @ x.unsqueeze(-2) = scale x to all streams
    #                       h_res = H_res @ streams = streams (identity H_res)
    #                       output = h_res + h_post
    
    # Let me compute what we expect:
    x_flat = x.view(1, 5, n_streams * hidden_size)
    H_pre, H_post, H_res = mhc_layer.compute_coefficients(x_flat)
    
    print(f"\n  H_pre shape: {H_pre.shape}, values: {H_pre[0, 0, 0, :].tolist()}")
    print(f"  H_post shape: {H_post.shape}, values: {H_post[0, 0, :, 0].tolist()}")
    print(f"  H_res shape: {H_res.shape}")
    print(f"  H_res[0,0]:\n{H_res[0, 0]}")
    
    # H_pre should be ~[0.25, 0.25, 0.25, 0.25] for averaging
    h_pre_expected = torch.tensor([1/n_streams] * n_streams)
    h_pre_diff = (H_pre[0, 0, 0, :] - h_pre_expected).abs().max().item()
    print(f"\n  H_pre difference from uniform: {h_pre_diff:.6e}")
    
    # H_post should be ~[1, 1, 1, 1] for copying
    h_post_expected = torch.tensor([1.0] * n_streams)
    h_post_diff = (H_post[0, 0, :, 0] - h_post_expected).abs().max().item()
    print(f"  H_post difference from 1.0: {h_post_diff:.6e}")
    
    # H_res should be identity
    identity = torch.eye(n_streams)
    h_res_diff = (H_res[0, 0] - identity).abs().max().item()
    print(f"  H_res difference from identity: {h_res_diff:.6e}")
    
    return h_pre_diff < 0.01 and h_post_diff < 0.1 and h_res_diff < 1e-4


def test_actual_model_layer():
    """Test with actual Qwen3 attention module."""
    print("\n" + "="*60)
    print("TEST 4: Testing with Actual Qwen3 Attention")
    print("="*60)
    
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RMSNorm
        from src.qwen3_mhc_model import Qwen3MHCConfig
        
        # Load original model config
        original_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float32,  # Use float32 for debugging
            device_map="cpu",
            trust_remote_code=True,
        )
        
        original_config = original_model.config
        original_layer = original_model.model.layers[0]
        
        # Create mHC config
        mhc_config = Qwen3MHCConfig(
            vocab_size=original_config.vocab_size,
            hidden_size=original_config.hidden_size,
            intermediate_size=original_config.intermediate_size,
            num_hidden_layers=original_config.num_hidden_layers,
            num_attention_heads=original_config.num_attention_heads,
            num_key_value_heads=original_config.num_key_value_heads,
            head_dim=getattr(original_config, 'head_dim', original_config.hidden_size // original_config.num_attention_heads),
            hidden_act=original_config.hidden_act,
            max_position_embeddings=original_config.max_position_embeddings,
            initializer_range=original_config.initializer_range,
            rms_norm_eps=original_config.rms_norm_eps,
            use_cache=original_config.use_cache,
            tie_word_embeddings=original_config.tie_word_embeddings,
            rope_theta=original_config.rope_theta,
            attention_dropout=getattr(original_config, 'attention_dropout', 0.0),
            attention_bias=getattr(original_config, 'attention_bias', False),
            n_streams=4,
            mhc_alpha_init=0.01,
        )
        
        mhc_cfg = mhc_config.get_mhc_config()
        
        # Create MHC layer wrapping original attention
        mhc_attn = MHCLayer(
            config=mhc_cfg,
            sublayer=original_layer.self_attn,
            layernorm=original_layer.input_layernorm,
        )
        
        # Create test input
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        
        # Get embeddings
        input_embeds = original_model.model.embed_tokens(inputs.input_ids)
        
        print(f"  Input embeddings shape: {input_embeds.shape}")
        
        # Get original output (single stream)
        with torch.no_grad():
            # Original path: layernorm -> attention -> residual
            normed = original_layer.input_layernorm(input_embeds)
            attn_output, _, _ = original_layer.self_attn(
                normed,
                attention_mask=None,
                position_ids=None,
            )
            original_output = input_embeds + attn_output
        
        print(f"  Original output shape: {original_output.shape}")
        
        # Get mHC output (multi-stream)
        expand = StreamExpand(4)
        collapse = StreamCollapse(4)
        
        with torch.no_grad():
            x_expanded = expand(input_embeds)
            mhc_output_expanded = mhc_attn(x_expanded)
            mhc_output = collapse(mhc_output_expanded)
        
        print(f"  MHC output shape: {mhc_output.shape}")
        
        # Compare
        diff = (original_output - mhc_output).abs().max().item()
        mean_diff = (original_output - mhc_output).abs().mean().item()
        
        print(f"\n  Max difference: {diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")
        print(f"  ✓ PASS" if diff < 0.1 else f"  ✗ FAIL")
        
        # Debug: Check intermediate values
        print("\n  === Debugging intermediate values ===")
        x = x_expanded
        B, S, n, C = x.shape
        x_flat = x.view(B, S, n * C)
        H_pre, H_post, H_res = mhc_attn.compute_coefficients(x_flat)
        
        print(f"  H_pre (stream weights): {H_pre[0, 0, 0, :].tolist()}")
        print(f"  H_post (scaling): {H_post[0, 0, :, 0].tolist()}")
        
        # What h_in should be (weighted average of streams)
        h_in = torch.matmul(H_pre, x).squeeze(-2)  # (B, S, C)
        h_in_vs_original = (h_in - input_embeds).abs().max().item()
        print(f"  h_in vs original embeddings max diff: {h_in_vs_original:.6e}")
        
        # h_out from sublayer
        h_norm = mhc_attn.layernorm(h_in)
        h_out = mhc_attn._forward_sublayer(h_norm)
        if isinstance(h_out, tuple):
            h_out = h_out[0]
        
        # Compare h_out with original attention output
        h_out_vs_original = (h_out - attn_output).abs().max().item()
        print(f"  h_out vs original attn output max diff: {h_out_vs_original:.6e}")
        
        # Check h_res
        h_res = torch.matmul(H_res, x)
        h_res_vs_original = (h_res[:, :, 0, :] - input_embeds).abs().max().item()
        print(f"  h_res[stream 0] vs original max diff: {h_res_vs_original:.6e}")
        
        # Check identity of H_res
        identity = torch.eye(4)
        h_res_identity_diff = (H_res[0, 0] - identity).abs().max().item()
        print(f"  H_res identity diff: {h_res_identity_diff:.6e}")
        
        del original_model
        
        return diff < 0.1
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mhc_forward_math():
    """Trace through the exact math of MHC forward pass."""
    print("\n" + "="*60)
    print("TEST 5: MHC Forward Pass Mathematical Analysis")
    print("="*60)
    
    n_streams = 4
    hidden_size = 8
    
    config = MHCConfig(
        n_streams=n_streams,
        hidden_size=hidden_size,
        alpha_init=0.01,
        phi_init_std=0.0,
        b_res_diagonal_init=20.0,
    )
    
    # Simple linear sublayer for testing
    class SimpleSublayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.eye(hidden_size) * 2)  # Scale by 2
        
        def forward(self, x, **kwargs):
            return x @ self.weight
    
    class IdentityNorm(nn.Module):
        def forward(self, x):
            return x
    
    mhc_layer = MHCLayer(
        config=config,
        sublayer=SimpleSublayer(),
        layernorm=IdentityNorm(),
    )
    
    # Create simple input where all streams are identical
    x_single = torch.randn(1, 2, hidden_size)
    expand = StreamExpand(n_streams)
    x = expand(x_single)  # All streams identical
    
    print(f"  x_single (first token): {x_single[0, 0, :4].tolist()}")
    print(f"  All streams identical: {torch.allclose(x[:, :, 0, :], x[:, :, 1, :])}")
    
    # Manual forward pass
    B, S, n, C = x.shape
    x_flat = x.view(B, S, n * C)
    
    # Step 1: Compute coefficients
    H_pre, H_post, H_res = mhc_layer.compute_coefficients(x_flat)
    
    print(f"\n  Step 1 - Coefficients:")
    print(f"    H_pre: {H_pre[0, 0, 0, :].tolist()} (should be ~[0.25, 0.25, 0.25, 0.25])")
    print(f"    H_post: {H_post[0, 0, :, 0].tolist()} (should be ~[1, 1, 1, 1])")
    print(f"    H_res diagonal: {[H_res[0, 0, i, i].item() for i in range(n)]}")
    
    # Step 2: Pre-mapping (squeeze n streams to 1)
    h_in = torch.matmul(H_pre, x).squeeze(-2)  # (B, S, C)
    
    print(f"\n  Step 2 - Pre-mapping:")
    print(f"    h_in = H_pre @ x (weighted average)")
    print(f"    h_in (first token): {h_in[0, 0, :4].tolist()}")
    print(f"    x_single (first token): {x_single[0, 0, :4].tolist()}")
    print(f"    h_in == x_single: {torch.allclose(h_in, x_single, atol=1e-5)}")
    
    # Step 3: Sublayer
    h_norm = mhc_layer.layernorm(h_in)
    h_out = mhc_layer.sublayer(h_norm)
    
    print(f"\n  Step 3 - Sublayer:")
    print(f"    h_out = sublayer(h_in) = 2 * h_in")
    print(f"    h_out (first token): {h_out[0, 0, :4].tolist()}")
    expected_out = x_single * 2
    print(f"    Expected (2*x_single): {expected_out[0, 0, :4].tolist()}")
    
    # Step 4: Post-mapping (broadcast to n streams)
    h_out_unsq = h_out.unsqueeze(-2)  # (B, S, 1, C)
    h_post = torch.matmul(H_post, h_out_unsq)  # (B, S, n, C)
    
    print(f"\n  Step 4 - Post-mapping:")
    print(f"    h_post = H_post @ h_out (broadcast with scaling)")
    print(f"    H_post values: {H_post[0, 0, :, 0].tolist()}")
    print(f"    h_post stream 0: {h_post[0, 0, 0, :4].tolist()}")
    print(f"    h_post stream 1: {h_post[0, 0, 1, :4].tolist()}")
    
    # Step 5: Residual mapping
    h_res_out = torch.matmul(H_res, x)  # (B, S, n, C)
    
    print(f"\n  Step 5 - Residual:")
    print(f"    h_res = H_res @ x (stream mixing via doubly-stochastic matrix)")
    print(f"    h_res stream 0: {h_res_out[0, 0, 0, :4].tolist()}")
    print(f"    original x stream 0: {x[0, 0, 0, :4].tolist()}")
    
    # Step 6: Combine
    output = h_res_out + h_post
    
    print(f"\n  Step 6 - Output:")
    print(f"    output = h_res + h_post")
    print(f"    output stream 0: {output[0, 0, 0, :4].tolist()}")
    
    # What should it be for equivalence?
    # Original transformer: output = x + sublayer(layernorm(x))
    # = x + 2*x = 3*x
    expected_single = x_single * 3
    print(f"\n  Expected for equivalence (x + 2x = 3x):")
    print(f"    3*x_single: {expected_single[0, 0, :4].tolist()}")
    
    # After collapse by averaging
    collapse = StreamCollapse(n_streams)
    output_collapsed = collapse(output)
    
    print(f"\n  After StreamCollapse (averaging):")
    print(f"    collapsed output: {output_collapsed[0, 0, :4].tolist()}")
    
    diff = (output_collapsed - expected_single).abs().max().item()
    print(f"\n  Difference from expected: {diff:.6e}")
    print(f"  ✓ PASS" if diff < 0.01 else f"  ✗ FAIL")
    
    return diff < 0.01


if __name__ == "__main__":
    print("="*60)
    print("MHC EQUIVALENCE DEBUG SCRIPT")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Stream Expand/Collapse", test_stream_expand_collapse()))
    results.append(("Sinkhorn Identity", test_sinkhorn_identity()))
    results.append(("MHC Layer Coefficients", test_mhc_layer_equivalence()))
    results.append(("MHC Forward Math", test_mhc_forward_math()))
    results.append(("Actual Model Layer", test_actual_model_layer()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
