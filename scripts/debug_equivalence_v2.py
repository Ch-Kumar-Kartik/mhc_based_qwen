"""
Debug script v3 - Tests actual model layer with proper API calls.

Traces through the full model conversion to identify equivalence issues.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/ch_kumar_kartik/qwen3-mhc')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MHCConfig
from src.mhc_layer import MHCLayer
from src.stream_ops import StreamExpand, StreamCollapse
from src.conversion import convert_qwen3_to_mhc, create_mhc_config_from_qwen3, create_mhc_model_from_qwen3
from src.qwen3_mhc_model import Qwen3MHCConfig, Qwen3MHCForCausalLM


def test_single_layer_detailed():
    """Test a single layer conversion in detail."""
    print("\n" + "="*60)
    print("TEST: Single Layer Detailed Comparison")
    print("="*60)
    
    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,  # float32 for precision debugging
        device_map="cpu",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Create input
    inputs = tokenizer("Hello world", return_tensors="pt")
    input_ids = inputs.input_ids
    
    print(f"Input: {tokenizer.decode(input_ids[0])}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Run original model layer by layer
    print("\n--- Original Model Forward Pass ---")
    with torch.no_grad():
        original_model.eval()
        
        # Get embeddings
        embeds = original_model.model.embed_tokens(input_ids)
        print(f"Embeddings shape: {embeds.shape}")
        print(f"Embeddings sample: {embeds[0, 0, :5].tolist()}")
        
        # Get position info
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cache_position = torch.arange(seq_len)
        
        # Get rotary embeddings
        rotary_emb = original_model.model.rotary_emb
        position_embeddings = rotary_emb(embeds, position_ids)
        print(f"Position embeddings (cos, sin) shapes: {position_embeddings[0].shape}, {position_embeddings[1].shape}")
        
        # Run through first layer only
        layer0 = original_model.model.layers[0]
        
        # Save input
        layer0_input = embeds.clone()
        
        # LayerNorm before attention
        attn_input = layer0.input_layernorm(embeds)
        print(f"After layernorm sample: {attn_input[0, 0, :5].tolist()}")
        
        # Attention - correct signature: (hidden_states, position_embeddings, attention_mask, ...)
        attn_output = layer0.self_attn(
            attn_input,
            position_embeddings,  # positional arg
            None,  # attention_mask - positional arg
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        print(f"Attention output sample: {attn_output[0, 0, :5].tolist()}")
        
        # Residual
        hidden = embeds + attn_output
        print(f"After attn residual sample: {hidden[0, 0, :5].tolist()}")
        
        # LayerNorm before MLP
        mlp_input = layer0.post_attention_layernorm(hidden)
        print(f"After post_attn layernorm sample: {mlp_input[0, 0, :5].tolist()}")
        
        # MLP
        mlp_output = layer0.mlp(mlp_input)
        print(f"MLP output sample: {mlp_output[0, 0, :5].tolist()}")
        
        # Residual
        layer0_output = hidden + mlp_output
        print(f"Layer 0 final output sample: {layer0_output[0, 0, :5].tolist()}")
    
    # Now create mHC model and compare
    print("\n--- Creating mHC Model ---")
    mhc_config = create_mhc_config_from_qwen3(original_model.config, n_streams=4)
    print(f"mHC config n_streams: {mhc_config.n_streams}")
    
    mhc_model = create_mhc_model_from_qwen3(
        original_model,
        mhc_config,
        device="cpu",
        torch_dtype=torch.float32,
    )
    
    print("\n--- mHC Model Forward Pass ---")
    with torch.no_grad():
        mhc_model.eval()
        
        # Get embeddings (should be same as original)
        mhc_embeds = mhc_model.model.embed_tokens(input_ids)
        embed_diff = (embeds - mhc_embeds).abs().max().item()
        print(f"Embedding difference: {embed_diff:.6e}")
        
        # Expand to multi-stream
        expanded = mhc_model.model.stream_expand(mhc_embeds)
        print(f"Expanded shape: {expanded.shape}")
        
        # Get mHC layer 0
        mhc_layer0 = mhc_model.model.layers[0]
        
        # Check MHC parameters
        print(f"\nmHC Attention layer parameters:")
        print(f"  alpha_pre: {mhc_layer0.mhc_attn.alpha_pre.item():.6f}")
        print(f"  alpha_post: {mhc_layer0.mhc_attn.alpha_post.item():.6f}")
        print(f"  alpha_res: {mhc_layer0.mhc_attn.alpha_res.item():.6f}")
        print(f"  phi_pre max: {mhc_layer0.mhc_attn.phi_pre.weight.abs().max().item():.6e}")
        print(f"  phi_post max: {mhc_layer0.mhc_attn.phi_post.weight.abs().max().item():.6e}")
        print(f"  phi_res max: {mhc_layer0.mhc_attn.phi_res.weight.abs().max().item():.6e}")
        
        # Get position embeddings for mHC model
        mhc_rotary = mhc_model.model.rotary_emb
        mhc_position_embeddings = mhc_rotary(mhc_embeds, position_ids)
        
        # Run mHC layer 0
        mhc_layer0_output = mhc_layer0(
            expanded,
            position_embeddings=mhc_position_embeddings,
        )
        if isinstance(mhc_layer0_output, tuple):
            mhc_layer0_output = mhc_layer0_output[0]
        
        print(f"mHC layer 0 output shape: {mhc_layer0_output.shape}")
        
        # Collapse for comparison
        collapsed = mhc_model.model.stream_collapse(mhc_layer0_output)
        print(f"Collapsed shape: {collapsed.shape}")
        print(f"Collapsed sample: {collapsed[0, 0, :5].tolist()}")
        
        # Compare
        layer_diff = (layer0_output - collapsed).abs().max().item()
        print(f"\n*** Layer 0 output difference: {layer_diff:.6e} ***")
        
        if layer_diff > 0.001:
            print("\nDebugging why layer outputs differ...")
            
            # Check attention sublayer weights match
            orig_attn = layer0.self_attn
            mhc_attn_sublayer = mhc_layer0.mhc_attn.sublayer
            
            # Check Q projection
            q_diff = (orig_attn.q_proj.weight - mhc_attn_sublayer.q_proj.weight).abs().max().item()
            print(f"  Q projection weight diff: {q_diff:.6e}")
            
            # Check layernorm weights match
            ln_diff = (layer0.input_layernorm.weight - mhc_layer0.mhc_attn.layernorm.weight).abs().max().item()
            print(f"  Input layernorm weight diff: {ln_diff:.6e}")
            
            # Trace through manually
            B, S, n, C = expanded.shape
            x_flat = expanded.view(B, S, n * C)
            H_pre, H_post, H_res = mhc_layer0.mhc_attn.compute_coefficients(x_flat)
            
            print(f"\n  H_pre values: {H_pre[0, 0, 0, :].tolist()}")
            print(f"  H_post values: {H_post[0, 0, :, 0].tolist()}")
            print(f"  H_res diagonal: {[H_res[0, 0, i, i].item() for i in range(n)]}")
            
            # h_in = weighted average
            h_in = torch.matmul(H_pre, expanded).squeeze(-2)
            h_in_vs_orig = (h_in - mhc_embeds).abs().max().item()
            print(f"  h_in vs original embeds: {h_in_vs_orig:.6e}")
            
            # After layernorm
            h_norm = mhc_layer0.mhc_attn.layernorm(h_in)
            norm_diff = (h_norm - attn_input).abs().max().item()
            print(f"  h_norm vs original attn_input: {norm_diff:.6e}")
            
            # After attention sublayer - use correct signature
            h_out = mhc_layer0.mhc_attn.sublayer(h_norm, mhc_position_embeddings, None)
            if isinstance(h_out, tuple):
                h_out = h_out[0]
            attn_diff = (h_out - attn_output).abs().max().item()
            print(f"  h_out (mhc attn) vs original attn_output: {attn_diff:.6e}")
            
            # h_post (broadcast)
            h_out_unsq = h_out.unsqueeze(-2)
            h_post_out = torch.matmul(H_post, h_out_unsq)
            print(f"  h_post shape: {h_post_out.shape}")
            
            # h_res (should be identity on expanded)
            h_res_out = torch.matmul(H_res, expanded)
            h_res_vs_expanded = (h_res_out - expanded).abs().max().item()
            print(f"  h_res vs expanded (should be ~same): {h_res_vs_expanded:.6e}")
            
            # Final mhc attn output
            mhc_attn_out = h_res_out + h_post_out
            
            print(f"\n  Checking math...")
            # Expected collapsed = embeds + attn_output  
            expected = mhc_embeds + h_out
            exp_diff = (expected - layer0_output[:, :, :]).abs().max().item()
            print(f"  Expected (embeds + h_out) vs original layer0 output (after attn only): {exp_diff:.6e}")
            
    # Clean up
    del original_model
    del mhc_model
    
    return layer_diff < 0.1


def test_layer_by_layer_accumulation():
    """Test layer-by-layer to see where error accumulates."""
    print("\n" + "="*60)
    print("TEST: Layer-by-Layer Error Accumulation")
    print("="*60)
    
    # Load models
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Create mHC model
    print("Creating mHC model...")
    mhc_config = create_mhc_config_from_qwen3(original_model.config, n_streams=4)
    mhc_model = create_mhc_model_from_qwen3(
        original_model,
        mhc_config,
        device="cpu",
        torch_dtype=torch.float32,
    )
    
    # Test input
    inputs = tokenizer("Hi", return_tensors="pt")
    input_ids = inputs.input_ids
    
    n_layers = original_model.config.num_hidden_layers
    print(f"Number of layers: {n_layers}")
    
    with torch.no_grad():
        original_model.eval()
        mhc_model.eval()
        
        # Setup position info
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0)
        
        # Original embeddings
        orig_hidden = original_model.model.embed_tokens(input_ids)
        
        # MHC embeddings
        mhc_hidden = mhc_model.model.embed_tokens(input_ids)
        mhc_expanded = mhc_model.model.stream_expand(mhc_hidden)
        
        print(f"\nInitial embedding diff: {(orig_hidden - mhc_hidden).abs().max().item():.6e}")
        
        # Get position embeddings
        orig_pos = original_model.model.rotary_emb(orig_hidden, position_ids)
        mhc_pos = mhc_model.model.rotary_emb(mhc_hidden, position_ids)
        
        # Process layer by layer
        print("\nLayer-by-layer comparison:")
        print("-" * 60)
        
        for layer_idx in range(min(n_layers, 5)):  # First 5 layers
            # Original layer
            orig_layer = original_model.model.layers[layer_idx]
            
            # Attention residual - correct signature
            attn_norm = orig_layer.input_layernorm(orig_hidden)
            attn_out = orig_layer.self_attn(attn_norm, orig_pos, None)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            orig_hidden = orig_hidden + attn_out
            
            # MLP residual  
            mlp_norm = orig_layer.post_attention_layernorm(orig_hidden)
            mlp_out = orig_layer.mlp(mlp_norm)
            orig_hidden = orig_hidden + mlp_out
            
            # MHC layer
            mhc_layer = mhc_model.model.layers[layer_idx]
            mhc_layer_out = mhc_layer(mhc_expanded, position_embeddings=mhc_pos)
            if isinstance(mhc_layer_out, tuple):
                mhc_layer_out = mhc_layer_out[0]
            mhc_expanded = mhc_layer_out
            
            # Collapse for comparison
            mhc_collapsed = mhc_model.model.stream_collapse(mhc_expanded)
            
            diff = (orig_hidden - mhc_collapsed).abs().max().item()
            mean_diff = (orig_hidden - mhc_collapsed).abs().mean().item()
            
            print(f"Layer {layer_idx:2d}: max_diff = {diff:.6e}, mean_diff = {mean_diff:.6e}")
    
    del original_model
    del mhc_model
    
    return True


def test_full_model():
    """Test full model conversion."""
    print("\n" + "="*60)
    print("TEST: Full Model Comparison (single forward pass)")
    print("="*60)
    
    # Load models
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    
    # Create mHC model
    print("Creating mHC model...")
    mhc_config = create_mhc_config_from_qwen3(original_model.config, n_streams=4)
    mhc_model = create_mhc_model_from_qwen3(
        original_model,
        mhc_config,
        device="cpu",
        torch_dtype=torch.float32,
    )
    
    # Test input
    inputs = tokenizer("The quick brown fox", return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        original_model.eval()
        mhc_model.eval()
        
        orig_outputs = original_model(**inputs)
        mhc_outputs = mhc_model(**inputs)
        
        orig_logits = orig_outputs.logits
        mhc_logits = mhc_outputs.logits
        
        diff = (orig_logits - mhc_logits).abs().max().item()
        mean_diff = (orig_logits - mhc_logits).abs().mean().item()
        
        print(f"Logits shape: {orig_logits.shape}")
        print(f"Max difference: {diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")
        
        # Check per-position differences
        print("\nPer-position max differences:")
        for i in range(orig_logits.shape[1]):
            pos_diff = (orig_logits[0, i] - mhc_logits[0, i]).abs().max().item()
            print(f"  Position {i}: {pos_diff:.6e}")
    
    del original_model
    del mhc_model
    
    return diff < 0.1


if __name__ == "__main__":
    print("="*60)
    print("MHC EQUIVALENCE DEBUG SCRIPT V3")
    print("="*60)
    
    # Run tests
    test_single_layer_detailed()
    test_layer_by_layer_accumulation()
    test_full_model()
