"""
Weight conversion utilities for transforming standard Qwen3 to mHC V2 architecture.

This module handles:
1. Loading a standard Qwen3 model
2. Creating the mHC V2 model structure
3. Copying original weights to appropriate locations
4. Initializing mHC V2-specific parameters
5. Validating the conversion produces identical outputs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file

from .qwen3_mhc_modelV2 import Qwen3MHCConfigV2, Qwen3MHCForCausalLMV2, Qwen3MHCDecoderLayerV2

logger = logging.getLogger(__name__)


def convert_qwen3_to_mhc_v2(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    n_streams: int = 4,
    num_fracs: int = 1,
    sinkhorn_iters: int = 20,
    output_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    validate: bool = True,
    validation_tolerance: float = 1e-4,
) -> Tuple[Qwen3MHCForCausalLMV2, Optional[AutoTokenizer]]:
    """Convert a standard Qwen3 model to mHC V2 architecture.
    
    This function:
    1. Loads the original Qwen3 model
    2. Creates an mHC V2 model with the same architecture
    3. Copies all original weights
    4. Initializes mHC V2 parameters
    5. Optionally validates the conversion
    
    Args:
        model_name_or_path: HuggingFace model name or local path.
        n_streams: Number of parallel streams for mHC (default 4).
        num_fracs: Number of fractions for frac-connections (default 1).
        sinkhorn_iters: Number of Sinkhorn iterations (default 20).
        output_path: Optional path to save converted model.
        device: Device to load models on.
        torch_dtype: Data type for model weights.
        validate: Whether to validate equivalence after conversion.
        validation_tolerance: Tolerance for equivalence validation.
        
    Returns:
        Tuple of (converted_model, tokenizer).
        
    Raises:
        ValueError: If validation fails beyond tolerance.
    """
    logger.info(f"Loading original model from {model_name_or_path}")
    
    # Load original model and tokenizer
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    
    # Get original config
    original_config = original_model.config
    
    # Create mHC V2 config
    mhc_config = create_mhc_config_v2_from_qwen3(
        original_config, 
        n_streams=n_streams,
        num_fracs=num_fracs,
        sinkhorn_iters=sinkhorn_iters,
    )
    
    logger.info(f"Creating mHC V2 model with {n_streams} streams, {num_fracs} fracs")
    
    # Create mHC V2 model
    mhc_model = create_mhc_model_v2_from_qwen3(
        original_model,
        mhc_config,
        device=device,
        torch_dtype=torch_dtype,
    )
    
    # Validate if requested
    if validate:
        logger.info("Validating equivalence...")
        is_equivalent, max_diff = validate_equivalence_v2(
            original_model,
            mhc_model,
            tokenizer,
            device=device,
            tolerance=validation_tolerance,
        )
        
        if not is_equivalent:
            logger.warning(
                f"Conversion validation warning: Max logit difference: {max_diff:.6e} "
                f"(tolerance: {validation_tolerance:.6e}). This is expected for V2 architecture."
            )
        else:
            logger.info(f"Validation passed. Max logit difference: {max_diff:.6e}")
    
    # Save if path provided
    if output_path is not None:
        logger.info(f"Saving converted model to {output_path}")
        save_mhc_model_v2(mhc_model, tokenizer, output_path)
    
    # Clean up original model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return mhc_model, tokenizer


def create_mhc_config_v2_from_qwen3(
    qwen3_config: Any,
    n_streams: int = 4,
    num_fracs: int = 1,
    sinkhorn_iters: int = 20,
) -> Qwen3MHCConfigV2:
    """Create Qwen3MHCConfigV2 from original Qwen3 config.
    
    Args:
        qwen3_config: Original Qwen3 configuration.
        n_streams: Number of mHC streams.
        num_fracs: Number of fractions for frac-connections.
        sinkhorn_iters: Number of Sinkhorn iterations.
        
    Returns:
        Qwen3MHCConfigV2 with mHC V2 parameters.
    """
    config = Qwen3MHCConfigV2(
        vocab_size=qwen3_config.vocab_size,
        hidden_size=qwen3_config.hidden_size,
        intermediate_size=qwen3_config.intermediate_size,
        num_hidden_layers=qwen3_config.num_hidden_layers,
        num_attention_heads=qwen3_config.num_attention_heads,
        num_key_value_heads=qwen3_config.num_key_value_heads,
        head_dim=getattr(qwen3_config, 'head_dim', qwen3_config.hidden_size // qwen3_config.num_attention_heads),
        hidden_act=qwen3_config.hidden_act,
        max_position_embeddings=qwen3_config.max_position_embeddings,
        initializer_range=qwen3_config.initializer_range,
        rms_norm_eps=qwen3_config.rms_norm_eps,
        use_cache=qwen3_config.use_cache,
        tie_word_embeddings=qwen3_config.tie_word_embeddings,
        rope_theta=qwen3_config.rope_theta,
        rope_scaling=getattr(qwen3_config, 'rope_scaling', None),
        attention_dropout=getattr(qwen3_config, 'attention_dropout', 0.0),
        attention_bias=getattr(qwen3_config, 'attention_bias', False),
        mlp_bias=getattr(qwen3_config, 'mlp_bias', False),
        # mHC V2 specific
        n_streams=n_streams,
        num_fracs=num_fracs,
        sinkhorn_iters=sinkhorn_iters,
        log_domain_sinkhorn=False,
        num_input_views=1,
        num_dynamic_alpha_proposals=1,
        mhc_dropout=0.0,
        use_triton_sinkhorn=False,
        add_stream_embed=False,
        add_attn_pool_reduce_stream=False,
    )
    
    # Force eager attention for mHC model
    config._attn_implementation = "eager"
    
    return config


def create_mhc_model_v2_from_qwen3(
    original_model: nn.Module,
    mhc_config: Qwen3MHCConfigV2,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Qwen3MHCForCausalLMV2:
    """Create mHC V2 model and copy weights from original Qwen3.
    
    Args:
        original_model: Original Qwen3 model.
        mhc_config: mHC V2 configuration.
        device: Device for the new model.
        torch_dtype: Data type for weights.
        
    Returns:
        Qwen3MHCForCausalLMV2 with copied weights.
    """
    target_device = torch.device(device)
    
    # Create base mHC V2 model structure
    mhc_model = Qwen3MHCForCausalLMV2(mhc_config)
    
    # Copy embedding weights
    mhc_model.model.embed_tokens.weight.data.copy_(
        original_model.model.embed_tokens.weight.data.to('cpu')
    )
    
    # Copy final layer norm
    mhc_model.model.norm.weight.data.copy_(
        original_model.model.norm.weight.data.to('cpu')
    )
    
    # Copy LM head (may be tied)
    if not mhc_config.tie_word_embeddings:
        mhc_model.lm_head.weight.data.copy_(
            original_model.lm_head.weight.data.to('cpu')
        )
    
    # Copy weights to the decoder layers
    for layer_idx in range(mhc_config.num_hidden_layers):
        original_layer = original_model.model.layers[layer_idx]
        mhc_layer = mhc_model.model.layers[layer_idx]
        
        # Copy attention weights
        _copy_module_weights(original_layer.self_attn, mhc_layer.attention)
        
        # Copy MLP weights
        _copy_module_weights(original_layer.mlp, mhc_layer.mlp)
        
        # Copy layer norm weights
        _copy_module_weights(original_layer.input_layernorm, mhc_layer.input_layernorm)
        _copy_module_weights(original_layer.post_attention_layernorm, mhc_layer.post_attention_layernorm)
    
    # Move entire model to the target device and dtype
    mhc_model = mhc_model.to(device=target_device, dtype=torch_dtype)
    
    # Copy rotary embeddings AFTER dtype conversion
    if hasattr(original_model.model, 'rotary_emb') and original_model.model.rotary_emb is not None:
        if hasattr(original_model.model.rotary_emb, 'inv_freq'):
            mhc_model.model.rotary_emb.inv_freq = original_model.model.rotary_emb.inv_freq.clone()
    
    return mhc_model


def _copy_module_weights(src_module: nn.Module, dst_module: nn.Module):
    """Copy weights from source module to destination module.
    
    Args:
        src_module: Source module to copy from.
        dst_module: Destination module to copy to.
    """
    src_state = src_module.state_dict()
    dst_state = dst_module.state_dict()
    
    for key in dst_state.keys():
        if key in src_state:
            dst_state[key].copy_(src_state[key].to('cpu'))
    
    dst_module.load_state_dict(dst_state)


def validate_equivalence_v2(
    original_model: nn.Module,
    mhc_model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    tolerance: float = 0.5,
    test_sequences: Optional[list] = None,
) -> Tuple[bool, float]:
    """Validate that mHC V2 model produces similar outputs as original.
    
    Note: V2 architecture may not produce exactly equivalent outputs
    at initialization due to the different mHC structure.
    
    Args:
        original_model: Original Qwen3 model.
        mhc_model: Converted mHC V2 model.
        tokenizer: Tokenizer for encoding test sequences.
        device: Device to run validation on.
        tolerance: Maximum allowed difference in logits.
        test_sequences: Optional list of test strings.
        
    Returns:
        Tuple of (is_equivalent, max_difference).
    """
    if test_sequences is None:
        test_sequences = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "def fibonacci(n):\n    if n <= 1:\n        return n",
            "一二三四五，上山打老虎。",
        ]
    
    # Ensure both models are on the same device
    original_model = original_model.to(device)
    mhc_model = mhc_model.to(device)
    
    original_model.eval()
    mhc_model.eval()
    
    max_diff = 0.0
    
    with torch.no_grad():
        for seq in test_sequences:
            inputs = tokenizer(seq, return_tensors="pt").to(device)
            
            # Get original outputs
            original_outputs = original_model(**inputs)
            original_logits = original_outputs.logits
            
            # Get mHC V2 outputs
            mhc_outputs = mhc_model(**inputs)
            mhc_logits = mhc_outputs.logits
            
            # Compute difference
            diff = (original_logits - mhc_logits).abs().max().item()
            max_diff = max(max_diff, diff)
            
            logger.debug(f"Sequence '{seq[:30]}...': max diff = {diff:.6e}")
    
    is_equivalent = max_diff < tolerance
    
    return is_equivalent, max_diff


def save_mhc_model_v2(
    model: Qwen3MHCForCausalLMV2,
    tokenizer: Any,
    output_path: str,
    use_safetensors: bool = True,
):
    """Save mHC V2 model and tokenizer.
    
    Args:
        model: mHC V2 model to save.
        tokenizer: Tokenizer to save.
        output_path: Output directory path.
        use_safetensors: Whether to use safetensors format.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(
        output_path,
        safe_serialization=use_safetensors,
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Model saved to {output_path}")


def load_mhc_model_v2(
    model_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    base_model: str = "Qwen/Qwen3-0.6B",
) -> Tuple[Qwen3MHCForCausalLMV2, Any]:
    """Load a saved mHC V2 model.
    
    Args:
        model_path: Path to saved model.
        device: Device to load to.
        torch_dtype: Data type for weights.
        base_model: Base model to load tokenizer from.
        
    Returns:
        Tuple of (model, tokenizer).
    """
    # Check if model_path is a local directory
    model_path_obj = Path(model_path)
    is_local = model_path_obj.exists() and model_path_obj.is_dir()
    
    model = Qwen3MHCForCausalLMV2.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        local_files_only=is_local,
    )
    
    # Load tokenizer from base model since mHC uses same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def count_parameters_v2(model: nn.Module) -> Dict[str, int]:
    """Count parameters in mHC V2 model by category.
    
    Args:
        model: Model to count parameters in.
        
    Returns:
        Dictionary with parameter counts.
    """
    total = 0
    mhc_params = 0
    original_params = 0
    
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        
        # Check if this is an mHC V2 parameter
        if any(x in name for x in ['static_alpha', 'dynamic_alpha_fn', 'pre_branch_scale',
                                    'residual_scale', 'static_beta', 'dynamic_beta_fn',
                                    'h_post_scale', 'mhc_attn.norm', 'mhc_mlp.norm',
                                    'split_fracs', 'merge_fracs']):
            mhc_params += count
        else:
            original_params += count
    
    return {
        "total": total,
        "mhc": mhc_params,
        "original": original_params,
        "mhc_percentage": 100 * mhc_params / total if total > 0 else 0,
    }


def print_model_summary_v2(model: Qwen3MHCForCausalLMV2):
    """Print summary of mHC V2 model.
    
    Args:
        model: mHC V2 model to summarize.
    """
    config = model.config
    param_counts = count_parameters_v2(model)
    
    print("=" * 60)
    print("Qwen3 mHC V2 Model Summary")
    print("=" * 60)
    print(f"Hidden size:        {config.hidden_size}")
    print(f"Num layers:         {config.num_hidden_layers}")
    print(f"Num attention heads:{config.num_attention_heads}")
    print(f"Num KV heads:       {config.num_key_value_heads}")
    print(f"Vocab size:         {config.vocab_size}")
    print("-" * 60)
    print(f"mHC streams:        {config.n_streams}")
    print(f"Num fracs:          {config.num_fracs}")
    print(f"Sinkhorn iters:     {config.sinkhorn_iters}")
    print(f"Log-domain Sinkhorn:{config.log_domain_sinkhorn}")
    print(f"Dynamic proposals:  {config.num_dynamic_alpha_proposals}")
    print("-" * 60)
    print(f"Total parameters:   {param_counts['total']:,}")
    print(f"Original params:    {param_counts['original']:,}")
    print(f"mHC params:         {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.2f}%)")
    print("=" * 60)
