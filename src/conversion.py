"""
Weight conversion utilities for transforming standard Qwen3 to mHC architecture.

This module handles:
1. Loading a standard Qwen3 model
2. Creating the mHC model structure
3. Copying original weights to appropriate locations
4. Initializing mHC-specific parameters for equivalence
5. Validating the conversion produces identical outputs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file

from .config import MHCConfig
from .qwen3_mhc_model import Qwen3MHCConfig, Qwen3MHCForCausalLM, Qwen3MHCDecoderLayer

logger = logging.getLogger(__name__)


def convert_qwen3_to_mhc(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    n_streams: int = 4,
    output_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    validate: bool = True,
    validation_tolerance: float = 1e-4,
) -> Tuple[Qwen3MHCForCausalLM, Optional[AutoTokenizer]]:
    """Convert a standard Qwen3 model to mHC architecture.
    
    This function:
    1. Loads the original Qwen3 model
    2. Creates an mHC model with the same architecture
    3. Copies all original weights
    4. Initializes mHC parameters for equivalence
    5. Optionally validates the conversion
    
    Args:
        model_name_or_path: HuggingFace model name or local path.
        n_streams: Number of parallel streams for mHC (default 4).
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
    # Use eager attention for consistency with mHC model (which requires eager)
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
    
    # Create mHC config
    mhc_config = create_mhc_config_from_qwen3(original_config, n_streams)
    
    logger.info(f"Creating mHC model with {n_streams} streams")
    
    # Create mHC model
    mhc_model = create_mhc_model_from_qwen3(
        original_model,
        mhc_config,
        device=device,
        torch_dtype=torch_dtype,
    )
    
    # Validate if requested
    if validate:
        logger.info("Validating equivalence...")
        is_equivalent, max_diff = validate_equivalence(
            original_model,
            mhc_model,
            tokenizer,
            device=device,
            tolerance=validation_tolerance,
        )
        
        if not is_equivalent:
            raise ValueError(
                f"Conversion validation failed! Max logit difference: {max_diff:.6e} "
                f"(tolerance: {validation_tolerance:.6e})"
            )
        
        logger.info(f"Validation passed. Max logit difference: {max_diff:.6e}")
    
    # Save if path provided
    if output_path is not None:
        logger.info(f"Saving converted model to {output_path}")
        save_mhc_model(mhc_model, tokenizer, output_path)
    
    # Clean up original model
    del original_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return mhc_model, tokenizer


def create_mhc_config_from_qwen3(
    qwen3_config: Any,
    n_streams: int = 4,
) -> Qwen3MHCConfig:
    """Create Qwen3MHCConfig from original Qwen3 config.
    
    Args:
        qwen3_config: Original Qwen3 configuration.
        n_streams: Number of mHC streams.
        
    Returns:
        Qwen3MHCConfig with mHC parameters.
    """
    config = Qwen3MHCConfig(
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
        # mHC specific
        n_streams=n_streams,
        mhc_alpha_init=0.01,
        mhc_sinkhorn_iterations=20,
        mhc_sinkhorn_eps=1e-8,
    )
    
    # Force eager attention for mHC model (SDPA not supported for custom architectures)
    # This is functionally equivalent but may affect validation against SDPA models
    config._attn_implementation = "eager"
    
    return config


def create_mhc_model_from_qwen3(
    original_model: nn.Module,
    mhc_config: Qwen3MHCConfig,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Qwen3MHCForCausalLM:
    """Create mHC model and copy weights from original Qwen3.
    
    Args:
        original_model: Original Qwen3 model.
        mhc_config: mHC configuration.
        device: Device for the new model.
        torch_dtype: Data type for weights.
        
    Returns:
        Qwen3MHCForCausalLM with copied weights.
    """
    target_device = torch.device(device)
    
    # Create base mHC model structure (this now creates layers with placeholder weights)
    mhc_model = Qwen3MHCForCausalLM(mhc_config)
    
    # Re-initialize MHC parameters (post_init may have overwritten them)
    for layer in mhc_model.model.layers:
        layer.mhc_attn._init_for_equivalence()
        layer.mhc_mlp._init_for_equivalence()
    
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
    # The layers were already created in __init__, we just need to copy weights
    for layer_idx in range(mhc_config.num_hidden_layers):
        original_layer = original_model.model.layers[layer_idx]
        mhc_layer = mhc_model.model.layers[layer_idx]
        
        # Copy attention weights
        _copy_module_weights(original_layer.self_attn, mhc_layer.mhc_attn.sublayer)
        
        # Copy MLP weights
        _copy_module_weights(original_layer.mlp, mhc_layer.mhc_mlp.sublayer)
        
        # Copy layer norm weights
        _copy_module_weights(original_layer.input_layernorm, mhc_layer.mhc_attn.layernorm)
        _copy_module_weights(original_layer.post_attention_layernorm, mhc_layer.mhc_mlp.layernorm)
    
    # Move entire model to the target device and dtype
    mhc_model = mhc_model.to(device=target_device, dtype=torch_dtype)
    
    # Copy rotary embeddings AFTER dtype conversion
    # inv_freq is a buffer that should stay in float32 for precision
    # The original model keeps inv_freq as float32 even when model is bfloat16
    if hasattr(original_model.model, 'rotary_emb') and original_model.model.rotary_emb is not None:
        if hasattr(original_model.model.rotary_emb, 'inv_freq'):
            mhc_model.model.rotary_emb.inv_freq = original_model.model.rotary_emb.inv_freq.clone()
    
    # Verify mHC parameters are initialized for equivalence
    _verify_mhc_initialization(mhc_model)
    
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



def _verify_mhc_initialization(model: Qwen3MHCForCausalLM):
    """Verify mHC parameters are correctly initialized for equivalence.
    
    Args:
        model: The mHC model to verify.
    """
    # Use larger tolerance for bfloat16
    tol = 1e-2
    
    for layer_idx, layer in enumerate(model.model.layers):
        for name, mhc_layer in [("attn", layer.mhc_attn), ("mlp", layer.mhc_mlp)]:
            # Check alpha values (allow tolerance for dtype conversion)
            alpha_pre_val = mhc_layer.alpha_pre.float().item()
            alpha_post_val = mhc_layer.alpha_post.float().item()
            alpha_res_val = mhc_layer.alpha_res.float().item()
            
            assert abs(alpha_pre_val - 0.01) < tol, \
                f"Layer {layer_idx} {name}: alpha_pre={alpha_pre_val}, expected 0.01"
            assert abs(alpha_post_val - 0.01) < tol, \
                f"Layer {layer_idx} {name}: alpha_post={alpha_post_val}, expected 0.01"
            assert abs(alpha_res_val - 0.01) < tol, \
                f"Layer {layer_idx} {name}: alpha_res={alpha_res_val}, expected 0.01"
            
            # Check phi weights are near zero (use tolerance for dtype)
            phi_pre_max = mhc_layer.phi_pre.weight.float().abs().max().item()
            phi_post_max = mhc_layer.phi_post.weight.float().abs().max().item()
            phi_res_max = mhc_layer.phi_res.weight.float().abs().max().item()
            
            assert phi_pre_max < tol, \
                f"Layer {layer_idx} {name}: phi_pre max={phi_pre_max}, expected ~0"
            assert phi_post_max < tol, \
                f"Layer {layer_idx} {name}: phi_post max={phi_post_max}, expected ~0"
            assert phi_res_max < tol, \
                f"Layer {layer_idx} {name}: phi_res max={phi_res_max}, expected ~0"


def validate_equivalence(
    original_model: nn.Module,
    mhc_model: nn.Module,
    tokenizer: Any,
    device: str = "cuda",
    tolerance: float = 0.5,
    test_sequences: Optional[list] = None,
) -> Tuple[bool, float]:
    """Validate that mHC model produces same outputs as original.
    
    Args:
        original_model: Original Qwen3 model.
        mhc_model: Converted mHC model.
        tokenizer: Tokenizer for encoding test sequences.
        device: Device to run validation on.
        tolerance: Maximum allowed difference in logits (default 0.5 for bfloat16).
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
            
            # Get mHC outputs
            mhc_outputs = mhc_model(**inputs)
            mhc_logits = mhc_outputs.logits
            
            # Compute difference
            diff = (original_logits - mhc_logits).abs().max().item()
            max_diff = max(max_diff, diff)
            
            logger.debug(f"Sequence '{seq[:30]}...': max diff = {diff:.6e}")
    
    is_equivalent = max_diff < tolerance
    
    return is_equivalent, max_diff


def save_mhc_model(
    model: Qwen3MHCForCausalLM,
    tokenizer: Any,
    output_path: str,
    use_safetensors: bool = True,
):
    """Save mHC model and tokenizer.
    
    Args:
        model: mHC model to save.
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


def load_mhc_model(
    model_path: str,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    base_model: str = "Qwen/Qwen3-0.6B",
) -> Tuple[Qwen3MHCForCausalLM, Any]:
    """Load a saved mHC model.
    
    Args:
        model_path: Path to saved model.
        device: Device to load to.
        torch_dtype: Data type for weights.
        base_model: Base model to load tokenizer from (mHC models don't have their own tokenizer).
        
    Returns:
        Tuple of (model, tokenizer).
    """
    model = Qwen3MHCForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    
    # Load tokenizer from base model since mHC uses same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_weight_mapping() -> Dict[str, str]:
    """Get mapping from original Qwen3 weight names to mHC weight names.
    
    Returns:
        Dictionary mapping original names to mHC names.
    """
    # Base mappings (unchanged)
    mapping = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Layer mappings
    # Original: model.layers[i].{component}
    # mHC: model.layers[i].mhc_{attn|mlp}.sublayer.{component}
    # or: model.layers[i].mhc_{attn|mlp}.layernorm.{component}
    
    return mapping


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model by category.
    
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
        
        # Check if this is an mHC parameter
        if any(x in name for x in ['phi_pre', 'phi_post', 'phi_res', 
                                    'b_pre', 'b_post', 'b_res',
                                    'alpha_pre', 'alpha_post', 'alpha_res',
                                    'coef_norm']):
            mhc_params += count
        else:
            original_params += count
    
    return {
        "total": total,
        "mhc": mhc_params,
        "original": original_params,
        "mhc_percentage": 100 * mhc_params / total if total > 0 else 0,
    }


def print_model_summary(model: Qwen3MHCForCausalLM):
    """Print summary of mHC model.
    
    Args:
        model: mHC model to summarize.
    """
    config = model.config
    param_counts = count_parameters(model)
    
    print("=" * 60)
    print("Qwen3 mHC Model Summary")
    print("=" * 60)
    print(f"Hidden size:        {config.hidden_size}")
    print(f"Num layers:         {config.num_hidden_layers}")
    print(f"Num attention heads:{config.num_attention_heads}")
    print(f"Num KV heads:       {config.num_key_value_heads}")
    print(f"Vocab size:         {config.vocab_size}")
    print("-" * 60)
    print(f"mHC streams:        {config.n_streams}")
    print(f"Expanded hidden:    {config.n_streams * config.hidden_size}")
    print(f"mHC alpha init:     {config.mhc_alpha_init}")
    print(f"Sinkhorn iters:     {config.mhc_sinkhorn_iterations}")
    print("-" * 60)
    print(f"Total parameters:   {param_counts['total']:,}")
    print(f"Original params:    {param_counts['original']:,}")
    print(f"mHC params:         {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.2f}%)")
    print("=" * 60)
