"""
Qwen3 model with Manifold-Constrained Hyper-Connections V2 (mHC V2).

This module provides mHC-enhanced versions of Qwen3 model components using
the V2 implementation based on the hyper-connections paper approach,
transforming the single-stream residual architecture into a multi-stream
architecture with learned inter-stream mixing via Sinkhorn-constrained matrices.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Any
from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from .mhc_layerV2 import (
    ManifoldConstrainedHyperConnectionsV2,
    mHCV2,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
    Residual,
    RMSNorm,
)


class Qwen3MHCConfigV2(PretrainedConfig):
    """Configuration for Qwen3 with mHC V2.
    
    Extends the base Qwen3 config with mHC V2-specific parameters.
    Includes all Qwen3 config attributes needed by Qwen3Attention and Qwen3MLP.
    """
    model_type = "qwen3_mhc_v2"
    
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
        attention_dropout: float = 0.0,
        # Additional Qwen3 config attributes needed by Qwen3Attention
        attention_bias: bool = False,
        mlp_bias: bool = False,
        # Sliding window attention config (required by Qwen3Attention)
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        layer_types: list = None,
        # mHC V2 specific
        n_streams: int = 4,
        num_fracs: int = 1,  # Frac-connections paper parameter
        sinkhorn_iters: int = 20,
        log_domain_sinkhorn: bool = False,
        num_input_views: int = 1,
        num_dynamic_alpha_proposals: int = 1,
        mhc_dropout: float = 0.0,
        use_triton_sinkhorn: bool = False,
        add_stream_embed: bool = False,
        add_attn_pool_reduce_stream: bool = False,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        
        # Sliding window attention config (required by Qwen3Attention)
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        
        # Initialize layer_types like Qwen3Config does
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        
        # mHC V2 parameters
        self.n_streams = n_streams
        self.num_fracs = num_fracs
        self.sinkhorn_iters = sinkhorn_iters
        self.log_domain_sinkhorn = log_domain_sinkhorn
        self.num_input_views = num_input_views
        self.num_dynamic_alpha_proposals = num_dynamic_alpha_proposals
        self.mhc_dropout = mhc_dropout
        self.use_triton_sinkhorn = use_triton_sinkhorn
        self.add_stream_embed = add_stream_embed
        self.add_attn_pool_reduce_stream = add_attn_pool_reduce_stream


class Qwen3MHCDecoderLayerV2(nn.Module):
    """Qwen3 decoder layer with mHC V2 wrappers.
    
    Wraps both the attention and MLP sublayers with ManifoldConstrainedHyperConnectionsV2,
    enabling multi-stream residual computation with Sinkhorn-constrained mixing.
    
    Structure:
        mhc_attn: mHCV2 wrapping [input_layernorm + self_attn]
        mhc_mlp: mHCV2 wrapping [post_attention_layernorm + mlp]
    """
    
    def __init__(
        self,
        config: Qwen3MHCConfigV2,
        layer_idx: int,
        attention_module: nn.Module,
        mlp_module: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
    ):
        """Initialize mHC V2 decoder layer.
        
        Args:
            config: Model configuration.
            layer_idx: Index of this layer.
            attention_module: Original attention module.
            mlp_module: Original MLP module.
            input_layernorm: LayerNorm before attention.
            post_attention_layernorm: LayerNorm before MLP.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_streams = config.n_streams
        
        # Create wrapped attention branch (layernorm + attention)
        self.input_layernorm = input_layernorm
        self.attention = attention_module
        
        # Create wrapped MLP branch (layernorm + mlp)
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp_module
        
        # Create mHC V2 layers for attention and MLP
        self.mhc_attn = ManifoldConstrainedHyperConnectionsV2(
            num_residual_streams=config.n_streams,
            dim=config.hidden_size,
            layer_index=layer_idx * 2,  # Each decoder layer has 2 mHC sublayers
            dropout=config.mhc_dropout,
            num_fracs=config.num_fracs,
            sinkhorn_iters=config.sinkhorn_iters,
            log_domain_sinkhorn=config.log_domain_sinkhorn,
            num_input_views=config.num_input_views,
            num_dynamic_alpha_proposals=config.num_dynamic_alpha_proposals,
            use_triton_sinkhorn=config.use_triton_sinkhorn,
        )
        
        self.mhc_mlp = ManifoldConstrainedHyperConnectionsV2(
            num_residual_streams=config.n_streams,
            dim=config.hidden_size,
            layer_index=layer_idx * 2 + 1,
            dropout=config.mhc_dropout,
            num_fracs=config.num_fracs,
            sinkhorn_iters=config.sinkhorn_iters,
            log_domain_sinkhorn=config.log_domain_sinkhorn,
            num_input_views=config.num_input_views,
            num_dynamic_alpha_proposals=config.num_dynamic_alpha_proposals,
            use_triton_sinkhorn=config.use_triton_sinkhorn,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through mHC V2 decoder layer.
        
        Args:
            hidden_states: Multi-stream hidden states (B, S, n, C).
            attention_mask: Attention mask.
            position_ids: Position IDs for RoPE.
            past_key_value: KV cache.
            output_attentions: Whether to output attention weights.
            use_cache: Whether to use/return cache.
            cache_position: Cache position indices.
            position_embeddings: Precomputed RoPE embeddings.
            
        Returns:
            Tuple of (hidden_states, present_key_value, ...).
        """
        # ======== Attention sublayer with mHC V2 ========
        # Get branch input and residual function from mHC
        attn_branch_input, add_attn_residual = self.mhc_attn(hidden_states)
        
        # Apply layernorm and attention
        attn_normed = self.input_layernorm(attn_branch_input)
        attn_output = self.attention(
            attn_normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Handle attention output (may be tuple with cache)
        if isinstance(attn_output, tuple):
            attn_out_hidden = attn_output[0]
        else:
            attn_out_hidden = attn_output
        
        # Add residual via mHC
        hidden_states = add_attn_residual(attn_out_hidden)
        
        # ======== MLP sublayer with mHC V2 ========
        # Get branch input and residual function from mHC
        mlp_branch_input, add_mlp_residual = self.mhc_mlp(hidden_states)
        
        # Apply layernorm and MLP
        mlp_normed = self.post_attention_layernorm(mlp_branch_input)
        mlp_output = self.mlp(mlp_normed)
        
        # Add residual via mHC
        hidden_states = add_mlp_residual(mlp_output)
        
        outputs = (hidden_states,)
        
        return outputs


class Qwen3MHCModelV2(PreTrainedModel):
    """Qwen3 model with mHC V2 multi-stream architecture.
    
    Transforms the base Qwen3 architecture:
    - Adds stream expansion after embedding
    - Wraps each decoder layer with mHC V2
    - Adds stream collapse before final norm
    """
    
    config_class = Qwen3MHCConfigV2
    base_model_prefix = "model"
    _supports_gradient_checkpointing = True
    
    def __init__(self, config: Qwen3MHCConfigV2):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        
        # Embedding (unchanged from original)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        
        # Get expand/reduce functions from mHC V2
        self.expand_fn, self.reduce_fn = get_expand_reduce_stream_functions(
            num_streams=config.n_streams,
            add_stream_embed=config.add_stream_embed,
            add_attn_pool_reduce_stream=config.add_attn_pool_reduce_stream,
            dim=config.hidden_size,
            disable=False,
        )
        
        # Create decoder layers with mHC V2 wrappers
        self.layers = nn.ModuleList([
            self._create_decoder_layer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm - use Qwen3RMSNorm for numerical compatibility with original model
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Rotary embeddings - create our own since we need them for attention
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        
        self.gradient_checkpointing = False
    
    def _create_decoder_layer(self, config: Qwen3MHCConfigV2, layer_idx: int) -> Qwen3MHCDecoderLayerV2:
        """Create a decoder layer with mHC V2 wrappers."""
        # Import Qwen3 components
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP, Qwen3RMSNorm
        
        # Create attention module
        attention = Qwen3Attention(config, layer_idx)
        
        # Create MLP module
        mlp = Qwen3MLP(config)
        
        # Create layer norms
        input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        return Qwen3MHCDecoderLayerV2(
            config=config,
            layer_idx=layer_idx,
            attention_module=attention,
            mlp_module=mlp,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
        )
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Forward pass through the mHC V2 model."""
        from transformers.masking_utils import create_causal_mask
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Expand to multi-stream using mHC V2 expand function
        hidden_states = self.expand_fn(inputs_embeds)  # (B, S, n, C)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # Handle position_ids - create if not provided
        batch_size, seq_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if cache_position is None:
            cache_position = torch.arange(seq_length, dtype=torch.long, device=position_ids.device)
        
        # Create 4D causal attention mask from 2D mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        
        # Get position embeddings from rotary_emb
        position_embeddings = None
        if self.rotary_emb is not None:
            dummy_x = inputs_embeds  # Use embeddings for device/dtype info
            position_embeddings = self.rotary_emb(dummy_x, position_ids)
        
        # Forward through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)
        
        # Collapse back to single stream using mHC V2 reduce function
        hidden_states = self.reduce_fn(hidden_states)  # (B, S, C)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen3MHCForCausalLMV2(PreTrainedModel, GenerationMixin):
    """Qwen3 with mHC V2 for causal language modeling."""
    
    config_class = Qwen3MHCConfigV2
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_param_buffer_assignment = False  # Prevent post_init from reinitializing MHC params
    _supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize weights, skipping mHC V2 modules which handle their own init."""
        if isinstance(module, ManifoldConstrainedHyperConnectionsV2):
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module=None, value=False):
        """Enable or disable gradient checkpointing (compatible with transformers API)."""
        if module is None:
            self.model.gradient_checkpointing = value
        else:
            self.model.gradient_checkpointing = value
    
    def __init__(self, config: Qwen3MHCConfigV2):
        super().__init__(config)
        self.config = config
        
        self.model = Qwen3MHCModelV2(config)
        self.vocab_size = config.vocab_size
        
        # LM head (may be tied to embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_decoder(self):
        return self.model
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass for causal LM."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # For numerical stability
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation."""
        # If we have cache, only use the last token
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        
        # If inputs_embeds provided, only use them for first generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        
        model_inputs.update(
            {
                "position_ids": cache_position.unsqueeze(0) if cache_position is not None else None,
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        
        return model_inputs
    
    def get_mhc_parameters(self) -> List[nn.Parameter]:
        """Get all mHC V2-specific parameters for separate optimization."""
        mhc_params = []
        for layer in self.model.layers:
            # Attention mHC V2 parameters
            mhc_params.append(layer.mhc_attn.static_alpha)
            mhc_params.append(layer.mhc_attn.dynamic_alpha_fn)
            mhc_params.append(layer.mhc_attn.pre_branch_scale)
            mhc_params.append(layer.mhc_attn.residual_scale)
            if layer.mhc_attn.add_branch_out_to_residual:
                mhc_params.append(layer.mhc_attn.static_beta)
                mhc_params.append(layer.mhc_attn.dynamic_beta_fn)
                mhc_params.append(layer.mhc_attn.h_post_scale)
            mhc_params.extend(layer.mhc_attn.norm.parameters())
            
            # MLP mHC V2 parameters
            mhc_params.append(layer.mhc_mlp.static_alpha)
            mhc_params.append(layer.mhc_mlp.dynamic_alpha_fn)
            mhc_params.append(layer.mhc_mlp.pre_branch_scale)
            mhc_params.append(layer.mhc_mlp.residual_scale)
            if layer.mhc_mlp.add_branch_out_to_residual:
                mhc_params.append(layer.mhc_mlp.static_beta)
                mhc_params.append(layer.mhc_mlp.dynamic_beta_fn)
                mhc_params.append(layer.mhc_mlp.h_post_scale)
            mhc_params.extend(layer.mhc_mlp.norm.parameters())
        
        return mhc_params
    
    def get_original_parameters(self) -> List[nn.Parameter]:
        """Get all original (non-mHC) parameters."""
        mhc_param_ids = {id(p) for p in self.get_mhc_parameters()}
        return [p for p in self.parameters() if id(p) not in mhc_param_ids]
    
    def get_monitoring_stats(self, hidden_states: torch.Tensor) -> dict:
        """Get monitoring statistics from all mHC V2 layers."""
        stats = {
            "n_streams": self.config.n_streams,
            "num_fracs": self.config.num_fracs,
            "sinkhorn_iters": self.config.sinkhorn_iters,
        }
        return stats
