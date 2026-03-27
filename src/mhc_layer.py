"""
mHC Layer wrapper module.

Wraps a transformer sublayer (Attention or MLP) with Manifold-Constrained 
Hyper-Connection machinery, enabling multi-stream residual computation with
learned inter-stream mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math

from .config import MHCConfig
from .sinkhorn import sinkhorn_knopp


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Used for normalizing the flattened hidden states before computing
    dynamic mHC coefficients.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class MHCLayer(nn.Module):
    """Manifold-Constrained Hyper-Connection layer.
    
    Wraps a transformer sublayer with mHC machinery:
    - Computes H_pre, H_post, H_res coefficients from hidden state
    - Squeezes n streams to 1 via H_pre for sublayer input
    - Expands 1 stream to n via H_post for sublayer output
    - Mixes streams via H_res (doubly stochastic) for residual
    
    Forward pass:
        1. x ∈ (B, S, n, C) - expanded hidden state
        2. h_in = H_pre @ x, squeeze → (B, S, C)
        3. h_out = sublayer(layernorm(h_in)) → (B, S, C)
        4. h_post = H_post.T @ h_out → (B, S, n, C)
        5. h_res = H_res @ x → (B, S, n, C)
        6. output = h_res + h_post → (B, S, n, C)
    
    Attributes:
        config: MHC configuration.
        sublayer: The wrapped transformer sublayer (Attention or MLP).
        layernorm: LayerNorm applied before sublayer.
    """
    
    def __init__(
        self,
        config: MHCConfig,
        sublayer: nn.Module,
        layernorm: nn.Module,
    ):
        """Initialize MHC layer.
        
        Args:
            config: MHC configuration.
            sublayer: Transformer sublayer to wrap (Attention or MLP).
            layernorm: LayerNorm to apply before sublayer.
        """
        super().__init__()
        self.config = config
        self.sublayer = sublayer
        self.layernorm = layernorm
        
        n = config.n_streams
        expanded_size = config.expanded_size
        
        # Coefficient computation normalization
        self.coef_norm = RMSNorm(expanded_size, eps=config.coef_norm_eps)
        
        # Dynamic projection matrices (initialized to zero for equivalence)
        self.phi_pre = nn.Linear(expanded_size, n, bias=False)
        self.phi_post = nn.Linear(expanded_size, n, bias=False)
        self.phi_res = nn.Linear(expanded_size, n * n, bias=False)
        
        # Static bias terms
        self.b_pre = nn.Parameter(torch.empty(1, 1, n))
        self.b_post = nn.Parameter(torch.empty(1, 1, n))
        self.b_res = nn.Parameter(torch.empty(1, 1, n, n))
        
        # Learnable gating scalars
        self.alpha_pre = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(config.alpha_init))
        
        # Initialize for equivalence
        self._init_for_equivalence()
    
    def _init_for_equivalence(self):
        """Initialize parameters for equivalence with original model.
        
        At initialization:
        - H_pre averages n streams (softmax of equal values = 1/n each)
        - H_post copies to all streams equally (2*sigmoid(0) = 1)
        - H_res is identity (no mixing)
        """
        config = self.config
        n = config.n_streams
        
        # phi weights: zero for no dynamic contribution
        if config.phi_init_std == 0:
            nn.init.zeros_(self.phi_pre.weight)
            nn.init.zeros_(self.phi_post.weight)
            nn.init.zeros_(self.phi_res.weight)
        else:
            nn.init.normal_(self.phi_pre.weight, std=config.phi_init_std)
            nn.init.normal_(self.phi_post.weight, std=config.phi_init_std)
            nn.init.normal_(self.phi_res.weight, std=config.phi_init_std)
        
        # b_pre: softmax of equal values gives 1/n for each stream
        # Any equal values work, 0 is simplest
        self.b_pre.data.zero_()
        
        # b_post: 2*sigmoid(b_post) = 1, so sigmoid = 0.5, so b_post = 0
        self.b_post.data.fill_(config.b_post_init)
        
        # b_res: large diagonal for identity after Sinkhorn
        self.b_res.data.zero_()
        for i in range(n):
            self.b_res.data[0, 0, i, i] = config.b_res_diagonal_init
    
    def compute_coefficients(
        self,
        x_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute H_pre, H_post, H_res from flattened hidden state.
        
        Args:
            x_flat: Flattened hidden state of shape (B, S, n*C).
            
        Returns:
            Tuple of (H_pre, H_post, H_res):
            - H_pre: (B, S, 1, n) pre-mapping coefficients
            - H_post: (B, S, n, 1) post-mapping coefficients
            - H_res: (B, S, n, n) residual mixing matrix (doubly stochastic)
        """
        n = self.config.n_streams
        
        # Normalize for coefficient computation
        x_norm = self.coef_norm(x_flat)
        
        # Compute dynamic components
        h_pre_dyn = self.phi_pre(x_norm)    # (B, S, n)
        h_post_dyn = self.phi_post(x_norm)  # (B, S, n)
        h_res_dyn = self.phi_res(x_norm)    # (B, S, n*n)
        
        # Reshape residual dynamic component
        B, S, _ = h_res_dyn.shape
        h_res_dyn = h_res_dyn.view(B, S, n, n)
        
        # Combine dynamic and static with gating
        h_pre_raw = self.alpha_pre * h_pre_dyn + self.b_pre
        h_post_raw = self.alpha_post * h_post_dyn + self.b_post
        h_res_raw = self.alpha_res * h_res_dyn + self.b_res
        
        # Apply constraints
        # H_pre: softmax to ensure exact sum to 1 (better numerical stability than sigmoid)
        H_pre = torch.softmax(h_pre_raw, dim=-1).unsqueeze(-2)  # (B, S, 1, n)
        
        # H_post: 2*sigmoid for [0, 2] range, then transpose for matmul
        H_post = (2.0 * torch.sigmoid(h_post_raw)).unsqueeze(-1)  # (B, S, n, 1)
        
        # H_res: Sinkhorn-Knopp for doubly stochastic
        H_res = sinkhorn_knopp(
            h_res_raw,
            num_iterations=self.config.sinkhorn_iterations,
            eps=self.config.sinkhorn_eps,
        )  # (B, S, n, n)
        
        return H_pre, H_post, H_res
    
    def forward(
        self,
        x: torch.Tensor,
        **sublayer_kwargs,
    ) -> torch.Tensor:
        """Forward pass through mHC layer.
        
        Args:
            x: Expanded hidden state of shape (B, S, n, C).
            **sublayer_kwargs: Additional arguments passed to sublayer
                (e.g., attention_mask, position_ids, cache, etc.).
                
        Returns:
            Output hidden state of shape (B, S, n, C).
        """
        B, S, n, C = x.shape
        
        # Flatten for coefficient computation
        x_flat = x.view(B, S, n * C)
        
        # Compute coefficients
        H_pre, H_post, H_res = self.compute_coefficients(x_flat)
        
        # Pre-mapping: squeeze n streams to 1
        # H_pre: (B, S, 1, n), x: (B, S, n, C)
        # Result: (B, S, 1, C) -> squeeze -> (B, S, C)
        h_in = torch.matmul(H_pre, x).squeeze(-2)  # (B, S, C)
        
        # Apply layernorm and sublayer
        h_norm = self.layernorm(h_in)
        
        # Handle sublayer forward - different sublayers have different signatures
        h_out = self._forward_sublayer(h_norm, **sublayer_kwargs)
        
        # If sublayer returns tuple (like attention with cache), extract hidden state
        if isinstance(h_out, tuple):
            h_out = h_out[0]
        
        # Post-mapping: broadcast 1 stream to n
        # h_out: (B, S, C) -> unsqueeze -> (B, S, 1, C)
        # H_post: (B, S, n, 1)
        # Result: (B, S, n, C)
        h_out = h_out.unsqueeze(-2)  # (B, S, 1, C)
        h_post = torch.matmul(H_post, h_out)  # (B, S, n, C)
        
        # Residual mapping: mix streams
        # H_res: (B, S, n, n), x: (B, S, n, C)
        # Result: (B, S, n, C)
        h_res = torch.matmul(H_res, x)  # (B, S, n, C)
        
        # Combine
        output = h_res + h_post
        
        return output
    
    def _forward_sublayer(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the wrapped sublayer.
        
        Handles different sublayer types (Attention vs MLP) which may have
        different forward signatures.
        
        Args:
            hidden_states: Normalized hidden states (B, S, C).
            **kwargs: Sublayer-specific arguments.
            
        Returns:
            Sublayer output.
        """
        return self.sublayer(hidden_states, **kwargs)
    
    def get_monitoring_stats(self, x: torch.Tensor) -> dict:
        """Get monitoring statistics for training stability.
        
        Args:
            x: Input hidden state (B, S, n, C).
            
        Returns:
            Dictionary of monitoring metrics.
        """
        B, S, n, C = x.shape
        x_flat = x.view(B, S, n * C)
        
        H_pre, H_post, H_res = self.compute_coefficients(x_flat)
        
        # Compute stability metrics
        from .sinkhorn import (
            compute_forward_gain,
            compute_backward_gain,
            compute_identity_distance,
        )
        
        return {
            "forward_gain": compute_forward_gain(H_res).item(),
            "backward_gain": compute_backward_gain(H_res).item(),
            "identity_distance": compute_identity_distance(H_res).mean().item(),
            "alpha_pre": self.alpha_pre.item(),
            "alpha_post": self.alpha_post.item(),
            "alpha_res": self.alpha_res.item(),
        }
