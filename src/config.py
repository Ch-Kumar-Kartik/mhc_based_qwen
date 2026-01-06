"""
Configuration dataclass for mHC parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class MHCConfig:
    """Configuration for Manifold-Constrained Hyper-Connections.
    
    Attributes:
        n_streams: Number of parallel residual streams (expansion rate).
        hidden_size: Original model hidden dimension.
        alpha_init: Initial value for learnable gating scalars.
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations for doubly stochastic projection.
        sinkhorn_eps: Small epsilon for numerical stability in Sinkhorn-Knopp.
        b_pre_init: Initial value for pre-mapping bias (logit(1/n) for averaging).
        b_post_init: Initial value for post-mapping bias (0 for sigmoid(0)=0.5, 2*0.5=1).
        b_res_diagonal_init: Initial diagonal value for residual mapping (large for identity).
        phi_init_std: Standard deviation for phi weight initialization (0 for equivalence).
    """
    
    # Core parameters
    n_streams: int = 4
    hidden_size: int = 1024
    
    # Gating initialization
    alpha_init: float = 0.01
    
    # Sinkhorn-Knopp parameters
    sinkhorn_iterations: int = 20
    sinkhorn_eps: float = 1e-8
    
    # Bias initialization for equivalence
    # b_pre: sigmoid(b_pre) should equal 1/n for averaging
    # For n=4: logit(0.25) = log(0.25/0.75) = log(1/3) ≈ -1.0986
    b_pre_init: Optional[float] = None  # Computed from n_streams if None
    
    # b_post: 2*sigmoid(b_post) should equal 1 for copying
    # sigmoid(0) = 0.5, 2*0.5 = 1
    b_post_init: float = 0.0
    
    # b_res: After Sinkhorn on exp(b_res), result should be identity
    # Large diagonal values dominate after normalization
    b_res_diagonal_init: float = 20.0
    
    # Dynamic projection initialization (0 for equivalence)
    phi_init_std: float = 0.0
    
    # Coefficient norm epsilon
    coef_norm_eps: float = 1e-6
    
    def __post_init__(self):
        """Compute derived values."""
        if self.b_pre_init is None:
            # logit(1/n) = log((1/n) / (1 - 1/n)) = log(1/(n-1))
            self.b_pre_init = math.log(1.0 / (self.n_streams - 1))
    
    @property
    def expanded_size(self) -> int:
        """Total expanded hidden dimension (n * C)."""
        return self.n_streams * self.hidden_size
    
    @property
    def phi_pre_shape(self) -> tuple:
        """Shape of phi_pre projection: (n*C, n)."""
        return (self.expanded_size, self.n_streams)
    
    @property
    def phi_post_shape(self) -> tuple:
        """Shape of phi_post projection: (n*C, n)."""
        return (self.expanded_size, self.n_streams)
    
    @property
    def phi_res_shape(self) -> tuple:
        """Shape of phi_res projection: (n*C, n^2)."""
        return (self.expanded_size, self.n_streams * self.n_streams)
    
    @classmethod
    def for_qwen3_0_6b(cls, n_streams: int = 4) -> "MHCConfig":
        """Create config for Qwen3-0.6B model.
        
        Args:
            n_streams: Number of parallel streams (default 4 as per paper).
            
        Returns:
            MHCConfig configured for Qwen3-0.6B.
        """
        return cls(
            n_streams=n_streams,
            hidden_size=1024,
        )


@dataclass
class MHCTrainingConfig:
    """Training configuration for mHC models.
    
    Based on paper's recommendations for 3B model, scaled for 0.6B.
    """
    
    # Optimizer
    learning_rate: float = 8.6e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-20
    
    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 30000
    lr_decay_style: str = "step"  # step decay at 80% and 90%
    lr_decay_ratio_1: float = 0.316  # at 80%
    lr_decay_ratio_2: float = 0.1    # at 90%
    
    # Batch
    batch_size: int = 320
    sequence_length: int = 4096
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    checkpoint_interval: int = 1000
    gradient_checkpointing: bool = True
    checkpoint_every_n_layers: int = 4  # √(n*L/(n+2)) for n=4, L=28
    
    # Stability monitoring thresholds
    gradient_norm_threshold: float = 10.0
    forward_gain_threshold: float = 2.0
    backward_gain_threshold: float = 2.0
    loss_spike_threshold: float = 2.0  # 2x increase is warning
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
