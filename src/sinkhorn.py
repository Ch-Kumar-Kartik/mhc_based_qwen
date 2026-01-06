"""
Sinkhorn-Knopp algorithm for doubly stochastic matrix projection.

Projects arbitrary matrices onto the Birkhoff polytope (set of doubly stochastic 
matrices where all rows and columns sum to 1).

Reference: Sinkhorn & Knopp, "Concerning nonnegative matrices and doubly stochastic 
matrices", Pacific Journal of Mathematics, 1967
"""

import torch
import torch.nn as nn
from typing import Optional


def sinkhorn_knopp(
    M: torch.Tensor,
    num_iterations: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply Sinkhorn-Knopp algorithm to project matrix onto doubly stochastic manifold.
    
    Projects an arbitrary matrix M onto the Birkhoff polytope by:
    1. Exponentiating to make all entries positive
    2. Alternating row and column normalization until convergence
    
    The resulting matrix H satisfies:
    - All entries >= 0
    - Each row sums to 1
    - Each column sums to 1
    
    Properties of doubly stochastic matrices:
    - Spectral norm <= 1 (non-expansive)
    - Product of DS matrices is DS
    - Bounded signal propagation across layers
    
    Args:
        M: Input matrix of shape (..., n, n). Can be batched.
        num_iterations: Number of alternating normalization iterations.
        eps: Small constant for numerical stability.
        
    Returns:
        Doubly stochastic matrix of same shape as M.
        
    Example:
        >>> M = torch.randn(4, 4)
        >>> H = sinkhorn_knopp(M, num_iterations=20)
        >>> print(H.sum(dim=-1))  # Each row sums to ~1
        >>> print(H.sum(dim=-2))  # Each column sums to ~1
    """
    # Exponentiate to ensure positivity
    # Use float32 for numerical stability even if input is bfloat16
    input_dtype = M.dtype
    M = M.float()
    H = torch.exp(M)
    
    # Alternating row and column normalization
    for _ in range(num_iterations):
        # Normalize rows
        row_sums = H.sum(dim=-1, keepdim=True)
        H = H / (row_sums + eps)
        
        # Normalize columns
        col_sums = H.sum(dim=-2, keepdim=True)
        H = H / (col_sums + eps)
    
    return H.to(input_dtype)


class SinkhornKnopp(nn.Module):
    """Differentiable Sinkhorn-Knopp layer.
    
    A module wrapper around the Sinkhorn-Knopp algorithm that can be used
    as a differentiable layer in neural networks. Gradients flow through
    the iterative normalization process.
    
    The gradient is computed by differentiating through the forward pass.
    For more efficient backward pass, one could implement implicit differentiation
    through the fixed-point equation, but the explicit approach works well for
    small n (like n=4).
    
    Attributes:
        num_iterations: Number of Sinkhorn iterations.
        eps: Numerical stability constant.
    """
    
    def __init__(
        self,
        num_iterations: int = 20,
        eps: float = 1e-8,
    ):
        """Initialize Sinkhorn-Knopp layer.
        
        Args:
            num_iterations: Number of alternating normalization iterations.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps
    
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """Apply Sinkhorn-Knopp projection.
        
        Args:
            M: Input matrix of shape (..., n, n).
            
        Returns:
            Doubly stochastic matrix of same shape.
        """
        return sinkhorn_knopp(M, self.num_iterations, self.eps)
    
    def extra_repr(self) -> str:
        return f"num_iterations={self.num_iterations}, eps={self.eps}"


def validate_doubly_stochastic(
    H: torch.Tensor,
    tol: float = 1e-5,
) -> dict:
    """Validate that a matrix is doubly stochastic.
    
    Args:
        H: Matrix to validate, shape (..., n, n).
        tol: Tolerance for sum checks.
        
    Returns:
        Dictionary with validation results:
        - is_valid: Whether matrix satisfies doubly stochastic constraints
        - row_sum_error: Maximum deviation of row sums from 1
        - col_sum_error: Maximum deviation of column sums from 1
        - min_value: Minimum entry (should be >= 0)
    """
    row_sums = H.sum(dim=-1)
    col_sums = H.sum(dim=-2)
    
    row_sum_error = (row_sums - 1.0).abs().max().item()
    col_sum_error = (col_sums - 1.0).abs().max().item()
    min_value = H.min().item()
    
    is_valid = (
        row_sum_error < tol and
        col_sum_error < tol and
        min_value >= -tol  # Allow small numerical errors
    )
    
    return {
        "is_valid": is_valid,
        "row_sum_error": row_sum_error,
        "col_sum_error": col_sum_error,
        "min_value": min_value,
    }


def compute_forward_gain(H: torch.Tensor) -> torch.Tensor:
    """Compute forward gain of doubly stochastic matrix.
    
    The forward gain is the maximum row sum, which for a DS matrix should be 1.
    This is useful for monitoring training stability.
    
    Args:
        H: Doubly stochastic matrix of shape (..., n, n).
        
    Returns:
        Maximum row sum (should be ~1.0 for DS matrices).
    """
    return H.sum(dim=-1).max()


def compute_backward_gain(H: torch.Tensor) -> torch.Tensor:
    """Compute backward gain of doubly stochastic matrix.
    
    The backward gain is the maximum column sum, which for a DS matrix should be 1.
    This is useful for monitoring training stability.
    
    Args:
        H: Doubly stochastic matrix of shape (..., n, n).
        
    Returns:
        Maximum column sum (should be ~1.0 for DS matrices).
    """
    return H.sum(dim=-2).max()


def compute_identity_distance(H: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius distance from identity matrix.
    
    Useful for monitoring how much inter-stream mixing is being learned
    (initialized as identity, distance increases as mixing is learned).
    
    Args:
        H: Matrix of shape (..., n, n).
        
    Returns:
        Frobenius norm of (H - I).
    """
    n = H.shape[-1]
    device = H.device
    dtype = H.dtype
    I = torch.eye(n, device=device, dtype=dtype)
    
    # Expand I to match batch dimensions
    while I.dim() < H.dim():
        I = I.unsqueeze(0)
    
    return torch.norm(H - I, p='fro', dim=(-2, -1))
