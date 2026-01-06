"""
Tests for Sinkhorn-Knopp algorithm implementation.
"""

import pytest
import torch
import math

from src.sinkhorn import (
    sinkhorn_knopp,
    SinkhornKnopp,
    validate_doubly_stochastic,
    compute_forward_gain,
    compute_backward_gain,
    compute_identity_distance,
)


class TestSinkhornKnopp:
    """Tests for Sinkhorn-Knopp algorithm."""
    
    def test_output_is_doubly_stochastic(self):
        """Test that output satisfies doubly stochastic constraints."""
        M = torch.randn(4, 4)
        H = sinkhorn_knopp(M, num_iterations=20)
        
        result = validate_doubly_stochastic(H)
        
        assert result["is_valid"], (
            f"Not doubly stochastic: row_err={result['row_sum_error']:.6e}, "
            f"col_err={result['col_sum_error']:.6e}, min={result['min_value']:.6e}"
        )
    
    def test_output_is_non_negative(self):
        """Test that all entries are non-negative."""
        M = torch.randn(4, 4) * 10  # Large values
        H = sinkhorn_knopp(M)
        
        assert H.min() >= -1e-8, f"Negative entry found: {H.min()}"
    
    def test_batched_input(self):
        """Test with batched input."""
        B, S, n = 2, 10, 4
        M = torch.randn(B, S, n, n)
        H = sinkhorn_knopp(M)
        
        assert H.shape == (B, S, n, n)
        
        # Check each batch element
        for b in range(B):
            for s in range(S):
                result = validate_doubly_stochastic(H[b, s])
                assert result["is_valid"]
    
    def test_convergence_in_20_iterations(self):
        """Test that 20 iterations achieves good convergence for n=4."""
        M = torch.randn(4, 4)
        H = sinkhorn_knopp(M, num_iterations=20)
        
        result = validate_doubly_stochastic(H, tol=1e-8)
        
        assert result["row_sum_error"] < 1e-8
        assert result["col_sum_error"] < 1e-8
    
    def test_large_diagonal_gives_near_identity(self):
        """Test that large diagonal values produce near-identity matrix."""
        n = 4
        M = torch.zeros(n, n)
        M.fill_diagonal_(20.0)  # Large diagonal
        
        H = sinkhorn_knopp(M)
        I = torch.eye(n)
        
        diff = (H - I).abs().max()
        assert diff < 1e-6, f"Expected near-identity, got diff={diff:.6e}"
    
    def test_gradient_flow(self):
        """Test that gradients flow through Sinkhorn-Knopp."""
        M = torch.randn(4, 4, requires_grad=True)
        H = sinkhorn_knopp(M)
        
        # Create a loss and backprop
        loss = H.sum()
        loss.backward()
        
        assert M.grad is not None
        assert not torch.isnan(M.grad).any()
        assert not torch.isinf(M.grad).any()
    
    def test_numerical_stability_with_extreme_values(self):
        """Test numerical stability with extreme input values."""
        # Very large values
        M_large = torch.randn(4, 4) * 100
        H_large = sinkhorn_knopp(M_large)
        assert not torch.isnan(H_large).any()
        
        # Very small values
        M_small = torch.randn(4, 4) * 0.001
        H_small = sinkhorn_knopp(M_small)
        assert not torch.isnan(H_small).any()
        
        # Mixed signs
        M_mixed = torch.randn(4, 4) * 50 - 25
        H_mixed = sinkhorn_knopp(M_mixed)
        assert not torch.isnan(H_mixed).any()


class TestSinkhornKnoppModule:
    """Tests for SinkhornKnopp nn.Module."""
    
    def test_module_initialization(self):
        """Test module can be initialized."""
        sk = SinkhornKnopp(num_iterations=20, eps=1e-8)
        assert sk.num_iterations == 20
        assert sk.eps == 1e-8
    
    def test_module_forward(self):
        """Test module forward pass."""
        sk = SinkhornKnopp()
        M = torch.randn(4, 4)
        H = sk(M)
        
        result = validate_doubly_stochastic(H)
        assert result["is_valid"]
    
    def test_module_in_network(self):
        """Test module can be used in a neural network."""
        class TestNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)
                self.sk = SinkhornKnopp()
            
            def forward(self, x):
                x = self.linear(x)
                x = x.view(-1, 4, 4)
                return self.sk(x)
        
        net = TestNet()
        x = torch.randn(2, 16)
        y = net(x)
        
        assert y.shape == (2, 4, 4)
        
        # Test backward
        loss = y.sum()
        loss.backward()
        
        assert net.linear.weight.grad is not None


class TestMonitoringFunctions:
    """Tests for monitoring utility functions."""
    
    def test_forward_gain_for_ds_matrix(self):
        """Forward gain of DS matrix should be ~1."""
        M = torch.randn(4, 4)
        H = sinkhorn_knopp(M)
        
        gain = compute_forward_gain(H)
        assert abs(gain.item() - 1.0) < 1e-6
    
    def test_backward_gain_for_ds_matrix(self):
        """Backward gain of DS matrix should be ~1."""
        M = torch.randn(4, 4)
        H = sinkhorn_knopp(M)
        
        gain = compute_backward_gain(H)
        assert abs(gain.item() - 1.0) < 1e-6
    
    def test_identity_distance_for_identity(self):
        """Identity distance for identity matrix should be 0."""
        I = torch.eye(4)
        dist = compute_identity_distance(I)
        
        assert dist.item() < 1e-6
    
    def test_identity_distance_for_large_diagonal(self):
        """Large diagonal initialization should give near-zero distance."""
        M = torch.zeros(4, 4)
        M.fill_diagonal_(20.0)
        H = sinkhorn_knopp(M)
        
        dist = compute_identity_distance(H)
        assert dist.item() < 1e-5


class TestValidation:
    """Tests for validation utility."""
    
    def test_validate_identity_matrix(self):
        """Identity matrix is doubly stochastic."""
        I = torch.eye(4)
        result = validate_doubly_stochastic(I)
        
        assert result["is_valid"]
        assert result["row_sum_error"] < 1e-10
        assert result["col_sum_error"] < 1e-10
        assert result["min_value"] >= 0
    
    def test_validate_uniform_matrix(self):
        """Uniform matrix (1/n entries) is doubly stochastic."""
        n = 4
        U = torch.ones(n, n) / n
        result = validate_doubly_stochastic(U)
        
        assert result["is_valid"]
    
    def test_validate_non_ds_matrix(self):
        """Non-DS matrix should fail validation."""
        # Matrix with row sums != 1
        M = torch.randn(4, 4).abs()  # Non-negative but not normalized
        result = validate_doubly_stochastic(M)
        
        assert not result["is_valid"]


class TestDtypeHandling:
    """Tests for different data types."""
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
    def test_preserves_dtype(self, dtype):
        """Test that output dtype matches input dtype."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
        
        M = torch.randn(4, 4, dtype=dtype)
        H = sinkhorn_knopp(M)
        
        assert H.dtype == dtype
    
    def test_bfloat16_still_accurate(self):
        """Test that bfloat16 computation is still accurate."""
        if not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
        
        M = torch.randn(4, 4, dtype=torch.bfloat16, device='cuda')
        H = sinkhorn_knopp(M)
        
        # Convert to float for validation
        H_float = H.float()
        result = validate_doubly_stochastic(H_float, tol=1e-3)  # Looser tolerance for bf16
        
        assert result["is_valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
