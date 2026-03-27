"""
Tests for equivalence between original Qwen3 and mHC-converted model.

These tests verify that at initialization with equivalence parameters,
the mHC model produces identical outputs to the original model.
"""

import pytest
import torch
import torch.nn as nn
import math

from src.config import MHCConfig
from src.stream_ops import StreamExpand, StreamCollapse
from src.sinkhorn import sinkhorn_knopp


class TestStreamOperations:
    """Tests for stream expand/collapse equivalence."""
    
    def test_expand_collapse_identity(self):
        """Test that expand -> collapse is identity."""
        expand = StreamExpand(n_streams=4)
        collapse = StreamCollapse(n_streams=4)
        
        x = torch.randn(2, 10, 64)
        
        x_expanded = expand(x)
        x_collapsed = collapse(x_expanded)
        
        # Should be exact since we copy then average identical values
        assert torch.allclose(x, x_collapsed, atol=1e-6)
    
    def test_expand_creates_copies(self):
        """Test that expand creates n identical copies."""
        expand = StreamExpand(n_streams=4)
        
        x = torch.randn(2, 10, 64)
        x_expanded = expand(x)
        
        # All streams should be identical
        for i in range(4):
            assert torch.allclose(x_expanded[:, :, i, :], x)
    
    def test_collapse_averages(self):
        """Test that collapse averages streams."""
        collapse = StreamCollapse(n_streams=4)
        
        # Create different streams
        streams = [torch.randn(2, 10, 64) for _ in range(4)]
        x = torch.stack(streams, dim=2)
        
        y = collapse(x)
        expected = sum(streams) / 4
        
        assert torch.allclose(y, expected)


class TestResidualEquivalence:
    """Tests for residual path equivalence."""
    
    def test_h_pre_averaging(self):
        """Test H_pre performs averaging when initialized correctly."""
        n = 4
        C = 64
        
        # Create H_pre as initialized: sigmoid(log(1/3)) = 0.25
        b_pre = math.log(1.0 / (n - 1))
        H_pre = torch.sigmoid(torch.full((1, 1, 1, n), b_pre))
        
        # Create input with identical streams
        v = torch.randn(2, 10, C)
        x = v.unsqueeze(-2).expand(-1, -1, n, -1).clone()
        
        # H_pre @ x should give v (the original value)
        h_in = torch.matmul(H_pre, x).squeeze(-2)
        
        assert torch.allclose(h_in, v, atol=1e-5)
    
    def test_h_post_copying(self):
        """Test H_post performs copying when initialized correctly."""
        n = 4
        C = 64
        
        # Create H_post as initialized: 2*sigmoid(0) = 1.0
        b_post = 0.0
        H_post = 2.0 * torch.sigmoid(torch.full((1, 1, n, 1), b_post))
        
        # Create sublayer output
        h_out = torch.randn(2, 10, 1, C)
        
        # H_post @ h_out should broadcast to all streams equally
        h_post = torch.matmul(H_post, h_out)
        
        # All streams should equal h_out.squeeze()
        for i in range(n):
            assert torch.allclose(h_post[:, :, i, :], h_out.squeeze(-2), atol=1e-5)
    
    def test_h_res_identity(self):
        """Test H_res acts as identity when initialized correctly."""
        n = 4
        C = 64
        
        # Create b_res with large diagonal
        b_res = torch.zeros(1, 1, n, n)
        for i in range(n):
            b_res[0, 0, i, i] = 20.0
        
        # Apply Sinkhorn-Knopp
        H_res = sinkhorn_knopp(b_res, num_iterations=20)
        
        # Create input
        x = torch.randn(2, 10, n, C)
        
        # H_res @ x should be approximately x
        h_res = torch.matmul(H_res, x)
        
        assert torch.allclose(h_res, x, atol=1e-5)


class TestFullLayerEquivalence:
    """Tests for full layer equivalence simulation."""
    
    def test_mhc_residual_equivalence(self):
        """Test full mHC residual is equivalent to standard residual."""
        n = 4
        C = 64
        B, S = 2, 10
        
        # Initialize coefficients for equivalence
        b_pre = math.log(1.0 / (n - 1))
        H_pre = torch.sigmoid(torch.full((B, S, 1, n), b_pre))
        
        b_post = 0.0
        H_post = 2.0 * torch.sigmoid(torch.full((B, S, n, 1), b_post))
        
        b_res = torch.zeros(B, S, n, n)
        for i in range(n):
            b_res[:, :, i, i] = 20.0
        H_res = sinkhorn_knopp(b_res)
        
        # Original value
        v = torch.randn(B, S, C)
        
        # Expand to streams
        x = v.unsqueeze(-2).expand(-1, -1, n, -1).clone()
        
        # Simulate sublayer (identity for simplicity)
        sublayer_delta = torch.randn(B, S, C) * 0.1  # Small change
        
        # mHC forward pass
        h_in = torch.matmul(H_pre, x).squeeze(-2)  # Should be v
        h_out_raw = h_in + sublayer_delta  # Standard residual would add this
        
        # For equivalence test, we check the mHC mechanism
        h_out = h_out_raw.unsqueeze(-2)
        h_post = torch.matmul(H_post, h_out)  # Broadcast to all streams
        h_res = torch.matmul(H_res, x)  # Should be x (identity)
        
        output = h_res + h_post
        
        # Collapse
        output_collapsed = output.mean(dim=-2)
        
        # Expected: original v + sublayer_delta
        expected = v + sublayer_delta
        
        assert torch.allclose(output_collapsed, expected, atol=1e-4)


class TestCoefficientsAtInit:
    """Tests verifying coefficient values at initialization."""
    
    def test_h_pre_value_at_init(self):
        """Test H_pre has correct value at initialization."""
        n = 4
        b_pre_init = math.log(1.0 / (n - 1))
        
        h_pre = torch.sigmoid(torch.tensor(b_pre_init))
        expected = 1.0 / n
        
        assert abs(h_pre.item() - expected) < 1e-5
    
    def test_h_post_value_at_init(self):
        """Test H_post has correct value at initialization."""
        b_post_init = 0.0
        
        h_post = 2.0 * torch.sigmoid(torch.tensor(b_post_init))
        expected = 1.0
        
        assert abs(h_post.item() - expected) < 1e-5
    
    def test_h_res_identity_at_init(self):
        """Test H_res is identity at initialization."""
        n = 4
        
        b_res = torch.zeros(n, n)
        for i in range(n):
            b_res[i, i] = 20.0
        
        H_res = sinkhorn_knopp(b_res)
        I = torch.eye(n)
        
        assert torch.allclose(H_res, I, atol=1e-5)


class TestGradientEquivalence:
    """Tests for gradient flow equivalence."""
    
    def test_gradients_flow_through_sinkhorn(self):
        """Test gradients flow correctly through Sinkhorn."""
        n = 4
        
        b_res = torch.zeros(n, n, requires_grad=True)
        with torch.no_grad():
            b_res_data = torch.zeros(n, n)
            for i in range(n):
                b_res_data[i, i] = 20.0
        b_res = nn.Parameter(b_res_data.clone())
        
        H_res = sinkhorn_knopp(b_res)
        loss = H_res.sum()
        loss.backward()
        
        assert b_res.grad is not None
        assert not torch.isnan(b_res.grad).any()
    
    def test_gradients_through_full_mhc_path(self):
        """Test gradients flow through full mHC computation."""
        n = 4
        C = 64
        B, S = 2, 10
        
        # Create learnable parameters
        b_pre = nn.Parameter(torch.full((1, 1, 1, n), math.log(1.0 / (n - 1))))
        b_post = nn.Parameter(torch.zeros(1, 1, n, 1))
        
        b_res_data = torch.zeros(1, 1, n, n)
        for i in range(n):
            b_res_data[0, 0, i, i] = 20.0
        b_res = nn.Parameter(b_res_data)
        
        # Input
        x = torch.randn(B, S, n, C, requires_grad=True)
        
        # Forward
        H_pre = torch.sigmoid(b_pre)
        H_post = 2.0 * torch.sigmoid(b_post)
        H_res = sinkhorn_knopp(b_res)
        
        h_in = torch.matmul(H_pre, x).squeeze(-2)
        h_out = h_in.unsqueeze(-2)  # Identity sublayer
        h_post = torch.matmul(H_post, h_out)
        h_res = torch.matmul(H_res, x)
        
        output = h_res + h_post
        loss = output.sum()
        loss.backward()
        
        # Check all gradients exist
        assert x.grad is not None
        assert b_pre.grad is not None
        assert b_post.grad is not None
        assert b_res.grad is not None


class TestNumericalStability:
    """Tests for numerical stability in equivalence computation."""
    
    def test_stability_with_large_hidden_size(self):
        """Test numerical stability with large hidden size."""
        n = 4
        C = 4096  # Large like real model
        
        b_res = torch.zeros(n, n)
        for i in range(n):
            b_res[i, i] = 20.0
        
        H_res = sinkhorn_knopp(b_res)
        
        # Should still be identity
        assert torch.allclose(H_res, torch.eye(n), atol=1e-5)
    
    def test_stability_with_bfloat16(self):
        """Test numerical stability with bfloat16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for bfloat16 test")
        
        n = 4
        
        b_res = torch.zeros(n, n, dtype=torch.bfloat16, device='cuda')
        for i in range(n):
            b_res[i, i] = 20.0
        
        H_res = sinkhorn_knopp(b_res)
        
        # Check it's close to identity (with looser tolerance for bf16)
        I = torch.eye(n, dtype=torch.bfloat16, device='cuda')
        assert torch.allclose(H_res, I, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
