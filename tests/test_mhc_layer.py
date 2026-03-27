"""
Tests for MHCLayer module.
"""

import pytest
import torch
import torch.nn as nn
import math

from src.config import MHCConfig
from src.mhc_layer import MHCLayer, RMSNorm


class SimpleSublayer(nn.Module):
    """Simple sublayer for testing."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, **kwargs):
        return self.linear(x)


class TestRMSNorm:
    """Tests for RMSNorm."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        norm = RMSNorm(1024)
        x = torch.randn(2, 10, 1024)
        y = norm(x)
        
        assert y.shape == x.shape
    
    def test_normalization(self):
        """Test that output has unit RMS."""
        norm = RMSNorm(1024)
        x = torch.randn(2, 10, 1024) * 10
        y = norm(x)
        
        # Check RMS is approximately 1
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestMHCLayerInitialization:
    """Tests for MHCLayer initialization."""
    
    @pytest.fixture
    def mhc_layer(self):
        """Create an MHCLayer for testing."""
        config = MHCConfig(n_streams=4, hidden_size=1024)
        sublayer = SimpleSublayer(1024)
        layernorm = nn.LayerNorm(1024)
        return MHCLayer(config, sublayer, layernorm)
    
    def test_alpha_initialization(self, mhc_layer):
        """Test alpha values are initialized to 0.01."""
        assert abs(mhc_layer.alpha_pre.item() - 0.01) < 1e-6
        assert abs(mhc_layer.alpha_post.item() - 0.01) < 1e-6
        assert abs(mhc_layer.alpha_res.item() - 0.01) < 1e-6
    
    def test_phi_zero_initialization(self, mhc_layer):
        """Test phi weights are initialized to zero."""
        assert torch.allclose(
            mhc_layer.phi_pre.weight,
            torch.zeros_like(mhc_layer.phi_pre.weight)
        )
        assert torch.allclose(
            mhc_layer.phi_post.weight,
            torch.zeros_like(mhc_layer.phi_post.weight)
        )
        assert torch.allclose(
            mhc_layer.phi_res.weight,
            torch.zeros_like(mhc_layer.phi_res.weight)
        )
    
    def test_b_pre_initialization(self, mhc_layer):
        """Test b_pre is initialized for averaging (logit(1/4))."""
        expected = math.log(1.0 / 3.0)  # logit(0.25) = log(0.25/0.75)
        assert torch.allclose(
            mhc_layer.b_pre,
            torch.full_like(mhc_layer.b_pre, expected),
            atol=1e-4
        )
    
    def test_b_post_initialization(self, mhc_layer):
        """Test b_post is initialized to zero (for 2*sigmoid=1)."""
        assert torch.allclose(
            mhc_layer.b_post,
            torch.zeros_like(mhc_layer.b_post)
        )
    
    def test_b_res_diagonal_initialization(self, mhc_layer):
        """Test b_res has large diagonal for identity behavior."""
        n = 4
        b_res = mhc_layer.b_res.squeeze()
        
        # Check diagonal is 20
        for i in range(n):
            assert abs(b_res[i, i].item() - 20.0) < 1e-6
        
        # Check off-diagonal is 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(b_res[i, j].item()) < 1e-6


class TestMHCLayerForward:
    """Tests for MHCLayer forward pass."""
    
    @pytest.fixture
    def mhc_layer(self):
        """Create an MHCLayer for testing."""
        config = MHCConfig(n_streams=4, hidden_size=64)  # Smaller for testing
        sublayer = SimpleSublayer(64)
        layernorm = nn.LayerNorm(64)
        return MHCLayer(config, sublayer, layernorm)
    
    def test_output_shape(self, mhc_layer):
        """Test output shape is (B, S, n, C)."""
        x = torch.randn(2, 10, 4, 64)
        y = mhc_layer(x)
        
        assert y.shape == (2, 10, 4, 64)
    
    def test_gradient_flow(self, mhc_layer):
        """Test gradients flow to all parameters."""
        x = torch.randn(2, 10, 4, 64, requires_grad=True)
        y = mhc_layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check input gradient
        assert x.grad is not None
        
        # Check mHC parameter gradients
        assert mhc_layer.alpha_pre.grad is not None
        assert mhc_layer.alpha_post.grad is not None
        assert mhc_layer.alpha_res.grad is not None
        assert mhc_layer.b_pre.grad is not None
        assert mhc_layer.b_post.grad is not None
        assert mhc_layer.b_res.grad is not None
        assert mhc_layer.phi_pre.weight.grad is not None
        assert mhc_layer.phi_post.weight.grad is not None
        assert mhc_layer.phi_res.weight.grad is not None
    
    def test_no_nan_in_output(self, mhc_layer):
        """Test no NaN values in output."""
        x = torch.randn(2, 10, 4, 64)
        y = mhc_layer(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


class TestMHCLayerEquivalence:
    """Tests for equivalence behavior at initialization."""
    
    def test_identity_stream_behavior(self):
        """Test that identical streams produce identical output streams."""
        config = MHCConfig(n_streams=4, hidden_size=64)
        
        # Simple identity sublayer
        class IdentitySublayer(nn.Module):
            def forward(self, x, **kwargs):
                return x
        
        sublayer = IdentitySublayer()
        layernorm = nn.Identity()  # Skip layernorm for this test
        mhc_layer = MHCLayer(config, sublayer, layernorm)
        
        # Create input with identical streams
        v = torch.randn(2, 10, 64)
        x = v.unsqueeze(-2).expand(-1, -1, 4, -1).clone()
        
        with torch.no_grad():
            y = mhc_layer(x)
        
        # Check output streams are similar (not exactly equal due to residual)
        stream_diffs = []
        for i in range(1, 4):
            diff = (y[:, :, 0, :] - y[:, :, i, :]).abs().max().item()
            stream_diffs.append(diff)
        
        # All streams should be very similar
        assert max(stream_diffs) < 0.1  # Reasonable tolerance
    
    def test_h_pre_averages_streams(self):
        """Test that H_pre coefficients average to 1/n."""
        config = MHCConfig(n_streams=4, hidden_size=64)
        sublayer = SimpleSublayer(64)
        layernorm = nn.LayerNorm(64)
        mhc_layer = MHCLayer(config, sublayer, layernorm)
        
        x = torch.randn(2, 10, 4, 64)
        x_flat = x.view(2, 10, 4 * 64)
        
        with torch.no_grad():
            H_pre, H_post, H_res = mhc_layer.compute_coefficients(x_flat)
        
        # H_pre should be approximately 0.25 for each entry
        h_pre_values = H_pre.squeeze(-2)
        expected = 0.25
        
        assert torch.allclose(h_pre_values, torch.full_like(h_pre_values, expected), atol=0.01)
    
    def test_h_post_copies_equally(self):
        """Test that H_post coefficients are all 1.0."""
        config = MHCConfig(n_streams=4, hidden_size=64)
        sublayer = SimpleSublayer(64)
        layernorm = nn.LayerNorm(64)
        mhc_layer = MHCLayer(config, sublayer, layernorm)
        
        x = torch.randn(2, 10, 4, 64)
        x_flat = x.view(2, 10, 4 * 64)
        
        with torch.no_grad():
            H_pre, H_post, H_res = mhc_layer.compute_coefficients(x_flat)
        
        # H_post should be approximately 1.0 for each entry
        h_post_values = H_post.squeeze(-1)
        expected = 1.0
        
        assert torch.allclose(h_post_values, torch.full_like(h_post_values, expected), atol=0.01)
    
    def test_h_res_is_near_identity(self):
        """Test that H_res is near identity matrix."""
        config = MHCConfig(n_streams=4, hidden_size=64)
        sublayer = SimpleSublayer(64)
        layernorm = nn.LayerNorm(64)
        mhc_layer = MHCLayer(config, sublayer, layernorm)
        
        x = torch.randn(2, 10, 4, 64)
        x_flat = x.view(2, 10, 4 * 64)
        
        with torch.no_grad():
            H_pre, H_post, H_res = mhc_layer.compute_coefficients(x_flat)
        
        # H_res should be near identity
        I = torch.eye(4).unsqueeze(0).unsqueeze(0)
        
        # Check mean across batch
        H_res_mean = H_res.mean(dim=(0, 1))
        
        diff = (H_res_mean - I.squeeze()).abs().max()
        assert diff < 0.01, f"H_res not near identity, max diff: {diff}"


class TestMHCLayerMonitoring:
    """Tests for monitoring functionality."""
    
    def test_get_monitoring_stats(self):
        """Test monitoring stats are computed correctly."""
        config = MHCConfig(n_streams=4, hidden_size=64)
        sublayer = SimpleSublayer(64)
        layernorm = nn.LayerNorm(64)
        mhc_layer = MHCLayer(config, sublayer, layernorm)
        
        x = torch.randn(2, 10, 4, 64)
        
        with torch.no_grad():
            stats = mhc_layer.get_monitoring_stats(x)
        
        assert "forward_gain" in stats
        assert "backward_gain" in stats
        assert "identity_distance" in stats
        assert "alpha_pre" in stats
        assert "alpha_post" in stats
        assert "alpha_res" in stats
        
        # Forward/backward gains should be ~1 for DS matrix
        assert abs(stats["forward_gain"] - 1.0) < 0.01
        assert abs(stats["backward_gain"] - 1.0) < 0.01
        
        # Identity distance should be small at init
        assert stats["identity_distance"] < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
