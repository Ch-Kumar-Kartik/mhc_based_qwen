"""
Tests for weight conversion from Qwen3 to mHC.
"""

import pytest
import torch
import torch.nn as nn

from src.config import MHCConfig
from src.qwen3_mhc_model import Qwen3MHCConfig, Qwen3MHCForCausalLM
from src.conversion import (
    create_mhc_config_from_qwen3,
    count_parameters,
)


class MockQwen3Config:
    """Mock Qwen3 configuration for testing."""
    
    def __init__(self):
        self.vocab_size = 151936
        self.hidden_size = 1024
        self.intermediate_size = 3072
        self.num_hidden_layers = 28
        self.num_attention_heads = 16
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.hidden_act = "silu"
        self.max_position_embeddings = 40960
        self.initializer_range = 0.02
        self.rms_norm_eps = 1e-6
        self.use_cache = True
        self.tie_word_embeddings = True
        self.rope_theta = 1000000.0
        self.attention_dropout = 0.0


class TestCreateMHCConfig:
    """Tests for creating mHC config from Qwen3 config."""
    
    def test_creates_valid_config(self):
        """Test that mHC config is created correctly."""
        qwen3_config = MockQwen3Config()
        mhc_config = create_mhc_config_from_qwen3(qwen3_config, n_streams=4)
        
        assert mhc_config.vocab_size == 151936
        assert mhc_config.hidden_size == 1024
        assert mhc_config.num_hidden_layers == 28
        assert mhc_config.n_streams == 4
        assert mhc_config.mhc_alpha_init == 0.01
        assert mhc_config.mhc_sinkhorn_iterations == 20
    
    def test_different_stream_counts(self):
        """Test with different stream counts."""
        qwen3_config = MockQwen3Config()
        
        for n in [2, 4, 8]:
            mhc_config = create_mhc_config_from_qwen3(qwen3_config, n_streams=n)
            assert mhc_config.n_streams == n


class TestQwen3MHCConfig:
    """Tests for Qwen3MHCConfig."""
    
    def test_get_mhc_config(self):
        """Test getting MHCConfig from Qwen3MHCConfig."""
        config = Qwen3MHCConfig(
            hidden_size=1024,
            n_streams=4,
            mhc_alpha_init=0.01,
        )
        
        mhc_config = config.get_mhc_config()
        
        assert isinstance(mhc_config, MHCConfig)
        assert mhc_config.n_streams == 4
        assert mhc_config.hidden_size == 1024
        assert mhc_config.alpha_init == 0.01


class TestParameterCounting:
    """Tests for parameter counting utility."""
    
    def test_count_simple_model(self):
        """Test counting parameters in a simple model."""
        model = nn.Sequential(
            nn.Linear(100, 200),  # 100*200 + 200 = 20200
            nn.Linear(200, 50),   # 200*50 + 50 = 10050
        )
        
        counts = count_parameters(model)
        
        assert counts["total"] == 20200 + 10050
        assert counts["mhc"] == 0  # No mHC parameters
        assert counts["original"] == 20200 + 10050
    
    def test_identifies_mhc_parameters(self):
        """Test that mHC parameters are identified correctly."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 100)
                self.phi_pre = nn.Linear(100, 4)  # mHC param
                self.b_res = nn.Parameter(torch.randn(4, 4))  # mHC param
                self.alpha_pre = nn.Parameter(torch.tensor(0.01))  # mHC param
        
        model = MockModel()
        counts = count_parameters(model)
        
        # phi_pre: 100*4 + 4 = 404
        # b_res: 16
        # alpha_pre: 1
        assert counts["mhc"] == 404 + 16 + 1
        
        # linear: 100*100 + 100 = 10100
        assert counts["original"] == 10100


class TestMHCConfigValues:
    """Tests for MHCConfig computed values."""
    
    def test_expanded_size(self):
        """Test expanded_size property."""
        config = MHCConfig(n_streams=4, hidden_size=1024)
        assert config.expanded_size == 4096
    
    def test_phi_shapes(self):
        """Test phi projection shapes."""
        config = MHCConfig(n_streams=4, hidden_size=1024)
        
        assert config.phi_pre_shape == (4096, 4)
        assert config.phi_post_shape == (4096, 4)
        assert config.phi_res_shape == (4096, 16)
    
    def test_b_pre_init_value(self):
        """Test b_pre_init is computed correctly."""
        config = MHCConfig(n_streams=4, hidden_size=1024)
        
        # For n=4: logit(1/4) = log(1/3) ≈ -1.0986
        import math
        expected = math.log(1.0 / 3.0)
        
        assert abs(config.b_pre_init - expected) < 1e-4


class TestQwen3MHCModel:
    """Tests for Qwen3MHCForCausalLM structure."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for fast testing."""
        return Qwen3MHCConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            n_streams=4,
        )
    
    def test_model_creation(self, small_config):
        """Test model can be created."""
        model = Qwen3MHCForCausalLM(small_config)
        
        assert model is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'lm_head')
    
    def test_stream_ops_present(self, small_config):
        """Test stream expand/collapse modules are present."""
        model = Qwen3MHCForCausalLM(small_config)
        
        assert hasattr(model.model, 'stream_expand')
        assert hasattr(model.model, 'stream_collapse')
        assert model.model.stream_expand.n_streams == 4
        assert model.model.stream_collapse.n_streams == 4
    
    def test_get_mhc_parameters(self, small_config):
        """Test getting mHC-specific parameters."""
        model = Qwen3MHCForCausalLM(small_config)
        
        # Need to create decoder layers first
        # This would require more setup, so we just check the method exists
        assert hasattr(model, 'get_mhc_parameters')
    
    def test_get_original_parameters(self, small_config):
        """Test getting original parameters."""
        model = Qwen3MHCForCausalLM(small_config)
        
        assert hasattr(model, 'get_original_parameters')


class TestWeightMapping:
    """Tests for weight mapping between original and mHC models."""
    
    def test_embedding_unchanged(self):
        """Test embedding weights are unchanged in mapping."""
        # The embedding should map directly
        from src.conversion import get_weight_mapping
        
        mapping = get_weight_mapping()
        
        assert "model.embed_tokens.weight" in mapping
        assert mapping["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    
    def test_final_norm_unchanged(self):
        """Test final norm weights are unchanged."""
        from src.conversion import get_weight_mapping
        
        mapping = get_weight_mapping()
        
        assert "model.norm.weight" in mapping
        assert mapping["model.norm.weight"] == "model.norm.weight"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
