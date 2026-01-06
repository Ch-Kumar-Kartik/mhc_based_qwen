"""
Manifold-Constrained Hyper-Connections (mHC) for Qwen3-0.6B

This package implements mHC as described in arXiv:2512.24880 (DeepSeek-AI, Dec 2025).
"""

from .config import MHCConfig
from .sinkhorn import sinkhorn_knopp, SinkhornKnopp
from .mhc_layer import MHCLayer
from .stream_ops import StreamExpand, StreamCollapse
from .qwen3_mhc_model import Qwen3MHCForCausalLM, Qwen3MHCModel
from .conversion import convert_qwen3_to_mhc, validate_equivalence
from .data import load_reasoning_dataset, create_dataloader, print_available_datasets

__version__ = "0.1.0"
__all__ = [
    "MHCConfig",
    "sinkhorn_knopp",
    "SinkhornKnopp",
    "MHCLayer",
    "StreamExpand",
    "StreamCollapse",
    "Qwen3MHCForCausalLM",
    "Qwen3MHCModel",
    "convert_qwen3_to_mhc",
    "validate_equivalence",
    "load_reasoning_dataset",
    "create_dataloader",
    "print_available_datasets",
]
