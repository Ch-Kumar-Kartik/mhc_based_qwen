"""
Manifold-Constrained Hyper-Connections (mHC) for Qwen3-0.6B

This package implements mHC as described in arXiv:2512.24880 (DeepSeek-AI, Dec 2025).
Includes V2 implementation based on the hyper-connections paper approach.
"""

from .config import MHCConfig, MHCTrainingConfig
from .sinkhorn import sinkhorn_knopp, SinkhornKnopp
from .mhc_layer import MHCLayer
from .stream_ops import StreamExpand, StreamCollapse
from .qwen3_mhc_model import Qwen3MHCForCausalLM, Qwen3MHCModel
from .conversion import convert_qwen3_to_mhc, validate_equivalence
from .data import load_reasoning_dataset, create_dataloader, print_available_datasets

# V2 imports (hyper-connections paper implementation)
from .mhc_layerV2 import (
    ManifoldConstrainedHyperConnectionsV2,
    mHCV2,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
)
from .qwen3_mhc_modelV2 import (
    Qwen3MHCConfigV2,
    Qwen3MHCForCausalLMV2,
    Qwen3MHCModelV2,
    Qwen3MHCDecoderLayerV2,
)
from .conversionV2 import (
    convert_qwen3_to_mhc_v2,
    load_mhc_model_v2,
    save_mhc_model_v2,
    count_parameters_v2,
    print_model_summary_v2,
)

__version__ = "0.2.0"
__all__ = [
    # V1 exports
    "MHCConfig",
    "MHCTrainingConfig",
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
    # V2 exports
    "ManifoldConstrainedHyperConnectionsV2",
    "mHCV2",
    "get_expand_reduce_stream_functions",
    "get_init_and_expand_reduce_stream_functions",
    "Qwen3MHCConfigV2",
    "Qwen3MHCForCausalLMV2",
    "Qwen3MHCModelV2",
    "Qwen3MHCDecoderLayerV2",
    "convert_qwen3_to_mhc_v2",
    "load_mhc_model_v2",
    "save_mhc_model_v2",
    "count_parameters_v2",
    "print_model_summary_v2",
]
