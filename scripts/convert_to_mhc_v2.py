#!/usr/bin/env python3
"""
Convert Qwen3 model to mHC V2 architecture.

Usage:
    python -m scripts.convert_to_mhc_v2 --model Qwen/Qwen3-0.6B --output ./output/qwen3_mhc_v2
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversionV2 import (
    convert_qwen3_to_mhc_v2,
    print_model_summary_v2,
)


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3 to mHC V2")
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output/qwen3_mhc_v2_converted",
        help="Output directory for converted model",
    )
    
    parser.add_argument(
        "--n-streams",
        type=int,
        default=4,
        help="Number of mHC streams",
    )
    
    parser.add_argument(
        "--num-fracs",
        type=int,
        default=1,
        help="Number of fractions for frac-connections",
    )
    
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for conversion",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type for model weights",
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip equivalence validation",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    
    logger.info(f"Converting {args.model} to mHC V2")
    logger.info(f"  n_streams: {args.n_streams}")
    logger.info(f"  num_fracs: {args.num_fracs}")
    logger.info(f"  sinkhorn_iters: {args.sinkhorn_iters}")
    logger.info(f"  device: {args.device}")
    logger.info(f"  dtype: {args.dtype}")
    
    # Convert model
    model, tokenizer = convert_qwen3_to_mhc_v2(
        model_name_or_path=args.model,
        n_streams=args.n_streams,
        num_fracs=args.num_fracs,
        sinkhorn_iters=args.sinkhorn_iters,
        output_path=args.output,
        device=args.device,
        torch_dtype=torch_dtype,
        validate=not args.no_validate,
        validation_tolerance=1.0,  # Relaxed for V2
    )
    
    # Print summary
    print_model_summary_v2(model)
    
    logger.info(f"Model saved to {args.output}")
    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
