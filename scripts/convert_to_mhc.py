#!/usr/bin/env python3
"""
CLI script for converting Qwen3 models to mHC architecture.

Usage:
    python -m scripts.convert_to_mhc --model Qwen/Qwen3-0.6B --output ./qwen3-mhc --streams 4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversion import convert_qwen3_to_mhc, print_model_summary


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 model to mHC architecture"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-0.6B)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model",
    )
    
    parser.add_argument(
        "--streams",
        type=int,
        default=4,
        help="Number of mHC streams (default: 4)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for conversion",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights (default: bfloat16)",
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip equivalence validation",
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Tolerance for equivalence validation (default: 2.0 for bfloat16)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]
    
    logger.info(f"Converting {args.model} to mHC architecture")
    logger.info(f"  Streams: {args.streams}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Dtype: {args.dtype}")
    logger.info(f"  Output: {args.output}")
    
    try:
        model, tokenizer = convert_qwen3_to_mhc(
            model_name_or_path=args.model,
            n_streams=args.streams,
            output_path=args.output,
            device=args.device,
            torch_dtype=torch_dtype,
            validate=not args.no_validate,
            validation_tolerance=args.tolerance,
        )
        
        print("\n")
        print_model_summary(model)
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
