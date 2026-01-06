#!/usr/bin/env python3
"""
CLI script for validating equivalence between original and mHC models.

Usage:
    python -m scripts.validate_equivalence --original Qwen/Qwen3-0.6B --mhc ./qwen3-mhc
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.conversion import validate_equivalence, load_mhc_model


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
        description="Validate equivalence between original Qwen3 and mHC model"
    )
    
    parser.add_argument(
        "--original",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Original model name or path",
    )
    
    parser.add_argument(
        "--mhc",
        type=str,
        required=True,
        help="mHC model path",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for equivalence (default: 1e-4)",
    )
    
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Custom test sequences",
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
    
    logger.info("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    logger.info("Loading mHC model...")
    mhc_model, _ = load_mhc_model(
        args.mhc,
        device="cuda",
        torch_dtype=torch_dtype,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.original,
        trust_remote_code=True,
    )
    
    logger.info("Running validation...")
    is_equivalent, max_diff = validate_equivalence(
        original_model,
        mhc_model,
        tokenizer,
        device="cuda",
        tolerance=args.tolerance,
        test_sequences=args.sequences,
    )
    
    print("\n" + "=" * 60)
    print("Equivalence Validation Results")
    print("=" * 60)
    print(f"Original model: {args.original}")
    print(f"mHC model:      {args.mhc}")
    print(f"Tolerance:      {args.tolerance:.2e}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Status:         {'PASS ✓' if is_equivalent else 'FAIL ✗'}")
    print("=" * 60)
    
    sys.exit(0 if is_equivalent else 1)


if __name__ == "__main__":
    main()
