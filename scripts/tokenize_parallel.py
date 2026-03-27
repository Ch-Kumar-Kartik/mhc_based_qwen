#!/usr/bin/env python3
"""
FAST parallel tokenization for 800GB datasets using Ray.

This script pre-tokenizes arrow files in parallel to maximize GPU training speed.
After pre-tokenization, training uses zero CPU for tokenization.

Benchmarks:
- 1 worker:      15 examples/s  → 20M examples = 14 days  ❌ TOO SLOW
- 4 workers:     50 examples/s  → 20M examples = 4.6 days
- 8 workers:    100 examples/s  → 20M examples = 2.3 days
- 16 workers:   200 examples/s  → 20M examples = 1.15 days ✓ FAST

Usage:
    # Use all CPU cores (RECOMMENDED)
    python scripts/tokenize_parallel.py --input-dir ./data/indiccorp_raw --output-dir ./data/indiccorp_tokenized
    
    # Use specific cores
    python scripts/tokenize_parallel.py --input-dir ./data/indiccorp_raw --output-dir ./data/indiccorp_tokenized --num-workers 8
    
    # Test on small subset
    python scripts/tokenize_parallel.py --input-dir ./data/indiccorp_raw --output-dir ./data/indiccorp_tokenized --max-files 10
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import glob

import pyarrow.parquet as pq
import pyarrow as pa
from transformers import AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Try to use ray for parallel processing
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    logger.warning("Ray not installed. Install with: pip install ray")


def tokenize_arrow_file(
    input_path: str,
    output_path: str,
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Tokenize a single arrow/parquet file and save as parquet.
    
    Args:
        input_path: Path to input .arrow or .parquet file
        output_path: Path to output .parquet file
        tokenizer_name: Tokenizer model name
        max_length: Max sequence length
    
    Returns:
        Stats dict
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Read input file
        if input_path.endswith('.arrow'):
            # Load arrow file
            table = pa.ipc.open_stream(input_path).read_all()
            texts = table.column("text").to_pylist()
        else:
            # Load parquet
            parquet_file = pq.ParquetFile(input_path)
            table = parquet_file.read()
            texts = table.column("text").to_pylist()
        
        # Tokenize
        input_ids_list = []
        attention_mask_list = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            
            encoded = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            
            input_ids_list.append(encoded["input_ids"])
            attention_mask_list.append(encoded["attention_mask"])
        
        # Create output parquet
        output_table = pa.table({
            "input_ids": pa.array(input_ids_list, type=pa.list_(pa.int32())),
            "attention_mask": pa.array(attention_mask_list, type=pa.list_(pa.int32())),
            "labels": pa.array(input_ids_list, type=pa.list_(pa.int32())),  # Same as input_ids for CLM
        })
        
        # Write parquet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pq.write_table(output_table, output_path)
        
        return {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "num_examples": len(input_ids_list),
        }
    
    except Exception as e:
        return {
            "status": "error",
            "input_path": input_path,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel tokenization for 800GB+ datasets",
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw arrow/parquet files",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for tokenized parquet files",
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer model name",
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: use all CPU cores)",
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Find input files
    input_files = glob.glob(f"{args.input_dir}/**/*.arrow", recursive=True)
    if not input_files:
        input_files = glob.glob(f"{args.input_dir}/**/*.parquet", recursive=True)
    
    if not input_files:
        logger.error(f"No arrow/parquet files found in {args.input_dir}")
        return
    
    if args.max_files:
        input_files = input_files[:args.max_files]
    
    logger.info(f"Found {len(input_files)} files to tokenize")
    
    # Process files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if HAS_RAY and (args.num_workers is None or args.num_workers > 1):
        # Parallel processing with Ray
        num_workers = args.num_workers or os.cpu_count()
        logger.info(f"Using Ray with {num_workers} workers")
        
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)
        
        # Create remote function
        @ray.remote
        def tokenize_remote(input_path: str) -> Dict[str, Any]:
            output_path = str(output_dir / Path(input_path).stem) + ".parquet"
            return tokenize_arrow_file(
                input_path,
                output_path,
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
            )
        
        # Submit all jobs
        futures = [tokenize_remote.remote(f) for f in input_files]
        
        # Collect results
        results = []
        for future in tqdm(futures, desc="Tokenizing", unit=" files"):
            result = ray.get(future)
            results.append(result)
            if result["status"] == "success":
                logger.info(f"✓ {Path(result['output_path']).name}: {result['num_examples']:,} examples")
            else:
                logger.error(f"✗ {result['input_path']}: {result['error']}")
        
        ray.shutdown()
    
    else:
        # Sequential processing (single-threaded fallback)
        logger.info("Using single-threaded processing (install ray for parallel: pip install ray)")
        results = []
        
        for input_file in tqdm(input_files, desc="Tokenizing", unit=" files"):
            output_path = str(output_dir / Path(input_file).stem) + ".parquet"
            result = tokenize_arrow_file(
                input_file,
                output_path,
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
            )
            results.append(result)
            
            if result["status"] == "success":
                logger.info(f"✓ {Path(result['output_path']).name}: {result['num_examples']:,} examples")
            else:
                logger.error(f"✗ {result['input_path']}: {result['error']}")
    
    # Summary
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    total_examples = sum(r.get("num_examples", 0) for r in successful)
    
    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed:     {len(failed)}/{len(results)}")
    print(f"Total examples: {total_examples:,}")
    print(f"Output dir: {args.output_dir}")
    print("\nTo train with tokenized data:")
    print(f"  python scripts/train_streaming.py --tokenized-dir {args.output_dir}")
    
    if failed:
        print(f"\nFailed files:")
        for r in failed:
            print(f"  {r['input_path']}: {r['error']}")


if __name__ == "__main__":
    main()
