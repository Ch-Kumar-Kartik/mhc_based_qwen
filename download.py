"""
Download FineWeb-Edu dataset and tokenize with Qwen3 tokenizer.

This script downloads raw text from HuggingFace and tokenizes it for
use with Qwen3-based mHC models.

Usage:
    python download.py --num_samples 100000 --output_dir ./data
    python download.py --num_samples 10000 --shard_size 5000 --output_dir ./data

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SUBSET = "sample-10BT"  # Smaller subset: ~10B tokens
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data"
DEFAULT_MAX_LENGTH = 4096


def download_and_tokenize(
    num_samples: int,
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    subset: str = DEFAULT_SUBSET,
    max_length: int = DEFAULT_MAX_LENGTH,
    shard_size: int = 10000,
):
    """Download FineWeb-Edu and tokenize with specified tokenizer.
    
    Args:
        num_samples: Number of samples to download and tokenize
        output_dir: Directory to save tokenized data
        model_name: HuggingFace model name for tokenizer
        subset: FineWeb-Edu subset name (e.g., 'sample-10BT', 'sample-100BT')
        max_length: Maximum sequence length for tokenization
        shard_size: Number of samples per shard
    """
    print(f"FineWeb-Edu Download & Tokenization")
    print(f"=" * 50)
    print(f"Model tokenizer: {model_name}")
    print(f"Dataset subset:  HuggingFaceFW/fineweb-edu ({subset})")
    print(f"Samples:         {num_samples:,}")
    print(f"Max length:      {max_length}")
    print(f"Output dir:      {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  Pad token:  {tokenizer.pad_token}")
    print()
    
    # Load dataset in streaming mode for efficiency
    print(f"Loading dataset (streaming)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )
    
    # Collect samples
    print(f"Collecting {num_samples:,} samples...")
    samples = []
    for i, example in enumerate(tqdm(dataset, total=num_samples, desc="Downloading")):
        if i >= num_samples:
            break
        samples.append(example["text"])
    
    print(f"  Collected {len(samples):,} samples")
    print()
    
    # Tokenize in batches
    print("Tokenizing...")
    all_input_ids = []
    all_attention_masks = []
    
    batch_size = 1000
    for i in tqdm(range(0, len(samples), batch_size), desc="Tokenizing"):
        batch = samples[i:i + batch_size]
        
        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        
        all_input_ids.extend(encoded["input_ids"])
        all_attention_masks.extend(encoded["attention_mask"])
    
    print(f"  Tokenized {len(all_input_ids):,} samples")
    print()
    
    # Create dataset and save
    print("Saving tokenized dataset...")
    tokenized_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_input_ids,  # For causal LM, labels = input_ids
    })
    
    # Save as Arrow format
    output_path = Path(output_dir) / "fineweb_edu_qwen3"
    tokenized_dataset.save_to_disk(str(output_path))
    
    print(f"  Saved to: {output_path}")
    print()
    
    # Print sample statistics
    print("Dataset statistics:")
    print(f"  Total samples:    {len(tokenized_dataset):,}")
    print(f"  Sequence length:  {max_length}")
    print(f"  Total tokens:     {len(tokenized_dataset) * max_length:,}")
    print()
    
    # Print sample
    print("Sample tokenization:")
    sample = tokenized_dataset[0]
    sample_tokens = sample["input_ids"][:50]
    decoded = tokenizer.decode(sample_tokens)
    print(f"  First 50 tokens: {sample_tokens}")
    print(f"  Decoded: {decoded[:200]}...")
    print()
    
    print("Done!")
    print()
    print("To use with train.py:")
    print(f"  python -m scripts.train --data {output_path} ...")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu and tokenize with Qwen3 tokenizer"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to download (default: 10000)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for tokenized data",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name for tokenizer (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default=DEFAULT_SUBSET,
        help=f"FineWeb-Edu subset (default: {DEFAULT_SUBSET})",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Samples per shard (default: 10000)",
    )
    
    args = parser.parse_args()
    
    download_and_tokenize(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        model_name=args.model,
        subset=args.subset,
        max_length=args.max_length,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
