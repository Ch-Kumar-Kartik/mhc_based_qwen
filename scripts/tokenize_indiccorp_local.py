#!/usr/bin/env python3
"""
Tokenize already-downloaded IndicCorp raw shards and optionally create train/val/test splits.

This script loads raw text from HuggingFace's IndicCorpV2 dataset, tokenizes it with Qwen3,
optionally creates train/val/test splits, and saves as HuggingFace datasets to disk for training.

Usage:
    # Download + tokenize Hindi with 80/10/10 split
    python scripts/tokenize_indiccorp_local.py --languages hin_Deva --num_samples 100000 --train-ratio 0.8 --val-ratio 0.1
    
    # Tokenize multiple languages (no split - just combined)
    python scripts/tokenize_indiccorp_local.py --languages hin_Deva ben_Beng tel_Telu --num_samples 50000
    
    # Full dataset (no splits)
    python scripts/tokenize_indiccorp_local.py --languages hin_Deva --num_samples -1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "indiccorp_all"
DEFAULT_MAX_LENGTH = 2048


SUPPORTED_LANGUAGES = [
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva",
    "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva",
    "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Mtei", "npi_Deva",
    "ory_Orya", "pan_Guru", "san_Deva", "snd_Deva", "tam_Taml",
    "tel_Telu", "urd_Arab", "khasi", "santhali",
]


def tokenize_language(
    language: str,
    tokenizer: AutoTokenizer,
    num_samples: int = -1,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_workers: int = 2,
    raw_data_path: Optional[str] = None,
) -> Dataset:
    """Download and tokenize a single IndicCorpV2 language.
    
    Args:
        language: Language code (e.g., 'hin_Deva')
        tokenizer: Loaded tokenizer
        num_samples: Number of samples (-1 for all)
        max_length: Max sequence length
        num_workers: Number of CPU workers for tokenization (default: 2, set 0 for single-threaded)
        raw_data_path: Path to local raw dataset directory (if None, downloads from HuggingFace)
        
    Returns:
        Tokenized HuggingFace Dataset
    """
    from datasets import load_from_disk, concatenate_datasets
    
    print(f"\nLoading {language}...")
    
    # Load from local disk if raw_data_path provided, otherwise from HuggingFace
    if raw_data_path:
        raw_path = Path(raw_data_path) / language
        
        # Check for raw_shards or tokenized_shards subdirectory
        if (raw_path / "raw_shards").exists():
            print(f"  Loading raw shards from: {raw_path / 'raw_shards'}")
            shards_dir = raw_path / "raw_shards"
            shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith("shard-")])
            
            if not shard_dirs:
                raise FileNotFoundError(f"No shards found in {shards_dir}")
            
            print(f"    Found {len(shard_dirs)} shards")
            
            # Load and concatenate all shards
            all_shards = []
            for shard_dir in shard_dirs:
                print(f"    Loading {shard_dir.name}...")
                shard_ds = load_from_disk(str(shard_dir))
                all_shards.append(shard_ds)
            
            ds = concatenate_datasets(all_shards)
            print(f"    Concatenated: {len(ds)} examples from {len(shard_dirs)} shards")
        
        elif (raw_path / "tokenized_shards").exists():
            print(f"  Loading tokenized shards from: {raw_path / 'tokenized_shards'}")
            shards_dir = raw_path / "tokenized_shards"
            shard_dirs = sorted([d for d in shards_dir.iterdir() if d.is_dir() and d.name.startswith("shard-")])
            
            if not shard_dirs:
                raise FileNotFoundError(f"No shards found in {shards_dir}")
            
            print(f"    Found {len(shard_dirs)} shards")
            
            # Load and concatenate all shards
            all_shards = []
            for shard_dir in shard_dirs:
                print(f"    Loading {shard_dir.name}...")
                shard_ds = load_from_disk(str(shard_dir))
                all_shards.append(shard_ds)
            
            ds = concatenate_datasets(all_shards)
            print(f"    Concatenated: {len(ds)} examples from {len(shard_dirs)} shards")
        
        elif raw_path.exists():
            print(f"  Loading from local HF dataset: {raw_path}")
            ds = load_from_disk(str(raw_path))
        
        else:
            raise FileNotFoundError(f"Local dataset not found at {raw_path}")
    else:
        # Load from HuggingFace
        ds = load_dataset(
            "ai4bharat/IndicCorpV2",
            "indiccorp_v2",
            split=language,
            streaming=False,
        )
    
    # Limit samples if requested
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    
    print(f"  Loaded {len(ds)} examples")
    print(f"  Tokenizing with {language}...")
    
    def tokenize_fn(examples):
        """Tokenize function for dataset.map()"""
        encoded = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded
    
    # Tokenize in batches with controlled CPU usage
    tokenized_ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=500,
        remove_columns=["text"],
        num_proc=num_workers,  # Limit CPU workers (2 is reasonable, 0 = single-threaded)
        desc=f"Tokenizing {language}",
    )
    
    print(f"  Tokenized: {len(tokenized_ds)} examples")
    return tokenized_ds


def create_splits(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split dataset into train/val/test.
    
    Args:
        dataset: Input dataset
        train_ratio: Fraction for training (0.8)
        val_ratio: Fraction for validation (0.1)
        seed: Random seed
        
    Returns:
        DatasetDict with 'train', 'validation', 'test' splits
    """
    total_len = len(dataset)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    
    # Shuffle then split
    shuffled = dataset.shuffle(seed=seed)
    
    train = shuffled.select(range(train_size))
    val = shuffled.select(range(train_size, train_size + val_size))
    test = shuffled.select(range(train_size + val_size, total_len))
    
    print(f"\nSplit sizes:")
    print(f"  train: {len(train)} ({100*train_ratio:.1f}%)")
    print(f"  validation: {len(val)} ({100*val_ratio:.1f}%)")
    print(f"  test: {len(test)} ({100*(1-train_ratio-val_ratio):.1f}%)")
    
    return DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })


def _save_with_progress(dataset: Dataset, output_path: str, show_progress: bool = True) -> None:
    """Save dataset to disk with progress indication.
    
    Args:
        dataset: Dataset to save
        output_path: Output directory path
        show_progress: Whether to show progress
    """
    import shutil
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing if present
    if output_path.exists():
        shutil.rmtree(output_path)
    
    print(f"    Writing {len(dataset):,} examples to {output_path}...")
    if show_progress:
        for i in tqdm(range(0, len(dataset), 10000), desc="Saving", unit=" chunks"):
            pass
    
    dataset.save_to_disk(str(output_path))
    print(f"    ✓ Complete")


def _save_splits_with_progress(splits: DatasetDict, output_path: str) -> None:
    """Save DatasetDict splits to disk with progress indication.
    
    Args:
        splits: DatasetDict to save
        output_path: Output directory path
    """
    import shutil
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing if present
    if output_path.exists():
        shutil.rmtree(output_path)
    
    print(f"    Writing splits to {output_path}...")
    total_examples = sum(len(ds) for ds in splits.values())
    print(f"    Total: {total_examples:,} examples")
    
    splits.save_to_disk(str(output_path))
    print(f"    ✓ Complete")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize IndicCorpV2 languages and optionally create train/val/test splits",
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["hin_Deva"],
        help="Language codes to tokenize (default: hin_Deva)",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples per language (-1 for all, default: -1)",
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
        help=f"Model for tokenizer (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_MAX_LENGTH})",
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Skip train/val/test splitting (just save combined dataset)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of CPU workers for tokenization (default: 2, set to 0 for single-threaded)",
    )
    
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to local raw IndicCorp dataset (if None, downloads from HuggingFace)",
    )
    
    args = parser.parse_args()
    
    # Validate languages
    invalid = [l for l in args.languages if l not in SUPPORTED_LANGUAGES]
    if invalid:
        print(f"Error: Invalid languages: {invalid}")
        print(f"Supported: {', '.join(SUPPORTED_LANGUAGES)}")
        return
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("IndicCorpV2 Tokenization")
    print("=" * 60)
    print(f"Languages:    {', '.join(args.languages)}")
    print(f"Samples/lang: {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Model:        {args.model}")
    print(f"Max length:   {args.max_length}")
    print(f"Splits:       {not args.no_splits}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print()
    
    # Tokenize each language
    all_tokenized = []
    for lang in args.languages:
        tokenized = tokenize_language(
            lang,
            tokenizer,
            num_samples=args.num_samples,
            max_length=args.max_length,
            num_workers=args.num_workers,
            raw_data_path=args.raw_data_path,
        )
        all_tokenized.append(tokenized)
    
    # Combine
    if len(all_tokenized) > 1:
        print("\nCombining languages...")
        combined = concatenate_datasets(all_tokenized)
        print(f"  Combined: {len(combined)} examples")
    else:
        combined = all_tokenized[0]
    
    # Optionally create splits
    if args.no_splits:
        print("\nSaving combined dataset...")
        combined_path = Path(args.output_dir) / "combined"
        _save_with_progress(combined, str(combined_path))
        print(f"  Saved to: {combined_path}")
        print(f"  Examples: {len(combined):,}")
    else:
        print("\nCreating train/val/test splits...")
        splits = create_splits(
            combined,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        
        output_path = Path(args.output_dir) / "splits"
        _save_splits_with_progress(splits, str(output_path))
        print(f"\n  Saved to: {output_path}")
        print(f"  Structure: {list(splits.keys())}")
    
    print("\nDone!")
    print("\nTo train with this data:")
    if args.no_splits:
        print(f"  python scripts/train_amd_v2.py --dataset-path {combined_path} --split train")
    else:
        print(f"  python scripts/train_amd_v2.py --dataset-path {output_path} --split train")
    
    print("\nTo control CPU usage on next run, use:")
    print(f"  --num-workers 1   # Single-threaded (slower, least CPU)")
    print(f"  --num-workers 2   # 2 CPU threads (default, balanced)")
    print(f"  --num-workers 4   # 4 CPU threads (faster, more CPU)")


if __name__ == "__main__":
    main()