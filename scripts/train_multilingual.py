#!/usr/bin/env python3
"""
Prepare a multilingual IndicCorpV2 dataset (no training).

What it does:
1. Discovers and downloads all IndicCorpV2 splits from Hugging Face (cached per split)
2. Optionally balances and shuffles by split weights (defaults to equal)
3. Saves the combined dataset to <output>/combined_dataset.jsonl

Training is intentionally disabled; this script only downloads and builds data.

Usage examples:
    python scripts/train_multilingual.py --max-examples-per-lang 10000
    python scripts/train_multilingual.py --download-only --max-examples-per-lang 50000
    python scripts/train_multilingual.py --max-examples-per-lang -1  # WARNING: very large
    python scripts/train_multilingual.py --output ./output/multilingual_mhc
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_all_splits() -> List[str]:
    """Fetch all available splits for IndicCorpV2 (each split is a language)."""
    from datasets import get_dataset_split_names

    try:
        splits = get_dataset_split_names("ai4bharat/IndicCorpV2", "indiccorp_v2")
        logger.info(f"Discovered {len(splits)} splits from Hugging Face")
        return splits
    except Exception as e:
        logger.error(f"Failed to fetch splits for ai4bharat/IndicCorpV2: {e}")
        return []


def download_all_datasets(
    cache_dir: str = "./data/indiccorp_cache",
    max_examples_per_lang: Optional[int] = None,
    streaming: bool = True,
    splits: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Download all IndicCorpV2 language datasets (one split per language).

    Args:
        cache_dir: Directory to cache downloaded data
        max_examples_per_lang: Max examples per language (None = all)
        streaming: Use streaming mode for large datasets
        splits: Optional list of split names to download; if None, discover all

    Returns:
        Dictionary mapping split name to list of examples
    """
    from datasets import load_dataset

    try:
        from datasets import get_dataset_config_info  # type: ignore
    except Exception:  # pragma: no cover
        get_dataset_config_info = None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def load_jsonl_cache(path: Path) -> List[Dict[str, Any]]:
        """Load a jsonl cache without reading the whole file into memory."""
        examples: List[Dict[str, Any]] = []
        limit = max_examples_per_lang if max_examples_per_lang and max_examples_per_lang > 0 else None
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"  Stopped reading {path.name} at line {i + 1} due to decode error: {e}")
                    break
        logger.info(f"  Loaded {len(examples)} examples")
        return examples

    split_names = splits or get_all_splits()
    if not split_names:
        logger.error("No splits found; exiting early")
        return {}

    split_sizes: Dict[str, int] = {}
    if get_dataset_config_info is not None:
        try:
            cfg = get_dataset_config_info("ai4bharat/IndicCorpV2", "indiccorp_v2")
            split_sizes = {
                name: info.num_examples
                for name, info in cfg.splits.items()
                if info.num_examples is not None
            }
            if split_sizes:
                logger.info(f"Found split sizes for {len(split_sizes)} splits")
        except Exception as e:
            logger.warning(f"Could not retrieve split sizes; percent progress will be approximate: {e}")

    all_data: Dict[str, List[Dict[str, Any]]] = {}

    for split_name in split_names:
        limit_str = max_examples_per_lang if max_examples_per_lang not in (None, -1, 0) else "all"
        cache_file = cache_path / f"{split_name}_{limit_str}.jsonl"
        legacy_json = cache_path / f"{split_name}_{limit_str}.json"

        if legacy_json.exists() and not cache_file.exists():
            logger.warning(
                f"Found legacy cache {legacy_json.name}; it will be ignored to avoid large in-memory loads. "
                "A jsonl cache will be regenerated."
            )

        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {split_name} from cache: {cache_file}")
            all_data[split_name] = load_jsonl_cache(cache_file)
            continue

        logger.info(f"Downloading split {split_name}...")

        try:
            ds = load_dataset(
                "ai4bharat/IndicCorpV2",
                "indiccorp_v2",
                split=split_name,
                streaming=streaming,
            )

            examples = []
            limit = max_examples_per_lang if max_examples_per_lang and max_examples_per_lang > 0 else None
            tmp_cache = cache_file.with_suffix(cache_file.suffix + ".tmp")
            total_split = split_sizes.get(split_name)

            with open(tmp_cache, "w", encoding="utf-8") as f:
                for i, example in enumerate(ds):
                    if limit is not None and i >= limit:
                        break
                    record = {"text": example.get("text", ""), "lang": split_name}
                    examples.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if (i + 1) % 10000 == 0:
                        if total_split:
                            pct = 100.0 * (i + 1) / total_split
                            logger.info(f"  {split_name}: {i + 1} examples... ({pct:.2f}% of split)")
                        else:
                            logger.info(f"  {split_name}: {i + 1} examples...")

            tmp_cache.replace(cache_file)
            all_data[split_name] = examples
            logger.info(f"  Downloaded {len(examples)} examples for {split_name}")
            logger.info(f"  Cached to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to download split {split_name}: {e}")
            all_data[split_name] = []

    return all_data


def create_combined_dataset(
    all_data: Dict[str, List[Dict[str, Any]]],
    shuffle: bool = True,
    balanced: bool = True,
    lang_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Combine all language datasets into one.
    
    Args:
        all_data: Dictionary of language data keyed by split name
        shuffle: Whether to shuffle the combined data
        balanced: If True, sample proportionally to language weights
        lang_weights: Optional weights per split (defaults to 1.0 each)
        
    Returns:
        Combined list of examples
    """
    weights = lang_weights or {k: 1.0 for k in all_data.keys()}

    if balanced:
        # Calculate target samples per language based on weights
        total_available = sum(len(data) for data in all_data.values())
        total_weight = sum(weights.get(k, 1.0) for k in all_data.keys())
        
        combined = []
        for lang_key, data in all_data.items():
            weight = weights.get(lang_key, 1.0)
            target_ratio = weight / total_weight if total_weight > 0 else 0
            target_samples = min(len(data), int(total_available * target_ratio))
            
            if len(data) > target_samples and target_samples > 0:
                sampled = random.sample(data, target_samples)
            else:
                sampled = data
            
            combined.extend(sampled)
            logger.info(f"  {lang_key}: {len(sampled)} samples (weight={weight})")
    else:
        combined = []
        for data in all_data.values():
            combined.extend(data)
    
    if shuffle:
        random.shuffle(combined)
    
    logger.info(f"Combined dataset: {len(combined)} total examples")
    return combined


class MultilingualDataset:
    """PyTorch Dataset for multilingual text corpus."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 1024,
    ):
        import torch
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.torch = torch
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        text = example.get("text", "")
        
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_multilingual(
    combined_data: List[Dict[str, Any]],
    output_dir: str,
    model_name: str = "Qwen/Qwen3-0.6B",
    batch_size: int = 4,
    grad_accum: int = 8,
    max_length: int = 1024,
    total_steps: int = 100000,
    learning_rate: float = 5e-5,
    save_interval: int = 1000,
    use_compile: bool = True,
    n_streams: int = 4,
    num_fracs: int = 1,
):
    """Train mHC model on combined multilingual data."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    from src.conversionV2 import convert_qwen3_to_mhc_v2, count_parameters_v2
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Multilingual mHC Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Training examples: {len(combined_data)}")
    logger.info(f"Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    logger.info(f"Max length: {max_length}")
    logger.info(f"Total steps: {total_steps}")
    
    # Convert model
    logger.info(f"Converting {model_name} to mHC V2...")
    model, tokenizer = convert_qwen3_to_mhc_v2(
        model_name_or_path=model_name,
        n_streams=n_streams,
        num_fracs=num_fracs,
        device=str(device),
        torch_dtype=dtype,
    )
    
    # Save converted model
    converted_path = output_path / "converted_model"
    model.save_pretrained(converted_path)
    tokenizer.save_pretrained(converted_path)
    logger.info(f"Saved converted model to {converted_path}")
    
    # Model info
    param_counts = count_parameters_v2(model)
    logger.info(f"Parameters: {param_counts['total']:,}")
    
    # Create dataset and dataloader
    dataset = MultilingualDataset(combined_data, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Move model to device
    model = model.to(device=device, dtype=dtype)
    model.train()
    
    # Enable gradient checkpointing
    try:
        model.model.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    except:
        pass
    
    # Compile model
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="default")
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler with warmup
    warmup_steps = int(total_steps * 0.1)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    global_step = 0
    data_iter = iter(dataloader)
    epoch = 0
    accum_loss = 0.0
    
    pbar = tqdm(total=total_steps, desc="Training")
    
    while global_step < total_steps:
        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                logger.info(f"Epoch {epoch}")
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            
            loss.backward()
            accum_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        global_step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        if global_step % 10 == 0:
            logger.info(f"Step {global_step}: loss={accum_loss:.4f}")
        
        accum_loss = 0.0
        
        # Save checkpoint
        if global_step % save_interval == 0:
            ckpt_dir = output_path / f"checkpoint-{global_step}"
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            model_to_save.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")
    
    pbar.close()
    
    # Save final model
    final_dir = output_path / "final"
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_to_save.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and combine all IndicCorpV2 languages (no training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data options
    parser.add_argument("--cache-dir", type=str, default="./data/indiccorp_cache",
                        help="Directory to cache downloaded datasets")
    parser.add_argument("--max-examples-per-lang", type=int, default=100_000,
                        help="Max examples per language (-1 for all; default capped to avoid OOM)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download datasets, skip combining/saving")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Balance dataset across splits (equal weights by default)")
    
    # Model options
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model to convert")
    parser.add_argument("--n-streams", type=int, default=4,
                        help="Number of mHC streams")
    parser.add_argument("--num-fracs", type=int, default=1,
                        help="Number of fractions")
    
    # Training options
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--total-steps", type=int, default=100000,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    
    # Output
    parser.add_argument("--output", type=str, default="./output/multilingual_mhc",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Step 1: Download all datasets
    logger.info("=" * 60)
    logger.info("Step 1: Downloading IndicCorpV2 Datasets")
    logger.info("=" * 60)
    
    max_examples = args.max_examples_per_lang
    if max_examples is not None and max_examples <= 0:
        logger.warning("max-examples-per-lang set to <= 0; downloading full split may exhaust memory/disk.")
        max_examples = None

    all_data = download_all_datasets(
        cache_dir=args.cache_dir,
        max_examples_per_lang=max_examples,
    )
    
    total_examples = sum(len(d) for d in all_data.values())
    logger.info(f"Total downloaded: {total_examples} examples across {len(all_data)} languages")
    
    if args.download_only:
        logger.info("Download complete! (--download-only flag set)")
        return
    
    # Step 2: Combine datasets
    logger.info("=" * 60)
    logger.info("Step 2: Creating Combined Dataset")
    logger.info("=" * 60)
    
    combined = create_combined_dataset(all_data, shuffle=True, balanced=args.balanced)
    
    # Step 3: Save combined dataset (no training)
    logger.info("=" * 60)
    logger.info("Step 3: Saving Combined Dataset (training disabled)")
    logger.info("=" * 60)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_file = output_path / "combined_dataset.jsonl"

    with open(dataset_file, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved combined dataset to {dataset_file} ({len(combined)} examples)")
    logger.info("Script finished. Training is intentionally skipped.")


if __name__ == "__main__":
    main()
