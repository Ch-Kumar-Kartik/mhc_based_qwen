"""
High-performance streaming data loader for Hugging Face datasets (100GB+).

Handles indiccorp and similar Hugging Face datasets with proper shard support.

Key optimizations:
- Loads HF datasets from disk cache
- Streams with proper shard distribution to workers
- Tokenization on-the-fly or pre-tokenized
- Efficient memory usage with prefetching
"""

import logging
from typing import Optional, Iterator, Dict, Any
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _collate_batch_fn(examples) -> Dict[str, Any]:
    """Collate function to convert examples to tensor batches.
    
    Defined at module level to be pickleable for multiprocessing on Windows.
    """
    batch = {
        "input_ids": torch.stack([torch.tensor(ex["input_ids"], dtype=torch.long) for ex in examples]),
        "attention_mask": torch.stack([torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in examples]),
    }
    # Add labels if available
    if "labels" in examples[0]:
        batch["labels"] = torch.stack([torch.tensor(ex["labels"], dtype=torch.long) for ex in examples])
    else:
        batch["labels"] = batch["input_ids"].clone()
    return batch


@dataclass
class StreamConfig:
    """Configuration for streaming datasets."""
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    max_length: int = 2048
    drop_last: bool = True
    shuffle_buffer_size: int = 10000
    seed: int = 42


class HFDataset(IterableDataset):
    """
    Stream Hugging Face datasets with proper shard handling.
    
    For 800GB+ datasets:
    - Loads dataset shards from disk
    - Tokenizes on-demand per worker
    - Distributes shards across workers
    - Memory-efficient streaming
    """
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 2048,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        text_column: str = "text",
    ):
        from datasets import load_from_disk
        
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.text_column = text_column
        self.rank = 0
        self.world_size = 1
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}...")
        
        # Check if path exists first
        import os
        if not os.path.exists(dataset_path):
            available_datasets = []
            data_dir = os.path.dirname(dataset_path) or "./data"
            if os.path.exists(data_dir):
                for d in os.listdir(data_dir):
                    full_path = os.path.join(data_dir, d)
                    if os.path.isdir(full_path) and os.listdir(full_path):
                        available_datasets.append(d)
            msg = f"Dataset path does not exist: {dataset_path}"
            if available_datasets:
                msg += f"\n  Available datasets in '{data_dir}': {available_datasets}"
            msg += "\n  Run python -m scripts.data.download_indiccorp first to create the dataset."
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        try:
            self.dataset = load_from_disk(dataset_path)
        except Exception as e:
            # Check if it's a multi-language directory (indiccorp_all structure)
            import os
            contents = os.listdir(dataset_path) if os.path.isdir(dataset_path) else []
            
            # Check if it's a multi-language directory with language subdirectories
            language_dirs = [d for d in contents if os.path.isdir(os.path.join(dataset_path, d)) 
                            and not d.startswith('.')]
            
            if language_dirs and not any(f.endswith('.arrow') or f == 'dataset_info.json' for f in contents):
                # This looks like indiccorp_all structure - load all language shards
                logger.info(f"Detected multi-language directory with {len(language_dirs)} language folders")
                from datasets import concatenate_datasets
                
                all_datasets = []
                for lang_dir in sorted(language_dirs):
                    lang_path = os.path.join(dataset_path, lang_dir)
                    raw_shards_path = os.path.join(lang_path, "raw_shards")
                    
                    # Look for raw_shards subdirectory
                    if os.path.exists(raw_shards_path):
                        logger.info(f"  Loading {lang_dir} from raw_shards...")
                        shard_dirs = sorted([d for d in os.listdir(raw_shards_path) 
                                           if os.path.isdir(os.path.join(raw_shards_path, d)) 
                                           and d.startswith('shard-')])
                        
                        for shard_dir in shard_dirs:
                            shard_path = os.path.join(raw_shards_path, shard_dir)
                            try:
                                shard_ds = load_from_disk(shard_path)
                                all_datasets.append(shard_ds)
                            except Exception as shard_e:
                                logger.warning(f"    Failed to load shard {shard_dir}: {shard_e}")
                
                if all_datasets:
                    self.dataset = concatenate_datasets(all_datasets)
                    logger.info(f"Concatenated {len(all_datasets)} shards into dataset with {len(self.dataset):,} examples")
                else:
                    raise FileNotFoundError(f"No valid shards found in {dataset_path}")
            else:
                # Original error handling
                if not contents:
                    msg = (
                        f"Dataset directory is empty: {dataset_path}\n"
                        "  Run python -m scripts.data.download_indiccorp to populate it."
                    )
                elif not any(f.endswith('.arrow') or f == 'dataset_info.json' for f in contents):
                    msg = f"Directory is not a valid HuggingFace dataset: {dataset_path}\n  Contents: {contents}\n  Expected: .arrow files and dataset_info.json or language subdirectories"
                else:
                    msg = f"Failed to load dataset: {e}"
                logger.error(msg)
                raise FileNotFoundError(msg) from e
        
        logger.info(f"Dataset loaded: {len(self.dataset):,} examples")
        
        # Verify text column exists
        if text_column not in self.dataset.column_names:
            logger.error(f"Column '{text_column}' not found. Available: {self.dataset.column_names}")
            raise ValueError(f"Column '{text_column}' not found in dataset")
    
    def _tokenize_example(self, text: str) -> Dict[str, Any]:
        """Tokenize a single example."""
        if not self.tokenizer:
            return {"input_ids": [0] * self.max_length, "attention_mask": [1] * self.max_length}
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        encoded["labels"] = encoded["input_ids"][:]
        return encoded
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through dataset examples."""
        # Get worker info for distributed iteration
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Distribute examples across workers
        total_examples = len(self.dataset)
        examples_per_worker = total_examples // num_workers
        start_idx = worker_id * examples_per_worker
        end_idx = start_idx + examples_per_worker if worker_id < num_workers - 1 else total_examples
        
        logger.info(f"Worker {worker_id}/{num_workers}: Processing examples {start_idx}-{end_idx}")
        
        buffer = []
        rng = np.random.RandomState(self.seed + worker_id)
        
        for idx in range(start_idx, end_idx):
            example = self.dataset[int(idx)]
            
            text = example.get(self.text_column, "")
            if not isinstance(text, str) or not text.strip():
                continue
            
            tokenized = self._tokenize_example(text)
            buffer.append(tokenized)
            
            # Shuffle within buffer window
            if len(buffer) >= self.shuffle_buffer_size:
                rng.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []
        
        # Yield remaining buffered items
        if buffer:
            rng.shuffle(buffer)
            for item in buffer:
                yield item


def load_hf_dataloader(
    dataset_path: str,
    tokenizer: Optional[PreTrainedTokenizer],
    config: StreamConfig,
    text_column: str = "text",
) -> DataLoader:
    """
    Load HF dataset from disk and create streaming dataloader.
    
    Usage:
        dataloader = load_hf_dataloader("./data/indiccorp_all", tokenizer, config)
        for batch in dataloader:
            # batch["input_ids"], batch["labels"], etc.
    """
    dataset = HFDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
        text_column=text_column,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        drop_last=config.drop_last,
        pin_memory=True,
        collate_fn=_collate_batch_fn,
    )
    
    return dataloader
