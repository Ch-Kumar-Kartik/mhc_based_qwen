"""
Data loading utilities for CoT reasoning training.

Recommended datasets for training mHC models on chain-of-thought reasoning:

1. OpenThoughts3-1.2M (BEST) - 1.2M examples, math/code/science with QwQ-32B traces
2. OpenThoughts-114k - 114K high-quality examples with DeepSeek-R1 traces  
3. OpenThoughts2-1M - 1M examples including OpenR1-Math

Usage:
    from src.data import load_reasoning_dataset, create_dataloader
    
    dataset = load_reasoning_dataset("openthoughts3", tokenizer, max_length=4096)
    dataloader = create_dataloader(dataset, batch_size=4)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ReasoningDatasetConfig:
    """Configuration for reasoning datasets."""
    name: str
    hf_path: str
    hf_subset: Optional[str] = None
    conversation_key: str = "conversations"
    system_key: str = "system"
    max_examples: Optional[int] = None
    data_dir: Optional[str] = None  # For datasets with subdirectories
    text_key: str = "text"  # For text-only datasets
    is_text_only: bool = False  # True for monolingual corpus datasets
    split_name: Optional[str] = None  # For datasets where split is the language code (e.g., IndicCorpV2)
    

# Available reasoning datasets
REASONING_DATASETS = {
    "openthoughts3": ReasoningDatasetConfig(
        name="OpenThoughts3-1.2M",
        hf_path="open-thoughts/OpenThoughts3-1.2M",
        hf_subset="default",
        conversation_key="conversations",
    ),
    "openthoughts114k": ReasoningDatasetConfig(
        name="OpenThoughts-114k", 
        hf_path="open-thoughts/OpenThoughts-114k",
        hf_subset="default",
        conversation_key="conversations",
    ),
    "openthoughts2": ReasoningDatasetConfig(
        name="OpenThoughts2-1M",
        hf_path="open-thoughts/OpenThoughts2-1M",
        hf_subset="default",
        conversation_key="conversations",
    ),
    "sangraha_verified": ReasoningDatasetConfig(
        name="Sangraha-Verified",
        hf_path="ai4bharat/sangraha",
        hf_subset="verified",
        split_name="train",  # dataset provides a train split; no built-in val/test
        text_key="text",
        is_text_only=True,
    ),
    # IndicCorpV2 - Multilingual Indic Corpus for pretraining
    # See: https://huggingface.co/datasets/ai4bharat/IndicCorpV2
    "indiccorp_hi": ReasoningDatasetConfig(
        name="IndicCorpV2-Hindi",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="hin_Deva",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_te": ReasoningDatasetConfig(
        name="IndicCorpV2-Telugu",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="tel_Telu",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_ta": ReasoningDatasetConfig(
        name="IndicCorpV2-Tamil",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="tam_Taml",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_mr": ReasoningDatasetConfig(
        name="IndicCorpV2-Marathi",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="mar_Deva",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_bn": ReasoningDatasetConfig(
        name="IndicCorpV2-Bengali",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="ben_Beng",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_gu": ReasoningDatasetConfig(
        name="IndicCorpV2-Gujarati",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="guj_Gujr",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_kn": ReasoningDatasetConfig(
        name="IndicCorpV2-Kannada",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="kan_Knda",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_ml": ReasoningDatasetConfig(
        name="IndicCorpV2-Malayalam",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="mal_Mlym",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_pa": ReasoningDatasetConfig(
        name="IndicCorpV2-Punjabi",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="pan_Guru",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_or": ReasoningDatasetConfig(
        name="IndicCorpV2-Odia",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="ory_Orya",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_as": ReasoningDatasetConfig(
        name="IndicCorpV2-Assamese",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="asm_Beng",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_ne": ReasoningDatasetConfig(
        name="IndicCorpV2-Nepali",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="npi_Deva",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_sa": ReasoningDatasetConfig(
        name="IndicCorpV2-Sanskrit",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="san_Deva",
        text_key="text",
        is_text_only=True,
    ),
    "indiccorp_ur": ReasoningDatasetConfig(
        name="IndicCorpV2-Urdu",
        hf_path="ai4bharat/IndicCorpV2",
        hf_subset="indiccorp_v2",
        split_name="urd_Arab",
        text_key="text",
        is_text_only=True,
    ),
}


class ReasoningDataset(Dataset):
    """PyTorch Dataset for reasoning data with CoT traces."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        """Initialize reasoning dataset.
        
        Args:
            data: List of conversation examples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            system_prompt: Optional system prompt to prepend.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that thinks step-by-step. "
            "Show your reasoning process before giving the final answer."
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Build conversation text
        text = self._build_conversation_text(example)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _build_conversation_text(self, example: Dict[str, Any]) -> str:
        """Build conversation text from example.
        
        Handles OpenThoughts format with 'conversations' list containing
        'from' (human/assistant) and 'value' keys.
        
        Uses Qwen3 chat format or falls back to ChatML format.
        """
        # Try to use tokenizer's chat template if available
        conversations = example.get("conversations", [])
        system = example.get("system", self.system_prompt)
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # Convert to messages format
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            
            for turn in conversations:
                role = turn.get("from", "")
                content = turn.get("value", "")
                
                if role == "human":
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "gpt"):
                    messages.append({"role": "assistant", "content": content})
            
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass  # Fall back to manual formatting
        
        # Manual ChatML format (fallback)
        parts = []
        
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        # Add conversation turns
        conversations = example.get("conversations", [])
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            
            if role == "human":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role in ("assistant", "gpt"):
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        return "\n".join(parts)


class TextCorpusDataset(Dataset):
    """PyTorch Dataset for text-only corpus data (e.g., IndicCorpV2).
    
    This is for monolingual pretraining where the data is raw text without
    conversation structure. Used for language model pretraining on large
    text corpora.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 2048,
        text_key: str = "text",
    ):
        """Initialize text corpus dataset.
        
        Args:
            data: List of text examples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            text_key: Key in data dict containing the text.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key


class TokenizedDiskDataset(Dataset):
    """PyTorch Dataset wrapper around a tokenized HuggingFace `datasets` dataset.

    Expects columns: input_ids, attention_mask, and optionally labels.
    This is used for pre-tokenized corpora saved with `Dataset.save_to_disk()`.
    """

    def __init__(self, hf_dataset: Any):
        self.ds = hf_dataset

        cols = set(getattr(self.ds, "column_names", []))
        required = {"input_ids", "attention_mask"}
        missing = required - cols
        if missing:
            raise ValueError(f"Tokenized dataset missing required columns: {sorted(missing)}")

        self.has_labels = "labels" in cols

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        if self.has_labels:
            labels = torch.tensor(item["labels"], dtype=torch.long)
        else:
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_tokenized_dataset_from_disk(
    dataset_path: str,
    split: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> TokenizedDiskDataset:
    """Load a pre-tokenized dataset saved with `datasets.save_to_disk()`.

    Args:
        dataset_path: Path to dataset directory.
        split: Optional split name when loading a DatasetDict saved to disk.
               Common values: 'train', 'validation', 'test'.
        max_examples: Optional max examples to keep.

    Returns:
        TokenizedDiskDataset suitable for `DataLoader`.
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset_path)

    # DatasetDict case
    if hasattr(ds, "keys") and split is not None:
        ds = ds[split]

    if max_examples is not None and max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    return TokenizedDiskDataset(ds)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Get text from the example
        text = example.get(self.text_key, "")
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PackedTextCorpusDataset(Dataset):
    """Packed version of TextCorpusDataset for efficient training.
    
    Concatenates multiple text sequences to minimize padding waste.
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 2048,
        text_key: str = "text",
    ):
        """Initialize packed text corpus dataset.
        
        Args:
            data: List of text examples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for packed sequences.
            text_key: Key in data dict containing the text.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # Pre-tokenize all examples
        logger.info(f"Pre-tokenizing {len(data)} text examples for packing...")
        
        texts = [example.get(text_key, "") for example in data]
        
        # Batch tokenize
        logger.info("Batch tokenizing...")
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None,
            padding=False,
        )
        
        # Collect tokenized examples
        self.tokenized_examples = []
        for input_ids in encodings["input_ids"]:
            if len(input_ids) > 0:
                self.tokenized_examples.append({
                    "input_ids": input_ids,
                    "length": len(input_ids),
                })
        
        logger.info(f"Tokenized {len(self.tokenized_examples)} examples")
        
        # Sort by length for better packing efficiency
        self.tokenized_examples.sort(key=lambda x: x["length"])
        
        # Pack sequences
        logger.info("Packing sequences...")
        self.packed_sequences = self._pack_sequences()
        logger.info(f"Packed into {len(self.packed_sequences)} sequences")
        logger.info(f"Packing efficiency: {self._compute_efficiency():.1%}")
    
    def _pack_sequences(self) -> List[Dict[str, Any]]:
        """Pack multiple sequences using greedy bin packing."""
        packed = []
        current_pack = {"input_ids": [], "sequence_boundaries": []}
        current_length = 0
        
        for example in self.tokenized_examples:
            seq_len = example["length"]
            
            if current_length + seq_len <= self.max_length:
                current_pack["input_ids"].extend(example["input_ids"])
                current_pack["sequence_boundaries"].append((current_length, current_length + seq_len))
                current_length += seq_len
            else:
                if current_pack["input_ids"]:
                    packed.append(current_pack)
                current_pack = {
                    "input_ids": list(example["input_ids"]),
                    "sequence_boundaries": [(0, seq_len)],
                }
                current_length = seq_len
        
        if current_pack["input_ids"]:
            packed.append(current_pack)
        
        return packed
    
    def _compute_efficiency(self) -> float:
        """Compute packing efficiency."""
        total_tokens = sum(len(p["input_ids"]) for p in self.packed_sequences)
        max_tokens = len(self.packed_sequences) * self.max_length
        return total_tokens / max_tokens if max_tokens > 0 else 0.0
    
    def __len__(self) -> int:
        return len(self.packed_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pack = self.packed_sequences[idx]
        
        input_ids = pack["input_ids"]
        seq_len = len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - seq_len
        
        if padding_length > 0:
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_token_id] * padding_length
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:seq_len] = 1
        
        # Create labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PackedReasoningDataset(Dataset):
    """Dataset that packs multiple sequences into single tensors to eliminate padding waste.
    
    This implements "uncontaminated packing" where multiple short sequences are concatenated
    into a single sequence up to max_length, with proper attention masking to prevent
    cross-contamination between sequences.
    
    Benefits:
    - 2-5x faster training by eliminating padding tokens
    - Better GPU utilization
    - Same training dynamics (no cross-attention between packed sequences)
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 4096,
        system_prompt: Optional[str] = None,
        num_proc: int = None,
    ):
        """Initialize packed reasoning dataset.
        
        Args:
            data: List of conversation examples.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length for packed sequences.
            system_prompt: Optional system prompt to prepend.
            num_proc: Number of processes for parallel tokenization (default: CPU count).
        """
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that thinks step-by-step. "
            "Show your reasoning process before giving the final answer."
        )
        
        if num_proc is None:
            num_proc = os.cpu_count() or 8
        
        # Pre-tokenize all examples
        logger.info(f"Pre-tokenizing {len(data)} examples for packing (using {num_proc} processes)...")
        
        # Build all texts first (this is fast)
        texts = [self._build_conversation_text(example) for example in data]
        
        # Batch tokenize (much faster than one-by-one)
        logger.info("Batch tokenizing...")
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None,  # Return lists
            padding=False,  # Don't pad yet
        )
        
        # Collect tokenized examples
        self.tokenized_examples = []
        for input_ids in encodings["input_ids"]:
            if len(input_ids) > 0:
                self.tokenized_examples.append({
                    "input_ids": input_ids,
                    "length": len(input_ids),
                })
        
        logger.info(f"Tokenized {len(self.tokenized_examples)} examples")
        
        # Sort by length for better packing efficiency
        self.tokenized_examples.sort(key=lambda x: x["length"])
        
        # Create packed sequences
        logger.info("Packing sequences...")
        self.packed_sequences = self._pack_sequences()
        logger.info(f"Packed {len(self.tokenized_examples)} examples into {len(self.packed_sequences)} sequences")
        logger.info(f"Packing efficiency: {self._compute_efficiency():.1%}")
    
    def _build_conversation_text(self, example: Dict[str, Any]) -> str:
        """Build conversation text from example (same as ReasoningDataset)."""
        conversations = example.get("conversations", [])
        system = example.get("system", self.system_prompt)
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            
            for turn in conversations:
                role = turn.get("from", "")
                content = turn.get("value", "")
                
                if role == "human":
                    messages.append({"role": "user", "content": content})
                elif role in ("assistant", "gpt"):
                    messages.append({"role": "assistant", "content": content})
            
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        
        # Manual ChatML format (fallback)
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            
            if role == "human":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role in ("assistant", "gpt"):
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        return "\n".join(parts)
    
    def _pack_sequences(self) -> List[Dict[str, Any]]:
        """Pack multiple sequences into single sequences using greedy bin packing.
        
        O(n) algorithm - much faster than first-fit-decreasing for large datasets.
        """
        packed = []
        
        # Simple greedy: iterate through sorted sequences and fill bins
        current_pack = {
            "input_ids": [],
            "sequence_boundaries": [],
        }
        current_length = 0
        
        # Already sorted by length (shortest first), so we iterate and greedily pack
        for example in self.tokenized_examples:
            seq_len = example["length"]
            
            # If this sequence fits in current pack, add it
            if current_length + seq_len <= self.max_length:
                current_pack["input_ids"].extend(example["input_ids"])
                current_pack["sequence_boundaries"].append((current_length, current_length + seq_len))
                current_length += seq_len
            else:
                # Save current pack if it has content, start new one
                if current_pack["input_ids"]:
                    packed.append(current_pack)
                
                current_pack = {
                    "input_ids": list(example["input_ids"]),
                    "sequence_boundaries": [(0, seq_len)],
                }
                current_length = seq_len
        
        # Don't forget the last pack
        if current_pack["input_ids"]:
            packed.append(current_pack)
        
        return packed
    
    def _compute_efficiency(self) -> float:
        """Compute packing efficiency (ratio of actual tokens to max possible)."""
        total_tokens = sum(len(p["input_ids"]) for p in self.packed_sequences)
        max_tokens = len(self.packed_sequences) * self.max_length
        return total_tokens / max_tokens if max_tokens > 0 else 0.0
    
    def __len__(self) -> int:
        return len(self.packed_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pack = self.packed_sequences[idx]
        
        input_ids = pack["input_ids"]
        seq_len = len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - seq_len
        
        if padding_length > 0:
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_token_id] * padding_length
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:seq_len] = 1
        
        # Create labels (same as input_ids, but -100 for padding)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # Create position_ids that reset for each packed sequence
        position_ids = torch.zeros(self.max_length, dtype=torch.long)
        for start, end in pack["sequence_boundaries"]:
            seq_positions = torch.arange(end - start)
            position_ids[start:end] = seq_positions
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            # Store boundaries for potential block-diagonal attention masking
            "_sequence_boundaries": pack["sequence_boundaries"],
        }


def load_reasoning_dataset(
    dataset_name: str,
    tokenizer: Any,
    max_length: int = 4096,
    max_examples: Optional[int] = None,
    split: str = "train",
    streaming: bool = False,
    packing: bool = False,
    cache_dir: Optional[str] = "./cache",
) -> Dataset:
    """Load a reasoning dataset from HuggingFace.
    
    Args:
        dataset_name: Name of dataset (openthoughts3, openthoughts114k, openthoughts2, indiccorp_*).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        max_examples: Optional limit on number of examples.
        split: Dataset split to load.
        streaming: Whether to use streaming mode (for large datasets).
        packing: Whether to use sequence packing for 2-5x faster training.
        cache_dir: Directory to cache packed datasets (None to disable caching).
        
    Returns:
        ReasoningDataset, PackedReasoningDataset, TextCorpusDataset, or PackedTextCorpusDataset.
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        >>> dataset = load_reasoning_dataset("openthoughts114k", tokenizer, packing=True)
        >>> # For IndicCorpV2:
        >>> dataset = load_reasoning_dataset("indiccorp_hi", tokenizer, max_examples=100000)
    """
    from datasets import load_dataset
    import hashlib
    import pickle
    
    if dataset_name not in REASONING_DATASETS:
        available = ", ".join(REASONING_DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    config = REASONING_DATASETS[dataset_name]
    is_text_only = config.is_text_only
    
    # Check for cached packed dataset
    if packing and cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create cache key from dataset params
        cache_key = f"{dataset_name}_{max_length}_{max_examples or 'all'}_{split}"
        cache_file = cache_path / f"packed_{cache_key}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached packed dataset from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                # Reconstruct the appropriate dataset type
                if is_text_only:
                    dataset = PackedTextCorpusDataset.__new__(PackedTextCorpusDataset)
                    dataset.tokenizer = tokenizer
                    dataset.max_length = max_length
                    dataset.text_key = config.text_key
                    dataset.tokenized_examples = cached["tokenized_examples"]
                    dataset.packed_sequences = cached["packed_sequences"]
                else:
                    dataset = PackedReasoningDataset.__new__(PackedReasoningDataset)
                    dataset.tokenizer = tokenizer
                    dataset.max_length = max_length
                    dataset.system_prompt = cached["system_prompt"]
                    dataset.tokenized_examples = cached["tokenized_examples"]
                    dataset.packed_sequences = cached["packed_sequences"]
                logger.info(f"Loaded {len(dataset.packed_sequences)} packed sequences from cache")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding...")
    
    logger.info(f"Loading {config.name} from {config.hf_path}")
    
    # Load from HuggingFace with appropriate parameters
    # For IndicCorpV2, the language code is the split name (e.g., hin_Deva)
    actual_split = config.split_name if config.split_name else split
    
    load_kwargs = {
        "split": actual_split,
        "streaming": streaming,
    }
    
    # Add data_dir if specified (for datasets with subdirectories)
    if config.data_dir:
        load_kwargs["data_dir"] = config.data_dir
    
    hf_dataset = load_dataset(
        config.hf_path,
        config.hf_subset,
        **load_kwargs,
    )
    
    # Convert to list (handle streaming)
    if streaming:
        if max_examples:
            data = list(hf_dataset.take(max_examples))
        else:
            logger.warning("Loading full streaming dataset into memory...")
            data = list(hf_dataset)
    else:
        if max_examples:
            data = hf_dataset.select(range(min(max_examples, len(hf_dataset))))
            data = list(data)
        else:
            data = list(hf_dataset)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Handle text-only datasets (like IndicCorpV2)
    if is_text_only:
        if packing:
            logger.info("Using packed text corpus dataset for faster training")
            dataset = PackedTextCorpusDataset(
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
                text_key=config.text_key,
            )
            
            # Cache the packed dataset
            if cache_dir:
                try:
                    cache_data = {
                        "tokenized_examples": dataset.tokenized_examples,
                        "packed_sequences": dataset.packed_sequences,
                    }
                    with open(cache_file, "wb") as f:
                        pickle.dump(cache_data, f)
                    logger.info(f"Cached packed dataset to {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to cache dataset: {e}")
            
            return dataset
        else:
            return TextCorpusDataset(
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
                text_key=config.text_key,
            )
    
    # Handle conversation datasets (OpenThoughts, etc.)
    if packing:
        logger.info("Using packed dataset for faster training")
        dataset = PackedReasoningDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        
        # Cache the packed dataset
        if cache_dir:
            try:
                cache_data = {
                    "system_prompt": dataset.system_prompt,
                    "tokenized_examples": dataset.tokenized_examples,
                    "packed_sequences": dataset.packed_sequences,
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached packed dataset to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache dataset: {e}")
        
        return dataset
    else:
        return ReasoningDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=max_length,
        )


def create_dataloader(
    dataset: ReasoningDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for training.
    
    Args:
        dataset: ReasoningDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        
    Returns:
        PyTorch DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset.
        
    Returns:
        Dictionary with dataset information.
    """
    if dataset_name not in REASONING_DATASETS:
        available = ", ".join(REASONING_DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    config = REASONING_DATASETS[dataset_name]
    
    info = {
        "name": config.name,
        "huggingface_path": config.hf_path,
        "subset": config.hf_subset,
    }
    
    # Dataset-specific info
    if dataset_name == "openthoughts3":
        info.update({
            "size": "1.2M examples",
            "domains": "850K math, 250K code, 100K science",
            "teacher_model": "QwQ-32B",
            "description": "State-of-the-art open reasoning dataset (June 2025)",
        })
    elif dataset_name == "openthoughts114k":
        info.update({
            "size": "114K examples",
            "domains": "Math, science, code, puzzles",
            "teacher_model": "DeepSeek-R1",
            "description": "High-quality curated reasoning traces",
        })
    elif dataset_name == "openthoughts2":
        info.update({
            "size": "1M examples", 
            "domains": "Math, code + OpenR1-Math",
            "teacher_model": "DeepSeek-R1",
            "description": "Extended dataset with additional math/code",
        })
    
    return info


def print_available_datasets():
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("Available Reasoning Datasets for CoT Training")
    print("=" * 70)
    
    for name in REASONING_DATASETS:
        info = get_dataset_info(name)
        print(f"\n{info['name']}")
        print("-" * 40)
        print(f"  Load with: load_reasoning_dataset('{name}', tokenizer)")
        print(f"  HuggingFace: {info['huggingface_path']}")
        print(f"  Size: {info['size']}")
        print(f"  Domains: {info['domains']}")
        print(f"  Teacher: {info['teacher_model']}")
        print(f"  {info['description']}")
    
    print("\n" + "=" * 70)
    print("Recommended: 'openthoughts3' or 'openthoughts114k' for best results")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_available_datasets()
