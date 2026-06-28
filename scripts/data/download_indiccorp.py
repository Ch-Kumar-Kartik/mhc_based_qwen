"""
Download IndicCorpV2 dataset (optionally tokenize with Qwen3 tokenizer).

This script downloads multilingual Indic text from HuggingFace and optionally
tokenizes it for use with Qwen3-based mHC models.

Dataset: https://huggingface.co/datasets/ai4bharat/IndicCorpV2

Supported languages (24 splits):
    asm_Beng  - Assamese (Bengali script)
    ben_Beng  - Bengali
    brx_Deva  - Bodo (Devanagari script)
    doi_Deva  - Dogri (Devanagari script)
    gom_Deva  - Konkani (Devanagari script)
    guj_Gujr  - Gujarati
    hin_Deva  - Hindi (Devanagari script)
    kan_Knda  - Kannada
    kas_Arab  - Kashmiri (Arabic script)
    kas_Deva  - Kashmiri (Devanagari script)
    mai_Deva  - Maithili (Devanagari script)
    mal_Mlym  - Malayalam
    mar_Deva  - Marathi (Devanagari script)
    mni_Mtei  - Manipuri (Meitei script)
    npi_Deva  - Nepali (Devanagari script)
    ory_Orya  - Odia
    pan_Guru  - Punjabi (Gurmukhi script)
    san_Deva  - Sanskrit (Devanagari script)
    snd_Deva  - Sindhi (Devanagari script)
    tam_Taml  - Tamil
    tel_Telu  - Telugu
    urd_Arab  - Urdu (Arabic script)
    khasi     - Khasi
    santhali  - Santhali

Usage:
    # Download raw text (no tokenization) - works without PyTorch
    python -m scripts.data.download_indiccorp --num_samples 10000 --raw
    
    # Download specific languages
    python -m scripts.data.download_indiccorp --languages hin_Deva ben_Beng tam_Taml --num_samples 10000 --raw
    
    # Download and tokenize (requires PyTorch + transformers)
    python -m scripts.data.download_indiccorp --languages tel_Telu --num_samples 50000 --max_length 2048
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, HfFileSystem
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def _get_hf_token() -> Optional[str]:
    """Return Hugging Face token from environment, if present.

    We intentionally do not accept tokens via CLI args to avoid accidentally
    leaking them via shell history or process lists.
    """
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token
    return None

# Optional: tokenization requires PyTorch + transformers
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "indiccorp"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_TOKENIZE_BATCH_SIZE = 2000

# All supported languages/splits in IndicCorpV2 (from HuggingFace API)
SUPPORTED_LANGUAGES = [
    "asm_Beng",  # Assamese
    "ben_Beng",  # Bengali
    "brx_Deva",  # Bodo
    "doi_Deva",  # Dogri
    "gom_Deva",  # Konkani
    "guj_Gujr",  # Gujarati
    "hin_Deva",  # Hindi
    "kan_Knda",  # Kannada
    "kas_Arab",  # Kashmiri (Arabic)
    "kas_Deva",  # Kashmiri (Devanagari)
    "mai_Deva",  # Maithili
    "mal_Mlym",  # Malayalam
    "mar_Deva",  # Marathi
    "mni_Mtei",  # Manipuri (Meitei)
    "npi_Deva",  # Nepali
    "ory_Orya",  # Odia
    "pan_Guru",  # Punjabi
    "san_Deva",  # Sanskrit
    "snd_Deva",  # Sindhi (Devanagari)
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
    "urd_Arab",  # Urdu
    "khasi",     # Khasi
    "santhali",  # Santhali
]

# Mapping from IndicCorpV2 split codes to files actually present in the repo.
# Repo layout (as of commit 2d7285e): data/*.txt (large monolingual corpora)
LANG_TO_REPO_FILES = {
    "asm_Beng": ["data/as.txt"],
    "ben_Beng": ["data/bn.txt"],
    "brx_Deva": ["data/bd.txt"],
    "doi_Deva": ["data/dg.txt"],
    "gom_Deva": ["data/gom.txt"],
    "guj_Gujr": ["data/gu.txt"],
    "hin_Deva": ["data/hi-1.txt", "data/hi-2.txt", "data/hi-3.txt"],
    "kan_Knda": ["data/kn.txt"],
    "kas_Arab": ["data/ks.txt"],
    "kas_Deva": ["data/ks.txt"],
    "mai_Deva": ["data/mai.txt"],
    "mal_Mlym": ["data/ml.txt"],
    "mar_Deva": ["data/mr.txt"],
    "mni_Mtei": ["data/mni.txt"],
    "npi_Deva": ["data/ne.txt"],
    "ory_Orya": ["data/or.txt"],
    "pan_Guru": ["data/pa.txt"],
    "san_Deva": ["data/sa.txt"],
    "snd_Deva": ["data/sd.txt"],
    "tam_Taml": ["data/ta.txt"],
    "tel_Telu": ["data/te.txt"],
    "urd_Arab": ["data/ur.txt"],
    "khasi": ["data/kha.txt"],
    "santhali": ["data/sat.txt"],
}

# Language code to full name mapping
LANGUAGE_NAMES = {
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "brx_Deva": "Bodo",
    "doi_Deva": "Dogri",
    "gom_Deva": "Konkani",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri (Arabic)",
    "kas_Deva": "Kashmiri (Devanagari)",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Mtei": "Manipuri (Meitei)",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "snd_Deva": "Sindhi (Devanagari)",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
    "khasi": "Khasi",
    "santhali": "Santhali",
}


def list_languages():
    """Print all supported languages."""
    print("\nSupported languages in IndicCorpV2:")
    print("=" * 50)
    for code in SUPPORTED_LANGUAGES:
        print(f"  {code:<12} - {LANGUAGE_NAMES[code]}")
    print()


def _pack_and_chunk_token_ids(
    tokenized_input_ids: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
    mask_pad_labels: bool = False,
) -> dict:
    """Pack multiple documents into contiguous token stream and chunk into fixed blocks.

    This is typically faster than padding every document to max_length and also
    produces denser training blocks.

    Returns a dict with input_ids/attention_mask/labels.
    """
    input_ids_blocks: List[List[int]] = []
    attention_mask_blocks: List[List[int]] = []
    labels_blocks: List[List[int]] = []

    pool: List[int] = []
    for ids in tokenized_input_ids:
        if not ids:
            continue
        pool.extend(ids)
        if eos_token_id is not None:
            pool.append(eos_token_id)

        while len(pool) >= max_length:
            block = pool[:max_length]
            pool = pool[max_length:]
            input_ids_blocks.append(block)
            attention_mask_blocks.append([1] * max_length)
            labels_blocks.append(block.copy())

    if pool:
        block = pool[:max_length]
        attn = [1] * len(block)
        pad_len = 0
        if len(block) < max_length:
            pad_len = max_length - len(block)
            block = block + [pad_token_id] * pad_len
            attn = attn + [0] * pad_len

        labels = block.copy()
        if mask_pad_labels and pad_len > 0:
            for i in range(max_length - pad_len, max_length):
                labels[i] = -100

        input_ids_blocks.append(block)
        attention_mask_blocks.append(attn)
        labels_blocks.append(labels)

    return {
        "input_ids": input_ids_blocks,
        "attention_mask": attention_mask_blocks,
        "labels": labels_blocks,
    }


def download_language(
    language: str,
    num_samples: int,
    tokenizer=None,
    max_length: int = DEFAULT_MAX_LENGTH,
    streaming: bool = True,
    raw: bool = False,
    shard_size: int = 10000,
    tokenize_batch_size: int = DEFAULT_TOKENIZE_BATCH_SIZE,
    pack_sequences: bool = False,
    mask_pad_labels: bool = False,
    repo_id: str = "ai4bharat/IndicCorpV2",
    hf_cache_dir: Optional[str] = None,
    download_files: bool = False,
) -> Optional[dict]:
    """Download and optionally tokenize a single language subset.
    
    Args:
        language: Language code (e.g., 'hin_Deva', 'tel_Telu')
        num_samples: Number of samples to download
        tokenizer: HuggingFace tokenizer (None if raw=True)
        max_length: Maximum sequence length
        streaming: Whether to use streaming mode
        raw: If True, return raw text without tokenization
        
    Returns:
        Dictionary with text or tokenized data, plus language info
    """
    print(f"\n  Downloading {language} ({LANGUAGE_NAMES.get(language, 'Unknown')})...")
    
    try:
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is required to download IndicCorpV2 raw files. Install with: pip install huggingface_hub"
            )

        hf_token = _get_hf_token()

        repo_files = LANG_TO_REPO_FILES.get(language)
        if not repo_files:
            raise RuntimeError(
                f"No repo file mapping found for language '{language}'. Update LANG_TO_REPO_FILES in this script."
            )

        fs = None
        if HF_HUB_AVAILABLE and not download_files:
            try:
                fs = HfFileSystem(token=hf_token)
            except TypeError:
                # Older/newer versions may not accept token kwarg.
                fs = HfFileSystem()

        # Iterate lines from one or more repo .txt files.
        # By default, stream remotely via HfFileSystem to avoid downloading multi-GB files.
        # If download_files=True, download to local cache first.
        def iter_lines():
            for repo_file in repo_files:
                if download_files:
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=repo_file,
                        repo_type="dataset",
                        cache_dir=hf_cache_dir,
                        token=hf_token,
                    )
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                        for line in handle:
                            text_line = line.strip()
                            if text_line:
                                yield text_line
                else:
                    # Path format for HfFileSystem: datasets/<repo_id>/<path>
                    hf_path = f"datasets/{repo_id}/{repo_file}"
                    with fs.open(hf_path, mode="r", encoding="utf-8", errors="ignore") as handle:
                        for line in handle:
                            text_line = line.strip()
                            if text_line:
                                yield text_line

        # If we are downloading all samples, don't try to buffer everything in memory.
        unlimited = num_samples <= 0
        target_total = None if unlimited else num_samples

        if raw or tokenizer is None:
            # Raw: shard to disk by default when unlimited, otherwise return in-memory for small runs
            buffer_text: List[str] = []
            language_col: List[str] = []
            shard_index = 0
            total_written = 0

            def flush_raw_shard():
                nonlocal shard_index, total_written
                if not buffer_text:
                    return
                shard_ds = Dataset.from_dict({"text": buffer_text, "language": language_col})
                shard_dir = Path(DEFAULT_OUTPUT_DIR) / "_tmp"  # overridden by caller; safe default
                shard_dir = Path(os.environ.get("INDICCORP_OUTPUT_DIR", str(shard_dir)))
                out_dir = shard_dir / language / "raw_shards" / f"shard-{shard_index:05d}"
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                shard_ds.save_to_disk(str(out_dir))
                total_written += len(buffer_text)
                shard_index += 1
                buffer_text.clear()
                language_col.clear()

            for i, text in enumerate(tqdm(iter_lines(), total=target_total, desc=f"    {language}", leave=False)):
                if not unlimited and i >= num_samples:
                    break
                buffer_text.append(text)
                language_col.append(language)
                if unlimited and len(buffer_text) >= shard_size:
                    flush_raw_shard()

            if unlimited:
                flush_raw_shard()
                print(f"    Saved raw shards: {total_written:,} samples")
                return {"raw_shards": True, "language": language}

            # Finite: return in-memory data to let the caller build a combined dataset
            print(f"    Collected {len(buffer_text):,} samples")
            return {"text": buffer_text, "language": language_col}

        # Tokenized: if unlimited, shard to disk; if finite, collect then tokenize (faster)
        if not unlimited:
            samples: List[str] = []
            for i, text in enumerate(tqdm(iter_lines(), total=target_total, desc=f"    {language}", leave=False)):
                if i >= num_samples:
                    break
                samples.append(text)

            if not samples:
                print(f"    Warning: No samples collected for {language}")
                return None

            print(f"    Collected {len(samples):,} samples")

            # Fast path: pack multiple documents then chunk into fixed-length blocks.
            # Avoids per-example padding overhead and yields denser blocks.
            if pack_sequences:
                eos_id = getattr(tokenizer, "eos_token_id", None)
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = eos_id if eos_id is not None else 0

                tokenized_ids: List[List[int]] = []
                bs = max(1, int(tokenize_batch_size))
                for j in range(0, len(samples), bs):
                    batch = samples[j : j + bs]
                    encoded = tokenizer(
                        batch,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                    )
                    tokenized_ids.extend(encoded["input_ids"])

                packed = _pack_and_chunk_token_ids(
                    tokenized_input_ids=tokenized_ids,
                    max_length=max_length,
                    pad_token_id=int(pad_id),
                    eos_token_id=int(eos_id) if eos_id is not None else None,
                    mask_pad_labels=mask_pad_labels,
                )

                n = len(packed["input_ids"])
                return {
                    "input_ids": packed["input_ids"],
                    "attention_mask": packed["attention_mask"],
                    "labels": packed["labels"] if mask_pad_labels else packed["input_ids"],
                    "language": [language] * n,
                }

            # Default path: straightforward padding to max_length
            all_input_ids = []
            all_attention_masks = []
            bs = max(1, int(tokenize_batch_size))
            for j in range(0, len(samples), bs):
                batch = samples[j : j + bs]
                encoded = tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )
                all_input_ids.extend(encoded["input_ids"])
                all_attention_masks.extend(encoded["attention_mask"])

            labels = all_input_ids
            if mask_pad_labels:
                pad_id = tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                masked = []
                for ids, attn in zip(all_input_ids, all_attention_masks):
                    # trust attention_mask over token-id checks
                    lab = [tok if m == 1 else -100 for tok, m in zip(ids, attn)]
                    masked.append(lab)
                labels = masked

            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
                "labels": labels,
                "language": [language] * len(all_input_ids),
            }

        # Unlimited tokenization: shard to disk
        token_buffer: List[str] = []
        shard_index = 0
        total_written = 0

        def flush_token_shard():
            nonlocal shard_index, total_written
            if not token_buffer:
                return
            encoded = tokenizer(
                token_buffer,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )
            shard_ds = Dataset.from_dict(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["input_ids"],
                    "language": [language] * len(encoded["input_ids"]),
                }
            )
            shard_dir = Path(DEFAULT_OUTPUT_DIR) / "_tmp"  # overridden by caller; safe default
            shard_dir = Path(os.environ.get("INDICCORP_OUTPUT_DIR", str(shard_dir)))
            out_dir = shard_dir / language / "tokenized_shards" / f"shard-{shard_index:05d}"
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            shard_ds.save_to_disk(str(out_dir))
            total_written += len(token_buffer)
            shard_index += 1
            token_buffer.clear()

        for text in tqdm(iter_lines(), total=None, desc=f"    {language}", leave=False):
            token_buffer.append(text)
            if len(token_buffer) >= shard_size:
                flush_token_shard()

        flush_token_shard()
        print(f"    Saved tokenized shards: {total_written:,} samples")
        return {"tokenized_shards": True, "language": language}

    except Exception as e:
        print(f"    Error downloading {language}: {e}")
        return None


def download_indiccorp(
    languages: Optional[List[str]] = None,
    num_samples_per_lang: int = 10000,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    streaming: bool = True,
    save_per_language: bool = False,
    raw: bool = False,
    shard_size: int = 10000,
    tokenize_batch_size: int = DEFAULT_TOKENIZE_BATCH_SIZE,
    pack_sequences: bool = False,
    mask_pad_labels: bool = False,
    trust_remote_code: bool = True,
    use_fast_tokenizer: bool = True,
    hf_cache_dir: Optional[str] = None,
    download_files: bool = False,
):
    """Download IndicCorpV2 and optionally tokenize with specified tokenizer.
    
    Args:
        languages: List of language codes to download (None for all)
        num_samples_per_lang: Number of samples per language
        output_dir: Directory to save tokenized data
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length for tokenization
        streaming: Whether to use streaming mode for downloading
        save_per_language: Whether to save each language separately
        raw: If True, save raw text without tokenization
    """
    # Validate languages
    if languages is None:
        languages = SUPPORTED_LANGUAGES
    else:
        invalid = [l for l in languages if l not in SUPPORTED_LANGUAGES]
        if invalid:
            print(f"Error: Invalid language codes: {invalid}")
            list_languages()
            return None
    
    # Check tokenizer availability
    tokenizer = None
    if not raw:
        if not TOKENIZER_AVAILABLE:
            print("Warning: PyTorch/transformers not available. Switching to raw mode.")
            print("         Install with: pip install torch transformers")
            print()
            raw = True
        else:
            print("Loading tokenizer...")
            try:
                # Enable Rust-tokenizers internal thread pool when available.
                # (Only applies to fast tokenizers; safe no-op otherwise.)
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"  Vocab size: {tokenizer.vocab_size:,}")
                print(f"  Pad token:  {tokenizer.pad_token}")
                if getattr(tokenizer, "is_fast", False):
                    print("  Tokenizer:  fast")
                else:
                    print("  Tokenizer:  python (slow)")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
                print("         Switching to raw mode.")
                raw = True
    
    print(f"\nIndicCorpV2 Download")
    print(f"=" * 60)
    print(f"Dataset:             ai4bharat/IndicCorpV2")
    print(f"Languages:           {len(languages)}")
    samples_label = "ALL" if num_samples_per_lang <= 0 else f"{num_samples_per_lang:,}"
    print(f"Samples per lang:    {samples_label}")
    print(f"Output dir:          {output_dir}")
    print(f"Streaming mode:      {streaming}")
    print(f"Raw text mode:       {raw}")
    if not raw:
        print(f"Model tokenizer:     {model_name}")
        print(f"Max length:          {max_length}")
        print(f"Tokenize batch size: {tokenize_batch_size}")
        print(f"Pack sequences:      {pack_sequences}")
        print(f"Mask pad labels:     {mask_pad_labels}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make output dir available to shard writers inside download_language
    os.environ["INDICCORP_OUTPUT_DIR"] = str(Path(output_dir))

    # Download each language
    print(f"Downloading {len(languages)} language(s)...")
    
    if raw:
        all_data = {
            "text": [],
            "language": [],
        }
    else:
        all_data = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "language": [],
        }
    
    successful_languages = []
    failed_languages = []
    wrote_shards = False
    
    for lang in languages:
        result = download_language(
            language=lang,
            num_samples=num_samples_per_lang,
            tokenizer=tokenizer,
            max_length=max_length,
            streaming=streaming,
            raw=raw,
            shard_size=shard_size,
            tokenize_batch_size=tokenize_batch_size,
            pack_sequences=pack_sequences,
            mask_pad_labels=mask_pad_labels,
            hf_cache_dir=hf_cache_dir,
            download_files=download_files,
        )
        
        if result is not None:
            successful_languages.append(lang)
            
            # If we created shards (unlimited mode), skip combined aggregation
            if result.get("raw_shards") or result.get("tokenized_shards"):
                wrote_shards = True
                continue

            # Add to combined data (finite mode)
            for key in all_data:
                if key in result:
                    all_data[key].extend(result[key])
            
            # Optionally save per-language
            if save_per_language:
                lang_dataset = Dataset.from_dict(result)
                lang_path = Path(output_dir) / lang
                lang_dataset.save_to_disk(str(lang_path))
                print(f"    Saved to: {lang_path}")
        else:
            failed_languages.append(lang)
    
    print()
    
    # Create combined dataset (only if we actually aggregated in-memory data)
    has_data = bool(all_data.get("text") or all_data.get("input_ids"))
    if has_data:
        print("Creating combined dataset...")
        combined_dataset = Dataset.from_dict(all_data)
        
        # Save combined dataset
        combined_path = Path(output_dir) / "combined"
        combined_dataset.save_to_disk(str(combined_path))
        
        print(f"  Saved combined dataset to: {combined_path}")
        print()
        
        # Print statistics
        print("Dataset Statistics:")
        print(f"=" * 60)
        print(f"  Successful languages: {len(successful_languages)}")
        print(f"  Failed languages:     {len(failed_languages)}")
        print(f"  Total samples:        {len(combined_dataset):,}")
        if not raw:
            print(f"  Sequence length:      {max_length}")
            print(f"  Total tokens:         {len(combined_dataset) * max_length:,}")
        print()
        
        # Language distribution
        print("Samples per language:")
        from collections import Counter
        lang_counts = Counter(all_data["language"])
        for lang, count in sorted(lang_counts.items()):
            print(f"    {lang:<12}: {count:>8,} ({LANGUAGE_NAMES.get(lang, 'Unknown')})")
        print()
        
        if failed_languages:
            print("Failed languages:")
            for lang in failed_languages:
                print(f"    {lang} ({LANGUAGE_NAMES.get(lang, 'Unknown')})")
            print()
        
        # Print sample
        print("Sample data:")
        sample = combined_dataset[0]
        print(f"  Language: {sample['language']}")
        if raw:
            text = sample['text']
            print(f"  Text preview: {text[:300]}...")
        else:
            sample_tokens = sample["input_ids"][:50]
            decoded = tokenizer.decode(sample_tokens)
            print(f"  First 50 tokens: {sample_tokens}")
            print(f"  Decoded: {decoded[:200]}...")
        print()
        
        print("Done!")
        print()
        print("To use with training:")
        print(f"  python -m scripts.train_multilingual --data {combined_path} ...")
        
        return combined_path
    if wrote_shards and successful_languages:
        print("Done! (sharded)")
        print()
        print("Unlimited download writes shards per language to:")
        if raw:
            print(f"  {Path(output_dir) / '<language>' / 'raw_shards' / 'shard-00000' }")
        else:
            print(f"  {Path(output_dir) / '<language>' / 'tokenized_shards' / 'shard-00000' }")
        print()
        print("Tip: start with one language first if this is your first run.")
        return Path(output_dir)

    print("Error: No data was successfully downloaded!")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download IndicCorpV2 dataset (optionally tokenize with Qwen3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download ALL samples as raw text (works without PyTorch)
    python download_indiccorp.py --raw
    
    # Download specific languages (raw)
    python download_indiccorp.py --languages hin_Deva ben_Beng tam_Taml tel_Telu --raw
    
    # Download and tokenize (requires PyTorch + transformers)
    python download_indiccorp.py --languages hin_Deva --num_samples 100000
    
    # List all supported languages
    python download_indiccorp.py --list_languages
        """,
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Language codes to download (default: all). Use --list_languages to see options.",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples per language (default: -1 for ALL). Use -1 for all samples.",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for data",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name for tokenizer (default: {DEFAULT_MODEL})",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_LENGTH})",
    )

    parser.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=DEFAULT_TOKENIZE_BATCH_SIZE,
        help=f"Batch size for tokenization (default: {DEFAULT_TOKENIZE_BATCH_SIZE}). Larger is usually faster until you hit RAM limits.",
    )

    parser.add_argument(
        "--pack",
        action="store_true",
        help="Pack many lines into a token stream and chunk into fixed-length blocks (often faster and denser than padding each line). Only applies when tokenizing with finite --num_samples.",
    )

    parser.add_argument(
        "--mask_pad_labels",
        action="store_true",
        help="Set labels=-100 on padding positions (recommended for training). Default keeps current behavior (labels=input_ids).",
    )

    parser.add_argument(
        "--no_fast_tokenizer",
        action="store_true",
        help="Force python (slow) tokenizer backend (not recommended).",
    )

    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trust_remote_code when loading tokenizer (faster/safer if model supports it).",
    )
    
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode (loads entire dataset into memory)",
    )
    
    parser.add_argument(
        "--save_per_language",
        action="store_true",
        help="Save each language as a separate dataset",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Shard size for unlimited downloads (-1). For finite downloads this is ignored (default: 10000).",
    )

    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache dir for downloaded raw .txt files.",
    )

    parser.add_argument(
        "--download_files",
        action="store_true",
        help="Download the raw .txt files locally before reading (VERY large; default is remote streaming).",
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Download raw text without tokenization (no PyTorch needed)",
    )
    
    parser.add_argument(
        "--list_languages",
        action="store_true",
        help="List all supported languages and exit",
    )
    
    args = parser.parse_args()
    
    if args.list_languages:
        list_languages()
        return
    
    download_indiccorp(
        languages=args.languages,
        num_samples_per_lang=args.num_samples,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        streaming=not args.no_streaming,
        save_per_language=args.save_per_language,
        raw=args.raw,
        shard_size=args.shard_size,
        tokenize_batch_size=args.tokenize_batch_size,
        pack_sequences=args.pack,
        mask_pad_labels=args.mask_pad_labels,
        trust_remote_code=not args.no_trust_remote_code,
        use_fast_tokenizer=not args.no_fast_tokenizer,
        hf_cache_dir=args.hf_cache_dir,
        download_files=args.download_files,
    )


if __name__ == "__main__":
    main()
