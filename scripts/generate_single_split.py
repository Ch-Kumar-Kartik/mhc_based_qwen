#!/usr/bin/env python3
"""Generate a single tokenized Arrow split from a dataset split.

Example:
    python scripts/generate_single_split.py \
        --dataset ai4bharat/sangraha \
        --dataset-config verified \
        --split train \
    --cache-dir E:/mhc_based_qwen-2/hf_cache \
    --tokenizer-model Qwen/Qwen3-0.6B \
        --output-dir E:/mhc_based_qwen-2/data/sangraha_arrow/verified \
    --max-length 2048 --num-proc 1
"""
import argparse
from pathlib import Path
import sys

# Support both `python -m scripts.generate_single_split` and direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset, DatasetDict, Dataset

from scripts.data.prepare_dataset import tokenize_and_pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--dataset-config', default='verified', help="Dataset config/subset name (sangraha uses 'verified')")
    p.add_argument('--split', required=True, help='Split name (e.g. train/validation/test)')
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--tokenizer-model', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--num-proc', type=int, default=1)
    p.add_argument('--hf-login', action='store_true', help='Run huggingface_hub.login() before downloading')
    p.add_argument('--raw-file', type=str, default=None, help='Optional path to raw text file for this split')
    args = p.parse_args()

    # Optional HF login (if provided)
    if args.hf_login:
        try:
            from huggingface_hub import login as hf_login
            hf_login()
        except Exception:
            pass

    # If user provided a raw file, or we can auto-find it in the cache, load using the 'text' loader
    raw_path = None
    if args.raw_file:
        raw_path = Path(args.raw_file)
        if not raw_path.exists():
            raise FileNotFoundError(f"raw file not found: {raw_path}")
    elif args.cache_dir:
        # try to find a file in the cache that matches the split name
        cache_root = Path(args.cache_dir)
        hub_root = cache_root / 'hub'
        candidates = []
        if hub_root.exists():
            for p in hub_root.rglob('*'):
                if p.is_file() and args.split.lower() in p.name.lower() and p.suffix in ('.txt', '.dat', '.text'):
                    candidates.append(p)
        # fallback: scan cache root
        if not candidates:
            for p in cache_root.rglob('*'):
                if p.is_file() and args.split.lower() in p.name.lower() and p.suffix in ('.txt', '.dat', '.text'):
                    candidates.append(p)

        if candidates:
            # pick the largest candidate (likely the full raw file)
            raw_path = max(candidates, key=lambda p: p.stat().st_size)
            print(f"Auto-detected raw file for {args.config}: {raw_path}")

    if raw_path is not None:
        ds = load_dataset('text', data_files={'train': str(raw_path)}, cache_dir=args.cache_dir)
    else:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, cache_dir=args.cache_dir)

    # Normalize to a DatasetDict with a single 'train' split
    if isinstance(ds, DatasetDict):
        if 'train' in ds:
            dd = DatasetDict({'train': ds['train']})
        else:
            first = list(ds.keys())[0]
            dd = DatasetDict({'train': ds[first]})
    elif isinstance(ds, Dataset):
        dd = DatasetDict({'train': ds})
    else:
        raise RuntimeError('Unexpected dataset type')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenize_and_pack(
        dd,
        tokenizer_name=args.tokenizer_model,
        text_columns=['text'],
        output_dir=out_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )


if __name__ == '__main__':
    main()
