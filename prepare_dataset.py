#!/usr/bin/env python3
"""
Prepare and tokenize the ai4bharat/IndicCorpV2 dataset and save Arrow tokenized splits to disk.

Requirements implemented:
- Deterministic 80/10/10 split
- Tokenize once with `datasets.map(..., batched=True, num_proc=...)`
- Remove raw text columns after tokenization
- Pack tokens into fixed-length blocks (sequence packing)
- Save each split with `save_to_disk()` (Arrow, memory-mapped)

Usage example:
  python prepare_dataset.py --dataset ai4bharat/IndicCorpV2 \
      --tokenizer-model Qwen/Qwen3-0.6B \
      --output-dir C:/datasets/indiccorp_arrow --max-length 2048 --num-proc 16
"""
import argparse
import json
import platform
import hashlib
from itertools import chain
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset, get_dataset_config_names, DatasetDict, Dataset
from transformers import AutoTokenizer


_TOKENIZER = None
_TOKENIZER_NAME = None


def _get_tokenizer(tokenizer_name: str):
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is None or _TOKENIZER_NAME != tokenizer_name:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _TOKENIZER = tok
        _TOKENIZER_NAME = tokenizer_name
    return _TOKENIZER


def _tokenize_batch(examples, *, tokenizer_name: str, text_columns: List[str]):
    """Pickle-safe tokenization function for datasets.map(num_proc>1) on Windows."""
    tokenizer = _get_tokenizer(tokenizer_name)

    n = 0
    for col in text_columns:
        if col in examples and examples[col] is not None:
            try:
                n = max(n, len(examples[col]))
            except Exception:
                pass
    if n == 0:
        return {}

    per_example: List[str] = []
    for i in range(n):
        parts: List[str] = []
        for col in text_columns:
            if col in examples and examples[col] is not None and i < len(examples[col]):
                val = examples[col][i]
                if val is None:
                    continue
                parts.append(str(val))
        per_example.append('\n'.join(parts))

    return tokenizer(per_example, return_attention_mask=True)


def _pack_batch(examples, *, max_length: int):
    # Concatenate across the batch then chunk into fixed blocks.
    # Using itertools.chain avoids O(n^2) behavior of sum(list_of_lists, []).
    all_input_ids = list(chain.from_iterable(examples['input_ids']))

    all_attention = None
    if 'attention_mask' in examples and examples['attention_mask'] is not None:
        all_attention = list(chain.from_iterable(examples['attention_mask']))

    total_length = (len(all_input_ids) // max_length) * max_length
    result = {
        'input_ids': [all_input_ids[i : i + max_length] for i in range(0, total_length, max_length)]
    }
    if all_attention is not None:
        result['attention_mask'] = [all_attention[i : i + max_length] for i in range(0, total_length, max_length)]
    result['labels'] = result['input_ids'].copy()
    return result


def _stable_fingerprint(prefix: str, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha1(data).hexdigest()}"


def deterministic_split(ds, seed: int = 42):
    # ds is a Dataset (single split) - we want 80/10/10
    first = ds.train_test_split(test_size=0.2, seed=seed)
    # first['train'] is 80%, first['test'] is 20%
    second = first['test'].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        'train': first['train'],
        'validation': second['train'],
        'test': second['test'],
    })


def tokenize_and_pack(
    ds_dict: DatasetDict,
    tokenizer_name: str,
    text_columns: List[str],
    output_dir: Path,
    max_length: int = 2048,
    num_proc: int = 16,
    overwrite_output: bool = False,
):
    # IMPORTANT (Windows/multiprocess): don't prime the global tokenizer cache in the parent.
    # `datasets.map(num_proc>1)` pickles the function with dill and may try to serialize globals.
    # Keeping the global tokenizer as None avoids recursion/pickling blowups.
    global _TOKENIZER, _TOKENIZER_NAME
    _TOKENIZER = None
    _TOKENIZER_NAME = None

    # Tokenize each split
    tokenized = {}
    for split_name, ds in ds_dict.items():
        print(f"Tokenizing split: {split_name} (num_proc={num_proc})")
        remove_cols = [c for c in text_columns if c in ds.column_names]
        if not remove_cols:
            raise RuntimeError(
                f"None of --text-columns {text_columns} exist in split '{split_name}'. "
                f"Available columns: {ds.column_names}"
            )

        tok_fp = _stable_fingerprint(
            "tok",
            {
                "split": split_name,
                "tokenizer": tokenizer_name,
                "text_columns": text_columns,
            },
        )
        tokenized_ds = ds.map(
            _tokenize_batch,
            batched=True,
            fn_kwargs={'tokenizer_name': tokenizer_name, 'text_columns': text_columns},
            remove_columns=remove_cols,
            num_proc=num_proc,
            load_from_cache_file=False,
            new_fingerprint=tok_fp,
            desc=f"Tokenizing {split_name}",
        )

        print(f"Packing tokens into blocks of {max_length} for {split_name}")
        pack_fp = _stable_fingerprint(
            "pack",
            {
                "split": split_name,
                "max_length": max_length,
            },
        )
        packed = tokenized_ds.map(
            _pack_batch,
            batched=True,
            fn_kwargs={'max_length': max_length},
            remove_columns=[c for c in tokenized_ds.column_names if c not in ('input_ids', 'attention_mask')],
            desc=f"Packing {split_name}",
            num_proc=num_proc,
            load_from_cache_file=False,
            new_fingerprint=pack_fp,
        )

        # Ensure schema contains only the required fields
        cols_to_keep = ['input_ids', 'attention_mask', 'labels']
        existing = [c for c in cols_to_keep if c in packed.column_names]
        packed = packed.remove_columns([c for c in packed.column_names if c not in existing])

        out_path = output_dir / split_name
        if overwrite_output and out_path.exists():
            import shutil

            shutil.rmtree(out_path, ignore_errors=True)
        print(f"Saving tokenized {split_name} to {out_path}")
        packed.save_to_disk(str(out_path))
        tokenized[split_name] = out_path

    return tokenized


def _dataset_cache_root(cache_dir: Optional[str], dataset_name: str) -> Optional[Path]:
    if not cache_dir:
        return None
    # datasets uses <cache_dir>/datasets/<org>--<name>
    return Path(cache_dir) / 'datasets' / dataset_name.replace('/', '--')


def _safe_load_dataset(
    dataset: str,
    dataset_config: Optional[str],
    cache_dir: Optional[str],
    *,
    force_redownload: bool,
):
    # DownloadMode exists in datasets, but to keep this script compatible across versions,
    # we only pass it when available.
    kwargs = {'cache_dir': cache_dir}
    if force_redownload:
        try:
            from datasets import DownloadMode

            kwargs['download_mode'] = DownloadMode.FORCE_REDOWNLOAD
        except Exception:
            # Older/newer datasets versions may not have DownloadMode in this location.
            pass
    try:
        if dataset_config is None:
            return load_dataset(dataset, **kwargs)
        return load_dataset(dataset, dataset_config, **kwargs)
    except TypeError:
        # If the local datasets version doesn't accept some kwargs, retry without them.
        kwargs.pop('download_mode', None)
        if dataset_config is None:
            return load_dataset(dataset, cache_dir=cache_dir)
        return load_dataset(dataset, dataset_config, cache_dir=cache_dir)


def _load_via_croissant(
    *,
    dataset: str,
    subset: str,
    cache_dir: Optional[str],
    croissant_url: Optional[str],
    max_records: Optional[int],
):
    """Load a HF dataset subset via its Croissant metadata (mlcroissant).

    Notes:
    - Hugging Face's Croissant endpoint may expose only a portion of very large datasets.
    - Authentication is handled by huggingface_hub (e.g., `huggingface-cli login`).
    """
    try:
        import requests
        from huggingface_hub.file_download import build_hf_headers
        from mlcroissant import Dataset as CroissantDataset
    except Exception as e:
        raise RuntimeError(
            "Croissant loading requires 'requests' and 'mlcroissant'. "
            "Install them (see requirements.txt) or use --source datasets."
        ) from e

    url = croissant_url or f"https://huggingface.co/api/datasets/{dataset}/croissant"
    headers = build_hf_headers()
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    jsonld = resp.json()

    # Build croissant dataset and record iterator
    croissant_ds = CroissantDataset(jsonld=jsonld)

    def record_iter():
        n = 0
        for record in croissant_ds.records(subset):
            # Ensure plain python dict for HF datasets
            if not isinstance(record, dict):
                try:
                    record = dict(record)
                except Exception:
                    record = {'text': str(record)}
            yield record
            n += 1
            if max_records is not None and n >= max_records:
                break

    # Materialize into an Arrow-backed Dataset
    from datasets import Dataset as HFDataset

    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    try:
        ds = HFDataset.from_generator(record_iter, **kwargs)
    except TypeError:
        ds = HFDataset.from_generator(record_iter)

    # Best-effort: persist the jsonld for debugging/repro
    if cache_dir:
        try:
            croissant_meta_path = Path(cache_dir) / 'croissant'
            croissant_meta_path.mkdir(parents=True, exist_ok=True)
            (croissant_meta_path / f"{dataset.replace('/', '--')}-{subset}.json").write_text(
                json.dumps(jsonld, ensure_ascii=False, indent=2), encoding='utf-8'
            )
        except Exception:
            pass

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ai4bharat/sangraha')
    parser.add_argument('--dataset-config', type=str, default='verified', help="Dataset config/subset name (sangraha uses 'verified')")
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument(
        '--source',
        type=str,
        choices=['datasets', 'croissant'],
        default='datasets',
        help="Data source: 'datasets' uses datasets.load_dataset; 'croissant' uses HF Croissant metadata via mlcroissant",
    )
    parser.add_argument('--croissant-url', type=str, default=None, help='Override Croissant metadata URL')
    parser.add_argument('--max-records', type=int, default=None, help='Optional cap on number of records (debug)')
    parser.add_argument('--force-redownload', action='store_true', help='Force re-download/re-prepare even if cache exists')
    parser.add_argument(
        '--auto-clear-cache-on-split-mismatch',
        action='store_true',
        help='If a NonMatchingSplitsSizesError occurs, delete this dataset\'s cache folder under --cache-dir and retry once',
    )
    parser.add_argument('--tokenizer-model', type=str, required=True)
    parser.add_argument('--text-columns', type=str, nargs='+', default=['text'])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--overwrite-output', action='store_true', help='Delete existing output split directories before saving')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--num-proc', type=int, default=16)
    parser.add_argument('--hf-login', action='store_true', help='Run huggingface_hub.login() before downloading')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally login
    if args.hf_login:
        try:
            from huggingface_hub import login as hf_login
            hf_login()
        except Exception:
            pass

    if platform.system().lower().startswith('win') and args.num_proc and args.num_proc > 1:
        print(
            f"WARNING: Windows + --num-proc {args.num_proc} can be unstable for datasets.map(). "
            "If you hit multiprocessing/pickling errors, retry with --num-proc 1."
        )

    print(f"Loading dataset {args.dataset} (non-streaming, source={args.source})")

    def load_primary():
        if args.source == 'croissant':
            # In croissant mode, dataset-config acts like the subset name
            return _load_via_croissant(
                dataset=args.dataset,
                subset=args.dataset_config,
                cache_dir=args.cache_dir,
                croissant_url=args.croissant_url,
                max_records=args.max_records,
            )
        return _safe_load_dataset(
            args.dataset,
            args.dataset_config,
            args.cache_dir,
            force_redownload=args.force_redownload,
        )

    try:
        ds = load_primary()
    except Exception as e:
        # Special-case: cached dataset_info.json can end up with empty expected splits (0 examples)
        # and later cause verify_splits() to fail. Clearing only this dataset cache is the most
        # reliable fix.
        if args.auto_clear_cache_on_split_mismatch and type(e).__name__ == 'NonMatchingSplitsSizesError':
            cache_root = _dataset_cache_root(args.cache_dir, args.dataset)
            if cache_root and cache_root.exists():
                print(f"NonMatchingSplitsSizesError: clearing dataset cache folder and retrying once: {cache_root}")
                import shutil

                shutil.rmtree(cache_root, ignore_errors=True)
                ds = load_primary()
            else:
                raise
        else:
            print(f"Dataset load failed ({type(e).__name__}), attempting to load any available config...")
            try:
                configs = get_dataset_config_names(args.dataset)
            except Exception:
                configs = []
            if not configs:
                raise
            if args.source == 'croissant':
                # Croissant does not have config names in the same sense; re-raise.
                raise
            ds = _safe_load_dataset(
                args.dataset,
                configs[0],
                args.cache_dir,
                force_redownload=args.force_redownload,
            )


    # Many HF corpora use a single split 'train' — unify to a single dataset
    if isinstance(ds, DatasetDict):
        if 'train' in ds:
            full = ds['train']
        else:
            # If multiple splits exist, concatenate them
            keys = list(ds.keys())
            datasets_list = [ds[k] for k in keys]
            full = datasets_list[0]
            for d in datasets_list[1:]:
                full = full.concatenate(d)
    elif isinstance(ds, Dataset):
        full = ds
    else:
        raise RuntimeError("Unexpected dataset type returned by load_dataset")

    print(f"Total examples: {len(full)}")

    splits = deterministic_split(full, seed=args.seed)

    tokenized_paths = tokenize_and_pack(
        splits,
        tokenizer_name=args.tokenizer_model,
        text_columns=args.text_columns,
        output_dir=out_dir,
        max_length=args.max_length,
        num_proc=args.num_proc,
        overwrite_output=args.overwrite_output,
    )

    print("Done. Tokenized splits saved:")
    for k, p in tokenized_paths.items():
        print(f" - {k}: {p}")


if __name__ == '__main__':
    main()
