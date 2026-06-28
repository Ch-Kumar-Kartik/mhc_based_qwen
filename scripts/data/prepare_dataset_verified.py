#!/usr/bin/env python3
"""
Prepare and tokenize ai4bharat/sangraha (verified) using Hugging Face Datasets.

This script:
- Loads the dataset with load_dataset("ai4bharat/sangraha", "verified")
- Creates a deterministic 80/10/10 split
- Tokenizes once, packs into fixed-length blocks
- Saves Arrow datasets to disk for train.py

Notes from the dataset card:
- Verified/Unverified can also be loaded via data_dir, but this script uses the
  explicit config as requested.
"""
import argparse
import hashlib
import json
import platform
from itertools import chain
from pathlib import Path
from typing import List

from datasets import Dataset, DatasetDict, load_dataset
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

    per_example = []
    for i in range(n):
        parts = []
        for col in text_columns:
            if col in examples and examples[col] is not None and i < len(examples[col]):
                val = examples[col][i]
                if val is None:
                    continue
                parts.append(str(val))
        per_example.append("\n".join(parts))

    return tokenizer(per_example, return_attention_mask=True)


def _pack_batch(examples, *, max_length: int):
    all_input_ids = list(chain.from_iterable(examples["input_ids"]))
    all_attention = None
    if "attention_mask" in examples and examples["attention_mask"] is not None:
        all_attention = list(chain.from_iterable(examples["attention_mask"]))

    total_length = (len(all_input_ids) // max_length) * max_length
    result = {
        "input_ids": [
            all_input_ids[i : i + max_length] for i in range(0, total_length, max_length)
        ]
    }
    if all_attention is not None:
        result["attention_mask"] = [
            all_attention[i : i + max_length] for i in range(0, total_length, max_length)
        ]
    result["labels"] = result["input_ids"].copy()
    return result


def _stable_fingerprint(prefix: str, payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"{prefix}-{hashlib.sha1(data).hexdigest()}"


def _dataset_cache_root(cache_dir: str | None, dataset_name: str) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / "datasets" / dataset_name.replace("/", "--")


def _safe_load_dataset(dataset: str, dataset_config: str, cache_dir: str | None, *, force_redownload: bool):
    kwargs = {"cache_dir": cache_dir}
    if force_redownload:
        try:
            from datasets import DownloadMode

            kwargs["download_mode"] = DownloadMode.FORCE_REDOWNLOAD
        except Exception:
            pass
    try:
        return load_dataset(dataset, dataset_config, **kwargs)
    except TypeError:
        kwargs.pop("download_mode", None)
        return load_dataset(dataset, dataset_config, cache_dir=cache_dir)


def deterministic_split(ds, seed: int = 42):
    first = ds.train_test_split(test_size=0.2, seed=seed)
    second = first["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": first["train"],
            "validation": second["train"],
            "test": second["test"],
        }
    )


def tokenize_and_pack(
    ds_dict: DatasetDict,
    tokenizer_name: str,
    text_columns: List[str],
    output_dir: Path,
    max_length: int = 2048,
    num_proc: int = 1,
    overwrite_output: bool = False,
):
    global _TOKENIZER, _TOKENIZER_NAME
    _TOKENIZER = None
    _TOKENIZER_NAME = None

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
            {"split": split_name, "tokenizer": tokenizer_name, "text_columns": text_columns},
        )
        tokenized_ds = ds.map(
            _tokenize_batch,
            batched=True,
            fn_kwargs={"tokenizer_name": tokenizer_name, "text_columns": text_columns},
            remove_columns=remove_cols,
            num_proc=num_proc,
            load_from_cache_file=False,
            new_fingerprint=tok_fp,
            desc=f"Tokenizing {split_name}",
        )

        pack_fp = _stable_fingerprint(
            "pack",
            {"split": split_name, "max_length": max_length},
        )
        print(f"Packing tokens into blocks of {max_length} for {split_name}")
        packed = tokenized_ds.map(
            _pack_batch,
            batched=True,
            fn_kwargs={"max_length": max_length},
            remove_columns=[
                c for c in tokenized_ds.column_names if c not in ("input_ids", "attention_mask")
            ],
            num_proc=num_proc,
            load_from_cache_file=False,
            new_fingerprint=pack_fp,
            desc=f"Packing {split_name}",
        )

        cols_to_keep = ["input_ids", "attention_mask", "labels"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ai4bharat/sangraha")
    parser.add_argument("--dataset-config", type=str, default="verified")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--auto-clear-cache-on-split-mismatch", action="store_true")
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--text-columns", type=str, nargs="+", default=["text"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--hf-login", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hf_login:
        try:
            from huggingface_hub import login as hf_login

            hf_login()
        except Exception:
            pass

    if platform.system().lower().startswith("win") and args.num_proc and args.num_proc > 1:
        print(
            f"WARNING: Windows + --num-proc {args.num_proc} can be unstable for datasets.map(). "
            "If you hit multiprocessing/pickling errors, retry with --num-proc 1."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset} config={args.dataset_config}")
    try:
        ds = _safe_load_dataset(
            args.dataset,
            args.dataset_config,
            args.cache_dir,
            force_redownload=args.force_redownload,
        )
    except Exception as e:
        if args.auto_clear_cache_on_split_mismatch and type(e).__name__ == "NonMatchingSplitsSizesError":
            cache_root = _dataset_cache_root(args.cache_dir, args.dataset)
            if cache_root and cache_root.exists():
                print(f"NonMatchingSplitsSizesError: clearing dataset cache folder and retrying once: {cache_root}")
                import shutil

                shutil.rmtree(cache_root, ignore_errors=True)
                ds = _safe_load_dataset(
                    args.dataset,
                    args.dataset_config,
                    args.cache_dir,
                    force_redownload=args.force_redownload,
                )
            else:
                raise
        else:
            raise

    if isinstance(ds, DatasetDict):
        if "train" in ds:
            full = ds["train"]
        else:
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


if __name__ == "__main__":
    main()
