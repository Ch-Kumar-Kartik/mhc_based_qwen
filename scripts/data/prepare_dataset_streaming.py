#!/usr/bin/env python3
"""
Streaming dataset preparation for ai4bharat/sangraha verified.

- Uses datasets streaming (no full raw download)
- Deterministic 80/10/10 split via hash of record index + seed
- Tokenizes in small batches and packs into fixed-length blocks
- Writes tokenized blocks to Parquet shards per split
- Converts shards into Arrow datasets via save_to_disk for train.py

Notes:
- This avoids storing the full raw dataset on disk.
- A temporary set of Parquet shards is created under <output_dir>/parquet_tmp.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, load_dataset as hf_load_dataset
from transformers import AutoTokenizer


def _stable_hash_bucket(seed: int, index: int) -> int:
    h = hashlib.sha1(f"{seed}-{index}".encode("utf-8")).hexdigest()
    return int(h, 16) % 100


def _choose_split(seed: int, index: int) -> str:
    bucket = _stable_hash_bucket(seed, index)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _get_text(record: dict, text_columns: List[str]) -> str:
    parts = []
    for col in text_columns:
        if col in record and record[col] is not None:
            parts.append(str(record[col]))
    return "\n".join(parts)


class TokenBuffer:
    def __init__(self):
        self.input_ids: List[int] = []
        self.attention_mask: List[int] = []
        self._start = 0

    def append(self, input_ids: List[int], attention_mask: Optional[List[int]] = None):
        self.input_ids.extend(input_ids)
        if attention_mask is None:
            self.attention_mask.extend([1] * len(input_ids))
        else:
            self.attention_mask.extend(attention_mask)

    def pop_block(self, max_length: int) -> Optional[Tuple[List[int], List[int]]]:
        if len(self.input_ids) - self._start < max_length:
            return None
        i0 = self._start
        i1 = i0 + max_length
        block_ids = self.input_ids[i0:i1]
        block_attn = self.attention_mask[i0:i1]
        self._start = i1
        if self._start > 1_000_000:
            # Trim lists occasionally to keep memory bounded
            self.input_ids = self.input_ids[self._start :]
            self.attention_mask = self.attention_mask[self._start :]
            self._start = 0
        return block_ids, block_attn


class ParquetShardWriter:
    def __init__(self, out_dir: Path, rows_per_shard: int = 1000):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self._rows: List[dict] = []
        self._shard_idx = 0

    def add(self, input_ids: List[int], attention_mask: List[int]):
        self._rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
            }
        )
        if len(self._rows) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows)
        shard_path = self.out_dir / f"shard-{self._shard_idx:06d}.parquet"
        pq.write_table(table, shard_path)
        self._rows = []
        self._shard_idx += 1

    def close(self):
        self.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ai4bharat/sangraha")
    parser.add_argument("--dataset-config", type=str, default="verified")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--source",
        type=str,
        choices=["datasets", "parquet-hf"],
        default="datasets",
        help="datasets=HF dataset builder (streaming), parquet-hf=HF parquet-converted branch via hf:// URL",
    )
    parser.add_argument(
        "--parquet-revision",
        type=str,
        default="refs/convert/parquet",
        help="HF revision for parquet-converted files",
    )
    parser.add_argument(
        "--parquet-glob",
        type=str,
        default="**/*.parquet",
        help="Glob pattern within the parquet branch (used with --source parquet-hf)",
    )
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--text-columns", type=str, nargs="+", default=["text"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-login", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    if args.hf_login:
        try:
            from huggingface_hub import login as hf_login

            hf_login()
        except Exception:
            pass

    out_dir = Path(args.output_dir)
    if out_dir.exists() and args.overwrite_output:
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_tmp = out_dir / "parquet_tmp"
    parquet_tmp.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Streaming load
    if args.source == "parquet-hf":
        # Use the parquet-converted branch via hf:// URL
        data_files = f"hf://datasets/{args.dataset}@{args.parquet_revision}/{args.parquet_glob}"
        ds = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=True,
            cache_dir=args.cache_dir,
        )
    else:
        ds = load_dataset(
            args.dataset,
            args.dataset_config,
            streaming=True,
            cache_dir=args.cache_dir,
        )
    # Determine stream split
    if isinstance(ds, dict) or hasattr(ds, "keys"):
        if "train" in ds:
            stream = ds["train"]
        else:
            first_key = list(ds.keys())[0]
            stream = ds[first_key]
    else:
        stream = ds

    buffers: Dict[str, TokenBuffer] = defaultdict(TokenBuffer)
    writers = {
        "train": ParquetShardWriter(parquet_tmp / "train", rows_per_shard=args.rows_per_shard),
        "validation": ParquetShardWriter(parquet_tmp / "validation", rows_per_shard=args.rows_per_shard),
        "test": ParquetShardWriter(parquet_tmp / "test", rows_per_shard=args.rows_per_shard),
    }

    batch_texts: List[str] = []
    batch_splits: List[str] = []

    def flush_batch():
        if not batch_texts:
            return
        enc = tokenizer(batch_texts, return_attention_mask=True)
        for i in range(len(batch_texts)):
            split = batch_splits[i]
            buffers[split].append(enc["input_ids"][i], enc.get("attention_mask", [None] * len(enc["input_ids"]))[i])
            while True:
                block = buffers[split].pop_block(args.max_length)
                if block is None:
                    break
                input_ids, attention_mask = block
                writers[split].add(input_ids, attention_mask)
        batch_texts.clear()
        batch_splits.clear()

    for idx, record in enumerate(stream):
        if args.max_records is not None and idx >= args.max_records:
            break
        text = _get_text(record, args.text_columns)
        if not text:
            continue
        split = _choose_split(args.seed, idx)
        batch_texts.append(text)
        batch_splits.append(split)
        if len(batch_texts) >= args.batch_size:
            flush_batch()

    flush_batch()
    for w in writers.values():
        w.close()

    # Convert Parquet shards to Arrow datasets (save_to_disk) for train.py
    for split in ("train", "validation", "test"):
        shard_glob = str((parquet_tmp / split / "*.parquet").resolve())
        ds_split = hf_load_dataset("parquet", data_files=shard_glob)["train"]
        split_out = out_dir / split
        ds_split.save_to_disk(str(split_out))

    print("Done. Tokenized splits saved:")
    for split in ("train", "validation", "test"):
        print(f" - {split}: {out_dir / split}")


if __name__ == "__main__":
    main()
