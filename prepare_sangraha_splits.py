#!/usr/bin/env python3

import argparse
import logging
import os
import gc
import json
from pathlib import Path
from itertools import chain

import numpy as np
from tqdm.auto import tqdm

from transformers import AutoTokenizer, logging as hf_logging
from datasets import (
    load_from_disk,
    concatenate_datasets,
    DatasetDict,
    config as datasets_config,
)

hf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------
# helpers
# -------------------------------------------------------

def resolve_num_proc(num_proc):
    if num_proc:
        return num_proc
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def get_dir_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def format_bytes(x):
    for unit in ["B","KB","MB","GB","TB"]:
        if x < 1024:
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} PB"


def summarize_shards(shard_paths):

    summary = []

    for shard_path in shard_paths:
        row_count = None
        try:
            shard_ds = load_from_disk(str(shard_path))
            row_count = len(shard_ds)
        except Exception:
            pass

        summary.append(
            {
                "name": shard_path.name,
                "path": str(shard_path),
                "rows": row_count,
                "size_bytes": get_dir_size(shard_path),
            }
        )

    return summary


# -------------------------------------------------------
# language iterator
# -------------------------------------------------------

def iter_languages(data_dir):

    data_path = Path(data_dir)

    langs = [
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ]

    langs = sorted(langs)

    for lang in langs:

        logger.info(f"Loading language {lang.name}")

        dataset = load_from_disk(str(lang))

        logger.info(f"{lang.name}: {len(dataset):,} examples")

        yield lang.name, dataset


# -------------------------------------------------------
# tokenization
# -------------------------------------------------------

def tokenize_function(examples, tokenizer):

    return tokenizer(
        examples["text"],
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )


# -------------------------------------------------------
# packing
# -------------------------------------------------------

def group_texts_numpy(examples, block_size):

    concatenated = np.concatenate(
        [np.array(x, dtype=np.int32) for x in examples["input_ids"]]
    )

    total_length = (len(concatenated) // block_size) * block_size

    if total_length == 0:
        return {"input_ids": []}

    result = concatenated[:total_length].reshape(-1, block_size)

    return {"input_ids": [x.tolist() for x in result]}


# -------------------------------------------------------
# cleanup
# -------------------------------------------------------

def cleanup_arrow_cache(cache_dir):

    cache = Path(cache_dir)

    freed = 0
    files = 0

    for f in cache.glob("*.arrow"):

        try:
            freed += f.stat().st_size
            f.unlink()
            files += 1
        except:
            pass

    logger.info(
        f"Cache cleanup: removed {files} files freed {format_bytes(freed)}"
    )


# -------------------------------------------------------
# tokenize + pack
# -------------------------------------------------------

def tokenize_and_pack(
    dataset,
    tokenizer,
    max_length,
    batch_size,
    num_proc,
):

    logger.info("Tokenizing")

    tokenized = dataset.map(

        lambda x: tokenize_function(x, tokenizer),

        batched=True,

        batch_size=batch_size,

        num_proc=num_proc,

        remove_columns=["text"],

        writer_batch_size=1000,

        load_from_cache_file=False,

        desc="Tokenizing",

    )

    logger.info("Packing")

    packed = tokenized.map(

        lambda x: group_texts_numpy(x, max_length),

        batched=True,

        batch_size=batch_size,

        num_proc=num_proc,

        remove_columns=tokenized.column_names,

        writer_batch_size=1000,

        load_from_cache_file=False,

        desc="Packing",

    )

    del tokenized
    gc.collect()

    return packed


# -------------------------------------------------------
# splits
# -------------------------------------------------------

def create_splits(dataset, seed):

    dataset = dataset.shuffle(seed=seed)

    train_test = dataset.train_test_split(
        test_size=0.2,
        seed=seed,
        shuffle=False
    )

    val_test = train_test["test"].train_test_split(
        test_size=0.5,
        seed=seed,
        shuffle=False
    )

    return DatasetDict({

        "train": train_test["train"],

        "validation": val_test["train"],

        "test": val_test["test"]

    })


# -------------------------------------------------------
# main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--arrow-cache-dir", required=True)

    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-0.6B")

    parser.add_argument("--max-length", type=int, default=2048)

    parser.add_argument("--batch-size", type=int, default=2000)

    parser.add_argument("--num-proc", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--compression", default="zstd")

    parser.add_argument(
        "--skip-merge-final",
        action="store_true",
        help="Keep per-language shard_* outputs only; skip concatenate/split/final save_to_disk.",
    )

    args = parser.parse_args()

    num_proc = resolve_num_proc(args.num_proc)

    if args.compression != "none":
        datasets_config.DEFAULT_COMPRESSION = args.compression

    arrow_cache = Path(args.arrow_cache_dir)
    arrow_cache.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
        use_fast=True
    )

    logger.info(f"Tokenizer vocab size {len(tokenizer):,}")

    data_size = get_dir_size(args.data_dir)

    logger.info(f"Input dataset size {format_bytes(data_size)}")

    shard_paths = []

    # ---------------------------------------------------
    # process each language separately
    # ---------------------------------------------------

    for lang, dataset in iter_languages(args.data_dir):

        shard_dir = Path(args.output_dir) / f"shard_{lang}"

        # 🔹 Resume safety
        if shard_dir.exists():
            logger.info(f"Skipping {lang} (already processed)")
            shard_paths.append(shard_dir)
            del dataset
            continue

        logger.info(f"Processing {lang}")

        packed = tokenize_and_pack(
            dataset,
            tokenizer,
            args.max_length,
            args.batch_size,
            num_proc,
        )

        logger.info(f"Saving shard {lang}")

        packed.save_to_disk(shard_dir)

        shard_paths.append(shard_dir)

        del dataset
        del packed
        gc.collect()

        cleanup_arrow_cache(args.arrow_cache_dir)

    # ---------------------------------------------------
    # merge shards
    # ---------------------------------------------------

    if args.skip_merge_final:

        logger.info("Skipping final merge/split as requested (--skip-merge-final)")

        shard_summary = summarize_shards(shard_paths)

        summary_path = Path(args.output_dir) / "shard_metadata.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer": args.tokenizer_model,
                    "max_length": args.max_length,
                    "num_shards": len(shard_summary),
                    "total_sequences": sum(
                        s["rows"] for s in shard_summary if isinstance(s.get("rows"), int)
                    ),
                    "total_size_bytes": sum(s["size_bytes"] for s in shard_summary),
                    "shards": shard_summary,
                },
                f,
                indent=2,
            )

        logger.info("========== STORAGE REPORT ==========")
        logger.info(f"Input size: {format_bytes(data_size)}")
        logger.info(
            "Shard output size: "
            f"{format_bytes(sum(s['size_bytes'] for s in shard_summary))}"
        )
        logger.info(f"Wrote shard metadata: {summary_path}")
        logger.info("Arrow cache can be deleted safely")
        return

    logger.info("Loading shards")

    shards = [load_from_disk(str(p)) for p in shard_paths]

    logger.info("Concatenating shards")

    dataset = concatenate_datasets(shards)

    logger.info("Creating splits")

    splits = create_splits(dataset, args.seed)

    final_dir = Path(args.output_dir) / "final"

    logger.info("Saving final dataset")

    splits.save_to_disk(final_dir)

    # ---------------------------------------------------
    # metadata
    # ---------------------------------------------------

    metadata = {

        "tokenizer": args.tokenizer_model,

        "max_length": args.max_length,

        "splits": {

            "train": len(splits["train"]),

            "validation": len(splits["validation"]),

            "test": len(splits["test"]),

        },

        "total_sequences": sum(len(splits[k]) for k in splits)

    }

    with open(final_dir / "metadata.json","w") as f:
        json.dump(metadata,f,indent=2)

    output_size = get_dir_size(final_dir)

    logger.info("========== STORAGE REPORT ==========")

    logger.info(f"Input size: {format_bytes(data_size)}")

    logger.info(f"Output size: {format_bytes(output_size)}")

    logger.info("Arrow cache can be deleted safely")


if __name__ == "__main__":
    main()