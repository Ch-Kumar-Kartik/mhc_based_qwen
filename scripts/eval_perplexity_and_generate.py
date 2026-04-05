#!/usr/bin/env python3
"""Evaluate language-model perplexity and produce sample generations.

Supports:
- Base Hugging Face CausalLM checkpoints
- Local mHC V2 checkpoints (model_type=qwen3_mhc_v2)

Examples:
    python -m scripts.eval_perplexity_and_generate \
        --model-path output/qwen3_mhc_v2 \
        --dataset-path data/sangraha_packed \
        --split validation \
        --max-eval-batches 50

    python -m scripts.eval_perplexity_and_generate \
        --model-path output/qwen3_mhc_v2 \
        --prompt "Translate to Hindi: How are you?" \
        --prompt "Write a short poem about rain." \
        --max-new-tokens 64 --do-sample --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.conversionV2 import load_mhc_model_v2


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_device(device: str) -> str:
    device = device.strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA requested but not available; falling back to CPU")
        return "cpu"
    return device


def _detect_model_type(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return "base"

    cfg = path / "config.json"
    if not cfg.exists():
        return "base"

    try:
        config = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return "base"

    if config.get("model_type") == "qwen3_mhc_v2":
        return "mhc_v2"

    arch = config.get("architectures", [])
    if isinstance(arch, list) and any("Qwen3MHCForCausalLMV2" in str(x) for x in arch):
        return "mhc_v2"

    return "base"


def _load_eval_dataset(dataset_path: str, split: str):
    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    split_path = root / split
    if split_path.exists() and split_path.is_dir():
        return load_from_disk(str(split_path))

    train_path = root / "train"
    val_path = root / "validation"
    if train_path.exists() and train_path.is_dir():
        if split == "train":
            return load_from_disk(str(train_path))
        if split == "validation":
            if val_path.exists() and val_path.is_dir():
                return load_from_disk(str(val_path))
            raise FileNotFoundError(
                f"Requested split '{split}' not found under {dataset_path}. "
                "Available: train"
            )

    shards = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("shard_")])
    if shards:
        valid_shards = []
        skipped = []
        for shard in shards:
            if (shard / "state.json").exists() and (shard / "dataset_info.json").exists():
                valid_shards.append(shard)
            else:
                skipped.append(shard.name)

        if skipped:
            print(
                "[warning] Skipping invalid shard directories: "
                + ", ".join(skipped)
            )

        if not valid_shards:
            raise FileNotFoundError(
                "No valid shard datasets found. Each shard_* directory must contain "
                "state.json and dataset_info.json"
            )

        if split not in {"train", "validation", "all"}:
            print(
                f"[warning] split='{split}' ignored for shard layout; using concatenated shards"
            )
        return concatenate_datasets([load_from_disk(str(s)) for s in valid_shards])

    return load_from_disk(str(root))


def _collate_batch(examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    keys = set()
    for ex in examples:
        keys.update(ex.keys())

    batch: Dict[str, torch.Tensor] = {}
    for key in keys:
        values = [ex[key] for ex in examples if key in ex]
        if len(values) != len(examples):
            continue
        batch[key] = torch.as_tensor(values)

    if "input_ids" not in batch:
        raise ValueError("Batch does not include 'input_ids'; cannot evaluate perplexity")

    if "labels" not in batch:
        batch["labels"] = batch["input_ids"].clone()

    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.long)

    return batch


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.inference_mode()
def _evaluate_perplexity(
    model,
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_eval_batches: int,
    device: str,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_batch,
    )

    total_weighted_nll = 0.0
    total_tokens = 0
    total_loss = 0.0
    total_batches = 0

    started = time.perf_counter()

    for i, raw_batch in enumerate(loader):
        if max_eval_batches > 0 and i >= max_eval_batches:
            break

        batch = _move_batch_to_device(raw_batch, device=device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
            use_cache=False,
        )
        loss_value = float(outputs.loss.item())

        labels = batch["labels"]
        if labels.ndim == 2 and labels.shape[1] > 1:
            valid_tokens = (labels[:, 1:] != -100).sum().item()
        else:
            valid_tokens = labels.numel()

        valid_tokens = int(valid_tokens)
        if valid_tokens > 0:
            total_weighted_nll += loss_value * valid_tokens
            total_tokens += valid_tokens

        total_loss += loss_value
        total_batches += 1

    elapsed = time.perf_counter() - started

    if total_batches == 0:
        raise RuntimeError("No evaluation batches were processed")

    if total_tokens > 0:
        mean_loss = total_weighted_nll / float(total_tokens)
    else:
        mean_loss = total_loss / float(total_batches)

    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "num_batches": total_batches,
        "num_tokens": total_tokens,
        "elapsed_sec": elapsed,
    }


def _read_prompts(inline_prompts: List[str], prompts_file: Optional[str]) -> List[str]:
    prompts = list(inline_prompts)

    if prompts_file:
        path = Path(prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

        lines = path.read_text(encoding="utf-8").splitlines()
        file_prompts = [line.strip() for line in lines if line.strip()]
        prompts.extend(file_prompts)

    deduped = []
    seen = set()
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)

    return deduped


@torch.inference_mode()
def _generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
) -> List[Dict[str, Any]]:
    generations: List[Dict[str, Any]] = []

    for idx, prompt in enumerate(prompts, start=1):
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

        started = time.perf_counter()
        out = model.generate(**enc, **gen_kwargs)
        elapsed = time.perf_counter() - started

        input_len = enc["input_ids"].shape[1]
        generated_ids = out[0, input_len:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)

        sample = {
            "index": idx,
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "input_tokens": int(input_len),
            "generated_tokens": int(generated_ids.shape[0]),
            "elapsed_sec": elapsed,
        }
        generations.append(sample)

    return generations


def _print_perplexity_result(result: Dict[str, Any]) -> None:
    print("\n=== Perplexity ===")
    print(f"Mean loss:     {result['mean_loss']:.6f}")
    print(f"Perplexity:    {result['perplexity']:.6f}")
    print(f"Eval batches:  {result['num_batches']}")
    print(f"Eval tokens:   {result['num_tokens']}")
    print(f"Elapsed sec:   {result['elapsed_sec']:.2f}")


def _print_generation_results(samples: List[Dict[str, Any]]) -> None:
    print("\n=== Sample Generations ===")
    for sample in samples:
        print("-" * 80)
        print(f"[{sample['index']}] Prompt: {sample['prompt']}")
        print(f"Completion: {sample['completion']}")
        print(
            "Stats: "
            f"input_tokens={sample['input_tokens']} "
            f"generated_tokens={sample['generated_tokens']} "
            f"elapsed_sec={sample['elapsed_sec']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute model perplexity and/or generate sample outputs"
    )

    parser.add_argument("--model-path", required=True, help="Model path or Hugging Face model id")
    parser.add_argument(
        "--model-type",
        choices=["auto", "base", "mhc_v2"],
        default="auto",
        help="Model loader selection",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-0.6B",
        help="Base tokenizer source for mHC V2 checkpoints",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Override tokenizer source; defaults to model-path (base) or base-model (mHC V2)",
    )

    parser.add_argument("--dataset-path", default=None, help="Dataset root for perplexity evaluation")
    parser.add_argument("--split", default="validation", help="Dataset split name")
    parser.add_argument("--batch-size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means evaluate all batches")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader")

    parser.add_argument("--prompt", action="append", default=[], help="Inline prompt (repeatable)")
    parser.add_argument("--prompts-file", default=None, help="Text file with one prompt per line")

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    parser.add_argument("--device", default="cuda", help="Device string: cuda, cuda:0, cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--run-perplexity", action="store_true", help="Force perplexity mode")
    parser.add_argument("--run-generation", action="store_true", help="Force generation mode")

    parser.add_argument("--output-json", default=None, help="Write run summary JSON")
    parser.add_argument("--generations-jsonl", default=None, help="Write generations as JSONL")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    if device == "cpu" and torch_dtype == torch.float16:
        print("[warning] float16 on CPU is unsupported/slow on many setups; using float32")
        torch_dtype = torch.float32

    run_perplexity = args.run_perplexity
    run_generation = args.run_generation

    prompts = _read_prompts(args.prompt, args.prompts_file)

    if not run_perplexity and not run_generation:
        run_perplexity = args.dataset_path is not None
        run_generation = len(prompts) > 0

    if run_perplexity and not args.dataset_path:
        raise ValueError("Perplexity mode requires --dataset-path")

    if run_generation and not prompts:
        raise ValueError("Generation mode requires --prompt and/or --prompts-file")

    if not run_perplexity and not run_generation:
        raise ValueError(
            "Nothing to do. Provide --dataset-path for perplexity and/or prompts for generation"
        )

    model_type = args.model_type
    if model_type == "auto":
        model_type = _detect_model_type(args.model_path)

    print("=== Run Configuration ===")
    print(f"model_path:   {args.model_path}")
    print(f"model_type:   {model_type}")
    print(f"device:       {device}")
    print(f"dtype:        {torch_dtype}")
    print(f"perplexity:   {run_perplexity}")
    print(f"generation:   {run_generation}")

    if model_type == "mhc_v2":
        model, tokenizer = load_mhc_model_v2(
            model_path=args.model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=(args.tokenizer_path or args.base_model),
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    else:
        tokenizer_source = args.tokenizer_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            attn_implementation="eager",
        )
        model = model.to(device)

    model.eval()

    summary: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "model_path": args.model_path,
        "model_type": model_type,
        "device": device,
        "dtype": str(torch_dtype),
    }

    if run_perplexity:
        dataset = _load_eval_dataset(args.dataset_path, args.split)
        ppl_result = _evaluate_perplexity(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_eval_batches=args.max_eval_batches,
            device=device,
        )
        _print_perplexity_result(ppl_result)
        summary["perplexity"] = {
            "dataset_path": args.dataset_path,
            "split": args.split,
            **ppl_result,
        }

    generation_samples: List[Dict[str, Any]] = []
    if run_generation:
        generation_samples = _generate_samples(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
        )
        _print_generation_results(generation_samples)
        summary["generation"] = {
            "num_prompts": len(prompts),
            "settings": {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
                "repetition_penalty": args.repetition_penalty,
            },
            "samples": generation_samples,
        }

    if args.generations_jsonl and generation_samples:
        out_path = Path(args.generations_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in generation_samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote generations JSONL: {out_path}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
