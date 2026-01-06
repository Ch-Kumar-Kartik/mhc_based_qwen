#!/usr/bin/env python3
"""Benchmark performance: standard Qwen3 vs mHC Qwen3.

This script measures:
- Prefill (forward pass) latency and tokens/sec
- End-to-end generation latency and new-tokens/sec
- (CUDA only) peak allocated memory during the measured region

Example:
    python -m scripts.benchmark_performance \
        --original Qwen/Qwen3-0.6B \
        --mhc ./qwen3-0.6b-mhc \
        --device cuda \
        --dtype bfloat16 \
        --batch-size 1 \
        --prompt-len 256 \
        --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkResult:
    name: str
    prefill_ms_p50: float
    prefill_ms_p90: float
    prefill_tok_s: float
    gen_ms_p50: float
    gen_ms_p90: float
    gen_new_tok_s: float
    total_ms_p50: float
    total_ms_p90: float
    peak_mem_mb: Optional[float]


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _sync_if_cuda(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory_if_cuda(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb_if_cuda(device: str) -> Optional[float]:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    return None


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
        return "cpu"
    return device


def _build_batch(
    tokenizer,
    prompt: str,
    batch_size: int,
    prompt_len: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Trim/pad to prompt_len
    if input_ids.shape[1] > prompt_len:
        input_ids = input_ids[:, :prompt_len]
    elif input_ids.shape[1] < prompt_len:
        pad_amount = prompt_len - input_ids.shape[1]
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_id)

    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = (input_ids != pad_id).to(torch.long)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _load_original_model(
    model_id_or_path: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
):
    # Keep attention implementation consistent with mHC (uses eager)
    return AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )


def _load_mhc_model(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype,
):
    # Use repo helper if available (handles config/model wiring)
    from src.conversion import load_mhc_model

    mhc_model, _ = load_mhc_model(model_path, device=device, torch_dtype=torch_dtype)
    return mhc_model


@torch.inference_mode()
def _time_prefill(
    model,
    batch: Dict[str, torch.Tensor],
    device: str,
    repeats: int,
    warmup: int,
    use_cache: bool,
) -> Tuple[List[float], Optional[float]]:
    # Returns list of latencies (seconds), and peak memory (MB) over measured region.
    for _ in range(warmup):
        _ = model(**batch, use_cache=use_cache)
    _sync_if_cuda(device)

    times: List[float] = []
    _reset_peak_memory_if_cuda(device)

    for _ in range(repeats):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        _ = model(**batch, use_cache=use_cache)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mb = _peak_memory_mb_if_cuda(device)
    return times, peak_mb


@torch.inference_mode()
def _time_generate(
    model,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    device: str,
    repeats: int,
    warmup: int,
    max_new_tokens: int,
    use_cache: bool,
) -> Tuple[List[float], Optional[float]]:
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tokenizer.pad_token_id,
    )

    for _ in range(warmup):
        _ = model.generate(**batch, **gen_kwargs)
    _sync_if_cuda(device)

    times: List[float] = []
    _reset_peak_memory_if_cuda(device)

    for _ in range(repeats):
        _sync_if_cuda(device)
        t0 = time.perf_counter()
        _ = model.generate(**batch, **gen_kwargs)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mb = _peak_memory_mb_if_cuda(device)
    return times, peak_mb


def _summarize(
    name: str,
    prefill_s: List[float],
    gen_s: List[float],
    peak_mem_mb: Optional[float],
    batch_size: int,
    prompt_len: int,
    max_new_tokens: int,
) -> BenchmarkResult:
    prefill_ms = [t * 1000.0 for t in prefill_s]
    gen_ms = [t * 1000.0 for t in gen_s]
    total_s = [a + b for a, b in zip(prefill_s, gen_s)]
    total_ms = [t * 1000.0 for t in total_s]

    # Throughput
    prefill_tok_s = (batch_size * prompt_len) / statistics.mean(prefill_s)
    gen_new_tok_s = (batch_size * max_new_tokens) / statistics.mean(gen_s)

    return BenchmarkResult(
        name=name,
        prefill_ms_p50=_percentile(prefill_ms, 50),
        prefill_ms_p90=_percentile(prefill_ms, 90),
        prefill_tok_s=prefill_tok_s,
        gen_ms_p50=_percentile(gen_ms, 50),
        gen_ms_p90=_percentile(gen_ms, 90),
        gen_new_tok_s=gen_new_tok_s,
        total_ms_p50=_percentile(total_ms, 50),
        total_ms_p90=_percentile(total_ms, 90),
        peak_mem_mb=peak_mem_mb,
    )


def _print_result(r: BenchmarkResult):
    mem = f"{r.peak_mem_mb:.1f} MB" if r.peak_mem_mb is not None else "n/a"
    print(f"\n== {r.name} ==")
    print(
        f"prefill: p50={r.prefill_ms_p50:.2f} ms  p90={r.prefill_ms_p90:.2f} ms  "
        f"throughput={r.prefill_tok_s:.1f} tok/s"
    )
    print(
        f"gen:    p50={r.gen_ms_p50:.2f} ms  p90={r.gen_ms_p90:.2f} ms  "
        f"throughput={r.gen_new_tok_s:.1f} new_tok/s"
    )
    print(f"total:  p50={r.total_ms_p50:.2f} ms  p90={r.total_ms_p90:.2f} ms")
    print(f"peak mem (CUDA): {mem}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark standard Qwen3 vs mHC Qwen3")
    parser.add_argument("--original", default="Qwen/Qwen3-0.6B", help="Original model name/path")
    parser.add_argument("--mhc", required=True, help="mHC model path")

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading/inference (cuda/cpu). If CUDA unavailable, falls back to cpu.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention implementation for the ORIGINAL model. mHC uses eager.",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=256, help="Token length after pad/trim")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="The capital of France is")

    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true", help="Disable KV-cache")

    args = parser.parse_args()

    device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    use_cache = not args.no_cache

    print("Benchmark configuration:")
    print(f"  device:         {device}")
    print(f"  dtype:          {args.dtype}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  prompt_len:     {args.prompt_len}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  repeats:        {args.repeats} (warmup {args.warmup})")
    print(f"  use_cache:      {use_cache}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.original, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = _build_batch(
        tokenizer=tokenizer,
        prompt=args.prompt,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        device=device,
    )

    results: List[BenchmarkResult] = []

    print("\nLoading original model...")
    original = _load_original_model(
        args.original,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    original.eval()

    prefill_s, prefill_peak = _time_prefill(
        original,
        batch=batch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
        use_cache=use_cache,
    )
    gen_s, gen_peak = _time_generate(
        original,
        batch=batch,
        tokenizer=tokenizer,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
    )

    results.append(
        _summarize(
            name="Original",
            prefill_s=prefill_s,
            gen_s=gen_s,
            peak_mem_mb=max([m for m in [prefill_peak, gen_peak] if m is not None], default=None),
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
        )
    )

    # Free memory before loading mHC
    del original
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading mHC model...")
    mhc = _load_mhc_model(args.mhc, device=device, torch_dtype=torch_dtype)
    mhc.eval()

    prefill_s, prefill_peak = _time_prefill(
        mhc,
        batch=batch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
        use_cache=use_cache,
    )
    gen_s, gen_peak = _time_generate(
        mhc,
        batch=batch,
        tokenizer=tokenizer,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
    )

    results.append(
        _summarize(
            name="mHC",
            prefill_s=prefill_s,
            gen_s=gen_s,
            peak_mem_mb=max([m for m in [prefill_peak, gen_peak] if m is not None], default=None),
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            max_new_tokens=args.max_new_tokens,
        )
    )

    print("\n" + "=" * 72)
    print("Performance summary")
    print("=" * 72)
    for r in results:
        _print_result(r)


if __name__ == "__main__":
    main()
