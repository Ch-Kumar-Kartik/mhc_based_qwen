#!/usr/bin/env python3
"""Benchmark base Qwen vs mHC on IndicQA.

This script evaluates:
- Response ability: Exact Match (EM) and token-level F1
- Token consumption: prompt/output/total tokens
- Memory consumption: CUDA peak memory (and optional CPU RSS delta)
- Latency: end-to-end generation latency

Example:
    python -m scripts.benchmark_indicqa \
        --base-model Qwen/Qwen3-0.6B \
        --mhc-model ./output/qwen3_mhc_v2_converted \
        --mhc-type v2 \
        --split validation \
        --max-samples 200 \
        --device cuda \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None
    list_repo_files = None


try:
    import psutil
except Exception:
    psutil = None


@dataclass
class QARecord:
    record_id: str
    language: str
    context: str
    question: str
    references: List[str]


@dataclass
class SampleMetrics:
    model_name: str
    sample_index: int
    record_id: str
    language: str
    em: float
    f1: float
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    peak_cuda_mem_mb: Optional[float]
    rss_delta_mb: Optional[float]
    prediction: str
    reference: str


@dataclass
class ModelSummary:
    name: str
    samples_evaluated: int
    exact_match: float
    f1: float
    response_ability_score: float
    avg_prompt_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    total_tokens: int
    avg_latency_ms: float
    p90_latency_ms: float
    avg_peak_cuda_mem_mb: Optional[float]
    max_peak_cuda_mem_mb: Optional[float]
    avg_rss_delta_mb: Optional[float]
    max_rss_delta_mb: Optional[float]


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
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


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


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    floor_idx = int(k)
    ceil_idx = min(floor_idx + 1, len(sorted_values) - 1)
    if floor_idx == ceil_idx:
        return sorted_values[floor_idx]
    d0 = sorted_values[floor_idx] * (ceil_idx - k)
    d1 = sorted_values[ceil_idx] * (k - floor_idx)
    return d0 + d1


def _normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = text.casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    text = " ".join(text.split())
    return text


def _exact_match(prediction: str, references: Sequence[str]) -> float:
    pred = _normalize_answer(prediction)
    if not references:
        return 0.0
    for ref in references:
        if pred == _normalize_answer(ref):
            return 1.0
    return 0.0


def _f1_score(prediction: str, references: Sequence[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    if not references:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = _normalize_answer(ref).split()
        if not pred_tokens and not ref_tokens:
            score = 1.0
        elif not pred_tokens or not ref_tokens:
            score = 0.0
        else:
            common = {}
            for token in pred_tokens:
                common[token] = common.get(token, 0) + 1
            overlap = 0
            for token in ref_tokens:
                if common.get(token, 0) > 0:
                    overlap += 1
                    common[token] -= 1
            if overlap == 0:
                score = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(ref_tokens)
                score = (2.0 * precision * recall) / (precision + recall)
        best = max(best, score)
    return best


def _get_example_value(example: Dict[str, Any], candidates: Sequence[str]) -> Any:
    key_lookup = {k.lower(): k for k in example.keys()}
    for candidate in candidates:
        if candidate.lower() in key_lookup:
            return example[key_lookup[candidate.lower()]]
    return None


def _extract_references(example: Dict[str, Any]) -> List[str]:
    answers_value = _get_example_value(example, ["answers", "answer", "gold_answers", "label"])

    refs: List[str] = []
    if isinstance(answers_value, dict):
        text_values = answers_value.get("text")
        if isinstance(text_values, list):
            refs.extend([str(x).strip() for x in text_values if str(x).strip()])
        elif isinstance(text_values, str) and text_values.strip():
            refs.append(text_values.strip())

        if not refs:
            for value in answers_value.values():
                if isinstance(value, list):
                    refs.extend([str(x).strip() for x in value if str(x).strip()])
                elif isinstance(value, str) and value.strip():
                    refs.append(value.strip())
    elif isinstance(answers_value, list):
        for item in answers_value:
            if isinstance(item, str) and item.strip():
                refs.append(item.strip())
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    refs.append(txt.strip())
    elif isinstance(answers_value, str) and answers_value.strip():
        refs.append(answers_value.strip())

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for r in refs:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    return deduped


def _extract_qa_record(example: Dict[str, Any], index: int) -> Optional[QARecord]:
    context = _get_example_value(example, ["context", "passage", "paragraph", "article", "text"])
    question = _get_example_value(example, ["question", "query", "prompt"])
    references = _extract_references(example)

    if not context or not question or not references:
        return None

    record_id = _get_example_value(example, ["id", "qid", "question_id"])
    if record_id is None:
        record_id = str(index)

    language = _get_example_value(example, ["language", "lang", "locale", "iso", "split"])
    if language is None:
        language = "unknown"

    return QARecord(
        record_id=str(record_id),
        language=str(language),
        context=str(context),
        question=str(question),
        references=references,
    )


def _build_prompt(context: str, question: str) -> str:
    return (
        "You are given a context and a question. "
        "Answer using a short span copied from the context when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _extract_prediction(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    first_line = text.splitlines()[0].strip()
    return first_line


def _split_matches(record_id: str, split: str) -> bool:
    split_key = split.strip().lower()
    if split_key in {"all", "*", "full"}:
        return True

    bucket = int(hashlib.md5(record_id.encode("utf-8")).hexdigest(), 16) % 10
    if split_key in {"train", "training"}:
        return bucket <= 7
    if split_key in {"validation", "valid", "val", "dev"}:
        return bucket == 8
    if split_key in {"test", "testing"}:
        return bucket == 9
    return True


def _load_indicqa_records_from_repo_json(
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int,
    shuffle: bool,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> List[QARecord]:
    if hf_hub_download is None or list_repo_files is None:
        raise RuntimeError(
            "huggingface_hub is required for IndicQA fallback loading. "
            "Install it or upgrade the environment."
        )

    repo_files = list_repo_files(dataset_name, repo_type="dataset")
    json_files = sorted(
        [f for f in repo_files if f.startswith("data/indicqa.") and f.endswith(".json")]
    )
    if not json_files:
        raise RuntimeError(
            f"Could not find raw IndicQA language JSON files in dataset repo: {dataset_name}"
        )

    records: List[QARecord] = []

    for rel_path in json_files:
        lang = Path(rel_path).stem.split(".")[-1]
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=rel_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        with open(local_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for article in payload.get("data", []):
            for para in article.get("paragraphs", []):
                context = str(para.get("context", "")).strip()
                if not context:
                    continue

                for qa in para.get("qas", []):
                    question = str(qa.get("question", "")).strip()
                    if not question:
                        continue

                    record_id = str(qa.get("id") or f"{lang}_{len(records)}")
                    if not _split_matches(record_id, split):
                        continue

                    refs = []
                    for answer_obj in qa.get("answers", []):
                        if isinstance(answer_obj, dict):
                            ref = str(answer_obj.get("text", "")).strip()
                            if ref:
                                refs.append(ref)

                    # Deduplicate and keep order.
                    deduped_refs: List[str] = []
                    seen = set()
                    for ref in refs:
                        if ref not in seen:
                            deduped_refs.append(ref)
                            seen.add(ref)

                    if not deduped_refs:
                        continue

                    records.append(
                        QARecord(
                            record_id=record_id,
                            language=lang,
                            context=context,
                            question=question,
                            references=deduped_refs,
                        )
                    )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    if max_samples > 0:
        records = records[:max_samples]

    print(
        "Loaded "
        f"{len(records)} QA records from raw IndicQA repo JSON files "
        f"for split '{split}'."
    )
    return records


def _load_indicqa_records(
    dataset_name: str,
    split: str,
    max_samples: int,
    seed: int,
    shuffle: bool,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> List[QARecord]:
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
        )
    except Exception as first_error:
        error_text = str(first_error)
        if "Dataset scripts are no longer supported" in error_text or "IndicQA.py" in error_text:
            print(
                "Dataset script loading is unsupported in current datasets version. "
                "Falling back to raw language JSON files from dataset repo..."
            )
            return _load_indicqa_records_from_repo_json(
                dataset_name=dataset_name,
                split=split,
                max_samples=max_samples,
                seed=seed,
                shuffle=shuffle,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        print(f"Primary load failed ({first_error}). Trying fallback load path...")
        dataset_obj = load_dataset(dataset_name, cache_dir=cache_dir)
        if isinstance(dataset_obj, dict) and split in dataset_obj:
            dataset = dataset_obj[split]  # type: ignore[index]
        else:
            first_split = list(dataset_obj.keys())[0]  # type: ignore[arg-type]
            print(f"Requested split '{split}' missing. Falling back to '{first_split}'.")
            dataset = dataset_obj[first_split]  # type: ignore[index]

    if shuffle and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    records: List[QARecord] = []
    total = len(dataset) if hasattr(dataset, "__len__") else max_samples
    limit = min(max_samples, total) if max_samples > 0 else total

    for idx, example in enumerate(dataset):
        if max_samples > 0 and len(records) >= max_samples:
            break
        qa = _extract_qa_record(example, idx)
        if qa is not None:
            records.append(qa)

    print(f"Loaded {len(records)} valid QA records from split '{split}'.")
    return records


def _get_rss_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return None


def _load_base_model(
    model_name_or_path: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str,
    trust_remote_code: bool,
    local_files_only: bool,
):
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )


def _load_mhc_model_auto(
    model_path: str,
    mhc_type: str,
    base_model: str,
    device: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    local_files_only: bool,
):
    if mhc_type == "v2":
        from src.conversionV2 import load_mhc_model_v2

        model, _ = load_mhc_model_v2(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return model

    if mhc_type == "v1":
        from src.conversion import load_mhc_model

        model, _ = load_mhc_model(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
        )
        return model

    # auto mode
    last_error: Optional[Exception] = None

    try:
        from src.conversionV2 import load_mhc_model_v2

        model, _ = load_mhc_model_v2(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        print("Loaded mHC model using V2 loader.")
        return model
    except Exception as e:
        last_error = e
        print(f"V2 load failed: {e}")

    try:
        from src.conversion import load_mhc_model

        model, _ = load_mhc_model(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
        )
        print("Loaded mHC model using V1 loader.")
        return model
    except Exception as e:
        last_error = e
        print(f"V1 load failed: {e}")

    if last_error is not None:
        raise RuntimeError(f"Unable to load mHC model from {model_path}: {last_error}")
    raise RuntimeError(f"Unable to load mHC model from {model_path}")


@torch.inference_mode()
def _evaluate_model(
    model_name: str,
    model,
    tokenizer,
    records: Sequence[QARecord],
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    use_cache: bool,
) -> Tuple[ModelSummary, List[SampleMetrics]]:
    sample_metrics: List[SampleMetrics] = []

    for idx, record in enumerate(records):
        prompt = _build_prompt(record.context, record.question)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])

        _reset_peak_memory_if_cuda(device)
        rss_before = _get_rss_memory_mb()

        _sync_if_cuda(device)
        start = time.perf_counter()
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        _sync_if_cuda(device)
        end = time.perf_counter()

        peak_cuda_mem_mb = _peak_memory_mb_if_cuda(device)
        rss_after = _get_rss_memory_mb()

        output_tokens = int(output_ids.shape[1] - prompt_tokens)
        generated = tokenizer.decode(
            output_ids[0, prompt_tokens:],
            skip_special_tokens=True,
        )
        prediction = _extract_prediction(generated)

        em = _exact_match(prediction, record.references)
        f1 = _f1_score(prediction, record.references)

        rss_delta_mb = None
        if rss_before is not None and rss_after is not None:
            rss_delta_mb = max(0.0, rss_after - rss_before)

        sample_metrics.append(
            SampleMetrics(
                model_name=model_name,
                sample_index=idx,
                record_id=record.record_id,
                language=record.language,
                em=em,
                f1=f1,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=prompt_tokens + output_tokens,
                latency_ms=(end - start) * 1000.0,
                peak_cuda_mem_mb=peak_cuda_mem_mb,
                rss_delta_mb=rss_delta_mb,
                prediction=prediction,
                reference=record.references[0],
            )
        )

    if not sample_metrics:
        raise ValueError("No samples were evaluated. Check dataset fields/split and retry.")

    em_values = [m.em for m in sample_metrics]
    f1_values = [m.f1 for m in sample_metrics]
    prompt_values = [m.prompt_tokens for m in sample_metrics]
    output_values = [m.output_tokens for m in sample_metrics]
    total_values = [m.total_tokens for m in sample_metrics]
    latency_values = [m.latency_ms for m in sample_metrics]

    cuda_values = [m.peak_cuda_mem_mb for m in sample_metrics if m.peak_cuda_mem_mb is not None]
    rss_values = [m.rss_delta_mb for m in sample_metrics if m.rss_delta_mb is not None]

    em_score = 100.0 * statistics.mean(em_values)
    f1_score = 100.0 * statistics.mean(f1_values)

    summary = ModelSummary(
        name=model_name,
        samples_evaluated=len(sample_metrics),
        exact_match=em_score,
        f1=f1_score,
        response_ability_score=0.5 * (em_score + f1_score),
        avg_prompt_tokens=statistics.mean(prompt_values),
        avg_output_tokens=statistics.mean(output_values),
        avg_total_tokens=statistics.mean(total_values),
        total_tokens=sum(total_values),
        avg_latency_ms=statistics.mean(latency_values),
        p90_latency_ms=_percentile(latency_values, 90),
        avg_peak_cuda_mem_mb=statistics.mean(cuda_values) if cuda_values else None,
        max_peak_cuda_mem_mb=max(cuda_values) if cuda_values else None,
        avg_rss_delta_mb=statistics.mean(rss_values) if rss_values else None,
        max_rss_delta_mb=max(rss_values) if rss_values else None,
    )

    return summary, sample_metrics


def _save_outputs(
    output_dir: Path,
    base_summary: ModelSummary,
    mhc_summary: ModelSummary,
    base_metrics: Sequence[SampleMetrics],
    mhc_metrics: Sequence[SampleMetrics],
    args: argparse.Namespace,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"indicqa_benchmark_summary_{timestamp}.json"
    samples_path = output_dir / f"indicqa_benchmark_samples_{timestamp}.csv"

    comparison = {
        "response_ability_delta": {
            "exact_match": mhc_summary.exact_match - base_summary.exact_match,
            "f1": mhc_summary.f1 - base_summary.f1,
            "response_ability_score": (
                mhc_summary.response_ability_score - base_summary.response_ability_score
            ),
        },
        "token_consumption_delta": {
            "avg_prompt_tokens": mhc_summary.avg_prompt_tokens - base_summary.avg_prompt_tokens,
            "avg_output_tokens": mhc_summary.avg_output_tokens - base_summary.avg_output_tokens,
            "avg_total_tokens": mhc_summary.avg_total_tokens - base_summary.avg_total_tokens,
            "total_tokens": mhc_summary.total_tokens - base_summary.total_tokens,
        },
        "latency_delta_ms": {
            "avg_latency_ms": mhc_summary.avg_latency_ms - base_summary.avg_latency_ms,
            "p90_latency_ms": mhc_summary.p90_latency_ms - base_summary.p90_latency_ms,
        },
        "memory_delta": {
            "avg_peak_cuda_mem_mb": None
            if mhc_summary.avg_peak_cuda_mem_mb is None or base_summary.avg_peak_cuda_mem_mb is None
            else mhc_summary.avg_peak_cuda_mem_mb - base_summary.avg_peak_cuda_mem_mb,
            "max_peak_cuda_mem_mb": None
            if mhc_summary.max_peak_cuda_mem_mb is None or base_summary.max_peak_cuda_mem_mb is None
            else mhc_summary.max_peak_cuda_mem_mb - base_summary.max_peak_cuda_mem_mb,
            "avg_rss_delta_mb": None
            if mhc_summary.avg_rss_delta_mb is None or base_summary.avg_rss_delta_mb is None
            else mhc_summary.avg_rss_delta_mb - base_summary.avg_rss_delta_mb,
            "max_rss_delta_mb": None
            if mhc_summary.max_rss_delta_mb is None or base_summary.max_rss_delta_mb is None
            else mhc_summary.max_rss_delta_mb - base_summary.max_rss_delta_mb,
        },
    }

    payload = {
        "benchmark": "IndicQA",
        "dataset": args.dataset,
        "split": args.split,
        "max_samples": args.max_samples,
        "generation": {
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": not args.no_cache,
        },
        "runtime": {
            "device": args.device,
            "dtype": args.dtype,
        },
        "models": {
            "base": asdict(base_summary),
            "mhc": asdict(mhc_summary),
        },
        "comparison": comparison,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with samples_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "sample_index",
                "record_id",
                "language",
                "em",
                "f1",
                "prompt_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "peak_cuda_mem_mb",
                "rss_delta_mb",
                "prediction",
                "reference",
            ],
        )
        writer.writeheader()
        for row in list(base_metrics) + list(mhc_metrics):
            writer.writerow(asdict(row))

    return summary_path, samples_path


def _print_summary(base: ModelSummary, mhc: ModelSummary):
    def _fmt(v: Optional[float], suffix: str = "") -> str:
        if v is None:
            return "n/a"
        return f"{v:.3f}{suffix}"

    print("\n" + "=" * 80)
    print("IndicQA Benchmark Summary")
    print("=" * 80)
    print("Response ability (higher is better):")
    print(f"  Base - EM: {base.exact_match:.3f} | F1: {base.f1:.3f} | Score: {base.response_ability_score:.3f}")
    print(f"  mHC  - EM: {mhc.exact_match:.3f} | F1: {mhc.f1:.3f} | Score: {mhc.response_ability_score:.3f}")

    print("\nToken consumption (lower is better for efficiency):")
    print(
        "  Base - avg prompt/output/total: "
        f"{base.avg_prompt_tokens:.2f}/{base.avg_output_tokens:.2f}/{base.avg_total_tokens:.2f}"
    )
    print(
        "  mHC  - avg prompt/output/total: "
        f"{mhc.avg_prompt_tokens:.2f}/{mhc.avg_output_tokens:.2f}/{mhc.avg_total_tokens:.2f}"
    )

    print("\nLatency (ms, lower is better):")
    print(f"  Base - avg: {base.avg_latency_ms:.3f} | p90: {base.p90_latency_ms:.3f}")
    print(f"  mHC  - avg: {mhc.avg_latency_ms:.3f} | p90: {mhc.p90_latency_ms:.3f}")

    print("\nMemory consumption:")
    print(
        "  Base - avg/max CUDA peak MB: "
        f"{_fmt(base.avg_peak_cuda_mem_mb)} / {_fmt(base.max_peak_cuda_mem_mb)}"
    )
    print(
        "  mHC  - avg/max CUDA peak MB: "
        f"{_fmt(mhc.avg_peak_cuda_mem_mb)} / {_fmt(mhc.max_peak_cuda_mem_mb)}"
    )
    print(
        "  Base - avg/max RSS delta MB: "
        f"{_fmt(base.avg_rss_delta_mb)} / {_fmt(base.max_rss_delta_mb)}"
    )
    print(
        "  mHC  - avg/max RSS delta MB: "
        f"{_fmt(mhc.avg_rss_delta_mb)} / {_fmt(mhc.max_rss_delta_mb)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark base Qwen vs mHC model on IndicQA (response ability, tokens, memory)."
    )
    parser.add_argument("--dataset", default="ai4bharat/IndicQA", help="HF dataset name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum valid QA samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before selecting samples")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")

    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Base model name/path")
    parser.add_argument("--mhc-model", required=True, help="mHC model path")
    parser.add_argument(
        "--mhc-type",
        default="auto",
        choices=["auto", "v1", "v2"],
        help="mHC loader type",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa"],
        help="Attention implementation for base model",
    )
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache")

    parser.add_argument("--trust-remote-code", action="store_true", help="Enable remote model/dataset code")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local HF cache/files")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache directory")
    parser.add_argument("--output-dir", default="output/benchmark_indicqa", help="Output directory")

    args = parser.parse_args()

    args.device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)

    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("CPU execution with float16/bfloat16 may be unstable. Using float32 instead.")
        torch_dtype = torch.float32

    print("Benchmark configuration:")
    print(f"  dataset:          {args.dataset}")
    print(f"  split:            {args.split}")
    print(f"  max_samples:      {args.max_samples}")
    print(f"  device:           {args.device}")
    print(f"  dtype:            {torch_dtype}")
    print(f"  base model:       {args.base_model}")
    print(f"  mhc model:        {args.mhc_model}")
    print(f"  mhc type:         {args.mhc_type}")

    print("\nLoading dataset records...")
    records = _load_indicqa_records(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if not records:
        raise ValueError("No valid records found in dataset split.")

    print("\nLoading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nEvaluating base model...")
    base_model = _load_base_model(
        args.base_model,
        device=args.device,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    base_model.eval()

    base_summary, base_metrics = _evaluate_model(
        model_name="base",
        model=base_model,
        tokenizer=tokenizer,
        records=records,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
    )

    del base_model
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nEvaluating mHC model...")
    mhc_model = _load_mhc_model_auto(
        model_path=args.mhc_model,
        mhc_type=args.mhc_type,
        base_model=args.base_model,
        device=args.device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    mhc_model.eval()

    mhc_summary, mhc_metrics = _evaluate_model(
        model_name="mhc",
        model=mhc_model,
        tokenizer=tokenizer,
        records=records,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        use_cache=not args.no_cache,
    )

    _print_summary(base_summary, mhc_summary)

    summary_path, samples_path = _save_outputs(
        output_dir=Path(args.output_dir),
        base_summary=base_summary,
        mhc_summary=mhc_summary,
        base_metrics=base_metrics,
        mhc_metrics=mhc_metrics,
        args=args,
    )

    print("\nSaved outputs:")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Samples CSV:  {samples_path}")


if __name__ == "__main__":
    main()
