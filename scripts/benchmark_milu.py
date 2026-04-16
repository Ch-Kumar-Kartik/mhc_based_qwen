#!/usr/bin/env python3
"""Benchmark base Qwen vs mHC on AI4Bharat/MILU.

This script evaluates multiple-choice accuracy on MILU using a prompt format
compatible with the official task definition and supports local mHC checkpoints
through repository-native loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = [
    "English",
    "Bengali",
    "Hindi",
    "Tamil",
    "Telugu",
    "Malayalam",
    "Kannada",
    "Marathi",
    "Gujarati",
    "Punjabi",
    "Odia",
]


@dataclass
class MiluRecord:
    language: str
    question: str
    options: List[str]
    target_index: int


@dataclass
class SampleResult:
    model_name: str
    language: str
    sample_index: int
    prediction: str
    target: str
    prediction_option: str
    target_option: str
    correct: bool
    latency_ms: float


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_device(device: str) -> str:
    d = device.strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return d


def _load_base_model(
    model_name_or_path: str,
    device: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    local_files_only: bool,
):
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
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

    raise RuntimeError(f"Unable to load mHC model from {model_path}: {last_error}")


def _extract_record(example: Dict, language: str) -> Optional[MiluRecord]:
    try:
        question = str(example["question"]).strip()
        options = [
            str(example["option1"]).strip(),
            str(example["option2"]).strip(),
            str(example["option3"]).strip(),
            str(example["option4"]).strip(),
        ]
        target_raw = str(example["target"]).strip().lower()
        if not target_raw.startswith("option"):
            return None
        idx = int(target_raw.replace("option", "")) - 1
        if idx < 0 or idx > 3:
            return None
        return MiluRecord(language=language, question=question, options=options, target_index=idx)
    except Exception:
        return None


def _load_milu_records(
    language: str,
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
    token: Optional[str],
) -> List[MiluRecord]:
    dataset = load_dataset(
        "ai4bharat/MILU",
        language,
        split=split,
        cache_dir=cache_dir,
        token=token if token else True,
    )
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    records: List[MiluRecord] = []
    for example in dataset:
        rec = _extract_record(example, language)
        if rec is not None:
            records.append(rec)
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def _load_fewshot_records(
    language: str,
    split: str,
    num_fewshot: int,
    cache_dir: Optional[str],
    token: Optional[str],
) -> List[MiluRecord]:
    if num_fewshot <= 0:
        return []
    dataset = load_dataset(
        "ai4bharat/MILU",
        language,
        split=split,
        cache_dir=cache_dir,
        token=token if token else True,
    )

    records: List[MiluRecord] = []
    for example in dataset:
        rec = _extract_record(example, language)
        if rec is not None:
            records.append(rec)
            if len(records) >= num_fewshot:
                break
    return records


def _index_to_letter(idx: int) -> str:
    return ["A", "B", "C", "D"][idx]


def _build_question_block(record: MiluRecord, include_answer: bool) -> str:
    block = (
        f"Question: {record.question}\n"
        "Choices:\n"
        f"A. {record.options[0]}\n"
        f"B. {record.options[1]}\n"
        f"C. {record.options[2]}\n"
        f"D. {record.options[3]}\n"
        "Answer:"
    )
    if include_answer:
        block += f" {_index_to_letter(record.target_index)}"
    return block


def _build_prompt(record: MiluRecord, fewshot_records: Sequence[MiluRecord]) -> str:
    parts: List[str] = []
    for fs in fewshot_records:
        parts.append(_build_question_block(fs, include_answer=True))
    parts.append(_build_question_block(record, include_answer=False))
    return "\n\n".join(parts)


def _format_prompt(tokenizer, prompt: str, apply_chat_template: bool) -> str:
    if not apply_chat_template:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _extract_answer_letter(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"([A-D])", text.upper())
    if match:
        return match.group(1)
    return ""


def _option_logprob(
    model,
    tokenizer,
    formatted_prompt: str,
    option_text: str,
    device: str,
    max_input_tokens: int,
) -> float:
    option_suffix = " " + option_text
    full_text = formatted_prompt + option_suffix

    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_input_tokens,
    )["input_ids"]
    option_ids = tokenizer(option_suffix, add_special_tokens=False)["input_ids"]

    if not option_ids or len(full_ids) <= len(option_ids):
        return float("-inf")

    start = len(full_ids) - len(option_ids)
    if start <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0]

    total_logprob = 0.0
    for pos in range(start, len(full_ids)):
        tok_id = full_ids[pos]
        token_logprob = torch.log_softmax(logits[pos - 1], dim=-1)[tok_id].item()
        total_logprob += token_logprob

    return total_logprob


@torch.inference_mode()
def _evaluate_model(
    model_name: str,
    model,
    tokenizer,
    language_to_records: Dict[str, List[MiluRecord]],
    language_to_fewshot: Dict[str, List[MiluRecord]],
    device: str,
    max_input_tokens: int,
    apply_chat_template: bool,
) -> Tuple[Dict[str, float], float, List[SampleResult]]:
    sample_results: List[SampleResult] = []
    language_accuracy: Dict[str, float] = {}

    total = 0
    correct_total = 0

    for language, records in language_to_records.items():
        language_correct = 0
        language_total = 0
        fewshot = language_to_fewshot.get(language, [])

        for idx, record in enumerate(records):
            prompt = _build_prompt(record, fewshot)
            formatted_prompt = _format_prompt(tokenizer, prompt, apply_chat_template=apply_chat_template)

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            option_scores = [
                _option_logprob(
                    model=model,
                    tokenizer=tokenizer,
                    formatted_prompt=formatted_prompt,
                    option_text=opt,
                    device=device,
                    max_input_tokens=max_input_tokens,
                )
                for opt in record.options
            ]
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            pred_index = int(max(range(len(option_scores)), key=lambda i: option_scores[i]))
            pred = _index_to_letter(pred_index)
            target = _index_to_letter(record.target_index)
            ok = pred_index == record.target_index

            language_total += 1
            total += 1
            if ok:
                language_correct += 1
                correct_total += 1

            sample_results.append(
                SampleResult(
                    model_name=model_name,
                    language=language,
                    sample_index=idx,
                    prediction=pred,
                    target=target,
                    prediction_option=record.options[pred_index],
                    target_option=record.options[record.target_index],
                    correct=ok,
                    latency_ms=(end - start) * 1000.0,
                )
            )

        language_accuracy[language] = (language_correct / language_total) if language_total else 0.0

    overall_accuracy = (correct_total / total) if total else 0.0
    return language_accuracy, overall_accuracy, sample_results


def _write_outputs(
    output_dir: Path,
    payload: Dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"milu_benchmark_{timestamp}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results: {out_json}")


def _parse_languages(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(LANGUAGES)
    parsed = [x.strip() for x in value.split(",") if x.strip()]
    invalid = [x for x in parsed if x not in LANGUAGES]
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}. Supported: {LANGUAGES}")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark base Qwen vs mHC on MILU")
    parser.add_argument("--mhc-model", required=True, help="Path to mHC checkpoint directory")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Base model name/path")
    parser.add_argument("--mhc-type", default="auto", choices=["auto", "v1", "v2"]) 

    parser.add_argument("--languages", default="all", help="Comma-separated languages or 'all'")
    parser.add_argument("--split", default="test", help="MILU evaluation split")
    parser.add_argument("--fewshot-split", default="validation", help="MILU fewshot split")
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--max-samples-per-language", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--apply-chat-template", action="store_true")

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-token", default=None, help="HF token for gated MILU dataset")
    parser.add_argument("--output-dir", default="output/benchmark_milu")

    args = parser.parse_args()

    random.seed(args.seed)

    languages = _parse_languages(args.languages)
    device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("Preparing MILU records...")
    language_to_records: Dict[str, List[MiluRecord]] = {}
    language_to_fewshot: Dict[str, List[MiluRecord]] = {}
    for language in languages:
        records = _load_milu_records(
            language=language,
            split=args.split,
            max_samples=args.max_samples_per_language,
            seed=args.seed,
            cache_dir=args.cache_dir,
            token=hf_token,
        )
        fewshot = _load_fewshot_records(
            language=language,
            split=args.fewshot_split,
            num_fewshot=args.num_fewshot,
            cache_dir=args.cache_dir,
            token=hf_token,
        )
        language_to_records[language] = records
        language_to_fewshot[language] = fewshot
        print(f"  {language}: eval={len(records)} fewshot={len(fewshot)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = _load_base_model(
        model_name_or_path=args.base_model,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    base_model.eval()

    print("Evaluating base model...")
    base_lang_acc, base_overall, base_samples = _evaluate_model(
        model_name="base",
        model=base_model,
        tokenizer=tokenizer,
        language_to_records=language_to_records,
        language_to_fewshot=language_to_fewshot,
        device=device,
        max_input_tokens=args.max_input_tokens,
        apply_chat_template=args.apply_chat_template,
    )

    print("Loading mHC model...")
    mhc_model = _load_mhc_model_auto(
        model_path=args.mhc_model,
        mhc_type=args.mhc_type,
        base_model=args.base_model,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    mhc_model.eval()

    print("Evaluating mHC model...")
    mhc_lang_acc, mhc_overall, mhc_samples = _evaluate_model(
        model_name="mhc",
        model=mhc_model,
        tokenizer=tokenizer,
        language_to_records=language_to_records,
        language_to_fewshot=language_to_fewshot,
        device=device,
        max_input_tokens=args.max_input_tokens,
        apply_chat_template=args.apply_chat_template,
    )

    delta_lang = {lang: mhc_lang_acc.get(lang, 0.0) - base_lang_acc.get(lang, 0.0) for lang in languages}
    payload = {
        "config": {
            "mhc_model": args.mhc_model,
            "base_model": args.base_model,
            "mhc_type": args.mhc_type,
            "languages": languages,
            "split": args.split,
            "fewshot_split": args.fewshot_split,
            "num_fewshot": args.num_fewshot,
            "max_samples_per_language": args.max_samples_per_language,
            "seed": args.seed,
            "device": device,
            "dtype": args.dtype,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "apply_chat_template": args.apply_chat_template,
        },
        "base": {
            "overall_accuracy": base_overall,
            "language_accuracy": base_lang_acc,
            "samples": [asdict(x) for x in base_samples],
        },
        "mhc": {
            "overall_accuracy": mhc_overall,
            "language_accuracy": mhc_lang_acc,
            "samples": [asdict(x) for x in mhc_samples],
        },
        "delta": {
            "overall": mhc_overall - base_overall,
            "language": delta_lang,
        },
    }

    print(f"Base overall accuracy: {base_overall:.4f}")
    print(f"mHC overall accuracy:  {mhc_overall:.4f}")
    print(f"Delta (mHC-base):      {mhc_overall - base_overall:+.4f}")

    _write_outputs(Path(args.output_dir), payload)


if __name__ == "__main__":
    main()
