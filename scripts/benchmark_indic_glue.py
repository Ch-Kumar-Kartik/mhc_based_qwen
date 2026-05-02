#!/usr/bin/env python3
"""Benchmark base Qwen vs mHC on IndicGLUE subsets.

This script evaluates zero-shot or few-shot multiple-choice and small-label
classification tasks in ai4bharat/indic_glue and reports accuracy, latency,
per-subset/task/language breakdowns, and deltas between base and mHC models.
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
from datasets import ClassLabel, Sequence as DatasetsSequence, get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_NAME = "ai4bharat/indic_glue"
DEFAULT_TASK_PREFIXES = ["copa", "wnli"]

LANGUAGE_CODE_MAP = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "od": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "en": "English",
}


@dataclass
class ChoiceRecord:
    subset: str
    task: str
    language: str
    prompt: str
    choices: List[str]
    target_index: int


@dataclass
class SampleResult:
    model_name: str
    subset: str
    task: str
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


def _split_subset(subset: str) -> Tuple[str, str, str]:
    if "." in subset:
        task, lang_code = subset.rsplit(".", 1)
    else:
        task, lang_code = subset, "unknown"
    language = LANGUAGE_CODE_MAP.get(lang_code, lang_code)
    return task, lang_code, language


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_subsets(all_subsets: Sequence[str], subset_value: str, task_prefixes: Sequence[str]) -> List[str]:
    if subset_value and subset_value.lower() != "auto":
        return _parse_csv(subset_value)
    prefixes = [p.strip().lower() for p in task_prefixes if p.strip()]
    if not prefixes:
        return list(all_subsets)
    selected = []
    for subset in all_subsets:
        task, _, _ = _split_subset(subset)
        if any(task.lower().startswith(prefix) for prefix in prefixes):
            selected.append(subset)
    return selected


def _is_token_classification_feature(label_feature) -> bool:
    return isinstance(label_feature, DatasetsSequence)


def _resolve_label_choices(dataset, max_labels: int, scan_limit: int) -> Optional[List[str]]:
    label_feature = dataset.features.get("label") if hasattr(dataset, "features") else None
    if isinstance(label_feature, ClassLabel):
        return list(label_feature.names)
    if _is_token_classification_feature(label_feature):
        return None

    labels: List[str] = []
    seen = set()
    scan_dataset = dataset
    if scan_limit > 0 and hasattr(dataset, "select"):
        try:
            scan_dataset = dataset.select(range(min(scan_limit, len(dataset))))
        except Exception:
            scan_dataset = dataset

    for example in scan_dataset:
        value = example.get("label")
        if isinstance(value, list):
            return None
        if value is None:
            continue
        text_value = str(value)
        if text_value in seen:
            continue
        seen.add(text_value)
        labels.append(text_value)
        if max_labels > 0 and len(labels) > max_labels:
            return None
    return labels


def _collect_choice_fields(example: Dict) -> Optional[List[str]]:
    for prefix in ("choice", "option"):
        choices: List[str] = []
        idx = 1
        while f"{prefix}{idx}" in example:
            value = example.get(f"{prefix}{idx}")
            if value is None:
                break
            choices.append(str(value).strip())
            idx += 1
        if len(choices) >= 2:
            return choices
    return None


def _parse_target_index(label_value, choices: Sequence[str]) -> Optional[int]:
    if label_value is None:
        return None
    if isinstance(label_value, (int, float)):
        idx = int(label_value)
        if 0 <= idx < len(choices):
            return idx
    if isinstance(label_value, str):
        value = label_value.strip()
        match = re.match(r"^(option|choice)(\d+)$", value, re.IGNORECASE)
        if match:
            idx = int(match.group(2)) - 1
            if 0 <= idx < len(choices):
                return idx
        if len(value) == 1 and value.isalpha():
            idx = ord(value.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return idx
        if value in choices:
            return choices.index(value)
        if value.isdigit():
            idx = int(value)
            if 0 <= idx < len(choices):
                return idx
    return None


def _build_prompt_text(example: Dict, task: str) -> Optional[str]:
    lines: List[str] = []
    if "premise" in example:
        lines.append(f"Premise: {str(example['premise']).strip()}")
    if "hypothesis" in example:
        lines.append(f"Hypothesis: {str(example['hypothesis']).strip()}")
    if "sentence1" in example:
        lines.append(f"Sentence 1: {str(example['sentence1']).strip()}")
    if "sentence2" in example:
        lines.append(f"Sentence 2: {str(example['sentence2']).strip()}")
    if "context" in example:
        lines.append(f"Context: {str(example['context']).strip()}")
    if "question" in example and task.lower().startswith("copa"):
        lines.append(f"Question: Which option is the {str(example['question']).strip()}?")
    elif "question" in example:
        lines.append(f"Question: {str(example['question']).strip()}")
    if "text" in example:
        lines.append(f"Text: {str(example['text']).strip()}")
    if "sentence" in example:
        lines.append(f"Sentence: {str(example['sentence']).strip()}")
    if not lines:
        return None
    return "\n".join(lines)


def _build_choice_record(
    example: Dict,
    subset: str,
    task: str,
    language: str,
    label_choices: Optional[List[str]],
    max_choice_count: int,
) -> Optional[ChoiceRecord]:
    choices = _collect_choice_fields(example)
    if choices is None and label_choices:
        choices = label_choices
    if not choices:
        return None
    if max_choice_count > 0 and len(choices) > max_choice_count:
        return None

    label_value = example.get("label")
    target_index = _parse_target_index(label_value, choices)
    if target_index is None:
        return None

    prompt_text = _build_prompt_text(example, task)
    if not prompt_text:
        return None

    return ChoiceRecord(
        subset=subset,
        task=task,
        language=language,
        prompt=prompt_text,
        choices=list(choices),
        target_index=target_index,
    )


def _index_to_letter(idx: int) -> str:
    if idx < 0 or idx >= 26:
        return ""
    return chr(ord("A") + idx)


def _build_question_block(record: ChoiceRecord, include_answer: bool) -> str:
    lines = [record.prompt, "Choices:"]
    for idx, choice in enumerate(record.choices):
        letter = _index_to_letter(idx)
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    block = "\n".join(lines)
    if include_answer:
        block += f" {_index_to_letter(record.target_index)}"
    return block


def _build_prompt(record: ChoiceRecord, fewshot_records: Sequence[ChoiceRecord]) -> str:
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
    subset_to_records: Dict[str, List[ChoiceRecord]],
    subset_to_fewshot: Dict[str, List[ChoiceRecord]],
    device: str,
    max_input_tokens: int,
    apply_chat_template: bool,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, List[SampleResult]]:
    sample_results: List[SampleResult] = []
    subset_accuracy: Dict[str, float] = {}
    task_accuracy: Dict[str, float] = {}
    language_accuracy: Dict[str, float] = {}

    total = 0
    correct_total = 0
    task_counts: Dict[str, Tuple[int, int]] = {}
    language_counts: Dict[str, Tuple[int, int]] = {}

    for subset, records in subset_to_records.items():
        subset_correct = 0
        subset_total = 0
        fewshot = subset_to_fewshot.get(subset, [])

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
                for opt in record.choices
            ]
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            pred_index = int(max(range(len(option_scores)), key=lambda i: option_scores[i]))
            pred = _index_to_letter(pred_index)
            target = _index_to_letter(record.target_index)
            ok = pred_index == record.target_index

            subset_total += 1
            total += 1
            if ok:
                subset_correct += 1
                correct_total += 1

            task_correct, task_total = task_counts.get(record.task, (0, 0))
            task_total += 1
            if ok:
                task_correct += 1
            task_counts[record.task] = (task_correct, task_total)

            lang_correct, lang_total = language_counts.get(record.language, (0, 0))
            lang_total += 1
            if ok:
                lang_correct += 1
            language_counts[record.language] = (lang_correct, lang_total)

            sample_results.append(
                SampleResult(
                    model_name=model_name,
                    subset=record.subset,
                    task=record.task,
                    language=record.language,
                    sample_index=idx,
                    prediction=pred,
                    target=target,
                    prediction_option=record.choices[pred_index],
                    target_option=record.choices[record.target_index],
                    correct=ok,
                    latency_ms=(end - start) * 1000.0,
                )
            )

        subset_accuracy[subset] = (subset_correct / subset_total) if subset_total else 0.0

    for task, (correct, count) in task_counts.items():
        task_accuracy[task] = (correct / count) if count else 0.0
    for language, (correct, count) in language_counts.items():
        language_accuracy[language] = (correct / count) if count else 0.0

    overall_accuracy = (correct_total / total) if total else 0.0
    return subset_accuracy, task_accuracy, language_accuracy, overall_accuracy, sample_results


def _write_outputs(output_dir: Path, payload: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"indic_glue_benchmark_{timestamp}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results: {out_json}")


def _load_records_for_subset(
    subset: str,
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
    token: Optional[str],
    max_labels: int,
    label_scan_size: int,
    max_choice_count: int,
) -> List[ChoiceRecord]:
    dataset = load_dataset(
        DATASET_NAME,
        subset,
        split=split,
        cache_dir=cache_dir,
        token=token if token else True,
    )
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    task, _, language = _split_subset(subset)
    label_choices = _resolve_label_choices(dataset, max_labels=max_labels, scan_limit=label_scan_size)

    records: List[ChoiceRecord] = []
    for example in dataset:
        record = _build_choice_record(
            example=example,
            subset=subset,
            task=task,
            language=language,
            label_choices=label_choices,
            max_choice_count=max_choice_count,
        )
        if record is not None:
            records.append(record)
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def _load_fewshot_records(
    subset: str,
    split: str,
    num_fewshot: int,
    cache_dir: Optional[str],
    token: Optional[str],
    max_labels: int,
    label_scan_size: int,
    max_choice_count: int,
) -> List[ChoiceRecord]:
    if num_fewshot <= 0:
        return []

    dataset = load_dataset(
        DATASET_NAME,
        subset,
        split=split,
        cache_dir=cache_dir,
        token=token if token else True,
    )

    task, _, language = _split_subset(subset)
    label_choices = _resolve_label_choices(dataset, max_labels=max_labels, scan_limit=label_scan_size)

    records: List[ChoiceRecord] = []
    for example in dataset:
        record = _build_choice_record(
            example=example,
            subset=subset,
            task=task,
            language=language,
            label_choices=label_choices,
            max_choice_count=max_choice_count,
        )
        if record is not None:
            records.append(record)
            if len(records) >= num_fewshot:
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark base Qwen vs mHC on IndicGLUE")
    parser.add_argument("--mhc-model", help="Path to mHC checkpoint directory")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Base model name/path")
    parser.add_argument("--mhc-type", default="auto", choices=["auto", "v1", "v2"])

    parser.add_argument(
        "--subsets",
        default="auto",
        help="Comma-separated IndicGLUE subsets or 'auto' for task prefixes",
    )
    parser.add_argument(
        "--task-prefixes",
        default=",".join(DEFAULT_TASK_PREFIXES),
        help="Comma-separated task prefixes to include when --subsets=auto",
    )
    parser.add_argument("--list-subsets", action="store_true", help="List available subsets and exit")
    parser.add_argument("--split", default="validation", help="Evaluation split (train/validation/test)")
    parser.add_argument("--fewshot-split", default="train", help="Few-shot split")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--max-samples-per-subset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-labels", type=int, default=8, help="Skip tasks with more labels (0=disable)")
    parser.add_argument("--label-scan-size", type=int, default=2000)
    parser.add_argument("--max-choice-count", type=int, default=26)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--apply-chat-template", action="store_true")

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-token", default=None, help="HF token if needed")
    parser.add_argument("--output-dir", default="output/benchmark_indic_glue")

    args = parser.parse_args()

    all_subsets = get_dataset_config_names(DATASET_NAME)
    if args.list_subsets:
        print("Available subsets:")
        for subset in all_subsets:
            print(f"  {subset}")
        return

    if not args.mhc_model:
        raise SystemExit("--mhc-model is required unless --list-subsets is set.")

    task_prefixes = _parse_csv(args.task_prefixes)
    selected_subsets = _select_subsets(all_subsets, args.subsets, task_prefixes)
    if not selected_subsets:
        raise SystemExit("No subsets selected. Use --list-subsets or adjust --subsets/--task-prefixes.")

    random.seed(args.seed)

    device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("Preparing IndicGLUE records...")
    subset_to_records: Dict[str, List[ChoiceRecord]] = {}
    subset_to_fewshot: Dict[str, List[ChoiceRecord]] = {}

    for subset in selected_subsets:
        try:
            records = _load_records_for_subset(
                subset=subset,
                split=args.split,
                max_samples=args.max_samples_per_subset,
                seed=args.seed,
                cache_dir=args.cache_dir,
                token=hf_token,
                max_labels=args.max_labels,
                label_scan_size=args.label_scan_size,
                max_choice_count=args.max_choice_count,
            )
            fewshot = _load_fewshot_records(
                subset=subset,
                split=args.fewshot_split,
                num_fewshot=args.num_fewshot,
                cache_dir=args.cache_dir,
                token=hf_token,
                max_labels=args.max_labels,
                label_scan_size=args.label_scan_size,
                max_choice_count=args.max_choice_count,
            )
        except Exception as exc:
            print(f"Skipping {subset}: {exc}")
            continue

        if not records:
            print(f"Skipping {subset}: no usable records found")
            continue

        subset_to_records[subset] = records
        subset_to_fewshot[subset] = fewshot
        task, _, language = _split_subset(subset)
        print(f"  {subset}: task={task} language={language} eval={len(records)} fewshot={len(fewshot)}")

    if not subset_to_records:
        raise SystemExit("No usable IndicGLUE records were found for the selected subsets.")

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
    base_subset_acc, base_task_acc, base_lang_acc, base_overall, base_samples = _evaluate_model(
        model_name="base",
        model=base_model,
        tokenizer=tokenizer,
        subset_to_records=subset_to_records,
        subset_to_fewshot=subset_to_fewshot,
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
    mhc_subset_acc, mhc_task_acc, mhc_lang_acc, mhc_overall, mhc_samples = _evaluate_model(
        model_name="mhc",
        model=mhc_model,
        tokenizer=tokenizer,
        subset_to_records=subset_to_records,
        subset_to_fewshot=subset_to_fewshot,
        device=device,
        max_input_tokens=args.max_input_tokens,
        apply_chat_template=args.apply_chat_template,
    )

    delta_subset = {subset: mhc_subset_acc.get(subset, 0.0) - base_subset_acc.get(subset, 0.0)
                    for subset in subset_to_records}
    delta_task = {task: mhc_task_acc.get(task, 0.0) - base_task_acc.get(task, 0.0)
                  for task in set(base_task_acc) | set(mhc_task_acc)}
    delta_language = {language: mhc_lang_acc.get(language, 0.0) - base_lang_acc.get(language, 0.0)
                      for language in set(base_lang_acc) | set(mhc_lang_acc)}

    payload = {
        "config": {
            "mhc_model": args.mhc_model,
            "base_model": args.base_model,
            "mhc_type": args.mhc_type,
            "subsets": list(subset_to_records.keys()),
            "split": args.split,
            "fewshot_split": args.fewshot_split,
            "num_fewshot": args.num_fewshot,
            "max_samples_per_subset": args.max_samples_per_subset,
            "seed": args.seed,
            "device": device,
            "dtype": args.dtype,
            "max_input_tokens": args.max_input_tokens,
            "apply_chat_template": args.apply_chat_template,
            "max_labels": args.max_labels,
            "label_scan_size": args.label_scan_size,
            "max_choice_count": args.max_choice_count,
        },
        "base": {
            "overall_accuracy": base_overall,
            "subset_accuracy": base_subset_acc,
            "task_accuracy": base_task_acc,
            "language_accuracy": base_lang_acc,
            "samples": [asdict(x) for x in base_samples],
        },
        "mhc": {
            "overall_accuracy": mhc_overall,
            "subset_accuracy": mhc_subset_acc,
            "task_accuracy": mhc_task_acc,
            "language_accuracy": mhc_lang_acc,
            "samples": [asdict(x) for x in mhc_samples],
        },
        "delta": {
            "overall": mhc_overall - base_overall,
            "subset": delta_subset,
            "task": delta_task,
            "language": delta_language,
        },
    }

    print(f"Base overall accuracy: {base_overall:.4f}")
    print(f"mHC overall accuracy:  {mhc_overall:.4f}")
    print(f"Delta (mHC-base):      {mhc_overall - base_overall:+.4f}")

    _write_outputs(Path(args.output_dir), payload)


if __name__ == "__main__":
    main()
