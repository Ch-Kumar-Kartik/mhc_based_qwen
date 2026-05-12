#!/usr/bin/env python3
"""Benchmark a single model on IndicNLG (5 tasks x 11 Indic languages).

Tasks (ai4bharat/IndicNLG collection):
- IndicParaphrase
- IndicWikiBio
- IndicQuestionGeneration
- IndicSentenceSummarization
- IndicHeadlineGeneration

Metrics:
- BLEU-4 (simple corpus BLEU with brevity penalty)
- ROUGE-L (LCS F1)
- Latency and token counts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None
    list_repo_files = None

INDIC_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
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
}

TASK_SPECS = {
    "ai4bharat/IndicParaphrase": {
        "task": "paraphrase",
        "input_candidates": ["sentence1", "sent1", "text", "source", "input", "sentence"],
        "target_candidates": ["sentence2", "sent2", "paraphrase", "target", "reference", "output"],
        "prompt": "Rewrite the sentence with the same meaning:\n{source}\nParaphrase:",
    },
    "ai4bharat/IndicWikiBio": {
        "task": "wikibio",
        "input_candidates": ["input_text", "table", "infobox", "facts", "content", "text"],
        "target_candidates": ["target_text", "bio", "biography", "summary", "output"],
        "prompt": "Write a short biography from the facts below:\n{source}\nBiography:",
    },
    "ai4bharat/IndicQuestionGeneration": {
        "task": "question_generation",
        "input_candidates": ["context", "passage", "paragraph", "text"],
        "answer_candidates": ["answer", "answers", "answer_text", "span"],
        "target_candidates": ["question", "questions", "target", "output"],
        "prompt": "Given the passage and answer, write a question.\nPassage:\n{source}\nAnswer: {answer}\nQuestion:",
        "prompt_no_answer": "Write a question based on the passage:\n{source}\nQuestion:",
    },
    "ai4bharat/IndicSentenceSummarization": {
        "task": "sentence_summarization",
        "input_candidates": ["text", "document", "article", "context", "input"],
        "target_candidates": ["summary", "target", "output"],
        "prompt": "Summarize the following text in one sentence:\n{source}\nSummary:",
    },
    "ai4bharat/IndicHeadlineGeneration": {
        "task": "headline_generation",
        "input_candidates": ["text", "document", "article", "context", "input"],
        "target_candidates": ["headline", "title", "summary", "target", "output"],
        "prompt": "Generate a news headline for the article:\n{source}\nHeadline:",
    },
}


@dataclass
class NLGRecord:
    dataset_name: str
    task: str
    language_code: str
    language_name: str
    source: str
    target: str
    references: List[str]


@dataclass
class SampleResult:
    dataset_name: str
    task: str
    language_code: str
    language_name: str
    sample_index: int
    prompt: str
    prediction: str
    reference: str
    bleu: float
    rouge_l: float
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
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
    device = device.strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device


def _apply_perf_settings(device: str, enable_tf32: bool, enable_cudnn_benchmark: bool) -> None:
    if not device.startswith("cuda"):
        return
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True


@torch.inference_mode()
def _auto_tune_batch_size(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_batch: int,
    step: int,
) -> int:
    # Auto-tune feature removed; keep function stub to avoid accidental use.
    raise RuntimeError("_auto_tune_batch_size was removed; control batch size with --batch-size")


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_example_value(example: Dict[str, Any], candidates: Sequence[str]) -> Any:
    key_lookup = {k.lower(): k for k in example.keys()}
    for candidate in candidates:
        if candidate.lower() in key_lookup:
            return example[key_lookup[candidate.lower()]]
    return None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _as_references(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        refs = [str(x).strip() for x in value if str(x).strip()]
        return refs
    return [str(value).strip()] if str(value).strip() else []


def _extract_nlg_record(dataset_name: str, example: Dict[str, Any]) -> Optional[NLGRecord]:
    spec = TASK_SPECS[dataset_name]
    source = _get_example_value(example, spec["input_candidates"])
    target = _get_example_value(example, spec["target_candidates"])

    if source is None or target is None:
        return None

    source_text = _normalize_text(source)
    refs = _as_references(target)
    if not source_text or not refs:
        return None

    language_value = _get_example_value(example, ["language", "lang", "locale", "iso"])
    language_code = str(language_value).strip() if language_value is not None else "unknown"
    language_name = LANGUAGE_CODE_MAP.get(language_code, language_code)

    return NLGRecord(
        dataset_name=dataset_name,
        task=spec["task"],
        language_code=language_code,
        language_name=language_name,
        source=source_text,
        target=refs[0],
        references=refs,
    )


def _resolve_split(dataset: Dataset, requested: str) -> str:
    try:
        available = list(dataset.keys())
    except Exception:
        return requested

    if requested in available:
        return requested
    for fallback in ("test", "validation", "train"):
        if fallback in available:
            return fallback
    return requested


def _iter_dataset_splits(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str],
    token: Optional[str],
    trust_remote_code: bool,
):
    try:
        configs = get_dataset_config_names(dataset_name)
    except Exception:
        configs = []

    configs = [c for c in configs if c in INDIC_LANGS or c in {"od", "or"}]

    if configs:
        for config in configs:
            try:
                loaded = load_dataset(
                    dataset_name,
                    config,
                    cache_dir=cache_dir,
                    token=token if token else True,
                    trust_remote_code=trust_remote_code,
                )
                resolved = _resolve_split(loaded, split)
                yield config, loaded[resolved]
            except RuntimeError as exc:
                if _is_dataset_script_error(exc):
                    yield from _iter_dataset_splits_from_files(
                        dataset_name=dataset_name,
                        split=split,
                        cache_dir=cache_dir,
                        token=token,
                        local_files_only=False,
                    )
                    return
                raise
        return

    try:
        loaded = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            token=token if token else True,
            trust_remote_code=trust_remote_code,
        )
        resolved = _resolve_split(loaded, split)
        yield "all", loaded[resolved]
    except RuntimeError as exc:
        if _is_dataset_script_error(exc):
            yield from _iter_dataset_splits_from_files(
                dataset_name=dataset_name,
                split=split,
                cache_dir=cache_dir,
                token=token,
                local_files_only=False,
            )
            return
        raise


def _is_dataset_script_error(exc: Exception) -> bool:
    msg = str(exc)
    return "Dataset scripts are no longer supported" in msg


def _infer_language_from_path(path: str) -> Optional[str]:
    parts = re.split(r"[\\/._-]+", path.lower())
    for token in parts:
        if token in INDIC_LANGS:
            return token
        if token == "od":
            return "or"
    return None


def _infer_split_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    if "train" in lower:
        return "train"
    if "test" in lower:
        return "test"
    if "dev" in lower:
        return "validation"
    if "validation" in lower or "valid" in lower or "dev" in lower:
        return "validation"
    return None


def _iter_dataset_splits_from_files(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str],
    token: Optional[str],
    local_files_only: bool,
):
    if list_repo_files is None or hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is required to load IndicNLG datasets that still use scripts."
        )

    repo_files = list_repo_files(dataset_name, repo_type="dataset")
    raw_files = [
        f
        for f in repo_files
        if f.endswith((".parquet", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".zip"))
    ]
    if not raw_files:
        raise RuntimeError(f"No data files found in dataset repo: {dataset_name}")

    grouped: Dict[str, Dict[str, List[str]]] = {}
    extracted_root = tempfile.mkdtemp(prefix="indicnlg_extract_")

    for path in raw_files:
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=path,
            cache_dir=cache_dir,
            token=token if token else None,
            local_files_only=local_files_only,
        )

        if path.endswith(".zip"):
            with zipfile.ZipFile(local_path, "r") as zf:
                for member in zf.namelist():
                    if not member.endswith((".json", ".jsonl", ".csv", ".tsv", ".txt")):
                        continue
                    lang = _infer_language_from_path(member) or _infer_language_from_path(path) or "all"
                    split_name = _infer_split_from_path(member) or _infer_split_from_path(path) or "train"
                    target_dir = os.path.join(extracted_root, dataset_name.replace("/", "_"))
                    os.makedirs(target_dir, exist_ok=True)
                    out_path = os.path.join(target_dir, os.path.basename(member))
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    grouped.setdefault(lang, {}).setdefault(split_name, []).append(out_path)
            continue

        lang = _infer_language_from_path(path) or "all"
        split_name = _infer_split_from_path(path) or "train"
        grouped.setdefault(lang, {}).setdefault(split_name, []).append(local_path)

    selected_langs = [lang for lang in grouped if lang in INDIC_LANGS]
    if not selected_langs and "all" in grouped:
        selected_langs = ["all"]

    for lang in sorted(selected_langs):
        split_files = grouped.get(lang, {})
        if not split_files:
            continue

        resolved_split = split if split in split_files else None
        if resolved_split is None:
            for fallback in ("test", "validation", "train"):
                if fallback in split_files:
                    resolved_split = fallback
                    break
        if resolved_split is None:
            continue

        local_paths = split_files[resolved_split]
        sample_path = local_paths[0]
        if sample_path.endswith(".parquet"):
            file_format = "parquet"
        elif sample_path.endswith((".json", ".jsonl")):
            file_format = "json"
        elif sample_path.endswith((".csv", ".tsv")):
            file_format = "csv"
        else:
            file_format = "text"

        load_kwargs: Dict[str, Any] = {
            "data_files": {resolved_split: local_paths},
            "cache_dir": cache_dir,
        }
        if file_format == "csv" and sample_path.endswith(".tsv"):
            load_kwargs["delimiter"] = "\t"

        loaded = load_dataset(file_format, **load_kwargs)
        yield lang, loaded[resolved_split]


def _format_prompt(record: NLGRecord) -> str:
    spec = TASK_SPECS[record.dataset_name]
    if record.dataset_name == "ai4bharat/IndicQuestionGeneration":
        answer = ""
        return spec["prompt"].format(source=record.source, answer=answer)
    return spec["prompt"].format(source=record.source)


def _format_prompt_qg(example: Dict[str, Any], record: NLGRecord) -> str:
    spec = TASK_SPECS[record.dataset_name]
    answer = _get_example_value(example, spec["answer_candidates"])
    answer_refs = _as_references(answer)
    if not answer_refs:
        return spec["prompt_no_answer"].format(source=record.source)
    return spec["prompt"].format(source=record.source, answer=_normalize_text(answer_refs[0]))


def _apply_chat_template(tokenizer, prompt: str, enabled: bool) -> str:
    if not enabled:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for j, token_b in enumerate(b, start=1):
            temp = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_l_f1(pred: str, refs: Sequence[str]) -> float:
    pred_tokens = pred.split()
    if not pred_tokens or not refs:
        return 0.0

    best = 0.0
    for ref in refs:
        ref_tokens = str(ref).split()
        if not ref_tokens:
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        score = (2.0 * precision * recall) / (precision + recall)
        best = max(best, score)
    return best


def _bleu_score(pred: str, refs: Sequence[str], max_n: int = 4) -> float:
    pred_tokens = pred.split()
    if not pred_tokens or not refs:
        return 0.0

    ref_tokens_list = [str(r).split() for r in refs if str(r).strip()]
    if not ref_tokens_list:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngram_counts(pred_tokens, n)
        max_ref_counts: Dict[Tuple[str, ...], int] = {}
        for ref_tokens in ref_tokens_list:
            ref_counts = _ngram_counts(ref_tokens, n)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

        overlap = 0
        total = 0
        for ngram, count in pred_ngrams.items():
            overlap += min(count, max_ref_counts.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(overlap / total)

    # Brevity penalty
    pred_len = len(pred_tokens)
    ref_lens = [len(r) for r in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda rlen: (abs(rlen - pred_len), rlen))
    if pred_len == 0:
        bp = 0.0
    elif pred_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / max(pred_len, 1)))

    # Geometric mean of precisions with smoothing
    smooth = 1e-9
    log_sum = 0.0
    for p in precisions:
        log_sum += (1.0 / max_n) * math.log(p + smooth)
    bleu = bp * math.exp(log_sum)
    return float(bleu)


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _load_model(
    model_path: str,
    model_type: str,
    base_model: str,
    device: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    local_files_only: bool,
):
    if model_type == "base":
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    if model_type == "mhc-v2":
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

    if model_type == "mhc-v1":
        from src.conversion import load_mhc_model

        model, _ = load_mhc_model(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
        )
        return model

    last_error: Optional[Exception] = None

    for candidate in ("mhc-v2", "mhc-v1", "base"):
        try:
            return _load_model(
                model_path=model_path,
                model_type=candidate,
                base_model=base_model,
                device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            last_error = exc
            print(f"Auto load failed for {candidate}: {exc}")

    raise RuntimeError(f"Unable to load model from {model_path}: {last_error}")


@torch.inference_mode()
def _generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[str, int, int, float]:
    formatted = prompt
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

    prompt_tokens = int(inputs["input_ids"].shape[1])
    output_tokens = int(generated.shape[0])
    total_tokens = prompt_tokens + output_tokens
    latency_ms = (end - start) * 1000.0

    return prediction, prompt_tokens, output_tokens, latency_ms


@torch.inference_mode()
def _generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[int], List[int], float]:
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})
    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    input_lengths = inputs["input_ids"].shape[1]
    predictions: List[str] = []
    prompt_tokens_list: List[int] = []
    output_tokens_list: List[int] = []
    for row in outputs:
        generated = row[input_lengths:]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        prompt_tokens_list.append(int(input_lengths))
        output_tokens_list.append(int(generated.shape[0]))

    latency_ms = (end - start) * 1000.0
    return predictions, prompt_tokens_list, output_tokens_list, latency_ms


def _write_output(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"indicnlg_benchmark_{timestamp}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark results: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a model on IndicNLG (5 tasks x 11 languages)")
    parser.add_argument("--model", required=True, help="Model path or HF id")
    parser.add_argument("--model-type", default="auto", choices=["auto", "base", "mhc-v1", "mhc-v2"])
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Base model for mHC loaders")

    parser.add_argument("--split", default="test", help="Preferred split: test/validation/train")
    parser.add_argument("--max-samples-per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--enable-cudnn-benchmark", action="store_true")
    

    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hf-token", default=None, help="HF token if needed")
    parser.add_argument("--output-dir", default="output/benchmark_indicnlg")

    args = parser.parse_args()

    random.seed(args.seed)
    device = _resolve_device(args.device)
    torch_dtype = _parse_dtype(args.dtype)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    _apply_perf_settings(device, args.enable_tf32, args.enable_cudnn_benchmark)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if args.model_type != "base" else args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = _load_model(
        model_path=args.model,
        model_type=args.model_type,
        base_model=args.base_model,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()

    # Auto-tuning removed: batch size is controlled via --batch-size directly.

    sample_results: List[SampleResult] = []
    summary: Dict[str, Dict[str, float]] = {}
    language_summary: Dict[str, Dict[str, float]] = {}

    print("Preparing IndicNLG records...")
    for dataset_name in TASK_SPECS:
        spec = TASK_SPECS[dataset_name]
        task_name = spec["task"]
        task_records = []

        for config, dataset in _iter_dataset_splits(
            dataset_name,
            args.split,
            args.cache_dir,
            hf_token,
            args.trust_remote_code,
        ):
            lang_code = config if config != "all" else "unknown"
            lang_name = LANGUAGE_CODE_MAP.get(lang_code, lang_code)

            dataset_iter = dataset
            if hasattr(dataset_iter, "shuffle"):
                dataset_iter = dataset_iter.shuffle(seed=args.seed)
            lang_counts: Dict[str, int] = {}
            pending: List[Tuple[NLGRecord, Dict[str, Any], str]] = []
            for idx, example in enumerate(dataset_iter):
                record = _extract_nlg_record(dataset_name, example)
                if record is None:
                    continue

                if config == "all":
                    if record.language_code not in INDIC_LANGS and record.language_code not in {"od", "or"}:
                        continue
                else:
                    record.language_code = lang_code
                    record.language_name = lang_name

                lang_key = record.language_code
                count = lang_counts.get(lang_key, 0)
                if args.max_samples_per_language > 0 and count >= args.max_samples_per_language:
                    continue

                if dataset_name == "ai4bharat/IndicQuestionGeneration":
                    prompt = _format_prompt_qg(example, record)
                else:
                    prompt = _format_prompt(record)

                prompt = _apply_chat_template(tokenizer, prompt, args.apply_chat_template)
                pending.append((record, example, prompt))

                if len(pending) < max(1, args.batch_size):
                    continue

                prompts = [p[2] for p in pending]
                predictions, prompt_tokens_list, output_tokens_list, latency_ms = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                per_sample_latency = latency_ms / max(len(pending), 1)
                for (rec, _, prompt_text), pred, p_tokens, o_tokens in zip(
                    pending, predictions, prompt_tokens_list, output_tokens_list
                ):
                    bleu = _bleu_score(pred, rec.references)
                    rouge_l = _rouge_l_f1(pred, rec.references)

                    sample_results.append(
                        SampleResult(
                            dataset_name=dataset_name,
                            task=task_name,
                            language_code=rec.language_code,
                            language_name=rec.language_name,
                            sample_index=len(sample_results),
                            prompt=prompt_text,
                            prediction=pred,
                            reference=rec.references[0] if rec.references else rec.target,
                            bleu=bleu,
                            rouge_l=rouge_l,
                            prompt_tokens=p_tokens,
                            output_tokens=o_tokens,
                            total_tokens=p_tokens + o_tokens,
                            latency_ms=per_sample_latency,
                        )
                    )

                    task_records.append((rec, bleu, rouge_l, per_sample_latency, p_tokens, o_tokens))
                    lang_counts[rec.language_code] = lang_counts.get(rec.language_code, 0) + 1

                pending = []

                if args.max_samples_per_language > 0:
                    if config != "all" and lang_counts[lang_key] >= args.max_samples_per_language:
                        break
                    if config == "all" and len(lang_counts) >= len(INDIC_LANGS):
                        if all(lang_counts.get(code, 0) >= args.max_samples_per_language for code in INDIC_LANGS):
                            break

            if pending:
                prompts = [p[2] for p in pending]
                predictions, prompt_tokens_list, output_tokens_list, latency_ms = _generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                per_sample_latency = latency_ms / max(len(pending), 1)
                for (rec, _, prompt_text), pred, p_tokens, o_tokens in zip(
                    pending, predictions, prompt_tokens_list, output_tokens_list
                ):
                    bleu = _bleu_score(pred, rec.references)
                    rouge_l = _rouge_l_f1(pred, rec.references)

                    sample_results.append(
                        SampleResult(
                            dataset_name=dataset_name,
                            task=task_name,
                            language_code=rec.language_code,
                            language_name=rec.language_name,
                            sample_index=len(sample_results),
                            prompt=prompt_text,
                            prediction=pred,
                            reference=rec.references[0] if rec.references else rec.target,
                            bleu=bleu,
                            rouge_l=rouge_l,
                            prompt_tokens=p_tokens,
                            output_tokens=o_tokens,
                            total_tokens=p_tokens + o_tokens,
                            latency_ms=per_sample_latency,
                        )
                    )

                    task_records.append((rec, bleu, rouge_l, per_sample_latency, p_tokens, o_tokens))
                    lang_counts[rec.language_code] = lang_counts.get(rec.language_code, 0) + 1

        if not task_records:
            print(f"No usable records for {dataset_name}")
            continue

        task_bleu = sum(x[1] for x in task_records) / len(task_records)
        task_rouge = sum(x[2] for x in task_records) / len(task_records)
        task_latency = sum(x[3] for x in task_records) / len(task_records)
        task_prompt_tokens = sum(x[4] for x in task_records) / len(task_records)
        task_output_tokens = sum(x[5] for x in task_records) / len(task_records)

        summary[task_name] = {
            "samples": len(task_records),
            "bleu": task_bleu,
            "rouge_l": task_rouge,
            "avg_latency_ms": task_latency,
            "avg_prompt_tokens": task_prompt_tokens,
            "avg_output_tokens": task_output_tokens,
        }

    if sample_results:
        for lang in INDIC_LANGS:
            lang_results = [x for x in sample_results if x.language_code == lang]
            if not lang_results:
                continue
            language_summary[lang] = {
                "language": LANGUAGE_CODE_MAP.get(lang, lang),
                "samples": len(lang_results),
                "bleu": sum(x.bleu for x in lang_results) / len(lang_results),
                "rouge_l": sum(x.rouge_l for x in lang_results) / len(lang_results),
                "avg_latency_ms": sum(x.latency_ms for x in lang_results) / len(lang_results),
                "avg_prompt_tokens": sum(x.prompt_tokens for x in lang_results) / len(lang_results),
                "avg_output_tokens": sum(x.output_tokens for x in lang_results) / len(lang_results),
            }

    payload = {
        "config": {
            "model": args.model,
            "model_type": args.model_type,
            "base_model": args.base_model,
            "split": args.split,
            "max_samples_per_language": args.max_samples_per_language,
            "seed": args.seed,
            "device": device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "apply_chat_template": args.apply_chat_template,
            "batch_size": args.batch_size,
            
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "task_summary": summary,
        "language_summary": language_summary,
        "samples": [asdict(x) for x in sample_results],
    }

    _write_output(Path(args.output_dir), payload)


if __name__ == "__main__":
    main()
