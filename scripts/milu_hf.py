"""Helpers for scoring MILU using Hugging Face models.

Expose small helpers to load tokenizer/model and compute option log-probabilities
for the ranking-based MILU evaluation.
"""
from __future__ import annotations

from typing import Tuple

import torch


def load_tokenizer_and_model(model_name_or_path: str, trust_remote_code: bool = False, local_files_only: bool = False, torch_dtype=None, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def option_logprob(
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
