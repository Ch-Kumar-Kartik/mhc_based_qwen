#!/usr/bin/env python3
"""Run parity diagnostics between original Qwen3 and converted mHC models.

Checks implemented:
1. Postfix parity: final hidden states parity (post-model-norm representation).
2. LM-head parity: final logits parity.
3. Layerwise LM-head parity: parity of LM-head projections on hidden states
   from each returned hidden-state slot.

Results are saved to a JSON report.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversion import load_mhc_model
from src.conversionV2 import load_mhc_model_v2


LOGGER = logging.getLogger(__name__)


def parse_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compute parity statistics between tensors."""
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    mse = torch.mean((a_f - b_f) ** 2)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rmse": float(torch.sqrt(mse).item()),
    }


def collapse_mhc_hidden_state(mhc_model: torch.nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """Collapse mHC multi-stream hidden state to a standard (B, S, C) tensor."""
    if hidden_state.dim() == 3:
        return hidden_state

    if hidden_state.dim() != 4:
        raise ValueError(f"Unsupported hidden-state rank for mHC collapse: {hidden_state.dim()}")

    reduce_fn = getattr(getattr(mhc_model, "model", None), "reduce_fn", None)
    if reduce_fn is not None:
        try:
            return reduce_fn(hidden_state)
        except Exception as err:  # pragma: no cover - fallback path
            LOGGER.warning("reduce_fn collapse failed (%s); falling back to stream mean", err)

    return hidden_state.mean(dim=-2)


def load_converted_model(
    model_path: str,
    base_model: str,
    device: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Tuple[torch.nn.Module, str]:
    """Load converted model, trying V2 then V1."""
    try:
        model, _ = load_mhc_model_v2(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return model, "v2"
    except Exception as v2_err:
        LOGGER.warning("Failed to load converted model as V2, trying V1: %s", v2_err)

    model, _ = load_mhc_model(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        base_model=base_model,
    )
    return model, "v1"


def run_checks(args: argparse.Namespace) -> Dict[str, Any]:
    torch_dtype = parse_dtype(args.dtype)

    LOGGER.info("Loading original model: %s", args.original)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )

    LOGGER.info("Loading tokenizer: %s", args.original)
    tokenizer = AutoTokenizer.from_pretrained(
        args.original,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading converted model: %s", args.converted)
    converted_model, converted_variant = load_converted_model(
        model_path=args.converted,
        base_model=args.original,
        device=args.device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    original_model.eval()
    converted_model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    LOGGER.info("Running forwards with hidden-state outputs")
    with torch.no_grad():
        original_outputs = original_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        converted_outputs = converted_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    postfix_parity = tensor_stats(
        original_outputs.hidden_states[-1],
        converted_outputs.hidden_states[-1],
    )

    lm_head_parity = tensor_stats(
        original_outputs.logits,
        converted_outputs.logits,
    )

    layerwise: List[Dict[str, Any]] = []
    original_hs = list(original_outputs.hidden_states)
    converted_hs = list(converted_outputs.hidden_states)
    n_layers_compared = min(len(original_hs), len(converted_hs))

    for idx in range(n_layers_compared):
        orig_h = original_hs[idx]
        conv_h = collapse_mhc_hidden_state(converted_model, converted_hs[idx])

        if orig_h.shape != conv_h.shape:
            layerwise.append(
                {
                    "index": idx,
                    "status": "skipped_shape_mismatch",
                    "original_shape": list(orig_h.shape),
                    "converted_shape": list(conv_h.shape),
                }
            )
            continue

        # Restrict to recent positions to keep layerwise LM-head comparisons fast.
        if args.layerwise_tokens > 0:
            orig_h = orig_h[:, -args.layerwise_tokens :, :]
            conv_h = conv_h[:, -args.layerwise_tokens :, :]

        with torch.no_grad():
            orig_layer_logits = original_model.lm_head(orig_h).float()
            conv_layer_logits = converted_model.lm_head(conv_h).float()

        stats = tensor_stats(orig_layer_logits, conv_layer_logits)
        stats.update(
            {
                "index": idx,
                "status": "ok",
                "positions_compared": int(orig_h.shape[1]),
            }
        )
        layerwise.append(stats)

    lm_head_weight_parity: Dict[str, Any]
    if (
        hasattr(original_model, "lm_head")
        and hasattr(converted_model, "lm_head")
        and original_model.lm_head.weight.shape == converted_model.lm_head.weight.shape
    ):
        lm_head_weight_parity = tensor_stats(
            original_model.lm_head.weight,
            converted_model.lm_head.weight,
        )
        lm_head_weight_parity["status"] = "ok"
    else:
        lm_head_weight_parity = {
            "status": "skipped_shape_mismatch",
            "original_shape": list(getattr(original_model.lm_head.weight, "shape", [])),
            "converted_shape": list(getattr(converted_model.lm_head.weight, "shape", [])),
        }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_model": args.original,
        "converted_model": str(Path(args.converted).resolve()),
        "converted_variant": converted_variant,
        "device": args.device,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "prompt_token_count": int(inputs["input_ids"].shape[1]),
        "checks": {
            "postfix_parity": postfix_parity,
            "lm_head_parity": lm_head_parity,
            "lm_head_weight_parity": lm_head_weight_parity,
            "layerwise_lm_head_parity": {
                "num_hidden_state_slots_compared": n_layers_compared,
                "per_slot": layerwise,
            },
        },
    }

    return report


def build_text_report(report: Dict[str, Any]) -> str:
    """Build a human-readable TXT report from the JSON structure."""
    checks = report["checks"]
    lines: List[str] = []
    lines.append("PARITY REPORT")
    lines.append("=" * 80)
    lines.append(f"timestamp_utc: {report['timestamp_utc']}")
    lines.append(f"original_model: {report['original_model']}")
    lines.append(f"converted_model: {report['converted_model']}")
    lines.append(f"converted_variant: {report['converted_variant']}")
    lines.append(f"device: {report['device']}")
    lines.append(f"dtype: {report['dtype']}")
    lines.append(f"prompt: {report['prompt']}")
    lines.append(f"prompt_token_count: {report['prompt_token_count']}")
    lines.append("")

    lines.append("POSTFIX PARITY")
    lines.append("-" * 80)
    lines.append(f"max_abs_diff: {checks['postfix_parity']['max_abs_diff']:.6e}")
    lines.append(f"mean_abs_diff: {checks['postfix_parity']['mean_abs_diff']:.6e}")
    lines.append(f"rmse: {checks['postfix_parity']['rmse']:.6e}")
    lines.append("")

    lines.append("LM HEAD PARITY")
    lines.append("-" * 80)
    lines.append(f"max_abs_diff: {checks['lm_head_parity']['max_abs_diff']:.6e}")
    lines.append(f"mean_abs_diff: {checks['lm_head_parity']['mean_abs_diff']:.6e}")
    lines.append(f"rmse: {checks['lm_head_parity']['rmse']:.6e}")
    lines.append("")

    lines.append("LM HEAD WEIGHT PARITY")
    lines.append("-" * 80)
    lm_w = checks["lm_head_weight_parity"]
    lines.append(f"status: {lm_w['status']}")
    if lm_w["status"] == "ok":
        lines.append(f"max_abs_diff: {lm_w['max_abs_diff']:.6e}")
        lines.append(f"mean_abs_diff: {lm_w['mean_abs_diff']:.6e}")
        lines.append(f"rmse: {lm_w['rmse']:.6e}")
    else:
        lines.append(f"original_shape: {lm_w.get('original_shape', [])}")
        lines.append(f"converted_shape: {lm_w.get('converted_shape', [])}")
    lines.append("")

    lines.append("LAYERWISE LM HEAD PARITY")
    lines.append("-" * 80)
    layerwise = checks["layerwise_lm_head_parity"]
    lines.append(
        f"num_hidden_state_slots_compared: {layerwise['num_hidden_state_slots_compared']}"
    )
    for slot in layerwise["per_slot"]:
        idx = slot["index"]
        status = slot["status"]
        if status == "ok":
            lines.append(
                "slot={idx} status={status} positions={positions} "
                "max_abs_diff={max_abs_diff:.6e} mean_abs_diff={mean_abs_diff:.6e} rmse={rmse:.6e}".format(
                    idx=idx,
                    status=status,
                    positions=slot["positions_compared"],
                    max_abs_diff=slot["max_abs_diff"],
                    mean_abs_diff=slot["mean_abs_diff"],
                    rmse=slot["rmse"],
                )
            )
        else:
            lines.append(
                f"slot={idx} status={status} original_shape={slot.get('original_shape')} converted_shape={slot.get('converted_shape')}"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run postfix parity, LM-head parity, and layerwise LM-head parity and save JSON report.",
    )
    parser.add_argument(
        "--original",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Original base model name/path",
    )
    parser.add_argument(
        "--converted",
        type=str,
        required=True,
        help="Path to converted mHC checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path for saved report",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=None,
        help="Optional TXT output path. Defaults to the JSON output path with .txt extension.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Prompt used for parity checks",
    )
    parser.add_argument(
        "--layerwise-tokens",
        type=int,
        default=1,
        help="How many trailing token positions to use for each layerwise LM-head parity projection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading and evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote code execution when loading from HF",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load from local cache/files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    report = run_checks(args)

    output_path = Path(args.output)
    output_txt_path = Path(args.output_txt) if args.output_txt else output_path.with_suffix(".txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text(build_text_report(report), encoding="utf-8")

    checks = report["checks"]
    LOGGER.info("Saved parity report: %s", output_path)
    LOGGER.info("Saved parity TXT report: %s", output_txt_path)
    LOGGER.info("postfix_parity.max_abs_diff: %.6e", checks["postfix_parity"]["max_abs_diff"])
    LOGGER.info("lm_head_parity.max_abs_diff: %.6e", checks["lm_head_parity"]["max_abs_diff"])


if __name__ == "__main__":
    main()
