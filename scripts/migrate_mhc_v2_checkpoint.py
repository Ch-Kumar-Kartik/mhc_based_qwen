#!/usr/bin/env python3
"""Migrate an existing mHC V2 checkpoint to a modified mHC V2 architecture.

Typical use case:
- Enable add_stream_embed on an older checkpoint trained with add_stream_embed=false.

The script performs a partial state-dict load (strict=False) so newly introduced
parameters are initialized from the target model init while existing compatible
weights are reused from the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from src.qwen3_mhc_modelV2 import Qwen3MHCConfigV2, Qwen3MHCForCausalLMV2


logger = logging.getLogger(__name__)


def _parse_torch_dtype(dtype_name: str, source_cfg_dtype: Optional[str]) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16

    # auto
    if isinstance(source_cfg_dtype, str):
        source_cfg_dtype = source_cfg_dtype.lower()
        if "bfloat16" in source_cfg_dtype:
            return torch.bfloat16
        if "float16" in source_cfg_dtype or "fp16" in source_cfg_dtype:
            return torch.float16

    return torch.float32


def _load_tokenizer(source_path: str, base_model: str, local_files_only: bool):
    try:
        tok = AutoTokenizer.from_pretrained(source_path, local_files_only=local_files_only)
        logger.info("Loaded tokenizer from source checkpoint")
        return tok
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
        logger.info("Loaded tokenizer from base model")
        return tok


def main():
    parser = argparse.ArgumentParser(description="Migrate mHC V2 checkpoint to modified architecture")
    parser.add_argument("--source", required=True, help="Source mHC V2 checkpoint directory")
    parser.add_argument("--output", required=True, help="Output directory for migrated checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="Fallback tokenizer source")

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Output checkpoint dtype",
    )

    parser.add_argument(
        "--add-stream-embed",
        action="store_true",
        default=None,
        help="Enable stream embeddings in target config",
    )
    parser.add_argument(
        "--no-add-stream-embed",
        dest="add_stream_embed",
        action="store_false",
        help="Disable stream embeddings in target config",
    )

    parser.add_argument("--device", default="cpu", help="Device for migration load/build")
    parser.add_argument("--local-files-only", action="store_true", help="Disable remote Hub fetches")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logs")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading source checkpoint: {source_path}")
    source_model = Qwen3MHCForCausalLMV2.from_pretrained(
        str(source_path),
        torch_dtype=torch.float32,
        device_map=args.device,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    source_model.eval()

    source_cfg = source_model.config
    source_cfg_dict = source_cfg.to_dict()

    if args.add_stream_embed is not None:
        source_cfg_dict["add_stream_embed"] = bool(args.add_stream_embed)

    target_cfg = Qwen3MHCConfigV2(**source_cfg_dict)

    logger.info("Building target model and loading transferable weights")
    target_model = Qwen3MHCForCausalLMV2(target_cfg)
    load_result = target_model.load_state_dict(source_model.state_dict(), strict=False)

    output_dtype = _parse_torch_dtype(args.dtype, getattr(source_cfg, "dtype", None))
    target_model = target_model.to(dtype=output_dtype)

    logger.info(f"Saving migrated checkpoint to: {output_path}")
    target_model.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = _load_tokenizer(str(source_path), args.base_model, args.local_files_only)
    tokenizer.save_pretrained(str(output_path))

    if getattr(source_model, "generation_config", None) is not None:
        source_model.generation_config.save_pretrained(str(output_path))

    report = {
        "source": str(source_path),
        "output": str(output_path),
        "source_config": {
            "n_streams": getattr(source_cfg, "n_streams", None),
            "sinkhorn_iters": getattr(source_cfg, "sinkhorn_iters", None),
            "add_stream_embed": getattr(source_cfg, "add_stream_embed", None),
        },
        "target_config": {
            "n_streams": getattr(target_cfg, "n_streams", None),
            "sinkhorn_iters": getattr(target_cfg, "sinkhorn_iters", None),
            "add_stream_embed": getattr(target_cfg, "add_stream_embed", None),
        },
        "output_dtype": str(output_dtype),
        "missing_keys_count": len(load_result.missing_keys),
        "unexpected_keys_count": len(load_result.unexpected_keys),
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
    }

    report_path = output_path / "migration_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Migration complete")
    logger.info(f"Missing keys: {len(load_result.missing_keys)}")
    logger.info(f"Unexpected keys: {len(load_result.unexpected_keys)}")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
