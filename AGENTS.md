# AGENTS.md

## Project Intent

This repository implements and studies Manifold-Constrained Hyper-Connections (mHC) on top of Qwen3-0.6B, with both V1 and V2 implementations.

Primary goals:
- Convert base Qwen3 models into mHC / mHC V2 variants.
- Preserve stable initialization and validate equivalence (or bounded divergence in V2).
- Train on multilingual pretokenized datasets (notably Sangraha shards).
- Analyze learned stream-mixing behavior after training.

## Research Map

Core papers reflected in code:
- mHC / manifold-constrained residual mixing
- Hyper-connections and frac-connections (V2 implementation)

Main source folders:
- src/: model internals, conversion, Sinkhorn, stream ops, config dataclasses.
- scripts/: conversion/training/validation/analysis utilities.
- tests/: unit tests for conversion, Sinkhorn, stream ops, and equivalence mechanics.
- configs/: YAML-based training defaults for V1/V2 and AMD.
- data/: local packed datasets (save_to_disk outputs and sharded layouts).

## Code Architecture

Key modules and responsibilities:
- src/config.py
  - Defines MHCConfig and MHCTrainingConfig.
  - Encodes equivalence-oriented initialization defaults, including b_pre_init derived from log(1/(n-1)).
- src/sinkhorn.py
  - Projects residual mixing matrices into doubly stochastic structure.
  - This is a core stability primitive.
- src/stream_ops.py
  - Stream expand/collapse helpers used by equivalence logic and model flow.
- src/mhc_layer.py, src/qwen3_mhc_model.py
  - V1 mHC implementation.
- src/mhc_layerV2.py, src/qwen3_mhc_modelV2.py
  - V2 mHC implementation based on hyper-connections style modules.
- src/conversion.py, src/conversionV2.py
  - Build mHC models from base Qwen3 checkpoints and transfer weights.

Top-level and script entrypoints:
- train.py
  - Main root training script; currently uses V2 conversion path.
  - Loads packed datasets from disk and supports checkpoint save/prune.
- scripts/train_v2.py
  - Config-driven V2 trainer with step LR decay and runtime tuning.
- scripts/convert_to_mhc_v2.py
  - CLI conversion to V2 model.
- scripts/validate_equivalence.py
  - Validation CLI (tries V2 first, falls back to V1).
- prepare_dataset.py, prepare_dataset_streaming.py, prepare_dataset_verified.py
  - Dataset prep/tokenization/packing pipelines.

## Data Layout and Assumptions

Expected dataset patterns:
- Split-based layout:
  - <tokenized-dir>/train
  - optional <tokenized-dir>/validation
- Shard-based layout:
  - <tokenized-dir>/shard_*

Important practical detail:
- A shard folder is only valid for load_from_disk when it contains Hugging Face metadata files (for example state.json and dataset_info.json). Arrow-only partial shards can exist and must be skipped or repaired.

## Verified Workflows

Conversion (V2):
- python -m scripts.convert_to_mhc_v2 --model Qwen/Qwen3-0.6B --output ./output/qwen3_mhc_v2_converted

Equivalence check:
- python -m scripts.validate_equivalence --original Qwen/Qwen3-0.6B --mhc ./output/qwen3_mhc_v2_converted

Root training example (argparse-based):
- python train.py --tokenized-dir data/sangraha_packed --output-dir output/qwen3_mhc_v2 --base-model Qwen/Qwen3-0.6B --n-streams 4 --sinkhorn-iters 20 --batch-size 2 --gradient-accumulation-steps 8 --total-steps 50000

Config-driven V2 training:
- python -m scripts.train_v2 --config configs/qwen3_0.6b_mhc_v2.yaml

Tests:
- pytest
- pytest -v

## Safety and Stability Guardrails

Treat these zones as high-sensitivity:
- src/sinkhorn.py
  - Do not alter normalization/projection logic without mathematical justification and test updates.
- src/config.py initialization constants/derivations
  - Changes can silently break warm-start behavior.
- src/stream_ops.py
  - Preserve expand/collapse identity properties.
- Conversion logic in src/conversion.py and src/conversionV2.py
  - Ensure checkpoint compatibility and output behavior are revalidated after edits.

Minimum validation after touching model math:
- pytest
- pytest tests/test_sinkhorn.py -v
- pytest tests/test_equivalence.py -v
- python -m scripts.validate_equivalence --mhc <path>

## Platform Notes (Observed)

Windows + large memory-mapped datasets:
- High dataloader worker counts on HDD can cause severe I/O thrash and apparent stalls.
- Safer defaults: low worker count (0..2), low prefetch, avoid persistent workers if disk is bottlenecked.

Offline model loading:
- Qwen3 paths using trust_remote_code=True may trigger extra remote file fetches.
- Prefer explicit offline-safe options where possible (trust_remote_code=False, local_files_only=True) for disconnected runs.

Gradient checkpointing compatibility:
- Ensure model classes use the current Transformers API expectations and disable use_cache during checkpointed training.

## Agent Working Rules for This Repo

When making changes as an autonomous coding agent:
1. Identify whether the target path is V1 or V2 before editing.
2. Preserve CLI compatibility for existing scripts unless change is explicitly requested.
3. Prefer minimal, localized patches over broad refactors.
4. If touching dataset loading, keep behavior robust to partial/corrupt shards.
5. Run targeted tests first, then broader tests if model logic changed.
6. For performance tuning claims, include reproducible measurement scripts/commands.
7. Never change initialization math and Sinkhorn behavior in the same patch unless the user explicitly requests algorithmic redesign.

## Open Research Questions in This Codebase

Potential directions for future contributors:
- V1 vs V2 comparison under identical training settings (loss curves, speed, memory).
- Quantifying stream specialization across layers over long runs.
- Stability/quality impact of Sinkhorn iteration count changes.
- Interaction between frac-connections parameters and multilingual corpus composition.
- Robustness of conversion/validation under pure offline operation.

## Quick Orientation for New Contributors

If you are new, read in this order:
1. README.md
2. SPEC.md
3. src/config.py
4. src/mhc_layerV2.py and src/qwen3_mhc_modelV2.py
5. src/conversionV2.py
6. train.py and scripts/train_v2.py
7. tests/test_equivalence.py and tests/test_sinkhorn.py
