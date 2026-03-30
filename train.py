#!/usr/bin/env python3
import argparse
from collections import deque
import importlib.util
import logging
import math
import os
import re
import shutil
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
import time
from typing import Optional

# Reduce allocator fragmentation on long-running CUDA jobs unless user overrides it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Improves kernel scheduling consistency for many transformer workloads.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_scheduler
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm

from src.conversionV2 import load_mhc_model_v2, convert_qwen3_to_mhc_v2, count_parameters_v2


def _has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def _get_underlying_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _set_original_parameter_trainability(model, requires_grad: bool) -> bool:
    base_model = _get_underlying_model(model)
    if not hasattr(base_model, "get_original_parameters"):
        return False

    for param in base_model.get_original_parameters():
        param.requires_grad = requires_grad
    return True


def _get_monitoring_stats(model):
    base_model = _get_underlying_model(model)
    if hasattr(base_model, "get_monitoring_stats"):
        return base_model.get_monitoring_stats()
    return {}


def setup_logging(output_dir: Path, verbose: bool = False, log_file_name: str = "train.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    # Rebuild handlers each run to avoid duplicate logs when re-entering main in-process.
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_dir / log_file_name, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def _get_gpu_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}

    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_alloc_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }

# =========================
# 🔥 CUDA OPTIMIZATION
# =========================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    # Avoid leaving unused VRAM headroom that can trigger avoidable OOMs.
    torch.cuda.set_per_process_memory_fraction(1.0)


# =========================
# DATA COLLATE
# =========================
def collate_pretokenized_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([
        item.get('attention_mask', torch.ones_like(item['input_ids']))
        for item in batch
    ])
    labels = torch.stack([
        item.get('labels', item['input_ids'])
        for item in batch
    ])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# =========================
# CHECKPOINT UTILS
# =========================
def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None

    def checkpoint_step(path: Path) -> Optional[int]:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match is None:
            return None
        return int(match.group(1))

    candidates = []
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        if checkpoint_step(d) is None:
            continue
        candidates.append(d)

    if not candidates:
        return None
    return max(candidates, key=lambda p: checkpoint_step(p) or -1)


def save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, global_step, step_times):
    ckpt = output_dir / f'checkpoint-{global_step}'
    ckpt.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))

    torch.save({
        'global_step': global_step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step_times': step_times
    }, ckpt / "training_state.pt")

    return ckpt


def prune_old_checkpoints(output_dir: Path, keep_last: int):
    if keep_last < 0:
        return
    ckpts = sorted([d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")],
                   key=lambda p: int(p.name.split("-")[-1]))
    for d in ckpts[:-keep_last]:
        shutil.rmtree(d, ignore_errors=True)


# =========================
# LOSS (optimized)
# =========================
def compute_loss(outputs, batch):
    if outputs.loss is not None:
        return outputs.loss

    logits = outputs.logits
    labels = batch['labels']

    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[..., 1:].reshape(-1)

    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


# =========================
# DATA LOADING
# =========================
def load_data(path, logger: Optional[logging.Logger] = None):
    train_path = Path(path) / "train"

    def _log(level: str, msg: str):
        if logger is None:
            print(msg)
            return
        getattr(logger, level)(msg)

    if train_path.exists():
        train_ds = load_from_disk(str(train_path))
        val_path = Path(path) / "validation"
        val_ds = load_from_disk(str(val_path)) if val_path.exists() else None
        return train_ds, val_ds


    shards = sorted([d for d in Path(path).iterdir() if d.is_dir() and d.name.startswith("shard_")])
    if shards:
        valid_shards = []
        skipped_shards = []

        for shard in shards:
            # A Hugging Face dataset directory must include these metadata files.
            if (shard / "state.json").exists() and (shard / "dataset_info.json").exists():
                valid_shards.append(shard)
            else:
                skipped_shards.append(shard.name)

        if skipped_shards:
            _log(
                "warning",
                f"Skipping {len(skipped_shards)} invalid shard directories: "
                + ", ".join(skipped_shards)
            )

        if not valid_shards:
            raise FileNotFoundError(
                "No valid shard datasets found under "
                f"{path}. Expected each shard_* directory to contain state.json and dataset_info.json."
            )

        ds = [load_from_disk(str(s)) for s in valid_shards]
        return concatenate_datasets(ds), None

    return load_from_disk(path), None


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenized-dir', required=True)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--base-model', type=str, default='Qwen/Qwen3-0.6B')

    parser.add_argument('--n-streams', type=int, default=2)
    parser.add_argument('--num-fracs', type=int, default=1)
    parser.add_argument('--sinkhorn-iters', type=int, default=5)
    parser.add_argument('--residual-mix-temperature', type=float, default=1.0)

    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('--total-steps', type=int, default=50000)
    parser.add_argument('--freeze-original-steps', type=int, default=500,
                        help='Freeze non-mHC backbone parameters for the first N optimizer steps.')

    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)

    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--save-steps', type=int, default=1000)
    parser.add_argument('--eval-steps', type=int, default=500)

    parser.add_argument('--keep-last-checkpoints', type=int, default=3)

    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=1)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--no-persistent-workers', action='store_true')

    parser.add_argument('--dtype', choices=['fp16', 'bf16', 'auto'], default='auto')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help="Checkpoint path or 'latest'. If omitted, auto-resumes from latest checkpoint in output-dir.")
    parser.add_argument('--no-resume', action='store_true',
                        help="Disable auto-resume from output-dir checkpoint-* directories.")
    parser.add_argument('--use-compile', action='store_true',
                        help="Enable torch.compile for speed if available.")
    parser.add_argument('--compile-mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help="torch.compile mode.")
    parser.add_argument('--use-fused-adamw', action='store_true',
                        help="Use fused AdamW when supported by this PyTorch/CUDA build.")
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                        help="Disable gradient checkpointing (faster, but higher VRAM use).")
    parser.add_argument('--log-interval', type=int, default=100,
                        help="Log optimizer-step metrics every N steps.")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable debug-level logging.")
    parser.add_argument('--log-file', type=str, default='train.log',
                        help="Log file name under output-dir.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, verbose=args.verbose, log_file_name=args.log_file)
    logger.info("Starting training run")
    logger.info(f"Workspace output directory: {output_dir}")
    logger.info(f"Device selected: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")

    if args.dtype == 'auto':
        if device.type == 'cuda' and torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            amp_dtype = torch.bfloat16
            use_grad_scaler = False
            logger.info("AMP dtype auto-selected: bf16")
        else:
            model_dtype = torch.float16
            amp_dtype = torch.float16
            use_grad_scaler = (device.type == 'cuda')
            logger.info("AMP dtype auto-selected: fp16")
    elif args.dtype == 'bf16':
        model_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
        use_grad_scaler = False
    else:
        model_dtype = torch.float16
        amp_dtype = torch.float16
        use_grad_scaler = (device.type == 'cuda')

    logger.info(
        "Training args: "
        f"batch_size={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, "
        f"total_steps={args.total_steps}, lr={args.learning_rate:.2e}, warmup={args.warmup_steps}, "
        f"save_steps={args.save_steps}, workers={args.num_workers}, compile={args.use_compile}, "
        f"dtype={args.dtype}, scaler_enabled={use_grad_scaler}, freeze_original_steps={args.freeze_original_steps}"
    )

    train_ds, val_ds = load_data(args.tokenized_dir, logger=logger)
    train_ds.set_format(type='torch')

    loader_kwargs = {
        'batch_size': args.batch_size,
        'sampler': SequentialSampler(train_ds),
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'collate_fn': collate_pretokenized_batch,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = not args.no_persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    loader = DataLoader(train_ds, **loader_kwargs)
    logger.info(
        f"Dataset loaded: train_examples={len(train_ds)}, "
        f"val_examples={(len(val_ds) if val_ds is not None else 0)}, "
        f"batches_per_epoch={len(loader)}"
    )

    checkpoint_to_resume = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "latest":
            checkpoint_to_resume = find_latest_checkpoint(output_dir)
            if checkpoint_to_resume is None:
                logger.info("No checkpoint-* found under output-dir; starting fresh.")
        else:
            checkpoint_to_resume = Path(args.resume_from_checkpoint)
            if not checkpoint_to_resume.exists():
                raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint_to_resume}")
    elif not args.no_resume:
        checkpoint_to_resume = find_latest_checkpoint(output_dir)
        if checkpoint_to_resume is not None:
            logger.info(f"Auto-resuming from latest checkpoint: {checkpoint_to_resume}")

    # =========================
    # MODEL
    # =========================
    if checkpoint_to_resume is not None:
        model, tokenizer = load_mhc_model_v2(
            model_path=str(checkpoint_to_resume),
            device="cuda" if device.type == "cuda" else "cpu",
            torch_dtype=model_dtype,
            base_model=args.base_model,
            local_files_only=args.local_files_only,
            trust_remote_code=False,
        )
    elif args.model:
        model, tokenizer = load_mhc_model_v2(
            model_path=args.model,
            device="cuda" if device.type == "cuda" else "cpu",
            torch_dtype=model_dtype,
            base_model=args.base_model,
            local_files_only=args.local_files_only,
            trust_remote_code=False,
        )
    else:
        model, tokenizer = convert_qwen3_to_mhc_v2(
            model_name_or_path=args.base_model,
            n_streams=args.n_streams,
            num_fracs=args.num_fracs,
            sinkhorn_iters=args.sinkhorn_iters,
            residual_mix_temperature=args.residual_mix_temperature,
            device="cuda" if device.type == "cuda" else "cpu",
            torch_dtype=model_dtype,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        )

    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    original_parameters_frozen = False
    if args.freeze_original_steps > 0:
        if _set_original_parameter_trainability(model, False):
            original_parameters_frozen = True
            logger.info(
                f"Stage 1 active: froze original parameters for first {args.freeze_original_steps} optimizer steps"
            )

    model.to(device)

    try:
        param_counts = count_parameters_v2(model)
        logger.info(
            f"Model parameters: total={param_counts['total']:,}, "
            f"original={param_counts['original']:,}, mhc={param_counts['mhc']:,} "
            f"({param_counts['mhc_percentage']:.2f}%)"
        )
    except Exception as exc:
        logger.warning(f"Could not compute parameter breakdown: {exc}")

    compiled_model_active = False
    if args.use_compile and hasattr(torch, "compile"):
        try:
            if not _has_triton():
                logger.warning("torch.compile requested but Triton is unavailable; continuing in eager mode.")
            else:
                # Reduce extra private-pool pressure from CUDA graph capture on smaller GPUs.
                os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                model = torch.compile(model, mode=args.compile_mode)
                compiled_model_active = True
                logger.info(f"torch.compile enabled with mode={args.compile_mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without compile: {e}")

    if args.use_fused_adamw:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
            logger.info("Using fused AdamW")
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            logger.warning("Fused AdamW unsupported in this build; using standard AdamW")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    model.train()
    global_step = 0
    step_times = []

    if checkpoint_to_resume is not None:
        training_state_path = checkpoint_to_resume / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location="cpu")
            global_step = int(state.get("global_step", 0))

            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state and use_grad_scaler:
                scaler.load_state_dict(state["scaler"])

            step_times = state.get("step_times", [])
            logger.info(f"Resumed optimizer/scheduler state from {training_state_path} at global_step={global_step}")
        else:
            logger.warning(f"Checkpoint found at {checkpoint_to_resume} but no training_state.pt; resuming weights only.")

    pbar = tqdm(total=args.total_steps, initial=global_step)
    recent_losses = deque(maxlen=50)
    recent_step_times = deque(maxlen=50)
    train_start_time = time.perf_counter()
    accumulated_micro_loss = 0.0
    micro_batch_count = 0

    logger.info("Entering training loop")

    while global_step < args.total_steps:
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=args.pin_memory) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                try:
                    outputs = model(**batch)
                except Exception as e:
                    if compiled_model_active and (
                        "Cannot find a working triton installation" in str(e)
                        or "BackendCompilerFailed" in e.__class__.__name__
                        or "backend='inductor' raised" in str(e)
                    ):
                        logger.warning("torch.compile backend failed at runtime; falling back to eager model.")
                        if hasattr(model, "_orig_mod"):
                            model = model._orig_mod
                        compiled_model_active = False
                        outputs = model(**batch)
                    else:
                        raise
                raw_loss = compute_loss(outputs, batch)
                loss = raw_loss / args.gradient_accumulation_steps

            accumulated_micro_loss += float(raw_loss.item())
            micro_batch_count += 1

            if torch.isnan(loss):
                logger.warning(f"NaN loss encountered at micro-step {step} (global_step={global_step}); skipping.")
                continue

            try:
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.exception("CUDA OOM during backward pass")
                raise RuntimeError(
                    "CUDA OOM during backward pass. Resume now works, but this run still exceeds VRAM. "
                    "Try --batch-size 1, keep gradient accumulation the same, and prefer --dtype bf16 if supported."
                )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer_step_start = time.perf_counter()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                optimizer_step_time = time.perf_counter() - optimizer_step_start
                step_times.append(optimizer_step_time)
                recent_step_times.append(optimizer_step_time)

                mean_loss = accumulated_micro_loss / max(1, micro_batch_count)
                recent_losses.append(mean_loss)
                accumulated_micro_loss = 0.0
                micro_batch_count = 0

                global_step += 1
                pbar.update(1)

                if original_parameters_frozen and global_step >= args.freeze_original_steps:
                    if _set_original_parameter_trainability(model, True):
                        original_parameters_frozen = False
                        logger.info(f"Stage 2 active: unfroze original parameters at step {global_step}")

                if global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else mean_loss
                    avg_step_time = sum(recent_step_times) / len(recent_step_times) if recent_step_times else optimizer_step_time
                    grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

                    seq_len = int(batch["input_ids"].shape[1])
                    effective_batch = args.batch_size * args.gradient_accumulation_steps
                    tokens_per_step = effective_batch * seq_len
                    tokens_per_sec = (tokens_per_step / avg_step_time) if avg_step_time > 0 else float("inf")
                    samples_per_sec = (effective_batch / avg_step_time) if avg_step_time > 0 else float("inf")

                    elapsed = time.perf_counter() - train_start_time
                    remaining_steps = max(0, args.total_steps - global_step)
                    eta_seconds = remaining_steps * avg_step_time

                    mem_stats = _get_gpu_memory_stats(device)
                    monitoring_stats = _get_monitoring_stats(model)
                    entropy = monitoring_stats.get("h_res_entropy_mean")
                    identity_dist = monitoring_stats.get("h_res_identity_distance_mean")

                    log_msg = (
                        f"step={global_step}/{args.total_steps} "
                        f"loss={mean_loss:.4f} avg_loss={avg_loss:.4f} lr={lr:.3e} grad_norm={grad_norm_value:.3f} "
                        f"step_time={optimizer_step_time:.3f}s avg_step_time={avg_step_time:.3f}s "
                        f"tokens/s={tokens_per_sec:.1f} samples/s={samples_per_sec:.2f} "
                        f"elapsed={elapsed/60.0:.1f}m eta={eta_seconds/60.0:.1f}m "
                        f"phase={'frozen' if original_parameters_frozen else 'unfrozen'}"
                    )

                    if mem_stats:
                        log_msg += (
                            f" gpu_alloc={mem_stats['alloc_gb']:.2f}GB"
                            f" gpu_reserved={mem_stats['reserved_gb']:.2f}GB"
                            f" gpu_max_alloc={mem_stats['max_alloc_gb']:.2f}GB"
                        )

                    if entropy is not None:
                        log_msg += f" h_res_entropy={entropy:.4f}"
                    if identity_dist is not None:
                        log_msg += f" h_res_identity_dist={identity_dist:.4f}"

                    logger.info(log_msg)
                    pbar.set_postfix({"loss": f"{mean_loss:.4f}", "lr": f"{lr:.2e}"})

                if global_step % args.save_steps == 0:
                    save_start = time.perf_counter()
                    ckpt = save_checkpoint(output_dir, model, tokenizer, optimizer, scheduler, global_step, step_times)
                    # Persist scaler state for exact AMP resume.
                    state_path = ckpt / "training_state.pt"
                    state = torch.load(state_path, map_location="cpu")
                    state["scaler"] = scaler.state_dict() if use_grad_scaler else None
                    torch.save(state, state_path)
                    prune_old_checkpoints(output_dir, args.keep_last_checkpoints)
                    save_dur = time.perf_counter() - save_start
                    logger.info(f"Checkpoint saved: {ckpt} (save_time={save_dur:.2f}s)")

                if global_step >= args.total_steps:
                    break

        if global_step >= args.total_steps:
            break

    pbar.close()
    total_time = time.perf_counter() - train_start_time
    avg_step_time = (sum(step_times) / len(step_times)) if step_times else 0.0
    logger.info(
        f"Training complete: global_step={global_step}, total_time={total_time/60.0:.2f}m, "
        f"avg_optimizer_step_time={avg_step_time:.3f}s"
    )


if __name__ == "__main__":
    main()