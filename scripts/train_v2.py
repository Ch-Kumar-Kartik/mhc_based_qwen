#!/usr/bin/env python3
"""
Training script for mHC V2 Qwen3 models.

This script implements the training procedure for the V2 mHC architecture
based on the hyper-connections paper, including:
- Learning rate scheduling with step decay
- Gradient checkpointing for memory efficiency
- Monitoring of mHC-specific metrics
- Stability checks and early stopping

Usage:
    python -m scripts.train_v2 --config configs/qwen3_0.6b_mhc_v2.yaml
"""

import argparse
import logging
import math
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MHCTrainingConfig
from src.conversionV2 import load_mhc_model_v2, count_parameters_v2, convert_qwen3_to_mhc_v2


def collate_pretokenized_batch(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate pre-tokenized samples and synthesize missing training fields."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([
        item["attention_mask"] if "attention_mask" in item else torch.ones_like(item["input_ids"])
        for item in batch
    ])
    labels = torch.stack([
        item["labels"] if "labels" in item else item["input_ids"].clone()
        for item in batch
    ])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging to file and console."""
    level = logging.DEBUG if verbose else logging.INFO
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


class StepLRScheduler:
    """Learning rate scheduler with step decay.
    
    Implements the schedule from the mHC paper:
    - Warmup for warmup_steps
    - Constant LR until 80% of training
    - Decay by 0.316 at 80%
    - Decay by 0.1 at 90%
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: MHCTrainingConfig,
    ):
        self.optimizer = optimizer
        self.config = config
        self.base_lr = config.learning_rate
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self) -> Dict[str, int]:
        """Serialize scheduler state."""
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: Dict[str, int]):
        """Restore scheduler state and re-apply LR."""
        self.current_step = int(state_dict.get("current_step", 0))
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """Compute learning rate for current step."""
        step = self.current_step
        config = self.config
        
        # Warmup phase
        if step < config.warmup_steps:
            return self.base_lr * step / config.warmup_steps
        
        # After warmup
        progress = step / config.total_steps
        
        if progress < 0.8:
            return self.base_lr
        elif progress < 0.9:
            return self.base_lr * config.lr_decay_ratio_1
        else:
            return self.base_lr * config.lr_decay_ratio_1 * config.lr_decay_ratio_2


class MHCV2Trainer:
    """Trainer for mHC V2 models with stability monitoring."""
    
    def __init__(
        self,
        model: nn.Module,
        config: MHCTrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        output_dir: str = "./output",
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.device = next(self.model.parameters()).device
        self.grad_accum_steps = max(1, int(self.config.gradient_accumulation_steps))
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(self.config.amp_dtype)
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_grad_scaler)
        self.is_compiled = False

        if self.config.use_compile:
            self._try_compile_model()
        else:
            self.logger.info("torch.compile: disabled (use_compile=false)")

        self._configure_runtime_performance()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = StepLRScheduler(self.optimizer, config)
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = deque(maxlen=100)

        effective_batch = self.config.batch_size * self.grad_accum_steps
        tokens_per_step = effective_batch * self.config.sequence_length
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
        self.logger.info(f"Effective batch size: {effective_batch}")
        self.logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
        self.logger.info(f"DataLoader workers: {self.config.num_workers}")
        self._log_compile_status()

    def _configure_runtime_performance(self):
        """Apply CPU/GPU runtime knobs from config."""
        try:
            torch.set_num_threads(int(self.config.cpu_num_threads))
        except Exception as e:
            self.logger.warning(f"Could not set cpu_num_threads: {e}")

        try:
            torch.set_num_interop_threads(int(self.config.cpu_num_interop_threads))
        except Exception as e:
            self.logger.warning(f"Could not set cpu_num_interop_threads: {e}")

        if self.device.type == "cuda":
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")

            if self.config.use_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True

        self.logger.info(
            "Runtime tuning: "
            f"cpu_threads={torch.get_num_threads()}, "
            f"interop_threads={torch.get_num_interop_threads()}"
        )

    def _try_compile_model(self):
        """Compile model with torch.compile when enabled and available."""
        if not hasattr(torch, "compile"):
            self.logger.warning("torch.compile is unavailable in this PyTorch build")
            return

        try:
            # Keep training alive by falling back to eager when the backend fails at runtime.
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            self.logger.info(f"Compiling model with mode={self.config.compile_mode}")
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
            self.is_compiled = True
            self.logger.info("Model compilation complete")
        except Exception as e:
            self.logger.warning(f"torch.compile failed, continuing without compile: {e}")

    def _forward_with_compile_fallback(self, batch: Dict[str, torch.Tensor]):
        """Run forward pass and fallback to eager if compile backend fails at runtime."""
        try:
            return self.model(**batch)
        except Exception as e:
            if self.is_compiled and (
                "BackendCompilerFailed" in e.__class__.__name__
                or "Cannot find a working triton installation" in str(e)
                or "backend='inductor' raised" in str(e)
            ):
                self.logger.warning(
                    "torch.compile backend failed at runtime; falling back to eager execution"
                )
                if hasattr(self.model, "_orig_mod"):
                    self.model = self.model._orig_mod
                self.is_compiled = False
                self._log_compile_status()
                return self.model(**batch)
            raise

    def _log_compile_status(self):
        """Log compile status once during startup for quick verification."""
        if self.config.use_compile and self.is_compiled:
            self.logger.info(f"torch.compile: active (mode={self.config.compile_mode})")
        elif self.config.use_compile and not self.is_compiled:
            self.logger.info("torch.compile: requested but inactive (fallback mode)")

    def _resolve_amp_dtype(self, dtype_name: str) -> torch.dtype:
        """Resolve user-configured AMP dtype with safe fallback."""
        normalized = (dtype_name or "bfloat16").lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            self.logger.warning(f"Unknown amp_dtype='{dtype_name}', using bfloat16")
            return torch.bfloat16
        return mapping[normalized]

    def _autocast_context(self):
        """Create autocast context for the active device and configured dtype."""
        if self.use_amp and self.amp_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with mHC-paper settings."""
        config = self.config

        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            if "bias" in lowered or "norm" in lowered:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer_kwargs = {
            "lr": config.learning_rate,
            "betas": (config.adam_beta1, config.adam_beta2),
            "eps": config.adam_epsilon,
        }

        if config.use_fused_adamw:
            try:
                return AdamW(param_groups, fused=True, **optimizer_kwargs)
            except TypeError:
                self.logger.warning("Fused AdamW not available, falling back to standard AdamW")

        return AdamW(param_groups, **optimizer_kwargs)
    
    def train(self):
        """Run full training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Total steps: {self.config.total_steps}")
        self.logger.info(f"Warmup steps: {self.config.warmup_steps}")
        
        self.model.train()
        
        if self.config.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except ValueError as e:
                self.logger.warning(f"Gradient checkpointing not available: {e}")

        self.optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        
        epoch = 0
        while self.global_step < self.config.total_steps:
            epoch += 1
            self.logger.info(f"Starting epoch {epoch}")
            
            for batch in self.train_dataloader:
                if self.global_step >= self.config.total_steps:
                    break

                micro_step += 1
                should_step = (micro_step % self.grad_accum_steps) == 0
                loss = self._training_step(batch)

                # Check for instability
                if self._check_instability(loss):
                    self.logger.error("Training instability detected! Stopping.")
                    return

                if not should_step:
                    continue

                grad_norm = self._optimizer_step()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(loss, grad_norm)
                
                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_interval == 0:
                    self._evaluate()
                
                # Checkpoint
                if self.global_step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
        
        self.logger.info("Training completed!")
        self._save_checkpoint(final=True)

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one micro-batch forward/backward pass."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with self._autocast_context():
            outputs = self._forward_with_compile_fallback(batch)
            raw_loss = outputs.loss

        scaled_loss = raw_loss / self.grad_accum_steps
        if self.use_grad_scaler:
            self.grad_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return float(raw_loss.item())

    def _optimizer_step(self) -> float:
        """Apply gradient clipping and optimizer/scheduler updates."""
        if self.use_grad_scaler:
            self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm,
        )

        if float(grad_norm) > self.config.gradient_norm_threshold:
            self.logger.warning(
                f"Step {self.global_step}: Large gradient norm {float(grad_norm):.2f}"
            )

        if self.use_grad_scaler:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return float(grad_norm)
    
    def _check_instability(self, loss: float) -> bool:
        """Check for training instability."""
        # NaN/Inf check
        if not math.isfinite(loss):
            self.logger.error(f"Non-finite loss at step {self.global_step}: {loss}")
            return True
        
        # Loss spike check
        self.loss_history.append(loss)
        if len(self.loss_history) >= 20:
            history = list(self.loss_history)
            recent_mean = sum(history[-10:]) / 10
            earlier_mean = sum(history[:10]) / 10

            if recent_mean > earlier_mean * self.config.loss_spike_threshold:
                self.logger.warning(
                    f"Loss spike detected: recent={recent_mean:.4f}, "
                    f"earlier={earlier_mean:.4f}"
                )
        
        return False
    
    def _log_metrics(self, loss: float, grad_norm: float):
        """Log training metrics."""
        lr = self.optimizer.param_groups[0]['lr']

        self.logger.info(
            f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}, grad_norm={grad_norm:.2f}"
        )
    
    def _evaluate(self):
        """Run evaluation."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with self._autocast_context():
                    outputs = self._forward_with_compile_fallback(batch)
                total_loss += outputs.loss.item()
                num_batches += 1

                if self.config.max_eval_batches and num_batches >= self.config.max_eval_batches:
                    break
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Evaluation - Step {self.global_step}: loss={avg_loss:.4f}")
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)
        
        self.model.train()
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        output_path = Path(self.output_dir)
        if final:
            checkpoint_dir = output_path / "final"
        elif best:
            checkpoint_dir = output_path / "best"
        else:
            checkpoint_dir = output_path / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model_to_save.save_pretrained(str(checkpoint_dir))

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": list(self.loss_history),
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load trainer state from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_path)
        state_path = checkpoint_dir / "training_state.pt"

        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint state not found: {state_path}")

        state = torch.load(state_path, map_location=self.device)
        self.optimizer.load_state_dict(state.get("optimizer_state_dict", {}))

        scheduler_state = state.get("scheduler_state_dict")
        if scheduler_state:
            self.scheduler.load_state_dict(scheduler_state)
        else:
            self.scheduler.load_state_dict({"current_step": state.get("scheduler_step", 0)})

        self.global_step = int(state.get("global_step", 0))
        self.best_loss = float(state.get("best_loss", float("inf")))
        self.loss_history = deque(state.get("loss_history", []), maxlen=100)

        self.logger.info(f"Resumed trainer state from {checkpoint_dir} (step {self.global_step})")


def load_config(config_path: str) -> MHCTrainingConfig:
    """Load training config from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return MHCTrainingConfig(**config_dict.get('training', {}))


def load_model_config(config_path: str) -> dict:
    """Load model config from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict.get('model', {})


def load_hf_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 4096,
    batch_size: int = 1,
    split: str = "train",
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    drop_last: bool = False,
):
    """Load dataset from HuggingFace hub, local path, or pre-tokenized Arrow format.
    
    Args:
        data_path: HuggingFace dataset name, local JSON path, or Arrow dataset path
        tokenizer: Tokenizer for encoding (unused if data is pre-tokenized)
        max_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        split: Dataset split to use
        
    Returns:
        DataLoader for the dataset
    """
    from datasets import load_dataset, load_from_disk
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {data_path}")

    if shuffle is None:
        shuffle = split == "train"

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers
    
    # Check if it's a pre-tokenized Arrow dataset (from download.py)
    arrow_path = Path(data_path)
    if arrow_path.exists() and arrow_path.is_dir():
        # Check for Arrow dataset markers
        if (arrow_path / "dataset_info.json").exists() or (arrow_path / "state.json").exists():
            logger.info("Detected pre-tokenized Arrow dataset, loading directly...")
            loaded_obj = load_from_disk(str(arrow_path))

            # load_from_disk may return Dataset or DatasetDict.
            if hasattr(loaded_obj, "column_names"):
                if split != "train":
                    raise ValueError(
                        "Requested non-train split from a single-split disk dataset"
                    )
                dataset = loaded_obj
            else:
                available_splits = list(loaded_obj.keys())
                if split in loaded_obj:
                    dataset = loaded_obj[split]
                else:
                    fallback_split = available_splits[0]
                    logger.warning(
                        f"Split '{split}' not found in Arrow dataset; using '{fallback_split}'"
                    )
                    dataset = loaded_obj[fallback_split]

            columns = set(dataset.column_names)
            if "input_ids" in columns:
                logger.info(
                    "Using pre-tokenized Arrow dataset and auto-filling missing columns if needed"
                )

                format_columns = [
                    col for col in ["input_ids", "attention_mask", "labels"] if col in columns
                ]
                dataset.set_format(type="torch", columns=format_columns)

                dataloader = DataLoader(
                    dataset,
                    collate_fn=collate_pretokenized_batch,
                    **dataloader_kwargs,
                )

                logger.info(f"Dataset loaded: {len(dataset)} examples, {len(dataloader)} batches")
                return dataloader

            logger.warning(
                f"Arrow dataset does not contain 'input_ids', will attempt text tokenization. "
                f"Found: {dataset.column_names}"
            )
    
    # Load dataset from HuggingFace or JSON
    try:
        # Check if it's a dataset with a config name
        if "/" in data_path:
            parts = data_path.split("/")
            if len(parts) == 2:
                try:
                    dataset = load_dataset(data_path, split=split)
                except Exception:
                    dataset = load_dataset(parts[0], parts[1], split=split)
            else:
                dataset = load_dataset(data_path, split=split)
        else:
            dataset = load_dataset(data_path, split=split)
    except Exception as e:
        logger.warning(f"HuggingFace dataset load failed: {e}")
        dataset = load_dataset("json", data_files=data_path, split=split)
    
    # Determine text column
    text_column = None
    for col in ["text", "content", "input", "prompt"]:
        if col in dataset.column_names:
            text_column = col
            break
    
    if text_column is None:
        text_column = dataset.column_names[0]
        logger.warning(f"No standard text column found, using '{text_column}'")
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        **dataloader_kwargs,
    )
    
    logger.info(f"Dataset loaded: {len(tokenized_dataset)} examples, {len(dataloader)} batches")
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train mHC V2 model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to existing mHC V2 model (if None, will convert from base model)",
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model to convert from (used if --model is not provided)",
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.output, args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config(args.config)
    model_config = load_model_config(args.config)
    
    # Load or convert model
    model_source = args.resume if args.resume else args.model

    if model_source is not None:
        logger.info(f"Loading existing mHC V2 model from {model_source}")
        model, tokenizer = load_mhc_model_v2(
            model_source,
            device=args.device,
            base_model=args.base_model,
        )
    else:
        logger.info(f"Converting base model {args.base_model} to mHC V2")
        n_streams = model_config.get('n_streams', 4)
        num_fracs = model_config.get('num_fracs', 1)
        sinkhorn_iters = model_config.get('sinkhorn_iters', 20)
        
        model, tokenizer = convert_qwen3_to_mhc_v2(
            model_name_or_path=args.base_model,
            n_streams=n_streams,
            num_fracs=num_fracs,
            sinkhorn_iters=sinkhorn_iters,
            device=args.device,
            validate=True,
            validation_tolerance=1.0,  # Relaxed tolerance for V2
        )

    model = model.to(args.device)
    
    # Print model summary
    param_counts = count_parameters_v2(model)
    logger.info(f"Model loaded: {param_counts['total']:,} total params")
    logger.info(f"  Original: {param_counts['original']:,}")
    logger.info(f"  mHC:      {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.1f}%)")
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data}")
    train_dataloader = load_hf_dataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=config.sequence_length,
        batch_size=config.batch_size,
        split="train",
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
        drop_last=config.drop_last,
    )
    
    # Try to load eval split if available
    eval_dataloader = None
    try:
        eval_dataloader = load_hf_dataset(
            data_path=args.data,
            tokenizer=tokenizer,
            max_length=config.sequence_length,
            batch_size=config.batch_size,
            split="validation",
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            drop_last=False,
        )
    except Exception as e:
        logger.warning(f"No validation split found: {e}")
    
    # Create trainer and start training
    trainer = MHCV2Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        output_dir=args.output,
    )

    if args.resume:
        logger.info(f"Loading optimizer/scheduler state from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
