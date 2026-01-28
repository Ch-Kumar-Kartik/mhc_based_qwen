#!/usr/bin/env python3
"""
Optimized training script for mHC Qwen3 on AMD Strix Halo iGPU.

Hardware target: AMD Strix Halo iGPU (gfx120x) with 128GB shared RAM
Optimizations:
- ROCm/HIP backend with torch.compile
- Large batch sizes leveraging unified memory
- Gradient checkpointing for memory efficiency
- Fused AdamW optimizer
- Mixed precision (bfloat16)
- Prefetching and async data loading

Usage:
    # Using HuggingFace reasoning datasets
    python scripts/train_amd.py --dataset openthoughts114k --max-examples 50000
    python scripts/train_amd.py --dataset openthoughts3  # Full 1.2M dataset
    
    # Using local pre-tokenized dataset from data/ directory (created by download.py)
    python scripts/train_amd.py --data-path ./data/fineweb_edu_qwen3
    python scripts/train_amd.py --data-path ./data/fineweb_edu_qwen3 --max-examples 5000
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk

from src.config import MHCTrainingConfig
from src.conversion import load_mhc_model, convert_qwen3_to_mhc, count_parameters
from src.data import load_reasoning_dataset, create_dataloader, REASONING_DATASETS


# ============================================================================
# AMD Strix Halo Optimized Configuration
# ============================================================================

@dataclass
class AMDStrixHaloConfig:
    """Optimized configuration for AMD Strix Halo iGPU with 128GB shared RAM.
    
    The Strix Halo has unique characteristics:
    - Large unified memory (128GB) allows bigger batches
    - iGPU has lower compute than discrete GPUs, so we optimize for throughput
    - ROCm backend with specific optimizations
    """
    
    # Device settings
    device: str = "cuda"  # ROCm uses cuda API
    dtype: torch.dtype = torch.bfloat16
    
    # Batch settings - optimized for 128GB unified memory
    # Larger batches = better GPU utilization on iGPU
    batch_size: int = 8  # Per-device batch size
    gradient_accumulation_steps: int = 4  # Effective batch = 32
    max_length: int = 2048  # Sequence length (reduce if OOM)
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    pin_memory: bool = False  # Not needed for unified memory
    
    # Dataloader settings
    num_workers: int = 8  # CPU cores for data loading
    prefetch_factor: int = 4
    
    # Optimizer settings (fused for speed)
    learning_rate: float = 5e-5  # Lower for fine-tuning
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_8bit_optimizer: bool = False  # Use bitsandbytes 8-bit AdamW
    
    # Schedule
    warmup_ratio: float = 0.1
    total_steps: int = 10000
    
    # Logging
    log_interval: int = 1
    eval_interval: int = 500
    save_interval: int = 100
    
    # Compilation (torch.compile for ROCm)
    use_compile: bool = True
    compile_mode: str = "default"  # "reduce-overhead" can cause issues with grad accumulation
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def setup_amd_environment():
    """Setup environment variables for optimal AMD ROCm performance."""
    
    # Silence tokenizer fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # ROCm optimizations
    os.environ.setdefault("HSA_FORCE_FINE_GRAIN_PCIE", "1")
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "8")
    
    # Memory settings for large unified memory
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    
    # Parallel compilation - use all available cores
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(os.cpu_count() or 16))
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON,ATen")
    
    # Disable memory efficient attention if causing issues
    # os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "0")
    
    # Enable TF32 equivalent on AMD
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default dtype
    torch.set_default_dtype(torch.bfloat16)


def get_device_info() -> Dict[str, Any]:
    """Get AMD GPU device information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "rocm_available": hasattr(torch.version, 'hip') and torch.version.hip is not None,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        
        # Memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory
        info["total_memory_gb"] = total_mem / (1024**3)
        
        # For unified memory, this might show differently
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info["free_memory_gb"] = free_mem / (1024**3)
        except:
            pass
    
    return info


# ============================================================================
# Optimized Trainer
# ============================================================================

class AMDOptimizedTrainer:
    """Trainer optimized for AMD Strix Halo iGPU."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: AMDStrixHaloConfig,
        output_dir: str,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(device=self.device, dtype=config.dtype)
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            try:
                # Direct setting - bypass HF's checks which can be overly restrictive
                self.model.model.gradient_checkpointing = True
                # Set the checkpointing function
                from torch.utils.checkpoint import checkpoint
                self.model.model._gradient_checkpointing_func = checkpoint
                self.logger.info("Gradient checkpointing enabled (direct)")
            except Exception as e:
                self.logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Compile model for speed (ROCm support)
        if config.use_compile and hasattr(torch, 'compile'):
            try:
                self.logger.info(f"Compiling model with mode={config.compile_mode}...")
                self.model = torch.compile(self.model, mode=config.compile_mode)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        
        # Setup optimizer (fused for AMD)
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Timing
        self.step_times = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer - 8-bit AdamW if available, else fused AdamW."""
        config = self.config
        
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'layernorm' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Try 8-bit AdamW first (saves ~50% optimizer memory)
        if config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    lr=config.learning_rate,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                )
                self.logger.info("Using 8-bit AdamW optimizer (bitsandbytes)")
                return optimizer
            except ImportError:
                self.logger.warning("bitsandbytes not installed, falling back to standard optimizer")
            except Exception as e:
                self.logger.warning(f"8-bit optimizer failed: {e}, falling back to standard")
        
        # Use fused AdamW if available (faster on GPU)
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                fused=True,  # Fused kernel
            )
            self.logger.info("Using fused AdamW optimizer")
        except TypeError:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
            )
            self.logger.info("Using standard AdamW optimizer")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        config = self.config
        warmup_steps = int(config.total_steps * config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (config.total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Run optimized training loop."""
        config = self.config
        
        self.logger.info("=" * 60)
        self.logger.info("Starting AMD-optimized training")
        self.logger.info("=" * 60)
        self.logger.info(f"Total steps: {config.total_steps}")
        self.logger.info(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
        self.logger.info(f"Sequence length: {config.max_length}")
        self.logger.info(f"Learning rate: {config.learning_rate}")
        
        self.model.train()
        accumulation_loss = 0.0
        
        epoch = 0
        data_iter = iter(self.train_dataloader)
        
        start_time = time.time()
        
        pbar = tqdm(total=config.total_steps, desc="Training", unit="step")
        
        while self.global_step < config.total_steps:
            step_start = time.time()
            
            # Accumulation loop
            for micro_step in range(config.gradient_accumulation_steps):
                # Mark step begin for CUDA graphs (needed for torch.compile reduce-overhead mode)
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    self.logger.info(f"Starting epoch {epoch}")
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                # Move to device (async for unified memory)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=config.dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                accumulation_loss += loss.item()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.max_grad_norm,
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Update tracking
            self.global_step += 1
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{accumulation_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step_time': f'{step_time:.2f}s'
            })
            
            # Logging
            if self.global_step % config.log_interval == 0:
                self._log_progress(accumulation_loss, grad_norm, step_time)
            
            accumulation_loss = 0.0
            
            # Evaluation
            if self.eval_dataloader and self.global_step % config.eval_interval == 0:
                self._evaluate()
            
            # Checkpoint
            if self.global_step % config.save_interval == 0:
                self._save_checkpoint()
        
        pbar.close()
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Average step time: {sum(self.step_times)/len(self.step_times):.3f}s")
        self.logger.info("=" * 60)
        
        self._save_checkpoint(final=True)
    
    def _log_progress(self, loss: float, grad_norm: float, step_time: float):
        """Log training progress."""
        lr = self.scheduler.get_last_lr()[0]
        
        # Calculate throughput
        tokens_per_step = self.config.effective_batch_size * self.config.max_length
        tokens_per_sec = tokens_per_step / step_time
        
        # Memory usage (if available)
        try:
            mem_used = torch.cuda.memory_allocated() / (1024**3)
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)
            mem_str = f", mem={mem_used:.1f}/{mem_reserved:.1f}GB"
        except:
            mem_str = ""
        
        self.logger.info(
            f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}, "
            f"grad_norm={grad_norm:.2f}, {tokens_per_sec:.0f} tok/s, "
            f"step_time={step_time:.2f}s{mem_str}"
        )
        
        self.loss_history.append(loss)
    
    def _evaluate(self):
        """Run evaluation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.amp.autocast(device_type='cuda', dtype=self.config.dtype):
                    outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                if num_batches >= 50:  # Limit eval batches
                    break
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"[EVAL] Step {self.global_step}: loss={avg_loss:.4f}")
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)
        
        self.model.train()
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save checkpoint."""
        if final:
            save_dir = self.output_dir / "final"
        elif best:
            save_dir = self.output_dir / "best"
        else:
            save_dir = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        # Unwrap compiled model if needed
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod
        
        model_to_save.save_pretrained(save_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history[-1000:],  # Keep last 1000
        }
        torch.save(state, save_dir / "training_state.pt")
        
        self.logger.info(f"Checkpoint saved to {save_dir}")


# ============================================================================
# Main
# ============================================================================

def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train mHC Qwen3 on AMD Strix Halo (optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="Base model or path to converted mHC model",
    )
    parser.add_argument(
        "--mhc-model", type=str, default=None,
        help="Path to pre-converted mHC model (skip conversion)",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="openthoughts114k",
        choices=list(REASONING_DATASETS.keys()),
        help="Reasoning dataset to use (ignored if --data-path is set)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to local pre-tokenized dataset (e.g., ./data/fineweb_edu_qwen3). If set, --dataset is ignored.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum training examples (None = full dataset)",
    )
    parser.add_argument(
        "--packing", action="store_true",
        help="Enable sequence packing for 2-5x faster training",
    )
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N steps")
    
    # Memory/Performance
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--8bit-optimizer", action="store_true", dest="use_8bit_optimizer", help="Use 8-bit AdamW (requires bitsandbytes)")
    parser.add_argument("--num-workers", type=int, default=8)
    
    # Output
    parser.add_argument("--output", type=str, default="./output/mhc_training")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.output, args.verbose)
    logger = logging.getLogger(__name__)
    
    # Setup AMD environment
    setup_amd_environment()
    
    # Print device info
    logger.info("=" * 60)
    logger.info("AMD Strix Halo Optimized Training")
    logger.info("=" * 60)
    
    device_info = get_device_info()
    for k, v in device_info.items():
        logger.info(f"  {k}: {v}")
    
    # Create config
    config = AMDStrixHaloConfig(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        learning_rate=args.lr,
        total_steps=args.total_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=not args.no_grad_checkpoint,
        use_compile=not args.no_compile,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        use_8bit_optimizer=args.use_8bit_optimizer,
    )
    
    logger.info(f"Effective batch size: {config.effective_batch_size}")
    
    # Load or convert model
    if args.mhc_model:
        logger.info(f"Loading pre-converted mHC model from {args.mhc_model}")
        model, tokenizer = load_mhc_model(args.mhc_model, device="cuda")
    else:
        logger.info(f"Converting {args.model} to mHC architecture...")
        model, tokenizer = convert_qwen3_to_mhc(
            model_name_or_path=args.model,
            n_streams=4,
            device="cuda",
            torch_dtype=config.dtype,
            validate=False,  # Skip validation for speed
        )
        
        # Save converted model
        converted_path = Path(args.output) / "converted_model"
        logger.info(f"Saving converted model to {converted_path}")
        model.save_pretrained(converted_path)
        tokenizer.save_pretrained(converted_path)
    
    # Print model info
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts['total']:,}")
    logger.info(f"  Original: {param_counts['original']:,}")
    logger.info(f"  mHC: {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.1f}%)")
    
    # Load dataset
    if args.data_path:
        # Load local pre-tokenized dataset (from download.py)
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        logger.info(f"Loading local dataset from: {data_path}")
        train_dataset = load_from_disk(str(data_path))
        
        # Verify required columns exist
        required_cols = {"input_ids", "attention_mask", "labels"}
        if not required_cols.issubset(set(train_dataset.column_names)):
            raise ValueError(f"Dataset missing required columns. Found: {train_dataset.column_names}, need: {required_cols}")
        
        # Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Apply max_examples limit if specified
        if args.max_examples and args.max_examples < len(train_dataset):
            train_dataset = train_dataset.select(range(args.max_examples))
        
        logger.info(f"Dataset loaded: {len(train_dataset)} examples")
    else:
        # Load from HuggingFace reasoning datasets
        logger.info(f"Loading dataset: {args.dataset}")
        train_dataset = load_reasoning_dataset(
            args.dataset,
            tokenizer,
            max_length=config.max_length,
            max_examples=args.max_examples,
            packing=args.packing,
        )
    logger.info(f"Training examples: {len(train_dataset)}")
    
    # Custom collate function for packed datasets (filters out metadata)
    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            if key.startswith("_"):  # Skip metadata like _sequence_boundaries
                continue
            result[key] = torch.stack([item[key] for item in batch])
        return result
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn if args.packing else None,
    )
    
    # Optional: create eval dataloader from subset
    eval_dataloader = None
    if len(train_dataset) > 1000:
        from torch.utils.data import Subset
        eval_indices = list(range(0, min(500, len(train_dataset)), 1))
        eval_subset = Subset(train_dataset, eval_indices)
        eval_dataloader = DataLoader(
            eval_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
        )
    
    # Create trainer
    trainer = AMDOptimizedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=args.output,
    )
    
    # Train!
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
