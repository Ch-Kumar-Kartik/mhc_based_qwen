#!/usr/bin/env python3
"""
Optimized training script for mHC V2 Qwen3 on NVIDIA RTX 4060.

Hardware target: RTX 4060 (12GB typical)
Optimizations:
- CUDA backend with torch.compile (default) and TF32 enabled
- Mixed precision (bfloat16) with configurable dtype override
- Gradient checkpointing for memory efficiency
- Fused AdamW optimizer (PyTorch) with optional 8-bit AdamW
- Tuned dataloader defaults for PCIe GPU (pin_memory, prefetch)
- Resume training from an existing checkpoint directory

Usage:
    python scripts/train_amd_v2.py --dataset openthoughts114k --max-examples 50000 --total-steps 50000
    python scripts/train_amd_v2.py --dataset openthoughts3  # Full 1.2M dataset
    python scripts/train_amd_v2.py --resume ./output/mhc_v2_training/checkpoint-1000
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

from src.config import MHCTrainingConfig
from src.conversionV2 import load_mhc_model_v2, convert_qwen3_to_mhc_v2, count_parameters_v2
from src.data import load_reasoning_dataset, create_dataloader, REASONING_DATASETS, load_tokenized_dataset_from_disk


# ============================================================================
# NVIDIA RTX 4060 Optimized Configuration
# ============================================================================

@dataclass
class AMDStrixHaloConfigV2:
    """Optimized configuration for NVIDIA RTX 4060 (12GB typical).
    
    Defaults favor CUDA + PCIe data path: pinned host memory, mixed precision,
    and moderate worker counts to balance CPU overhead.
    """
    
    # Device settings
    device: str = "cuda"  # ROCm uses cuda API
    dtype: torch.dtype = torch.bfloat16
    
    # Batch settings - tuned for 12GB @ 2048 tokens; lower if you hit OOM
    batch_size: int = 2  # Per-device batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    max_length: int = 2048  # Sequence length (reduce if OOM)
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    pin_memory: bool = True  # Prefer pinned host memory for PCIe GPUs
    
    # Dataloader settings
    num_workers: int = 6
    prefetch_factor: int = 2
    
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
    
    # Compilation (torch.compile for CUDA)
    use_compile: bool = True
    compile_mode: str = "default"  # "reduce-overhead" can cause issues with grad accumulation
    
    # mHC V2 specific
    n_streams: int = 4
    num_fracs: int = 1
    sinkhorn_iters: int = 20
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def setup_nvidia_environment(target_dtype: torch.dtype = torch.bfloat16):
    """Setup environment variables for optimal NVIDIA CUDA performance."""
    
    # Silence tokenizer fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # CUDA/Inductor knobs
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(os.cpu_count() or 16))
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON,ATen")
    os.environ.setdefault("TORCHINDUCTOR_USE_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    
    # TF32 for matmuls/convs (safe on 4090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Prefer high precision accumulations
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    torch.set_default_dtype(target_dtype)


def get_device_info() -> Dict[str, Any]:
    """Get CUDA device information."""
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
        except Exception:
            pass
    
    return info


# ============================================================================
# Optimized Trainer for V2
# ============================================================================

class AMDOptimizedTrainerV2:
    """Trainer optimized for mHC V2 on high-end CUDA GPUs (tuned for RTX 4090)."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: AMDStrixHaloConfigV2,
        output_dir: str,
        resume_checkpoint: Optional[Path] = None,
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
        
        # Compile model for speed (CUDA)
        if config.use_compile and hasattr(torch, 'compile'):
            try:
                self.logger.info(f"Compiling model with mode={config.compile_mode}...")
                self.model = torch.compile(self.model, mode=config.compile_mode)
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        
        # Setup optimizer (prefers fused CUDA kernels)
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        self.step_times = []

        # Optionally resume
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)
    
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
        self.logger.info("Starting CUDA-optimized mHC V2 training")
        self.logger.info("=" * 60)
        self.logger.info(f"Total steps: {config.total_steps}")
        self.logger.info(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
        self.logger.info(f"Sequence length: {config.max_length}")
        self.logger.info(f"Learning rate: {config.learning_rate}")
        self.logger.info(f"mHC V2 streams: {config.n_streams}")
        self.logger.info(f"mHC V2 fracs: {config.num_fracs}")
        self.logger.info(f"Sinkhorn iterations: {config.sinkhorn_iters}")
        if self.global_step > 0:
            self.logger.info(f"Resuming at global_step={self.global_step}")
        if self.global_step >= config.total_steps:
            self.logger.info("Global step already at/above total steps; nothing to do.")
            return
        
        self.model.train()
        accumulation_loss = 0.0
        
        epoch = 0
        data_iter = iter(self.train_dataloader)
        
        start_time = time.time()
        
        pbar = tqdm(total=config.total_steps, initial=self.global_step, desc="Training", unit="step")
        
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

    def _load_checkpoint(self, checkpoint_dir: Path):
        """Load training state from checkpoint directory."""
        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            self.logger.warning(f"No training_state.pt found in {checkpoint_dir}, starting fresh")
            return
        try:
            state = torch.load(state_path, map_location=self.device)
        except Exception as exc:
            self.logger.warning(f"Failed to load training state from {state_path}: {exc}")
            return
        # Restore optimizer/scheduler and counters
        try:
            self.optimizer.load_state_dict(state.get("optimizer_state_dict", {}))
            self.scheduler.load_state_dict(state.get("scheduler_state_dict", {}))
        except Exception as exc:
            self.logger.warning(f"Failed to load optimizer/scheduler state: {exc}")
        self.global_step = int(state.get("global_step", 0))
        self.best_loss = float(state.get("best_loss", float("inf")))
        self.loss_history = state.get("loss_history", [])
        self.step_times = state.get("step_times", [])
        self.logger.info(f"Resumed from {checkpoint_dir} at step {self.global_step} (best loss {self.best_loss:.4f})")
    
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
            "step_times": self.step_times[-1000:],
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
        description="Train mHC V2 Qwen3 on NVIDIA RTX 4060 (optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset-specific defaults (only applied when the user does not pass the flag)
    dataset_default_total_steps = {
        "openthoughts114k": 50000,
        # IndicCorpV2 defaults (large corpus, adjust based on max-examples)
        "indiccorp_hi": 100000,  # Hindi
        "indiccorp_te": 50000,   # Telugu
        "indiccorp_ta": 50000,   # Tamil
        "indiccorp_mr": 50000,   # Marathi
        "indiccorp_bn": 50000,   # Bengali
        "indiccorp_gu": 50000,   # Gujarati
        "indiccorp_kn": 50000,   # Kannada
        "indiccorp_ml": 50000,   # Malayalam
        "indiccorp_pa": 30000,   # Punjabi
        "indiccorp_or": 30000,   # Odia
        "indiccorp_as": 20000,   # Assamese
        "indiccorp_ne": 20000,   # Nepali
        "indiccorp_sa": 10000,   # Sanskrit
        "indiccorp_ur": 50000,   # Urdu
    }
    
    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B",
        help="Base model or path to converted mHC model",
    )
    parser.add_argument(
        "--mhc-model", type=str, default=None,
        help="Path to pre-converted mHC V2 model (skip conversion)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory containing training_state.pt to resume",
    )
    
    # mHC V2 specific
    parser.add_argument("--n-streams", type=int, default=4, help="Number of mHC streams")
    parser.add_argument("--num-fracs", type=int, default=1, help="Number of fractions for frac-connections")
    parser.add_argument("--sinkhorn-iters", type=int, default=20, help="Sinkhorn iterations")
    
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="openthoughts114k",
        choices=list(REASONING_DATASETS.keys()),
        help="Dataset to use. Options include: openthoughts3, openthoughts114k, openthoughts2, "
             "and IndicCorpV2 languages: indiccorp_hi (Hindi), indiccorp_te (Telugu), "
             "indiccorp_ta (Tamil), indiccorp_mr (Marathi), indiccorp_bn (Bengali), indiccorp_gu (Gujarati), "
             "indiccorp_kn (Kannada), indiccorp_ml (Malayalam), indiccorp_pa (Punjabi), indiccorp_or (Odia), "
             "indiccorp_as (Assamese), indiccorp_ne (Nepali), indiccorp_sa (Sanskrit), indiccorp_ur (Urdu)",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to pre-tokenized dataset on disk (saved via datasets.save_to_disk). Overrides --dataset if provided.",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (e.g., 'train', 'validation', 'test') when loading from --dataset-path.",
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total optimizer steps (default is dataset-dependent)",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute dtype")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g., cuda or cuda:1")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-interval", type=int, default=100, help="Save checkpoint every N steps")
    
    # Memory/Performance
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--8bit-optimizer", action="store_true", dest="use_8bit_optimizer", help="Use 8-bit AdamW (requires bitsandbytes)")
    parser.add_argument("--num-workers", type=int, default=6)
    
    # Output
    parser.add_argument("--output", type=str, default="./output/mhc_v2_training")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()

    # Apply dataset-specific default steps if user didn't specify --total-steps
    if args.total_steps is None:
        args.total_steps = dataset_default_total_steps.get(args.dataset, 10000)
    
    # Setup
    setup_logging(args.output, args.verbose)
    logger = logging.getLogger(__name__)
    
    # Map dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]
    
    # Setup CUDA environment
    setup_nvidia_environment(target_dtype)
    
    # Print device info
    logger.info("=" * 60)
    logger.info("NVIDIA RTX 4060 Optimized Training (mHC V2)")
    logger.info("=" * 60)
    
    device_info = get_device_info()
    for k, v in device_info.items():
        logger.info(f"  {k}: {v}")
    
    # Create config
    config = AMDStrixHaloConfigV2(
        device=args.device,
        dtype=target_dtype,
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
        n_streams=args.n_streams,
        num_fracs=args.num_fracs,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    
    logger.info(f"Effective batch size: {config.effective_batch_size}")
    logger.info(f"mHC V2 config: n_streams={config.n_streams}, num_fracs={config.num_fracs}, sinkhorn_iters={config.sinkhorn_iters}")
    
    resume_checkpoint = Path(args.resume) if args.resume else None
    if resume_checkpoint and not resume_checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_checkpoint}")

    # Load or convert model
    if resume_checkpoint:
        logger.info(f"Resuming model from {resume_checkpoint}")
        model, tokenizer = load_mhc_model_v2(resume_checkpoint, device=config.device)
    elif args.mhc_model:
        logger.info(f"Loading pre-converted mHC V2 model from {args.mhc_model}")
        model, tokenizer = load_mhc_model_v2(args.mhc_model, device=config.device)
    else:
        logger.info(f"Converting {args.model} to mHC V2 architecture...")
        model, tokenizer = convert_qwen3_to_mhc_v2(
            model_name_or_path=args.model,
            n_streams=config.n_streams,
            num_fracs=config.num_fracs,
            sinkhorn_iters=config.sinkhorn_iters,
            device=config.device,
            torch_dtype=config.dtype,
            validate=False,  # Skip validation for speed
        )
        
        # Save converted model
        converted_path = Path(args.output) / "converted_model"
        logger.info(f"Saving converted model to {converted_path}")
        model.save_pretrained(converted_path)
        tokenizer.save_pretrained(converted_path)
    
    # Print model info
    param_counts = count_parameters_v2(model)
    logger.info(f"Model parameters: {param_counts['total']:,}")
    logger.info(f"  Original: {param_counts['original']:,}")
    logger.info(f"  mHC: {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.1f}%)")
    
    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading pre-tokenized dataset from: {args.dataset_path}")
        logger.info(f"  Split: {args.split}")
        train_dataset = load_tokenized_dataset_from_disk(
            args.dataset_path,
            split=args.split,
            max_examples=args.max_examples,
        )
    else:
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
    trainer = AMDOptimizedTrainerV2(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=args.output,
        resume_checkpoint=resume_checkpoint,
    )
    
    # Train!
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
