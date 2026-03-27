#!/usr/bin/env python3
"""
FAST training script using streaming data loaders for 800GB+ datasets.

Optimized for streaming arrow and parquet files with minimal preprocessing.

Two modes:
1. ON-THE-FLY: Stream raw arrow files, tokenize during training (5 min setup)
2. PRE-TOKENIZED: Stream pre-tokenized parquet files (fastest speed)

Usage:
    # Start training IMMEDIATELY with on-the-fly tokenization
    python scripts/train_streaming.py --arrow-dir ./data/indiccorp_raw --total-steps 100000
    
    # Use pre-tokenized data (faster)
    python scripts/train_streaming.py --tokenized-dir ./data/indiccorp_tokenized --total-steps 100000
    
    # Resume from checkpoint
    python scripts/train_streaming.py --arrow-dir ./data/indiccorp_raw --resume ./output/checkpoint-1000
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming_data import StreamConfig, load_hf_dataloader
from src.conversionV2 import load_mhc_model_v2, count_parameters_v2, convert_qwen3_to_mhc_v2
from transformers import AutoTokenizer
import torch.optim as optim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)




# ============================================================================
# Streaming Training Configuration (Optimized for Large Datasets)
# ============================================================================

@dataclass
class StreamingTrainingConfig:
    """Configuration for streaming training on 800GB+ datasets.
    
    Defaults favor large dataset streaming: mmap file reads, 
    minimal tokenization overhead, and efficient GPU utilization.
    """
    
    # Device settings
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    
    # Batch settings - tuned for 12GB @ 2048 tokens
    batch_size: int = 2  # Per-device batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 16
    max_length: int = 2048  # Sequence length
    
    # Memory optimizations
    gradient_checkpointing: bool = True
    pin_memory: bool = True  # Pinned host memory for PCIe GPUs
    
    # Dataloader settings
    num_workers: int = 8
    prefetch_factor: int = 2
    shuffle_buffer_size: int = 10000
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_8bit_optimizer: bool = False
    
    # Schedule
    warmup_ratio: float = 0.1
    total_steps: int = 10000
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 100
    output_dir: str = "./output/streaming_training"
    
    # Compilation
    use_compile: bool = False  # torch.compile (experimental with streaming)
    compile_mode: str = "default"
    
    # mHC V2 specific
    use_mhc: bool = True
    n_streams: int = 4
    num_fracs: int = 1
    sinkhorn_iters: int = 20
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def setup_nvidia_environment(target_dtype: torch.dtype = torch.bfloat16):
    """Setup environment for optimal NVIDIA CUDA performance."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(os.cpu_count() or 16))
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON,ATen")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
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
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        info["total_memory_gb"] = total_mem / (1024**3)
        
        try:
            free_mem, _ = torch.cuda.mem_get_info(0)
            info["free_memory_gb"] = free_mem / (1024**3)
        except Exception:
            pass
    
    return info


# ============================================================================
# Optimized Streaming Trainer
# ============================================================================

class StreamingTrainer:
    """Trainer optimized for streaming 800GB+ datasets."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: StreamingTrainingConfig,
        output_dir: str,
        resume_checkpoint: Optional[Path] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
        
        # Move model to device
        self.model = self.model.to(device=self.device)
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            try:
                self.model.model.gradient_checkpointing = True
                from torch.utils.checkpoint import checkpoint
                self.model.model._gradient_checkpointing_func = checkpoint
                self.logger.info("Gradient checkpointing enabled")
            except Exception as e:
                self.logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.loss_history = []
        self.step_times = []
        
        # Optionally resume
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer - 8-bit AdamW if available, else fused AdamW."""
        config = self.config
        
        # Separate parameters with/without weight decay
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
        
        # Try 8-bit AdamW first
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
        
        # Use fused AdamW if available
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                fused=True,
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
        """Run streaming training loop."""
        config = self.config
        
        self.logger.info("=" * 70)
        self.logger.info("STREAMING TRAINING - 800GB DATASET")
        self.logger.info("=" * 70)
        self.logger.info(f"Total steps: {config.total_steps}")
        self.logger.info(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
        self.logger.info(f"Sequence length: {config.max_length}")
        self.logger.info(f"Learning rate: {config.learning_rate}")
        if config.use_mhc:
            self.logger.info(f"mHC V2 streams: {config.n_streams}")
            self.logger.info(f"mHC V2 fracs: {config.num_fracs}")
        if self.global_step > 0:
            self.logger.info(f"Resuming at step {self.global_step}")
        
        if self.global_step >= config.total_steps:
            self.logger.info("Already completed training")
            return
        
        self.model.train()
        accumulation_loss = 0.0
        
        start_time = time.time()
        data_iter = iter(self.train_dataloader)
        
        pbar = tqdm(total=config.total_steps, initial=self.global_step, desc="Training", unit="step")
        
        while self.global_step < config.total_steps:
            step_start = time.time()
            
            # Accumulation loop
            for micro_step in range(config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                # Move to device
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                       dtype=config.dtype):
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
            self.optimizer.zero_grad(set_to_none=True)
            
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
            
            # Checkpoint
            if self.global_step % config.save_interval == 0:
                self._save_checkpoint()
        
        pbar.close()
        total_time = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Average step time: {sum(self.step_times)/len(self.step_times):.3f}s")
        self.logger.info("=" * 70)
        
        self._save_checkpoint(final=True)
    
    def _load_checkpoint(self, checkpoint_dir: Path):
        """Load training state from checkpoint."""
        state_path = checkpoint_dir / "training_state.pt"
        if not state_path.exists():
            self.logger.warning(f"No training_state.pt in {checkpoint_dir}, starting fresh")
            return
        
        try:
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state.get("optimizer_state_dict", {}))
            self.scheduler.load_state_dict(state.get("scheduler_state_dict", {}))
            self.global_step = int(state.get("global_step", 0))
            self.loss_history = state.get("loss_history", [])
            self.step_times = state.get("step_times", [])
            self.logger.info(f"Resumed from {checkpoint_dir} at step {self.global_step}")
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
    
    def _log_progress(self, loss: float, grad_norm: float, step_time: float):
        """Log training progress."""
        lr = self.scheduler.get_last_lr()[0]
        
        # Throughput
        tokens_per_step = self.config.effective_batch_size * self.config.max_length
        tokens_per_sec = tokens_per_step / step_time
        
        # Memory usage
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
    
    def _save_checkpoint(self, final: bool = False):
        """Save checkpoint."""
        if final:
            save_dir = self.output_dir / "final_model"
        else:
            save_dir = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod
        
        model_to_save.save_pretrained(save_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "step_times": self.step_times,
        }
        torch.save(state, save_dir / "training_state.pt")
        
        if not final:
            self.logger.info(f"Saved checkpoint to {save_dir}")





def main():
    parser = argparse.ArgumentParser(
        description="Fast streaming training for 800GB+ datasets",
    )
    
    parser.add_argument(
        "--arrow-dir",
        type=str,
        default=None,
        help="Directory with raw arrow files (on-the-fly tokenization)",
    )
    
    parser.add_argument(
        "--tokenized-dir",
        type=str,
        default=None,
        help="Directory with pre-tokenized parquet files (fastest)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total training steps",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max sequence length",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/streaming_training",
        help="Output directory",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name",
    )
    
    parser.add_argument(
        "--mhc-model-path",
        type=str,
        default=None,
        help="Path to existing mHC model (if provided, uses this instead of creating new one)",
    )
    
    parser.add_argument(
        "--use-mhc",
        action="store_true",
        default=True,
        help="Use mHC V2 model",
    )
    
    parser.add_argument(
        "--no-mhc",
        action="store_true",
        help="Don't use mHC (use base model)",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval",
    )
    
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Checkpoint save interval",
    )
    
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_nvidia_environment()
    
    # Validate inputs
    if not args.arrow_dir and not args.tokenized_dir:
        parser.error("Must provide --arrow-dir or --tokenized-dir")
    
    # Config
    config = StreamingTrainingConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        total_steps=args.total_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        warmup_ratio=args.warmup_ratio,
        use_mhc=args.use_mhc and not args.no_mhc,
    )
    
    # Device info
    device_info = get_device_info()
    logger.info("=" * 70)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 70)
    for key, value in device_info.items():
        logger.info(f"{key}: {value}")
    logger.info("")
    
    # Load model
    logger.info("Loading model...")
    if config.use_mhc:
        if args.mhc_model_path:
            # Load existing mHC model
            logger.info(f"Loading existing mHC model from {args.mhc_model_path}")
            model, _ = load_mhc_model_v2(
                model_path=args.mhc_model_path,
                device=config.device,
                torch_dtype=config.dtype,
                base_model=args.model,
            )
        else:
            # Convert from base model to create new mHC model
            model, _ = convert_qwen3_to_mhc_v2(
                model_name_or_path=args.model,
                n_streams=config.n_streams,
                num_fracs=config.num_fracs,
                sinkhorn_iters=config.sinkhorn_iters,
                device=config.device,
                torch_dtype=config.dtype,
                validate=False,  # Skip validation for speed
            )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=config.dtype,
            device_map="auto",
        )
    
    total_params = count_parameters_v2(model)
    logger.info(f"Model parameters: {total_params}")
    logger.info("")
    
    # Load tokenizer (for on-the-fly tokenization)
    if args.arrow_dir:
        logger.info("Loading tokenizer for on-the-fly tokenization...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
    
    # Create dataloader
    logger.info("Creating dataloader...")
    stream_config = StreamConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_length=config.max_length,
    )
    
    if args.arrow_dir:
        logger.info(f"Loading HF dataset from {args.arrow_dir}")
        dataloader = load_hf_dataloader(args.arrow_dir, tokenizer, stream_config)
    elif args.tokenized_dir:
        logger.info(f"Loading HF dataset from {args.tokenized_dir}")
        dataloader = load_hf_dataloader(args.tokenized_dir, None, stream_config)
    else:
        raise ValueError("Must provide --arrow-dir or --tokenized-dir")
    
    logger.info("")
    
    # Create trainer and run
    trainer = StreamingTrainer(
        model=model,
        train_dataloader=dataloader,
        config=config,
        output_dir=config.output_dir,
        resume_checkpoint=Path(args.resume) if args.resume else None,
    )
    
    trainer.train()
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Training complete. Output: {config.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
