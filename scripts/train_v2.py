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
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MHCTrainingConfig
from src.conversionV2 import load_mhc_model_v2, count_parameters_v2, convert_qwen3_to_mhc_v2
from src.qwen3_mhc_modelV2 import Qwen3MHCForCausalLMV2


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
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = StepLRScheduler(self.optimizer, config)
        
        # Tracking
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with mHC-paper settings."""
        config = self.config
        
        # Separate mHC and original parameters for potential different treatment
        if hasattr(self.model, 'get_mhc_parameters'):
            mhc_params = self.model.get_mhc_parameters()
            original_params = self.model.get_original_parameters()
            
            param_groups = [
                {"params": original_params, "weight_decay": config.weight_decay},
                {"params": mhc_params, "weight_decay": config.weight_decay},
            ]
        else:
            param_groups = [
                {"params": self.model.parameters(), "weight_decay": config.weight_decay}
            ]
        
        return AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
    
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
        
        epoch = 0
        while self.global_step < self.config.total_steps:
            epoch += 1
            self.logger.info(f"Starting epoch {epoch}")
            
            for batch in self.train_dataloader:
                if self.global_step >= self.config.total_steps:
                    break
                
                loss, grad_norm = self._training_step(batch)
                
                # Check for instability
                if self._check_instability(loss):
                    self.logger.error("Training instability detected! Stopping.")
                    return
                
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
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> tuple[float, float]:
        """Execute single training step."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )
        
        # Check gradient norm
        if grad_norm > self.config.gradient_norm_threshold:
            self.logger.warning(
                f"Step {self.global_step}: Large gradient norm {grad_norm:.2f}"
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item(), float(grad_norm)
    
    def _check_instability(self, loss: float) -> bool:
        """Check for training instability."""
        # NaN/Inf check
        if not math.isfinite(loss):
            self.logger.error(f"Non-finite loss at step {self.global_step}: {loss}")
            return True
        
        # Loss spike check
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
            recent_mean = sum(self.loss_history[-10:]) / 10
            earlier_mean = sum(self.loss_history[:50]) / 50
            
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
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Evaluation - Step {self.global_step}: loss={avg_loss:.4f}")
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(best=True)
        
        self.model.train()
    
    def _save_checkpoint(self, best: bool = False, final: bool = False):
        """Save model checkpoint."""
        if final:
            checkpoint_dir = os.path.join(self.output_dir, "final")
        elif best:
            checkpoint_dir = os.path.join(self.output_dir, "best")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
        }
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")


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
    
    # Check if it's a pre-tokenized Arrow dataset (from download.py)
    arrow_path = Path(data_path)
    if arrow_path.exists() and arrow_path.is_dir():
        # Check for Arrow dataset markers
        if (arrow_path / "dataset_info.json").exists() or (arrow_path / "state.json").exists():
            logger.info("Detected pre-tokenized Arrow dataset, loading directly...")
            dataset = load_from_disk(str(arrow_path))
            
            # Verify it has required columns
            required_cols = {"input_ids", "attention_mask", "labels"}
            if required_cols.issubset(set(dataset.column_names)):
                logger.info("Dataset already tokenized, skipping tokenization")
                dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                )
                
                logger.info(f"Dataset loaded: {len(dataset)} examples, {len(dataloader)} batches")
                return dataloader
            else:
                logger.warning(f"Arrow dataset missing columns, will re-tokenize. Found: {dataset.column_names}")
    
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
        batch_size=batch_size,
        shuffle=(split == "train"),
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
    if args.model is not None:
        logger.info(f"Loading existing mHC V2 model from {args.model}")
        model, tokenizer = load_mhc_model_v2(
            args.model,
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
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
