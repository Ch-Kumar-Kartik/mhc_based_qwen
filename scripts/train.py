#!/usr/bin/env python3
"""
Training script for mHC Qwen3 models.

This script implements the training procedure described in the mHC paper,
including:
- Learning rate scheduling with step decay
- Gradient checkpointing for memory efficiency
- Monitoring of mHC-specific metrics
- Stability checks and early stopping

Usage:
    python -m scripts.train --config configs/qwen3_0.6b_mhc.yaml
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
from src.conversion import load_mhc_model, count_parameters


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


class MHCTrainer:
    """Trainer for mHC models with stability monitoring."""
    
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
            self.model.gradient_checkpointing_enable()
        
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
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        
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
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
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


def main():
    parser = argparse.ArgumentParser(description="Train mHC model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to mHC model",
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
    
    args = parser.parse_args()
    
    setup_logging(args.output, args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config(args.config)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model, tokenizer = load_mhc_model(args.model)
    
    # Print model summary
    param_counts = count_parameters(model)
    logger.info(f"Model loaded: {param_counts['total']:,} total params")
    logger.info(f"  Original: {param_counts['original']:,}")
    logger.info(f"  mHC:      {param_counts['mhc']:,} ({param_counts['mhc_percentage']:.1f}%)")
    
    # TODO: Load dataset and create dataloaders
    # This would be customized based on the specific training data
    logger.info("Note: Dataset loading not implemented - please provide your own DataLoader")
    
    # Example of how training would be started:
    # trainer = MHCTrainer(
    #     model=model,
    #     config=config,
    #     train_dataloader=train_dataloader,
    #     eval_dataloader=eval_dataloader,
    #     output_dir=args.output,
    # )
    # trainer.train()
    
    logger.info("Training script loaded successfully. Implement data loading to start training.")


if __name__ == "__main__":
    main()
