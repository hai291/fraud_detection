"""
Trainer class for training models with Accelerate integration
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, Callable, List, Tuple
import time
import json
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.wandb_utils import WandbLogger
from ..evaluation.evaluator import Evaluator
from ..evaluation.metrics import MetricCalculator


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.monitor_op = torch.lt if mode == 'min' else torch.gt
        self.min_delta *= 1 if mode == 'min' else -1
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Main trainer class with Accelerate integration for distributed training
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        accelerator: Optional[Accelerator] = None,
        wandb_logger: Optional[WandbLogger] = None,
        evaluator: Optional[Evaluator] = None,
        metric_calculator: Optional[MetricCalculator] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config or {}
        self.callbacks = callbacks or []
        
        # Initialize accelerator
        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
                mixed_precision=self.config.get('mixed_precision', 'no'),
                log_with=self.config.get('log_with', None),
                project_dir=self.config.get('output_dir', './outputs')
            )
        else:
            self.accelerator = accelerator
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Initialize wandb logger
        self.wandb_logger = wandb_logger
        
        # Initialize evaluator and metric calculator
        self.evaluator = evaluator
        self.metric_calculator = metric_calculator
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.training_history = []
        
        # Early stopping
        early_stopping_config = self.config.get('early_stopping', {})
        if early_stopping_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.0),
                mode=early_stopping_config.get('mode', 'min')
            )
        else:
            self.early_stopping = None
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare model, optimizer, and dataloaders with accelerator
        self._prepare_for_training()
    
    def _prepare_for_training(self):
        """Prepare model, optimizer, and dataloaders for training"""
        # Set seed for reproducibility
        if 'seed' in self.config:
            set_seed(self.config['seed'])
        
        # Prepare with accelerator
        components = [self.model, self.train_dataloader]
        
        if self.optimizer is not None:
            components.append(self.optimizer)
        
        if self.scheduler is not None:
            components.append(self.scheduler)
        
        if self.val_dataloader is not None:
            components.append(self.val_dataloader)
        
        prepared = self.accelerator.prepare(*components)
        
        # Unpack prepared components
        self.model = prepared[0]
        self.train_dataloader = prepared[1]
        idx = 2
        
        if self.optimizer is not None:
            self.optimizer = prepared[idx]
            idx += 1
        
        if self.scheduler is not None:
            self.scheduler = prepared[idx]
            idx += 1
        
        if self.val_dataloader is not None:
            self.val_dataloader = prepared[idx]
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config,
            'training_history': self.training_history
        }
        
        self.accelerator.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(self.checkpoint_dir / 'best_model.pt')
            self.accelerator.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model state
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                if self.loss_fn is not None:
                    if isinstance(outputs, dict):
                        loss = self.loss_fn(outputs['logits'], batch['labels'])
                    else:
                        loss = self.loss_fn(outputs, batch['labels'])
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Update parameters
                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update progress
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Log metrics
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': current_lr,
                    'step': self.global_step
                })
                
                # Log to wandb
                if self.wandb_logger and self.global_step % self.config.get('logging_steps', 100) == 0:
                    self.wandb_logger.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': current_lr,
                        'train/step': self.global_step
                    })
                
                # Run callbacks
                for callback in self.callbacks:
                    callback(self, batch_idx, loss.item())
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def evaluate(self, dataloader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating ({prefix})",
            disable=not self.accelerator.is_local_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                if self.loss_fn is not None:
                    if isinstance(outputs, dict):
                        loss = self.loss_fn(outputs['logits'], batch['labels'])
                        logits = outputs['logits']
                    else:
                        loss = self.loss_fn(outputs, batch['labels'])
                        logits = outputs
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(self.accelerator.gather(predictions).cpu().numpy())
                all_labels.extend(self.accelerator.gather(batch['labels']).cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {f'{prefix}_loss': avg_loss}
        
        # Calculate additional metrics if metric calculator is available
        if self.metric_calculator:
            additional_metrics = self.metric_calculator.calculate_metrics(all_predictions, all_labels)
            metrics.update({f'{prefix}_{k}': v for k, v in additional_metrics.items()})
        
        return metrics
    
    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None):
        """Main training loop"""
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Total training steps: {len(self.train_dataloader) * num_epochs}")
        
        # Initialize wandb if configured
        if self.wandb_logger:
            self.wandb_logger.watch(self.model)
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self.evaluate(self.val_dataloader, 'val')
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log epoch metrics
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs} - " + 
                           " - ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
            
            # Log to wandb
            if self.wandb_logger:
                self.wandb_logger.log({
                    **epoch_metrics,
                    'epoch': epoch + 1
                })
            
            # Save training history
            self.training_history.append({
                'epoch': epoch + 1,
                'metrics': epoch_metrics,
                'timestamp': time.time()
            })
            
            # Save checkpoint
            checkpoint_path = str(self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')
            is_best = False
            
            # Check if this is the best model
            monitor_metric = self.config.get('monitor_metric', 'val_loss')
            if monitor_metric in epoch_metrics:
                current_metric = epoch_metrics[monitor_metric]
                if self.best_metric is None or \
                   (self.config.get('monitor_mode', 'min') == 'min' and current_metric < self.best_metric) or \
                   (self.config.get('monitor_mode', 'min') == 'max' and current_metric > self.best_metric):
                    self.best_metric = current_metric
                    is_best = True
            
            # Save checkpoint
            if self.accelerator.is_main_process:
                self.save_checkpoint(checkpoint_path, is_best)
            
            # Early stopping
            if self.early_stopping and monitor_metric in epoch_metrics:
                if self.early_stopping(epoch_metrics[monitor_metric]):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Final evaluation on test set
        if self.test_dataloader is not None:
            test_metrics = self.evaluate(self.test_dataloader, 'test')
            self.logger.info("Test Results: " + 
                           " - ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))
            
            if self.wandb_logger:
                self.wandb_logger.log(test_metrics)
        
        # Training summary
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        if self.accelerator.is_main_process:
            final_checkpoint_path = str(self.checkpoint_dir / 'final_model.pt')
            self.save_checkpoint(final_checkpoint_path)
        
        # Close wandb
        if self.wandb_logger:
            self.wandb_logger.finish()
    
    def predict(self, dataloader: DataLoader) -> Tuple[List, List]:
        """Make predictions on a dataset"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                outputs = self.model(**batch)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(self.accelerator.gather(predictions).cpu().numpy())
                all_probabilities.extend(self.accelerator.gather(probabilities).cpu().numpy())
        
        return all_predictions, all_probabilities
    
    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history
    
    def save_training_history(self, filepath: str):
        """Save training history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved to {filepath}")


class MultiTaskTrainer(Trainer):
    """
    Trainer for multi-task learning
    """
    
    def __init__(self, *args, task_weights: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with multi-task support"""
        self.model.train()
        task_losses = {}
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate multi-task loss
                if isinstance(outputs, dict) and 'task_losses' in outputs:
                    losses = outputs['task_losses']
                    weighted_loss = 0
                    
                    for task, loss in losses.items():
                        weight = self.task_weights.get(task, 1.0)
                        weighted_loss += weight * loss
                        
                        if task not in task_losses:
                            task_losses[task] = 0
                        task_losses[task] += loss.item()
                    
                    loss = weighted_loss
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Update parameters
                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update progress
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
        
        # Calculate average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {'train_loss': avg_loss}
        
        # Add task-specific losses
        for task, total_task_loss in task_losses.items():
            metrics[f'train_{task}_loss'] = total_task_loss / num_batches
        
        return metrics
