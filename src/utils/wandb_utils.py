"""
Wandb integration utilities
"""

import os
import wandb
from typing import Dict, Any, Optional, List
import json


def initialize_wandb(
    project: str,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    resume: Optional[str] = None,
    group: Optional[str] = None
) -> None:
    """
    Initialize Weights & Biases
    
    Args:
        project (str): Project name
        entity (Optional[str]): Entity (team/username)
        name (Optional[str]): Run name
        config (Optional[Dict[str, Any]]): Configuration dictionary
        tags (Optional[List[str]]): Tags for the run
        notes (Optional[str]): Notes for the run
        resume (Optional[str]): Resume mode ('allow', 'must', 'never')
        group (Optional[str]): Group name for organizing runs
    """
    wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        resume=resume,
        group=group
    )


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to wandb
    
    Args:
        metrics (Dict[str, float]): Metrics dictionary
        step (Optional[int]): Step number
    """
    wandb.log(metrics, step=step)


def log_model_metrics(
    train_loss: float,
    val_loss: float,
    val_accuracy: float,
    epoch: int,
    learning_rate: float,
    additional_metrics: Optional[Dict[str, float]] = None
) -> None:
    """
    Log model training metrics
    
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        val_accuracy (float): Validation accuracy
        epoch (int): Current epoch
        learning_rate (float): Current learning rate
        additional_metrics (Optional[Dict[str, float]]): Additional metrics
    """
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'epoch': epoch,
        'learning_rate': learning_rate
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    wandb.log(metrics)


def log_gradients(model, step: Optional[int] = None) -> None:
    """
    Log model gradients
    
    Args:
        model: PyTorch model
        step (Optional[int]): Step number
    """
    gradients = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[f"grad_{name}"] = param.grad.norm().item()
    
    if gradients:
        wandb.log(gradients, step=step)


def log_model_architecture(model, input_shape: tuple) -> None:
    """
    Log model architecture
    
    Args:
        model: PyTorch model
        input_shape (tuple): Input shape for the model
    """
    try:
        import torch
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Log model graph
        wandb.watch(model, log="all", log_freq=100)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            'model_total_params': total_params,
            'model_trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        })
        
    except ImportError:
        print("PyTorch not available for model logging")


def log_dataset_info(dataset_info: Dict[str, Any]) -> None:
    """
    Log dataset information
    
    Args:
        dataset_info (Dict[str, Any]): Dataset information
    """
    wandb.config.update({
        'dataset': dataset_info
    })


def log_hyperparameters(config: Dict[str, Any]) -> None:
    """
    Log hyperparameters
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    wandb.config.update(config)


def log_confusion_matrix(y_true, y_pred, class_names: Optional[List[str]] = None) -> None:
    """
    Log confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names (Optional[List[str]]): Class names
    """
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        wandb.log({'confusion_matrix': wandb.Image(plt)})
        plt.close()
        
    except ImportError:
        print("Required libraries not available for confusion matrix logging")


def log_learning_curve(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None
) -> None:
    """
    Log learning curves
    
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        train_accuracies (Optional[List[float]]): Training accuracies
        val_accuracies (Optional[List[float]]): Validation accuracies
    """
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(train_losses) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        axes[0].plot(epochs, train_losses, label='Training Loss')
        axes[0].plot(epochs, val_losses, label='Validation Loss')
        axes[0].set_title('Loss Curve')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        if train_accuracies and val_accuracies:
            axes[1].plot(epochs, train_accuracies, label='Training Accuracy')
            axes[1].plot(epochs, val_accuracies, label='Validation Accuracy')
            axes[1].set_title('Accuracy Curve')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        wandb.log({'learning_curves': wandb.Image(fig)})
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for learning curve logging")


def log_text_samples(
    texts: List[str],
    predictions: List[str],
    true_labels: List[str],
    title: str = "Sample Predictions"
) -> None:
    """
    Log text samples with predictions
    
    Args:
        texts (List[str]): Input texts
        predictions (List[str]): Predicted labels
        true_labels (List[str]): True labels
        title (str): Title for the table
    """
    table = wandb.Table(columns=["Text", "Prediction", "True Label", "Correct"])
    
    for text, pred, true_label in zip(texts, predictions, true_labels):
        is_correct = pred == true_label
        table.add_data(text[:200], pred, true_label, is_correct)
    
    wandb.log({title.lower().replace(' ', '_'): table})


def log_model_artifacts(
    model_path: str,
    model_name: str = "model",
    description: str = "Trained model",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log model as artifact
    
    Args:
        model_path (str): Path to model file
        model_name (str): Name of the model artifact
        description (str): Description of the model
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description,
        metadata=metadata
    )
    
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def log_dataset_artifacts(
    dataset_path: str,
    dataset_name: str = "dataset",
    description: str = "Training dataset",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log dataset as artifact
    
    Args:
        dataset_path (str): Path to dataset file/directory
        dataset_name (str): Name of the dataset artifact
        description (str): Description of the dataset
        metadata (Optional[Dict[str, Any]]): Additional metadata
    """
    artifact = wandb.Artifact(
        name=dataset_name,
        type="dataset",
        description=description,
        metadata=metadata
    )
    
    if os.path.isdir(dataset_path):
        artifact.add_dir(dataset_path)
    else:
        artifact.add_file(dataset_path)
    
    wandb.log_artifact(artifact)


def log_experiment_summary(
    best_metrics: Dict[str, float],
    training_time: float,
    total_epochs: int,
    early_stopping_epoch: Optional[int] = None
) -> None:
    """
    Log experiment summary
    
    Args:
        best_metrics (Dict[str, float]): Best metrics achieved
        training_time (float): Total training time in seconds
        total_epochs (int): Total number of epochs
        early_stopping_epoch (Optional[int]): Epoch where early stopping occurred
    """
    summary = {
        'best_' + k: v for k, v in best_metrics.items()
    }
    
    summary.update({
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'total_epochs': total_epochs
    })
    
    if early_stopping_epoch:
        summary['early_stopping_epoch'] = early_stopping_epoch
    
    wandb.summary.update(summary)


def finish_wandb() -> None:
    """Finish wandb run"""
    wandb.finish()


class WandbCallback:
    """
    Callback for logging to wandb during training
    """
    
    def __init__(self, log_freq: int = 10):
        self.log_freq = log_freq
        self.step = 0
    
    def on_batch_end(self, batch_idx: int, logs: Dict[str, float]) -> None:
        """Called at the end of each batch"""
        self.step += 1
        
        if self.step % self.log_freq == 0:
            wandb.log(logs, step=self.step)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]) -> None:
        """Called at the end of each epoch"""
        logs['epoch'] = epoch
        wandb.log(logs)
    
    def on_train_end(self, logs: Dict[str, float]) -> None:
        """Called at the end of training"""
        wandb.summary.update(logs)
