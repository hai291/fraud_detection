"""
Main training script
"""

import os
import argparse
import yaml
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any

from ..data.dataset import CustomDataset
from ..models.transformer_model import TransformerModel
from ..models.cnn_model import CNNModel
from ..utils.config import load_config
from ..utils.logging import setup_logging
from ..utils.wandb_utils import initialize_wandb
from .trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training script')
    
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model configuration file')
    parser.add_argument('--training_config', type=str, required=True,
                        help='Path to training configuration file')
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to data configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--logging_dir', type=str, default='logs',
                        help='Logging directory')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_model(model_config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on configuration"""
    model_type = model_config.get('architecture', 'transformer')
    
    if model_type == 'transformer':
        return TransformerModel(model_config)
    elif model_type == 'cnn':
        return CNNModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_datasets(data_config: Dict[str, Any]) -> tuple:
    """Create train and validation datasets"""
    tokenizer_name = data_config['tokenizer']['name']
    max_length = data_config['max_length']
    
    train_dataset = CustomDataset(
        data_path=data_config['train_path'],
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        text_column=data_config['dataset']['text_column'],
        label_column=data_config['dataset']['label_column']
    )
    
    val_dataset = CustomDataset(
        data_path=data_config['val_path'],
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        text_column=data_config['dataset']['text_column'],
        label_column=data_config['dataset']['label_column']
    )
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    training_config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> tuple:
    """Create train and validation dataloaders"""
    batch_size = training_config['batch_size']
    num_workers = data_config.get('num_workers', 4)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_config: Dict[str, Any],
    num_training_steps: int
):
    """Create optimizer and learning rate scheduler"""
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        eps=training_config.get('adam_epsilon', 1e-8),
        betas=(training_config.get('adam_beta1', 0.9), 
               training_config.get('adam_beta2', 0.999))
    )
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    data_config = load_config(args.data_config)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        mixed_precision=training_config.get('mixed_precision', 'fp16'),
        log_with="wandb" if training_config.get('wandb', {}).get('project') else None,
        project_dir=args.output_dir
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        log_file=os.path.join(args.logging_dir, "training.log")
    )
    
    # Initialize wandb
    if training_config.get('wandb', {}).get('project'):
        initialize_wandb(
            project=training_config['wandb']['project'],
            entity=training_config['wandb'].get('entity'),
            name=training_config['wandb'].get('name'),
            config={
                'model_config': model_config,
                'training_config': training_config,
                'data_config': data_config
            }
        )
    
    # Create model
    model = create_model(model_config['model'])
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(data_config['data'])
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, val_dataset, training_config['training'], data_config['data']
    )
    
    # Calculate total training steps
    num_training_steps = len(train_dataloader) * training_config['training']['num_epochs']
    num_training_steps //= training_config['training'].get('gradient_accumulation_steps', 1)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, training_config['training'], num_training_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=training_config['training'],
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        logger=logger
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("final_model")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
