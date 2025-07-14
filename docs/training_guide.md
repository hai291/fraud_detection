# Training Guide

## Overview
This document provides comprehensive guidance for training models in this research framework.

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd base-research-repo

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
uv pip install accelerate wandb torch transformers datasets
```

### 2. Prepare Data
```bash
# Process raw data
bash scripts/preprocess.sh
```

### 3. Configure Training
Edit configuration files in `config/`:
- `model_config.yaml`: Model architecture settings
- `training_config.yaml`: Training hyperparameters
- `data_config.yaml`: Data processing settings

### 4. Start Training
```bash
# Configure accelerate (first time only)
accelerate config

# Start training
bash scripts/train.sh
```

## Detailed Training Process

### 1. Data Preparation

#### Data Format
Ensure your data is in the correct format:
```json
[
  {
    "text": "This is a sample text",
    "label": 0
  },
  {
    "text": "Another sample text",
    "label": 1
  }
]
```

#### Data Preprocessing
```bash
python src/data/preprocessing.py \
    --input_dir dataset/raw \
    --output_dir dataset/processed \
    --config config/data_config.yaml
```

### 2. Model Configuration

#### Transformer Model
```yaml
model:
  name: "transformer_model"
  architecture: "transformer"
  pretrained_model: "bert-base-uncased"
  num_labels: 2
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
```

#### CNN Model
```yaml
model:
  name: "cnn_model"
  architecture: "cnn"
  vocab_size: 30522
  embed_size: 128
  num_filters: 100
  filter_sizes: [3, 4, 5]
  num_labels: 2
  dropout_prob: 0.5
```

### 3. Training Configuration

#### Basic Settings
```yaml
training:
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 10
  warmup_steps: 500
  weight_decay: 0.01
  
  # Gradient settings
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  
  # Mixed precision
  fp16: true
  
  # Logging
  logging_steps: 100
  eval_steps: 500
```

#### Advanced Settings
```yaml
training:
  # Optimizer
  optimizer: "adamw"
  adam_epsilon: 1e-8
  adam_beta1: 0.9
  adam_beta2: 0.999
  
  # Scheduler
  scheduler_type: "linear"
  
  # Early stopping
  early_stopping_patience: 3
  early_stopping_threshold: 0.01
  
  # Checkpointing
  save_steps: 1000
  save_total_limit: 3
```

### 4. Accelerate Configuration

#### Single GPU
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

#### Multi-GPU
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4  # Number of GPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 5. Wandb Integration

#### Setup
```bash
# Login to wandb
wandb login
```

#### Configuration
```yaml
wandb:
  project: "your-project-name"
  entity: "your-wandb-entity"
  name: null  # Auto-generated
  tags: ["experiment", "baseline"]
  notes: "Baseline model training"
```

#### Tracking Metrics
The framework automatically tracks:
- Training loss
- Validation loss
- Validation accuracy
- Learning rate
- Gradient norm
- Model parameters

## Training Strategies

### 1. Hyperparameter Tuning

#### Learning Rate Finding
```python
from src.utils.lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_dataloader, end_lr=1, num_iter=100)
lr_finder.plot()
```

#### Grid Search
```python
import itertools

learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
batch_sizes = [16, 32, 64]
dropout_rates = [0.1, 0.3, 0.5]

for lr, bs, dr in itertools.product(learning_rates, batch_sizes, dropout_rates):
    config = {
        'learning_rate': lr,
        'batch_size': bs,
        'dropout_prob': dr
    }
    # Run training with config
```

### 2. Transfer Learning

#### Fine-tuning Strategy
1. **Freeze base model**: Train only classification head
2. **Unfreeze gradually**: Unfreeze layers progressively
3. **Different learning rates**: Use different LR for different layers

```python
# Freeze base model
model.freeze_parameters(['transformer'])

# Train classification head
trainer.train(epochs=2)

# Unfreeze and continue training
model.unfreeze_parameters(['transformer'])
trainer.optimizer.param_groups[0]['lr'] = 1e-5  # Lower LR for base model
trainer.train(epochs=8)
```

### 3. Regularization Techniques

#### Dropout
```yaml
model:
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
```

#### Weight Decay
```yaml
training:
  weight_decay: 0.01
```

#### Gradient Clipping
```yaml
training:
  max_grad_norm: 1.0
```

#### Early Stopping
```yaml
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.01
```

### 4. Data Augmentation

#### Text Augmentation
```yaml
data:
  augmentation:
    enabled: true
    techniques:
      - "synonym_replacement"
      - "random_insertion"
      - "random_swap"
      - "random_deletion"
    probability: 0.1
```

#### Implementation
```python
from src.data.augmentation import DataAugmenter

augmenter = DataAugmenter(augmentation_prob=0.1)
augmented_data = augmenter.augment_dataset(
    data=train_data,
    text_column="text",
    augmentation_factor=1
)
```

## Monitoring and Debugging

### 1. Training Metrics

#### Loss Curves
Monitor training and validation loss:
- **Decreasing**: Good training progress
- **Plateauing**: Need to adjust learning rate
- **Increasing**: Overfitting or learning rate too high

#### Validation Metrics
Track validation accuracy and F1-score:
- **Improving**: Good generalization
- **Plateauing**: Potential overfitting
- **Decreasing**: Definite overfitting

### 2. Common Issues

#### Overfitting
**Symptoms**: Training loss decreases but validation loss increases
**Solutions**:
- Increase regularization (dropout, weight decay)
- Reduce model complexity
- Use early stopping
- Add more training data

#### Underfitting
**Symptoms**: Both training and validation loss are high
**Solutions**:
- Increase model complexity
- Reduce regularization
- Increase learning rate
- Train for more epochs

#### Slow Convergence
**Symptoms**: Loss decreases very slowly
**Solutions**:
- Increase learning rate
- Use learning rate scheduling
- Check gradient flow
- Improve data quality

### 3. Performance Optimization

#### Memory Optimization
```yaml
training:
  gradient_accumulation_steps: 4  # Simulate larger batch size
  fp16: true  # Use mixed precision
  dataloader_num_workers: 4  # Optimize data loading
```

#### Speed Optimization
```python
# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# Optimize data loading
train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## Advanced Techniques

### 1. Curriculum Learning
Start with easier examples and gradually increase difficulty:
```python
from src.training.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    initial_difficulty=0.3,
    final_difficulty=1.0,
    num_epochs=10
)
```

### 2. Multi-task Learning
Train on multiple related tasks simultaneously:
```python
from src.training.multitask import MultiTaskTrainer

trainer = MultiTaskTrainer(
    models={'task1': model1, 'task2': model2},
    datasets={'task1': dataset1, 'task2': dataset2},
    loss_weights={'task1': 0.7, 'task2': 0.3}
)
```

### 3. Knowledge Distillation
Train a smaller model using a larger teacher model:
```python
from src.training.distillation import DistillationTrainer

trainer = DistillationTrainer(
    student_model=small_model,
    teacher_model=large_model,
    temperature=4.0,
    alpha=0.7
)
```

## Evaluation and Testing

### 1. Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold

# K-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X, y):
    # Train and validate
    pass
```

### 2. Test Set Evaluation
```bash
# Run evaluation
python src/evaluation/evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_config config/data_config.yaml \
    --output_dir evaluation_results
```

### 3. Statistical Significance
```python
from scipy import stats

# Compare two models
scores1 = [0.85, 0.87, 0.86, 0.88, 0.85]
scores2 = [0.82, 0.84, 0.83, 0.85, 0.82]

t_stat, p_value = stats.ttest_rel(scores1, scores2)
print(f"p-value: {p_value}")
```

## Troubleshooting

### Common Errors

#### CUDA Out of Memory
```bash
# Solutions:
# 1. Reduce batch size
# 2. Use gradient accumulation
# 3. Use mixed precision
# 4. Clear cache
torch.cuda.empty_cache()
```

#### Slow Training
```bash
# Check:
# 1. Data loading bottleneck
# 2. GPU utilization
# 3. Model complexity
# 4. Batch size optimization
```

#### NaN Loss
```bash
# Causes:
# 1. Learning rate too high
# 2. Gradient explosion
# 3. Numerical instability
# 4. Bad data

# Solutions:
# 1. Reduce learning rate
# 2. Use gradient clipping
# 3. Check data quality
# 4. Use mixed precision carefully
```

### Debugging Tools

#### Gradient Checking
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

#### Model Inspection
```python
# Check model structure
print(model)

# Check parameter counts
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Check device placement
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
```

## Best Practices

### 1. Reproducibility
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2. Checkpointing
```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'config': config
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 3. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

### 4. Version Control
```bash
# Always commit your config files
git add config/
git commit -m "Add training configuration for experiment X"

# Tag important experiments
git tag -a v1.0-baseline -m "Baseline model results"
```

## References

1. Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks.
2. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification.
3. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
