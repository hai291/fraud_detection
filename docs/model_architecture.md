# Model Architecture

## Overview
This document describes the model architectures used in this research project.

## Base Model Architecture

### Design Principles
- **Modularity**: All models inherit from a base class for consistency
- **Flexibility**: Easy to configure and extend for different tasks
- **Efficiency**: Optimized for training and inference performance
- **Reproducibility**: Deterministic behavior with proper seeding

### Base Model Class
```python
class BaseModel(nn.Module, ABC):
    def __init__(self, config: Dict[str, Any])
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
    def get_num_parameters(self) -> int
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> None
```

## Transformer Model

### Architecture Overview
The transformer model is based on the BERT architecture with a classification head.

```
Input Text
    ↓
[CLS] Token1 Token2 ... TokenN [SEP]
    ↓
Embedding Layer (vocab_size → hidden_size)
    ↓
Positional Encoding
    ↓
Transformer Encoder Layers (×12)
    ├── Multi-Head Self-Attention
    ├── Add & Norm
    ├── Feed-Forward Network
    └── Add & Norm
    ↓
Pooler Layer ([CLS] representation)
    ↓
Dropout Layer
    ↓
Classification Head (hidden_size → num_labels)
    ↓
Output Logits
```

### Key Components

#### 1. Embedding Layer
- **Vocabulary size**: 30,522 (BERT tokenizer)
- **Hidden size**: 768
- **Position embeddings**: Up to 512 tokens
- **Token type embeddings**: For sentence pair tasks

#### 2. Transformer Encoder
- **Number of layers**: 12
- **Number of attention heads**: 12
- **Intermediate size**: 3,072
- **Activation function**: GELU
- **Dropout rate**: 0.1

#### 3. Classification Head
- **Input size**: 768 (hidden_size)
- **Output size**: Number of classes
- **Activation**: None (raw logits)

### Configuration Example
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

### Usage
```python
from src.models.transformer_model import TransformerModel

model = TransformerModel(config)
outputs = model({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels  # optional
})
```

## CNN Model

### Architecture Overview
The CNN model uses multiple convolutional filters to capture n-gram features.

```
Input Text
    ↓
Token IDs
    ↓
Embedding Layer (vocab_size → embed_size)
    ↓
Multiple Conv1D Layers (filter_sizes: [3, 4, 5])
    ├── Conv1D + ReLU
    ├── Max Pooling
    └── Feature Extraction
    ↓
Concatenate Features
    ↓
Dropout Layer
    ↓
Linear Classification Layer
    ↓
Output Logits
```

### Key Components

#### 1. Embedding Layer
- **Vocabulary size**: 30,522
- **Embedding size**: 128 (configurable)
- **Initialization**: Uniform distribution [-0.1, 0.1]

#### 2. Convolutional Layers
- **Filter sizes**: [3, 4, 5] (configurable)
- **Number of filters**: 100 per size
- **Activation**: ReLU
- **Pooling**: Max pooling over sequence length

#### 3. Classification Head
- **Input size**: Total filters (300 with default config)
- **Output size**: Number of classes
- **Dropout**: 0.5

### Configuration Example
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

## Advanced TextCNN Model

### Architecture Overview
Enhanced CNN model with attention mechanism and batch normalization.

```
Input Text
    ↓
Embedding Layer
    ↓
Self-Attention Layer
    ├── Multi-Head Attention
    └── Global Average Pooling
    ↓
CNN Branch
    ├── Multiple Conv1D + BatchNorm + ReLU
    ├── Max Pooling
    └── Adaptive Pooling
    ↓
Feature Combination
    ├── Concatenate [CNN Features, Attention Features]
    └── Linear + ReLU + Dropout
    ↓
Classification Layers
    ├── Linear + ReLU + Dropout
    └── Final Linear Layer
    ↓
Output Logits
```

### Key Features
1. **Multi-head attention**: Captures long-range dependencies
2. **Batch normalization**: Stabilizes training
3. **Adaptive pooling**: Handles variable sequence lengths
4. **Feature combination**: Combines CNN and attention features
5. **Deep classification**: Multiple linear layers for better representation

## Model Comparison

| Model | Parameters | Training Time | Inference Speed | Accuracy |
|-------|------------|---------------|-----------------|----------|
| Transformer | ~110M | High | Medium | High |
| CNN | ~5M | Low | High | Medium |
| TextCNN | ~10M | Medium | High | Medium-High |

## Training Considerations

### Memory Requirements
- **Transformer**: ~8GB GPU memory for batch size 16
- **CNN**: ~2GB GPU memory for batch size 64
- **TextCNN**: ~4GB GPU memory for batch size 32

### Optimization
- **Transformer**: AdamW with linear warmup
- **CNN**: Adam with step decay
- **TextCNN**: AdamW with cosine annealing

### Regularization
- **Dropout**: Applied to all models
- **Weight decay**: L2 regularization
- **Gradient clipping**: Prevents exploding gradients

## Hyperparameter Tuning

### Search Space
```python
search_space = {
    'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
    'batch_size': [16, 32, 64],
    'dropout_prob': [0.1, 0.3, 0.5],
    'weight_decay': [0.01, 0.1, 0.2]
}
```

### Tuning Strategy
1. **Grid search**: For small search spaces
2. **Random search**: For larger search spaces
3. **Bayesian optimization**: For expensive evaluations

## Model Selection

### Criteria
1. **Validation accuracy**: Primary metric
2. **Training time**: Practical considerations
3. **Inference speed**: Deployment requirements
4. **Memory usage**: Hardware constraints
5. **Interpretability**: Domain requirements

### Ensemble Methods
- **Voting**: Majority vote from multiple models
- **Stacking**: Meta-learner combines predictions
- **Bagging**: Bootstrap aggregating for robustness

## References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
3. Vaswani, A., et al. (2017). Attention is All You Need.
