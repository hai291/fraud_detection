"""
CNN-based models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from .base_model import BaseModel


class CNNModel(BaseModel):
    """
    CNN model for text classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config.get('vocab_size', 30522)
        self.embed_size = config.get('embed_size', 128)
        self.num_filters = config.get('num_filters', 100)
        self.filter_sizes = config.get('filter_sizes', [3, 4, 5])
        self.num_labels = config.get('num_labels', 2)
        self.dropout_prob = config.get('dropout_prob', 0.5)
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.embed_size))
            for k in self.filter_sizes
        ])
        
        # Dropout and classification
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(
            len(self.filter_sizes) * self.num_filters,
            self.num_labels
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs (Dict[str, torch.Tensor]): Input tensors
                - input_ids: (batch_size, seq_len)
                - labels: (batch_size,) - optional
                
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        input_ids = inputs['input_ids']
        labels = inputs.get('labels', None)
        
        # Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_size)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_size)
        
        # Convolution and pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, seq_len - k + 1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len - k + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all features
        features = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters)
        
        # Classification
        features = self.dropout(features)
        logits = self.classifier(features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'features': features
        }


class TextCNN(BaseModel):
    """
    Advanced TextCNN model with multiple techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config.get('vocab_size', 30522)
        self.embed_size = config.get('embed_size', 300)
        self.num_filters = config.get('num_filters', 128)
        self.filter_sizes = config.get('filter_sizes', [2, 3, 4, 5])
        self.num_labels = config.get('num_labels', 2)
        self.dropout_prob = config.get('dropout_prob', 0.5)
        self.use_pretrained = config.get('use_pretrained_embeddings', False)
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        if self.use_pretrained:
            # Freeze pretrained embeddings
            self.embedding.weight.requires_grad = False
        
        # Multiple convolutional layers
        self.convs = nn.ModuleList()
        for filter_size in self.filter_sizes:
            conv = nn.Sequential(
                nn.Conv1d(self.embed_size, self.num_filters, filter_size, padding=1),
                nn.BatchNorm1d(self.num_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )
            self.convs.append(conv)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_size,
            num_heads=8,
            dropout=self.dropout_prob
        )
        
        # Feature combination
        total_filters = len(self.filter_sizes) * self.num_filters
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_filters + self.embed_size, total_filters),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(total_filters // 2, self.num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        if not self.use_pretrained:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs (Dict[str, torch.Tensor]): Input tensors
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len) - optional
                - labels: (batch_size,) - optional
                
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        labels = inputs.get('labels', None)
        
        # Embedding
        x = self.embedding(input_ids)  # (batch_size, seq_len, embed_size)
        
        # Self-attention
        if attention_mask is not None:
            # Apply attention mask
            x_masked = x * attention_mask.unsqueeze(-1).float()
            attn_output, _ = self.attention(
                x_masked.transpose(0, 1),
                x_masked.transpose(0, 1),
                x_masked.transpose(0, 1)
            )
            attn_output = attn_output.transpose(0, 1)
        else:
            attn_output, _ = self.attention(
                x.transpose(0, 1),
                x.transpose(0, 1),
                x.transpose(0, 1)
            )
            attn_output = attn_output.transpose(0, 1)
        
        # Global average pooling for attention output
        attn_pooled = torch.mean(attn_output, dim=1)  # (batch_size, embed_size)
        
        # CNN features
        x_cnn = x.transpose(1, 2)  # (batch_size, embed_size, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_cnn)  # (batch_size, num_filters, reduced_seq_len)
            # Global max pooling
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate CNN features
        cnn_features = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters)
        
        # Combine CNN and attention features
        combined_features = torch.cat([cnn_features, attn_pooled], dim=1)
        features = self.feature_combiner(combined_features)
        
        # Classification
        logits = self.classifier(features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'features': features,
            'attention_output': attn_output
        }
