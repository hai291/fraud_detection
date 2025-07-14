"""
Transformer-based models
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional
from .base_model import BaseModel


class TransformerModel(BaseModel):
    """
    Transformer-based model for classification tasks
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.num_labels = config.get('num_labels', 2)
        self.hidden_size = config.get('hidden_size', 768)
        self.dropout_prob = config.get('hidden_dropout_prob', 0.1)
        
        # Load pretrained transformer
        model_name = config.get('pretrained_model', 'bert-base-uncased')
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs (Dict[str, torch.Tensor]): Input tensors
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - labels: (batch_size,) - optional
                
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.get('labels', None)
        
        # Transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get pooled output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': outputs.last_hidden_state
        }


class TransformerForGeneration(BaseModel):
    """
    Transformer-based model for text generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config.get('vocab_size', 50257)
        self.hidden_size = config.get('hidden_size', 768)
        self.max_length = config.get('max_length', 512)
        
        # Load pretrained transformer
        model_name = config.get('pretrained_model', 'gpt2')
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            inputs (Dict[str, torch.Tensor]): Input tensors
                - input_ids: (batch_size, seq_len)
                - attention_mask: (batch_size, seq_len)
                - labels: (batch_size, seq_len) - optional
                
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.get('labels', None)
        
        # Transformer forward pass
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Language modeling
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text
        
        Args:
            input_ids (torch.Tensor): Input token ids
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p sampling
            do_sample (bool): Whether to use sampling
            
        Returns:
            torch.Tensor: Generated token ids
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward({
                    'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids)
                })
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                if top_p > 0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample next token
                if do_sample:
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits, dim=-1),
                        num_samples=1
                    )
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if next_token.item() == self.transformer.config.eos_token_id:
                    break
        
        return input_ids
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
