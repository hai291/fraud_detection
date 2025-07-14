"""
Base model class
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseModel(nn.Module, ABC):
    """
    Base model class for all neural network models
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get('name', 'base_model')
        
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model
        
        Args:
            inputs (Dict[str, torch.Tensor]): Input tensors
            
        Returns:
            Dict[str, torch.Tensor]: Model outputs
        """
        pass
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_parameters(self, module_names: Optional[list] = None) -> None:
        """
        Freeze model parameters
        
        Args:
            module_names (Optional[list]): List of module names to freeze
        """
        if module_names is None:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific modules
            for name, module in self.named_modules():
                if any(module_name in name for module_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = False
    
    def unfreeze_parameters(self, module_names: Optional[list] = None) -> None:
        """
        Unfreeze model parameters
        
        Args:
            module_names (Optional[list]): List of module names to unfreeze
        """
        if module_names is None:
            # Unfreeze all parameters
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific modules
            for name, module in self.named_modules():
                if any(module_name in name for module_name in module_names):
                    for param in module.parameters():
                        param.requires_grad = True
    
    def save_model(self, path: str) -> None:
        """
        Save model state dict
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load model state dict
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.model_name = checkpoint.get('model_name', self.model_name)
    
    def print_model_info(self) -> None:
        """Print model information"""
        print(f"Model: {self.model_name}")
        print(f"Total parameters: {self.get_num_parameters():,}")
        print(f"Trainable parameters: {self.get_trainable_parameters():,}")
        print(f"Model size: {self.get_num_parameters() * 4 / 1024 / 1024:.2f} MB")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary"""
        return {
            'model_name': self.model_name,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_trainable_parameters(),
            'model_size_mb': self.get_num_parameters() * 4 / 1024 / 1024,
            'config': self.config
        }
