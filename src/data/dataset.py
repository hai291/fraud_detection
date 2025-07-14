"""
Custom Dataset classes for PyTorch
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    """
    Custom Dataset class for text classification tasks
    
    Args:
        data_path (str): Path to the dataset file
        tokenizer_name (str): Name of the tokenizer to use
        max_length (int): Maximum sequence length
        text_column (str): Name of the text column
        label_column (str): Name of the label column
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        text_column: str = "text",
        label_column: str = "label"
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Get text and label
        text = item[self.text_column]
        label = item[self.label_column]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextGenerationDataset(Dataset):
    """
    Dataset class for text generation tasks
    
    Args:
        data_path (str): Path to the dataset file
        tokenizer_name (str): Name of the tokenizer to use
        max_length (int): Maximum sequence length
        input_column (str): Name of the input text column
        target_column (str): Name of the target text column
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        input_column: str = "input",
        target_column: str = "target"
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.input_column = input_column
        self.target_column = target_column
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Get input and target
        input_text = item[self.input_column]
        target_text = item[self.target_column]
        
        # Combine input and target
        full_text = f"{input_text} {self.tokenizer.eos_token} {target_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }
