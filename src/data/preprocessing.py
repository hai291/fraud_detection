"""
Data preprocessing utilities
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
import argparse
import yaml


class DataPreprocessor:
    """
    Data preprocessing utilities
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove special characters if needed
        # Add more cleaning steps as needed
        
        return text.strip()
    
    def process_dataset(
        self,
        input_path: str,
        output_dir: str,
        text_column: str = "text",
        label_column: str = "label",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> None:
        """
        Process dataset and split into train/val/test
        
        Args:
            input_path (str): Path to input dataset
            output_dir (str): Directory to save processed data
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test split ratio
            val_size (float): Validation split ratio
            random_state (int): Random seed
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Clean text
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df[text_column].str.len() > 0]
        
        # Split data
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_size + val_size, 
            random_state=random_state,
            stratify=df[label_column] if label_column in df.columns else None
        )
        
        if val_size > 0:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_size / (test_size + val_size),
                random_state=random_state,
                stratify=temp_df[label_column] if label_column in temp_df.columns else None
            )
        else:
            test_df = temp_df
            val_df = pd.DataFrame()
        
        # Save splits
        train_df.to_json(
            os.path.join(output_dir, 'train.json'),
            orient='records',
            indent=2,
            force_ascii=False
        )
        
        if not val_df.empty:
            val_df.to_json(
                os.path.join(output_dir, 'val.json'),
                orient='records',
                indent=2,
                force_ascii=False
            )
        
        test_df.to_json(
            os.path.join(output_dir, 'test.json'),
            orient='records',
            indent=2,
            force_ascii=False
        )
        
        print(f"Dataset split completed:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Saved to: {output_dir}")


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.config)
    
    # Process all files in input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith(('.csv', '.json')):
            input_path = os.path.join(args.input_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                preprocessor.process_dataset(
                    input_path=input_path,
                    output_dir=args.output_dir
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
