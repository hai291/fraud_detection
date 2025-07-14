"""
Model evaluation utilities
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse

from ..data.dataset import CustomDataset
from ..models.transformer_model import TransformerModel
from ..models.cnn_model import CNNModel
from ..utils.config import load_config
from ..utils.logging import setup_logging
from .metrics import MetricsCalculator


class Evaluator:
    """
    Model evaluator class
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        self.model = model
        self.device = device
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on given dataloader
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            return_predictions (bool): Whether to return predictions
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Collect predictions
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                # Collect loss
                if outputs.get('loss') is not None:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities
        )
        
        # Add average loss
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        
        # Prepare results
        results = {'metrics': metrics}
        
        if return_predictions:
            results['predictions'] = all_predictions
        
        if return_probabilities:
            results['probabilities'] = all_probabilities
        
        results['labels'] = all_labels
        
        return results
    
    def evaluate_single_example(
        self,
        input_text: str,
        tokenizer,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Evaluate single text example
        
        Args:
            input_text (str): Input text
            tokenizer: Tokenizer for preprocessing
            max_length (int): Maximum sequence length
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Tokenize input
        encoding = tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
        
        # Get predictions
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
        return {
            'prediction': prediction.cpu().item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'confidence': torch.max(probabilities).cpu().item()
        }
    
    def evaluate_with_error_analysis(
        self,
        dataloader: DataLoader,
        class_names: Optional[List[str]] = None,
        top_k_errors: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate with error analysis
        
        Args:
            dataloader (DataLoader): Data loader for evaluation
            class_names (Optional[List[str]]): Class names
            top_k_errors (int): Number of top errors to analyze
            
        Returns:
            Dict[str, Any]: Evaluation results with error analysis
        """
        results = self.evaluate(
            dataloader,
            return_predictions=True,
            return_probabilities=True
        )
        
        predictions = results['predictions']
        probabilities = results['probabilities']
        labels = results['labels']
        
        # Find errors
        errors = []
        for i, (pred, prob, label) in enumerate(zip(predictions, probabilities, labels)):
            if pred != label:
                confidence = prob[pred]
                errors.append({
                    'index': i,
                    'predicted': pred,
                    'true': label,
                    'confidence': confidence,
                    'probabilities': prob
                })
        
        # Sort errors by confidence (most confident errors first)
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Analysis
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions),
            'top_k_errors': errors[:top_k_errors]
        }
        
        # Class-wise error analysis
        if class_names:
            class_errors = {}
            for class_name in class_names:
                class_idx = class_names.index(class_name)
                class_errors[class_name] = {
                    'total_samples': sum(1 for l in labels if l == class_idx),
                    'errors': sum(1 for pred, label in zip(predictions, labels) 
                                if label == class_idx and pred != label)
                }
            
            error_analysis['class_errors'] = class_errors
        
        results['error_analysis'] = error_analysis
        
        return results
    
    def benchmark_inference_speed(
        self,
        dataloader: DataLoader,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed
        
        Args:
            dataloader (DataLoader): Data loader for benchmarking
            num_runs (int): Number of runs for averaging
            
        Returns:
            Dict[str, float]: Timing results
        """
        import time
        
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(batch)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        total_samples = len(dataloader.dataset)
        
        return {
            'avg_time_per_run': np.mean(times),
            'std_time_per_run': np.std(times),
            'avg_time_per_sample': np.mean(times) / total_samples,
            'samples_per_second': total_samples / np.mean(times)
        }


def create_evaluator(
    model_path: str,
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Evaluator:
    """
    Create evaluator from model checkpoint
    
    Args:
        model_path (str): Path to model checkpoint
        model_config (Dict[str, Any]): Model configuration
        device (Optional[torch.device]): Device to use
        
    Returns:
        Evaluator: Configured evaluator
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model_type = model_config.get('architecture', 'transformer')
    
    if model_type == 'transformer':
        model = TransformerModel(model_config)
    elif model_type == 'cnn':
        model = CNNModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return Evaluator(model, device)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model configuration')
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to data configuration')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Load configurations
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = create_evaluator(args.model_path, model_config['model'], device)
    
    # Create test dataset
    test_dataset = CustomDataset(
        data_path=data_config['data']['test_path'],
        tokenizer_name=data_config['data']['tokenizer']['name'],
        max_length=data_config['data']['max_length'],
        text_column=data_config['data']['dataset']['text_column'],
        label_column=data_config['data']['dataset']['label_column']
    )
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Evaluating model on {len(test_dataset)} samples...")
    
    # Evaluate model
    results = evaluator.evaluate_with_error_analysis(
        test_dataloader,
        class_names=data_config['data']['dataset'].get('class_names')
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in results['metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
