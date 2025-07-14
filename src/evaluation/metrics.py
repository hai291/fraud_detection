"""
Metrics calculation utilities
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)


class MetricsCalculator:
    """
    Metrics calculator for model evaluation
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names
    
    def calculate_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        metrics = {}
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_prob is not None:
            y_prob = np.array(y_prob)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Class-wise metrics
        if average == 'weighted':
            precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            metrics['precision_per_class'] = precision_class.tolist()
            metrics['recall_per_class'] = recall_class.tolist()
            metrics['f1_per_class'] = f1_class.tolist()
            metrics['support_per_class'] = support_class.tolist()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # ROC AUC and Average Precision (for binary and multi-class)
        if y_prob is not None:
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average=average
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovo', average=average
                    )
                except ValueError:
                    # Handle cases where not all classes are present
                    pass
        
        # Error analysis
        metrics['total_samples'] = len(y_true)
        metrics['correct_predictions'] = np.sum(y_true == y_pred)
        metrics['incorrect_predictions'] = np.sum(y_true != y_pred)
        metrics['error_rate'] = 1 - metrics['accuracy']
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict[str, float]: Regression metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Additional regression metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return metrics
    
    def calculate_ranking_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_score: Union[List, np.ndarray],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate ranking metrics
        
        Args:
            y_true: True relevance scores
            y_score: Predicted relevance scores
            k: Cut-off for ranking metrics
            
        Returns:
            Dict[str, float]: Ranking metrics
        """
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # Sort by predicted scores
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Precision@k
        precision_at_k = np.sum(y_true_sorted[:k]) / k
        
        # Recall@k
        total_relevant = np.sum(y_true)
        recall_at_k = np.sum(y_true_sorted[:k]) / total_relevant if total_relevant > 0 else 0
        
        # NDCG@k
        dcg_at_k = np.sum(y_true_sorted[:k] / np.log2(np.arange(2, k + 2)))
        idcg_at_k = np.sum(np.sort(y_true)[::-1][:k] / np.log2(np.arange(2, k + 2)))
        ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0
        
        return {
            f'precision_at_{k}': precision_at_k,
            f'recall_at_{k}': recall_at_k,
            f'ndcg_at_{k}': ndcg_at_k
        }
    
    def get_classification_report(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray]
    ) -> str:
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Classification report
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            zero_division=0
        )
    
    def calculate_per_class_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict[str, Dict[str, float]]: Per-class metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        unique_classes = np.unique(y_true)
        per_class_metrics = {}
        
        for class_id in unique_classes:
            class_name = self.class_names[class_id] if self.class_names else f"Class_{class_id}"
            
            # Binary classification for this class vs rest
            y_true_binary = (y_true == class_id).astype(int)
            y_pred_binary = (y_pred == class_id).astype(int)
            
            per_class_metrics[class_name] = {
                'precision': precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average='binary', zero_division=0
                )[0],
                'recall': precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average='binary', zero_division=0
                )[1],
                'f1_score': precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average='binary', zero_division=0
                )[2],
                'support': np.sum(y_true == class_id)
            }
        
        return per_class_metrics
    
    def calculate_confidence_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Union[List, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate confidence-based metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dict[str, float]: Confidence metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Get confidence scores (max probability)
        confidences = np.max(y_prob, axis=1)
        
        # Accuracy vs confidence
        correct_predictions = (y_true == y_pred)
        
        # Binned accuracy by confidence
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_predictions[mask])
                bin_confidence = np.mean(confidences[mask])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        # Expected Calibration Error (ECE)
        ece = 0
        for i in range(len(bin_accuracies)):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                ece += (np.sum(mask) / len(confidences)) * abs(bin_accuracies[i] - bin_confidences[i])
        
        return {
            'average_confidence': np.mean(confidences),
            'confidence_of_correct': np.mean(confidences[correct_predictions]),
            'confidence_of_incorrect': np.mean(confidences[~correct_predictions]),
            'expected_calibration_error': ece,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences
        }
    
    def calculate_statistical_significance(
        self,
        y_true: Union[List, np.ndarray],
        y_pred1: Union[List, np.ndarray],
        y_pred2: Union[List, np.ndarray],
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Calculate statistical significance between two models
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            metric: Metric to compare
            
        Returns:
            Dict[str, float]: Statistical test results
        """
        from scipy import stats
        
        y_true = np.array(y_true)
        y_pred1 = np.array(y_pred1)
        y_pred2 = np.array(y_pred2)
        
        if metric == 'accuracy':
            scores1 = (y_true == y_pred1).astype(int)
            scores2 = (y_true == y_pred2).astype(int)
        else:
            raise ValueError(f"Metric {metric} not supported for significance testing")
        
        # McNemar's test for paired predictions
        # Contingency table
        n01 = np.sum((scores1 == 0) & (scores2 == 1))  # Model 1 wrong, Model 2 correct
        n10 = np.sum((scores1 == 1) & (scores2 == 0))  # Model 1 correct, Model 2 wrong
        
        # McNemar's test statistic
        if n01 + n10 > 0:
            mcnemar_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        # Paired t-test
        t_stat, t_p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'mcnemar_statistic': mcnemar_stat,
            'mcnemar_p_value': p_value,
            't_statistic': t_stat,
            't_p_value': t_p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01
        }
