"""
Loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = targets.unsqueeze(1)
        
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets, 1.0 - self.smoothing)
        
        loss = torch.sum(-smooth_targets * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    """
    
    def __init__(self, weights: Optional[Tensor] = None, reduction: str = 'mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weights, reduction=self.reduction)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings1: Tensor, embeddings2: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: 1 for similar pairs, 0 for dissimilar pairs
        """
        distances = F.pairwise_distance(embeddings1, embeddings2)
        
        positive_loss = labels * torch.pow(distances, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        loss = positive_loss + negative_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        loss = 1 - dice_score
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that can combine multiple loss functions
    """
    
    def __init__(self, losses: Dict[str, nn.Module], weights: Dict[str, float]):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
    
    def forward(self, inputs: Tensor, targets: Tensor, **kwargs) -> Tensor:
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            if name in kwargs:
                loss_value = loss_fn(inputs, targets, **kwargs[name])
            else:
                loss_value = loss_fn(inputs, targets)
            
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
            loss_dict[name] = loss_value.item()
        
        return total_loss, loss_dict


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, reduction: str = 'mean'):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, student_logits: Tensor, teacher_logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            targets: True labels
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss
        distillation_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, targets, reduction=self.reduction)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


def get_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create loss function based on configuration
    
    Args:
        loss_config: Configuration dictionary for loss function
        
    Returns:
        Loss function module
    """
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction=loss_config.get('reduction', 'mean'))
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            num_classes=loss_config['num_classes'],
            smoothing=loss_config.get('smoothing', 0.1),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'weighted_ce':
        weights = loss_config.get('weights')
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        return WeightedCrossEntropyLoss(
            weights=weights,
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            margin=loss_config.get('margin', 1.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'triplet':
        return TripletLoss(
            margin=loss_config.get('margin', 1.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=loss_config.get('smooth', 1.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    elif loss_type == 'combined':
        # Create individual loss functions
        losses = {}
        weights = {}
        
        for loss_name, loss_cfg in loss_config['losses'].items():
            losses[loss_name] = get_loss_function(loss_cfg)
            weights[loss_name] = loss_cfg.get('weight', 1.0)
        
        return CombinedLoss(losses, weights)
    
    elif loss_type == 'distillation':
        return DistillationLoss(
            temperature=loss_config.get('temperature', 4.0),
            alpha=loss_config.get('alpha', 0.7),
            reduction=loss_config.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def calculate_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for balanced training
    
    Args:
        labels (torch.Tensor): Training labels
        num_classes (int): Number of classes
        
    Returns:
        torch.Tensor: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=labels.numpy()
    )
    
    return torch.tensor(class_weights, dtype=torch.float32)
