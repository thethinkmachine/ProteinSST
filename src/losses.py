"""
Loss functions for protein secondary structure prediction.
Includes Focal Loss, Weighted CE, and Multi-task loss combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import SST8_WEIGHTS, SST3_WEIGHTS


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (γ=0 is standard CE, γ=2 is recommended)
        reduction: 'mean', 'sum', or 'none'
        ignore_index: Index to ignore in loss computation (e.g., padding)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (batch, seq_len, num_classes) or (batch*seq_len, num_classes)
            targets: Labels of shape (batch, seq_len) or (batch*seq_len,)
        """
        # Flatten if needed
        if inputs.dim() == 3:
            batch_size, seq_len, num_classes = inputs.shape
            inputs = inputs.view(-1, num_classes)
            targets = targets.view(-1)
        
        # Create mask for valid positions
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# Weighted Cross Entropy
# =============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with class weights.
    
    Args:
        weight: Class weights tensor
        ignore_index: Index to ignore (padding)
        label_smoothing: Label smoothing factor (0.0 to 0.1 recommended)
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten if 3D
        if inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        
        return F.cross_entropy(
            inputs, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


# =============================================================================
# Label Smoothing Cross Entropy
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for regularization.
    
    Smoothed labels: (1 - ε) * one_hot + ε / num_classes
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 3:
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)
        
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        num_classes = inputs.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.full_like(inputs, self.smoothing / num_classes)
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)
        
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)
        
        return loss.mean()


# =============================================================================
# Multi-Task Loss
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for joint Q8 and Q3 prediction.
    
    Total Loss = λ_q8 * L_q8 + λ_q3 * L_q3
    
    Args:
        q8_loss_fn: Loss function for Q8 prediction
        q3_loss_fn: Loss function for Q3 prediction
        q8_weight: Weight for Q8 loss (λ_q8)
        q3_weight: Weight for Q3 loss (λ_q3)
    """
    
    def __init__(
        self,
        q8_loss_fn: nn.Module = None,
        q3_loss_fn: nn.Module = None,
        q8_weight: float = 1.0,
        q3_weight: float = 0.5,
    ):
        super().__init__()
        
        # Default to Focal Loss with class weights
        self.q8_loss_fn = q8_loss_fn or FocalLoss(alpha=SST8_WEIGHTS, gamma=2.0)
        self.q3_loss_fn = q3_loss_fn or FocalLoss(alpha=SST3_WEIGHTS, gamma=2.0)
        
        self.q8_weight = q8_weight
        self.q3_weight = q3_weight
    
    def forward(
        self,
        q8_logits: torch.Tensor,
        q8_targets: torch.Tensor,
        q3_logits: torch.Tensor,
        q3_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (total_loss, q8_loss, q3_loss)
        """
        q8_loss = self.q8_loss_fn(q8_logits, q8_targets)
        q3_loss = self.q3_loss_fn(q3_logits, q3_targets)
        
        total_loss = self.q8_weight * q8_loss + self.q3_weight * q3_loss
        
        return total_loss, q8_loss, q3_loss


# =============================================================================
# Dynamic Weight Averaging (DWA) Multi-Task Loss
# =============================================================================

class DynamicWeightMultiTaskLoss(nn.Module):
    """
    Multi-task loss with dynamic weight averaging based on loss trends.
    
    Automatically adjusts task weights based on relative learning progress.
    Reference: "End-to-End Multi-Task Learning with Attention" (Liu et al., 2019)
    """
    
    def __init__(
        self,
        q8_loss_fn: nn.Module = None,
        q3_loss_fn: nn.Module = None,
        temperature: float = 2.0,
    ):
        super().__init__()
        
        self.q8_loss_fn = q8_loss_fn or FocalLoss(alpha=SST8_WEIGHTS, gamma=2.0)
        self.q3_loss_fn = q3_loss_fn or FocalLoss(alpha=SST3_WEIGHTS, gamma=2.0)
        
        self.temperature = temperature
        
        # Track previous losses for dynamic weighting
        self.prev_q8_loss = None
        self.prev_q3_loss = None
    
    def forward(
        self,
        q8_logits: torch.Tensor,
        q8_targets: torch.Tensor,
        q3_logits: torch.Tensor,
        q3_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        q8_loss = self.q8_loss_fn(q8_logits, q8_targets)
        q3_loss = self.q3_loss_fn(q3_logits, q3_targets)
        
        # Compute dynamic weights
        if self.prev_q8_loss is not None and self.prev_q3_loss is not None:
            # Relative improvement rates
            r8 = q8_loss.item() / (self.prev_q8_loss + 1e-8)
            r3 = q3_loss.item() / (self.prev_q3_loss + 1e-8)
            
            # Softmax with temperature
            w8 = torch.exp(torch.tensor(r8 / self.temperature))
            w3 = torch.exp(torch.tensor(r3 / self.temperature))
            
            # Normalize
            w_sum = w8 + w3
            w8 = 2 * w8 / w_sum  # Scale to sum to 2 (like original weights)
            w3 = 2 * w3 / w_sum
        else:
            w8, w3 = 1.0, 0.5  # Initial weights
        
        # Update previous losses
        self.prev_q8_loss = q8_loss.item()
        self.prev_q3_loss = q3_loss.item()
        
        total_loss = w8 * q8_loss + w3 * q3_loss
        
        return total_loss, q8_loss, q3_loss


# =============================================================================
# Factory Functions
# =============================================================================

def get_loss_function(
    loss_type: str = 'focal',
    task: str = 'q8',
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: 'focal', 'weighted_ce', 'label_smoothing', or 'ce'
        task: 'q8' or 'q3' (determines class weights)
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
    
    Returns:
        Loss function module
    """
    weights = SST8_WEIGHTS if task == 'q8' else SST3_WEIGHTS
    
    if loss_type == 'focal':
        return FocalLoss(alpha=weights, gamma=gamma)
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_multitask_loss(
    loss_type: str = 'focal',
    q8_weight: float = 1.0,
    q3_weight: float = 0.5,
    dynamic_weights: bool = False,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function for multi-task loss.
    
    Args:
        loss_type: Base loss type ('focal', 'weighted_ce', etc.)
        q8_weight: Weight for Q8 loss
        q3_weight: Weight for Q3 loss
        dynamic_weights: Use dynamic weight averaging
        gamma: Focal loss gamma
    
    Returns:
        Multi-task loss module
    """
    q8_loss = get_loss_function(loss_type, 'q8', gamma)
    q3_loss = get_loss_function(loss_type, 'q3', gamma)
    
    if dynamic_weights:
        return DynamicWeightMultiTaskLoss(q8_loss, q3_loss)
    else:
        return MultiTaskLoss(q8_loss, q3_loss, q8_weight, q3_weight)
