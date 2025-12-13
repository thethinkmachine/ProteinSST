"""
Evaluation metrics for protein secondary structure prediction.
Includes Q3, Q8 accuracy, per-class metrics, SOV score, and visualization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix,
)

from .config import (
    SST8_CLASSES, SST3_CLASSES, SST8_NAMES, SST3_NAMES,
    IDX_TO_SST8, IDX_TO_SST3,
)


# =============================================================================
# Basic Accuracy Metrics
# =============================================================================

def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute per-residue accuracy.
    
    Args:
        predictions: Predicted class indices (batch, seq_len) or logits (batch, seq_len, classes)
        targets: Ground truth indices (batch, seq_len)
        ignore_index: Index to ignore (padding)
    
    Returns:
        Accuracy as float
    """
    # Handle logits
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)
    
    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    # Mask valid positions
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(targets) == 0:
        return 0.0
    
    correct = (predictions == targets).sum().item()
    total = len(targets)
    
    return correct / total


def compute_q8_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute Q8 (8-state) per-residue accuracy."""
    return compute_accuracy(predictions, targets, ignore_index)


def compute_q3_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """Compute Q3 (3-state) per-residue accuracy."""
    return compute_accuracy(predictions, targets, ignore_index)


# =============================================================================
# Per-Class Metrics
# =============================================================================

def compute_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: List[str],
    ignore_index: int = -100,
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 for each class.
    
    Returns:
        Dictionary with per-class metrics
    """
    # Handle logits
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)
    
    # Flatten and mask
    predictions = predictions.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]
    
    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    
    # Build result dictionary
    results = {}
    for i, name in enumerate(class_names):
        results[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
    
    # Add macro averages
    results['macro_avg'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'support': int(np.sum(support)),
    }
    
    return results


def compute_sst8_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics for SST8."""
    return compute_per_class_metrics(
        predictions, targets, 8, SST8_CLASSES, ignore_index
    )


def compute_sst3_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics for SST3."""
    return compute_per_class_metrics(
        predictions, targets, 3, SST3_CLASSES, ignore_index
    )


# =============================================================================
# Segment Overlap (SOV) Score
# =============================================================================

def compute_sov_score(
    predictions: List[str],
    targets: List[str],
    classes: List[str],
) -> float:
    """
    Compute Segment Overlap (SOV) score.
    
    SOV evaluates the quality of predicted secondary structure segments,
    accounting for boundary variations.
    
    Args:
        predictions: List of predicted structure strings
        targets: List of ground truth structure strings
        classes: List of structure classes to consider
    
    Returns:
        SOV score (0-100)
    """
    def get_segments(sequence: str) -> List[Tuple[str, int, int]]:
        """Extract contiguous segments from structure string."""
        if not sequence:
            return []
        
        segments = []
        start = 0
        current_class = sequence[0]
        
        for i in range(1, len(sequence)):
            if sequence[i] != current_class:
                segments.append((current_class, start, i - 1))
                start = i
                current_class = sequence[i]
        
        segments.append((current_class, start, len(sequence) - 1))
        return segments
    
    def segment_overlap(s1_start, s1_end, s2_start, s2_end):
        """Compute overlap between two segments."""
        overlap_start = max(s1_start, s2_start)
        overlap_end = min(s1_end, s2_end)
        
        if overlap_start <= overlap_end:
            return overlap_end - overlap_start + 1
        return 0
    
    total_sov = 0.0
    total_length = 0
    
    for pred_seq, true_seq in zip(predictions, targets):
        if len(pred_seq) != len(true_seq):
            continue
        
        true_segments = get_segments(true_seq)
        pred_segments = get_segments(pred_seq)
        
        for true_class, t_start, t_end in true_segments:
            if true_class not in classes:
                continue
            
            t_len = t_end - t_start + 1
            total_length += t_len
            
            # Find overlapping predicted segments of same class
            for pred_class, p_start, p_end in pred_segments:
                if pred_class != true_class:
                    continue
                
                overlap = segment_overlap(t_start, t_end, p_start, p_end)
                if overlap > 0:
                    # Compute SOV contribution
                    minov = overlap
                    maxov = max(t_end, p_end) - min(t_start, p_start) + 1
                    
                    delta = min(
                        maxov - minov,
                        minov,
                        t_len // 2,
                        (p_end - p_start + 1) // 2
                    )
                    
                    sov_contribution = t_len * (minov + delta) / maxov
                    total_sov += sov_contribution
    
    if total_length == 0:
        return 0.0
    
    return 100.0 * total_sov / total_length


# =============================================================================
# Confusion Matrix
# =============================================================================

def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)
    
    predictions = predictions.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]
    
    return sklearn_confusion_matrix(
        targets, predictions,
        labels=list(range(num_classes)),
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        normalize: Whether to normalize by row (true labels)
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        cm_plot = cm_normalized
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'
    
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Evaluation Report
# =============================================================================

@dataclass
class EvaluationReport:
    """Container for all evaluation metrics."""
    
    # Basic accuracy
    q8_accuracy: float
    q3_accuracy: float
    
    # Per-class metrics
    q8_per_class: Dict[str, Dict[str, float]]
    q3_per_class: Dict[str, Dict[str, float]]
    
    # Macro F1
    q8_macro_f1: float
    q3_macro_f1: float
    
    # SOV scores (optional)
    q8_sov: Optional[float] = None
    q3_sov: Optional[float] = None
    
    # Confusion matrices
    q8_confusion_matrix: Optional[np.ndarray] = None
    q3_confusion_matrix: Optional[np.ndarray] = None
    
    def __repr__(self):
        return (
            f"EvaluationReport(\n"
            f"  Q8 Accuracy: {self.q8_accuracy:.4f}\n"
            f"  Q3 Accuracy: {self.q3_accuracy:.4f}\n"
            f"  Q8 Macro F1: {self.q8_macro_f1:.4f}\n"
            f"  Q3 Macro F1: {self.q3_macro_f1:.4f}\n"
            f")"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'q8_accuracy': self.q8_accuracy,
            'q3_accuracy': self.q3_accuracy,
            'q8_macro_f1': self.q8_macro_f1,
            'q3_macro_f1': self.q3_macro_f1,
            'q8_sov': self.q8_sov,
            'q3_sov': self.q3_sov,
            'q8_per_class': self.q8_per_class,
            'q3_per_class': self.q3_per_class,
        }
    
    def print_report(self):
        """Print detailed evaluation report."""
        print("=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n{'='*30} Q8 (8-state) {'='*30}")
        print(f"Accuracy: {self.q8_accuracy:.4f} ({self.q8_accuracy*100:.2f}%)")
        print(f"Macro F1: {self.q8_macro_f1:.4f}")
        if self.q8_sov is not None:
            print(f"SOV Score: {self.q8_sov:.2f}")
        
        print("\nPer-class metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 60)
        for cls in SST8_CLASSES:
            if cls in self.q8_per_class:
                m = self.q8_per_class[cls]
                name = f"{cls} ({SST8_NAMES.get(cls, '')})"
                print(f"{name:<15} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['support']}")
        
        print(f"\n{'='*30} Q3 (3-state) {'='*30}")
        print(f"Accuracy: {self.q3_accuracy:.4f} ({self.q3_accuracy*100:.2f}%)")
        print(f"Macro F1: {self.q3_macro_f1:.4f}")
        if self.q3_sov is not None:
            print(f"SOV Score: {self.q3_sov:.2f}")
        
        print("\nPer-class metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 60)
        for cls in SST3_CLASSES:
            if cls in self.q3_per_class:
                m = self.q3_per_class[cls]
                name = f"{cls} ({SST3_NAMES.get(cls, '')})"
                print(f"{name:<15} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}       {m['support']}")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    compute_sov: bool = False,
) -> EvaluationReport:
    """
    Perform full evaluation on a model.
    
    Args:
        model: Trained model
        dataloader: Validation/test DataLoader
        device: Device to use
        compute_sov: Whether to compute SOV scores (slower)
    
    Returns:
        EvaluationReport with all metrics
    """
    model.eval()
    
    all_q8_preds = []
    all_q8_targets = []
    all_q3_preds = []
    all_q3_targets = []
    
    # For SOV computation (per-protein)
    all_q8_pred_strs = []
    all_q8_target_strs = []
    all_q3_pred_strs = []
    all_q3_target_strs = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            q8_targets = batch['sst8'].to(device)
            q3_targets = batch['sst3'].to(device)
            lengths = batch['lengths']
            
            q8_logits, q3_logits = model(features)
            
            q8_preds = q8_logits.argmax(dim=-1)  # (B, L)
            q3_preds = q3_logits.argmax(dim=-1)
            
            # --------------------------------------------------
            # MASK + FLATTEN FOR TOKEN-LEVEL METRICS
            # --------------------------------------------------
            mask = q8_targets != -100  # (B, L)
            
            all_q8_preds.append(q8_preds[mask])      # (N,)
            all_q8_targets.append(q8_targets[mask])
            all_q3_preds.append(q3_preds[mask])
            all_q3_targets.append(q3_targets[mask])
            
            # --------------------------------------------------
            # PER-PROTEIN STRINGS FOR SOV
            # --------------------------------------------------
            if compute_sov:
                for i, length in enumerate(lengths):
                    q8_pred_str = ''.join(
                        IDX_TO_SST8[idx.item()] for idx in q8_preds[i, :length]
                    )
                    q8_target_str = ''.join(
                        IDX_TO_SST8[idx.item()]
                        for idx in q8_targets[i, :length]
                        if idx.item() != -100
                    )
                    
                    q3_pred_str = ''.join(
                        IDX_TO_SST3[idx.item()] for idx in q3_preds[i, :length]
                    )
                    q3_target_str = ''.join(
                        IDX_TO_SST3[idx.item()]
                        for idx in q3_targets[i, :length]
                        if idx.item() != -100
                    )
                    
                    all_q8_pred_strs.append(q8_pred_str)
                    all_q8_target_strs.append(q8_target_str)
                    all_q3_pred_strs.append(q3_pred_str)
                    all_q3_target_strs.append(q3_target_str)
    
    # Concatenate flattened tensors (SAFE)
    all_q8_preds = torch.cat(all_q8_preds, dim=0)
    all_q8_targets = torch.cat(all_q8_targets, dim=0)
    all_q3_preds = torch.cat(all_q3_preds, dim=0)
    all_q3_targets = torch.cat(all_q3_targets, dim=0)
    
    # Metrics
    q8_accuracy = compute_q8_accuracy(all_q8_preds, all_q8_targets)
    q3_accuracy = compute_q3_accuracy(all_q3_preds, all_q3_targets)
    
    q8_per_class = compute_sst8_per_class_metrics(all_q8_preds, all_q8_targets)
    q3_per_class = compute_sst3_per_class_metrics(all_q3_preds, all_q3_targets)
    
    q8_confusion = compute_confusion_matrix(all_q8_preds, all_q8_targets, 8)
    q3_confusion = compute_confusion_matrix(all_q3_preds, all_q3_targets, 3)
    
    # SOV
    q8_sov = None
    q3_sov = None
    if compute_sov:
        q8_sov = compute_sov_score(all_q8_pred_strs, all_q8_target_strs, SST8_CLASSES)
        q3_sov = compute_sov_score(all_q3_pred_strs, all_q3_target_strs, SST3_CLASSES)
    
    return EvaluationReport(
        q8_accuracy=q8_accuracy,
        q3_accuracy=q3_accuracy,
        q8_per_class=q8_per_class,
        q3_per_class=q3_per_class,
        q8_macro_f1=q8_per_class['macro_avg']['f1'],
        q3_macro_f1=q3_per_class['macro_avg']['f1'],
        q8_sov=q8_sov,
        q3_sov=q3_sov,
        q8_confusion_matrix=q8_confusion,
        q3_confusion_matrix=q3_confusion,
    )
