"""
Training infrastructure for ProteinSST.
Includes Trainer class with training loop, validation, and checkpointing.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .metrics import compute_q8_accuracy, compute_q3_accuracy, EvaluationReport, evaluate_model
from .losses import MultiTaskLoss


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Unified training class for all model tiers.
    
    Args:
        model: PyTorch model with forward(x) -> (q8_logits, q3_logits)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        loss_fn: Multi-task loss function
        optimizer: Optimizer (AdamW recommended)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory for saving checkpoints
        gradient_clip: Maximum gradient norm
        log_every: Log training metrics every N batches
        use_amp: Use automatic mixed precision (FP16)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        gradient_clip: float = 1.0,
        log_every: int = 100,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.use_amp = use_amp
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler() if use_amp else None
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'q8_accuracy': [],
            'q3_accuracy': [],
            'learning_rate': [],
        }
        self.best_val_loss = float('inf')
        self.best_q8_accuracy = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_q8_loss = 0.0
        total_q3_loss = 0.0
        total_q8_correct = 0
        total_q3_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            q8_targets = batch['sst8'].to(self.device)
            q3_targets = batch['sst3'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    q8_logits, q3_logits = self.model(features)
                    loss, q8_loss, q3_loss = self.loss_fn(
                        q8_logits, q8_targets, q3_logits, q3_targets
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                q8_logits, q3_logits = self.model(features)
                loss, q8_loss, q3_loss = self.loss_fn(
                    q8_logits, q8_targets, q3_logits, q3_targets
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # Update scheduler (if step-based)
            if self.scheduler is not None and hasattr(self.scheduler, 'step_batch'):
                self.scheduler.step()
            
            # Tracking
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_q8_loss += q8_loss.item() * batch_size
            total_q3_loss += q3_loss.item() * batch_size
            total_samples += batch_size
            
            # Compute batch accuracy
            q8_preds = q8_logits.argmax(dim=-1)
            q3_preds = q3_logits.argmax(dim=-1)
            
            mask = q8_targets != -100
            total_q8_correct += ((q8_preds == q8_targets) & mask).sum().item()
            total_q3_correct += ((q3_preds == q3_targets) & mask).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'q8_loss': q8_loss.item(),
                'q3_loss': q3_loss.item(),
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / total_samples
        avg_q8_loss = total_q8_loss / total_samples
        avg_q3_loss = total_q3_loss / total_samples
        
        return {
            'loss': avg_loss,
            'q8_loss': avg_q8_loss,
            'q3_loss': avg_q3_loss,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_q8_loss = 0.0
        total_q3_loss = 0.0
        all_q8_preds = []
        all_q8_targets = []
        all_q3_preds = []
        all_q3_targets = []
        total_samples = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            q8_targets = batch['sst8'].to(self.device)
            q3_targets = batch['sst3'].to(self.device)
            
            q8_logits, q3_logits = self.model(features)
            loss, q8_loss, q3_loss = self.loss_fn(
                q8_logits, q8_targets, q3_logits, q3_targets
            )
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_q8_loss += q8_loss.item() * batch_size
            total_q3_loss += q3_loss.item() * batch_size
            total_samples += batch_size
            
            all_q8_preds.append(q8_logits)
            all_q8_targets.append(q8_targets)
            all_q3_preds.append(q3_logits)
            all_q3_targets.append(q3_targets)
        
        # Concatenate predictions
        all_q8_preds = torch.cat(all_q8_preds, dim=0)
        all_q8_targets = torch.cat(all_q8_targets, dim=0)
        all_q3_preds = torch.cat(all_q3_preds, dim=0)
        all_q3_targets = torch.cat(all_q3_targets, dim=0)
        
        # Compute accuracy
        q8_accuracy = compute_q8_accuracy(all_q8_preds, all_q8_targets)
        q3_accuracy = compute_q3_accuracy(all_q3_preds, all_q3_targets)
        
        return {
            'loss': total_loss / total_samples,
            'q8_loss': total_q8_loss / total_samples,
            'q3_loss': total_q3_loss / total_samples,
            'q8_accuracy': q8_accuracy,
            'q3_accuracy': q3_accuracy,
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_q8_accuracy': self.best_q8_accuracy,
            'history': self.history,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_q8_accuracy = checkpoint['best_q8_accuracy']
        self.history = checkpoint['history']
    
    def train(
        self,
        num_epochs: int,
        patience: int = 10,
        save_every: int = 5,
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            save_every: Save checkpoint every N epochs
        
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=patience, mode='min')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler (if epoch-based)
            if self.scheduler is not None:
                if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step()
                elif isinstance(self.scheduler, OneCycleLR):
                    pass  # OneCycleLR steps per batch
                else:
                    self.scheduler.step(val_metrics['loss'])
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['q8_accuracy'].append(val_metrics['q8_accuracy'])
            self.history['q3_accuracy'].append(val_metrics['q3_accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_q8_accuracy = val_metrics['q8_accuracy']
            
            # Save checkpoints
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', is_best=is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Q8 Acc: {val_metrics['q8_accuracy']:.4f} | "
                f"Q3 Acc: {val_metrics['q3_accuracy']:.4f} | "
                f"LR: {current_lr:.6f}" +
                (" *" if is_best else "")
            )
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Q8 Accuracy: {self.best_q8_accuracy:.4f}")
        
        return self.history


# =============================================================================
# Utility Functions
# =============================================================================

def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = 'adamw',
) -> torch.optim.Optimizer:
    """Create optimizer with proper weight decay handling."""
    
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    if optimizer_type == 'adamw':
        return AdamW(param_groups, lr=lr)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 50,
    warmup_steps: int = 500,
    steps_per_epoch: int = 100,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    
    if scheduler_type == 'cosine':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs // 3,
            T_mult=2,
            eta_min=1e-6,
        )
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['q8_accuracy'], label='Q8')
    axes[0, 1].plot(history['q3_accuracy'], label='Q3')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Q8 & Q3 Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final metrics bar
    final_metrics = {
        'Q8 Accuracy': history['q8_accuracy'][-1],
        'Q3 Accuracy': history['q3_accuracy'][-1],
    }
    axes[1, 1].bar(final_metrics.keys(), final_metrics.values(), color=['steelblue', 'coral'])
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Final Validation Metrics')
    axes[1, 1].set_ylim(0, 1)
    for i, (k, v) in enumerate(final_metrics.items()):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
