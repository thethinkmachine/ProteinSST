#!/usr/bin/env python3
"""
Tier 1 Training Script: Baseline Model

Architecture: PLM Embeddings → FC → MTL Head

This is the simplest tier for establishing baseline performance.

Usage:
    python notebooks/training/tier1_baseline.py --plm ankh_base --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from src.config import (
    Tier1Config, 
    LEAKAGE_TRAIN_IDS, 
    SST8_WEIGHTS, 
    SST3_WEIGHTS,
    get_embedding_dim,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier1Baseline
from src.losses import MultiTaskLoss
from src.metrics import compute_accuracy, compute_per_class_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tier 1 Baseline Model')
    
    # Model
    parser.add_argument('--plm', type=str, default='ankh_base',
                       choices=['ankh_base', 'ankh_large', 'esm2_35m', 'esm2_650m', 'protbert'])
    parser.add_argument('--head_strategy', type=str, default='q3discarding',
                       choices=['q3discarding', 'q3guided'])
    parser.add_argument('--fc_hidden', type=int, default=512)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    
    # Data
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--embeddings', type=str, default=None,
                       help='Path to HDF5 embeddings (default: data/embeddings/{plm}.h5)')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_q8_correct = 0
    total_q3_correct = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        features = batch['features'].to(device)
        sst8_labels = batch['sst8'].to(device)
        sst3_labels = batch['sst3'].to(device)
        lengths = batch['lengths']
        
        # Forward
        optimizer.zero_grad()
        q8_logits, q3_logits = model(features, return_q3=True)
        
        # Loss
        loss = criterion(q8_logits, sst8_labels, q3_logits, sst3_labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * features.size(0)
        
        # Mask for padding
        mask = sst8_labels != -100
        
        q8_pred = q8_logits.argmax(dim=-1)
        q8_correct = ((q8_pred == sst8_labels) & mask).sum().item()
        total_q8_correct += q8_correct
        
        if q3_logits is not None:
            q3_pred = q3_logits.argmax(dim=-1)
            q3_correct = ((q3_pred == sst3_labels) & mask).sum().item()
            total_q3_correct += q3_correct
        
        total_tokens += mask.sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'q8_acc': f'{total_q8_correct / max(total_tokens, 1) * 100:.1f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'q8_acc': total_q8_correct / max(total_tokens, 1) * 100,
        'q3_acc': total_q3_correct / max(total_tokens, 1) * 100,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_q8_correct = 0
    total_q3_correct = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        features = batch['features'].to(device)
        sst8_labels = batch['sst8'].to(device)
        sst3_labels = batch['sst3'].to(device)
        
        # Forward
        q8_logits, q3_logits = model(features, return_q3=True)
        
        # Loss
        loss = criterion(q8_logits, sst8_labels, q3_logits, sst3_labels)
        total_loss += loss.item() * features.size(0)
        
        # Mask for padding
        mask = sst8_labels != -100
        
        q8_pred = q8_logits.argmax(dim=-1)
        total_q8_correct += ((q8_pred == sst8_labels) & mask).sum().item()
        
        if q3_logits is not None:
            q3_pred = q3_logits.argmax(dim=-1)
            total_q3_correct += ((q3_pred == sst3_labels) & mask).sum().item()
        
        total_tokens += mask.sum().item()
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'q8_acc': total_q8_correct / max(total_tokens, 1) * 100,
        'q3_acc': total_q3_correct / max(total_tokens, 1) * 100,
    }


def main():
    args = parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    set_seed(args.seed)
    
    print("=" * 60)
    print("Tier 1: Baseline Training")
    print("=" * 60)
    print(f"PLM: {args.plm}")
    print(f"Head Strategy: {args.head_strategy}")
    print(f"Device: {device}")
    print()
    
    # Embeddings path
    embeddings_path = args.embeddings or f'data/embeddings/{args.plm}.h5'
    
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        print(f"   Run: python scripts/extract_embeddings.py --plm {args.plm}")
        sys.exit(1)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = HDF5EmbeddingDataset(
        csv_path=args.train_csv,
        h5_path=embeddings_path,
        dataset_name='train',
        max_length=args.max_length,
        exclude_ids=LEAKAGE_TRAIN_IDS,
    )
    
    # Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Model
    embedding_dim = get_embedding_dim(args.plm)
    print(f"\nCreating model (embedding_dim={embedding_dim})...")
    
    model = Tier1Baseline(
        embedding_dim=embedding_dim,
        fc_hidden=args.fc_hidden,
        head_strategy=args.head_strategy,
    ).to(device)
    
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Loss
    criterion = MultiTaskLoss(
        q8_weight=1.0,
        q3_weight=0.5,
        q8_class_weights=SST8_WEIGHTS.to(device),
        q3_class_weights=SST3_WEIGHTS.to(device),
        focal_gamma=2.0,
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Q8: {train_metrics['q8_acc']:.2f}%, Q3: {train_metrics['q3_acc']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Q8: {val_metrics['q8_acc']:.2f}%, Q3: {val_metrics['q3_acc']:.2f}%")
        
        # Early stopping
        if val_metrics['q8_acc'] > best_val_acc:
            best_val_acc = val_metrics['q8_acc']
            patience_counter = 0
            
            # Save best model
            checkpoint_path = Path('checkpoints/tier1_baseline_best.pt')
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': {
                    'plm': args.plm,
                    'embedding_dim': embedding_dim,
                    'fc_hidden': args.fc_hidden,
                    'head_strategy': args.head_strategy,
                },
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Q8: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print(f"\n✅ Training complete! Best Val Q8: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
