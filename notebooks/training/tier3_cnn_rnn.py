#!/usr/bin/env python3
"""
Tier 3 Training Script: CNN + RNN Model

Architecture: PLM Embeddings → CNN → BiLSTM/BiGRU/BiRNN → MTL Head

Most expressive architecture combining local motif extraction with sequential modeling.

Usage:
    python notebooks/training/tier3_cnn_rnn.py --plm ankh_base --cnn_type multiscale --rnn_type lstm
    python notebooks/training/tier3_cnn_rnn.py --plm esm2_650m --cnn_type deep --rnn_type gru
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

from src.config import (
    Tier3Config,
    LEAKAGE_TRAIN_IDS,
    SST8_WEIGHTS,
    SST3_WEIGHTS,
    get_embedding_dim,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier3CNNRNN
from src.losses import MultiTaskLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tier 3 CNN+RNN Model')
    
    # Model
    parser.add_argument('--plm', type=str, default='ankh_base')
    
    # CNN
    parser.add_argument('--cnn_type', type=str, default='multiscale',
                       choices=['multiscale', 'deep'])
    parser.add_argument('--kernel_sizes', type=str, default='3,5,7',
                       help='Comma-separated kernel sizes for multiscale')
    parser.add_argument('--dilations', type=str, default='1,2,4,8',
                       help='Comma-separated dilations for deep CNN')
    parser.add_argument('--cnn_channels', type=int, default=64)
    
    # RNN
    parser.add_argument('--rnn_type', type=str, default='lstm',
                       choices=['lstm', 'gru', 'rnn'])
    parser.add_argument('--rnn_hidden', type=int, default=256)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_dropout', type=float, default=0.3)
    
    # Head
    parser.add_argument('--head_strategy', type=str, default='q3discarding')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    
    # Data
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Device
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
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
        
        optimizer.zero_grad()
        q8_logits, q3_logits = model(features, lengths=lengths, return_q3=True)
        loss = criterion(q8_logits, sst8_labels, q3_logits, sst3_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        mask = sst8_labels != -100
        
        q8_pred = q8_logits.argmax(dim=-1)
        total_q8_correct += ((q8_pred == sst8_labels) & mask).sum().item()
        
        if q3_logits is not None:
            q3_pred = q3_logits.argmax(dim=-1)
            total_q3_correct += ((q3_pred == sst3_labels) & mask).sum().item()
        
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
    model.eval()
    total_loss = 0
    total_q8_correct = 0
    total_q3_correct = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        features = batch['features'].to(device)
        sst8_labels = batch['sst8'].to(device)
        sst3_labels = batch['sst3'].to(device)
        lengths = batch['lengths']
        
        q8_logits, q3_logits = model(features, lengths=lengths, return_q3=True)
        loss = criterion(q8_logits, sst8_labels, q3_logits, sst3_labels)
        total_loss += loss.item() * features.size(0)
        
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
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    set_seed(args.seed)
    
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    dilations = [int(d) for d in args.dilations.split(',')]
    
    print("=" * 60)
    print("Tier 3: CNN + RNN Training")
    print("=" * 60)
    print(f"PLM: {args.plm}")
    print(f"CNN: {args.cnn_type} (kernels={kernel_sizes if args.cnn_type == 'multiscale' else dilations})")
    print(f"RNN: {args.rnn_type} (hidden={args.rnn_hidden}, layers={args.rnn_layers})")
    print(f"Device: {device}")
    print()
    
    embeddings_path = args.embeddings or f'data/embeddings/{args.plm}.h5'
    
    if not Path(embeddings_path).exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        print(f"   Run: python scripts/extract_embeddings.py --plm {args.plm}")
        sys.exit(1)
    
    print("Loading dataset...")
    full_dataset = HDF5EmbeddingDataset(
        csv_path=args.train_csv,
        h5_path=embeddings_path,
        dataset_name='train',
        max_length=args.max_length,
        exclude_ids=LEAKAGE_TRAIN_IDS,
    )
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    embedding_dim = get_embedding_dim(args.plm)
    print(f"\nCreating model (embedding_dim={embedding_dim})...")
    
    model = Tier3CNNRNN(
        embedding_dim=embedding_dim,
        cnn_type=args.cnn_type,
        kernel_sizes=kernel_sizes,
        cnn_out_channels=args.cnn_channels,
        cnn_dilations=dilations,
        rnn_type=args.rnn_type,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        head_strategy=args.head_strategy,
    ).to(device)
    
    print(f"  Parameters: {model.count_parameters():,}")
    
    criterion = MultiTaskLoss(
        q8_weight=1.0, q3_weight=0.5,
        q8_class_weights=SST8_WEIGHTS.to(device),
        q3_class_weights=SST3_WEIGHTS.to(device),
        focal_gamma=2.0,
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_acc = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Q8: {train_metrics['q8_acc']:.2f}%, Q3: {train_metrics['q3_acc']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Q8: {val_metrics['q8_acc']:.2f}%, Q3: {val_metrics['q3_acc']:.2f}%")
        
        if val_metrics['q8_acc'] > best_val_acc:
            best_val_acc = val_metrics['q8_acc']
            patience_counter = 0
            
            checkpoint_path = Path(f'checkpoints/tier3_{args.cnn_type}_{args.rnn_type}_best.pt')
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'config': {
                    'plm': args.plm,
                    'cnn_type': args.cnn_type,
                    'rnn_type': args.rnn_type,
                    'rnn_hidden': args.rnn_hidden,
                    'rnn_layers': args.rnn_layers,
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
