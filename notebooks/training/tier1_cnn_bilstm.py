# %% [markdown]
# # Tier 1: CNN + BiLSTM Training
# 
# This notebook implements training for the **Tier 1** architecture:
# - Multi-scale 1D CNN for local feature extraction
# - Stacked BiLSTM for sequential modeling
# - Dual output heads for Q8 and Q3 prediction
# 
# ## Expected Performance
# - Q3 Accuracy: ~82-85%
# - Q8 Accuracy: ~70-74%

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import sys
sys.path.insert(0, '../..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import os

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# %%
# Import our modules
from src.config import Tier1Config, LEAKAGE_TRAIN_IDS
from src.data import ProteinDataset, create_dataloaders, collate_fn
from src.models.tier1_cnn_bilstm import CNNBiLSTM
from src.losses import get_multitask_loss
from src.augmentation import SequenceAugmenter, describe_augmentation_levels
from src.metrics import evaluate_model, plot_confusion_matrix, EvaluationReport
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

# %% [markdown]
# ## 2. Configuration

# %%
# Experiment configuration
config = Tier1Config(
    # Data
    max_seq_length=512,
    batch_size=32,
    
    # Model
    input_dim=40,  # 20 (one-hot) + 20 (BLOSUM62)
    use_blosum=True,
    cnn_filters=64,
    cnn_kernels=[3, 5, 7],
    lstm_hidden=256,
    lstm_layers=2,
    lstm_dropout=0.3,
    fc_hidden=256,
    fc_dropout=0.2,
    
    # Training
    learning_rate=1e-4,
    weight_decay=0.01,
    max_epochs=50,
    patience=10,
    gradient_clip=1.0,
    
    # Loss
    focal_gamma=2.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Augmentation
    augmentation_level=2,  # Light masking
    
    # Checkpointing
    checkpoint_dir='../../checkpoints/tier1_cnn_bilstm',
)

print("Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Max sequence length: {config.max_seq_length}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Augmentation level: {config.augmentation_level}")

# %% [markdown]
# ## 3. Data Loading

# %%
# Preview augmentation levels
describe_augmentation_levels()

# %%
# Create augmenter
augmenter = SequenceAugmenter(level=config.augmentation_level, seed=SEED)
print(f"\nUsing augmentation: {augmenter}")

# %%
# Create data loaders
train_loader, val_loader = create_dataloaders(
    train_csv='../../data/train.csv',
    val_split=0.1,
    batch_size=config.batch_size,
    max_length=config.max_seq_length,
    use_blosum=config.use_blosum,
    use_positional=False,
    augmentation=augmenter,
    num_workers=4,
    seed=SEED,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %%
# Inspect a batch
sample_batch = next(iter(train_loader))
print(f"\nBatch contents:")
print(f"  Features shape: {sample_batch['features'].shape}")
print(f"  SST8 shape: {sample_batch['sst8'].shape}")
print(f"  SST3 shape: {sample_batch['sst3'].shape}")
print(f"  Lengths: {sample_batch['lengths'][:5]}")

# %% [markdown]
# ## 4. Model Initialization

# %%
# Create model
model = CNNBiLSTM(
    input_dim=config.input_dim,
    cnn_filters=config.cnn_filters,
    cnn_kernels=config.cnn_kernels,
    lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers,
    lstm_dropout=config.lstm_dropout,
    fc_hidden=config.fc_hidden,
    fc_dropout=config.fc_dropout,
)

print(model)
print(f"\nTotal parameters: {model.count_parameters():,}")

# %%
# Test forward pass
model = model.to(DEVICE)
test_input = sample_batch['features'].to(DEVICE)
q8_out, q3_out = model(test_input)
print(f"Q8 output shape: {q8_out.shape}")
print(f"Q3 output shape: {q3_out.shape}")

# %% [markdown]
# ## 5. Loss Function Setup

# %%
# Create multi-task loss with focal loss and class weights
loss_fn = get_multitask_loss(
    loss_type='focal',
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    dynamic_weights=False,
    gamma=config.focal_gamma,
)

print(f"Loss function: {loss_fn}")

# %% [markdown]
# ## 6. Training

# %%
# Create optimizer and scheduler
optimizer = create_optimizer(
    model,
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    optimizer_type='adamw',
)

scheduler = create_scheduler(
    optimizer,
    scheduler_type='cosine',
    num_epochs=config.max_epochs,
)

# %%
# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    checkpoint_dir=config.checkpoint_dir,
    gradient_clip=config.gradient_clip,
    log_every=100,
    use_amp=torch.cuda.is_available(),  # Use FP16 on GPU
)

# %%
# Train!
history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

# %% [markdown]
# ## 7. Training Visualization

# %%
# Plot training history
fig = plot_training_history(history, save_path=f'{config.checkpoint_dir}/training_history.png')
fig.show()

# %% [markdown]
# ## 8. Evaluation

# %%
# Load best model
checkpoint = torch.load(f'{config.checkpoint_dir}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")

# %%
# Full evaluation
report = evaluate_model(
    model=model,
    dataloader=val_loader,
    device=DEVICE,
    compute_sov=True,
)

report.print_report()

# %%
# Plot confusion matrices
from src.config import SST8_CLASSES, SST3_CLASSES

fig_q8 = plot_confusion_matrix(
    report.q8_confusion_matrix,
    SST8_CLASSES,
    title='Q8 Confusion Matrix (Tier 1)',
    save_path=f'{config.checkpoint_dir}/q8_confusion_matrix.png',
)

fig_q3 = plot_confusion_matrix(
    report.q3_confusion_matrix,
    SST3_CLASSES,
    title='Q3 Confusion Matrix (Tier 1)',
    save_path=f'{config.checkpoint_dir}/q3_confusion_matrix.png',
)

# %% [markdown]
# ## 9. Summary

# %%
print("=" * 60)
print("TIER 1 TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest Results:")
print(f"  Q8 Accuracy: {report.q8_accuracy:.4f} ({report.q8_accuracy*100:.2f}%)")
print(f"  Q3 Accuracy: {report.q3_accuracy:.4f} ({report.q3_accuracy*100:.2f}%)")
print(f"  Q8 Macro F1: {report.q8_macro_f1:.4f}")
print(f"  Q3 Macro F1: {report.q3_macro_f1:.4f}")
if report.q8_sov:
    print(f"  Q8 SOV: {report.q8_sov:.2f}")
    print(f"  Q3 SOV: {report.q3_sov:.2f}")

print(f"\nCheckpoints saved to: {config.checkpoint_dir}")
