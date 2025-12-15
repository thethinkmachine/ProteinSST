# %% [markdown]
# # 🧬 Tier 2: CNN Model Training
#
# This notebook implements the **Tier 2 CNN** architecture for protein secondary structure prediction.
#
# ## Architecture Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                           TIER 2: CNN                                   │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │   PLM Embeddings (L, D_plm)                                             │
# │          │                                                              │
# │          ▼                                                              │
# │   ┌─────────────────────────────────────────────────┐                   │
# │   │            CNN BLOCK (choose one)               │                   │
# │   │                                                 │                   │
# │   │   MultiscaleCNN                 DeepCNN         │                   │
# │   │   ┌───┬───┬───┬───┐            ┌───────────┐    │                   │
# │   │   │k=3│k=5│k=7│k=11            │ Conv d=1  │    │                   │
# │   │   └─┬─┴─┬─┴─┬─┴─┬─┘            ├───────────┤    │                   │
# │   │     └───┼───┼───┘              │ Conv d=2  │    │                   │
# │   │         │concat                ├───────────┤    │                   │
# │   │         ▼                      │ Conv d=4  │    │                   │
# │   │   (L, 4*64=256)                ├───────────┤    │                   │
# │   │                                │ Conv d=8  │    │                   │
# │   │                                └─────┬─────┘    │                   │
# │   │                                      ▼          │                   │
# │   │                                 (L, 256)        │                   │
# │   └─────────────────────────────────────────────────┘                   │
# │                        │                                                │
# │                        ▼                                                │
# │   ┌─────────────────────────────────────────────────┐                   │
# │   │  MTL Head (q3discarding OR q3guided)            │                   │
# │   └─────────────────────────────────────────────────┘                   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
# ```
#
# ## CNN Block Types
#
# | Type | Description | Params | Best For |
# |------|-------------|--------|----------|
# | **MultiscaleCNN** | Parallel branches, different kernel sizes | ~840K | Local patterns |
# | **DeepCNN** | Stacked layers, increasing dilation | ~275K | Long-range context |

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader, random_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# %%
from src.config import (
    Tier2Config, LEAKAGE_TRAIN_IDS,
    SST8_WEIGHTS, SST3_WEIGHTS,
    get_embedding_dim, PLM_EMBEDDING_DIMS,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier2CNN
from src.losses import get_multitask_loss
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

print("✓ Library modules imported")

# %% [markdown]
# ## 2. Configuration

# %%
# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Choose CNN type: 'multiscale' or 'deep'
CNN_TYPE = 'multiscale'

config = Tier2Config(
    # PLM Selection
    plm_name='ankh_base',
    embeddings_path='../../data/embeddings/ankh_base.h5',
    
    # CNN Architecture
    cnn_type=CNN_TYPE,
    
    # MultiscaleCNN params
    kernel_sizes=[3, 5, 7, 11],
    cnn_out_channels=64,
    
    # DeepCNN params
    cnn_num_layers=4,
    cnn_dilations=[1, 2, 4, 8],
    cnn_residual=True,
    
    # Common
    cnn_activation='relu',
    cnn_dropout=0.0,
    
    # MTL Head
    head_strategy='q3discarding',
    head_hidden=256,
    head_dropout=0.1,
    
    # Training
    max_seq_length=512,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_epochs=50,
    patience=10,
    gradient_clip=1.0,
    
    # Loss
    focal_gamma=2.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir=f'../../checkpoints/tier2_{CNN_TYPE}',
    
    # Tracking
    use_tracking=False,
    experiment_name=f'tier2_{CNN_TYPE}',
)

# %%
print("\n" + "═" * 60)
print(f"TIER 2 CNN CONFIGURATION ({CNN_TYPE.upper()})")
print("═" * 60)
print(f"\n📦 PLM: {config.plm_name}")
print(f"   Embedding Dim: {get_embedding_dim(config.plm_name)}")
print(f"\n🏗️  CNN Architecture:")
print(f"   Type: {config.cnn_type}")
if config.cnn_type == 'multiscale':
    print(f"   Kernel Sizes: {config.kernel_sizes}")
    print(f"   Channels per Branch: {config.cnn_out_channels}")
    print(f"   Total Output: {config.cnn_out_channels * len(config.kernel_sizes)}")
else:
    print(f"   Layers: {config.cnn_num_layers}")
    print(f"   Dilations: {config.cnn_dilations}")
    print(f"   Residual: {config.cnn_residual}")
print(f"\n🎯 Head Strategy: {config.head_strategy}")
print("═" * 60)

# %% [markdown]
# ## 3. Data Loading

# %%
embeddings_path = Path(config.embeddings_path)
if not embeddings_path.exists():
    print(f"❌ Run: python scripts/extract_embeddings.py --plm {config.plm_name}")
else:
    print(f"✓ Embeddings: {embeddings_path}")

# %%
full_dataset = HDF5EmbeddingDataset(
    csv_path='../../data/train.csv',
    h5_path=config.embeddings_path,
    dataset_name='train',
    max_length=config.max_seq_length,
    exclude_ids=LEAKAGE_TRAIN_IDS,
)

val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    collate_fn=collate_fn, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False,
    collate_fn=collate_fn, num_workers=4, pin_memory=True
)

print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"   Batches: {len(train_loader)} train, {len(val_loader)} val")

# %% [markdown]
# ## 4. Model Initialization

# %%
embedding_dim = get_embedding_dim(config.plm_name)

model = Tier2CNN(
    embedding_dim=embedding_dim,
    cnn_type=config.cnn_type,
    kernel_sizes=config.kernel_sizes,
    cnn_out_channels=config.cnn_out_channels,
    cnn_num_layers=config.cnn_num_layers,
    cnn_dilations=config.cnn_dilations,
    cnn_activation=config.cnn_activation,
    cnn_dropout=config.cnn_dropout,
    cnn_residual=config.cnn_residual,
    head_strategy=config.head_strategy,
    head_hidden=config.head_hidden,
    head_dropout=config.head_dropout,
).to(DEVICE)

print("\n🏗️  Model Summary:")
print("═" * 60)
print(f"CNN Type: {config.cnn_type}")
print(f"CNN Output Channels: {model.cnn.out_channels}")
print(f"\n📈 Total Parameters: {model.count_parameters():,}")
print("═" * 60)

# %%
# Compare with alt CNN type
alt_type = 'deep' if config.cnn_type == 'multiscale' else 'multiscale'
alt_model = Tier2CNN(
    embedding_dim=embedding_dim,
    cnn_type=alt_type,
    kernel_sizes=config.kernel_sizes,
    cnn_out_channels=config.cnn_out_channels,
    cnn_num_layers=config.cnn_num_layers,
    cnn_dilations=config.cnn_dilations,
)

print("\n📊 CNN Type Comparison:")
print("─" * 40)
print(f"  {config.cnn_type:12} │ {model.count_parameters():,} params ← selected")
print(f"  {alt_type:12} │ {alt_model.count_parameters():,} params")
print("─" * 40)
del alt_model

# %%
# Test forward pass
sample_batch = next(iter(train_loader))
model.eval()
with torch.no_grad():
    test_input = sample_batch['features'].to(DEVICE)
    q8_out, q3_out = model(test_input)

print(f"\n✓ Forward Pass: Input {test_input.shape} → Q8 {q8_out.shape}, Q3 {q3_out.shape}")

# %% [markdown]
# ## 5. Loss & Optimizer

# %%
loss_fn = get_multitask_loss(
    loss_type='focal',
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    q8_class_weights=SST8_WEIGHTS.to(DEVICE),
    q3_class_weights=SST3_WEIGHTS.to(DEVICE),
    gamma=config.focal_gamma,
)

optimizer = create_optimizer(model, lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = create_scheduler(optimizer, scheduler_type='cosine', num_epochs=config.max_epochs)

print("✓ Loss, optimizer, scheduler configured")

# %% [markdown]
# ## 6. Training

# %%
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
    use_amp=torch.cuda.is_available(),
    use_tracking=config.use_tracking,
    experiment_name=config.experiment_name,
    training_config=config.__dict__,
)

print("✓ Trainer initialized")

# %%
history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

# %% [markdown]
# ## 7. Visualization

# %%
fig = plot_training_history(
    history,
    save_path=str(Path(config.checkpoint_dir) / 'training_curves.png')
)

# %% [markdown]
# ## 8. Summary

# %%
print("\n" + "═" * 60)
print(f"🎉 TIER 2 {config.cnn_type.upper()} CNN TRAINING COMPLETE")
print("═" * 60)
print(f"\n📈 Best Results:")
print(f"   Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")
print(f"\n💾 Checkpoints: {config.checkpoint_dir}")
print("═" * 60)
