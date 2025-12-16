# %% [markdown]
# # 🧬 Tier 1: Baseline PLM Model Training
#
# This notebook implements the **Tier 1 Baseline** architecture for protein secondary structure prediction.
#
# ## Architecture Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                         TIER 1: BASELINE                                │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │   PLM Embeddings (L, D_plm)                                             │
# │          │                                                              │
# │          ▼                                                              │
# │   ┌─────────────────┐                                                   │
# │   │  Linear(D_plm,  │                                                   │
# │   │     512)        │  Feature Projection                               │
# │   │   + GELU        │                                                   │
# │   │   + LayerNorm   │                                                   │
# │   │   + Dropout     │                                                   │
# │   └────────┬────────┘                                                   │
# │            │                                                            │
# │            ▼                                                            │
# │   ┌─────────────────┐                                                   │
# │   │  MTL Head       │  q3discarding OR q3guided                         │
# │   │  ┌───────────┐  │                                                   │
# │   │  │ Q8 Head   │──┼──▶ Q8 Logits (L, 8)                               │
# │   │  └───────────┘  │                                                   │
# │   │  ┌───────────┐  │                                                   │
# │   │  │ Q3 Head   │──┼──▶ Q3 Logits (L, 3)                               │
# │   │  └───────────┘  │                                                   │
# │   └─────────────────┘                                                   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
# ```
#
# ## Key Features
#
# | Feature | Description |
# |---------|-------------|
# | **PLM Support** | ESM-2 (8M, 35M, 650M), ProtBert |
# | **Parameters** | ~500-660K (lightweight) |
# | **Training Time** | ~5min/epoch on GPU |
# | **Use Case** | Baseline establishment, fast experimentation |

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Set seeds for reproducibility
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
# Import library modules
from src.config import (
    Tier1Config, LEAKAGE_TRAIN_IDS,
    get_embedding_dim, PLM_EMBEDDING_DIMS,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier1Baseline
from src.losses import get_multitask_loss
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

print("✓ Library modules imported")

# %% [markdown]
# ## 2. Configuration
#
# Choose your PLM and configure training hyperparameters.

# %%
# Available PLMs
print("Available PLMs:")
print("─" * 50)
for name, dim in PLM_EMBEDDING_DIMS.items():
    print(f"  {name:15} │ dim={dim:4}")
print("─" * 50)

# %%
# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION - Modify these values
# ═══════════════════════════════════════════════════════════════════

PLM_NAME = 'protbert'  # Options: 'esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert'

config = Tier1Config(
    # PLM Selection
    plm_name=PLM_NAME,
    embeddings_path=f'../../data/embeddings/{PLM_NAME}.h5',
    
    # Model Architecture
    fc_hidden=512,
    fc_dropout=0.1,
    
    # MTL Head Strategy: 'q3discarding' or 'q3guided'
    head_strategy='q3guided',
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
    
    # Loss - Options: 'focal', 'weighted_ce', 'label_smoothing', 'ce', 'crf'
    loss_type='focal',  # Use 'crf' for CRF Negative Log-Likelihood
    focal_gamma=1.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir=f'../../checkpoints/tier1_{PLM_NAME}',
    
    # Experiment Tracking (enabled by default)
    use_tracking=True,
    trackio_space_id='thethinkmachine/trackio',
    hub_model_id=f'thethinkmachine/ProteinSST-{PLM_NAME}',
    experiment_name=f'tier1_{PLM_NAME}',
)

# %%
# Print configuration summary
print("\n" + "═" * 60)
print("TIER 1 BASELINE CONFIGURATION")
print("═" * 60)
print(f"\n📦 PLM: {config.plm_name}")
print(f"   Embedding Dim: {get_embedding_dim(config.plm_name)}")
print(f"   Embeddings: {config.embeddings_path}")
print(f"\n🏗️  Architecture:")
print(f"   FC Hidden: {config.fc_hidden}")
print(f"   Head Strategy: {config.head_strategy}")
print(f"\n⚙️  Training:")
print(f"   Batch Size: {config.batch_size}")
print(f"   Learning Rate: {config.learning_rate}")
print(f"   Max Epochs: {config.max_epochs}")
print(f"   Early Stopping Patience: {config.patience}")
print(f"\n📊 Tracking: {'Enabled' if config.use_tracking else 'Disabled'}")
print("═" * 60)

# %% [markdown]
# ## 3. Data Loading
#
# Load pre-computed PLM embeddings from HDF5 file.

# %%
# Check embeddings exist
embeddings_path = Path(config.embeddings_path)

if not embeddings_path.exists():
    print(f"❌ Embeddings not found: {embeddings_path}")
    print(f"\n   Run extraction first:")
    print(f"   python scripts/extract_embeddings.py --plm {config.plm_name}")
else:
    import h5py
    with h5py.File(embeddings_path, 'r') as f:
        train_count = len(f['train']) if 'train' in f else 0
        cb513_count = len(f['cb513']) if 'cb513' in f else 0
        plm_name = f.attrs.get('plm_name', 'unknown')
        embedding_dim = f.attrs.get('embedding_dim', 0)
    
    print(f"✓ Embeddings found: {embeddings_path}")
    print(f"   PLM: {plm_name}")
    print(f"   Embedding Dim: {embedding_dim}")
    print(f"   Train samples: {train_count}")
    print(f"   CB513 samples: {cb513_count}")

# %%
# Create dataset
print("Loading dataset...")

full_dataset = HDF5EmbeddingDataset(
    csv_path='../../data/train.csv',
    h5_path=config.embeddings_path,
    dataset_name='train',
    max_length=config.max_seq_length,
    exclude_ids=LEAKAGE_TRAIN_IDS,
)

# Train/Val split
val_split = 0.1
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"\n📊 Dataset Split:")
print(f"   Train: {len(train_dataset):,} samples")
print(f"   Val:   {len(val_dataset):,} samples")

# %%
# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

print(f"📦 DataLoaders:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches:   {len(val_loader)}")

# %%
# Inspect a batch
sample_batch = next(iter(train_loader))

print("\n🔍 Sample Batch:")
print(f"   Features shape: {sample_batch['features'].shape}")
print(f"   SST8 shape:     {sample_batch['sst8'].shape}")
print(f"   SST3 shape:     {sample_batch['sst3'].shape}")
print(f"   Lengths:        {sample_batch['lengths'][:5].tolist()}...")

# %% [markdown]
# ## 4. Model Initialization

# %%
# Get embedding dimension from PLM
embedding_dim = get_embedding_dim(config.plm_name)

# Create model
model = Tier1Baseline(
    embedding_dim=embedding_dim,
    fc_hidden=config.fc_hidden,
    fc_dropout=config.fc_dropout,
    head_strategy=config.head_strategy,
    head_hidden=config.head_hidden,
    head_dropout=config.head_dropout,
)

print("\n🏗️  Model Created:")
print(f"   Type: Tier1Baseline")
print(f"   Embedding Dim: {embedding_dim}")
print(f"   FC Hidden: {config.fc_hidden}")
print(f"   Head Strategy: {config.head_strategy}")
print(f"\n📈 Total Parameters: {model.count_parameters():,}")

# %%
# Test forward pass
model = model.to(DEVICE)
model.eval()
with torch.no_grad():
    test_input = sample_batch['features'].to(DEVICE)
    q8_out, q3_out = model(test_input)

print("\n✓ Forward Pass Test:")
print(f"   Input:  {test_input.shape}")
print(f"   Q8 Out: {q8_out.shape}")
print(f"   Q3 Out: {q3_out.shape}")

# %% [markdown]
# ## 5. Loss Function & Optimizer

# %%
# Multi-task loss
# Options: 'focal' (default), 'weighted_ce', 'label_smoothing', 'ce', 'crf'
# Use 'crf' for CRF Negative Log-Likelihood loss which models label transitions
loss_fn = get_multitask_loss(
    loss_type=config.loss_type,
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    gamma=config.focal_gamma,  # Only used for 'focal' loss type
)

loss_type_display = config.loss_type.upper()
if config.loss_type == 'crf':
    print("📉 Loss Function: MultiTaskCRFLoss (CRF NLL)")
    print("   Uses Viterbi decoding for optimal sequence prediction")
else:
    print(f"📉 Loss Function: MultiTaskLoss ({loss_type_display})")
print(f"   Q8 Weight: {config.q8_loss_weight}")
print(f"   Q3 Weight: {config.q3_loss_weight}")
if config.loss_type == 'focal':
    print(f"   Focal Gamma: {config.focal_gamma}")

# %%
# Optimizer
optimizer = create_optimizer(
    model,
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    optimizer_type='adamw',
)

# Scheduler
scheduler = create_scheduler(
    optimizer,
    scheduler_type='cosine',
    num_epochs=config.max_epochs,
)

print("\n⚡ Optimizer: AdamW")
print(f"   Learning Rate: {config.learning_rate}")
print(f"   Weight Decay: {config.weight_decay}")
print("\n📅 Scheduler: CosineAnnealingWarmRestarts")

# %% [markdown]
# ## 6. Training with Library Trainer
#
# Using the `Trainer` class from `src/training.py` which includes:
# - Early stopping based on harmonic mean of Q8 and Q3 F1 scores
# - Automatic checkpointing
# - Experiment tracking (Trackio/W&B)

# %%
# Create trainer using library
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
    use_amp=torch.cuda.is_available(),
    use_tracking=config.use_tracking,
    trackio_space_id=config.trackio_space_id,
    hub_model_id=config.hub_model_id,
    experiment_name=config.experiment_name,
    training_config=config.__dict__,
)

print("✓ Trainer initialized")
print(f"   Checkpoint dir: {config.checkpoint_dir}")
print(f"   Mixed Precision: {trainer.use_amp}")
print(f"   Tracking: {trainer.use_tracking}")

# %%
# Run training
history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

# %% [markdown]
# ## 7. Training Visualization

# %%
# Plot training history using library function
fig = plot_training_history(
    history,
    save_path=str(Path(config.checkpoint_dir) / 'training_curves.png')
)

# %% [markdown]
# ## 8. Evaluation on CB513 Test Set

# %%
# Load CB513 test set
cb513_path = Path(config.embeddings_path)

if cb513_path.exists():
    try:
        cb513_dataset = HDF5EmbeddingDataset(
            csv_path='../../data/cb513.csv',
            h5_path=config.embeddings_path,
            dataset_name='cb513',
            max_length=config.max_seq_length,
        )
        
        cb513_loader = DataLoader(
            cb513_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )
        
        print(f"✓ CB513 test set loaded: {len(cb513_dataset)} samples")
        
        # Load best model
        best_checkpoint = torch.load(
            Path(config.checkpoint_dir) / 'best_model.pt',
            map_location=DEVICE
        )
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"✓ Best model loaded (epoch {best_checkpoint.get('epoch', 'unknown')})")
        
        # Evaluate on CB513
        original_val_loader = trainer.val_loader
        trainer.val_loader = cb513_loader
        test_metrics = trainer.validate()
        trainer.val_loader = original_val_loader
        
        print("\n" + "═" * 60)
        print("📊 CB513 TEST SET RESULTS")
        print("═" * 60)
        print(f"   Q8 Accuracy: {test_metrics['q8_accuracy']:.4f}")
        print(f"   Q3 Accuracy: {test_metrics['q3_accuracy']:.4f}")
        print(f"   Q8 F1:       {test_metrics['q8_f1']:.4f}")
        print(f"   Q3 F1:       {test_metrics['q3_f1']:.4f}")
        print("═" * 60)
        
    except Exception as e:
        print(f"⚠️ Could not evaluate on CB513: {e}")
else:
    print("⚠️ CB513 embeddings not found. Run extraction first.")

# %% [markdown]
# ## 9. Summary

# %%
print("\n" + "═" * 60)
print("🎉 TRAINING COMPLETE")
print("═" * 60)
print(f"\n📈 Best Validation Results:")
print(f"   Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")
print(f"\n💾 Checkpoints saved to: {config.checkpoint_dir}")
print("═" * 60)
