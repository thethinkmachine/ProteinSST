# %% [markdown]
# # 🧬 Tier 3: CNN + RNN Model Training
#
# This notebook implements the **Tier 3 CNN+RNN** architecture - the most expressive model.
#
# ## Architecture Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        TIER 3: CNN + RNN                                │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │   PLM Embeddings (L, D_plm)                                             │
# │          │                                                              │
# │          ▼                                                              │
# │   ┌─────────────────────────────────────────────────┐                   │
# │   │      CNN BLOCK (MultiscaleCNN or DeepCNN)       │                   │
# │   │   ┌───┬───┬───┐           ┌─────────────┐       │                   │
# │   │   │k=3│k=5│k=7│    OR     │ Stacked CNN │       │                   │
# │   │   └─┬─┴─┬─┴─┬─┘           │ d=1,2,4,8   │       │                   │
# │   │     └───┼───┘             └──────┬──────┘       │                   │
# │   └─────────┼─────────────────────────┼─────────────┘                   │
# │             ▼                         ▼                                 │
# │   ┌─────────────────────────────────────────────────┐                   │
# │   │          RNN BLOCK (LSTM / GRU / RNN)           │                   │
# │   │   ┌─────────────────────────────────────────┐   │                   │
# │   │   │  → Layer1 → Layer2 →                    │   │                   │
# │   │   │  ← Layer1 ← Layer2 ←  (bidirectional)   │   │                   │
# │   │   └─────────────────────────────────────────┘   │                   │
# │   │   Output: hidden × 2 (bidirectional)            │                   │
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
# ## RNN Types
#
# | Type | Description | Parameters | Speed |
# |------|-------------|------------|-------|
# | **LSTM** | Long Short-Term Memory | Most | Slowest |
# | **GRU** | Gated Recurrent Unit | 75% of LSTM | Faster |
# | **RNN** | Vanilla RNN (tanh) | Fewest | Fastest |

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
    Tier3Config, LEAKAGE_TRAIN_IDS,
    get_embedding_dim, PLM_EMBEDDING_DIMS,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier3CNNRNN
from src.losses import get_multitask_loss
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

print("✓ Library modules imported")

# %% [markdown]
# ## 2. Configuration

# %%
# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PLM_NAME = 'esm2_35m'  # Options: 'esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert'
CNN_TYPE = 'multiscale'  # Options: 'multiscale' or 'deep'
RNN_TYPE = 'lstm'  # Options: 'lstm', 'gru', or 'rnn'

config = Tier3Config(
    # PLM
    plm_name=PLM_NAME,
    embeddings_path=f'../../data/embeddings/{PLM_NAME}.h5',
    
    # CNN
    cnn_type=CNN_TYPE,
    kernel_sizes=[3, 5, 7],
    cnn_out_channels=64,
    cnn_num_layers=4,
    cnn_dilations=[1, 2, 4, 8],
    cnn_residual=True,
    cnn_dropout=0.0,
    
    # RNN
    rnn_type=RNN_TYPE,
    rnn_hidden=256,
    rnn_layers=2,
    rnn_dropout=0.3,
    rnn_bidirectional=True,
    
    # Head - Try q3guided for this tier!
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
    loss_type='crf',  # Use 'crf' for CRF Negative Log-Likelihood
    focal_gamma=1.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir=f'../../checkpoints/tier3_{PLM_NAME}_{CNN_TYPE}_{RNN_TYPE}',
    
    # Tracking (enabled by default)
    use_tracking=True,
    trackio_space_id='thethinkmachine/trackio',
    hub_model_id=f'thethinkmachine/ProteinSST-{PLM_NAME}-{CNN_TYPE}-{RNN_TYPE}',
    experiment_name=f'tier3_{PLM_NAME}_{CNN_TYPE}_{RNN_TYPE}',
)

# %%
print("\n" + "═" * 60)
print("TIER 3 CNN+RNN CONFIGURATION")
print("═" * 60)
print(f"\n📦 PLM: {config.plm_name} (dim={get_embedding_dim(config.plm_name)})")
print(f"\n🏗️  CNN: {config.cnn_type}")
if config.cnn_type == 'multiscale':
    print(f"   Kernels: {config.kernel_sizes}")
else:
    print(f"   Dilations: {config.cnn_dilations}")
print(f"\n🔄 RNN: {config.rnn_type.upper()}")
print(f"   Hidden: {config.rnn_hidden}")
print(f"   Layers: {config.rnn_layers}")
print(f"   Bidirectional: {config.rnn_bidirectional}")
print(f"   Output dim: {config.rnn_hidden * (2 if config.rnn_bidirectional else 1)}")
print(f"\n🎯 Head: {config.head_strategy}")
print(f"\n📊 Tracking: {'Enabled' if config.use_tracking else 'Disabled'}")
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

model = Tier3CNNRNN(
    embedding_dim=embedding_dim,
    # CNN
    cnn_type=config.cnn_type,
    kernel_sizes=config.kernel_sizes,
    cnn_out_channels=config.cnn_out_channels,
    cnn_num_layers=config.cnn_num_layers,
    cnn_dilations=config.cnn_dilations,
    cnn_dropout=config.cnn_dropout,
    cnn_residual=config.cnn_residual,
    # RNN
    rnn_type=config.rnn_type,
    rnn_hidden=config.rnn_hidden,
    rnn_layers=config.rnn_layers,
    rnn_dropout=config.rnn_dropout,
    rnn_bidirectional=config.rnn_bidirectional,
    # Head
    head_strategy=config.head_strategy,
    head_hidden=config.head_hidden,
    head_dropout=config.head_dropout,
).to(DEVICE)

print("\n🏗️  Model Summary:")
print("═" * 60)
print(f"CNN: {config.cnn_type}, output={model.cnn.out_channels}")
print(f"RNN: {config.rnn_type}, output={model.rnn.out_channels}")
print(f"Head: {config.head_strategy}")
print(f"\n📈 Total Parameters: {model.count_parameters():,}")
print("═" * 60)

# %%
# Compare RNN types
print("\n📊 Parameter Comparison by RNN Type:")
print("─" * 45)
for rnn_type in ['lstm', 'gru', 'rnn']:
    temp_model = Tier3CNNRNN(
        embedding_dim=embedding_dim,
        cnn_type=config.cnn_type,
        kernel_sizes=config.kernel_sizes,
        cnn_out_channels=config.cnn_out_channels,
        rnn_type=rnn_type,
        rnn_hidden=config.rnn_hidden,
        rnn_layers=config.rnn_layers,
    )
    selected = " ← selected" if rnn_type == config.rnn_type else ""
    print(f"  {rnn_type.upper():5} │ {temp_model.count_parameters():>10,} params{selected}")
    del temp_model
print("─" * 45)

# %%
# Test forward pass
sample_batch = next(iter(train_loader))
model.eval()
with torch.no_grad():
    test_input = sample_batch['features'].to(DEVICE)
    lengths = sample_batch['lengths']
    q8_out, q3_out = model(test_input, lengths=lengths)

print(f"\n✓ Forward Pass: Input {test_input.shape} → Q8 {q8_out.shape}, Q3 {q3_out.shape}")

# %% [markdown]
# ## 5. Loss & Optimizer

# %%
# Multi-task loss
# Options: 'focal' (default), 'weighted_ce', 'label_smoothing', 'ce', 'crf'
loss_fn = get_multitask_loss(
    loss_type=config.loss_type,
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    gamma=config.focal_gamma,
)

optimizer = create_optimizer(model, lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = create_scheduler(optimizer, scheduler_type='cosine', num_epochs=config.max_epochs)

print(f"✓ Loss ({config.loss_type}), optimizer, scheduler configured")

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
# ## 8. Evaluation on CB513 Test Set

# %%
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
# ## 9. Q3-Guided Analysis
#
# If using `q3guided` strategy, analyze how well Q3 guides Q8.

# %%
if config.head_strategy == 'q3guided':
    print("\n🔍 Q3-Guided Strategy Analysis:")
    print("─" * 50)
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        features = batch['features'].to(DEVICE)
        q8_logits, q3_logits = model(features, return_q3=True)
        
        # Check consistency
        consistent = model.head.check_consistency(q8_logits, q3_logits)
        mask = batch['sst8'] != -100
        valid_consistent = consistent[mask.to(DEVICE)]
        
        print(f"   Q8/Q3 Consistency: {valid_consistent.float().mean() * 100:.1f}%")
        print(f"   (How often Q8 predictions fall within Q3's predicted macro-class)")
        print("─" * 50)
else:
    print("ℹ️  Using q3discarding strategy - Q3 is trained but discarded at inference")

# %% [markdown]
# ## 10. Summary

# %%
print("\n" + "═" * 60)
print(f"🎉 TIER 3 {config.cnn_type.upper()} + {config.rnn_type.upper()} TRAINING COMPLETE")
print("═" * 60)
print(f"\n📈 Best Validation Results:")
print(f"   Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")
print(f"\n💾 Checkpoints: {config.checkpoint_dir}")
print("═" * 60)
