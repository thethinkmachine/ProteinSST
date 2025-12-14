# %% [markdown]
# # Tier 3: PLM (ESM-2) Embeddings + BiLSTM Training
# 
# This notebook implements training for the **Tier 3** architecture:
# - **Pre-computed ESM-2 embeddings** as input (1280-dim)
# - Optional 1D CNN for local refinement
# - BiLSTM for sequential modeling
# 
# ## Prerequisites
# Run the embedding extraction script first to generate ESM-2 embeddings:
# ```bash
# python scripts/extract_embeddings.py
# ```
# 
# ## Expected Performance
# - Q3 Accuracy: ~88-91%
# - Q8 Accuracy: ~77-82%

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
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# %%
from src.config import Tier3Config, LEAKAGE_TRAIN_IDS
from src.data import PLMEmbeddingDataset, collate_fn
from src.models.tier3_plm_bilstm import PLMBiLSTM
from src.losses import get_multitask_loss
from src.augmentation import EmbeddingAugmenter
from src.metrics import evaluate_model, plot_confusion_matrix
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

# %% [markdown]
# ## 2. Configuration

# %%
config = Tier3Config(
    # Data
    max_seq_length=512,
    batch_size=32,
    
    # Model
    embedding_dim=1280,  # ESM-2 650M
    embeddings_path='../../data/embeddings',
    
    use_cnn=True,
    cnn_filters=128,
    cnn_kernels=[3, 5],
    
    lstm_hidden=256,
    lstm_layers=2,
    lstm_dropout=0.2,
    
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
    augmentation_level=2,  # Light noise injection for embeddings
    
    # Checkpointing
    checkpoint_dir='../../checkpoints/tier3_plm_bilstm',
)

print("Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Embedding dim: {config.embedding_dim}")
print(f"  Use CNN: {config.use_cnn}")

# %% [markdown]
# ## 3. Check Embeddings Availability

# %%
embeddings_dir = Path(config.embeddings_path)

if not embeddings_dir.exists():
    print("⚠️  Embeddings directory not found!")
    print(f"   Expected at: {embeddings_dir.absolute()}")
    print("\nTo extract embeddings, run:")
    print("   python scripts/extract_embeddings.py")
    print("\nAlternatively, using on-the-fly embedding extraction (slower)...")
    USE_PRECOMPUTED = False
else:
    embedding_files = list(embeddings_dir.glob("*.pt"))
    print(f"✅ Found {len(embedding_files)} pre-computed embeddings")
    USE_PRECOMPUTED = True

# %% [markdown]
# ## 4. Data Loading

# %%
if USE_PRECOMPUTED:
    # Use PLM embedding dataset
    import pandas as pd
    
    # Load full data for splitting
    train_df = pd.read_csv('../../data/train.csv')
    train_df = train_df[~train_df['id'].isin(LEAKAGE_TRAIN_IDS)].reset_index(drop=True)
    
    # Split
    np.random.seed(SEED)
    val_size = int(len(train_df) * 0.1)
    val_indices = np.random.choice(len(train_df), val_size, replace=False)
    train_indices = [i for i in range(len(train_df)) if i not in val_indices]
    
    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    train_split.to_csv('/tmp/plm_train.csv', index=False)
    val_split.to_csv('/tmp/plm_val.csv', index=False)
    
    # Create datasets
    train_dataset = PLMEmbeddingDataset(
        '/tmp/plm_train.csv',
        config.embeddings_path,
        max_length=config.max_seq_length,
    )
    
    val_dataset = PLMEmbeddingDataset(
        '/tmp/plm_val.csv',
        config.embeddings_path,
        max_length=config.max_seq_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
else:
    # On-the-fly embedding extraction using OnTheFlyPLMDataset
    print("Using on-the-fly embedding extraction...")
    print("⚠️  This is slower. Pre-compute embeddings for faster training.")
    
    from transformers import EsmTokenizer, EsmModel
    from src.data import OnTheFlyPLMDataset
    import pandas as pd
    
    # Load ESM-2 model
    ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading {ESM_MODEL}...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)
    esm_model = EsmModel.from_pretrained(ESM_MODEL)
    esm_model = esm_model.to(DEVICE)
    esm_model.eval()
    print(f"✅ ESM-2 loaded")
    
    # Load and split data
    train_df = pd.read_csv('../../data/train.csv')
    train_df = train_df[~train_df['id'].isin(LEAKAGE_TRAIN_IDS)].reset_index(drop=True)
    
    np.random.seed(SEED)
    val_size = int(len(train_df) * 0.1)
    val_indices = np.random.choice(len(train_df), val_size, replace=False)
    train_indices = [i for i in range(len(train_df)) if i not in val_indices]
    
    train_split = train_df.iloc[train_indices].reset_index(drop=True)
    val_split = train_df.iloc[val_indices].reset_index(drop=True)
    
    train_split.to_csv('/tmp/plm_train.csv', index=False)
    val_split.to_csv('/tmp/plm_val.csv', index=False)
    
    # Create on-the-fly datasets
    train_dataset = OnTheFlyPLMDataset(
        '/tmp/plm_train.csv',
        esm_model=esm_model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=config.max_seq_length,
    )
    
    val_dataset = OnTheFlyPLMDataset(
        '/tmp/plm_val.csv',
        esm_model=esm_model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=config.max_seq_length,
    )
    
    # Use smaller batch size due to on-the-fly extraction
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, config.batch_size // 4),  # Smaller batch
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # No multiprocessing with GPU model
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, config.batch_size // 4),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %% [markdown]
# ## 5. Model Initialization

# %%
model = PLMBiLSTM(
    embedding_dim=config.embedding_dim,
    use_cnn=config.use_cnn,
    cnn_filters=config.cnn_filters,
    cnn_kernels=config.cnn_kernels,
    lstm_hidden=config.lstm_hidden,
    lstm_layers=config.lstm_layers,
    lstm_dropout=config.lstm_dropout,
    fc_hidden=config.fc_hidden,
    fc_dropout=config.fc_dropout,
)

print(f"Model parameters: {model.count_parameters():,}")
model = model.to(DEVICE)

# %%
# Test forward pass
sample_batch = next(iter(train_loader))
test_input = sample_batch['features'].to(DEVICE)
q8_out, q3_out = model(test_input)
print(f"Q8 output shape: {q8_out.shape}")
print(f"Q3 output shape: {q3_out.shape}")

# %% [markdown]
# ## 6. Loss Function & Training

# %%
loss_fn = get_multitask_loss(
    loss_type='focal',
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    gamma=config.focal_gamma,
)

optimizer = create_optimizer(
    model,
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

scheduler = create_scheduler(
    optimizer,
    scheduler_type='cosine',
    num_epochs=config.max_epochs,
)

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
    use_tracking=True,
    experiment_name='tier3_plm_bilstm',
    hub_model_id='thethinkmachine/ProteinSST-PLMBiLSTM',
    training_config=config,
)

# %%
history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

# %% [markdown]
# ## 7. Evaluation

# %%
fig = plot_training_history(history, save_path=f'{config.checkpoint_dir}/training_history.png')
fig.show()

# %%
checkpoint = torch.load(f'{config.checkpoint_dir}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

report = evaluate_model(
    model=model,
    dataloader=val_loader,
    device=DEVICE,
    compute_sov=True,
)

report.print_report()

# %%
from src.config import SST8_CLASSES, SST3_CLASSES

plot_confusion_matrix(
    report.q8_confusion_matrix,
    SST8_CLASSES,
    title='Q8 Confusion Matrix (Tier 3 - PLM)',
    save_path=f'{config.checkpoint_dir}/q8_confusion_matrix.png',
)

plot_confusion_matrix(
    report.q3_confusion_matrix,
    SST3_CLASSES,
    title='Q3 Confusion Matrix (Tier 3 - PLM)',
    save_path=f'{config.checkpoint_dir}/q3_confusion_matrix.png',
)

# %% [markdown]
# ## 8. Summary

# %%
print("=" * 60)
print("TIER 3 (PLM + BiLSTM) TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest Results:")
print(f"  Q8 Accuracy: {report.q8_accuracy:.4f} ({report.q8_accuracy*100:.2f}%)")
print(f"  Q3 Accuracy: {report.q3_accuracy:.4f} ({report.q3_accuracy*100:.2f}%)")
print(f"  Q8 Macro F1: {report.q8_macro_f1:.4f}")
print(f"  Q3 Macro F1: {report.q3_macro_f1:.4f}")

print(f"\nThe power of PLM embeddings:")
print(f"  - Pre-trained on 250M+ protein sequences")
print(f"  - Captures evolutionary and structural information")
print(f"  - No need for MSA (faster inference)")
