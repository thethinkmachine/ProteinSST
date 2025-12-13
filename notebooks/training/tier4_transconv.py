# %% [markdown]
# # Tier 4: TransConv (Transformer + Dilated CNN) Training
# 
# This notebook implements training for the **Tier 4** architecture:
# - **Transformer encoder** for global context via self-attention
# - **Dilated CNN** for multi-scale local feature extraction
# - Feature fusion combining both branches
# 
# ## Expected Performance
# - Q3 Accuracy: ~89-92%
# - Q8 Accuracy: ~78-83%

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
from src.config import Tier4Config, LEAKAGE_TRAIN_IDS
from src.data import PLMEmbeddingDataset, collate_fn
from src.models.tier4_transconv import TransConv
from src.losses import get_multitask_loss
from src.metrics import evaluate_model, plot_confusion_matrix
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

# %% [markdown]
# ## 2. Configuration

# %%
config = Tier4Config(
    # Data
    max_seq_length=512,
    batch_size=16,  # Smaller batch for larger model
    
    # Model
    embedding_dim=1280,
    embeddings_path='../../data/embeddings',
    
    # Transformer
    transformer_dim=512,
    num_transformer_layers=4,
    num_heads=8,
    transformer_dropout=0.1,
    
    # Dilated CNN
    cnn_filters=256,
    dilations=[1, 2, 4, 8],
    
    fc_hidden=256,
    fc_dropout=0.2,
    
    # Training
    learning_rate=5e-5,  # Lower LR for transformer
    weight_decay=0.01,
    max_epochs=50,
    patience=10,
    gradient_clip=1.0,
    
    # Loss
    focal_gamma=2.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir='../../checkpoints/tier4_transconv',
)

print("Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Transformer layers: {config.num_transformer_layers}")
print(f"  Attention heads: {config.num_heads}")
print(f"  Dilations: {config.dilations}")

# %% [markdown]
# ## 3. Data Loading

# %%
import pandas as pd

# Load and split data
train_df = pd.read_csv('../../data/train.csv')
train_df = train_df[~train_df['id'].isin(LEAKAGE_TRAIN_IDS)].reset_index(drop=True)

np.random.seed(SEED)
val_size = int(len(train_df) * 0.1)
val_indices = np.random.choice(len(train_df), val_size, replace=False)
train_indices = [i for i in range(len(train_df)) if i not in val_indices]

train_split = train_df.iloc[train_indices].reset_index(drop=True)
val_split = train_df.iloc[val_indices].reset_index(drop=True)

train_split.to_csv('/tmp/transconv_train.csv', index=False)
val_split.to_csv('/tmp/transconv_val.csv', index=False)

# Check for embeddings
embeddings_dir = Path(config.embeddings_path)
if not embeddings_dir.exists():
    raise FileNotFoundError(
        f"Embeddings not found at {embeddings_dir}. "
        "Run scripts/extract_embeddings.py first."
    )

# Create datasets
train_dataset = PLMEmbeddingDataset(
    '/tmp/transconv_train.csv',
    config.embeddings_path,
    max_length=config.max_seq_length,
)

val_dataset = PLMEmbeddingDataset(
    '/tmp/transconv_val.csv', 
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

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %% [markdown]
# ## 4. Model Initialization

# %%
model = TransConv(
    embedding_dim=config.embedding_dim,
    transformer_dim=config.transformer_dim,
    num_transformer_layers=config.num_transformer_layers,
    num_heads=config.num_heads,
    transformer_dropout=config.transformer_dropout,
    cnn_filters=config.cnn_filters,
    dilations=config.dilations,
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
# ## 5. Loss & Training

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
    warmup_steps=500,
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
)

# %%
history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

# %% [markdown]
# ## 6. Evaluation

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
    title='Q8 Confusion Matrix (Tier 4 - TransConv)',
    save_path=f'{config.checkpoint_dir}/q8_confusion_matrix.png',
)

plot_confusion_matrix(
    report.q3_confusion_matrix,
    SST3_CLASSES,
    title='Q3 Confusion Matrix (Tier 4 - TransConv)',
    save_path=f'{config.checkpoint_dir}/q3_confusion_matrix.png',
)

# %% [markdown]
# ## 7. Summary

# %%
print("=" * 60)
print("TIER 4 (TransConv) TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest Results:")
print(f"  Q8 Accuracy: {report.q8_accuracy:.4f} ({report.q8_accuracy*100:.2f}%)")
print(f"  Q3 Accuracy: {report.q3_accuracy:.4f} ({report.q3_accuracy*100:.2f}%)")
print(f"  Q8 Macro F1: {report.q8_macro_f1:.4f}")
print(f"  Q3 Macro F1: {report.q3_macro_f1:.4f}")

print(f"\nTransConv combines:")
print(f"  - Transformer: Global context via self-attention")
print(f"  - Dilated CNN: Multi-scale local patterns")
print(f"  - Near state-of-the-art performance")
