# %% [markdown]
# # Tier 5: Fine-tuned ESM-2 Training
# 
# This notebook implements training for the **Tier 5** architecture:
# - **Fine-tuned ESM-2** pre-trained language model
# - Task-specific output heads for Q8 and Q3
# - Gradient checkpointing for memory efficiency
# - Layer-wise learning rate decay
# 
# ## Requirements
# - GPU with at least 16GB VRAM (24GB+ recommended)
# - `transformers` library
# 
# ## Expected Performance
# - Q3 Accuracy: ~91-93%
# - Q8 Accuracy: ~80-85%

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

if DEVICE == 'cpu':
    print("⚠️  Warning: Fine-tuning ESM-2 on CPU will be very slow!")
    print("   Consider using Tier 3 or 4 instead for CPU training.")

# %%
# Check GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 12:
        print("⚠️  Limited GPU memory. Using smaller ESM-2 model.")
        ESM_MODEL = "facebook/esm2_t12_35M_UR50D"  # 35M params
    else:
        ESM_MODEL = "facebook/esm2_t33_650M_UR50D"  # 650M params
else:
    ESM_MODEL = "facebook/esm2_t6_8M_UR50D"  # Smallest for CPU

print(f"Using ESM-2 model: {ESM_MODEL}")

# %%
from src.config import Tier5Config, LEAKAGE_TRAIN_IDS
from src.models.tier5_esm2_finetune import ESM2FineTune, ESM2Dataset, esm2_collate_fn
from src.losses import get_multitask_loss
from src.metrics import evaluate_model, plot_confusion_matrix
from src.training import Trainer, plot_training_history

# %% [markdown]
# ## 2. Configuration

# %%
config = Tier5Config(
    # Data
    max_seq_length=512,
    batch_size=8,  # Small for memory
    
    # Model
    esm_model=ESM_MODEL,
    freeze_layers=0,  # Full fine-tune (set >0 to freeze layers)
    
    fc_hidden=512,
    fc_dropout=0.1,
    gradient_checkpointing=True,
    
    # Training
    learning_rate=1e-5,  # Low LR for fine-tuning
    weight_decay=0.01,
    max_epochs=30,  # Fewer epochs for fine-tuning
    patience=7,
    gradient_clip=1.0,
    
    # Loss
    focal_gamma=2.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir='../../checkpoints/tier5_esm2_finetune',
)

print("Configuration:")
print(f"  Model: {config.esm_model}")
print(f"  Frozen layers: {config.freeze_layers}")
print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
print(f"  Learning rate: {config.learning_rate}")

# %% [markdown]
# ## 3. Data Loading

# %%
from transformers import EsmTokenizer
import pandas as pd

# Load tokenizer
tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)

# Load and split data
train_df = pd.read_csv('../../data/train.csv')
train_df = train_df[~train_df['id'].isin(LEAKAGE_TRAIN_IDS)].reset_index(drop=True)

np.random.seed(SEED)
val_size = int(len(train_df) * 0.1)
val_indices = np.random.choice(len(train_df), val_size, replace=False)
train_indices = [i for i in range(len(train_df)) if i not in val_indices]

train_split = train_df.iloc[train_indices].reset_index(drop=True)
val_split = train_df.iloc[val_indices].reset_index(drop=True)

train_split.to_csv('/tmp/esm2_train.csv', index=False)
val_split.to_csv('/tmp/esm2_val.csv', index=False)

# %%
# Create datasets
train_dataset = ESM2Dataset(
    '/tmp/esm2_train.csv',
    tokenizer,
    max_length=config.max_seq_length,
)

val_dataset = ESM2Dataset(
    '/tmp/esm2_val.csv',
    tokenizer,
    max_length=config.max_seq_length,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=esm2_collate_fn,
    num_workers=2,  # Fewer workers for tokenization
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=esm2_collate_fn,
    num_workers=2,
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# %% [markdown]
# ## 4. Model Initialization

# %%
model = ESM2FineTune(
    model_name=ESM_MODEL,
    freeze_layers=config.freeze_layers,
    fc_hidden=config.fc_hidden,
    fc_dropout=config.fc_dropout,
    gradient_checkpointing=config.gradient_checkpointing,
)

total_params = model.count_parameters(trainable_only=False)
trainable_params = model.count_parameters(trainable_only=True)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

model = model.to(DEVICE)

# %%
# Test forward pass
sample_batch = next(iter(train_loader))
input_ids = sample_batch['input_ids'].to(DEVICE)
attention_mask = sample_batch['attention_mask'].to(DEVICE)

with torch.no_grad():
    q8_out, q3_out = model(input_ids, attention_mask)

print(f"Q8 output shape: {q8_out.shape}")
print(f"Q3 output shape: {q3_out.shape}")

# %% [markdown]
# ## 5. Loss & Training Setup

# %%
loss_fn = get_multitask_loss(
    loss_type='focal',
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    gamma=config.focal_gamma,
)

# %%
# Layer-wise learning rate decay
param_groups = model.get_layer_lrs(
    base_lr=config.learning_rate,
    lr_decay=0.95,
)

optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

# Print LR by layer
print("Layer-wise learning rates:")
for pg in param_groups[:5]:  # First 5
    print(f"  {pg['name']}: {pg['lr']:.2e}")
print("  ...")

# %%
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=config.max_epochs // 3,
    T_mult=2,
    eta_min=1e-7,
)

# %% [markdown]
# ## 6. Custom Training Loop for ESM-2

# %%
# ESM-2 requires special handling for forward pass
class ESM2Trainer(Trainer):
    """Modified trainer for ESM-2 fine-tuning."""
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            q8_targets = batch['sst8'].to(self.device)
            q3_targets = batch['sst3'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    q8_logits, q3_logits = self.model(input_ids, attention_mask)
                    loss, q8_loss, q3_loss = self.loss_fn(
                        q8_logits, q8_targets, q3_logits, q3_targets
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                q8_logits, q3_logits = self.model(input_ids, attention_mask)
                loss, q8_loss, q3_loss = self.loss_fn(
                    q8_logits, q8_targets, q3_logits, q3_targets
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': total_loss / total_samples, 'q8_loss': 0, 'q3_loss': 0}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0.0
        all_q8_preds = []
        all_q8_targets = []
        all_q3_preds = []
        all_q3_targets = []
        total_samples = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            q8_targets = batch['sst8'].to(self.device)
            q3_targets = batch['sst3'].to(self.device)
            
            q8_logits, q3_logits = self.model(input_ids, attention_mask)
            loss, _, _ = self.loss_fn(q8_logits, q8_targets, q3_logits, q3_targets)
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_q8_preds.append(q8_logits)
            all_q8_targets.append(q8_targets)
            all_q3_preds.append(q3_logits)
            all_q3_targets.append(q3_targets)
        
        from src.metrics import compute_q8_accuracy, compute_q3_accuracy
        
        all_q8_preds = torch.cat(all_q8_preds, dim=0)
        all_q8_targets = torch.cat(all_q8_targets, dim=0)
        all_q3_preds = torch.cat(all_q3_preds, dim=0)
        all_q3_targets = torch.cat(all_q3_targets, dim=0)
        
        q8_accuracy = compute_q8_accuracy(all_q8_preds, all_q8_targets)
        q3_accuracy = compute_q3_accuracy(all_q3_preds, all_q3_targets)
        
        return {
            'loss': total_loss / total_samples,
            'q8_loss': 0,
            'q3_loss': 0,
            'q8_accuracy': q8_accuracy,
            'q3_accuracy': q3_accuracy,
        }

# %%
trainer = ESM2Trainer(
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
    trackio_space_id='thethinkmachine/trackio',  # HuggingFace Space for logs
    experiment_name='tier5_finetuned_esm2',
    hub_model_id='thethinkmachine/ProteinSST-ESM2',
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
# Load best model for evaluation
checkpoint = torch.load(f'{config.checkpoint_dir}/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Custom evaluation for ESM-2
model.eval()

all_q8_preds = []
all_q8_targets = []
all_q3_preds = []
all_q3_targets = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        q8_targets = batch['sst8'].to(DEVICE)
        q3_targets = batch['sst3'].to(DEVICE)
        
        q8_logits, q3_logits = model(input_ids, attention_mask)
        
        all_q8_preds.append(q8_logits)
        all_q8_targets.append(q8_targets)
        all_q3_preds.append(q3_logits)
        all_q3_targets.append(q3_targets)

all_q8_preds = torch.cat(all_q8_preds, dim=0)
all_q8_targets = torch.cat(all_q8_targets, dim=0)
all_q3_preds = torch.cat(all_q3_preds, dim=0)
all_q3_targets = torch.cat(all_q3_targets, dim=0)

# %%
from src.metrics import (
    compute_q8_accuracy, compute_q3_accuracy,
    compute_sst8_per_class_metrics, compute_sst3_per_class_metrics,
    compute_confusion_matrix,
)
from src.config import SST8_CLASSES, SST3_CLASSES

q8_accuracy = compute_q8_accuracy(all_q8_preds, all_q8_targets)
q3_accuracy = compute_q3_accuracy(all_q3_preds, all_q3_targets)

q8_per_class = compute_sst8_per_class_metrics(all_q8_preds, all_q8_targets)
q3_per_class = compute_sst3_per_class_metrics(all_q3_preds, all_q3_targets)

q8_cm = compute_confusion_matrix(all_q8_preds, all_q8_targets, 8)
q3_cm = compute_confusion_matrix(all_q3_preds, all_q3_targets, 3)

# %%
print("=" * 60)
print("TIER 5 (Fine-tuned ESM-2) EVALUATION")
print("=" * 60)
print(f"\nQ8 Accuracy: {q8_accuracy:.4f} ({q8_accuracy*100:.2f}%)")
print(f"Q3 Accuracy: {q3_accuracy:.4f} ({q3_accuracy*100:.2f}%)")
print(f"\nQ8 Macro F1: {q8_per_class['macro_avg']['f1']:.4f}")
print(f"Q3 Macro F1: {q3_per_class['macro_avg']['f1']:.4f}")

# %%
plot_confusion_matrix(
    q8_cm,
    SST8_CLASSES,
    title='Q8 Confusion Matrix (Tier 5 - ESM-2)',
    save_path=f'{config.checkpoint_dir}/q8_confusion_matrix.png',
)

plot_confusion_matrix(
    q3_cm,
    SST3_CLASSES,
    title='Q3 Confusion Matrix (Tier 5 - ESM-2)',
    save_path=f'{config.checkpoint_dir}/q3_confusion_matrix.png',
)

# %% [markdown]
# ## 8. Summary

# %%
print("=" * 60)
print("TIER 5 (Fine-tuned ESM-2) TRAINING COMPLETE")
print("=" * 60)
print(f"\nFinal Results:")
print(f"  Q8 Accuracy: {q8_accuracy:.4f} ({q8_accuracy*100:.2f}%)")
print(f"  Q3 Accuracy: {q3_accuracy:.4f} ({q3_accuracy*100:.2f}%)")
print(f"  Q8 Macro F1: {q8_per_class['macro_avg']['f1']:.4f}")
print(f"  Q3 Macro F1: {q3_per_class['macro_avg']['f1']:.4f}")

print(f"\nFine-tuning ESM-2 provides:")
print(f"  - State-of-the-art performance")
print(f"  - Transfer learning from 250M+ protein sequences")
print(f"  - End-to-end differentiable pipeline")
print(f"\nNote: This is the most computationally expensive tier")
print(f"      but delivers the best accuracy.")
