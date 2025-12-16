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
from src.models import Tier3CNNRNN, SequenceDataset, collate_fn_sequences
from src.losses import get_multitask_loss
from src.training import Trainer, create_optimizer, create_scheduler, plot_training_history

print("✓ Library modules imported")

# %% [markdown]
# ## 2. Configuration

# %%
# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PLM_NAME = 'protbert'  # Options: 'esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert'
CNN_TYPE = 'multiscale'  # Options: 'multiscale' or 'deep'
RNN_TYPE = 'lstm'  # Options: 'lstm', 'gru', or 'rnn'

# Generate submission.csv from test.csv using trained model
GENERATE_SUBMISSION = True

# FFT Mode: Set False to train PLM end-to-end (requires more GPU memory)
# When True, uses pre-extracted frozen embeddings (default, memory efficient)
FROZEN_PLM = True

config = Tier3Config(
    # PLM
    plm_name=PLM_NAME,
    embeddings_path=f'../../data/embeddings/{PLM_NAME}.h5',
    
    # Training Mode
    frozen_plm=FROZEN_PLM,
    gradient_checkpointing=not FROZEN_PLM,  # Enable for FFT to save memory
    
    # CNN
    skip_cnn=not FROZEN_PLM,  # Optionally skip CNN in FFT mode (PLM already captures features)
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
    
    # Training - adjusted for mode
    max_seq_length=512,
    batch_size=4 if not FROZEN_PLM else 32,  # Smaller batch for FFT
    learning_rate=5e-6 if not FROZEN_PLM else 1e-4,  # Lower LR for FFT
    weight_decay=0.01,
    max_epochs=15 if not FROZEN_PLM else 50,  # Fewer epochs for FFT
    patience=5 if not FROZEN_PLM else 10,
    gradient_clip=1.0,
    
    # Loss - Options: 'focal', 'weighted_ce', 'label_smoothing', 'ce', 'crf'
    loss_type='crf',  # Use 'crf' for CRF Negative Log-Likelihood
    focal_gamma=1.0,
    q8_loss_weight=1.0,
    q3_loss_weight=0.5,
    
    # Checkpointing
    checkpoint_dir=f'../../checkpoints/tier3_{PLM_NAME}_{CNN_TYPE}_{RNN_TYPE}' + ('_fft' if not FROZEN_PLM else ''),
    
    # Tracking (enabled by default)
    use_tracking=True,
    trackio_space_id='thethinkmachine/trackio',
    hub_model_id=f'thethinkmachine/ProteinSST-{PLM_NAME}-{CNN_TYPE}-{RNN_TYPE}' + ('-fft' if not FROZEN_PLM else ''),
    experiment_name=f'tier3_{PLM_NAME}_{CNN_TYPE}_{RNN_TYPE}' + ('_fft' if not FROZEN_PLM else ''),
)

# %%
print("\n" + "═" * 60)
print("TIER 3 CNN+RNN CONFIGURATION")
print("═" * 60)
print(f"\n🔧 Mode: {'Frozen PLM' if config.frozen_plm else '🔥 Full Fine-Tuning (FFT)'}")
print(f"📦 PLM: {config.plm_name} (dim={get_embedding_dim(config.plm_name)})")
print(f"\n🏗️  CNN: {config.cnn_type}" + (" (SKIPPED in FFT)" if config.skip_cnn else ""))
if not config.skip_cnn:
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
print(f"\n⚡ Training Settings:")
print(f"   Batch Size: {config.batch_size}")
print(f"   Learning Rate: {config.learning_rate}")
print(f"   Max Epochs: {config.max_epochs}")
print(f"\n📊 Tracking: {'Enabled' if config.use_tracking else 'Disabled'}")
print("═" * 60)

# %% [markdown]
# ## 3. Data Loading
#
# - **Frozen mode**: Load pre-computed PLM embeddings from HDF5 file
# - **FFT mode**: Load raw sequences (PLM processes them on-the-fly)

# %%
if config.frozen_plm:
    embeddings_path = Path(config.embeddings_path)
    if not embeddings_path.exists():
        print(f"❌ Run: python scripts/extract_embeddings.py --plm {config.plm_name}")
    else:
        print(f"✓ Embeddings: {embeddings_path}")
else:
    print("🔥 FFT Mode: PLM will be trained end-to-end")
    print(f"   PLM: {config.plm_name}")
    print(f"   Gradient Checkpointing: {config.gradient_checkpointing}")
    embeddings_path = None

# %%
print("Loading dataset...")

if config.frozen_plm:
    full_dataset = HDF5EmbeddingDataset(
        csv_path='../../data/train.csv',
        h5_path=config.embeddings_path,
        dataset_name='train',
        max_length=config.max_seq_length,
        exclude_ids=LEAKAGE_TRAIN_IDS,
    )
    current_collate_fn = collate_fn
else:
    full_dataset = SequenceDataset(
        csv_path='../../data/train.csv',
        max_length=config.max_seq_length,
        exclude_ids=LEAKAGE_TRAIN_IDS,
    )
    current_collate_fn = collate_fn_sequences

val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True,
    collate_fn=current_collate_fn, num_workers=4 if config.frozen_plm else 0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False,
    collate_fn=current_collate_fn, num_workers=4 if config.frozen_plm else 0, pin_memory=True
)

print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"   Batches: {len(train_loader)} train, {len(val_loader)} val")

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
    skip_cnn=config.skip_cnn,
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
    # FFT
    frozen_plm=config.frozen_plm,
    plm_name=config.plm_name,
    gradient_checkpointing=config.gradient_checkpointing,
).to(DEVICE)

print("\n🏗️  Model Summary:")
print("═" * 60)
print(f"Mode: {'Frozen PLM' if config.frozen_plm else '🔥 Full Fine-Tuning (FFT)'}")
print(f"PLM: {config.plm_name}")
print(f"CNN: {config.cnn_type}" + (" (SKIPPED)" if config.skip_cnn else f", output={model.cnn.out_channels if not config.skip_cnn else 'N/A'}"))
print(f"RNN: {config.rnn_type}, output={model.rnn.out_channels}")
print(f"Head: {config.head_strategy}")

if config.frozen_plm:
    print(f"\n📈 Total Parameters: {model.count_parameters():,}")
else:
    total_params = model.count_parameters()
    head_params = model.count_head_parameters()
    plm_params = total_params - head_params
    print(f"\n📈 Parameter Breakdown:")
    print(f"   PLM Backbone: {plm_params:,} (trainable)")
    print(f"   Head Layers:  {head_params:,}")
    print(f"   Total:        {total_params:,}")
print("═" * 60)

# %%
# Compare RNN types (only in frozen mode to save memory)
if config.frozen_plm:
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
    lengths = sample_batch['lengths']
    if config.frozen_plm:
        test_input = sample_batch['features'].to(DEVICE)
        q8_out, q3_out = model(test_input, lengths=lengths)
        print(f"\n✓ Forward Pass: Input {test_input.shape} → Q8 {q8_out.shape}, Q3 {q3_out.shape}")
    else:
        test_seqs = sample_batch['sequences']
        q8_out, q3_out = model(sequences=test_seqs, lengths=lengths)
        print(f"\n✓ Forward Pass: Input {len(test_seqs)} seqs → Q8 {q8_out.shape}, Q3 {q3_out.shape}")

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
    frozen_plm=FROZEN_PLM,
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
if config.frozen_plm:
    cb513_path = Path(config.embeddings_path)
    cb513_available = cb513_path.exists()
else:
    cb513_available = Path('../../data/cb513.csv').exists()

if cb513_available:
    try:
        if config.frozen_plm:
            cb513_dataset = HDF5EmbeddingDataset(
                csv_path='../../data/cb513.csv',
                h5_path=config.embeddings_path,
                dataset_name='cb513',
                max_length=config.max_seq_length,
            )
            cb513_collate = collate_fn
        else:
            cb513_dataset = SequenceDataset(
                csv_path='../../data/cb513.csv',
                max_length=config.max_seq_length,
            )
            cb513_collate = collate_fn_sequences
        
        cb513_loader = DataLoader(
            cb513_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=cb513_collate,
            num_workers=4 if config.frozen_plm else 0,
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
    if config.frozen_plm:
        print("⚠️ CB513 embeddings not found. Run extraction first.")
    else:
        print("⚠️ CB513 CSV not found.")

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
        if config.frozen_plm:
            features = batch['features'].to(DEVICE)
            q8_logits, q3_logits = model(features, return_q3=True)
        else:
            sequences = batch['sequences']
            q8_logits, q3_logits = model(sequences=sequences, return_q3=True)
        
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
# ## 10. Generate Submission (Optional)

# %%
if GENERATE_SUBMISSION:
    from src.config import IDX_TO_SST8
    import pandas as pd
    
    print("\n" + "═" * 60)
    print("📝 GENERATING SUBMISSION")
    print("═" * 60)
    
    # Check if test data exists
    test_csv_path = Path('../../data/test.csv')
    
    if not test_csv_path.exists():
        print(f"❌ Test CSV not found: {test_csv_path}")
    elif config.frozen_plm and not Path(config.embeddings_path).exists():
        print(f"❌ Embeddings not found: {config.embeddings_path}")
    else:
        try:
            # Load test dataset
            if config.frozen_plm:
                test_dataset = HDF5EmbeddingDataset(
                    csv_path=str(test_csv_path),
                    h5_path=config.embeddings_path,
                    dataset_name='test',
                    max_length=config.max_seq_length,
                    is_test=True,
                )
                test_collate = collate_fn
            else:
                test_dataset = SequenceDataset(
                    csv_path=str(test_csv_path),
                    max_length=config.max_seq_length,
                    is_test=True,
                )
                test_collate = collate_fn_sequences
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=test_collate,
                num_workers=4 if config.frozen_plm else 0,
            )
            
            print(f"✓ Test set loaded: {len(test_dataset)} samples")
            
            # Load best model
            best_checkpoint = torch.load(
                Path(config.checkpoint_dir) / 'best_model.pt',
                map_location=DEVICE
            )
            model.load_state_dict(best_checkpoint['model_state_dict'])
            model.eval()
            print(f"✓ Best model loaded (epoch {best_checkpoint.get('epoch', 'unknown')})")
            
            # Generate predictions
            all_ids = []
            all_preds = []
            
            with torch.no_grad():
                for batch in test_loader:
                    lengths = batch['lengths']
                    ids = batch['ids']
                    
                    if config.frozen_plm:
                        features = batch['features'].to(DEVICE)
                        q8_logits, _ = model(features, lengths=lengths, return_q3=False)
                    else:
                        sequences = batch['sequences']
                        q8_logits, _ = model(sequences=sequences, lengths=lengths, return_q3=False)
                    
                    q8_preds = q8_logits.argmax(dim=-1)  # (batch, seq_len)
                    
                    for i, (sample_id, length) in enumerate(zip(ids, lengths)):
                        pred_indices = q8_preds[i, :length].cpu().numpy()
                        pred_str = ''.join([IDX_TO_SST8[idx] for idx in pred_indices])
                        all_ids.append(sample_id)
                        all_preds.append(pred_str)
            
            # Create submission DataFrame
            submission_df = pd.DataFrame({
                'id': all_ids,
                'sst8': all_preds,
            })
            
            # Save submission
            submission_path = Path(config.checkpoint_dir) / 'submission.csv'
            submission_df.to_csv(submission_path, index=False)
            
            print(f"\n✓ Submission saved: {submission_path}")
            print(f"   Total predictions: {len(submission_df)}")
            print(f"\n   Preview:")
            print(submission_df.head())
            
        except Exception as e:
            print(f"⚠️ Could not generate submission: {e}")
            import traceback
            traceback.print_exc()
else:
    print("ℹ️  Submission generation disabled. Set GENERATE_SUBMISSION = True to enable.")

# %% [markdown]
# ## 11. Summary

# %%
print("\n" + "═" * 60)
print(f"🎉 TIER 3 {config.cnn_type.upper()} + {config.rnn_type.upper()} TRAINING COMPLETE")
print("═" * 60)
print(f"\n🔧 Training Mode: {'Frozen PLM' if config.frozen_plm else '🔥 Full Fine-Tuning (FFT)'}")
print(f"   PLM: {config.plm_name}")
print(f"\n📈 Best Validation Results:")
print(f"   Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")
print(f"\n💾 Checkpoints: {config.checkpoint_dir}")
print("═" * 60)
