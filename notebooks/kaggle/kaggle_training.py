# %% [markdown]
# # 🧬 ProteinSST - Kaggle Training Notebook
#
# This notebook is designed for the **Kaggle Protein Secondary Structure Prediction** competition.
#
# ## Quick Start
# 1. Upload `src/` folder as a Kaggle Dataset (or use the cell below to install)
# 2. Upload pre-extracted embeddings as a Kaggle Dataset (or extract them here)
# 3. Configure the TIER and hyperparameters
# 4. Run all cells to train and generate `submission.csv`
#
# ## Architecture Tiers
#
# | Tier | Architecture | Parameters | Best For |
# |------|-------------|------------|----------|
# | 1 | PLM → FC → Head | ~500K | Fast baseline |
# | 2 | PLM → CNN → Head | ~800K | Local patterns |
# | 3 | PLM → CNN → RNN → Head | ~2M | Sequential dependencies |

# %% [markdown]
# ## 1. Setup & Installation

# %%
# ═══════════════════════════════════════════════════════════════════
# KAGGLE SETUP - Run this cell first!
# ═══════════════════════════════════════════════════════════════════

# Option 1: If you uploaded src/ as a dataset named 'proteinsst-src'
# import sys
# sys.path.insert(0, '/kaggle/input/proteinsst-src')

# Option 2: Clone from GitHub (uncomment if needed)
# !git clone https://github.com/thethinkmachine/ProteinSST.git
# import sys
# sys.path.insert(0, '/kaggle/working/ProteinSST')

# For local testing, use this:
import sys
sys.path.insert(0, '../..')

# Install dependencies (if not already installed)
# !pip install -q h5py transformers

# %%
import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Configuration
#
# **⚠️ IMPORTANT: Configure these settings before running!**

# %%
# ═══════════════════════════════════════════════════════════════════
# 🎯 MAIN CONFIGURATION - CHANGE THESE!
# ═══════════════════════════════════════════════════════════════════

# Model Selection
TIER = 1  # Options: 1 (baseline), 2 (CNN), 3 (CNN+RNN)
PLM_NAME = 'protbert'  # Options: 'protbert', 'esm2_8m', 'esm2_35m', 'esm2_650m'

# Architecture Options (Tier 2 & 3 only)
CNN_TYPE = 'multiscale'  # Options: 'multiscale', 'deep'
RNN_TYPE = 'lstm'  # Options: 'lstm', 'gru', 'rnn' (Tier 3 only)

# Training Settings
LOSS_TYPE = 'focal'  # Options: 'focal', 'crf', 'weighted_ce', 'ce'
HEAD_STRATEGY = 'q3guided'  # Options: 'q3guided', 'q3discarding'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10

# Output
GENERATE_SUBMISSION = True
SEED = 42

# ═══════════════════════════════════════════════════════════════════
# 📁 PATHS - Adjust for your Kaggle datasets
# ═══════════════════════════════════════════════════════════════════

# For Kaggle:
# TRAIN_CSV = '/kaggle/input/your-competition-name/train.csv'
# TEST_CSV = '/kaggle/input/your-competition-name/test.csv'
# EMBEDDINGS_PATH = '/kaggle/input/proteinsst-embeddings/protbert.h5'
# OUTPUT_DIR = '/kaggle/working'

# For local testing:
TRAIN_CSV = '../../data/train.csv'
TEST_CSV = '../../data/test.csv'
EMBEDDINGS_PATH = f'../../data/embeddings/{PLM_NAME}.h5'
OUTPUT_DIR = '../../checkpoints/kaggle'

# ═══════════════════════════════════════════════════════════════════

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"\n{'═' * 60}")
print(f"CONFIGURATION SUMMARY")
print(f"{'═' * 60}")
print(f"🏗️  Tier: {TIER}")
print(f"📦 PLM: {PLM_NAME}")
if TIER >= 2:
    print(f"🔲 CNN: {CNN_TYPE}")
if TIER >= 3:
    print(f"🔄 RNN: {RNN_TYPE}")
print(f"📉 Loss: {LOSS_TYPE}")
print(f"🎯 Head: {HEAD_STRATEGY}")
print(f"🖥️  Device: {DEVICE}")
print(f"{'═' * 60}")

# %% [markdown]
# ## 3. Import ProteinSST Modules

# %%
from src.config import (
    Tier1Config, Tier2Config, Tier3Config,
    LEAKAGE_TRAIN_IDS, get_embedding_dim, IDX_TO_SST8,
)
from src.data import HDF5EmbeddingDataset, collate_fn
from src.models import Tier1Baseline, Tier2CNN, Tier3CNNRNN
from src.losses import get_multitask_loss
from src.training import Trainer, create_optimizer, create_scheduler

print("✓ ProteinSST modules imported successfully!")

# %% [markdown]
# ## 4. Create Configuration

# %%
# Build config based on selected tier
embedding_dim = get_embedding_dim(PLM_NAME)

if TIER == 1:
    config = Tier1Config(
        plm_name=PLM_NAME,
        embeddings_path=EMBEDDINGS_PATH,
        fc_hidden=512,
        fc_dropout=0.1,
        head_strategy=HEAD_STRATEGY,
        head_hidden=256,
        head_dropout=0.1,
        max_seq_length=512,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        gradient_clip=1.0,
        loss_type=LOSS_TYPE,
        focal_gamma=1.0,
        q8_loss_weight=1.0,
        q3_loss_weight=0.5,
        checkpoint_dir=OUTPUT_DIR,
        use_tracking=False,  # Disable for Kaggle
    )
    ModelClass = Tier1Baseline

elif TIER == 2:
    config = Tier2Config(
        plm_name=PLM_NAME,
        embeddings_path=EMBEDDINGS_PATH,
        cnn_type=CNN_TYPE,
        kernel_sizes=[3, 5, 7, 11],
        cnn_out_channels=64,
        cnn_num_layers=4,
        cnn_dilations=[1, 2, 4, 8],
        cnn_residual=True,
        cnn_dropout=0.0,
        head_strategy=HEAD_STRATEGY,
        head_hidden=256,
        head_dropout=0.1,
        max_seq_length=512,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        gradient_clip=1.0,
        loss_type=LOSS_TYPE,
        focal_gamma=1.0,
        q8_loss_weight=1.0,
        q3_loss_weight=0.5,
        checkpoint_dir=OUTPUT_DIR,
        use_tracking=False,
    )
    ModelClass = Tier2CNN

elif TIER == 3:
    config = Tier3Config(
        plm_name=PLM_NAME,
        embeddings_path=EMBEDDINGS_PATH,
        skip_cnn=False,
        cnn_type=CNN_TYPE,
        kernel_sizes=[3, 5, 7],
        cnn_out_channels=64,
        cnn_num_layers=4,
        cnn_dilations=[1, 2, 4, 8],
        cnn_residual=True,
        cnn_dropout=0.0,
        rnn_type=RNN_TYPE,
        rnn_hidden=256,
        rnn_layers=2,
        rnn_dropout=0.3,
        rnn_bidirectional=True,
        head_strategy=HEAD_STRATEGY,
        head_hidden=256,
        head_dropout=0.1,
        max_seq_length=512,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        gradient_clip=1.0,
        loss_type=LOSS_TYPE,
        focal_gamma=1.0,
        q8_loss_weight=1.0,
        q3_loss_weight=0.5,
        checkpoint_dir=OUTPUT_DIR,
        use_tracking=False,
    )
    ModelClass = Tier3CNNRNN

print(f"✓ Tier {TIER} config created")

# %% [markdown]
# ## 5. Load Data

# %%
# Check embeddings exist
embeddings_path = Path(EMBEDDINGS_PATH)
if not embeddings_path.exists():
    print(f"❌ Embeddings not found: {embeddings_path}")
    print(f"\n   You need to either:")
    print(f"   1. Upload pre-extracted embeddings as a Kaggle dataset")
    print(f"   2. Run the embedding extraction cell below")
else:
    import h5py
    with h5py.File(embeddings_path, 'r') as f:
        train_count = len(f['train']) if 'train' in f else 0
        test_count = len(f['test']) if 'test' in f else 0
        plm_name = f.attrs.get('plm_name', 'unknown')
        emb_dim = f.attrs.get('embedding_dim', 0)
    
    print(f"✓ Embeddings found: {embeddings_path}")
    print(f"   PLM: {plm_name}, Dim: {emb_dim}")
    print(f"   Train: {train_count}, Test: {test_count}")

# %%
# Load training data
print("Loading training data...")

full_dataset = HDF5EmbeddingDataset(
    csv_path=TRAIN_CSV,
    h5_path=EMBEDDINGS_PATH,
    dataset_name='train',
    max_length=config.max_seq_length,
    exclude_ids=LEAKAGE_TRAIN_IDS,
)

# Train/Val split
val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print(f"\n📊 Dataset Split:")
print(f"   Train: {len(train_dataset):,} samples ({len(train_loader)} batches)")
print(f"   Val:   {len(val_dataset):,} samples ({len(val_loader)} batches)")

# %% [markdown]
# ## 6. Create Model

# %%
# Create model based on tier
if TIER == 1:
    model = ModelClass(
        embedding_dim=embedding_dim,
        fc_hidden=config.fc_hidden,
        fc_dropout=config.fc_dropout,
        head_strategy=config.head_strategy,
        head_hidden=config.head_hidden,
        head_dropout=config.head_dropout,
    )
    
elif TIER == 2:
    model = ModelClass(
        embedding_dim=embedding_dim,
        cnn_type=config.cnn_type,
        kernel_sizes=config.kernel_sizes,
        cnn_out_channels=config.cnn_out_channels,
        cnn_num_layers=config.cnn_num_layers,
        cnn_dilations=config.cnn_dilations,
        cnn_activation='relu',
        cnn_dropout=config.cnn_dropout,
        cnn_residual=config.cnn_residual,
        head_strategy=config.head_strategy,
        head_hidden=config.head_hidden,
        head_dropout=config.head_dropout,
    )
    
elif TIER == 3:
    model = ModelClass(
        embedding_dim=embedding_dim,
        skip_cnn=config.skip_cnn,
        cnn_type=config.cnn_type,
        kernel_sizes=config.kernel_sizes,
        cnn_out_channels=config.cnn_out_channels,
        cnn_num_layers=config.cnn_num_layers,
        cnn_dilations=config.cnn_dilations,
        cnn_dropout=config.cnn_dropout,
        cnn_residual=config.cnn_residual,
        rnn_type=config.rnn_type,
        rnn_hidden=config.rnn_hidden,
        rnn_layers=config.rnn_layers,
        rnn_dropout=config.rnn_dropout,
        rnn_bidirectional=config.rnn_bidirectional,
        head_strategy=config.head_strategy,
        head_hidden=config.head_hidden,
        head_dropout=config.head_dropout,
    )

model = model.to(DEVICE)

print(f"\n🏗️  Model: Tier {TIER}")
print(f"📈 Total Parameters: {model.count_parameters():,}")

# Test forward pass
sample_batch = next(iter(train_loader))
model.eval()
with torch.no_grad():
    features = sample_batch['features'].to(DEVICE)
    lengths = sample_batch['lengths']
    
    if TIER == 3:
        q8_out, q3_out = model(features, lengths=lengths)
    else:
        q8_out, q3_out = model(features)

print(f"✓ Forward pass: {features.shape} → Q8 {q8_out.shape}, Q3 {q3_out.shape}")

# %% [markdown]
# ## 7. Setup Training

# %%
# Loss function
loss_fn = get_multitask_loss(
    loss_type=config.loss_type,
    q8_weight=config.q8_loss_weight,
    q3_weight=config.q3_loss_weight,
    gamma=config.focal_gamma,
)

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

print(f"📉 Loss: {config.loss_type}")
print(f"⚡ Optimizer: AdamW (lr={config.learning_rate})")
print(f"📅 Scheduler: CosineAnnealing")

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
    use_amp=torch.cuda.is_available(),
    use_tracking=False,  # Disabled for Kaggle
)

print("✓ Trainer initialized")
print(f"   Checkpoint dir: {config.checkpoint_dir}")
print(f"   Mixed Precision: {trainer.use_amp}")

# %% [markdown]
# ## 8. Train Model 🚀

# %%
print("\n" + "═" * 60)
print("🚀 STARTING TRAINING")
print("═" * 60)

history = trainer.train(
    num_epochs=config.max_epochs,
    patience=config.patience,
    save_every=5,
)

print("\n" + "═" * 60)
print("✅ TRAINING COMPLETE")
print("═" * 60)
print(f"\n📈 Best Results:")
print(f"   Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")

# %% [markdown]
# ## 9. Generate Submission

# %%
if GENERATE_SUBMISSION:
    print("\n" + "═" * 60)
    print("📝 GENERATING SUBMISSION")
    print("═" * 60)
    
    # Load best model
    best_checkpoint = torch.load(
        Path(OUTPUT_DIR) / 'best_model.pt',
        map_location=DEVICE
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Best model loaded (epoch {best_checkpoint.get('epoch', 'unknown')})")
    
    # Load test data
    test_dataset = HDF5EmbeddingDataset(
        csv_path=TEST_CSV,
        h5_path=EMBEDDINGS_PATH,
        dataset_name='test',
        max_length=config.max_seq_length,
        is_test=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    
    print(f"✓ Test set loaded: {len(test_dataset)} samples")
    
    # Generate predictions
    all_ids = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            features = batch['features'].to(DEVICE)
            lengths = batch['lengths']
            ids = batch['ids']
            
            # Forward pass
            if TIER == 3:
                q8_logits, _ = model(features, lengths=lengths, return_q3=False)
            else:
                q8_logits, _ = model(features, return_q3=False)
            
            q8_preds = q8_logits.argmax(dim=-1)  # (batch, seq_len)
            
            # Convert to strings
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
    submission_path = Path(OUTPUT_DIR) / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved: {submission_path}")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"\n📋 Preview:")
    print(submission_df.head(10))
    
    # For Kaggle, also save to /kaggle/working for easy download
    # submission_df.to_csv('/kaggle/working/submission.csv', index=False)
else:
    print("ℹ️  Submission generation disabled. Set GENERATE_SUBMISSION = True to enable.")

# %% [markdown]
# ## 10. Summary

# %%
print("\n" + "═" * 60)
print("🎉 KAGGLE NOTEBOOK COMPLETE")
print("═" * 60)

print(f"\n🏗️  Model Configuration:")
print(f"   Tier: {TIER}")
print(f"   PLM: {PLM_NAME}")
if TIER >= 2:
    print(f"   CNN: {CNN_TYPE}")
if TIER >= 3:
    print(f"   RNN: {RNN_TYPE}")
print(f"   Loss: {LOSS_TYPE}")
print(f"   Head: {HEAD_STRATEGY}")

print(f"\n📈 Training Results:")
print(f"   Best Harmonic F1: {trainer.best_harmonic_f1:.4f}")
print(f"   Best Q8 F1:       {trainer.best_q8_f1:.4f}")
print(f"   Best Q8 Accuracy: {trainer.best_q8_accuracy:.4f}")

if GENERATE_SUBMISSION:
    print(f"\n📝 Submission:")
    print(f"   File: {submission_path}")
    print(f"   Predictions: {len(submission_df)}")

print(f"\n💾 Saved Files:")
print(f"   {OUTPUT_DIR}/best_model.pt")
print(f"   {OUTPUT_DIR}/submission.csv")

print("\n" + "═" * 60)
print("🚀 Ready to submit to Kaggle!")
print("═" * 60)

# %% [markdown]
# ---
#
# ## 📚 Appendix: Extract Embeddings (Run Once)
#
# If you don't have pre-extracted embeddings, run this cell to extract them.
# **Save the output as a Kaggle Dataset for reuse!**

# %%
# ═══════════════════════════════════════════════════════════════════
# EMBEDDING EXTRACTION (Optional - Run once and save as dataset)
# ═══════════════════════════════════════════════════════════════════

EXTRACT_EMBEDDINGS = False  # Set to True to extract

if EXTRACT_EMBEDDINGS:
    import h5py
    from transformers import AutoTokenizer, AutoModel
    
    # PLM to extract
    EXTRACT_PLM = 'protbert'  # or 'esm2_8m', 'esm2_35m', 'esm2_650m'
    
    # PLM registry
    PLM_REGISTRY = {
        'protbert': ('Rostlab/prot_bert_bfd', 1024),
        'esm2_8m': ('facebook/esm2_t6_8M_UR50D', 320),
        'esm2_35m': ('facebook/esm2_t12_35M_UR50D', 480),
        'esm2_650m': ('facebook/esm2_t33_650M_UR50D', 1280),
    }
    
    model_name, emb_dim = PLM_REGISTRY[EXTRACT_PLM]
    
    print(f"Loading {EXTRACT_PLM}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    plm_model = AutoModel.from_pretrained(model_name).eval()
    
    if torch.cuda.is_available():
        plm_model = plm_model.cuda()
    
    def extract_batch(sequences, batch_size=8):
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            
            # Tokenize (add spaces for ProtBert)
            if 'protbert' in EXTRACT_PLM:
                batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
            
            inputs = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = plm_model(**inputs).last_hidden_state
            
            # Extract embeddings (remove special tokens)
            for j, seq in enumerate(batch_seqs):
                if 'protbert' in EXTRACT_PLM:
                    seq_len = len(seq.split())
                    emb = outputs[j, 1:seq_len+1, :].cpu().numpy()
                else:
                    seq_len = len(sequences[i+j])
                    emb = outputs[j, 1:seq_len+1, :].cpu().numpy()
                embeddings.append(emb)
        
        return embeddings
    
    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"\nExtracting train embeddings ({len(train_df)} samples)...")
    train_embs = extract_batch(train_df['sequence'].tolist())
    
    print(f"\nExtracting test embeddings ({len(test_df)} samples)...")
    test_embs = extract_batch(test_df['sequence'].tolist())
    
    # Save to HDF5
    output_h5 = f'{OUTPUT_DIR}/{EXTRACT_PLM}.h5'
    print(f"\nSaving to {output_h5}...")
    
    with h5py.File(output_h5, 'w') as f:
        f.attrs['plm_name'] = EXTRACT_PLM
        f.attrs['embedding_dim'] = emb_dim
        
        train_grp = f.create_group('train')
        for i, emb in enumerate(train_embs):
            train_grp.create_dataset(str(train_df.iloc[i]['id']), data=emb)
        
        test_grp = f.create_group('test')
        for i, emb in enumerate(test_embs):
            test_grp.create_dataset(str(test_df.iloc[i]['id']), data=emb)
    
    print(f"✅ Embeddings saved to {output_h5}")
    print(f"   Download this file and upload as a Kaggle Dataset!")
