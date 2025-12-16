# ProteinSST - AI Coding Guidelines

## Project Overview

ProteinSST is a deep learning pipeline for **protein secondary structure prediction** (SST). It predicts 8-state (Q8: G,H,I,E,B,T,S,C) and 3-state (Q3: H,E,C) secondary structures from amino acid sequences.

## Architecture: Tiered Model Design

Models follow a progressive complexity hierarchy with two training modes:

| Tier | Architecture | Key Files |
|------|-------------|-----------|
| **Tier 1** | PLM → FC → MTL Head | `src/models/tier1_baseline.py` |
| **Tier 2** | PLM → CNN → MTL Head | `src/models/tier2_cnn.py` |
| **Tier 3** | PLM → CNN → RNN → MTL Head | `src/models/tier3_cnn_rnn.py` |

**All models output a tuple `(q8_logits, q3_logits)`** via the MTL classification head.

## Training Modes: Frozen vs FFT

The `frozen_plm` flag controls training mode:

| Mode | `frozen_plm` | Description |
|------|-------------|-------------|
| **Frozen** (default) | `True` | Uses pre-extracted PLM embeddings from HDF5. Memory efficient, fast training. |
| **Full Fine-Tuning (FFT)** | `False` | PLM is trained end-to-end. Higher GPU memory, better accuracy potential. |

### Frozen Mode (Default)
```python
from src.config import Tier1Config
config = Tier1Config(plm_name='protbert', frozen_plm=True)

# Data loading
from src.data import HDF5EmbeddingDataset, collate_fn
dataset = HDF5EmbeddingDataset('data/embeddings/protbert.h5', ...)
loader = DataLoader(dataset, collate_fn=collate_fn)

# Model forward
model = Tier1Baseline(embedding_dim=1024, frozen_plm=True)
q8, q3 = model(features)  # features: (batch, seq_len, dim)
```

### FFT Mode
```python
from src.config import Tier1Config
config = Tier1Config(
    plm_name='protbert',
    frozen_plm=False,
    gradient_checkpointing=True,  # Save memory
    batch_size=8,  # Smaller batch for FFT
    learning_rate=1e-5,  # Lower LR for FFT
)

# Data loading
from src.models import SequenceDataset, collate_fn_sequences
dataset = SequenceDataset('data/train.csv', ...)
loader = DataLoader(dataset, collate_fn=collate_fn_sequences, num_workers=0)

# Model forward
model = Tier1Baseline(
    embedding_dim=1024,
    frozen_plm=False,
    plm_name='protbert',
    gradient_checkpointing=True,
)
q8, q3 = model(sequences=sequences)  # sequences: list of strings
```

## Critical Patterns

### Multi-Task Learning (MTL) Heads
Two strategies in `src/models/classification_heads.py`:
- **`q3discarding`**: Independent Q8/Q3 heads; Q3 discarded at inference
- **`q3guided`**: Q3 computed first as prior for Q8 (easier→harder cascade)

### Configuration Dataclasses
All hyperparameters live in `src/config.py` as dataclasses (`Tier1Config`, `Tier2Config`, `Tier3Config`). Always use these—never hardcode hyperparameters:
```python
from src.config import Tier1Config
config = Tier1Config(plm_name='protbert', batch_size=32)
```

### PLM Backbone (FFT Mode)
The `PLMBackbone` class in `src/models/plm_backbone.py` wraps ESM-2/ProtBert models:
```python
from src.models import PLMBackbone
backbone = PLMBackbone(plm_name='protbert', freeze=False, gradient_checkpointing=True)
embeddings = backbone(sequences)  # sequences: list of AA strings
```

**Supported PLMs** (see `src/plm_registry.py`):
- `esm2_8m` (dim=320), `esm2_35m` (dim=480), `esm2_650m` (dim=1280)
- `protbert` (dim=1024)

### Data Leakage Prevention
Always exclude high-similarity sequences from training:
```python
from src.config import LEAKAGE_TRAIN_IDS
dataset = HDF5EmbeddingDataset(..., exclude_ids=LEAKAGE_TRAIN_IDS)
# or for FFT mode:
dataset = SequenceDataset(..., exclude_ids=LEAKAGE_TRAIN_IDS)
```

## Loss Functions

Located in `src/losses.py`. Use factory function:
```python
from src.losses import get_multitask_loss
loss_fn = get_multitask_loss(loss_type='focal', q8_weight=1.0, q3_weight=0.5)
```
Options: `focal` (recommended), `weighted_ce`, `label_smoothing`, `ce`, `crf`

## Training Workflow

Use the unified `Trainer` class from `src/training.py`:
```python
from src.training import Trainer, create_optimizer, create_scheduler
trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, ...)
trainer.train(max_epochs=50, patience=10)
```

**Early stopping** uses harmonic mean of Q8 and Q3 macro F1 scores.

## Data Augmentation

5 levels in `src/augmentation.py` (1=none, 5=aggressive). Set via `augmentation_level` in config.

## Key Commands

```bash
# Extract PLM embeddings (required for frozen mode)
python scripts/extract_embeddings.py --plm protbert --output data/embeddings/protbert.h5

# Training notebooks are in notebooks/training/
# Run cells sequentially in tier1_baseline.ipynb, tier2_cnn.ipynb, tier3_cnn_rnn.ipynb
# Set FROZEN_PLM = True/False to switch between modes
```

## File Organization

- `src/config.py` - All constants, class mappings, tier configs
- `src/data.py` - `ProteinDataset`, `HDF5EmbeddingDataset`, encoding utilities
- `src/models/` - Tier architectures + building blocks (CNN, RNN, heads)
- `src/models/plm_backbone.py` - `PLMBackbone`, `SequenceDataset` for FFT mode
- `src/training.py` - `Trainer` class with checkpointing, early stopping
- `src/metrics.py` - Q3/Q8 accuracy, per-class metrics, SOV score
- `src/hub.py` - HuggingFace Hub push/pull utilities
- `checkpoints/` - Saved model weights by tier

## Conventions

1. **Sequence-level operations**: All models process `(batch, seq_len, features)` tensors
2. **Ignore index**: Use `-100` for padding in loss computation
3. **Class weights**: Pre-computed inverse-frequency weights in `src/config.py` (`SST8_WEIGHTS`, `SST3_WEIGHTS`)
4. **Notebook imports**: Always add `sys.path.insert(0, '../..')` to access `src/`
5. **FFT Naming**: Checkpoints/experiments with `_fft` suffix indicate FFT-trained models
