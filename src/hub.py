"""
HuggingFace Hub utilities for ProteinSST models.
Enables push/pull of trained models to/from the Hub.
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Type
from datetime import datetime


try:
    from huggingface_hub import (
        PyTorchModelHubMixin,
        HfApi,
        create_repo,
        upload_folder,
    )
    HAS_HUB = True
except ImportError:
    HAS_HUB = False
    PyTorchModelHubMixin = object  # Fallback for type hints


# =============================================================================
# Hub Mixin for Models
# =============================================================================

class ProteinSSTHubMixin(PyTorchModelHubMixin if HAS_HUB else object):
    """
    Mixin to add HuggingFace Hub push/pull capabilities to ProteinSST models.
    
    Usage:
        class MyModel(nn.Module, ProteinSSTHubMixin):
            ...
        
        # Push to hub
        model.push_to_hub("username/my-model")
        
        # Load from hub
        model = MyModel.from_pretrained("username/my-model")
    """
    pass


# =============================================================================
# Hub Utilities
# =============================================================================

def push_model_to_hub(
    model: nn.Module,
    repo_id: str,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> str:
    """
    Push a trained model to HuggingFace Hub.
    
    Uses a single-repo-with-revisions approach: each training run becomes
    a new commit in the same repository, allowing version control via git.
    
    Args:
        model: Trained PyTorch model
        repo_id: HuggingFace Hub repository ID (e.g., "username/protein-sst-tier1")
        config: Model configuration dict (saved as config.json)
        metrics: Training metrics to include in commit message
        commit_message: Custom commit message (auto-generated if None)
        private: Whether to make the repo private
        token: HuggingFace API token (uses cached token if None)
        checkpoint_path: Path to additional checkpoint files to upload
    
    Returns:
        URL of the uploaded model
    """
    if not HAS_HUB:
        raise ImportError(
            "huggingface-hub is required for Hub push. "
            "Install with: pip install huggingface-hub"
        )
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token)
    except Exception as e:
        print(f"Note: Could not create repo (may already exist): {e}")
    
    # Prepare temporary directory for upload
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save model weights
        model_path = tmpdir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        
        # Save config
        if config is not None:
            config_path = tmpdir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        
        # Create model card
        model_card = _create_model_card(repo_id, config, metrics)
        readme_path = tmpdir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Copy additional checkpoint files if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            import shutil
            checkpoint_dir = Path(checkpoint_path)
            if checkpoint_dir.is_file():
                shutil.copy(checkpoint_dir, tmpdir / checkpoint_dir.name)
            else:
                for file in checkpoint_dir.glob("*"):
                    if file.is_file() and file.suffix in ['.json', '.png']:
                        shutil.copy(file, tmpdir / file.name)
        
        # Generate commit message
        if commit_message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            if metrics:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                commit_message = f"Training run {timestamp} | {metrics_str}"
            else:
                commit_message = f"Training run {timestamp}"
        
        # Upload
        url = api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
    
    print(f"✓ Model pushed to: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def load_model_from_hub(
    repo_id: str,
    model_class: Type[nn.Module],
    revision: Optional[str] = None,
    token: Optional[str] = None,
    **model_kwargs,
) -> nn.Module:
    """
    Load a model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace Hub repository ID
        model_class: Model class to instantiate
        revision: Git revision (commit hash, branch, or tag) to load
        token: HuggingFace API token
        **model_kwargs: Additional kwargs passed to model constructor
    
    Returns:
        Loaded model
    """
    if not HAS_HUB:
        raise ImportError(
            "huggingface-hub is required for Hub load. "
            "Install with: pip install huggingface-hub"
        )
    
    from huggingface_hub import hf_hub_download
    
    # Download config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            token=token,
        )
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Merge with model_kwargs (model_kwargs takes precedence)
        for key, value in config.items():
            if key not in model_kwargs:
                model_kwargs[key] = value
    except Exception:
        pass  # Config is optional
    
    # Download model weights
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
        revision=revision,
        token=token,
    )
    
    # Instantiate and load
    model = model_class(**model_kwargs)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    
    print(f"✓ Model loaded from: https://huggingface.co/{repo_id}" + 
          (f" (revision: {revision})" if revision else ""))
    
    return model


def _create_model_card(
    repo_id: str,
    config: Optional[Dict] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Create a comprehensive model card README for the Hub."""
    
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    # Determine model tier from name
    tier_info = _get_tier_info(model_name, config)
    
    card = f"""---
library_name: pytorch
tags:
- protein
- secondary-structure-prediction
- bioinformatics
- proteinsst
- sequence-labeling
- deep-learning
license: gpl-3.0
datasets:
- CB513
- CASP
pipeline_tag: token-classification
metrics:
- accuracy
- f1
---

# {model_name}

**Protein Secondary Structure Prediction** using {tier_info['architecture']}.

This model was trained using [ProteinSST](https://github.com/thethinkmachine/ProteinSST) for per-residue secondary structure prediction from amino acid sequences.

## Model Description

### Task
Predicts per-residue secondary structure labels:

| Classification | Classes | Description |
|----------------|---------|-------------|
| **Q8 (8-state)** | G, H, I, E, B, T, S, C | 3₁₀-helix, α-helix, π-helix, β-strand, β-bridge, turn, bend, coil |
| **Q3 (3-state)** | H, E, C | Helix (G+H+I), Strand (E+B), Coil (T+S+C) |

### Architecture
{tier_info['description']}

**Key Features:**
{tier_info['features']}

"""
    
    if metrics:
        card += """## Performance

### Validation Metrics

| Metric | Value |
|--------|-------|
"""
        for key, value in sorted(metrics.items()):
            display_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                card += f"| {display_key} | **{value:.4f}** |\n"
            else:
                card += f"| {display_key} | {value} |\n"
        
        card += """
### Expected Performance Range

Based on architecture tier:
| Metric | Expected Range |
|--------|----------------|
| Q3 Accuracy | """ + tier_info['q3_range'] + """ |
| Q8 Accuracy | """ + tier_info['q8_range'] + """ |

"""
    
    if config:
        card += """## Training Configuration

<details>
<summary>Click to expand configuration</summary>

```json
"""
        card += json.dumps(config, indent=2, default=str)
        card += """
```

</details>

### Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
        key_params = ['learning_rate', 'batch_size', 'max_epochs', 'max_seq_length', 
                      'focal_gamma', 'q8_loss_weight', 'q3_loss_weight', 'augmentation_level']
        for param in key_params:
            if config and param in config:
                card += f"| {param.replace('_', ' ').title()} | {config[param]} |\n"
        card += "\n"
    
    # Usage section
    card += """## Usage

### Installation

```bash
git clone https://github.com/thethinkmachine/ProteinSST.git
cd ProteinSST
pip install -e .
```

### Load Pre-trained Model

```python
from src.hub import load_model_from_hub
"""
    
    # Add appropriate model class import based on tier
    card += tier_info['import_statement']
    
    card += f"""
# Load the latest version
model = load_model_from_hub(
    repo_id="{repo_id}",
    model_class={tier_info['class_name']},
)

# Load a specific training run (use git commit hash)
model = load_model_from_hub(
    repo_id="{repo_id}",
    model_class={tier_info['class_name']},
    revision="<commit_hash>",  # From Hub revision history
)
```

### Inference

```python
import torch
from src.data import one_hot_encode, get_blosum62_features
from src.config import SST8_CLASSES, SST3_CLASSES

# Prepare input
sequence = "MVLSPADKTN..."  # Your protein sequence
{tier_info['input_prep']}

# Predict
model.eval()
with torch.no_grad():
    q8_logits, q3_logits = model(features.unsqueeze(0))  # Add batch dim

# Get predictions
q8_pred = q8_logits.argmax(dim=-1).squeeze()
q3_pred = q3_logits.argmax(dim=-1).squeeze()

# Convert to structure labels
q8_labels = ''.join([SST8_CLASSES[i] for i in q8_pred.tolist()])
q3_labels = ''.join([SST3_CLASSES[i] for i in q3_pred.tolist()])

print(f"Sequence: {{sequence}}")
print(f"Q8:       {{q8_labels}}")
print(f"Q3:       {{q3_labels}}")
```

## Training

This model was trained using the ProteinSST pipeline with:
- **Loss Function**: Focal Loss (γ=2.0) with class weights for imbalance
- **Multi-task Learning**: Joint Q8 and Q3 prediction
- **Early Stopping**: Based on harmonic mean of Q8 and Q3 macro F1
- **Data Augmentation**: Sequence masking, similar AA substitution

### Training Data
- ~7,262 protein sequences from CB513/CASP datasets
- 10% validation split (stratified by sequence length)
- Excluded 1 high-similarity pair for leakage prevention

## Limitations

- Maximum sequence length: 512 residues (longer sequences are center-cropped)
- Trained on globular proteins; performance on transmembrane/disordered regions may vary
- Q8 predictions for rare classes (I, B) may be less reliable due to class imbalance

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{proteinsst2024,
  title={{ProteinSST: Deep Learning for Protein Secondary Structure Prediction}},
  author={{ThinkMachine}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}},
}}
```

## License

This model is released under the GPL-3.0 License.
"""
    
    return card


def _get_tier_info(model_name: str, config: Optional[Dict]) -> Dict[str, str]:
    """Get tier-specific information for model card."""
    
    name_lower = model_name.lower()
    
    if 'esm' in name_lower or 'tier5' in name_lower:
        return {
            'architecture': 'Fine-tuned ESM-2',
            'description': 'Fine-tuned ESM-2 protein language model with task-specific classification heads.',
            'features': """- Pre-trained on 250M+ protein sequences
- Gradient checkpointing for memory efficiency
- Layer-wise learning rate decay
- End-to-end differentiable""",
            'q3_range': '91-93%',
            'q8_range': '80-85%',
            'class_name': 'ESM2FineTune',
            'import_statement': 'from src.models.tier5_esm2_finetune import ESM2FineTune',
            'input_prep': """# For ESM-2, use tokenizer
from transformers import EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
encoding = tokenizer(sequence, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']""",
        }
    elif 'transconv' in name_lower or 'tier4' in name_lower:
        return {
            'architecture': 'TransConv (Transformer + Dilated CNN)',
            'description': 'Hybrid architecture combining Transformer self-attention for global context with dilated CNNs for multi-scale local features.',
            'features': """- 4-layer Transformer encoder with 8 attention heads
- Dilated CNN with rates [1, 2, 4, 8]
- Feature fusion layer
- Uses ESM-2 embeddings as input""",
            'q3_range': '89-92%',
            'q8_range': '78-83%',
            'class_name': 'TransConv',
            'import_statement': 'from src.models.tier4_transconv import TransConv',
            'input_prep': """# Requires pre-computed ESM-2 embeddings
embedding = torch.load(f"embeddings/{seq_id}.pt")  # (L, 1280)
features = embedding""",
        }
    elif 'plm' in name_lower or 'tier3' in name_lower:
        return {
            'architecture': 'PLM Embeddings + BiLSTM',
            'description': 'BiLSTM sequence model operating on pre-computed ESM-2 protein language model embeddings.',
            'features': """- ESM-2 embeddings (1280-dim) as input
- Optional CNN for local refinement
- 2-layer BiLSTM (512 hidden units)
- No MSA required""",
            'q3_range': '88-91%',
            'q8_range': '77-82%',
            'class_name': 'PLMBiLSTM',
            'import_statement': 'from src.models.tier3_plm_bilstm import PLMBiLSTM',
            'input_prep': """# Requires pre-computed ESM-2 embeddings
embedding = torch.load(f"embeddings/{seq_id}.pt")  # (L, 1280)
features = embedding""",
        }
    elif 'attention' in name_lower or 'tier2' in name_lower:
        return {
            'architecture': 'CNN + BiLSTM + Multi-Head Attention',
            'description': 'Enhanced architecture with multi-scale CNN, BiLSTM, and self-attention for capturing both local and global patterns.',
            'features': """- Multi-scale 1D CNN (kernels: 3, 5, 7)
- 2-layer BiLSTM (512 hidden units)
- 8-head self-attention with residual connections
- Positional encoding""",
            'q3_range': '85-88%',
            'q8_range': '75-78%',
            'class_name': 'CNNBiLSTMAttention',
            'import_statement': 'from src.models.tier2_cnn_bilstm_attention import CNNBiLSTMAttention',
            'input_prep': """# One-hot + BLOSUM62 encoding
onehot = one_hot_encode(sequence)  # (L, 20)
blosum = get_blosum62_features(sequence)  # (L, 20)
features = torch.cat([onehot, blosum], dim=-1)  # (L, 40)""",
        }
    else:  # Default to Tier 1
        return {
            'architecture': 'CNN + BiLSTM',
            'description': 'Classic architecture combining multi-scale CNN for local feature extraction with BiLSTM for sequential modeling.',
            'features': """- Multi-scale 1D CNN (kernels: 3, 5, 7)
- 2-layer BiLSTM (512 hidden units)
- Dual output heads for Q8 and Q3
- ~2.8M trainable parameters""",
            'q3_range': '82-85%',
            'q8_range': '70-74%',
            'class_name': 'CNNBiLSTM',
            'import_statement': 'from src.models.tier1_cnn_bilstm import CNNBiLSTM',
            'input_prep': """# One-hot + BLOSUM62 encoding
onehot = one_hot_encode(sequence)  # (L, 20)
blosum = get_blosum62_features(sequence)  # (L, 20)
features = torch.cat([onehot, blosum], dim=-1)  # (L, 40)""",
        }
