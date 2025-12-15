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
        config: Model configuration dict or dataclass (saved as config.json)
        metrics: Training metrics to include in commit message
        commit_message: Custom commit message (auto-generated if None)
        private: Whether to make the repo private
        token: HuggingFace API token (uses cached token if None)
        checkpoint_path: Path to additional checkpoint files to upload
    
    Returns:
        URL of the uploaded model
    """
    # Convert dataclass to dict if necessary
    if config is not None and hasattr(config, '__dataclass_fields__'):
        import dataclasses
        config = dataclasses.asdict(config)
    
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
                      'loss_type', 'focal_gamma', 'q8_loss_weight', 'q3_loss_weight', 'augmentation_level']
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

# For CRF models, use Viterbi decoding instead:
# from src.losses import MultiTaskCRFLoss
# loss_fn = MultiTaskCRFLoss()
# q8_pred, q3_pred = loss_fn.decode(q8_logits, q3_logits)

# Convert to structure labels
q8_labels = ''.join([SST8_CLASSES[i] for i in q8_pred.tolist()])
q3_labels = ''.join([SST3_CLASSES[i] for i in q3_pred.tolist()])

print(f"Sequence: {{sequence}}")
print(f"Q8:       {{q8_labels}}")
print(f"Q3:       {{q3_labels}}")
```

## Training

"""
    
    # Add loss type specific information
    loss_type = config.get('loss_type', 'focal') if config else 'focal'
    if loss_type == 'crf':
        card += """This model was trained using the ProteinSST pipeline with:
- **Loss Function**: CRF Negative Log-Likelihood (models label transitions)
- **Decoding**: Viterbi algorithm for optimal sequence prediction
- **Multi-task Learning**: Joint Q8 and Q3 prediction with independent CRF layers
- **Early Stopping**: Based on harmonic mean of Q8 and Q3 macro F1
- **Data Augmentation**: Sequence masking, similar AA substitution

> **Note**: For inference, use the `MultiTaskCRFLoss.decode()` method for Viterbi decoding
> instead of `argmax()` to get optimal label sequences that respect transition constraints.

"""
    else:
        focal_gamma = config.get('focal_gamma', 2.0) if config else 2.0
        card += f"""This model was trained using the ProteinSST pipeline with:
- **Loss Function**: Focal Loss (γ={focal_gamma}) with class weights for imbalance
- **Multi-task Learning**: Joint Q8 and Q3 prediction
- **Early Stopping**: Based on harmonic mean of Q8 and Q3 macro F1
- **Data Augmentation**: Sequence masking, similar AA substitution

"""
    
    card += """### Training Data
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
    """Get model-specific information for model card based on tier and config."""
    
    model_name_lower = model_name.lower()
    
    # Detect tier from model name or config
    if 'tier3' in model_name_lower or (config and 'rnn_type' in config):
        # Tier 3: CNN + RNN
        cnn_type = config.get('cnn_type', 'multiscale') if config else 'multiscale'
        rnn_type = config.get('rnn_type', 'lstm') if config else 'lstm'
        return {
            'architecture': f'{cnn_type.title()}CNN + Bi{rnn_type.upper()}',
            'description': f"""Tier 3 model combining CNN for local pattern extraction with bidirectional {rnn_type.upper()} for sequential modeling.

**Architecture Flow:**
```
PLM Embeddings → {cnn_type.title()}CNN → Bi{rnn_type.upper()} → MTL Head → Q8/Q3
```""",
            'features': f"""- PLM embeddings ({config.get('plm_name', 'ESM-2') if config else 'ESM-2'})
- {cnn_type.title()} CNN for local motif extraction
- Bidirectional {rnn_type.upper()} for sequential dependencies
- Multi-task classification head
- Supports CRF loss for label transition modeling""",
            'q3_range': '88-92%',
            'q8_range': '74-80%',
            'class_name': 'Tier3CNNRNN',
            'import_statement': 'from src.models import Tier3CNNRNN',
            'input_prep': """# PLM embeddings required
from src.data import HDF5EmbeddingDataset
# Load pre-computed embeddings or extract on-the-fly""",
        }
    
    elif 'tier2' in model_name_lower or (config and 'cnn_type' in config and 'rnn_type' not in config):
        # Tier 2: CNN only
        cnn_type = config.get('cnn_type', 'multiscale') if config else 'multiscale'
        return {
            'architecture': f'{cnn_type.title()}CNN',
            'description': f"""Tier 2 model using {cnn_type} CNN for local pattern extraction.

**Architecture Flow:**
```
PLM Embeddings → {cnn_type.title()}CNN → MTL Head → Q8/Q3
```""",
            'features': f"""- PLM embeddings ({config.get('plm_name', 'ESM-2') if config else 'ESM-2'})
- {cnn_type.title()} CNN with {'multiple kernel sizes' if cnn_type == 'multiscale' else 'dilated convolutions'}
- Multi-task classification head
- Supports CRF loss for label transition modeling""",
            'q3_range': '86-90%',
            'q8_range': '72-78%',
            'class_name': 'Tier2CNN',
            'import_statement': 'from src.models import Tier2CNN',
            'input_prep': """# PLM embeddings required
from src.data import HDF5EmbeddingDataset
# Load pre-computed embeddings""",
        }
    
    elif 'tier1' in model_name_lower or (config and 'fc_hidden' in config and 'cnn_type' not in config):
        # Tier 1: Baseline FC
        return {
            'architecture': 'PLM + FC Baseline',
            'description': """Tier 1 baseline model with frozen PLM embeddings and feed-forward classification.

**Architecture Flow:**
```
PLM Embeddings → Linear → GELU → LayerNorm → MTL Head → Q8/Q3
```""",
            'features': f"""- PLM embeddings ({config.get('plm_name', 'ESM-2') if config else 'ESM-2'})
- Simple feed-forward feature projection
- Multi-task classification head
- Supports CRF loss for label transition modeling
- Lightweight (~500K parameters)""",
            'q3_range': '84-88%',
            'q8_range': '68-74%',
            'class_name': 'Tier1Baseline',
            'import_statement': 'from src.models import Tier1Baseline',
            'input_prep': """# PLM embeddings required
from src.data import HDF5EmbeddingDataset
# Load pre-computed embeddings""",
        }
    
    else:
        # Fallback: assume ESM2 fine-tune or generic
        return {
            'architecture': 'Fine-tuned ESM-2',
            'description': 'Fine-tuned ESM-2 protein language model with task-specific classification heads.',
            'features': """- Pre-trained on 250M+ protein sequences
- Gradient checkpointing for memory efficiency
- Layer-wise learning rate decay
- End-to-end differentiable
- Supports CRF loss for label transition modeling""",
            'q3_range': '91-93%',
            'q8_range': '80-85%',
            'class_name': 'ESM2FineTune',
            'import_statement': 'from src.models.esm2_finetune import ESM2FineTune',
            'input_prep': """# For ESM-2, use tokenizer
from transformers import EsmTokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
encoding = tokenizer(sequence, return_tensors='pt')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']""",
        }
