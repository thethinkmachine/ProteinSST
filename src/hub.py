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
    """Create a model card README for the Hub."""
    
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    card = f"""---
library_name: pytorch
tags:
- protein
- secondary-structure-prediction
- bioinformatics
- proteinsst
license: gpl-3.0
---

# {model_name}

This model was trained using [ProteinSST](https://github.com/your-org/ProteinSST) for protein secondary structure prediction.

## Model Description

Predicts per-residue secondary structure labels:
- **Q8 (8-state)**: G, H, I, E, B, T, S, C
- **Q3 (3-state)**: H (Helix), E (Strand), C (Coil)

"""
    
    if metrics:
        card += "## Performance\n\n"
        card += "| Metric | Value |\n|--------|-------|\n"
        for key, value in metrics.items():
            card += f"| {key} | {value:.4f} |\n"
        card += "\n"
    
    if config:
        card += "## Configuration\n\n```json\n"
        card += json.dumps(config, indent=2, default=str)
        card += "\n```\n\n"
    
    card += """## Usage

```python
from src.hub import load_model_from_hub
from src.models.tier1_cnn_bilstm import CNNBiLSTM  # or appropriate model class

model = load_model_from_hub(
    repo_id="{repo_id}",
    model_class=CNNBiLSTM,
)

# For a specific revision/run:
model = load_model_from_hub(
    repo_id="{repo_id}",
    model_class=CNNBiLSTM,
    revision="abc123def",  # commit hash
)
```
""".format(repo_id=repo_id)
    
    return card
