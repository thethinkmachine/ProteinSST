"""
PLM Registry - Unified interface for loading protein language models.

Supported PLMs:
- ESM-2 (BERT-based): esm2_8m, esm2_35m, esm2_650m
- ProtBert (BERT-based): protbert
"""

from dataclasses import dataclass
from typing import Tuple, Any, Optional
import torch
import torch.nn as nn


@dataclass
class PLMInfo:
    """Information about a protein language model."""
    model_id: str
    embedding_dim: int
    model_type: str  # 't5', 'esm', 'bert'
    description: str
    

# Registry of supported PLMs
PLM_REGISTRY = {
    # ESM-2 models (BERT architecture)
    'esm2_8m': PLMInfo(
        model_id='facebook/esm2_t6_8M_UR50D',
        embedding_dim=320,
        model_type='esm',
        description='ESM-2 8M - Fastest, smallest'
    ),
    'esm2_35m': PLMInfo(
        model_id='facebook/esm2_t12_35M_UR50D',
        embedding_dim=480,
        model_type='esm',
        description='ESM-2 35M - Good balance'
    ),
    'esm2_650m': PLMInfo(
        model_id='facebook/esm2_t33_650M_UR50D',
        embedding_dim=1280,
        model_type='esm',
        description='ESM-2 650M - Best quality'
    ),
    
    # ProtBert (BERT architecture)
    'protbert': PLMInfo(
        model_id='Rostlab/prot_bert',
        embedding_dim=1024,
        model_type='bert',
        description='ProtBert - BERT trained on proteins'
    ),
}


def get_plm_info(plm_name: str) -> PLMInfo:
    """Get PLM information by name."""
    if plm_name not in PLM_REGISTRY:
        available = ', '.join(PLM_REGISTRY.keys())
        raise ValueError(f"Unknown PLM: {plm_name}. Available: {available}")
    return PLM_REGISTRY[plm_name]


def get_embedding_dim(plm_name: str) -> int:
    """Get embedding dimension for a PLM."""
    return get_plm_info(plm_name).embedding_dim


def list_plms() -> None:
    """Print available PLMs."""
    print("Available Protein Language Models:")
    print("-" * 60)
    for name, info in PLM_REGISTRY.items():
        print(f"  {name:15} | dim={info.embedding_dim:4} | {info.description}")


def load_plm(plm_name: str, device: str = 'cuda') -> Tuple[nn.Module, Any]:
    """
    Load a protein language model and its tokenizer.
    
    Args:
        plm_name: Name of the PLM (e.g., 'esm2_650m', 'protbert')
        device: Device to load model to
        
    Returns:
        Tuple of (model, tokenizer)
    """
    info = get_plm_info(plm_name)
    
    if info.model_type == 'esm':
        return _load_esm(info.model_id, device)
    elif info.model_type == 'bert':
        return _load_protbert(info.model_id, device)
    else:
        raise ValueError(f"Unknown model type: {info.model_type}")




def _load_esm(model_id: str, device: str) -> Tuple[nn.Module, Any]:
    """Load ESM-2 model using transformers."""
    try:
        from transformers import EsmModel, EsmTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    model = EsmModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_protbert(model_id: str, device: str) -> Tuple[nn.Module, Any]:
    """Load ProtBert model using transformers."""
    try:
        from transformers import BertModel, BertTokenizer
    except ImportError:
        raise ImportError("Install transformers: pip install transformers")
    
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=False)
    model = BertModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    tokenizer: Any,
    sequences: list,
    plm_type: str,
    device: str = 'cuda',
    max_length: int = 1024,
) -> list:
    """
    Extract embeddings from a PLM for a list of sequences.
    
    Args:
        model: Loaded PLM model
        tokenizer: PLM tokenizer
        sequences: List of protein sequences (strings)
        plm_type: Type of PLM ('esm', 'bert')
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        List of embedding tensors, each of shape (seq_len, embedding_dim)
    """
    embeddings = []
    
    for seq in sequences:
        # Truncate if needed
        if len(seq) > max_length:
            seq = seq[:max_length]
        
        if plm_type == 'esm':
            # ESM-2: standard tokenization
            tokens = tokenizer(
                seq,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=max_length + 2,
            )
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[0]  # (seq_len+2, dim)
            # Remove BOS and EOS tokens
            emb = hidden[1:-1].cpu()
            
        elif plm_type == 'bert':
            # ProtBert: needs space-separated amino acids
            import re
            seq_spaced = ' '.join(list(seq))
            # Replace unknown amino acids
            seq_spaced = re.sub(r"[UZOB]", "X", seq_spaced)
            
            tokens = tokenizer(
                seq_spaced,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=max_length + 2,
            )
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[0]  # (seq_len+2, dim)
            # Remove [CLS] and [SEP] tokens
            emb = hidden[1:-1].cpu()
            
        else:
            raise ValueError(f"Unknown PLM type: {plm_type}")
        
        embeddings.append(emb)
    
    return embeddings
