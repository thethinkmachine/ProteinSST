"""
PLM Backbone module for Full Fine-Tuning (FFT) mode.

When frozen_plm=False, the PLM backbone is included in the model and
trained end-to-end. This module wraps various PLM architectures
(ESM-2, ProtBert) with a unified interface.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import re


class PLMBackbone(nn.Module):
    """
    Unified PLM backbone for fine-tuning.
    
    Wraps ESM-2 or ProtBert models with a consistent interface.
    Handles tokenization and removes special tokens from output.
    
    Args:
        plm_name: Name of PLM ('esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert')
        freeze: If True, freeze PLM weights (equivalent to using pre-extracted embeddings)
        gradient_checkpointing: Enable gradient checkpointing to save memory
        
    Example:
        backbone = PLMBackbone('protbert', freeze=False)
        embeddings = backbone(sequences)  # (batch, seq_len, embedding_dim)
    """
    
    PLM_CONFIGS = {
        'esm2_8m': {
            'model_id': 'facebook/esm2_t6_8M_UR50D',
            'embedding_dim': 320,
            'model_type': 'esm',
        },
        'esm2_35m': {
            'model_id': 'facebook/esm2_t12_35M_UR50D',
            'embedding_dim': 480,
            'model_type': 'esm',
        },
        'esm2_650m': {
            'model_id': 'facebook/esm2_t33_650M_UR50D',
            'embedding_dim': 1280,
            'model_type': 'esm',
        },
        'protbert': {
            'model_id': 'Rostlab/prot_bert',
            'embedding_dim': 1024,
            'model_type': 'bert',
        },
    }
    
    def __init__(
        self,
        plm_name: str = 'protbert',
        freeze: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        if plm_name not in self.PLM_CONFIGS:
            raise ValueError(f"Unknown PLM: {plm_name}. Available: {list(self.PLM_CONFIGS.keys())}")
        
        config = self.PLM_CONFIGS[plm_name]
        self.plm_name = plm_name
        self.model_type = config['model_type']
        self.embedding_dim = config['embedding_dim']
        self.freeze = freeze
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(config['model_id'])
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _load_model(self, model_id: str) -> Tuple[nn.Module, Any]:
        """Load the PLM model and tokenizer."""
        if self.model_type == 'esm':
            from transformers import EsmModel, EsmTokenizer
            tokenizer = EsmTokenizer.from_pretrained(model_id)
            model = EsmModel.from_pretrained(model_id)
        elif self.model_type == 'bert':
            from transformers import BertModel, BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=False)
            model = BertModel.from_pretrained(model_id)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model, tokenizer
    
    def tokenize(
        self,
        sequences: list,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            max_length: Maximum sequence length (excluding special tokens)
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask'
        """
        if self.model_type == 'bert':
            # ProtBert needs space-separated amino acids
            sequences = [' '.join(list(seq)) for seq in sequences]
            # Replace unknown amino acids
            sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
        
        encoding = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length + 2,  # Account for special tokens
        )
        
        return encoding
    
    def forward(
        self,
        sequences: list = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Extract embeddings from sequences.
        
        Args:
            sequences: List of protein sequence strings (will be tokenized)
            input_ids: Pre-tokenized input IDs (optional, use instead of sequences)
            attention_mask: Attention mask for pre-tokenized inputs
            max_length: Maximum sequence length
            
        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim)
            Note: seq_len excludes special tokens (BOS/EOS or CLS/SEP)
        """
        # Tokenize if sequences provided
        if sequences is not None:
            encoding = self.tokenize(sequences, max_length)
            input_ids = encoding['input_ids'].to(self.model.device)
            attention_mask = encoding['attention_mask'].to(self.model.device)
        
        if input_ids is None:
            raise ValueError("Either sequences or input_ids must be provided")
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.last_hidden_state  # (batch, seq_len+2, dim)
        
        # Remove special tokens (first and last)
        # For ESM: BOS and EOS tokens
        # For BERT: CLS and SEP tokens
        embeddings = hidden_states[:, 1:-1, :]  # (batch, seq_len, dim)
        
        return embeddings
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
    
    def train(self, mode: bool = True):
        """Set training mode, respecting freeze flag."""
        super().train(mode)
        if self.freeze:
            self.model.eval()  # Keep frozen model in eval mode
        return self
    
    def to(self, device):
        """Move to device."""
        super().to(device)
        return self


class SequenceDataset(torch.utils.data.Dataset):
    """
    Simple dataset for FFT mode that returns raw sequences.
    
    Args:
        csv_path: Path to CSV with columns [id, seq, sst8, sst3]
        max_length: Maximum sequence length
        exclude_ids: IDs to exclude (leakage prevention)
        is_test: If True, no labels expected
    """
    
    def __init__(
        self,
        csv_path: str,
        max_length: int = 512,
        exclude_ids: list = None,
        is_test: bool = False,
    ):
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.is_test = is_test
        
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        
        # Center crop if needed
        truncate_start = 0
        if len(seq) > self.max_length:
            truncate_start = (len(seq) - self.max_length) // 2
            seq = seq[truncate_start:truncate_start + self.max_length]
        
        result = {
            'id': row['id'],
            'sequence': seq,
            'length': len(seq),
        }
        
        if not self.is_test:
            sst8 = row['sst8']
            sst3 = row['sst3']
            
            # Apply same truncation
            if truncate_start > 0 or len(sst8) > self.max_length:
                start = truncate_start if truncate_start > 0 else (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
            
            from ..config import SST8_TO_IDX, SST3_TO_IDX
            result['sst8'] = torch.tensor([SST8_TO_IDX[c] for c in sst8], dtype=torch.long)
            result['sst3'] = torch.tensor([SST3_TO_IDX[c] for c in sst3], dtype=torch.long)
        
        return result


def collate_fn_sequences(batch: list) -> dict:
    """Collate function for sequence batches (FFT mode)."""
    from torch.nn.utils.rnn import pad_sequence
    
    # Sort by length
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    sequences = [item['sequence'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    ids = [item['id'] for item in batch]
    
    result = {
        'sequences': sequences,
        'lengths': lengths,
        'ids': ids,
    }
    
    if 'sst8' in batch[0]:
        sst8 = [item['sst8'] for item in batch]
        sst3 = [item['sst3'] for item in batch]
        result['sst8'] = pad_sequence(sst8, batch_first=True, padding_value=-100)
        result['sst3'] = pad_sequence(sst3, batch_first=True, padding_value=-100)
    
    return result
