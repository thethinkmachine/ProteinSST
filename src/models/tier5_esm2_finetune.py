"""
Tier 5: Fine-tuned ESM-2 Model for Protein Secondary Structure Prediction.

Architecture:
- ESM-2 pre-trained model (fine-tuned)
- Task-specific output heads for Q8 and Q3
- Gradient checkpointing for memory efficiency
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from transformers import EsmModel, EsmConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class OutputHead(nn.Module):
    """Output head for per-residue classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ESM2FineTune(nn.Module):
    """
    Tier 5: Fine-tuned ESM-2 model for SST prediction.
    
    Uses the HuggingFace transformers library for ESM-2.
    
    Args:
        model_name: ESM-2 model name from HuggingFace
            - "facebook/esm2_t6_8M_UR50D" (8M params, fastest)
            - "facebook/esm2_t12_35M_UR50D" (35M params, good balance)
            - "facebook/esm2_t33_650M_UR50D" (650M params, best quality)
        freeze_layers: Number of layers to freeze (0 = full fine-tune)
        fc_hidden: Hidden size for output heads
        fc_dropout: Dropout for output heads
        gradient_checkpointing: Enable gradient checkpointing to save memory
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        freeze_layers: int = 0,
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for ESM-2 fine-tuning. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        
        # Load pre-trained ESM-2
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.esm.gradient_checkpointing_enable()
        
        # Freeze layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        # Get hidden size from config
        hidden_size = self.esm.config.hidden_size
        
        # Task-specific output heads
        self.q8_head = OutputHead(hidden_size, fc_hidden, 8, fc_dropout)
        self.q3_head = OutputHead(hidden_size, fc_hidden, 3, fc_dropout)
        
        self._init_heads()
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first N transformer layers."""
        # Freeze embeddings
        for param in self.esm.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified number of layers
        for i, layer in enumerate(self.esm.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _init_heads(self):
        """Initialize output head weights."""
        for module in [self.q8_head, self.q3_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized sequence IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
        
        Returns:
            Tuple of (q8_logits, q3_logits)
        """
        # ESM-2 forward pass
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Get hidden states (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Remove special tokens ([CLS] at start, [EOS] at end)
        # For ESM-2, first token is BOS and last is EOS
        hidden_states = hidden_states[:, 1:-1, :]
        
        # Output predictions
        q8_logits = self.q8_head(hidden_states)
        q3_logits = self.q3_head(hidden_states)
        
        return q8_logits, q3_logits
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_layer_lrs(self, base_lr: float = 1e-5, lr_decay: float = 0.95):
        """
        Get layer-wise learning rates for fine-tuning.
        
        Deeper layers get higher LR, embeddings and early layers get lower LR.
        
        Args:
            base_lr: Base learning rate for output heads
            lr_decay: Decay factor per layer (going backwards)
        
        Returns:
            List of parameter groups with layer-wise LRs
        """
        param_groups = []
        
        # Output heads - full LR
        param_groups.append({
            'params': list(self.q8_head.parameters()) + list(self.q3_head.parameters()),
            'lr': base_lr,
            'name': 'heads',
        })
        
        # ESM layers - decreasing LR
        num_layers = len(self.esm.encoder.layer)
        for i, layer in enumerate(reversed(self.esm.encoder.layer)):
            layer_lr = base_lr * (lr_decay ** (i + 1))
            trainable_params = [p for p in layer.parameters() if p.requires_grad]
            if trainable_params:
                param_groups.append({
                    'params': trainable_params,
                    'lr': layer_lr,
                    'name': f'layer_{num_layers - 1 - i}',
                })
        
        # Embeddings - lowest LR
        emb_params = [p for p in self.esm.embeddings.parameters() if p.requires_grad]
        if emb_params:
            param_groups.append({
                'params': emb_params,
                'lr': base_lr * (lr_decay ** (num_layers + 1)),
                'name': 'embeddings',
            })
        
        return param_groups


class ESM2Dataset(torch.utils.data.Dataset):
    """
    Dataset for ESM-2 fine-tuning with on-the-fly tokenization.
    
    Args:
        csv_path: Path to data CSV
        tokenizer: ESM-2 tokenizer
        max_length: Maximum sequence length
        exclude_ids: IDs to exclude
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        exclude_ids: list = None,
        is_test: bool = False,
    ):
        import pandas as pd
        
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        
        # Truncate if needed
        if len(seq) > self.max_length:
            start = (len(seq) - self.max_length) // 2
            seq = seq[start:start + self.max_length]
        
        # Tokenize
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=self.max_length + 2,  # Account for special tokens
        )
        
        result = {
            'id': row['id'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'length': len(seq),
        }
        
        if not self.is_test:
            from .tier1_cnn_bilstm import encode_sst8, encode_sst3
            
            sst8 = row['sst8']
            sst3 = row['sst3']
            if len(sst8) > self.max_length:
                start = (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
            
            # Import encoding functions
            from ..config import SST8_TO_IDX, SST3_TO_IDX
            
            result['sst8'] = torch.tensor([SST8_TO_IDX[c] for c in sst8])
            result['sst3'] = torch.tensor([SST3_TO_IDX[c] for c in sst3])
        
        return result


def esm2_collate_fn(batch):
    """Collate function for ESM-2 fine-tuning."""
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    ids = [item['id'] for item in batch]
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=1)  # ESM-2 pad token
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    result = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'lengths': lengths,
        'ids': ids,
    }
    
    if 'sst8' in batch[0]:
        sst8 = [item['sst8'] for item in batch]
        sst3 = [item['sst3'] for item in batch]
        result['sst8'] = pad_sequence(sst8, batch_first=True, padding_value=-100)
        result['sst3'] = pad_sequence(sst3, batch_first=True, padding_value=-100)
    
    return result
