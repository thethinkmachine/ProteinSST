"""
Tier 1: Baseline Model - PLM Embeddings → Classification Head

Supports two modes:
- frozen_plm=True (default): Uses pre-extracted PLM embeddings from HDF5
- frozen_plm=False (FFT): Includes PLM backbone, trained end-to-end

Architecture:
    [PLM Backbone (optional)] → PLM Embeddings (L, D_plm) → FC → GELU → Dropout → MTL Head → Q8/Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .classification_heads import MTLClassificationHead, HeadConfig


class Tier1Baseline(nn.Module):
    """
    Tier 1: Baseline model using PLM embeddings.
    
    Simple feed-forward network for establishing baseline performance.
    Supports both frozen (pre-extracted) and fine-tuned PLM modes.
    
    Args:
        embedding_dim: PLM embedding dimension (768 for Ankh base, 1280 for ESM2-650M)
        fc_hidden: Hidden dimension for feature projection
        fc_dropout: Dropout probability
        head_strategy: MTL head strategy ('q3discarding' or 'q3guided')
        head_hidden: Hidden dimension for classification head
        head_dropout: Dropout for classification head
        # FFT Mode
        frozen_plm: If True, expects pre-extracted embeddings. If False, includes PLM backbone.
        plm_name: Name of PLM to use for FFT mode ('esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert')
        gradient_checkpointing: Enable gradient checkpointing for PLM (saves memory)
        
    Example:
        # Frozen mode (default) - use pre-extracted embeddings
        model = Tier1Baseline(embedding_dim=1024)
        q8, q3 = model(embeddings)
        
        # FFT mode - include PLM backbone
        model = Tier1Baseline(frozen_plm=False, plm_name='protbert')
        q8, q3 = model(sequences=['MKFLILLFNILCLFPVLAADNH...'])
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
        head_strategy: str = 'q3discarding',
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        # FFT Mode
        frozen_plm: bool = True,
        plm_name: str = 'protbert',
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.frozen_plm = frozen_plm
        self.plm_name = plm_name
        
        # PLM Backbone (only for FFT mode)
        if not frozen_plm:
            from .plm_backbone import PLMBackbone
            self.plm_backbone = PLMBackbone(
                plm_name=plm_name,
                freeze=False,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.embedding_dim = self.plm_backbone.embedding_dim
        else:
            self.plm_backbone = None
            self.embedding_dim = embedding_dim
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, fc_hidden),
            nn.GELU(),
            nn.LayerNorm(fc_hidden),
            nn.Dropout(fc_dropout),
        )
        
        # Classification head
        self.head = MTLClassificationHead(
            input_dim=fc_hidden,
            strategy=head_strategy,
            fc_hidden=head_hidden,
            fc_dropout=head_dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights (excluding PLM backbone)."""
        for name, module in self.named_modules():
            if 'plm_backbone' in name:
                continue  # Skip PLM backbone
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor = None,
        sequences: List[str] = None,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: PLM embeddings of shape (batch, seq_len, embedding_dim) - for frozen mode
            sequences: List of protein sequences (strings) - for FFT mode
            return_q3: Whether to compute Q3 predictions
            
        Returns:
            Tuple of (q8_logits, q3_logits or None)
        """
        # Get embeddings from PLM backbone or use provided features
        if not self.frozen_plm:
            if sequences is None:
                raise ValueError("sequences required for FFT mode (frozen_plm=False)")
            x = self.plm_backbone(sequences=sequences)
        else:
            if features is None:
                raise ValueError("features required for frozen mode (frozen_plm=True)")
            x = features
        
        x = self.fc(x)
        q8_logits, q3_logits = self.head(x, return_q3=return_q3)
        
        return q8_logits, q3_logits
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def count_head_parameters(self) -> int:
        """Count parameters excluding PLM backbone (for comparison)."""
        total = 0
        for name, param in self.named_parameters():
            if 'plm_backbone' not in name and param.requires_grad:
                total += param.numel()
        return total
