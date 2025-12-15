"""
Tier 1: Baseline Model - Frozen PLM Embeddings → Classification Head

This is the simplest tier, using only a feed-forward network on top of
frozen PLM embeddings. Good for establishing a baseline and fast experimentation.

Architecture:
    PLM Embeddings (L, D_plm) → FC → GELU → Dropout → MTL Head → Q8/Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .classification_heads import MTLClassificationHead, HeadConfig


class Tier1Baseline(nn.Module):
    """
    Tier 1: Baseline model using frozen PLM embeddings.
    
    Simple feed-forward network for establishing baseline performance.
    
    Args:
        embedding_dim: PLM embedding dimension (768 for Ankh base, 1280 for ESM2-650M)
        fc_hidden: Hidden dimension for feature projection
        fc_dropout: Dropout probability
        head_strategy: MTL head strategy ('q3discarding' or 'q3guided')
        head_hidden: Hidden dimension for classification head
        head_dropout: Dropout for classification head
        
    Example:
        model = Tier1Baseline(embedding_dim=768)
        q8, q3 = model(embeddings)
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
        head_strategy: str = 'q3discarding',
        head_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, fc_hidden),
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
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: PLM embeddings of shape (batch, seq_len, embedding_dim)
            return_q3: Whether to compute Q3 predictions
            
        Returns:
            Tuple of (q8_logits, q3_logits or None)
        """
        x = self.fc(features)
        q8_logits, q3_logits = self.head(x, return_q3=return_q3)
        
        return q8_logits, q3_logits
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
