"""
Tier 4: TransConv - Transformer + Dilated CNN Hybrid.

Architecture:
- Transformer encoder for global context
- Dilated CNN for multi-scale local features
- Feature fusion
- Dual output heads for Q8 and Q3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder blocks."""
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DilatedCNNBlock(nn.Module):
    """
    Dilated 1D CNN for multi-scale local feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int] = [1, 2, 4, 8],
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            for d in dilations
        ])
        
        self.out_channels = out_channels * len(dilations)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(self.out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len, out_channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.fusion(x)
        
        return x.transpose(1, 2)


class FeatureFusion(nn.Module):
    """Fuse features from Transformer and CNN branches."""
    
    def __init__(self, transformer_dim: int, cnn_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        
        total_dim = transformer_dim + cnn_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, trans_features: torch.Tensor, cnn_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([trans_features, cnn_features], dim=-1)
        return self.fusion(combined)


class OutputHead(nn.Module):
    """Output head for classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TransConv(nn.Module):
    """
    Tier 4: TransConv - Transformer + Dilated CNN hybrid model.
    
    Combines global context from Transformer with multi-scale local features from dilated CNN.
    
    Args:
        embedding_dim: Input embedding dimension (PLM embeddings)
        transformer_dim: Transformer model dimension
        num_transformer_layers: Number of Transformer layers
        num_heads: Number of attention heads
        transformer_dropout: Transformer dropout
        cnn_filters: CNN filter count
        dilations: Dilation rates for CNN
        fc_hidden: Output head hidden size
        fc_dropout: Output head dropout
    """
    
    def __init__(
        self,
        embedding_dim: int = 1280,
        transformer_dim: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        transformer_dropout: float = 0.1,
        cnn_filters: int = 256,
        dilations: List[int] = [1, 2, 4, 8],
        fc_hidden: int = 256,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(transformer_dim, dropout=transformer_dropout)
        
        # Transformer branch
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=transformer_dropout,
        )
        
        # CNN branch (operates on projected input)
        self.cnn = DilatedCNNBlock(
            in_channels=transformer_dim,
            out_channels=cnn_filters,
            dilations=dilations,
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(
            transformer_dim=transformer_dim,
            cnn_dim=self.cnn.out_channels,
            out_dim=transformer_dim,
            dropout=transformer_dropout,
        )
        
        # Output heads
        self.q8_head = OutputHead(transformer_dim, fc_hidden, 8, fc_dropout)
        self.q3_head = OutputHead(transformer_dim, fc_hidden, 3, fc_dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: PLM embeddings of shape (batch, seq_len, embedding_dim)
        
        Returns:
            Tuple of (q8_logits, q3_logits)
        """
        # Project input
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Parallel branches
        trans_out = self.transformer(x)
        cnn_out = self.cnn(x)
        
        # Fuse features
        fused = self.fusion(trans_out, cnn_out)
        
        # Output
        q8_logits = self.q8_head(fused)
        q3_logits = self.q3_head(fused)
        
        return q8_logits, q3_logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
