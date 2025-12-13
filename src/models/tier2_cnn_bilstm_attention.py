"""
Tier 2: CNN + BiLSTM + Multi-Head Attention Model.

Architecture:
- Multi-scale 1D CNN for local feature extraction
- BiLSTM for sequential modeling
- Multi-Head Self-Attention for global context
- Residual connections and LayerNorm
- Dual output heads for Q8 and Q3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiScaleCNN(nn.Module):
    """Multi-scale 1D CNN with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_sizes: List[int] = [3, 5, 7],
    ):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])
        
        self.out_channels = out_channels * len(kernel_sizes)
        
        # Projection for residual if dimensions don't match
        self.residual_proj = None
        if in_channels != self.out_channels:
            self.residual_proj = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len, out_channels)
        """
        x_t = x.transpose(1, 2)  # (batch, channels, seq_len)
        
        conv_outputs = [conv(x_t) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x_t)
        else:
            residual = x_t
        
        out = out + residual if out.shape == residual.shape else out
        
        return out.transpose(1, 2)


class BiLSTMStack(nn.Module):
    """Stacked Bidirectional LSTM."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.output_size = hidden_size * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.layer_norm(output)
        return self.dropout(output)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with residual connection.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            (batch, seq_len, embed_dim)
        """
        residual = x
        
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        output = residual + self.dropout(attn_output)
        output = self.layer_norm(output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.ff(x)
        return self.layer_norm(residual + output)


class OutputHead(nn.Module):
    """Output head for per-residue classification."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CNNBiLSTMAttention(nn.Module):
    """
    Tier 2: CNN + BiLSTM + Multi-Head Attention model.
    
    Args:
        input_dim: Input feature dimension
        use_positional: Whether to add positional encoding
        cnn_filters: Number of CNN filters per kernel
        cnn_kernels: CNN kernel sizes
        lstm_hidden: BiLSTM hidden size
        lstm_layers: Number of BiLSTM layers
        lstm_dropout: BiLSTM dropout
        num_heads: Number of attention heads
        attention_dropout: Attention dropout
        fc_hidden: Output head hidden size
        fc_dropout: Output head dropout
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        use_positional: bool = True,
        cnn_filters: int = 64,
        cnn_kernels: List[int] = [3, 5, 7],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        fc_hidden: int = 256,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        
        # Positional encoding
        self.use_positional = use_positional
        if use_positional:
            self.positional = PositionalEncoding(input_dim, dropout=0.1)
        
        # Multi-scale CNN
        self.cnn = MultiScaleCNN(input_dim, cnn_filters, cnn_kernels)
        
        # BiLSTM
        self.bilstm = BiLSTMStack(
            self.cnn.out_channels,
            lstm_hidden,
            lstm_layers,
            lstm_dropout,
        )
        
        # Multi-Head Attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=self.bilstm.output_size,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        
        # Feed-forward
        self.ff = FeedForward(
            embed_dim=self.bilstm.output_size,
            dropout=attention_dropout,
        )
        
        # Output heads
        self.q8_head = OutputHead(
            self.bilstm.output_size,
            fc_hidden,
            8,
            fc_dropout,
        )
        
        self.q3_head = OutputHead(
            self.bilstm.output_size,
            fc_hidden,
            3,
            fc_dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            Tuple of (q8_logits, q3_logits)
        """
        # Positional encoding
        if self.use_positional:
            x = self.positional(x)
        
        # CNN
        x = self.cnn(x)
        
        # BiLSTM
        x = self.bilstm(x)
        
        # Multi-Head Attention
        x = self.attention(x)
        
        # Feed-forward
        x = self.ff(x)
        
        # Output
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x)
        
        return q8_logits, q3_logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
