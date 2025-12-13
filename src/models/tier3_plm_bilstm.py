"""
Tier 3: PLM (ESM-2) Embeddings + BiLSTM Model.

Architecture:
- Pre-computed ESM-2 embeddings as input
- Optional 1D CNN for local refinement
- BiLSTM for sequential modeling
- Dual output heads for Q8 and Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class OptionalCNN(nn.Module):
    """Optional 1D CNN layer for refining PLM embeddings."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        kernel_sizes: List[int] = [3, 5],
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
        
        # Projection to match residual
        self.proj = nn.Linear(in_channels, self.out_channels)
        self.layer_norm = nn.LayerNorm(self.out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels)
        Returns:
            (batch, seq_len, out_channels)
        """
        residual = self.proj(x)
        
        x_t = x.transpose(1, 2)
        conv_outputs = [conv(x_t) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        
        return self.layer_norm(out + residual)


class BiLSTMStack(nn.Module):
    """Stacked Bidirectional LSTM."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
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


class OutputHead(nn.Module):
    """Output head for per-residue classification."""
    
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


class PLMBiLSTM(nn.Module):
    """
    Tier 3: PLM Embeddings + BiLSTM model.
    
    Uses pre-computed ESM-2 embeddings as input features.
    
    Args:
        embedding_dim: Dimension of PLM embeddings (1280 for ESM-2 650M)
        use_cnn: Whether to apply CNN refinement
        cnn_filters: CNN filter count
        cnn_kernels: CNN kernel sizes
        lstm_hidden: BiLSTM hidden size
        lstm_layers: Number of BiLSTM layers
        lstm_dropout: BiLSTM dropout
        fc_hidden: Output head hidden size
        fc_dropout: Output head dropout
    """
    
    def __init__(
        self,
        embedding_dim: int = 1280,
        use_cnn: bool = True,
        cnn_filters: int = 128,
        cnn_kernels: List[int] = [3, 5],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        fc_hidden: int = 256,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_cnn = use_cnn
        
        # Input projection (reduce dimension if too large)
        if embedding_dim > 512:
            self.input_proj = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
            )
            current_dim = 512
        else:
            self.input_proj = nn.Identity()
            current_dim = embedding_dim
        
        # Optional CNN
        if use_cnn:
            self.cnn = OptionalCNN(current_dim, cnn_filters, cnn_kernels)
            current_dim = self.cnn.out_channels
        else:
            self.cnn = None
        
        # BiLSTM
        self.bilstm = BiLSTMStack(
            current_dim,
            lstm_hidden,
            lstm_layers,
            lstm_dropout,
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
            x: PLM embeddings of shape (batch, seq_len, embedding_dim)
        
        Returns:
            Tuple of (q8_logits, q3_logits)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Optional CNN
        if self.cnn is not None:
            x = self.cnn(x)
        
        # BiLSTM
        x = self.bilstm(x)
        
        # Output
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x)
        
        return q8_logits, q3_logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
