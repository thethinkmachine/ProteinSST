"""
Tier 1: CNN + BiLSTM Model for Protein Secondary Structure Prediction.

Architecture:
- Multi-scale 1D CNN for local feature extraction
- Stacked BiLSTM for sequential modeling
- Dual output heads for Q8 and Q3 prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for extracting local features with different receptive fields.
    """
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
        
        Returns:
            Output tensor of shape (batch, seq_len, out_channels * num_kernels)
        """
        # Transpose for Conv1D: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each convolution and concatenate
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        
        # Transpose back: (batch, seq_len, channels)
        return x.transpose(1, 2)


class BiLSTMStack(nn.Module):
    """
    Stacked Bidirectional LSTM layers.
    """
    
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
        self.output_size = hidden_size * 2  # Bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size * 2)
        """
        output, _ = self.lstm(x)
        return self.dropout(output)


class OutputHead(nn.Module):
    """
    Output head for per-residue classification.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        return self.fc(x)


class CNNBiLSTM(nn.Module):
    """
    Tier 1: CNN + BiLSTM model for protein secondary structure prediction.
    
    Args:
        input_dim: Input feature dimension (default 20 for one-hot, 40 with BLOSUM)
        cnn_filters: Number of filters per CNN kernel size
        cnn_kernels: List of kernel sizes for multi-scale CNN
        lstm_hidden: Hidden size for BiLSTM
        lstm_layers: Number of BiLSTM layers
        lstm_dropout: Dropout for BiLSTM
        fc_hidden: Hidden size for output heads
        fc_dropout: Dropout for output heads
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        cnn_filters: int = 64,
        cnn_kernels: List[int] = [3, 5, 7],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        fc_hidden: int = 256,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        
        # Multi-scale CNN
        self.cnn = MultiScaleCNN(
            in_channels=input_dim,
            out_channels=cnn_filters,
            kernel_sizes=cnn_kernels,
        )
        
        # BiLSTM
        self.bilstm = BiLSTMStack(
            input_size=self.cnn.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        
        # Output heads
        self.q8_head = OutputHead(
            input_size=self.bilstm.output_size,
            hidden_size=fc_hidden,
            num_classes=8,
            dropout=fc_dropout,
        )
        
        self.q3_head = OutputHead(
            input_size=self.bilstm.output_size,
            hidden_size=fc_hidden,
            num_classes=3,
            dropout=fc_dropout,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        
        Returns:
            Tuple of (q8_logits, q3_logits), each of shape (batch, seq_len, num_classes)
        """
        # CNN feature extraction
        x = self.cnn(x)
        
        # BiLSTM sequence modeling
        x = self.bilstm(x)
        
        # Output predictions
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x)
        
        return q8_logits, q3_logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"CNNBiLSTM(\n"
            f"  CNN: {self.cnn.out_channels} channels\n"
            f"  BiLSTM: {self.bilstm.output_size} hidden\n"
            f"  Parameters: {self.count_parameters():,}\n"
            f")"
        )
