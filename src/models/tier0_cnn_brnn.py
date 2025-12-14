"""
Tier 0: CNN + BRNN Model for Protein Secondary Structure Prediction.

Architecture:
- Multi-scale 1D CNN for local feature extraction
- Stacked Bidirectional vanilla RNN for sequential modeling
- Dual output heads for Q8 and Q3 prediction

This is a simpler baseline compared to Tier 1's BiLSTM architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

# Reuse components from Tier 1
from .tier1_cnn_bilstm import MultiScaleCNN, OutputHead


class BiRNNStack(nn.Module):
    """
    Stacked Bidirectional vanilla RNN layers.
    
    Simpler than BiLSTM - no gating mechanism, but faster and
    serves as a baseline to measure the benefit of LSTM.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        nonlinearity: str = 'tanh',
    ):
        super().__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity=nonlinearity,
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
        output, _ = self.rnn(x)
        return self.dropout(output)


class CNNBRNN(nn.Module):
    """
    Tier 0: CNN + BRNN model for protein secondary structure prediction.
    
    Uses vanilla bidirectional RNN instead of BiLSTM for a simpler baseline.
    
    Args:
        input_dim: Input feature dimension (default 20 for one-hot, 40 with BLOSUM)
        cnn_filters: Number of filters per CNN kernel size
        cnn_kernels: List of kernel sizes for multi-scale CNN
        rnn_hidden: Hidden size for BiRNN
        rnn_layers: Number of BiRNN layers
        rnn_dropout: Dropout for BiRNN
        rnn_nonlinearity: Activation function ('tanh' or 'relu')
        fc_hidden: Hidden size for output heads
        fc_dropout: Dropout for output heads
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        cnn_filters: int = 64,
        cnn_kernels: List[int] = [3, 5, 7],
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_nonlinearity: str = 'tanh',
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
        
        # BiRNN
        self.birnn = BiRNNStack(
            input_size=self.cnn.out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            nonlinearity=rnn_nonlinearity,
        )
        
        # Output heads
        self.q8_head = OutputHead(
            input_size=self.birnn.output_size,
            hidden_size=fc_hidden,
            num_classes=8,
            dropout=fc_dropout,
        )
        
        self.q3_head = OutputHead(
            input_size=self.birnn.output_size,
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
            elif isinstance(module, nn.RNN):
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
        
        # BiRNN sequence modeling
        x = self.birnn(x)
        
        # Output predictions
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x)
        
        return q8_logits, q3_logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"CNNBRNN(\n"
            f"  CNN: {self.cnn.out_channels} channels\n"
            f"  BiRNN: {self.birnn.output_size} hidden\n"
            f"  Parameters: {self.count_parameters():,}\n"
            f")"
        )
