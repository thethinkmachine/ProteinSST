"""
RNN Building Blocks for Protein Secondary Structure Prediction.

Provides configurable bidirectional RNN with support for:
- LSTM (default)
- GRU
- Vanilla RNN
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class RNNConfig:
    """Configuration for RNN layer."""
    rnn_type: str = 'lstm'  # 'lstm', 'gru', 'rnn'
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    def __post_init__(self):
        valid_types = ['lstm', 'gru', 'rnn']
        if self.rnn_type not in valid_types:
            raise ValueError(f"rnn_type must be one of {valid_types}")


class ConfigurableRNN(nn.Module):
    """
    Configurable bidirectional RNN supporting LSTM, GRU, and vanilla RNN.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of stacked RNN layers
        rnn_type: Type of RNN ('lstm', 'gru', 'rnn')
        dropout: Dropout probability between layers
        bidirectional: Use bidirectional RNN
        
    Example:
        rnn = ConfigurableRNN(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            rnn_type='lstm',
            bidirectional=True,
        )
        out = rnn(x)  # (batch, seq_len, 512) for bidirectional
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Calculate output size
        self.out_channels = hidden_size * (2 if bidirectional else 1)
        
        # Select RNN class
        rnn_classes = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN,
        }
        
        if self.rnn_type not in rnn_classes:
            raise ValueError(f"rnn_type must be one of {list(rnn_classes.keys())}")
        
        rnn_class = rnn_classes[self.rnn_type]
        
        # Build RNN with appropriate kwargs
        rnn_kwargs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': bidirectional,
            'dropout': dropout if num_layers > 1 else 0,
        }
        
        # Vanilla RNN has optional nonlinearity parameter
        if self.rnn_type == 'rnn':
            rnn_kwargs['nonlinearity'] = 'tanh'
        
        self.rnn = rnn_class(**rnn_kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            lengths: Optional sequence lengths for packed sequence (not used currently)
            
        Returns:
            Output tensor of shape (batch, seq_len, hidden_size * num_directions)
        """
        # RNN forward pass
        output, _ = self.rnn(x)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


def create_rnn(
    input_size: int,
    config: Optional[RNNConfig] = None,
    **kwargs
) -> ConfigurableRNN:
    """
    Factory function to create RNN module.
    
    Args:
        input_size: Input feature dimension
        config: Optional RNNConfig object
        **kwargs: Override config parameters
        
    Returns:
        ConfigurableRNN module
    """
    if config is None:
        config = RNNConfig()
    
    return ConfigurableRNN(
        input_size=input_size,
        hidden_size=kwargs.get('hidden_size', config.hidden_size),
        num_layers=kwargs.get('num_layers', config.num_layers),
        rnn_type=kwargs.get('rnn_type', config.rnn_type),
        dropout=kwargs.get('dropout', config.dropout),
        bidirectional=kwargs.get('bidirectional', config.bidirectional),
    )
