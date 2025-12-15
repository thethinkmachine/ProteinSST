"""
CNN Building Blocks for Protein Secondary Structure Prediction.

Two types of CNN architectures:
- MultiscaleCNN: Parallel branches with different kernel sizes (widens in feature space)
- DeepCNN: Stacked layers with configurable dilation (deepens in context space)

All parameters are individually configurable per layer.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class CNNLayerConfig:
    """Configuration for a single CNN layer."""
    out_channels: int = 64
    kernel_size: int = 3
    dilation: int = 1
    stride: int = 1
    padding: Union[str, int] = 'same'  # 'same', 'valid', or int
    activation: str = 'relu'  # 'relu', 'gelu', 'silu', 'none'
    batch_norm: bool = True
    dropout: float = 0.0
    residual: bool = False  # Add residual connection (requires matching channels)
    
    def __post_init__(self):
        # Validate activation
        valid_activations = ['relu', 'gelu', 'silu', 'none']
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(inplace=True),
        'none': nn.Identity(),
    }
    return activations.get(name, nn.ReLU(inplace=True))


def calculate_padding(kernel_size: int, dilation: int, padding: Union[str, int]) -> int:
    """Calculate padding to achieve desired output size."""
    if padding == 'same':
        # For 'same' padding: output_size = input_size
        return (kernel_size - 1) * dilation // 2
    elif padding == 'valid':
        return 0
    else:
        return int(padding)


class CNNLayer(nn.Module):
    """Single CNN layer with optional batch norm, activation, dropout, and residual connection."""
    
    def __init__(self, in_channels: int, config: CNNLayerConfig):
        super().__init__()
        self.config = config
        self.use_residual = config.residual and (in_channels == config.out_channels)
        
        padding = calculate_padding(config.kernel_size, config.dilation, config.padding)
        
        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=padding,
                dilation=config.dilation,
            )
        ]
        
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(config.out_channels))
        
        layers.append(get_activation(config.activation))
        
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        
        self.conv = nn.Sequential(*layers)
        
        # Residual projection if channels don't match
        if config.residual and in_channels != config.out_channels:
            self.residual_proj = nn.Conv1d(in_channels, config.out_channels, kernel_size=1)
            self.use_residual = True
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        out = self.conv(x)
        
        if self.use_residual:
            residual = x if self.residual_proj is None else self.residual_proj(x)
            # Handle potential sequence length mismatch from dilated convolutions
            if out.shape[-1] != residual.shape[-1]:
                # Pad or crop residual to match
                diff = residual.shape[-1] - out.shape[-1]
                if diff > 0:
                    residual = residual[:, :, diff // 2: diff // 2 + out.shape[-1]]
                else:
                    out = out[:, :, -diff // 2: -diff // 2 + residual.shape[-1]]
            out = out + residual
        
        return out


class MultiscaleCNN(nn.Module):
    """
    Parallel CNN branches with different kernel sizes (widens in feature space).
    
    Each branch processes the input with a different kernel size, and outputs
    are concatenated along the channel dimension.
    
    Args:
        in_channels: Input embedding dimension
        layer_configs: List of CNNLayerConfig, one per branch
        
    Example:
        # Create multiscale CNN with kernels 3, 5, 7, 11
        configs = [
            CNNLayerConfig(out_channels=64, kernel_size=3),
            CNNLayerConfig(out_channels=64, kernel_size=5),
            CNNLayerConfig(out_channels=64, kernel_size=7),
            CNNLayerConfig(out_channels=64, kernel_size=11, dilation=2),
        ]
        cnn = MultiscaleCNN(in_channels=768, layer_configs=configs)
    """
    
    def __init__(
        self,
        in_channels: int,
        layer_configs: List[CNNLayerConfig] = None,
        # Convenience parameters for simple config
        kernel_sizes: List[int] = None,
        out_channels: int = 64,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Use provided configs or create from convenience parameters
        if layer_configs is None:
            if kernel_sizes is None:
                kernel_sizes = [3, 5, 7, 11]
            layer_configs = [
                CNNLayerConfig(
                    out_channels=out_channels,
                    kernel_size=k,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
                for k in kernel_sizes
            ]
        
        self.branches = nn.ModuleList([
            CNNLayer(in_channels, config) for config in layer_configs
        ])
        
        # Calculate total output channels
        self.out_channels = sum(config.out_channels for config in layer_configs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
        Returns:
            Output tensor of shape (batch, seq_len, out_channels)
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each branch and concatenate
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        
        # Transpose back: (batch, seq_len, channels)
        return out.transpose(1, 2)


class DeepCNN(nn.Module):
    """
    Stacked CNN layers (deepens in context space).
    
    Sequential layers with optional residual connections and configurable dilation.
    Increasing dilation allows capturing longer-range context without increasing
    kernel size.
    
    Args:
        in_channels: Input embedding dimension
        layer_configs: List of CNNLayerConfig for each sequential layer
        residual_every_n: Add residual connection every N layers (0 = disabled)
        
    Example:
        # Create deep CNN with increasing dilation
        configs = [
            CNNLayerConfig(out_channels=128, kernel_size=3, dilation=1),
            CNNLayerConfig(out_channels=128, kernel_size=3, dilation=2, residual=True),
            CNNLayerConfig(out_channels=128, kernel_size=3, dilation=4, residual=True),
            CNNLayerConfig(out_channels=256, kernel_size=3, dilation=8),
        ]
        cnn = DeepCNN(in_channels=768, layer_configs=configs)
    """
    
    def __init__(
        self,
        in_channels: int,
        layer_configs: List[CNNLayerConfig] = None,
        # Convenience parameters for simple config
        num_layers: int = None,
        hidden_channels: int = 128,
        out_channels: int = 256,
        kernel_size: int = 3,
        dilations: List[int] = None,
        activation: str = 'relu',
        batch_norm: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        
        # Use provided configs or create from convenience parameters
        if layer_configs is None:
            if num_layers is None:
                num_layers = 4
            if dilations is None:
                dilations = [1, 2, 4, 8][:num_layers]  # Exponentially increasing
            
            # Pad dilations if needed
            while len(dilations) < num_layers:
                dilations.append(dilations[-1])
            
            layer_configs = []
            for i in range(num_layers):
                is_last = (i == num_layers - 1)
                layer_configs.append(CNNLayerConfig(
                    out_channels=out_channels if is_last else hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    residual=residual and i > 0,  # Residual from 2nd layer
                ))
        
        # Build sequential layers
        layers = []
        current_channels = in_channels
        for config in layer_configs:
            layers.append(CNNLayer(current_channels, config))
            current_channels = config.out_channels
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, channels)
        Returns:
            Output tensor of shape (batch, seq_len, out_channels)
        """
        # Transpose for Conv1d: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply sequential layers
        for layer in self.layers:
            x = layer(x)
        
        # Transpose back: (batch, seq_len, channels)
        return x.transpose(1, 2)


def create_cnn(
    cnn_type: str,
    in_channels: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CNN module.
    
    Args:
        cnn_type: 'multiscale' or 'deep'
        in_channels: Input embedding dimension
        **kwargs: Additional arguments passed to the CNN constructor
        
    Returns:
        CNN module
    """
    if cnn_type == 'multiscale':
        return MultiscaleCNN(in_channels=in_channels, **kwargs)
    elif cnn_type == 'deep':
        return DeepCNN(in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown CNN type: {cnn_type}. Use 'multiscale' or 'deep'")
