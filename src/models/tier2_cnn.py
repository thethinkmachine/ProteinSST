"""
Tier 2: CNN Model - Frozen PLM Embeddings → CNN → Classification Head

Uses CNNs to extract local motifs from PLM embeddings.
Supports MultiscaleCNN (wider) or DeepCNN (deeper) architectures.

Architecture:
    PLM Embeddings (L, D_plm) → CNN Block → MTL Head → Q8/Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .cnn_blocks import MultiscaleCNN, DeepCNN, CNNLayerConfig, create_cnn
from .classification_heads import MTLClassificationHead, HeadConfig


class Tier2CNN(nn.Module):
    """
    Tier 2: CNN model using frozen PLM embeddings.
    
    Extracts local motifs from PLM embeddings using multiscale or deep CNNs.
    
    Args:
        embedding_dim: PLM embedding dimension
        cnn_type: 'multiscale' or 'deep'
        cnn_configs: List of CNNLayerConfig for CNN layers (optional)
        # Convenience params for MultiscaleCNN
        kernel_sizes: List of kernel sizes for multiscale branches
        cnn_out_channels: Output channels per branch (multiscale) or hidden (deep)
        # Convenience params for DeepCNN
        cnn_num_layers: Number of layers for deep CNN
        cnn_dilations: List of dilation values for deep CNN
        # Common CNN params
        cnn_activation: Activation function
        cnn_dropout: Dropout probability
        # Head params
        head_strategy: MTL head strategy
        head_hidden: Hidden dimension for classification head
        head_dropout: Dropout for classification head
        
    Example:
        # Multiscale CNN
        model = Tier2CNN(
            embedding_dim=768,
            cnn_type='multiscale',
            kernel_sizes=[3, 5, 7, 11],
        )
        
        # Deep CNN
        model = Tier2CNN(
            embedding_dim=768,
            cnn_type='deep',
            cnn_num_layers=4,
            cnn_dilations=[1, 2, 4, 8],
        )
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        cnn_type: str = 'multiscale',
        cnn_configs: List[CNNLayerConfig] = None,
        # Multiscale convenience params
        kernel_sizes: List[int] = None,
        cnn_out_channels: int = 64,
        # Deep convenience params
        cnn_num_layers: int = 4,
        cnn_dilations: List[int] = None,
        # Common CNN params
        cnn_activation: str = 'relu',
        cnn_batch_norm: bool = True,
        cnn_dropout: float = 0.0,
        cnn_residual: bool = True,
        # Head params
        head_strategy: str = 'q3discarding',
        head_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.cnn_type = cnn_type
        
        # Build CNN
        if cnn_type == 'multiscale':
            self.cnn = MultiscaleCNN(
                in_channels=embedding_dim,
                layer_configs=cnn_configs,
                kernel_sizes=kernel_sizes or [3, 5, 7, 11],
                out_channels=cnn_out_channels,
                activation=cnn_activation,
                batch_norm=cnn_batch_norm,
                dropout=cnn_dropout,
            )
        elif cnn_type == 'deep':
            self.cnn = DeepCNN(
                in_channels=embedding_dim,
                layer_configs=cnn_configs,
                num_layers=cnn_num_layers,
                hidden_channels=cnn_out_channels,
                out_channels=cnn_out_channels * 2,  # Final layer is wider
                dilations=cnn_dilations or [1, 2, 4, 8],
                activation=cnn_activation,
                batch_norm=cnn_batch_norm,
                dropout=cnn_dropout,
                residual=cnn_residual,
            )
        else:
            raise ValueError(f"Unknown cnn_type: {cnn_type}")
        
        cnn_output_dim = self.cnn.out_channels
        
        # Classification head
        self.head = MTLClassificationHead(
            input_dim=cnn_output_dim,
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
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
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
        # CNN expects (batch, seq_len, channels), outputs same shape
        x = self.cnn(features)
        
        # Classification
        q8_logits, q3_logits = self.head(x, return_q3=return_q3)
        
        return q8_logits, q3_logits
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
