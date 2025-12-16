"""
Tier 2: CNN Model - PLM Embeddings → CNN → Classification Head

Supports two modes:
- frozen_plm=True (default): Uses pre-extracted PLM embeddings from HDF5
- frozen_plm=False (FFT): Includes PLM backbone, trained end-to-end

Uses CNNs to extract local motifs from PLM embeddings.
Supports MultiscaleCNN (wider) or DeepCNN (deeper) architectures.

Architecture:
    [PLM Backbone (optional)] → PLM Embeddings (L, D_plm) → CNN Block → MTL Head → Q8/Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .cnn_blocks import MultiscaleCNN, DeepCNN, CNNLayerConfig, create_cnn
from .classification_heads import MTLClassificationHead, HeadConfig


class Tier2CNN(nn.Module):
    """
    Tier 2: CNN model using PLM embeddings.
    
    Extracts local motifs from PLM embeddings using multiscale or deep CNNs.
    Supports both frozen (pre-extracted) and fine-tuned PLM modes.
    
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
        # FFT Mode
        frozen_plm: If True, expects pre-extracted embeddings. If False, includes PLM backbone.
        plm_name: Name of PLM to use for FFT mode
        gradient_checkpointing: Enable gradient checkpointing for PLM
        
    Example:
        # Frozen mode - Multiscale CNN
        model = Tier2CNN(
            embedding_dim=768,
            cnn_type='multiscale',
            kernel_sizes=[3, 5, 7, 11],
        )
        
        # FFT mode - Deep CNN
        model = Tier2CNN(
            frozen_plm=False,
            plm_name='protbert',
            cnn_type='deep',
            cnn_num_layers=4,
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
        # FFT Mode
        frozen_plm: bool = True,
        plm_name: str = 'protbert',
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.frozen_plm = frozen_plm
        self.plm_name = plm_name
        self.cnn_type = cnn_type
        
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
        
        # Build CNN
        if cnn_type == 'multiscale':
            self.cnn = MultiscaleCNN(
                in_channels=self.embedding_dim,
                layer_configs=cnn_configs,
                kernel_sizes=kernel_sizes or [3, 5, 7, 11],
                out_channels=cnn_out_channels,
                activation=cnn_activation,
                batch_norm=cnn_batch_norm,
                dropout=cnn_dropout,
            )
        elif cnn_type == 'deep':
            self.cnn = DeepCNN(
                in_channels=self.embedding_dim,
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
        """Initialize weights (excluding PLM backbone)."""
        for name, module in self.named_modules():
            if 'plm_backbone' in name:
                continue  # Skip PLM backbone
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
        
        # CNN expects (batch, seq_len, channels), outputs same shape
        x = self.cnn(x)
        
        # Classification
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
