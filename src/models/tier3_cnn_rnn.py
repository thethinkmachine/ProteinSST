"""
Tier 3: CNN + RNN Model - PLM Embeddings → CNN → RNN → Classification Head

Supports two modes:
- frozen_plm=True (default): Uses pre-extracted PLM embeddings from HDF5
- frozen_plm=False (FFT): Includes PLM backbone, trained end-to-end

Uses CNNs to extract local motifs, then RNNs for sequential modeling.
Supports MultiscaleCNN or DeepCNN, and LSTM/GRU/RNN variants.

Architecture:
    [PLM Backbone (optional)] → PLM Embeddings (L, D_plm) → CNN Block → BiLSTM/BiGRU/BiRNN → MTL Head → Q8/Q3
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .cnn_blocks import MultiscaleCNN, DeepCNN, CNNLayerConfig
from .rnn_blocks import ConfigurableRNN, RNNConfig
from .classification_heads import MTLClassificationHead, HeadConfig


class Tier3CNNRNN(nn.Module):
    """
    Tier 3: CNN + RNN model using PLM embeddings.
    
    Combines local motif extraction (CNN) with sequential modeling (RNN).
    Most expressive architecture with the highest capacity.
    Supports both frozen (pre-extracted) and fine-tuned PLM modes.
    
    Args:
        embedding_dim: PLM embedding dimension
        skip_cnn: If True, bypass CNN entirely and pass PLM embeddings directly to RNN
        # CNN params (ignored if skip_cnn=True)
        cnn_type: 'multiscale' or 'deep'
        cnn_configs: List of CNNLayerConfig for CNN layers (optional)
        kernel_sizes: List of kernel sizes for multiscale branches
        cnn_out_channels: Output channels per branch (multiscale) or hidden (deep)
        cnn_num_layers: Number of layers for deep CNN
        cnn_dilations: List of dilation values for deep CNN
        cnn_activation: Activation function
        cnn_dropout: Dropout probability
        cnn_residual: Use residual connections in CNN
        # RNN params
        rnn_type: 'lstm', 'gru', or 'rnn'
        rnn_hidden: Hidden size for RNN
        rnn_layers: Number of RNN layers
        rnn_dropout: Dropout between RNN layers
        rnn_bidirectional: Use bidirectional RNN
        # Head params
        head_strategy: MTL head strategy
        head_hidden: Hidden dimension for classification head
        head_dropout: Dropout for classification head
        # FFT Mode
        frozen_plm: If True, expects pre-extracted embeddings. If False, includes PLM backbone.
        plm_name: Name of PLM to use for FFT mode
        gradient_checkpointing: Enable gradient checkpointing for PLM
        
    Example:
        # Frozen mode - With CNN
        model = Tier3CNNRNN(
            embedding_dim=768,
            cnn_type='multiscale',
            kernel_sizes=[3, 5, 7],
            rnn_type='lstm',
            rnn_hidden=256,
        )
        
        # FFT mode - Without CNN (PLM -> RNN directly)
        model = Tier3CNNRNN(
            frozen_plm=False,
            plm_name='protbert',
            skip_cnn=True,
            rnn_type='lstm',
            rnn_hidden=256,
        )
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        skip_cnn: bool = False,
        # CNN params
        cnn_type: str = 'multiscale',
        cnn_configs: List[CNNLayerConfig] = None,
        kernel_sizes: List[int] = None,
        cnn_out_channels: int = 64,
        cnn_num_layers: int = 4,
        cnn_dilations: List[int] = None,
        cnn_activation: str = 'relu',
        cnn_batch_norm: bool = True,
        cnn_dropout: float = 0.0,
        cnn_residual: bool = True,
        # RNN params
        rnn_type: str = 'lstm',
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = True,
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
        self.skip_cnn = skip_cnn
        self.cnn_type = cnn_type if not skip_cnn else None
        self.rnn_type = rnn_type
        
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
        
        # Build CNN (only if not skipping)
        if skip_cnn:
            self.cnn = None
            rnn_input_dim = self.embedding_dim
        elif cnn_type == 'multiscale':
            self.cnn = MultiscaleCNN(
                in_channels=self.embedding_dim,
                layer_configs=cnn_configs,
                kernel_sizes=kernel_sizes or [3, 5, 7],
                out_channels=cnn_out_channels,
                activation=cnn_activation,
                batch_norm=cnn_batch_norm,
                dropout=cnn_dropout,
            )
            rnn_input_dim = self.cnn.out_channels
        elif cnn_type == 'deep':
            self.cnn = DeepCNN(
                in_channels=self.embedding_dim,
                layer_configs=cnn_configs,
                num_layers=cnn_num_layers,
                hidden_channels=cnn_out_channels,
                out_channels=cnn_out_channels * 2,
                dilations=cnn_dilations or [1, 2, 4, 8],
                activation=cnn_activation,
                batch_norm=cnn_batch_norm,
                dropout=cnn_dropout,
                residual=cnn_residual,
            )
            rnn_input_dim = self.cnn.out_channels
        else:
            raise ValueError(f"Unknown cnn_type: {cnn_type}")
        
        # Build RNN
        self.rnn = ConfigurableRNN(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            rnn_type=rnn_type,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )
        
        rnn_output_dim = self.rnn.out_channels
        
        # Classification head
        self.head = MTLClassificationHead(
            input_dim=rnn_output_dim,
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
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for pname, param in module.named_parameters():
                    if 'weight_ih' in pname:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in pname:
                        nn.init.orthogonal_(param)
                    elif 'bias' in pname:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        features: torch.Tensor = None,
        sequences: List[str] = None,
        lengths: Optional[torch.Tensor] = None,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: PLM embeddings of shape (batch, seq_len, embedding_dim) - for frozen mode
            sequences: List of protein sequences (strings) - for FFT mode
            lengths: Optional sequence lengths (not used currently)
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
        
        # CNN: (batch, seq_len, channels) -> (batch, seq_len, cnn_out)
        # Skip CNN if configured
        x = x if self.skip_cnn else self.cnn(x)
        
        # RNN: (batch, seq_len, input_dim) -> (batch, seq_len, rnn_out)
        x = self.rnn(x, lengths)
        
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
