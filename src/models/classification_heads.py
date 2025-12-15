"""
Multi-Task Learning Classification Heads for Protein Secondary Structure Prediction.

Two strategies for MTL:
1. q3discarding: Independent Q8 and Q3 heads, Q3 discarded at inference
2. q3guided: Q3 computed first, then Q8 receives Q3 logits as a prior (easier→harder cascade)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal

# Import HeadConfig from central config to avoid duplication
from ..config import HeadConfig


class OutputProjection(nn.Module):
    """Simple output projection with optional hidden layer."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Q3DiscardingHead(nn.Module):
    """
    Independent Q8 and Q3 heads - Q3 discarded at inference.
    
    Both heads receive the same input features and are trained jointly.
    During inference, only Q8 predictions are used.
    
    Architecture:
        Input Features
            ├─→ Linear(hidden) → GELU → LN → Dropout → Linear(8) → Q8 logits
            └─→ Linear(hidden) → GELU → LN → Dropout → Linear(3) → Q3 logits [discarded]
    """
    
    def __init__(
        self,
        input_dim: int,
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.q8_head = OutputProjection(input_dim, fc_hidden, 8, fc_dropout)
        self.q3_head = OutputProjection(input_dim, fc_hidden, 3, fc_dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            return_q3: Whether to compute Q3 predictions (for training)
            
        Returns:
            Tuple of (q8_logits, q3_logits or None)
        """
        q8_logits = self.q8_head(x)
        q3_logits = self.q3_head(x) if return_q3 else None
        
        return q8_logits, q3_logits


class Q3GuidedHead(nn.Module):
    """
    Q3-Guided Q8 head: Q3 computed first as a prior for Q8.
    
    Since Q3 (3-class) is an easier task that converges faster, we use it
    to provide a strong prior signal to Q8. Q3 effectively "hints" to Q8
    which macro-class to focus on:
    
    - If Q3 says "Helix", Q8 focuses on distinguishing G, H, I
    - If Q3 says "Strand", Q8 focuses on distinguishing E, B
    - If Q3 says "Coil", Q8 focuses on distinguishing T, S, C
    
    Architecture:
        Input Features → Linear(hidden) → GELU → LN → Dropout → Linear(3) → Q3 logits
                                                                               │
        Input Features ─────────────────────────────────────┐                  │
                                                            ▼                  ▼
                                                        Concat([Features, Q3_logits])
                                                            │
                                                            ▼
                                        Linear(hidden) → GELU → LN → Dropout → Linear(8) → Q8 logits
    
    During training:
    - Q3 is computed first (easier task, converges faster)
    - Q3 logits are concatenated with features and fed to Q8
    - Q3 logits are detached to prevent gradient flow from Q8 → Q3
    - Both heads are trained with their respective losses
    
    During inference:
    - Q3 provides the prior, Q8 makes the final prediction
    - Can return both for consistency checking
    """
    
    def __init__(
        self,
        input_dim: int,
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Q3 head (computed first - easier task)
        self.q3_head = OutputProjection(input_dim, fc_hidden, 3, fc_dropout)
        
        # Q8 head receives features + Q3 logits as prior
        self.q8_head = OutputProjection(input_dim + 3, fc_hidden, 8, fc_dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            return_q3: Whether to return Q3 predictions (always computed internally)
            
        Returns:
            Tuple of (q8_logits, q3_logits or None)
        """
        # Step 1: Compute Q3 first (easier task, provides prior)
        q3_logits = self.q3_head(x)
        
        # Step 2: Use Q3 as prior for Q8
        # Detach Q3 logits to prevent gradient flow from Q8 loss back to Q3 head
        # Q3 should learn independently from its own loss, not be influenced by Q8
        q3_prior = q3_logits.detach()
        
        # Concatenate features with Q3 prior
        q8_input = torch.cat([x, q3_prior], dim=-1)
        
        # Compute Q8 with Q3 guidance
        q8_logits = self.q8_head(q8_input)
        
        return q8_logits, q3_logits if return_q3 else None


class MTLClassificationHead(nn.Module):
    """
    Multi-Task Learning classification head with strategy selection.
    
    Strategies:
    - 'q3discarding': Independent heads, Q3 discarded at inference
    - 'q3guided': Q3 computed first, provides prior signal to Q8
    
    Args:
        input_dim: Input feature dimension
        strategy: MTL strategy ('q3discarding' or 'q3guided')
        fc_hidden: Hidden dimension for output projections
        fc_dropout: Dropout probability
        
    Example:
        head = MTLClassificationHead(
            input_dim=512,
            strategy='q3guided',
            fc_hidden=256,
        )
        q8, q3 = head(features)
    """
    
    def __init__(
        self,
        input_dim: int,
        strategy: str = 'q3discarding',
        fc_hidden: int = 512,
        fc_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.strategy = strategy
        
        if strategy == 'q3discarding':
            self.head = Q3DiscardingHead(input_dim, fc_hidden, fc_dropout)
        elif strategy == 'q3guided':
            self.head = Q3GuidedHead(input_dim, fc_hidden, fc_dropout)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'q3discarding' or 'q3guided'")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_q3: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            return_q3: Whether to compute Q3 predictions
            
        Returns:
            Tuple of (q8_logits, q3_logits or None)
        """
        return self.head(x, return_q3=return_q3)
    
    def check_consistency(
        self,
        q8_logits: torch.Tensor,
        q3_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check consistency between Q8 and Q3 predictions.
        
        Returns a mask indicating where predictions are consistent.
        Useful for identifying low-confidence predictions.
        
        Args:
            q8_logits: Q8 predictions of shape (batch, seq_len, 8)
            q3_logits: Q3 predictions of shape (batch, seq_len, 3)
            
        Returns:
            Boolean tensor of shape (batch, seq_len) where True = consistent
        """
        # Q8 to Q3 mapping: G,H,I -> H(0), E,B -> E(1), T,S,C -> C(2)
        q8_to_q3 = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=q8_logits.device)
        
        q8_pred = q8_logits.argmax(dim=-1)  # (batch, seq_len)
        q3_pred = q3_logits.argmax(dim=-1)  # (batch, seq_len)
        
        # Map Q8 to Q3
        q8_as_q3 = q8_to_q3[q8_pred]
        
        # Check consistency
        consistent = (q8_as_q3 == q3_pred)
        
        return consistent


def create_classification_head(
    input_dim: int,
    config: Optional[HeadConfig] = None,
    **kwargs
) -> MTLClassificationHead:
    """
    Factory function to create classification head.
    
    Args:
        input_dim: Input feature dimension
        config: Optional HeadConfig object
        **kwargs: Override config parameters
        
    Returns:
        MTLClassificationHead module
    """
    if config is None:
        config = HeadConfig()
    
    return MTLClassificationHead(
        input_dim=input_dim,
        strategy=kwargs.get('strategy', config.strategy),
        fc_hidden=kwargs.get('fc_hidden', config.fc_hidden),
        fc_dropout=kwargs.get('fc_dropout', config.fc_dropout),
    )
