"""
Data augmentation for protein sequences.
Implements leveled augmentation (1-5) for controlled experiments.
"""

import random
import torch
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from .config import SIMILAR_AA, MASK_TOKEN, AMINO_ACIDS


# =============================================================================
# Augmentation Configuration
# =============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for a specific augmentation level."""
    
    # Masking
    mask_prob: float = 0.0
    
    # Similar amino acid substitution
    substitute_prob: float = 0.0
    
    # Noise injection (for embeddings)
    noise_std: float = 0.0
    
    # Sequence operations
    crop_prob: float = 0.0
    min_crop_ratio: float = 0.8  # Minimum length ratio after crop
    
    reverse_prob: float = 0.0


# Level configurations
AUGMENTATION_LEVELS = {
    1: AugmentationConfig(),  # No augmentation (baseline)
    
    2: AugmentationConfig(
        mask_prob=0.05,
    ),
    
    3: AugmentationConfig(
        mask_prob=0.10,
        substitute_prob=0.03,
        noise_std=0.01,
    ),
    
    4: AugmentationConfig(
        mask_prob=0.15,
        substitute_prob=0.05,
        noise_std=0.02,
        crop_prob=0.2,
        min_crop_ratio=0.7,
    ),
    
    5: AugmentationConfig(
        mask_prob=0.20,
        substitute_prob=0.08,
        noise_std=0.03,
        crop_prob=0.3,
        min_crop_ratio=0.6,
        reverse_prob=0.1,
    ),
}


# =============================================================================
# Augmentation Functions
# =============================================================================

def mask_residues(
    sequence: str,
    labels_sst8: str,
    labels_sst3: str,
    mask_prob: float,
    mask_token: str = 'X',  # Use 'X' as mask for one-hot (will be zero vector)
) -> Tuple[str, str, str]:
    """
    Randomly mask residues in the sequence.
    Labels at masked positions are preserved (model should still predict them).
    """
    if mask_prob <= 0:
        return sequence, labels_sst8, labels_sst3
    
    seq_list = list(sequence)
    
    for i in range(len(seq_list)):
        if random.random() < mask_prob:
            seq_list[i] = mask_token
    
    return ''.join(seq_list), labels_sst8, labels_sst3


def substitute_similar_aa(
    sequence: str,
    labels_sst8: str,
    labels_sst3: str,
    substitute_prob: float,
) -> Tuple[str, str, str]:
    """
    Replace amino acids with chemically similar ones.
    This is a conservative augmentation that shouldn't change structure much.
    """
    if substitute_prob <= 0:
        return sequence, labels_sst8, labels_sst3
    
    seq_list = list(sequence)
    
    for i in range(len(seq_list)):
        aa = seq_list[i]
        if aa in SIMILAR_AA and random.random() < substitute_prob:
            similar = SIMILAR_AA[aa]
            if similar:
                seq_list[i] = random.choice(similar)
    
    return ''.join(seq_list), labels_sst8, labels_sst3


def random_crop(
    sequence: str,
    labels_sst8: str,
    labels_sst3: str,
    crop_prob: float,
    min_ratio: float = 0.7,
) -> Tuple[str, str, str]:
    """
    Randomly crop a contiguous region of the sequence.
    Labels are cropped correspondingly.
    """
    if crop_prob <= 0 or random.random() > crop_prob:
        return sequence, labels_sst8, labels_sst3
    
    length = len(sequence)
    min_len = max(10, int(length * min_ratio))  # At least 10 residues
    
    new_len = random.randint(min_len, length)
    start = random.randint(0, length - new_len)
    
    return (
        sequence[start:start + new_len],
        labels_sst8[start:start + new_len],
        labels_sst3[start:start + new_len],
    )


def reverse_sequence(
    sequence: str,
    labels_sst8: str,
    labels_sst3: str,
    reverse_prob: float,
) -> Tuple[str, str, str]:
    """
    Reverse the entire sequence (experimental).
    Note: This changes structural context significantly.
    """
    if reverse_prob <= 0 or random.random() > reverse_prob:
        return sequence, labels_sst8, labels_sst3
    
    return (
        sequence[::-1],
        labels_sst8[::-1],
        labels_sst3[::-1],
    )


# =============================================================================
# Augmentation Pipeline
# =============================================================================

class SequenceAugmenter:
    """
    Applies sequence augmentations based on configured level.
    
    Args:
        level: Augmentation level (1-5)
        seed: Random seed for reproducibility (optional)
    """
    
    def __init__(self, level: int = 2, seed: Optional[int] = None):
        if level not in AUGMENTATION_LEVELS:
            raise ValueError(f"Level must be 1-5, got {level}")
        
        self.level = level
        self.config = AUGMENTATION_LEVELS[level]
        
        if seed is not None:
            random.seed(seed)
    
    def __call__(
        self,
        sequence: str,
        labels_sst8: str,
        labels_sst3: str,
    ) -> Tuple[str, str, str]:
        """Apply augmentations to sequence and labels."""
        
        # Level 1: No augmentation
        if self.level == 1:
            return sequence, labels_sst8, labels_sst3
        
        # Apply augmentations in order
        
        # 1. Cropping (must be first to reduce compute for subsequent ops)
        sequence, labels_sst8, labels_sst3 = random_crop(
            sequence, labels_sst8, labels_sst3,
            self.config.crop_prob,
            self.config.min_crop_ratio,
        )
        
        # 2. Masking
        sequence, labels_sst8, labels_sst3 = mask_residues(
            sequence, labels_sst8, labels_sst3,
            self.config.mask_prob,
        )
        
        # 3. Substitution
        sequence, labels_sst8, labels_sst3 = substitute_similar_aa(
            sequence, labels_sst8, labels_sst3,
            self.config.substitute_prob,
        )
        
        # 4. Reversal (experimental, only at level 5)
        sequence, labels_sst8, labels_sst3 = reverse_sequence(
            sequence, labels_sst8, labels_sst3,
            self.config.reverse_prob,
        )
        
        return sequence, labels_sst8, labels_sst3
    
    def __repr__(self):
        return f"SequenceAugmenter(level={self.level}, config={self.config})"


class EmbeddingAugmenter:
    """
    Applies augmentations to pre-computed embeddings.
    
    Args:
        level: Augmentation level (1-5)
    """
    
    def __init__(self, level: int = 2):
        if level not in AUGMENTATION_LEVELS:
            raise ValueError(f"Level must be 1-5, got {level}")
        
        self.level = level
        self.config = AUGMENTATION_LEVELS[level]
    
    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to embedding tensor.
        
        Args:
            embeddings: Tensor of shape (seq_len, embed_dim)
        
        Returns:
            Augmented tensor
        """
        if self.level == 1 or self.config.noise_std <= 0:
            return embeddings
        
        # Add Gaussian noise
        noise = torch.randn_like(embeddings) * self.config.noise_std
        return embeddings + noise
    
    def mask_embeddings(
        self,
        embeddings: torch.Tensor,
        mask_prob: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask random positions in embeddings.
        
        Returns:
            Tuple of (masked_embeddings, mask_indices)
        """
        mask_prob = mask_prob or self.config.mask_prob
        
        if mask_prob <= 0:
            return embeddings, torch.zeros(embeddings.shape[0], dtype=torch.bool)
        
        mask = torch.rand(embeddings.shape[0]) < mask_prob
        masked_embeddings = embeddings.clone()
        masked_embeddings[mask] = 0.0  # Zero out masked positions
        
        return masked_embeddings, mask


# =============================================================================
# Factory Functions
# =============================================================================

def get_augmenter(
    level: int = 2,
    for_embeddings: bool = False,
    seed: Optional[int] = None,
) -> Callable:
    """
    Create an augmenter based on level.
    
    Args:
        level: Augmentation level (1-5)
        for_embeddings: If True, return EmbeddingAugmenter
        seed: Random seed
    
    Returns:
        Augmenter callable
    """
    if for_embeddings:
        return EmbeddingAugmenter(level)
    return SequenceAugmenter(level, seed)


def describe_augmentation_levels():
    """Print description of all augmentation levels."""
    descriptions = {
        1: "No augmentation (baseline)",
        2: "Light masking (5%)",
        3: "Moderate: masking (10%) + substitution (3%) + noise",
        4: "Aggressive: masking (15%) + substitution (5%) + cropping (20%)",
        5: "Experimental: heavy masking (20%) + all augmentations + reversal (10%)",
    }
    
    print("=" * 60)
    print("AUGMENTATION LEVELS")
    print("=" * 60)
    for level, desc in descriptions.items():
        config = AUGMENTATION_LEVELS[level]
        print(f"\nLevel {level}: {desc}")
        print(f"  - Mask prob: {config.mask_prob}")
        print(f"  - Substitute prob: {config.substitute_prob}")
        print(f"  - Crop prob: {config.crop_prob}")
        print(f"  - Noise std: {config.noise_std}")
        print(f"  - Reverse prob: {config.reverse_prob}")
