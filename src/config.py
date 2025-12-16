"""
Configuration module for ProteinSST training pipeline.
Contains all hyperparameters, class mappings, and shared constants.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# SST Class Definitions
# =============================================================================

# SST8 (8-class) labels as defined by DSSP
SST8_CLASSES = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C']
SST8_NAMES = {
    'G': '3₁₀-helix',
    'H': 'α-helix',
    'I': 'π-helix',
    'E': 'β-strand',
    'B': 'β-bridge',
    'T': 'Turn',
    'S': 'Bend',
    'C': 'Coil'
}

# SST3 (3-class) labels
SST3_CLASSES = ['H', 'E', 'C']
SST3_NAMES = {
    'H': 'Helix',
    'E': 'Strand', 
    'C': 'Coil'
}

# Mapping from SST8 to SST3
SST8_TO_SST3 = {
    'G': 'H', 'H': 'H', 'I': 'H',  # Helix
    'E': 'E', 'B': 'E',             # Strand
    'T': 'C', 'S': 'C', 'C': 'C'    # Coil
}

# Character to index mappings
SST8_TO_IDX = {c: i for i, c in enumerate(SST8_CLASSES)}
SST3_TO_IDX = {c: i for i, c in enumerate(SST3_CLASSES)}
IDX_TO_SST8 = {i: c for c, i in SST8_TO_IDX.items()}
IDX_TO_SST3 = {i: c for c, i in SST3_TO_IDX.items()}


# =============================================================================
# Amino Acid Definitions
# =============================================================================

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# Special tokens
PAD_TOKEN = '<PAD>'
MASK_TOKEN = '<MASK>'
UNK_TOKEN = '<UNK>'

# Extended vocabulary with special tokens
VOCAB = [PAD_TOKEN, MASK_TOKEN, UNK_TOKEN] + AMINO_ACIDS
VOCAB_TO_IDX = {tok: i for i, tok in enumerate(VOCAB)}
IDX_TO_VOCAB = {i: tok for tok, i in VOCAB_TO_IDX.items()}

# BLOSUM62 substitution matrix (simplified - chemically similar groups)
SIMILAR_AA = {
    'A': ['G', 'S'],
    'C': ['S'],
    'D': ['E', 'N'],
    'E': ['D', 'Q'],
    'F': ['Y', 'W'],
    'G': ['A', 'S'],
    'H': ['N', 'Q'],
    'I': ['L', 'V', 'M'],
    'K': ['R', 'Q'],
    'L': ['I', 'V', 'M'],
    'M': ['L', 'I', 'V'],
    'N': ['D', 'Q', 'H'],
    'P': ['A'],
    'Q': ['E', 'N', 'K'],
    'R': ['K', 'Q'],
    'S': ['T', 'A', 'G'],
    'T': ['S', 'A'],
    'V': ['I', 'L', 'M'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W'],
}


# =============================================================================
# Data Paths
# =============================================================================

TRAIN_CSV_PATH = 'data/train.csv'
CB513_PATH = 'data/cb513.csv'
EMBEDDINGS_DIR = 'data/embeddings'


# =============================================================================
# Class Weights (from EDA - inverse frequency)
# =============================================================================

# SST8 class weights based on inverse frequency from training data
# Order: G, H, I, E, B, T, S, C
SST8_WEIGHTS = torch.tensor([
    3.5,   # G (3₁₀-helix) - rare (~4%)
    0.35,  # H (α-helix) - very common (~30%)
    15.0,  # I (π-helix) - very rare (<1%)
    0.6,   # E (β-strand) - common (~20%)
    8.0,   # B (β-bridge) - rare (~2%)
    1.2,   # T (Turn) - moderate (~8%)
    1.0,   # S (Bend) - moderate (~10%)
    0.3,   # C (Coil) - most common (~25%)
], dtype=torch.float32)

# SST3 class weights
# Order: H, E, C
SST3_WEIGHTS = torch.tensor([
    0.8,  # H (Helix) - ~35%
    1.2,  # E (Strand) - ~20%
    0.6,  # C (Coil) - ~45%
], dtype=torch.float32)


# =============================================================================
# Base Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    
    # Data
    max_seq_length: int = 512
    train_split: float = 0.9
    batch_size: int = 32
    num_workers: int = 4
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_epochs: int = 50
    gradient_clip: float = 1.0
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'none'
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Loss
    focal_gamma: float = 2.0
    loss_type: str = 'focal'  # 'focal', 'weighted_ce', 'label_smoothing', 'ce', 'crf'
    label_smoothing: float = 0.0
    q8_loss_weight: float = 1.0
    q3_loss_weight: float = 0.5
    
    # Augmentation (1-5)
    augmentation_level: int = 1
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 5
    
    # Logging & Tracking
    log_every: int = 100
    use_tracking: bool = False
    trackio_space_id: Optional[str] = None
    experiment_name: str = 'protein_sst'
    
    # HuggingFace Hub
    hub_model_id: Optional[str] = None
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# Component Configurations
# =============================================================================

@dataclass
class PLMConfig:
    """PLM embedding configuration."""
    plm_name: str = 'esm2_35m'  # 'esm2_8m', 'esm2_35m', 'esm2_650m', 'protbert'
    embeddings_path: str = 'data/embeddings'  # Path to HDF5 file or directory


@dataclass
class CNNConfig:
    """CNN architecture configuration."""
    cnn_type: str = 'multiscale'  # 'multiscale' or 'deep'
    
    # Multiscale params
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 11])
    out_channels: int = 64
    
    # Deep params
    num_layers: int = 4
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    hidden_channels: int = 128
    
    # Common params
    activation: str = 'relu'
    batch_norm: bool = True
    dropout: float = 0.0
    residual: bool = True


@dataclass
class RNNConfig:
    """RNN architecture configuration."""
    rnn_type: str = 'lstm'  # 'lstm', 'gru', 'rnn'
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    def __post_init__(self):
        valid_types = ['lstm', 'gru', 'rnn']
        if self.rnn_type not in valid_types:
            raise ValueError(f"rnn_type must be one of {valid_types}")


@dataclass
class HeadConfig:
    """Classification head configuration."""
    strategy: str = 'q3discarding'  # 'q3discarding' or 'q3guided'
    fc_hidden: int = 256
    fc_dropout: float = 0.1
    
    def __post_init__(self):
        valid_strategies = ['q3discarding', 'q3guided']
        if self.strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")


# =============================================================================
# Tier-Specific Configurations
# =============================================================================

@dataclass
class Tier1Config(TrainingConfig):
    """
    Tier 1: Baseline configuration.
    PLM Embeddings → FC → MTL Head
    
    Supports two modes via frozen_plm flag:
    - frozen_plm=True (default): Uses pre-extracted PLM embeddings
    - frozen_plm=False (FFT): Full Fine-Tuning with PLM backbone
    """
    model_name: str = 'tier1_baseline'
    
    # PLM Mode
    frozen_plm: bool = True  # True = use pre-extracted embeddings, False = FFT
    plm_name: str = 'esm2_35m'
    embeddings_path: str = 'data/embeddings/esm2_35m.h5'  # Only used when frozen_plm=True
    gradient_checkpointing: bool = False  # Only used when frozen_plm=False
    
    # FC layer
    fc_hidden: int = 512
    fc_dropout: float = 0.1
    
    # Head
    head_strategy: str = 'q3discarding'
    head_hidden: int = 256
    head_dropout: float = 0.1


@dataclass
class Tier2Config(TrainingConfig):
    """
    Tier 2: CNN configuration.
    PLM Embeddings → CNN → MTL Head
    
    Supports two modes via frozen_plm flag:
    - frozen_plm=True (default): Uses pre-extracted PLM embeddings
    - frozen_plm=False (FFT): Full Fine-Tuning with PLM backbone
    """
    model_name: str = 'tier2_cnn'
    
    # PLM Mode
    frozen_plm: bool = True  # True = use pre-extracted embeddings, False = FFT
    plm_name: str = 'esm2_35m'
    embeddings_path: str = 'data/embeddings/esm2_35m.h5'  # Only used when frozen_plm=True
    gradient_checkpointing: bool = False  # Only used when frozen_plm=False
    
    # CNN
    cnn_type: str = 'multiscale'  # 'multiscale' or 'deep'
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 11])
    cnn_out_channels: int = 64
    cnn_num_layers: int = 4
    cnn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cnn_activation: str = 'relu'
    cnn_dropout: float = 0.0
    cnn_residual: bool = True
    
    # Head
    head_strategy: str = 'q3discarding'
    head_hidden: int = 256
    head_dropout: float = 0.1


@dataclass
class Tier3Config(TrainingConfig):
    """
    Tier 3: CNN + RNN configuration.
    PLM Embeddings → CNN → BiLSTM/GRU/RNN → MTL Head
    
    Supports two modes via frozen_plm flag:
    - frozen_plm=True (default): Uses pre-extracted PLM embeddings
    - frozen_plm=False (FFT): Full Fine-Tuning with PLM backbone
    """
    model_name: str = 'tier3_cnn_rnn'
    
    # PLM Mode
    frozen_plm: bool = True  # True = use pre-extracted embeddings, False = FFT
    plm_name: str = 'esm2_35m'
    embeddings_path: str = 'data/embeddings/esm2_35m.h5'  # Only used when frozen_plm=True
    gradient_checkpointing: bool = False  # Only used when frozen_plm=False
    
    # CNN
    skip_cnn: bool = False  # Skip CNN and pass embeddings directly to RNN
    cnn_type: str = 'multiscale'  # 'multiscale' or 'deep'
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_out_channels: int = 64
    cnn_num_layers: int = 4
    cnn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cnn_activation: str = 'relu'
    cnn_dropout: float = 0.0
    cnn_residual: bool = True
    
    # RNN
    rnn_type: str = 'lstm'  # 'lstm', 'gru', 'rnn'
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_bidirectional: bool = True
    
    # Head
    head_strategy: str = 'q3discarding'
    head_hidden: int = 256
    head_dropout: float = 0.1


# =============================================================================
# PLM Embedding Dimensions (for convenience)
# =============================================================================

PLM_EMBEDDING_DIMS = {
    'esm2_8m': 320,
    'esm2_35m': 480,
    'esm2_650m': 1280,
    'protbert': 1024,
}


def get_embedding_dim(plm_name: str) -> int:
    """Get embedding dimension for a PLM."""
    if plm_name not in PLM_EMBEDDING_DIMS:
        raise ValueError(f"Unknown PLM: {plm_name}")
    return PLM_EMBEDDING_DIMS[plm_name]


# =============================================================================
# Leakage IDs to exclude
# =============================================================================

# From EDA: high-similarity pairs to exclude from training
LEAKAGE_TRAIN_IDS = [6552]  # Train IDs that are similar to test
LEAKAGE_TEST_IDS = [371]    # Corresponding test IDs
