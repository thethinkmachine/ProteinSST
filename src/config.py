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
# Training Configuration
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
    label_smoothing: float = 0.0
    q8_loss_weight: float = 1.0
    q3_loss_weight: float = 0.5
    
    # Augmentation (1-5)
    augmentation_level: int = 2
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 5
    
    # Logging & Tracking
    log_every: int = 100
    use_tracking: bool = False  # Enable Trackio/W&B experiment tracking
    experiment_name: str = 'protein_sst'
    
    # HuggingFace Hub
    hub_model_id: Optional[str] = None  # e.g., "username/protein-sst-tier1"
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    seed: int = 42


@dataclass
class Tier0Config(TrainingConfig):
    """Tier 0: CNN + BRNN (vanilla RNN) configuration."""
    model_name: str = 'tier0_cnn_brnn'
    
    # Model architecture
    input_dim: int = 20  # One-hot
    use_blosum: bool = True  # Add BLOSUM62 features
    
    # CNN
    cnn_filters: int = 64
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    # BiRNN (vanilla RNN)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_nonlinearity: str = 'tanh'  # 'tanh' or 'relu'
    
    # Output
    fc_hidden: int = 256
    fc_dropout: float = 0.2


@dataclass
class Tier1Config(TrainingConfig):
    """Tier 1: CNN + BiLSTM configuration."""
    model_name: str = 'tier1_cnn_bilstm'
    
    # Model architecture
    input_dim: int = 20  # One-hot
    use_blosum: bool = True  # Add BLOSUM62 features
    
    # CNN
    cnn_filters: int = 64
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    # BiLSTM
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # Output
    fc_hidden: int = 256
    fc_dropout: float = 0.2


@dataclass
class Tier2Config(TrainingConfig):
    """Tier 2: CNN + BiLSTM + Attention configuration."""
    model_name: str = 'tier2_cnn_bilstm_attention'
    
    # Inherits CNN and LSTM from Tier1
    input_dim: int = 20
    use_blosum: bool = True
    use_positional: bool = True
    
    cnn_filters: int = 64
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    
    # Attention
    num_heads: int = 8
    attention_dropout: float = 0.1
    
    fc_hidden: int = 256
    fc_dropout: float = 0.2


@dataclass 
class Tier3Config(TrainingConfig):
    """Tier 3: PLM (ESM-2) + BiLSTM configuration."""
    model_name: str = 'tier3_plm_bilstm'
    
    # PLM embeddings (pre-computed)
    embedding_dim: int = 1280  # ESM-2 650M
    embeddings_path: str = 'data/embeddings'
    
    # Optional CNN
    use_cnn: bool = True
    cnn_filters: int = 128
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5])
    
    # BiLSTM
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    
    fc_hidden: int = 256
    fc_dropout: float = 0.2


@dataclass
class Tier4Config(TrainingConfig):
    """Tier 4: TransConv (Transformer + CNN) configuration."""
    model_name: str = 'tier4_transconv'
    
    # PLM embeddings
    embedding_dim: int = 1280
    embeddings_path: str = 'data/embeddings'
    
    # Transformer
    num_transformer_layers: int = 4
    num_heads: int = 8
    transformer_dim: int = 512
    transformer_dropout: float = 0.1
    
    # Dilated CNN
    cnn_filters: int = 256
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    fc_hidden: int = 256
    fc_dropout: float = 0.2


@dataclass
class Tier5Config(TrainingConfig):
    """Tier 5: Fine-tuned ESM-2 configuration."""
    model_name: str = 'tier5_esm2_finetune'
    
    # ESM-2 model
    esm_model: str = 'facebook/esm2_t33_650M_UR50D'  # or t12_35M for smaller
    freeze_layers: int = 0  # Number of layers to freeze (0 = full fine-tune)
    
    # Task heads
    fc_hidden: int = 512
    fc_dropout: float = 0.1
    
    # Fine-tuning specific
    learning_rate: float = 1e-5  # Lower for fine-tuning
    gradient_checkpointing: bool = True  # Save memory
    batch_size: int = 8  # Smaller batch for memory


# =============================================================================
# Leakage IDs to exclude
# =============================================================================

# From EDA: high-similarity pairs to exclude from training
LEAKAGE_TRAIN_IDS = [6552]  # Train IDs that are similar to test
LEAKAGE_TEST_IDS = [371]    # Corresponding test IDs
