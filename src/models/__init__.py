# ProteinSST - Models Package

# PLM Backbone (for FFT mode)
from .plm_backbone import PLMBackbone, SequenceDataset, collate_fn_sequences

# Building blocks
from .cnn_blocks import (
    CNNLayerConfig,
    CNNLayer,
    MultiscaleCNN,
    DeepCNN,
    create_cnn,
)

from .rnn_blocks import (
    RNNConfig,
    ConfigurableRNN,
    create_rnn,
)

from .classification_heads import (
    HeadConfig,
    MTLClassificationHead,
    Q3DiscardingHead,
    Q3GuidedHead,
    create_classification_head,
)

# Tier models
from .tier1_baseline import Tier1Baseline
from .tier2_cnn import Tier2CNN
from .tier3_cnn_rnn import Tier3CNNRNN


__all__ = [
    # CNN blocks
    "CNNLayerConfig",
    "CNNLayer",
    "MultiscaleCNN",
    "DeepCNN",
    "create_cnn",
    # RNN blocks
    "RNNConfig",
    "ConfigurableRNN",
    "create_rnn",
    # Classification heads
    "HeadConfig",
    "MTLClassificationHead",
    "Q3DiscardingHead",
    "Q3GuidedHead",
    "create_classification_head",
    # Tier models
    "Tier1Baseline",
    "Tier2CNN",
    "Tier3CNNRNN",
]
