"""
Data loading and preprocessing for ProteinSST.
Includes Dataset, DataLoader, and encoding utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import (
    AMINO_ACIDS, AA_TO_IDX, VOCAB_TO_IDX,
    SST8_TO_IDX, SST3_TO_IDX, SST8_CLASSES, SST3_CLASSES,
    LEAKAGE_TRAIN_IDS, PAD_TOKEN, UNK_TOKEN
)


# =============================================================================
# BLOSUM62 Matrix (simplified representation)
# =============================================================================

# BLOSUM62 matrix values for each amino acid (normalized to 0-1 range)
BLOSUM62 = {
    'A': [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    'R': [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    'N': [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    'D': [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    'C': [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    'Q': [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    'E': [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    'G': [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    'H': [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    'I': [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    'L': [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    'K': [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    'M': [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    'F': [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    'P': [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    'S': [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    'T': [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    'W': [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    'Y': [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    'V': [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
}

def get_blosum62_features(sequence: str) -> torch.Tensor:
    """Convert sequence to BLOSUM62 feature matrix."""
    features = []
    for aa in sequence:
        if aa in BLOSUM62:
            features.append(BLOSUM62[aa])
        else:
            features.append([0] * 20)  # Unknown AA
    return torch.tensor(features, dtype=torch.float32) / 10.0  # Normalize


# =============================================================================
# Encoding Functions
# =============================================================================

def one_hot_encode(sequence: str, vocab: Dict[str, int] = None) -> torch.Tensor:
    """One-hot encode a protein sequence."""
    if vocab is None:
        vocab = AA_TO_IDX
    
    encoded = torch.zeros(len(sequence), len(vocab), dtype=torch.float32)
    for i, aa in enumerate(sequence):
        if aa in vocab:
            encoded[i, vocab[aa]] = 1.0
        # Unknown amino acids get zero vector
    return encoded


def encode_sequence(sequence: str, vocab: Dict[str, int] = None) -> torch.Tensor:
    """Encode sequence to integer indices."""
    if vocab is None:
        vocab = VOCAB_TO_IDX
    
    encoded = []
    for aa in sequence:
        if aa in vocab:
            encoded.append(vocab[aa])
        else:
            encoded.append(vocab[UNK_TOKEN])
    return torch.tensor(encoded, dtype=torch.long)


def encode_sst8(labels: str) -> torch.Tensor:
    """Encode SST8 labels to indices."""
    return torch.tensor([SST8_TO_IDX[c] for c in labels], dtype=torch.long)


def encode_sst3(labels: str) -> torch.Tensor:
    """Encode SST3 labels to indices."""
    return torch.tensor([SST3_TO_IDX[c] for c in labels], dtype=torch.long)


def positional_encoding(length: int, dim: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding."""
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =============================================================================
# Dataset Classes
# =============================================================================

class ProteinDataset(Dataset):
    """
    Dataset for protein secondary structure prediction.
    
    Args:
        csv_path: Path to CSV file with columns [id, seq, sst8, sst3]
        max_length: Maximum sequence length (longer sequences are truncated)
        use_blosum: Whether to include BLOSUM62 features
        use_positional: Whether to include positional encoding
        augmentation: Augmentation function to apply
        exclude_ids: List of IDs to exclude (for leakage prevention)
        is_test: If True, expects only [id, seq] columns
    """
    
    def __init__(
        self,
        csv_path: str,
        max_length: int = 512,
        use_blosum: bool = True,
        use_positional: bool = False,
        augmentation = None,
        exclude_ids: List[int] = None,
        is_test: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        self.use_blosum = use_blosum
        self.use_positional = use_positional
        self.augmentation = augmentation
        self.is_test = is_test
        
        # Exclude leakage IDs
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
        
        # Calculate feature dimension
        self.feature_dim = 20  # One-hot
        if use_blosum:
            self.feature_dim += 20
        if use_positional:
            self.feature_dim += 64  # Positional encoding dim
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = row['seq']
        
        # Truncate if needed
        if len(seq) > self.max_length:
            # Center crop for long sequences
            start = (len(seq) - self.max_length) // 2
            seq = seq[start:start + self.max_length]
        
        # Apply augmentation
        if self.augmentation is not None and not self.is_test:
            sst8 = row.get('sst8', '')
            sst3 = row.get('sst3', '')
            if len(sst8) > self.max_length:
                start = (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
            seq, sst8, sst3 = self.augmentation(seq, sst8, sst3)
        else:
            sst8 = row.get('sst8', '')
            sst3 = row.get('sst3', '')
            if not self.is_test and len(sst8) > self.max_length:
                start = (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
        
        # Build features
        features = [one_hot_encode(seq)]
        
        if self.use_blosum:
            features.append(get_blosum62_features(seq))
        
        if self.use_positional:
            features.append(positional_encoding(len(seq), 64))
        
        x = torch.cat(features, dim=-1)
        
        result = {
            'id': row['id'],
            'features': x,
            'length': len(seq),
            'sequence': seq,
        }
        
        if not self.is_test:
            result['sst8'] = encode_sst8(sst8)
            result['sst3'] = encode_sst3(sst3)
        
        return result


class PLMEmbeddingDataset(Dataset):
    """
    Dataset using pre-computed PLM embeddings (legacy .pt files).
    
    Args:
        csv_path: Path to CSV file with metadata
        embeddings_dir: Directory containing .pt embedding files
        max_length: Maximum sequence length
        exclude_ids: IDs to exclude
        is_test: If True, expects only sequence data
    """
    
    def __init__(
        self,
        csv_path: str,
        embeddings_dir: str,
        max_length: int = 512,
        exclude_ids: List[int] = None,
        is_test: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.max_length = max_length
        self.is_test = is_test
        
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = row['id']
        
        # Load pre-computed embedding
        emb_path = self.embeddings_dir / f"{sample_id}.pt"
        embedding = torch.load(emb_path)  # Shape: (seq_len, embed_dim)
        
        # Track truncation for consistent label handling
        truncate_start = 0
        if embedding.shape[0] > self.max_length:
            truncate_start = (embedding.shape[0] - self.max_length) // 2
            embedding = embedding[truncate_start:truncate_start + self.max_length]
        
        emb_len = embedding.shape[0]
        
        result = {
            'id': sample_id,
            'features': embedding,
            'length': emb_len,
        }
        
        if not self.is_test:
            sst8 = row['sst8']
            sst3 = row['sst3']
            
            # Apply same truncation as embedding
            if truncate_start > 0 or len(sst8) > self.max_length:
                start = truncate_start if truncate_start > 0 else (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
            
            # Ensure labels match embedding length exactly
            sst8 = sst8[:emb_len]
            sst3 = sst3[:emb_len]
            
            result['sst8'] = encode_sst8(sst8)
            result['sst3'] = encode_sst3(sst3)
        
        return result


class HDF5EmbeddingDataset(Dataset):
    """
    Dataset using pre-computed PLM embeddings from HDF5 file.
    
    This is the preferred method for loading embeddings as it reduces
    I/O overhead by storing all embeddings in a single file.
    
    Args:
        csv_path: Path to CSV file with metadata
        h5_path: Path to HDF5 file containing embeddings
        dataset_name: Name of dataset group in HDF5 ('train' or 'cb513')
        max_length: Maximum sequence length
        exclude_ids: IDs to exclude
        is_test: If True, expects only sequence data
    """
    
    def __init__(
        self,
        csv_path: str,
        h5_path: str,
        dataset_name: str = 'train',
        max_length: int = 512,
        exclude_ids: List[int] = None,
        is_test: bool = False,
    ):
        import h5py
        
        self.df = pd.read_csv(csv_path)
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.is_test = is_test
        
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
        
        # Open HDF5 file and get embedding dimension
        with h5py.File(h5_path, 'r') as f:
            self.embedding_dim = f.attrs.get('embedding_dim', None)
            # Verify dataset exists
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in {h5_path}")
        
        # Keep HDF5 file handle as None - will open per-worker
        self._h5_file = None
    
    def _get_h5_file(self):
        """Get or open HDF5 file handle (thread-safe for DataLoader workers)."""
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample_id = str(row['id'])
        
        # Load embedding from HDF5
        h5_file = self._get_h5_file()
        grp = h5_file[self.dataset_name]
        
        if sample_id not in grp:
            raise KeyError(f"Embedding for ID '{sample_id}' not found in {self.dataset_name}")
        
        embedding = torch.from_numpy(grp[sample_id][:].astype(np.float32))
        
        # Track truncation for consistent label handling
        truncate_start = 0
        if embedding.shape[0] > self.max_length:
            truncate_start = (embedding.shape[0] - self.max_length) // 2
            embedding = embedding[truncate_start:truncate_start + self.max_length]
        
        emb_len = embedding.shape[0]
        
        result = {
            'id': row['id'],
            'features': embedding,
            'length': emb_len,
        }
        
        if not self.is_test:
            sst8 = row['sst8']
            sst3 = row['sst3']
            
            # Apply same truncation as embedding
            if truncate_start > 0 or len(sst8) > self.max_length:
                start = truncate_start if truncate_start > 0 else (len(sst8) - self.max_length) // 2
                sst8 = sst8[start:start + self.max_length]
                sst3 = sst3[start:start + self.max_length]
            
            # Ensure labels match embedding length exactly
            sst8 = sst8[:emb_len]
            sst3 = sst3[:emb_len]
            
            result['sst8'] = encode_sst8(sst8)
            result['sst3'] = encode_sst3(sst3)
        
        return result
    
    def __del__(self):
        """Close HDF5 file handle on cleanup."""
        if self._h5_file is not None:
            self._h5_file.close()


class OnTheFlyPLMDataset(Dataset):
    """
    Dataset that extracts ESM-2 embeddings on-the-fly.
    
    Slower than using pre-computed embeddings, but works without extraction step.
    
    Args:
        csv_path: Path to CSV file
        esm_model: Loaded ESM-2 model
        tokenizer: ESM-2 tokenizer
        device: Device for model inference
        max_length: Maximum sequence length
        exclude_ids: IDs to exclude
        is_test: If True, expects only sequence data
    """
    
    def __init__(
        self,
        csv_path: str,
        esm_model,
        tokenizer,
        device: str = 'cuda',
        max_length: int = 512,
        exclude_ids: List[int] = None,
        is_test: bool = False,
    ):
        self.df = pd.read_csv(csv_path)
        self.esm_model = esm_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.is_test = is_test
        
        if exclude_ids:
            self.df = self.df[~self.df['id'].isin(exclude_ids)].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.df)
    
    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq = row['seq']
        
        # Keep original labels before any truncation
        sst8_orig = row['sst8'] if not self.is_test else None
        sst3_orig = row['sst3'] if not self.is_test else None
        
        # Truncate sequence if needed
        truncate_start = 0
        if len(seq) > self.max_length:
            truncate_start = (len(seq) - self.max_length) // 2
            seq = seq[truncate_start:truncate_start + self.max_length]
        
        # Tokenize and extract embedding
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=self.max_length + 2,
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass through ESM-2
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state.squeeze(0).cpu()  # (seq_len, hidden)
        
        # Remove special tokens (BOS and EOS)
        embedding = embedding[1:-1, :]
        
        # Get the actual embedding length (may differ slightly from seq length)
        emb_len = embedding.shape[0]
        
        result = {
            'id': row['id'],
            'features': embedding,
            'length': emb_len,
        }
        
        if not self.is_test:
            # Truncate labels to match the EXACT embedding length
            sst8 = sst8_orig[truncate_start:truncate_start + self.max_length]
            sst3 = sst3_orig[truncate_start:truncate_start + self.max_length]
            
            # Ensure labels match embedding length exactly
            sst8 = sst8[:emb_len]
            sst3 = sst3[:emb_len]
            
            result['sst8'] = encode_sst8(sst8)
            result['sst3'] = encode_sst3(sst3)
        
        return result


# =============================================================================
# Collate Functions
# =============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding for variable length sequences."""
    
    # Sort by length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    features = [item['features'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    ids = [item['id'] for item in batch]
    
    # Pad features
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    result = {
        'features': features_padded,
        'lengths': lengths,
        'ids': ids,
    }
    
    # Pad labels if present
    if 'sst8' in batch[0]:
        sst8 = [item['sst8'] for item in batch]
        sst3 = [item['sst3'] for item in batch]
        
        # Pad with -100 (ignored index for CrossEntropy)
        sst8_padded = pad_sequence(sst8, batch_first=True, padding_value=-100)
        sst3_padded = pad_sequence(sst3, batch_first=True, padding_value=-100)
        
        result['sst8'] = sst8_padded
        result['sst3'] = sst3_padded
    
    return result


def create_dataloaders(
    train_csv: str,
    val_split: float = 0.1,
    batch_size: int = 32,
    max_length: int = 512,
    use_blosum: bool = True,
    use_positional: bool = False,
    augmentation = None,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_df = pd.read_csv(train_csv)
    
    # Exclude leakage IDs
    full_df = full_df[~full_df['id'].isin(LEAKAGE_TRAIN_IDS)].reset_index(drop=True)
    
    # Stratified split by sequence length bins
    np.random.seed(seed)
    full_df['length_bin'] = pd.cut(full_df['seq'].str.len(), bins=10, labels=False)
    
    val_indices = full_df.groupby('length_bin', group_keys=False).apply(
        lambda x: x.sample(frac=val_split, random_state=seed),
        include_groups=False,
    ).index.tolist()
    
    train_indices = [i for i in range(len(full_df)) if i not in val_indices]
    
    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)
    
    # Save temp CSVs (or use in-memory datasets)
    train_df.to_csv('/tmp/train_split.csv', index=False)
    val_df.to_csv('/tmp/val_split.csv', index=False)
    
    # Create datasets
    train_dataset = ProteinDataset(
        '/tmp/train_split.csv',
        max_length=max_length,
        use_blosum=use_blosum,
        use_positional=use_positional,
        augmentation=augmentation,
    )
    
    val_dataset = ProteinDataset(
        '/tmp/val_split.csv',
        max_length=max_length,
        use_blosum=use_blosum,
        use_positional=use_positional,
        augmentation=None,  # No augmentation for validation
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
