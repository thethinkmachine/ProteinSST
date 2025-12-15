#!/usr/bin/env python3
"""
Download and prepare CB513 dataset for testing.

CB513 is a benchmark dataset for protein secondary structure prediction.
It contains 513 non-redundant protein chains with experimentally determined structures.

Usage:
    python scripts/download_cb513.py
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def download_cb513_from_huggingface() -> pd.DataFrame:
    """
    Download CB513 dataset from HuggingFace.
    
    Returns:
        DataFrame with columns [id, seq, sst8, sst3]
    """
    from datasets import load_dataset
    
    # Load from proteinea/secondary_structure_prediction which has CB513.csv
    print("Loading CB513 from HuggingFace proteinea/secondary_structure_prediction...")
    
    ds = load_dataset("proteinea/secondary_structure_prediction", data_files="CB513.csv", split="train")
    df = ds.to_pandas()
    
    # Ensure required columns exist and rename if needed
    col_mapping = {
        'input': 'seq',
        'sequence': 'seq',
        'dssp8': 'sst8',
        'dssp3': 'sst3',
        'ss8': 'sst8',
        'ss3': 'sst3',
    }
    
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    # Add ID column if missing
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    # Select only required columns
    df = df[['id', 'seq', 'sst8', 'sst3']]
    
    return df


def validate_cb513(df: pd.DataFrame) -> bool:
    """Validate the CB513 dataset."""
    
    print(f"\n📊 CB513 Dataset Summary:")
    print(f"   Proteins: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check required columns
    required = {'id', 'seq', 'sst8', 'sst3'}
    if not required.issubset(df.columns):
        print(f"   ❌ Missing columns: {required - set(df.columns)}")
        return False
    
    # Check sequence lengths
    seq_lengths = df['seq'].str.len()
    print(f"   Sequence lengths: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}")
    
    # Check that sst8 and sst3 have same length as seq
    mismatched = (df['seq'].str.len() != df['sst8'].str.len()).sum()
    if mismatched > 0:
        print(f"   ❌ {mismatched} proteins have mismatched seq/sst8 lengths")
        return False
    
    # Check valid SST8 characters
    valid_sst8 = set('GHIEBTSC')
    all_sst8_chars = set(''.join(df['sst8'].tolist()))
    invalid_chars = all_sst8_chars - valid_sst8
    if invalid_chars:
        print(f"   ⚠️ Unknown SST8 characters: {invalid_chars}")
    
    # Check valid SST3 characters
    valid_sst3 = set('HEC')
    all_sst3_chars = set(''.join(df['sst3'].tolist()))
    invalid_chars = all_sst3_chars - valid_sst3
    if invalid_chars:
        print(f"   ⚠️ Unknown SST3 characters: {invalid_chars}")
    
    print(f"   ✅ Validation passed!")
    return True


def main():
    output_path = Path('data/cb513.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"CB513 already exists at {output_path}")
        df = pd.read_csv(output_path)
        validate_cb513(df)
        return
    
    print("=" * 60)
    print("Downloading CB513 Dataset")
    print("=" * 60)
    
    # Download
    df = download_cb513_from_huggingface()
    
    # Validate
    if not validate_cb513(df):
        raise ValueError("Dataset validation failed")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
