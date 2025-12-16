#!/usr/bin/env python3
"""
Extract PLM embeddings for all protein sequences.

Supports ESM-2 and ProtBert models.
Outputs embeddings to a single HDF5 file for efficient storage.

Usage:
    python scripts/extract_embeddings.py --plm esm2_650m --output data/embeddings/esm2_650m.h5
    python scripts/extract_embeddings.py --plm protbert --output data/embeddings/protbert.h5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.plm_registry import (
    PLM_REGISTRY, 
    get_plm_info, 
    load_plm, 
    extract_embeddings,
    list_plms,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract PLM embeddings to HDF5')
    parser.add_argument(
        '--plm',
        type=str,
        default='esm2_650m',
        choices=list(PLM_REGISTRY.keys()),
        help='PLM to use for embedding extraction'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/train.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--cb513_csv',
        type=str,
        default='data/cb513.csv',
        help='Path to CB513 test CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HDF5 file path (default: data/embeddings/{plm}.h5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for extraction (1 recommended for variable lengths)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cuda, cpu)'
    )
    parser.add_argument(
        '--list_plms',
        action='store_true',
        help='List available PLMs and exit'
    )
    return parser.parse_args()


def extract_to_hdf5(
    model,
    tokenizer,
    plm_info,
    df: pd.DataFrame,
    output_path: Path,
    max_length: int = 1024,
    device: str = 'cuda',
    dataset_name: str = 'train',
):
    """Extract embeddings for all sequences in a dataframe and save to HDF5."""
    
    sequences = df['seq'].tolist()
    ids = df['id'].tolist()
    
    # Open HDF5 file in append mode
    with h5py.File(output_path, 'a') as f:
        # Create group for this dataset if it doesn't exist
        if dataset_name not in f:
            grp = f.create_group(dataset_name)
        else:
            grp = f[dataset_name]
        
        # Store metadata
        if 'plm_name' not in f.attrs:
            f.attrs['plm_name'] = plm_info.model_id
            f.attrs['embedding_dim'] = plm_info.embedding_dim
            f.attrs['model_type'] = plm_info.model_type
        
        for i, (seq_id, seq) in enumerate(tqdm(
            zip(ids, sequences), 
            total=len(sequences),
            desc=f'Extracting {dataset_name}'
        )):
            seq_id_str = str(seq_id)
            
            # Skip if already exists
            if seq_id_str in grp:
                continue
            
            # Extract embedding
            embeddings = extract_embeddings(
                model=model,
                tokenizer=tokenizer,
                sequences=[seq],
                plm_type=plm_info.model_type,
                device=device,
                max_length=max_length,
            )
            
            emb = embeddings[0].numpy().astype(np.float16)  # Save as float16 to reduce size
            
            # Store in HDF5
            grp.create_dataset(
                seq_id_str,
                data=emb,
                compression='gzip',
                compression_opts=4,
            )
        
        # Store ID list for this dataset
        if f'{dataset_name}_ids' not in f:
            f.create_dataset(
                f'{dataset_name}_ids',
                data=np.array([str(i) for i in ids], dtype='S20'),
            )


def main():
    args = parse_args()
    
    if args.list_plms:
        list_plms()
        return
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Get PLM info
    plm_info = get_plm_info(args.plm)
    
    print("=" * 60)
    print(f"PLM Embedding Extraction")
    print("=" * 60)
    print(f"PLM: {args.plm}")
    print(f"Model: {plm_info.model_id}")
    print(f"Embedding dim: {plm_info.embedding_dim}")
    print(f"Device: {device}")
    print()
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default relative to project root
        output_path = Path(__file__).parent.parent / f'data/embeddings/{args.plm}.h5'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}")
    print()
    
    # Load model
    print("Loading PLM...")
    model, tokenizer = load_plm(args.plm, device=device)
    print(f"Model loaded successfully!")
    print()
    
    # Resolve paths relative to project root if they don't exist in CWD
    project_root = Path(__file__).parent.parent
    
    train_path = Path(args.train_csv)
    if not train_path.exists() and (project_root / args.train_csv).exists():
        train_path = project_root / args.train_csv

    if train_path.exists():
        print(f"Processing training data from {train_path}")
        train_df = pd.read_csv(train_path)
        print(f"  Sequences: {len(train_df)}")
        
        extract_to_hdf5(
            model=model,
            tokenizer=tokenizer,
            plm_info=plm_info,
            df=train_df,
            output_path=output_path,
            max_length=args.max_length,
            device=device,
            dataset_name='train',
        )
    else:
        print(f"Warning: Training data not found at {args.train_csv}")
    
    # Process CB513 test data
    cb513_path = Path(args.cb513_csv)
    if not cb513_path.exists() and (project_root / args.cb513_csv).exists():
        cb513_path = project_root / args.cb513_csv

    if cb513_path.exists():
        print(f"\nProcessing CB513 test data from {cb513_path}")
        cb513_df = pd.read_csv(cb513_path)
        print(f"  Sequences: {len(cb513_df)}")
        
        extract_to_hdf5(
            model=model,
            tokenizer=tokenizer,
            plm_info=plm_info,
            df=cb513_df,
            output_path=output_path,
            max_length=args.max_length,
            device=device,
            dataset_name='cb513',
        )
    else:
        print(f"Warning: CB513 data not found at {args.cb513_csv}")
    
    # Summary
    if output_path.exists():
        with h5py.File(output_path, 'r') as f:
            train_count = len(f['train']) if 'train' in f else 0
            cb513_count = len(f['cb513']) if 'cb513' in f else 0
        
        file_size = output_path.stat().st_size / 1e9
        
        print()
        print("=" * 60)
        print(f"✅ Extraction complete!")
        print(f"   Train embeddings: {train_count}")
        print(f"   CB513 embeddings: {cb513_count}")
        print(f"   File size: {file_size:.2f} GB")
        print(f"   Location: {output_path.absolute()}")
        print("=" * 60)
    else:
        print("\nNo output file was created (no input data found or processed).")


if __name__ == '__main__':
    main()
