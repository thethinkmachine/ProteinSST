#!/usr/bin/env python3
"""
Extract ESM-2 embeddings for all protein sequences.

This script pre-computes embeddings for faster training in Tier 3-5.
Run this once before training PLM-based models.

Usage:
    python scripts/extract_embeddings.py --model esm2_t33_650M --batch_size 4
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from transformers import EsmTokenizer, EsmModel


# Model options
ESM_MODELS = {
    'esm2_t6_8M': 'facebook/esm2_t6_8M_UR50D',      # 8M params (fastest)
    'esm2_t12_35M': 'facebook/esm2_t12_35M_UR50D',   # 35M params
    'esm2_t33_650M': 'facebook/esm2_t33_650M_UR50D', # 650M params (best)
}


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ESM-2 embeddings')
    parser.add_argument(
        '--model', 
        type=str, 
        default='esm2_t33_650M',
        choices=list(ESM_MODELS.keys()),
        help='ESM-2 model to use'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/train.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--test_csv',
        type=str,
        default='data/test.csv',
        help='Path to test CSV'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for extraction'
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
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(
    sequences: list,
    ids: list,
    model: EsmModel,
    tokenizer: EsmTokenizer,
    output_dir: Path,
    batch_size: int = 4,
    max_length: int = 1024,
    device: str = 'cuda',
):
    """Extract and save embeddings for all sequences."""
    
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc='Extracting embeddings'):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(sequences))
        
        batch_seqs = sequences[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        
        # Skip already processed
        all_exist = all((output_dir / f"{id_}.pt").exists() for id_ in batch_ids)
        if all_exist:
            continue
        
        # Tokenize
        encoding = tokenizer(
            batch_seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # Save each sequence's embedding
        for j, (seq_id, seq) in enumerate(zip(batch_ids, batch_seqs)):
            # Get actual length (excluding padding and special tokens)
            seq_len = min(len(seq), max_length - 2)  # -2 for BOS/EOS
            
            # Extract embedding (remove BOS and EOS)
            embedding = hidden_states[j, 1:seq_len+1, :].cpu()
            
            # Save
            torch.save(embedding, output_dir / f"{seq_id}.pt")


def main():
    args = parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    
    # Load model
    model_name = ESM_MODELS[args.model]
    print(f"Loading {model_name}...")
    
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Output directory
    output_dir = Path(args.output_dir)
    
    # Process train data
    if Path(args.train_csv).exists():
        print(f"\nProcessing training data from {args.train_csv}")
        train_df = pd.read_csv(args.train_csv)
        
        extract_embeddings(
            sequences=train_df['seq'].tolist(),
            ids=train_df['id'].tolist(),
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
    
    # Process test data
    if Path(args.test_csv).exists():
        print(f"\nProcessing test data from {args.test_csv}")
        test_df = pd.read_csv(args.test_csv)
        
        extract_embeddings(
            sequences=test_df['seq'].tolist(),
            ids=test_df['id'].tolist(),
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
    
    # Summary
    num_files = len(list(output_dir.glob("*.pt")))
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
    
    print(f"\n✅ Extraction complete!")
    print(f"   Files: {num_files}")
    print(f"   Total size: {total_size / 1e9:.2f} GB")
    print(f"   Location: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
