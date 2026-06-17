import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

def process_dataset(dataset_name="bigcode/the-stack-smol", subset="data/python", split="train", out_dir="dataset/code_data"):
    print(f"Loading {dataset_name} ({subset}) dataset...")
    
    # We load a small subset of code, e.g. python code from the-stack-smol
    if subset:
        dataset = load_dataset(dataset_name, data_dir=subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
        
    print(f"Loaded {len(dataset)} examples.")

    # We use tiktoken's cl100k_base (used in GPT-4) or gpt2
    enc = tiktoken.get_encoding("gpt2")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_bin = out_dir / f"train.bin"
    val_bin = out_dir / f"val.bin"
    
    # split 90% train, 10% val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    for split_name, dset in zip(["train", "val"], [dataset["train"], dataset["test"]]):
        print(f"Processing {split_name} split...")
        # To save memory, we can stream or process in chunks
        arr_len = 0
        
        # Count total tokens first to preallocate memory if desired, but here we will append to file
        # We write to memmap to handle large datasets efficiently
        
        file_path = train_bin if split_name == "train" else val_bin
        
        # We use uint16 because GPT-2 vocab is 50257 (fits in uint16)
        dtype = np.uint16 
        
        # We append to file to avoid memory bloat
        with open(file_path, 'wb') as f:
            for example in tqdm(dset, desc=f"Tokenizing {split_name}"):
                # BigCode datasets usually have 'content' column for the code
                text = example.get('content', example.get('text', ''))
                if not text:
                    continue
                # Add a special token at the end of each document
                tokens = enc.encode_ordinary(text)
                tokens.append(enc.eot_token)
                
                # Write to file
                arr = np.array(tokens, dtype=dtype)
                f.write(arr.tobytes())
                arr_len += len(tokens)
                
        print(f"Saved {arr_len} tokens to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--out_dir", type=str, default="dataset/code_data")
    args = parser.parse_args()
    
    process_dataset(dataset_name=args.dataset, subset=args.subset, out_dir=args.out_dir)
