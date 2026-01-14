import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer

# uv run python scripts/tokenize_test.py --data_dir /home/tianle/Downloads/ 

def encode_txt_as_numpy_array(tokenizer: Tokenizer, path_to_txt: str, save_path: str):
    """
    Tokenize a text file and save it as a numpy memmap array.
    
    Args:
        tokenizer: The tokenizer instance.
        path_to_txt: Path to the input text file.
        save_path: Path to save the output numpy array.
    """
    if not os.path.exists(path_to_txt):
        print(f"Warning: Input file {path_to_txt} not found. Skipping.")
        return

    print(f"Processing {path_to_txt}...")
    
    # 1. Count lines for progress bar
    with open(path_to_txt, 'r', encoding='utf-8', errors='ignore') as f:
        num_lines = sum(1 for _ in f)
    
    # 2. First pass: Count total tokens
    total_tokens = 0
    with open(path_to_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            # Ensure line is stripped of trailing newline if desired, or keep it.
            # Usually we process line by line.
            # Note: tokenizer.encode returns list of ints
            total_tokens += len(tokenizer.encode(line))
            
    print(f"Total tokens found: {total_tokens}")

    # 3. Create memmap
    dtype = np.uint16 # Use uint16 if vocab size < 65535, else int32
    if len(tokenizer.vocab) > 65535:
        dtype = np.int32
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokens_mm = np.memmap(save_path, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 4. Second pass: Tokenize and write
    pos = 0
    with open(path_to_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n

    tokens_mm.flush()
    print(f"Saved tokenized data to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset using trained BPE tokenizer")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory containing vocab.json and merges.txt")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing input text files")
    parser.add_argument("--train_file", type=str, default="TinyStoriesV2-GPT4-train.txt", help="Training text filename")
    parser.add_argument("--valid_file", type=str, default="TinyStoriesV2-GPT4-valid.txt", help="Validation text filename")
    parser.add_argument("--output_train_file", type=str, default="train.dat", help="Output training data filename")
    parser.add_argument("--output_valid_file", type=str, default="valid.dat", help="Output validation data filename")
    
    args = parser.parse_args()

    # Paths
    vocab_path = os.path.join(args.tokenizer_dir, "vocab.json")
    merges_path = os.path.join(args.tokenizer_dir, "merges.txt")
    
    train_txt_path = os.path.join(args.data_dir, args.train_file)
    valid_txt_path = os.path.join(args.data_dir, args.valid_file)
    
    train_out_path = os.path.join(args.data_dir, args.output_train_file)
    valid_out_path = os.path.join(args.data_dir, args.output_valid_file)

    # Check tokenizer files
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"Error: Tokenizer files not found in {args.tokenizer_dir}")
        sys.exit(1)

    # Load Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=["<|endoftext|>"]
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # Test Tokenizer
    print("\n=== Testing Tokenizer ===")
    test_texts = [
        "Once upon a time, there was a little robot.",
        "Hello world! <|endoftext|> Some more text.",
        "<|endoftext|>",
        "你好，世界！"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        encoded = tokenizer.encode(text)
        print("Encoded IDs:", encoded)
        
        # Decode check
        decoded = tokenizer.decode(encoded)
        print("Decoded:", decoded)
        print("Match:", decoded == text)

    print("\n=== Tokenizing Data ===")
    encode_txt_as_numpy_array(tokenizer, train_txt_path, train_out_path)
    encode_txt_as_numpy_array(tokenizer, valid_txt_path, valid_out_path)

if __name__ == "__main__":
    main()
