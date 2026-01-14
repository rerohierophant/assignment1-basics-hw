import argparse
import os
import json
from cs336_basics.tokenizer import train_bpe

def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on TinyStories")
    parser.add_argument("--input_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens")
    parser.add_argument("--output_dir", type=str, default="./tokenizer", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Training BPE on {args.input_path} with vocab size {args.vocab_size}...")
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)
    
    # Save vocab
    os.makedirs(args.output_dir, exist_ok=True) # exist ok意思是如果目录存在则不报错直接保存
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    
    # Convert vocab to format suitable for JSON and Tokenizer.from_files
    # We attempt to decode as utf-8. If fail, we use latin-1 
    # (Note: This might cause issues with non-utf8 bytes when loading back with Tokenizer.from_files 
    # which assumes utf-8 encoding for strings, but it's the standard simple approach)
    vocab_json = {}
    for k, v in vocab.items():
        try:
            vocab_json[str(k)] = v.decode('utf-8')
        except UnicodeDecodeError:
            vocab_json[str(k)] = v.decode('latin-1')
            
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
        
    # Save merges
    merges_path = os.path.join(args.output_dir, "merges.txt")
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write("# version: 0.2\n")
        for t1, t2 in merges:
            try:
                s1 = t1.decode('utf-8')
            except:
                s1 = t1.decode('latin-1')
            try:
                s2 = t2.decode('utf-8')
            except:
                s2 = t2.decode('latin-1')
            f.write(f"{s1} {s2}\n")
            
    print(f"Saved vocab to {vocab_path} and merges to {merges_path}")

if __name__ == "__main__":
    main()
