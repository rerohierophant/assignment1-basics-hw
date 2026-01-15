import argparse
import os
import json
from cs336_basics.tokenizer import train_bpe, get_byte_encoder

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
    
    # Use GPT-2 style byte-to-unicode mapping for serialization
    byte_encoder = get_byte_encoder()
    
    def bytes_to_unicode_str(b_data: bytes) -> str:
        return "".join([byte_encoder[b] for b in b_data])

    vocab_json = {}
    for k, v in vocab.items():
        vocab_json[str(k)] = bytes_to_unicode_str(v)
            
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
        
    # Save merges
    merges_path = os.path.join(args.output_dir, "merges.txt")
    with open(merges_path, 'w', encoding='utf-8') as f:
        f.write("# version: 0.2\n")
        for t1, t2 in merges:
            s1 = bytes_to_unicode_str(t1)
            s2 = bytes_to_unicode_str(t2)
            f.write(f"{s1} {s2}\n")
            
    print(f"Saved vocab to {vocab_path} and merges to {merges_path}")

if __name__ == "__main__":
    main()
