"""
Text generation script for Transformer language model.

This script implements decoding functionality including temperature scaling,
top-p (nucleus) sampling, and text generation from trained models.
"""

import argparse
import torch
import numpy as np
from typing import List, Optional, Union
import sys
import os

# Add the project root to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修正导入路径和类名
from cs336_basics.model.transformer import TransformerLM
# 移除不存在的 check_pointing 导入，直接用 torch.load

def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling and softmax to logits."""
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax for numerical stability
    # Subtract max for numerical stability
    max_logits = torch.max(scaled_logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(scaled_logits - max_logits)
    probabilities = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    
    return probabilities


def top_p_sampling(probabilities: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Apply top-p (nucleus) sampling to probability distribution."""
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to keep (cumulative probability <= p)
    mask = cumulative_probs <= p
    
    # Always keep at least the top token
    mask[..., 0] = True
    
    # Zero out probabilities for tokens not in nucleus
    filtered_probs = sorted_probs * mask.float()
    
    # Renormalize
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)
    
    # Create output tensor with same shape as input
    output_probs = torch.zeros_like(probabilities)
    output_probs.scatter_(-1, sorted_indices, filtered_probs)
    
    return output_probs


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token: Optional[str] = "<|endoftext|>"
) -> str:
    """Generate text from a trained language model."""
    model.eval()
    
    # Encode the prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    # Convert to tensor and move to device
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens one by one
    generated_tokens = list(prompt_tokens) # Make a copy
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass through model
            # 注意：这里需要根据 TransformerLM 的 forward 签名传递参数
            # 如果你的 forward 只需要 input_ids，就这样写：
            logits = model(input_ids)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            # Apply temperature scaling and softmax
            probabilities = softmax_with_temperature(next_token_logits, temperature)
            
            # Apply top-p sampling if specified
            if top_p < 1.0:
                probabilities = top_p_sampling(probabilities, top_p)
            
            # Sample next token
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            
            # Add to generated sequence
            generated_tokens.append(next_token)
            
            # Check for end-of-sequence token
            if eos_token is not None:
                decoded_token = tokenizer.decode([next_token])
                # 简单检查，具体取决于 tokenizer 的 decode 实现
                if decoded_token == eos_token or (hasattr(tokenizer, 'eos_token') and next_token == tokenizer.eos_id):
                    break
            
            # Update input for next iteration
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # Truncate if sequence gets too long (to avoid memory issues)
            # 使用 model.max_len 而不是 context_length
            if input_ids.size(1) > model.max_len:
                input_ids = input_ids[:, -model.max_len:]
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def load_model_and_tokenizer(args, device: str = "cpu"):
    """Load a trained model and tokenizer."""
    
    # Import tokenizer
    try:
        from cs336_basics.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_files(args.vocab, args.merges)
    except ImportError:
        print("Error: Could not import Tokenizer from cs336_basics.tokenizer")
        return None, None
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None
    
    print("Initializing model structure...")
    # Create model structure
    # 注意：必须使用与训练时完全相同的参数！
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_len=args.context_len,
        theta=args.rope_theta
    ).to(device)
    
    print(f"Loading weights from {args.checkpoint}...")
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback for raw state dicts
        model.load_state_dict(checkpoint)
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Generate text with a trained Transformer language model')
    
    # Required paths
    parser.add_argument('--checkpoint', type=str, required=True, default='./checkpoints/checkpoint_final_6000.pt', help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, default='./tokenizer/vocab.json', help='Path to tokenizer vocabulary file')
    parser.add_argument('--merges', type=str, required=True, default='./tokenizer/merges.txt', help='Path to tokenizer merges file')
    
    # Model Configuration (MUST match training config)
    # 既然 checkpoint 里没存 config，这里就必须提供默认值或参数
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN dimension')
    parser.add_argument('--context_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p threshold for nucleus sampling')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer.")
        return
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Generate text
    print(f"\nGenerating {args.num_samples} sample(s) with prompt: '{args.prompt}'")
    print("-" * 80)
    
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\nSample {i+1}:")
        
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device
            )
            
            print(generated_text)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            break
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()