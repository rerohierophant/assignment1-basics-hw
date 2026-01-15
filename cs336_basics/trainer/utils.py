import torch
import numpy as np
import math
from typing import Iterable

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of inputs and targets from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of examples in the batch.
        context_length: Length of each example.
        device: Device to move the tensors to.
        
    Returns:
        x: Input tensor of shape (batch_size, context_length).
        y: Target tensor of shape (batch_size, context_length).
    """
    # We need to sample random starting indices such that i + context_length + 1 <= len(dataset)
    # The valid range for start index i is [0, len(dataset) - context_length - 1]
    # because we need context_length tokens for x and context_length tokens for y (shifted by 1)
    # x = dataset[i : i + context_length]
    # y = dataset[i + 1 : i + context_length + 1]
    
    max_start_index = len(dataset) - context_length
    ix = np.random.randint(0, max_start_index, (batch_size,))
    
    x_batch = []
    y_batch = []
    
    for i in ix:
        x_batch.append(dataset[i : i + context_length])
        y_batch.append(dataset[i + 1 : i + context_length + 1])
        
    x = torch.from_numpy(np.array(x_batch)).to(torch.long)
    y = torch.from_numpy(np.array(y_batch)).to(torch.long)
    
    if "cuda" in device and not torch.cuda.is_available():
        # Fallback or error handling if needed, but usually we assume the user knows what they are doing
        pass
        
    x = x.to(device)
    y = y.to(device)
    
    return x, y

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Calculate the learning rate at iteration `it` using a cosine schedule with warmup.
    
    Args:
        it: Current iteration number.
        max_learning_rate: Maximum learning rate after warmup.
        min_learning_rate: Minimum learning rate at the end of the cosine cycle.
        warmup_iters: Number of iterations for linear warmup.
        cosine_cycle_iters: Number of iterations for cosine decay.
        
    Returns:
        Current learning rate.
    """
    # 1. Warmup phase
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
        
    # 2. Post-cosine phase (constant min_learning_rate)
    if it > warmup_iters + cosine_cycle_iters:
        return min_learning_rate
        
    # 3. Cosine decay phase
    decay_ratio = (it - warmup_iters) / cosine_cycle_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip the gradients of the parameters to a maximum L2 norm.
    
    Args:
        parameters: Iterable of parameters to clip.
        max_l2_norm: Maximum L2 norm.
    """
    # Filter out parameters that don't have gradients
    params = [p for p in parameters if p.grad is not None]
    
    if not params:
        return
        
    # Calculate total norm
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2
    )
    
    # Clip gradients
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)
