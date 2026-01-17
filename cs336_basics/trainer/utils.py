import torch
import numpy as np
import math
from typing import Iterable

def get_batch(
    dataset: np.ndarray, # 数据集，一维token id数组
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从数据集中采样输入和目标。随机选batch size个起始索引，往后取context length长度的token
    
    Args:
        dataset: 整数 token ID 的一维 numpy 数组。
        batch_size: 批次中的样本数量。
        context_length: 每个样本的长度。
        device: 将张量移动到的设备。
        
    Returns:
        x: 形状为 (batch_size, context_length) 的输入张量。
        y: 形状为 (batch_size, context_length) 的目标张量。
    """
    # 我们需要采样随机起始索引，使得 i + context_length + 1 <= len(dataset)
    # 起始索引 i 的有效范围是 [0, len(dataset) - context_length - 1]

    max_start_index = len(dataset) - context_length
    ix = np.random.randint(0, max_start_index, (batch_size,))
    
    x_batch = []
    y_batch = []
    
    for i in ix:
        x_batch.append(dataset[i : i + context_length])
        y_batch.append(dataset[i + 1 : i + context_length + 1]) #因为是用来预测下一个token
        
    x = torch.from_numpy(np.array(x_batch)).to(torch.long)
    y = torch.from_numpy(np.array(y_batch)).to(torch.long) 
    
    if "cuda" in device and not torch.cuda.is_available():
        # 如果需要，进行回退或错误处理，但通常我们假设用户知道他们在做什么
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
    使用带有预热的余弦调度计算迭代 `it` 时的学习率。
    
    Args:
        it: 当前迭代次数。
        max_learning_rate: 预热后的最大学习率。
        min_learning_rate: 余弦周期结束时的最小学习率。
        warmup_iters: 线性预热的迭代次数。
        cosine_cycle_iters: 余弦衰减的迭代次数。
        
    Returns:
        当前学习率。
    """
    # 预热阶段，学习率线性增加
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
        
    # 后余弦阶段（恒定最小学习率）
    if it > warmup_iters + cosine_cycle_iters:
        return min_learning_rate
        
    # 余弦衰减阶段。预热结束后，学习率按照余弦曲线降低到min lr
    decay_ratio = (it - warmup_iters) / cosine_cycle_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    将参数的梯度裁剪到最大 L2 范数。防止梯度爆炸
    
    Args:
        parameters: 要裁剪的可迭代参数。
        max_l2_norm: 最大 L2 范数。
    """
    # 过滤掉没有梯度的参数
    params = [p for p in parameters if p.grad is not None]
    
    if not params:
        return
        
    # 计算总范数
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2 #detach从计算图中分离梯度，stack拼成一个向量
    )
    
    # 裁剪梯度
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef) #mul_原地操作，梯度乘以系数
