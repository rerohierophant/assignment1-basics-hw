import torch


# loss = -log(softmax(logits)[target]) = -logits[target] + log(sum(exp(logits)))
def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    计算交叉熵损失（数值稳定版本）
    
    Args:
        logits: (batch_size, vocab_size) 模型输出的未归一化分数
        targets: (batch_size,) 每个样本的正确类别索引
    Returns:
        标量，batch内的平均损失
    """

    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values  # (B, V)
    
    # 计算 log(sum(exp(shifted_logits)))
    log_sum_exp = torch.log(torch.exp(shifted_logits).sum(dim=-1))  # (B,)
    
    # 取出每个样本对应target位置的logit值（也是shifted后的）
    batch_size = logits.size(0)
    target_logits = shifted_logits[torch.arange(batch_size, device=logits.device), targets]  # (B,)
    
    loss_per_sample = -target_logits + log_sum_exp  # (B,)
    
    # 5. 返回 batch 内平均损失
    return loss_per_sample.mean()