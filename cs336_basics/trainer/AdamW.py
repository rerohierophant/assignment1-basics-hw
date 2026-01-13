import torch
import math
from typing import Iterable, Callable, Optional


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Args:
            params: 可迭代的参数集合
            lr: 学习率 α
            betas: (β₁, β₂) 动量系数
            eps: ε 数值稳定项
            weight_decay: λ 权重衰减系数
        """
        # 将超参数打包成 defaults 字典
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步优化
        
        Args:
            closure: 可选的闭包，用于重新计算loss
        Returns:
            loss（如果提供了closure）
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # 遍历所有参数组
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            # 遍历组内每个参数
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # 获取该参数的状态（存储动量等）
                state = self.state[p]
                
                # 初始化状态（第一步时）
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)  # 一阶动量
                    state["v"] = torch.zeros_like(p.data)  # 二阶动量
                
                # 取出状态
                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]
                
                # 更新一阶动量：m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 更新二阶动量：v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                # m̂_t = m_t / (1 - β₁^t)
                m_hat = m / bias_correction1
                
                # v̂_t = v_t / (1 - β₂^t)
                v_hat = v / bias_correction2
                
                # 计算更新量：α * m̂_t / (√v̂_t + ε)
                update = m_hat / (v_hat.sqrt() + eps)
                
                # 解耦的 weight decay：直接加在更新量上，不经过动量
                if weight_decay != 0:
                    update = update + weight_decay * p.data
                
                # 更新参数：θ_t = θ_{t-1} - α * update
                p.data.add_(update, alpha=-lr)
        
        return loss
