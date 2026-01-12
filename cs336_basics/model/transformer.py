import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import Linear, Embedding, RMSNorm, SwiGLU, MultiHeadAttentionWithRoPE


# Pre-norm Transformer Block
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        max_len: int = 4096,
        theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        
        # 两个 RMSNorm（Pre-norm 结构）
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # 多头自注意力（带 RoPE）
        self.attn = MultiHeadAttentionWithRoPE(d_model, n_heads, max_len, theta)
        
        # 前馈网络（SwiGLU）
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) 输入
            positions: (B, L) 或 (L,) 位置索引，None时自动生成
            mask: (B, L, L) 因果mask，0表示屏蔽
        Returns:
            (B, L, d_model) 输出
        """
        # 子层1：Pre-norm + MHA + 残差
        # y = x + MHA(RMSNorm(x))
        x = x + self.attn(self.norm1(x), positions, mask)
        
        # 子层2：Pre-norm + FFN + 残差
        # z = y + FFN(RMSNorm(y))
        x = x + self.ffn(self.norm2(x))
        
        return x


# Transformer Language Model
# 结构：Token Embedding -> num_layers × TransformerBlock -> Final RMSNorm -> LM Head
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int = 4096,
        theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token Embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # num_layers 个 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_len, theta)
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm（Pre-norm结构需要在最后加一个）
        self.final_norm = RMSNorm(d_model)
        
        # LM Head（输出投影到词表）
        self.lm_head = Linear(d_model, vocab_size, bias=False)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果mask：下三角为1，上三角为0
        位置 i 只能看到位置 <= i 的token
        """
        # (L, L) 下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask  # 1表示可见，0表示屏蔽
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) 输入的token ID序列
            positions: (B, L) 或 (L,) 位置索引，None时自动生成 [0, 1, ..., L-1]
        Returns:
            (B, L, vocab_size) 未归一化的logits
        """
        B, L = token_ids.size()
        device = token_ids.device
        
        # 1. Token Embedding
        x = self.token_embedding(token_ids)  # (B, L, d_model)
        
        # 2. 生成位置索引（如果没传）
        if positions is None:
            positions = torch.arange(L, device=device)
        
        # 3. 创建因果mask（decoder-only LM需要）
        causal_mask = self._create_causal_mask(L, device)  # (L, L)
        
        # 4. 通过所有 Transformer Block
        for layer in self.layers:
            x = layer(x, positions, causal_mask)
        
        # 5. Final RMSNorm
        x = self.final_norm(x)
        
        # 6. LM Head：投影到词表维度
        logits = self.lm_head(x)  # (B, L, vocab_size)
        
        return logits
