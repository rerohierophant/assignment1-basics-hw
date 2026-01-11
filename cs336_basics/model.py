import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Modules import Linear, RMSNorm, SwiGLU # .xx用来说明模块来自本地的Module.py文件

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, '-inf')
        scores = torch.softmax(scores, dim=-1)
        return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model // n_heads != 0"
        self.d_head = d_model // n_heads
        self.W_q = Linear(d_model, d_model, bias=False)
        self.W_k = Linear(d_model, d_model, bias=False)
        self.W_v = Linear(d_model, d_model, bias=False)
        self.W_o = Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention()
        self.RMSnorm = RMSNorm(d_model)

    def _split_heads(self, x):
        # x: B, L, d
        B, L, d = x.size()
        x = x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x):
        # x: B, H, L, d
        B, H, L, d = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return x

    def forward(self, q, k, v, mask: torch.Tensor | None = None):
        q = self._split_heads(self.W_q(q))
        k = self._split_heads(self.W_k(k))
        v = self._split_heads(self.W_v(v))
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.expand(-1, self.n_heads, -1, -1)
        attn = self.attn(q, k, v, mask)
        out = self._merge_heads(attn)
        return self.W_o(out)

class FFN(nn.Module):
    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.fc1 = Linear(d_model, dim_ff, bias=False)
        self.fc2 = Linear(dim_ff, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(self, x):
        return self.fc2(SwiGLU(self.fc1(x)))

        

