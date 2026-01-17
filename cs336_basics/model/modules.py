import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 本质做矩阵乘法
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter( #Parameter的作用在于告诉模型这个参数需要学习，让优化器更新
            torch.Tensor(out_features, in_features)
            #为什么先out再in：为了提升内存访问效率。belike [[neuron0权重], [neuron1权重]], 在内存里是连续的
            #以及转置操作很轻量，只改变步长
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        # 做Kaiming权重初始化, 防止Relu的梯度爆炸/梯度消失
        bound = 1.0 / self.in_features ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (batch, in_features)
        # weight: (out, in)
        y = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            y += self.bias
        return y


# 把tokenizer得到的token id映射成向量，维护一个词表大小 x embedding维度的矩阵，本质上是查表
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(vocab_size, embedding_dim))
        nn.init.normal(self.weight, mean=0.0, std=1.0 / embedding_dim ** 0.5)

    def forward(self, x:torch.Tensor):
        # x: b, l
        return self.weight[x]


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):
        x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values) #在实现中需要减去最大值，防止exp溢出
        return x / x.sum(dim=dim, keepdim=True)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# 激活函数，本质 x * sigmoid(x)
class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        sig = 1 / (1 + torch.exp(-x))
        return x * sig


# 基于SiLU的门控FFN
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff


        self.w1 = Linear(d_model, d_ff, bias=False)  # up projection
        self.w2 = Linear(d_model, d_ff, bias=False)  # gate projection
        self.w3 = Linear(d_ff, d_model, bias=False)  # down projection  
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d_model)
        # SwiGLU(x) = (xW1 ⊙ SiLU(xW2)) W3
        return self.w3(self.w1(x) * self.silu(self.w2(x))) 


# Scaled Dot-Product Attention 函数（非类，方便复用）
def scaled_dot_product_attention(q, k, v, mask: torch.Tensor | None = None):
    """
    Args:
        q: (..., seq_len, d_k)
        k: (..., seq_len, d_k)
        v: (..., seq_len, d_v)
        mask: (..., seq_len, seq_len) 或 None，0表示屏蔽
    Returns:
        (..., seq_len, d_v)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


# 多头自注意力（无位置编码）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.output_proj = Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        # x: (B, L, d_model) -> (B, n_heads, L, d_head)
        B, L, _ = x.size()
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        # x: (B, n_heads, L, d_head) -> (B, L, d_model)
        B, _, L, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (B, L, d_model) 输入序列（自注意力，Q=K=V都来自x）
            mask: (B, L, L) 或 (B, 1, L, L) 注意力mask，0表示屏蔽
        Returns:
            (B, L, d_model)
        """
        # 1. 线性投影 + 分头
        q = self._split_heads(self.q_proj(x))  # (B, n_heads, L, d_head)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        
        # 2. 处理 mask 维度
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L) 广播到所有heads
        
        # 3. 注意力计算
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # 4. 合并头 + 输出投影
        return self.output_proj(self._merge_heads(attn_output))

# 旋转位置编码：通过旋转变换将位置信息注入到Q/K中
# 核心思想：将向量按相邻维度配对，对每对应用2D旋转，旋转角度与位置相关
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_k: int, max_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        # 预计算频率：freq_i = 1 / theta^(2i/d_k)，i = 0, 1, ..., d_k/2 - 1
        # shape: (d_k // 2,)
        i = torch.arange(0, d_k, 2).float()  # [0, 2, 4, ..., d_k-2]
        freqs = 1.0 / (theta ** (i / d_k))   # 频率衰减，theta为可配置参数
        
        # 预计算所有位置的角度：pos * freq
        # shape: (max_seq_len, d_k // 2)
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, freqs)  # 计算外积，(max_seq_len, d_k // 2)
        
        # 预计算cos和sin，注册为buffer（不参与训练，但会保存和移动设备）
        self.register_buffer('cos_cached', angles.cos())  # (max_seq_len, d_k // 2)
        self.register_buffer('sin_cached', angles.sin())  # (max_seq_len, d_k // 2)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        # x: (..., seq_len, d_k) - 可以是Q或K
        # positions: (..., seq_len) - 每个token的位置索引
        
        # 获取对应位置的cos/sin值
        # cos/sin: (..., seq_len, d_k // 2)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        # 将x按相邻维度拆分成两半
        # x1: 偶数维度 (0, 2, 4, ...), x2: 奇数维度 (1, 3, 5, ...)
        x1 = x[..., 0::2]  # (..., seq_len, d_k // 2)
        x2 = x[..., 1::2]  # (..., seq_len, d_k // 2)
        
        # 应用2D旋转变换
        # [cos  -sin] [x1]   [x1*cos - x2*sin]
        # [sin   cos] [x2] = [x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 交错合并回原始形状
        # stack后: (..., seq_len, d_k // 2, 2)
        # flatten后: (..., seq_len, d_k)
        out = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return out.flatten(-2)  # 把最后两维展平


# 带RoPE的多头注意力
# RoPE只作用于Q和K，在点积之前各自旋转，V不需要位置编码
class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        
        # 投影层
        self.q_proj = Linear(d_model, d_model, bias=False)
        self.k_proj = Linear(d_model, d_model, bias=False)
        self.v_proj = Linear(d_model, d_model, bias=False)
        self.output_proj = Linear(d_model, d_model, bias=False)
        
        # RoPE：注意是对每个head的d_head维度做旋转，支持自定义theta
        self.rope = RotaryPositionEmbedding(self.d_head, max_len, theta)

    def _split_heads(self, x):
        # x: (B, L, d_model) -> (B, n_heads, L, d_head)
        B, L, _ = x.size()
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        # x: (B, n_heads, L, d_head) -> (B, L, d_model)
        B, _, L, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        """
        Args:
            x: (B, L, d_model) 输入序列
            positions: (B, L) 或 (L,) 每个token的位置索引，None时自动生成 [0, 1, ..., L-1]
            mask: (B, 1, L, L) 或 (B, L, L) 注意力mask，0表示屏蔽
        Returns:
            (B, L, d_model) 输出
        """
        B, L, _ = x.size()
        
        # 如果没传positions，自动生成 [0, 1, ..., L-1]
        if positions is None:
            positions = torch.arange(L, device=x.device)
        
        # 1. 线性投影 + 分头
        q = self._split_heads(self.q_proj(x))  # (B, n_heads, L, d_head)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        
        # 2. 对Q和K应用RoPE（关键步骤！）
        q = self.rope(q, positions)
        k = self.rope(k, positions)  # V不需要RoPE
        
        # 3. 处理 mask 维度
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L) 广播到所有heads
        
        # 4. 注意力计算
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        
        # 5. 合并头 + 输出投影
        return self.output_proj(self._merge_heads(attn_output))